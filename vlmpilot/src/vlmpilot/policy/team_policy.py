# DroneCATS
# Copyright (c) 2026-present NAVER Cloud Corp.
# Apache-2.0

"""team_policy — N frames → ONE VLM call → N Actions + a team decision record.

Multi-drone counterpart of :mod:`vlmpilot.policy.policy`. The per-drone dict → :class:`Action` step is
NOT re-implemented here: it is :func:`vlmpilot.policy.policy.action_from_dict`, the same function the
single-drone policy runs, called with ``vehicle=<name>`` so every ``[ACTION]``
line carries the `` drone=<name>`` tail (§8.1.1-A1). No file I/O, no HTTP here.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import re
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from vlmpilot.policy.action import ACTION_FINISHED, ACTION_GO, ACTION_ROTATE, ACTION_THINK, Action
from vlmpilot.policy.camera import PinholeCamera
from vlmpilot.policy.policy import (
    ACTION_RESPONSE_SCHEMA,
    HISTORY_LEN,
    PLAN_FIELD_MAX_CHARS,
    PLAN_FIELD_MAX_TOKENS,
    PLAN_RESPONSE_SCHEMA,
    POINT_SCALE,
    _extract_action_json,
    _HistCmd,
    action_from_dict,
    history_notes,
    read_plan_fields,
    resolve_model_name,
    summarize_action,
)

__all__ = [
    "MODE_MULTI",
    "TEAM_ACTION_SCHEMA_NAME",
    "TeamDecisionRecord",
    "TeamPolicy",
    "render_tile",
    "team_action_response_schema",
    "team_plan_response_schema",
    "tile_label",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODE_MULTI = "airsim_multi"
EMPTY_TEAM_HISTORY_LINE = "    (none yet — first step)"  # §4.3.1 (differs from the single-drone line on purpose)

# §4.4-M5: root=object, one single-drone action schema per drone key (drone_1..drone_N).
# Root is an object because OpenAI Structured Outputs rejects array roots (measured 400).
# All keys go in required so the model can't omit a drone. Depends on fleet
# size, so it is a builder function, not a constant.
TEAM_ACTION_SCHEMA_NAME = "vlmpilot_team_action"


def _drone_item_schema() -> dict:
    """Action schema for one drone: reuses the field and range constraints of the single-drone ACTION_RESPONSE_SCHEMA
    as-is (coordinate/angle ranges aren't maintained in two places)."""
    return {
        "type": "object",
        "properties": {k: dict(v) for k, v in ACTION_RESPONSE_SCHEMA["properties"].items()},
        "required": ["action"],
    }


def team_action_response_schema(n_drones: int, think_action: bool = True) -> dict:
    """For normal steps: ``{"drone_1": <action>, ..., "drone_N": <action>}``. All keys required.
    With think_action=False, think is removed from each drone's action enum (3 actions)."""
    keys = [f"drone_{i}" for i in range(1, int(n_drones) + 1)]
    def item() -> dict:
        d = _drone_item_schema()
        if not think_action:
            d = copy.deepcopy(d)
            d["properties"]["action"]["enum"] = ["go", "rotate", "finished"]
        return d
    return {
        "type": "object",
        "properties": {key: item() for key in keys},
        "required": keys,
        # No additionalProperties: Gemini doesn't know this keyword (400).
        # The parser ignores out-of-spec keys (same reason as the single-drone PLAN_RESPONSE_SCHEMA).
    }


def team_plan_response_schema(n_drones: int) -> dict:
    """For plan steps: the observation and plan string fields, then the drone keys. Field order is the
    reasoning order: fix observation and plan first, then pick actions (same design as the single-drone PLAN_RESPONSE_SCHEMA,
    reusing the same constants for the character limits)."""
    plan_fields = {k: dict(PLAN_RESPONSE_SCHEMA["properties"][k])
                   for k in ("current_frame", "action_planning")}
    schema = team_action_response_schema(n_drones)
    return {
        "type": "object",
        "properties": {**plan_fields, **schema["properties"]},
        "required": [*plan_fields, *schema["required"]],
    }

# Tile overlay (§4.7-M18 — fixed parameters)
_TILE_FONT = cv2.FONT_HERSHEY_SIMPLEX
_TILE_NAME_ORG = (24, 48)
_TILE_NAME_SCALE = 1.1
_TILE_NAME_BGR = (0, 255, 255)
_TILE_NAME_THICKNESS = 3
_TILE_ACTION_ORG = (24, 96)
_TILE_ACTION_SCALE = 0.9
_TILE_ACTION_BGR = (0, 0, 255)
_TILE_ACTION_THICKNESS = 3
_TILE_NOACTION_BGR = (160, 160, 160)
_TILE_NOACTION_THICKNESS = 2
_TILE_POINT_RADIUS = 14
_TILE_POINT_BGR = (0, 255, 0)

# ---------------------------------------------------------------------------
# Prompts (§4.2 / §4.3 — verbatim; rendered with a single-pass placeholder substitution so
# literal braces in the JSON examples and in the instruction never need escaping)
# ---------------------------------------------------------------------------

_SYSTEM_TEMPLATE = (
    """You are the commander of a team of {n} autonomous drones in a simulator.
You are given {n} images — ONE egocentric forward-camera view per drone, in order.
Each drone flies independently. Your job: command the TEAM to FIND the target and get
ONE drone to it. The mission is COMPLETE as soon as any single drone reaches the target
and declares it — the other drones do NOT need to arrive.

Because only one drone has to arrive, SPREAD OUT. Sending every drone down the same path
wastes the rest of the team. Different drones should look in different directions and
cover different areas; whichever one sees the target closes in on it.

First reason about what each drone sees, then output ONE action for EVERY drone.

## Drone roster (image order = this order)
{roster}

## Output — ONE JSON OBJECT with EXACTLY {n} keys: "drone_1" … "drone_{n}", one per drone, in drone order.
Each key holds that drone's action, so the mapping is unambiguous.

{
  "drone_1": { ...action... },
  "drone_2": { ...action... }{comma}
  ...
}

Each action is ONE of four:

1) GO — fly toward a point that drone can see. Use when the target, or the path toward it, is visible.
   {"action":"go","point":{point_fmt},"depth_m":<meters>}
   - point  : on the target (or along the way to it), IN THAT DRONE'S OWN IMAGE.
              x: 0-1000 left->right (500=center). y: 0-1000 top->bottom (500=center).
              Each image is {frame_w}x{frame_h} px. Do NOT answer in pixels — scale x by
              1000/{frame_w} and y by 1000/{frame_h}. A spot a quarter in from the left and a
              quarter down from the top is pixel ({quarter_x}, {quarter_y}) and is 250 on BOTH
              axes. The axes scale differently, so 500,500 is the exact centre of the image —
              do not fall back on it unless the target is centred.
   - depth_m: estimated straight-line distance to that point, in METERS.
              FAR target -> LARGE depth (take big strides). CLOSE target -> SMALL depth (creep in carefully).

2) ROTATE — that drone yaws in place (holds position and altitude), turning its camera to look around.
   Use to SEARCH when that drone cannot see the target.
   {"action":"rotate","angle_deg":<deg>}
   - angle_deg: + = turn right, - = turn left, up to +-90 per step.
     Keep sweeping the SAME direction across steps so the drone covers a full 360, don't oscillate.

3) THINK — that drone holds position and spends this step thinking. Use when its view is
   unclear, there are multiple candidates, or you are unsure what to do next. The step right
   AFTER a THINK you will reason at length before acting, so THINK BUYS you a careful
   decision — it is not a wasted step.
   {"action":"think"}

4) FINISHED — that drone declares IT has reached the target. One drone saying this
   completes the mission for the whole team.
   Use ONLY when that drone is within a few meters.
   How to tell it is close enough: the target must FILL a large part of ITS frame. If it still
   fits inside the view with empty margin all around it, that drone is more than 10 m away — keep going.
   For an object on the ground, its base should reach the lower third of the image.
   {"action":"finished"}

## How to decide (per drone — they see different things, so they should usually do different things)

- Judge EACH drone from ITS OWN image. A point you give to drone 3 is read in drone 3's image only.
- That drone can't see the target -> ROTATE to search. Keep sweeping the SAME direction, don't oscillate.
- Target visible but not there yet -> GO toward it. It must look BIGGER and more centered than the
  last step; SHRINK depth_m as it grows. If it isn't getting bigger, that drone is not closing in.
- Target FILLS much of that drone's frame, centered, clear path -> FINISHED. If there is still
  empty margin around it, that drone is too far — GO, don't declare.
- Stuck or unsure — view unclear, multiple candidates, or that drone's recent GOs aren't getting it
  closer (circling) -> THINK. The step right after a THINK, reason carefully, then commit.
- **Rotated a full turn and still nothing -> STOP rotating.** More rotation from the same spot
  shows that drone the same view. Check "Recent actions per drone": once a drone's ROTATEs add up
  to a full circle, it must do one of these instead:
    * GO to a new vantage point — pick a reachable open point (down a road, across a yard) and
      travel there, then look again from there, OR
    * GO UP for a wider view — point HIGH in the image (small y) so it rises and sees farther, OR
    * THINK once to rethink, then commit to a GO on the next step.
- **Declaring once does not freeze that drone.** If it can still get closer, or the target has
  moved, keep closing in — GO again and declare from nearer. A closer look is never wasted.
- SPREAD OUT — **only while searching**. If drones cannot see the target, send them to different
  areas/headings so the team covers more ground instead of sweeping the same way.
  Once a drone CAN see the target, send it straight there — never push a drone away from a target
  it already sees, even if another drone sees the same one.

Output ONLY the JSON object with the {n} drone keys. No text outside it."""
)

_USER_TEMPLATE = (
    """Mission: {instruction}

## Recent actions per drone (use them to avoid circling / repeating a mistake)
{history}

The {n} images below are this step's views, one per drone, in drone order.
Output ONLY the JSON object with the {n} drone keys. No text outside it."""
)

# Same design as single-drone policy._PLAN_FIELDS_RULE: the schema enforces both fields as required, so
# keep it in system permanently with no per-step instruction. Only the wording is adapted to the team form (drone keys).
_TEAM_PLAN_FIELDS_RULE = (
    "## When you are asked to plan\n"
    "\n"
    "The step right after a THINK, your reply carries two extra fields BEFORE the drone keys:\n"
    "  current_frame   — what each drone can actually see in its camera view right now\n"
    "  action_planning — the action each drone takes now to carry out the mission, and why\n"
    f"Each field is cut off at about {PLAN_FIELD_MAX_TOKENS} tokens (roughly "
    f"{PLAN_FIELD_MAX_CHARS} characters). Finish your sentences before that — do not run\n"
    "to the cap, a half-written sentence is worse than a short one.\n"
    "\n"
)

# Edits that drop THINK from the team prompt when think_action: false (same design as single-drone policy._NOTHINK_EDITS).
_NOTHINK_SYSTEM_EDITS = (
    ("Each action is ONE of four:", "Each action is ONE of three:"),
    ("3) THINK — that drone holds position and spends this step thinking. Use when its view is\n"
     "   unclear, there are multiple candidates, or you are unsure what to do next. The step right\n"
     "   AFTER a THINK you will reason at length before acting, so THINK BUYS you a careful\n"
     "   decision — it is not a wasted step.\n"
     '   {"action":"think"}\n'
     "\n"
     "4) FINISHED", "3) FINISHED"),
    ("- Stuck or unsure — view unclear, multiple candidates, or that drone's recent GOs aren't getting it\n"
     "  closer (circling) -> THINK. The step right after a THINK, reason carefully, then commit.\n",
     "- Stuck or unsure — view unclear, multiple candidates, or that drone's recent GOs aren't getting it\n"
     "  closer (circling) -> do NOT repeat the same action. ROTATE to re-check, or GO to a new vantage point.\n"),
    ("    * GO UP for a wider view — point HIGH in the image (small y) so it rises and sees farther, OR\n"
     "    * THINK once to rethink, then commit to a GO on the next step.\n",
     "    * GO UP for a wider view — point HIGH in the image (small y) so it rises and sees farther.\n"),
)

_MM_SYSTEM_EDITS = (
    ("You are given {n} images — ONE egocentric forward-camera view per drone, in order.\n",
     "You are given each drone's egocentric forward-camera views, grouped per drone in drone\n"
     "order: first that drone's recent past views, oldest first, each labelled with its number\n"
     "in that drone's action list and the action it took while looking at it; then its CURRENT\n"
     "view, labelled CURRENT — judge that drone from its CURRENT view.\n"),
    ("## Drone roster (image order = this order)\n",
     "## Drone roster (image groups come in this order)\n"),
    ('Check "Recent actions per drone": once', "Check each drone's view labels: once"),
)
# History lines go out once per view label, so the block would duplicate the same content.
_MM_USER_EDITS = (
    ("## Recent actions per drone (use them to avoid circling / repeating a mistake)\n{history}\n\n", ""),
    ("The {n} images below are this step's views, one per drone, in drone order.\n",
     "The images ABOVE are grouped per drone, in drone order, each labelled with that drone's\n"
     "step number and the action it took while looking at that view (use them to avoid circling /\n"
     "repeating a mistake); each drone's LAST image, labelled CURRENT, is its view this step.\n"),
)

_SYSTEM_PLACEHOLDERS = re.compile(
    r"\{(n|roster|point_fmt|comma|frame_w|frame_h|quarter_x|quarter_y)\}")
_USER_PLACEHOLDERS = re.compile(r"\{(instruction|history|n)\}")


def _render(template: str, pattern: re.Pattern[str], values: Mapping[str, str]) -> str:
    return pattern.sub(lambda m: values[m.group(1)], template)


def _out(line: str = "") -> None:
    print(line, flush=True)


# ---------------------------------------------------------------------------
# Decision record (§2.3, §4.7)
# ---------------------------------------------------------------------------


@dataclass
class TeamDecisionRecord:
    """One team step, ready for ``artifacts.save_team_decision``. No file I/O here."""

    timestamp: str  # "%Y%m%d_%H%M%S", taken at decide_team() entry (M20)
    tiles_bgr: list[np.ndarray | None]  # per-drone overlay tiles; None where there was no frame (M17)
    payload: dict  # dumped as decision_<ts>.json (M19)
    # Live Action references (private helper). The record is written AFTER execution (M16) and the
    # controller fills ``action.gt_depth_m`` in place, so the payload is refreshed from these on save.
    actions: list[Action | None] = field(default_factory=list)

    def sync_actions(self) -> None:
        """Copy the current Action fields (notably ``gt_depth_m``) into ``payload["actions"]``."""
        entries = self.payload.get("actions") if isinstance(self.payload, dict) else None
        if not isinstance(entries, list):
            return
        for entry, action in zip(entries, self.actions):
            if action is None or not isinstance(entry, dict):
                continue
            entry.update(dataclasses.asdict(action))


# ---------------------------------------------------------------------------
# Tile overlay (§4.7-M18)
# ---------------------------------------------------------------------------


def tile_label(action: Action | None) -> str:
    """Action label drawn on a drone's tile (M18 table)."""
    if action is None:
        return "(no action - hold)"
    t = action.kind
    if t == ACTION_GO:
        return "GO" if action.depth_m is None else f"GO depth={action.depth_m:.1f}m"
    if t == ACTION_ROTATE:
        return f"ROTATE {action.angle_deg:+.0f} deg"
    if t == ACTION_THINK:
        return "THINK - reason next"
    if t == ACTION_FINISHED:
        return "FINISHED - ARRIVAL DECLARED"
    return str(t)


def render_tile(frame_bgr: np.ndarray, index: int, name: str, action: Action | None) -> np.ndarray:
    """Copy of ``frame_bgr`` with the M18 overlay: ``D{i} {name}``, the action label, GO point."""
    canvas = np.array(frame_bgr, copy=True)
    if canvas.ndim == 2:
        canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
    cv2.putText(
        canvas, f"D{index + 1} {name}", _TILE_NAME_ORG, _TILE_FONT, _TILE_NAME_SCALE,
        _TILE_NAME_BGR, _TILE_NAME_THICKNESS, cv2.LINE_AA,
    )
    if action is None:
        cv2.putText(
            canvas, tile_label(None), _TILE_ACTION_ORG, _TILE_FONT, _TILE_ACTION_SCALE,
            _TILE_NOACTION_BGR, _TILE_NOACTION_THICKNESS, cv2.LINE_AA,
        )
        return canvas
    cv2.putText(
        canvas, tile_label(action), _TILE_ACTION_ORG, _TILE_FONT, _TILE_ACTION_SCALE,
        _TILE_ACTION_BGR, _TILE_ACTION_THICKNESS, cv2.LINE_AA,
    )
    if action.kind == ACTION_GO:
        cx = int(round(float(action.pixel_u)))
        cy = int(round(float(action.pixel_v)))
        cv2.circle(canvas, (cx, cy), _TILE_POINT_RADIUS, _TILE_POINT_BGR, -1, cv2.LINE_AA)
    return canvas


# ---------------------------------------------------------------------------
# Response parsing helpers (§4.5)
# ---------------------------------------------------------------------------

_ARRAY_KEYS = ("actions", "commands", "drones")  # M7 step 2, in this order
_NUMBER_KEYS = ("drone", "id", "index")  # M8 step 1, in this order
# Top-level keys of a normal response: drone_1, drone_2, … (enforced by the schema). For free decoding,
# read leniently on separators and case ("drone 1"/"Drone_1"/"drone1").
_DRONE_KEY_RE = re.compile(r"^drone[_\s]?(\d+)$", re.IGNORECASE)


def _drone_number(item: Mapping[str, Any]) -> int | None:
    """First present key of ``drone``/``id``/``index`` as int; None when absent or not an int."""
    for key in _NUMBER_KEYS:
        if key in item:
            try:
                return int(item[key])
            except (TypeError, ValueError):
                return None
    return None


def _drone_keyed_entries(data: Any) -> list[tuple[int, dict]] | None:
    """``{"drone_1": {...}, ...}`` → ``[(number, action dict)]``, **sorted by number**
    (doesn't rely on JSON object key order). None if there are no drone keys at all."""
    if not isinstance(data, Mapping):
        return None
    found: list[tuple[int, dict]] = []
    for key, value in data.items():
        m = _DRONE_KEY_RE.match(str(key).strip())
        if m and isinstance(value, Mapping):
            found.append((int(m.group(1)), dict(value)))
    if not found:
        return None
    found.sort(key=lambda kv: kv[0])
    return found


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------


class TeamPolicy:
    """One VLM call per team step. Backend contract: ``generate_multi(...)``."""

    def __init__(
        self,
        backend: Any,
        camera: PinholeCamera,
        drone_names: Sequence[str],
        *,
        point_order: str = "yx",
        point_scale: float = POINT_SCALE,
        history_len: int = HISTORY_LEN,
        api_provider: str = "gemini",
        model_name: str = "",
        plan_schema: bool = False,
        mm_history: bool | None = None,
        mm_history_scale: float | None = None,
        think_action: bool = True,
    ) -> None:
        names = [str(n) for n in drone_names]
        if not names:
            raise ValueError("drone_names must not be empty")
        self.backend = backend
        self.camera = camera
        self.drone_names: list[str] = names
        self.n_drones = len(names)
        # Same contract as the single-drone VlmPolicy: on think steps, instead of relaxing the schema, force the extended
        # schema and keep native reasoning off (want_think=False). Reasoning comes out as two fields,
        # not free text.
        self.plan_schema = bool(plan_schema)
        # Whether the THINK action is included (config: think_action). Same contract as the single-drone VlmPolicy.
        self.think_action = bool(think_action)
        if not self.think_action:
            _out("[POLICY] team think_action OFF — 3-action prompt, plan field rules excluded")
        # The schema depends on fleet size (all drone keys required); build once and reuse.
        self._action_response_schema = team_action_response_schema(self.n_drones, self.think_action)
        self._plan_response_schema = team_plan_response_schema(self.n_drones)
        self.point_order = "xy" if str(point_order).lower() == "xy" else "yx"
        # Same per-model convention knob as VlmPolicy: the range the model writes
        # its points in, and the range the history is rendered back in.
        self.point_scale = float(point_scale) if float(point_scale) > 0 else POINT_SCALE
        self.history_len = int(history_len)
        self.api_provider = str(api_provider)
        self.model_name = resolve_model_name(self.api_provider, model_name)
        # M1/M2: one buffer per drone; the four deques of a drone share maxlen so entries stay paired.
        self._histories: list[deque[str]] = [deque(maxlen=self.history_len) for _ in names]
        self._last_frame_wh: tuple[int, int] | None = None
        self._hist_t: list[deque[float]] = [deque(maxlen=self.history_len) for _ in names]
        self._hist_cmd: list[deque[_HistCmd]] = [deque(maxlen=self.history_len) for _ in names]
        self._hist_view: list[deque[Any]] = [deque(maxlen=self.history_len) for _ in names]
        self.mm_history = bool(mm_history)
        # Same contract as the single-drone VlmPolicy: past view width = current send width × mm_history_scale. Negatives not allowed (not checked).
        self.mm_history_scale = float(mm_history_scale) if mm_history_scale else None
        if self.mm_history:
            width = f" (past view width = current send width × {self.mm_history_scale:g})" if self.mm_history_scale else ""
            _out(f"[MMHIST] ON — attaching each drone's views of the past {self.history_len} steps with action labels{width}")
        # M15: values that cross method boundaries live on the instance, never in locals.
        self._last_think: bool = False
        self._last_system: str = ""
        self._last_user: str = ""
        self._last_raw: str | None = None
        self._last_usage: Any = None
        self._last_thoughts: str | None = None
        _out(
            f"[POLICY] team {self.api_provider}:{self.model_name} drones={','.join(names)} "
            f"point_order={self.point_order} history_len={self.history_len}"
        )

    @classmethod
    def from_config(
        cls, config: dict, camera: PinholeCamera, backend: Any, drone_names: Sequence[str]
    ) -> "TeamPolicy":
        cfg = config or {}
        return cls(
            backend,
            camera,
            drone_names,
            point_order=str(cfg.get("point_order", "yx") or "yx"),
            point_scale=float(cfg.get("point_scale") or POINT_SCALE),
            api_provider=str(cfg.get("api_provider", "gemini") or "gemini"),
            model_name=str(cfg.get("model_name", "") or ""),
            plan_schema=bool(cfg.get("plan_schema", True) or False),  # ON if the key is missing from config. All 47 deployed configs are true and no config
            # uses false. A config that omitted it executed THINK via native reasoning instead of the schema and
            # the real drone crashed (router measured 35-37 s vs Tello's 15 s no-command auto-land).
            # The base/A reproduction arms set plan_schema: false explicitly.
            history_len=int(cfg.get("history_len") or HISTORY_LEN),
            mm_history=cfg.get("mm_history"),
            mm_history_scale=cfg.get("mm_history_scale"),
            think_action=bool(cfg.get("think_action", True)),
        )

    # -- prompt ---------------------------------------------------------------

    @property
    def point_fmt(self) -> str:
        return "[x, y]" if self.point_order == "xy" else "[y, x]"

    def frame_fields(self) -> dict[str, int]:
        """The size the prompt states is the image size the model actually receives: not the camera setting
        but this step's incoming frame passed through the encoding rule."""
        src = self._last_frame_wh or (int(self.camera.width), int(self.camera.height))
        try:
            from vlmpilot.policy.vlm_backends import sent_frame_size  # type: ignore
            w, h = sent_frame_size(*src, getattr(self.backend, "max_width", None))
        except Exception:
            w, h = src
        return dict(frame_w=w, frame_h=h, quarter_x=w // 4, quarter_y=h // 4)

    def _note_frame(self, frames) -> None:
        """Every drone uses the same camera; remember the size of the first valid frame."""
        for f in frames or ():
            try:
                h, w = int(f.shape[0]), int(f.shape[1])
            except Exception:
                continue
            if w > 0 and h > 0:
                self._last_frame_wh = (w, h)
                return

    def roster_lines(self) -> str:
        """§4.2.1: ``  - DRONE {i} = {name}`` per drone, 1-based."""
        return "\n".join(f"  - DRONE {i} = {name}" for i, name in enumerate(self.drone_names, start=1))

    def history_items(self, i: int) -> list[str]:
        """Drone i's ``summary  [execution note]`` items, oldest first."""
        now = time.monotonic()
        notes = history_notes(list(self._hist_cmd[i]), list(self._hist_t[i]), now)
        return [f"{s}  {n}" for s, n in zip(self._histories[i], notes)]

    def history_labels(self, i: int) -> list[str]:
        """Drone i's ``step j of n: summary  [execution note]``: text lines and view labels use the same sentence."""
        items = self.history_items(i)
        n = len(items)
        labels = [f"step {j} of {n}: {s}" for j, s in enumerate(items, start=1)]
        if labels:
            labels[-1] += "  <- latest"
        return labels

    def history_lines(self) -> str:
        """§4.3.1: one block per drone, joined with newlines."""
        blocks: list[str] = []
        for i, name in enumerate(self.drone_names):
            labels = self.history_labels(i)
            body = "\n".join(f"    {lab}" for lab in labels) if labels else EMPTY_TEAM_HISTORY_LINE
            blocks.append(f"  DRONE {i + 1} ({name}):\n{body}")
        return "\n".join(blocks)

    def _with_mm_history(self, template: str, edits) -> str:
        # The first step has no past view to attach, so the original prompt goes out as-is.
        if not self.mm_history or not any(self._histories):
            return template
        for anchor, repl in edits:
            assert anchor in template, "mm-history anchor drifted from template"
            template = template.replace(anchor, repl, 1)
        return template

    def build_system_prompt(self) -> str:
        n = self.n_drones
        template = self._with_mm_history(_SYSTEM_TEMPLATE, _MM_SYSTEM_EDITS)
        if not self.think_action:
            for anchor, repl in _NOTHINK_SYSTEM_EDITS:
                assert anchor in template, "think_action anchor drifted from template"
                template = template.replace(anchor, repl, 1)
        if self.plan_schema and self.think_action:
            template = template.rstrip("\n") + "\n\n" + _TEAM_PLAN_FIELDS_RULE.rstrip("\n")
        return _render(
            template,
            _SYSTEM_PLACEHOLDERS,
            {"n": str(n), "roster": self.roster_lines(), "point_fmt": self.point_fmt,
             "comma": "," if n > 2 else "",
             **{k: str(v) for k, v in self.frame_fields().items()}},
        )

    def build_user_prompt(self, instruction: str) -> str:
        """Same on plan steps: the schema enforces both fields as required."""
        template = self._with_mm_history(_USER_TEMPLATE, _MM_USER_EDITS)
        return _render(
            template,
            _USER_PLACEHOLDERS,
            {"instruction": str(instruction), "history": self.history_lines(), "n": str(self.n_drones)},
        )

    def image_labels(self) -> list[str]:
        """M12: text label inserted right BEFORE each drone's image (mm-history off)."""
        return [f"DRONE {i + 1} ({name}) view:" for i, name in enumerate(self.drone_names)]

    def request_images(self, frames: Sequence[np.ndarray]) -> tuple[list[Any], list[str], list[int | None]]:
        """(images, labels, max_widths) in request order — grouped per drone when mm-history is on."""
        # With no past to attach (off, first step) the prompt is the original wording too, so no bundle label is used.
        if not self.mm_history or not any(self._hist_view):
            return list(frames), self.image_labels(), [None] * len(frames)
        images: list[Any] = []
        labels: list[str] = []
        widths: list[int | None] = []
        for i, (name, frame) in enumerate(zip(self.drone_names, frames)):
            head = f"DRONE {i + 1} ({name}) — "
            past = [head + lab for lab in self.history_labels(i)]
            images += list(self._hist_view[i]) + [frame]
            labels += past + [f"{head}CURRENT view, decide from this one:"]
            widths += [self._mm_past_width()] * len(past) + [None]
        return images, labels, widths

    def _mm_past_width(self) -> int | None:
        """Width cap in px for one past view (same rule as single-drone VlmPolicy._mm_past_width)."""
        if not self.mm_history_scale:
            return None
        return max(1, int(round(self.frame_fields()["frame_w"] * self.mm_history_scale)))

    def note_command(self, i: int, duration_s: float, distance_m: float | None = None) -> None:
        """Length / travel distance of the command issued by drone i's last action (called by the controller after plan)."""
        if 0 <= i < self.n_drones and self._hist_cmd[i]:
            self._hist_cmd[i][-1] = _HistCmd(
                float(duration_s or 0.0),
                None if distance_m is None else float(distance_m),
            )

    # -- parsing (M6–M8) ---------------------------------------------------------

    def parse_team_response(self, text: str) -> dict[int, dict]:
        """Model raw text → ``{drone_index(0-based): action dict}``. Raises on unusable input.

        Prints the ``[MULTI]`` numbering / duplicate lines of §4.5-M8. Drones missing from the
        result are the caller's business (M10).
        """
        cleaned = _extract_action_json(text)  # M6
        data = json.loads(cleaned)  # M7-1
        # Drone-key object (normal form). extract_action_json wraps a single object as [obj],
        # so also peel one list layer. Plan responses (current_frame/action_planning at the
        # same top level) yield actions through this path too; non-drone keys are ignored.
        keyed = _drone_keyed_entries(data)
        if keyed is None and isinstance(data, list) and len(data) == 1:
            keyed = _drone_keyed_entries(data[0])
        if keyed is not None:
            return self._map_drone_keyed(keyed)
        if isinstance(data, dict):  # M7-2
            for key in _ARRAY_KEYS:
                if isinstance(data.get(key), list):
                    data = data[key]
                    break
            else:
                data = [data]
        if not isinstance(data, list):  # M7-3
            raise TypeError(f"team response must be a JSON array, got {type(data).__name__}")
        items = [d for d in data if isinstance(d, dict)]  # M7-4

        # M8 — decide the numbering base ONCE from the whole response.
        n = self.n_drones
        numbers = [_drone_number(d) for d in items]
        present = [r for r in numbers if r is not None]
        base: int | None
        if present and all(1 <= r <= n for r in present):
            base = 1
        elif present and all(0 <= r <= n - 1 for r in present):
            base = 0
            _out("[MULTI] reply uses 0-based drone numbers — interpreting as 0-based")
        elif present:
            base = None
            _out(f"[MULTI] drone numbers out of range {present} — falling back to position order")
        else:
            # SPEC-GAP: no element carries a number at all → pure positional order, silently
            # (the "out of range" line would name an empty list).
            base = None

        result: dict[int, dict] = {}
        for pos, (item, number) in enumerate(zip(items, numbers)):
            idx = pos if (base is None or number is None) else number - base  # M8-3
            if not 0 <= idx < n:  # M8-4
                continue
            if idx in result:  # M8-5
                _out(f"[MULTI] drone {idx + 1} action came twice — using the first one only")
                continue
            result[idx] = item
        return result

    def _map_drone_keyed(self, keyed: list[tuple[int, dict]]) -> dict[int, dict]:
        """Drone key number → 0-based index. Same principle as M8: the numbering is decided once from the
        whole response, out-of-range entries are dropped, and only the first of duplicates is used. Since the key is
        the number there is no positional fallback: a missing drone is absent from the result and the caller keeps it holding (M10)."""
        n = self.n_drones
        numbers = [num for num, _ in keyed]
        if all(1 <= r <= n for r in numbers):
            base = 1
        elif all(0 <= r <= n - 1 for r in numbers):
            base = 0
            _out("[MULTI] reply uses 0-based drone numbers — interpreting as 0-based")
        else:
            base = 1
            bad = [r for r in numbers if not 1 <= r <= n]
            _out(f"[MULTI] drone numbers out of range {bad} — using only the in-range keys")
        result: dict[int, dict] = {}
        for num, item in keyed:
            idx = num - base
            if not 0 <= idx < n:
                continue
            if idx in result:  # keys that collide after normalization, like "drone_1" and "Drone 1"
                _out(f"[MULTI] drone {idx + 1} action came twice — using the first one only")
                continue
            result[idx] = item
        return result

    # -- think toggle (M13) -------------------------------------------------------

    def _think_flag(self) -> bool:
        if not self.think_action:
            return False
        return any(bool(h) and h[-1].startswith("THINK") for h in self._histories)

    # -- backend call (M12) -------------------------------------------------------

    def _call_backend(
        self, user_prompt: str, frames: Sequence[np.ndarray], system_prompt: str,
        think: bool, schema: Mapping[str, Any],
    ) -> tuple[str | None, Any, str | None]:
        """→ (raw text, usage, thoughts). Uses ``generate_multi_detailed`` when the backend has it."""
        images, labels, widths = self.request_images(frames)
        kwargs: dict[str, Any] = dict(think=think, system=system_prompt, response_schema=schema)
        if any(w is not None for w in widths):
            kwargs["max_widths"] = widths
        detailed = getattr(self.backend, "generate_multi_detailed", None)
        if callable(detailed):
            res = detailed(user_prompt, images, labels, **kwargs)
            return getattr(res, "text", None), getattr(res, "usage", None), getattr(res, "thoughts", None)
        return self.backend.generate_multi(user_prompt, images, labels, **kwargs), None, None

    # -- record (M17–M19) ---------------------------------------------------------

    def _make_payload(self, timestamp: str, instruction: str, actions: Sequence[Action | None],
                      *, plan: dict | None = None) -> dict:
        entries: list[dict] = []
        for i, (name, action) in enumerate(zip(self.drone_names, actions), start=1):
            if action is None:
                entries.append({"drone": i, "vehicle": name, "kind": None})  # exactly three keys
            else:
                entries.append({"drone": i, "vehicle": name, **dataclasses.asdict(action)})
        return {
            "timestamp": timestamp,
            "mode": MODE_MULTI,
            "api_provider": self.api_provider,
            "model": self.model_name,
            "instruction": instruction,
            "n_drones": self.n_drones,
            "system_prompt": self._last_system,
            "user_prompt": self._last_user,
            "raw_response": self._last_raw,
            "usage": self._last_usage,
            "thoughts": self._last_thoughts,
            "think_on": bool(self._last_think),
            # The plan step's observation and plan fields; same contract as single-drone payload["plan"].
            "plan": plan,
            "actions": entries,
        }

    def _make_tiles(self, frames: Sequence[np.ndarray | None], actions: Sequence[Action | None]) -> list[np.ndarray | None]:
        tiles: list[np.ndarray | None] = []
        for i, (name, frame, action) in enumerate(zip(self.drone_names, frames, actions)):
            if frame is None:
                tiles.append(None)  # M17: no frame → no tile
                continue
            try:
                tiles.append(render_tile(frame, i, name, action))
            except Exception as exc:  # noqa: BLE001 — a debug image must never cost the step
                _out(f"[MULTI] DRONE {i + 1} tile build failed (ignored): {exc}")
                tiles.append(None)
        return tiles

    # -- main entry -------------------------------------------------------------

    def decide_team(
        self, frames_bgr: Sequence[np.ndarray | None], instruction: str
    ) -> tuple[list[Action | None], TeamDecisionRecord | None]:
        """One team step. ``(actions[n], record)``; ``([None]*n, None)`` when nothing usable came back.

        Never raises (M14). A drone without a usable action gets ``None`` and keeps its history (M3/M10).
        """
        n = self.n_drones
        timestamp = time.strftime("%Y%m%d_%H%M%S")  # M20: at entry
        try:
            frames = list(frames_bgr or [])[:n]
            frames += [None] * (n - len(frames))
            self._note_frame(frames)
            if all(f is None for f in frames):
                _out("[MULTI] all input frames are None — skipping this step (no backend call)")
                return [None] * n, None
            # Fleet already substitutes black frames (§5.5-R26) and logs it; this is the same
            # placeholder for direct callers so image order == drone order always holds.
            send = [
                f if f is not None else np.zeros((self.camera.height, self.camera.width, 3), np.uint8)
                for f in frames
            ]

            think = self._think_flag()  # M13: judged BEFORE this step's actions enter the buffers
            # Same contract as single-drone policy.decide (plan_schema): the reasoning of the step right after THINK
            # is taken as two schema fields, not free text, and native reasoning stays off.
            plan_step = bool(think and self.plan_schema)
            schema = self._plan_response_schema if plan_step else self._action_response_schema
            want_think = False if plan_step else think
            system_prompt = self.build_system_prompt()
            user_prompt = self.build_user_prompt(instruction)
            if think:
                _out("[MULTI] previous THINK detected — thinking ON this step")
            self._last_think, self._last_system, self._last_user = think, system_prompt, user_prompt

            raw, usage, thoughts = self._call_backend(user_prompt, send, system_prompt, want_think, schema)
            self._last_raw = None if raw is None else str(raw)
            self._last_usage, self._last_thoughts = usage, thoughts
            if raw is None:
                _out("[MULTI] no backend reply (None) — every drone holds this step")
                return [None] * n, None
            # Episode log is the only place the raw reply is kept; prefixed so an echoed marker can
            # never satisfy a runner regex (same rule as the single-drone policy).
            for line in str(raw).splitlines() or [""]:
                _out(f"[RAW] {line}")

            plan = read_plan_fields(str(raw)) if plan_step else None

            try:
                per_drone = self.parse_team_response(str(raw))
            except Exception as exc:  # noqa: BLE001 — M11
                _out(f"[MULTI] reply parse failed — every drone holds this step: {exc}")
                return [None] * n, None

            actions: list[Action | None] = []
            for i, name in enumerate(self.drone_names):
                data = per_drone.get(i)
                if data is None:
                    _out(f"[MULTI] DRONE {i + 1} no action — holding this step")
                    actions.append(None)
                    continue
                try:
                    actions.append(action_from_dict(data, self.camera, self.point_order,
                                                    vehicle=name, point_scale=self.point_scale))
                except Exception as exc:  # noqa: BLE001 — M10: one drone's failure stays with that drone
                    _out(f"[MULTI] DRONE {i + 1} action parse failed (hold): {exc}")
                    actions.append(None)

            now = time.monotonic()
            for i, action in enumerate(actions):  # M3
                if action is None:
                    continue
                self._histories[i].append(summarize_action(action, self.camera, self.point_scale))
                self._hist_t[i].append(now)
                self._hist_cmd[i].append(_HistCmd())   # filled by the controller via note_command after plan
                if self.mm_history:
                    self._hist_view[i].append(send[i])

            tiles = self._make_tiles(frames, actions)
            payload = self._make_payload(timestamp, instruction, actions, plan=plan)
            return actions, TeamDecisionRecord(timestamp=timestamp, tiles_bgr=tiles, payload=payload, actions=actions)
        except Exception as exc:  # noqa: BLE001 — M14: never let anything out
            _out(f"[MULTI] team decision failed ({type(exc).__name__}: {exc}) — every drone holds this step")
            return [None] * n, None
