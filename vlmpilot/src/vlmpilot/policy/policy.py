# DroneCATS
# Copyright (c) 2026-present NAVER Cloud Corp.
# Apache-2.0

"""VLM decision step.

(frame, instruction, history) -> one :class:`Action` + a :class:`DecisionRecord` for artifacts.
No file I/O here; no HTTP here (that is the backend's job).
"""

from __future__ import annotations

import dataclasses
import copy
import json
import re
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Mapping, NamedTuple, Sequence

import cv2
import numpy as np

from vlmpilot.policy.action import (
    ACTION_FINISHED,
    ACTION_GO,
    ACTION_ROTATE,
    ACTION_THINK,
    Action,
)
from vlmpilot.policy.camera import PinholeCamera

# ---------------------------------------------------------------------------
# Constants (§4.2 — hard-coded, not config)
# ---------------------------------------------------------------------------

HISTORY_LEN = 5
DEFAULT_DEPTH_M = 3.0
DEPTH_MIN_M = 0.3
DEPTH_MAX_M = 200.0
POINT_SCALE = 1000.0
FALLBACK_ROTATE_DEG = 45.0
# The prompt asks for at most +-90 deg of yaw per step. It used to be a request
# only: the parser took whatever came back, and the suite executed rotations of
# up to 1080 deg. Clamping keeps the executed action inside the action space the
# prompt defines, and the log records every clamp so the violations stay
# countable rather than disappearing into a well-behaved number.
ROTATE_MAX_DEG = 90.0
FALLBACK_ROTATE_REASON = "no point returned"

# Overlay banner (P21.1 — fixed parameters)
_BANNER_ORG = (30, 60)
_BANNER_FONT = cv2.FONT_HERSHEY_SIMPLEX
_BANNER_SCALE = 1.0
_BANNER_COLOR_BGR = (0, 0, 255)
_BANNER_THICKNESS = 3

# Point marker / helper label (P21.2/3 — implementer's choice, not a contract)
_MARK_COLOR_BGR = (255, 255, 0)  # cyan, distinct from the red banner
_MARK_RADIUS = 14
_MARK_THICKNESS = 3
_LABEL_FONT = cv2.FONT_HERSHEY_SIMPLEX
_LABEL_SCALE = 0.6
_LABEL_COLOR_BGR = (255, 255, 0)
_LABEL_THICKNESS = 2

# Default model IDs (§4.3-M2) — external service identifiers
_DEFAULT_MODEL_OPENAI = "google/gemini-2.5-flash"
_DEFAULT_MODEL_OTHER = "gemini-2.5-flash"

EMPTY_HISTORY_LINE = "(none yet — this is the first step)"

# §3.3-P8 — owned by the policy, handed to the backend. Name is "vlmpilot_action".
ACTION_SCHEMA_NAME = "vlmpilot_action"
ACTION_RESPONSE_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["go", "rotate", "think", "finished"]},
        "point": {"type": "array",
                  "items": {"type": "integer", "minimum": 0, "maximum": int(POINT_SCALE)},
                  "minItems": 2, "maxItems": 2},
        "depth_m": {"type": "number", "minimum": 0, "maximum": 200},
        "angle_deg": {"type": "number", "minimum": -ROTATE_MAX_DEG, "maximum": ROTATE_MAX_DEG},
    },
    "required": ["action"],
}

# 3-action schema without the think action (config: think_action: false). Only the enum differs.
def action_response_schema(think_action: bool = True) -> dict:
    if think_action:
        return ACTION_RESPONSE_SCHEMA
    schema = copy.deepcopy(ACTION_RESPONSE_SCHEMA)
    schema["properties"]["action"]["enum"] = ["go", "rotate", "finished"]
    return schema


# Edits that drop THINK from the golden prompt when think_action: false — (anchor, replacement). Renumbers to three actions,
# and turns the "THINK when stuck" rule into an action possible without THINK. If an anchor vanishes from the template, the assert breaks first.
_NOTHINK_EDITS = (
    ("(choose one of four)", "(choose one of three)"),
    ("3) THINK — hold position and spend this step thinking. Use when the view is unclear, there are\n"
     "   multiple candidates, or you are unsure what to do next. The step right AFTER a THINK you will\n"
     "   reason at length before acting, so THINK BUYS you a careful decision — it is not a wasted step.\n"
     '   {{"action":"think"}}\n'
     "\n"
     "4) FINISHED", "3) FINISHED"),
    # In the split system template this line is last, so its trailing newline is rstripped; don't put a newline in the anchor.
    ("- Stuck or unsure — view unclear, multiple candidates, or your recent GOs aren't getting you\n"
     "  closer (circling) -> THINK. The step right after a THINK, reason carefully, then commit.",
     "- Stuck or unsure — view unclear, multiple candidates, or your recent GOs aren't getting you\n"
     "  closer (circling) -> do NOT repeat the same action. ROTATE to re-check, or GO to a new vantage point."),
)


# Character caps for the two fields. Measured (each server's /tokenize): qwen35-2b/4b 4.31 chars/token, cosmos
# 4.28, so 256 tokens is about 1,100 chars. Set 1200 so the grammar cuts around there, while
# the prompt tells the model in advance "it is cut at 256 tokens" so it closes its own sentences.
# The cap is a runaway breakwater, not a length control: a cut at the cap breaks a sentence mid-way
# and that text contaminates the action generated right after (measured).
PLAN_FIELD_MAX_CHARS = 1200
PLAN_FIELD_MAX_TOKENS = 256

# A system that differs only on plan steps breaks the prefix cache, so keep it in system permanently.
_PLAN_FIELDS_RULE = (
    "## When you are asked to plan\n"
    "\n"
    "The step right after a THINK, your reply carries these fields before the action:\n"
    "  target_in_view  — yes / no / unsure: is the target named in the command in this view?\n"
    "  target_match    — match / mismatch / unsure: do the candidate's colour AND kind both\n"
    "                    match the words of the command? If either differs it is NOT the target.\n"
    "  closing_in      — yes / no / unknown: did your last few actions get you closer?\n"
    "  current_frame   — ONLY the target named in the command. One short line each, nothing else:\n"
    "                    (1) the candidate you are looking at: its colour and what kind of object\n"
    "                        it is, AS YOU SEE THEM\n"
    "                    (2) do both match the words of the command? If the colour or the kind\n"
    "                        differs, it is NOT the target — say so instead of assuming it is.\n"
    "                    (3) where it sits and how much of the frame it fills, and whether it looks\n"
    "                        bigger or smaller than in your previous views.\n"
    "                    Do NOT describe anything that is neither the target nor in the way of\n"
    "                    reaching it. If nothing matches, say so in one line.\n"
    "  action_planning — ONLY the decision. One short line each:\n"
    "                    (1) did your last few actions get you closer? yes / no / can't tell\n"
    "                    (2) the ONE action you take now, and why the other options are wrong here\n"
    "                    (3) what you expect to see next step if this action worked\n"
    f"Each text field is cut off at about {PLAN_FIELD_MAX_TOKENS} tokens (roughly "
    f"{PLAN_FIELD_MAX_CHARS} characters). Finish your\n"
    "sentences before that — a half-written sentence is worse than a short one.\n"
    "\n"
)

# Extended schema for think steps (config: plan_schema). Field order is the reasoning order:
# fix observation and plan first, then pick the action. No ``pattern``:
# on our vLLM grammar compilation silently fails and out-of-spec keys get mixed in (measured).
PLAN_FIELD_KEYS = ("target_in_view", "target_match", "closing_in",
                   "current_frame", "action_planning")

PLAN_RESPONSE_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "target_in_view": {"type": "string", "enum": ["yes", "no", "unsure"]},
        "target_match": {"type": "string", "enum": ["match", "mismatch", "unsure"]},
        "closing_in": {"type": "string", "enum": ["yes", "no", "unknown"]},
        "current_frame": {"type": "string", "minLength": 20, "maxLength": PLAN_FIELD_MAX_CHARS},
        "action_planning": {"type": "string", "minLength": 20, "maxLength": PLAN_FIELD_MAX_CHARS},
        "action": {"type": "string", "enum": ["go", "rotate", "think", "finished"]},
        "point": {"type": "array", "items": {"type": "integer", "minimum": 0, "maximum": 1000},
                  "minItems": 2, "maxItems": 2},
        "depth_m": {"type": "number", "minimum": 0, "maximum": 200},
        "angle_deg": {"type": "number", "minimum": -90, "maximum": 90},
    },
    "required": ["target_in_view", "target_match", "closing_in",
                 "current_frame", "action_planning", "action"],
    # No additionalProperties: Gemini doesn't know this keyword
    # ("Unknown name \"additional_properties\"", 400). The parser ignores out-of-spec keys,
    # so there is practically no difference in enforcement.
}

# §3.4 golden prompt. JSON braces are doubled for str.format; ``{instruction}``,
# ``{point_fmt}``, ``{history_len}``, ``{history}`` are the only placeholders.
# No trailing newline; no trailing whitespace on any line.
_PROMPT_TEMPLATE = (
    "You are an autonomous drone pilot in a simulator. You see ONE egocentric\n"
    "image from the drone's forward camera. Your job: reach the target described below,\n"
    "then declare arrival. Decide WHERE the target is, then output ONE action.\n"
    "\n"
    "Command: {instruction}\n"
    "\n"
    "## Output — EXACTLY ONE action as a single JSON object (choose one of four):\n"
    "\n"
    "1) GO — fly toward a point you can see. Use when the target, or the path toward it, is visible.\n"
    '   {{"action":"go","point":{point_fmt},"depth_m":<meters>}}\n'
    "   - point  : on the target (or along the way to it). x: 0-1000 left->right (500=center).\n"
    "              y: 0-1000 top->bottom (500=center).\n"
    "              The image you receive is {frame_w}x{frame_h} px. Do NOT answer in pixels —\n"
    "              scale x by 1000/{frame_w} and y by 1000/{frame_h}. A spot a quarter in from\n"
    "              the left and a quarter down from the top is pixel ({quarter_x}, {quarter_y})\n"
    "              and is 250 on BOTH axes. The axes scale differently, so 500,500 is the exact\n"
    "              centre of the frame — do not fall back on it unless the target is centred.\n"
    "   - depth_m: how far AHEAD to fly, in METERS \u2014 the distance along the direction the\n"
    "              camera is facing, NOT the diagonal distance to the point.\n"
    "              FAR target -> LARGE depth (take big strides). CLOSE target -> SMALL depth (creep in carefully).\n"
    "\n"
    "2) ROTATE — yaw in place (hold position and altitude), turning the camera to look around.\n"
    "   Use to SEARCH when the target is NOT visible.\n"
    '   {{"action":"rotate","angle_deg":<deg>}}\n'
    "   - angle_deg: + = turn right, - = turn left, up to +-90 per step.\n"
    "     Keep sweeping the SAME direction across steps so you cover a full 360, don't oscillate.\n"
    "\n"
    "3) THINK — hold position and spend this step thinking. Use when the view is unclear, there are\n"
    "   multiple candidates, or you are unsure what to do next. The step right AFTER a THINK you will\n"
    "   reason at length before acting, so THINK BUYS you a careful decision — it is not a wasted step.\n"
    '   {{"action":"think"}}\n'
    "\n"
    "4) FINISHED — declare you have ARRIVED and hold. Use ONLY when you are within a few meters.\n"
    "   How to tell you are close enough: the target must FILL a large part of the frame. If it still\n"
    "   fits inside the view with empty margin all around it, you are more than 10 m away — keep going.\n"
    "   For an object on the ground, its base should reach the lower third of the image.\n"
    '   {{"action":"finished"}}\n'
    "\n"
    "## How to decide\n"
    "\n"
    "- Can't see the target -> ROTATE to search. Keep sweeping the SAME direction, don't oscillate.\n"
    "- Rotated a full turn and still nothing -> STOP rotating. More rotation from the same spot shows\n"
    "  you the same view. GO to a new vantage point (an open direction), then look again from there.\n"
    "- Target visible but not there yet -> GO toward it. It must look BIGGER and more centered than\n"
    "  the last step; SHRINK depth_m as it grows. If it isn't getting bigger, you are not closing in.\n"
    "- Target FILLS much of the frame, centered, clear path -> FINISHED. If there is still empty\n"
    "  margin around it, you are too far — GO, don't declare.\n"
    "- Stuck or unsure — view unclear, multiple candidates, or your recent GOs aren't getting you\n"
    "  closer (circling) -> THINK. The step right after a THINK, reason carefully, then commit.\n"
    "\n"
    "## Your last {history_len} actions (use them to avoid circling / repeating a mistake)\n"
    "{history}\n"
    "\n"
    "Output ONLY the single JSON object. No text outside it."
)

# system = golden minus the Command block and per-step tail; user = Command + history + closing line.
# Opt-in camera line. Off by default: measured zero-shot,
# neither model moved its depths when this line changed (elasticity ~0), so it
# is an experiment lever, not part of the golden prompt.
_INTRINSICS_ANCHOR = (
    "              FAR target -> LARGE depth (take big strides)."
    " CLOSE target -> SMALL depth (creep in carefully).\n"
)
_INTRINSICS_LINE = (
    "   - camera : the image is {w}x{h} px with a {hfov:g} deg horizontal FOV"
    " (focal length fx = fy = {f:.0f} px).\n"
)

# Opt-in depth convention (config: canonical_depth). Off by default: the golden
# prompt keeps the actual-metric wording. ON swaps the definition sentence for
# the canonical wording of the apdepth loadout, so the label the model is asked
# for matches the label it was trained on.
_DEPTH_BULLET_ACTUAL = (
    "   - depth_m: how far AHEAD to fly, in METERS \u2014 the distance along the direction the\n"
    "              camera is facing, NOT the diagonal distance to the point.\n"
)
_DEPTH_BULLET_CANONICAL = (
    "   - depth_m: canonical depth — the distance along the camera's forward axis,\n"
    "              expressed as it would appear to a reference camera with a 53\u00b0 horizontal\n"
    "              field of view.\n"
    "              NOT the diagonal distance to the point.\n"
)

_COMMAND_BLOCK = "Command: {instruction}\n\n"
_STEP_TAIL_START = "## Your last {history_len} actions"
_HISTORY_BLOCK = (
    "## Your last {history_len} actions (use them to avoid circling / repeating a mistake)\n"
    "{history}\n"
    "\n"
)
_idx = _PROMPT_TEMPLATE.index(_STEP_TAIL_START)
_SPLIT_SYSTEM_TEMPLATE = _PROMPT_TEMPLATE[:_idx].replace(_COMMAND_BLOCK, "", 1).rstrip("\n")
_SPLIT_USER_TEMPLATE = _COMMAND_BLOCK + _PROMPT_TEMPLATE[_idx:]
del _idx


def _out(line: str = "") -> None:
    print(line, flush=True)


# ---------------------------------------------------------------------------
# Raw-text cleaning (§3.3-P9) — delegated to vlm_backends.extract_action_json
# ---------------------------------------------------------------------------

_extract_impl: Callable[[str | None], str] | None = None


def _fallback_extract_action_json(raw: str | None) -> str:
    """Private re-statement of ``vlm_backends.extract_action_json`` used only when ``vlmpilot.policy.vlm_backends``
    cannot be imported (module absent or its SDK imports fail). Same contract:
    last ``</think>`` cut -> first code fence -> strip -> array-first greedy JSON grab,
    single object promoted to ``[obj]``. Never raises."""
    if raw is None:
        return ""
    text = str(raw)
    cut = text.rfind("</think>")
    if cut != -1:
        text = text[cut + len("</think>"):]
    for fence in ("```json", "```"):
        pos = text.find(fence)
        if pos != -1:
            start = pos + len(fence)
            end = text.find("```", start)
            text = text[start:] if end == -1 else text[start:end]
            break
    text = text.strip()
    m = re.search(r"\[\s*\{[\s\S]*\}\s*\]", text)
    if m:
        return m.group(0).strip()
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        return "[" + m.group(0) + "]"
    return text


def _extract_action_json(raw: str | None) -> str:
    global _extract_impl
    if _extract_impl is None:
        try:
            # SPEC-GAP: vlm_backends.py is written concurrently by another implementer; import
            # lazily and fall back to the private equivalent so this module (and its tests)
            # never depend on that file being present/importable.
            from vlmpilot.policy.vlm_backends import extract_action_json as impl  # type: ignore
        except Exception:
            impl = _fallback_extract_action_json
        _extract_impl = impl
    return _extract_impl(raw)


def read_plan_fields(text: str) -> dict | None:
    """Read the observation and plan fields from a plan step's response. None on failure.

    Shared by single (PLAN_RESPONSE_SCHEMA) and multi (team_policy's TEAM_PLAN schema):
    both fields are at the top level of the response in either case.
    """
    try:
        cleaned = _extract_action_json(text)
        try:
            data = json.loads(cleaned)
        except Exception:
            m = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if not m:
                return None
            data = json.loads(m.group(0))
    except Exception:
        return None
    if isinstance(data, list):       # _extract_action_json wraps an object in a list
        data = data[0] if data else None
    if not isinstance(data, dict):
        return None
    got = {k: data.get(k) for k in PLAN_FIELD_KEYS
           if isinstance(data.get(k), str) and data.get(k).strip()}
    return got or None


# ---------------------------------------------------------------------------
# Decision record
# ---------------------------------------------------------------------------


@dataclass
class DecisionRecord:
    """One step's decision, ready for ``artifacts.py`` to write. No file I/O here."""

    timestamp: str
    overlay_image_bgr: np.ndarray
    payload: dict
    # The frame as-is, without the baked overlay. For when an unmarked image is needed, like paper figures.
    # None when save_raw_frames is off, in which case artifacts writes nothing more.
    raw_image_bgr: np.ndarray | None = None


def resolve_model_name(api_provider: str, model_name: str | None) -> str:
    """§4.3 M1/M2: explicit (stripped) name wins; else provider default."""
    name = (model_name or "").strip()
    if name:
        return name
    if str(api_provider).strip().lower() == "openai":
        return _DEFAULT_MODEL_OPENAI
    return _DEFAULT_MODEL_OTHER


# With mm-history on, this sentence describing the image layout changes and the history block (_HISTORY_BLOCK)
# is dropped: the same sentence goes out once per image label, so the block would be a duplicate.
_ONE_IMAGE_ANCHOR = "You see ONE egocentric\nimage from the drone's forward camera. "
_MM_IMAGES_INTRO = (
    "You see SEVERAL egocentric\n"
    "images from the drone's forward camera, all ABOVE this text: first your recent past\n"
    "views, oldest first, each labelled with its number in the action list below and the\n"
    "action you chose while looking at it; then your CURRENT view, labelled CURRENT — decide\n"
    "from that one. "
)
MM_CURRENT_LABEL = "CURRENT view — decide from this one:"


class _HistCmd(NamedTuple):
    """Command issued by one action. ``distance_m`` None = not a move command (ROTATE/THINK), 0.0 = band stop (hold)."""

    duration_s: float = 0.0
    distance_m: float | None = None

    @property
    def kind(self) -> str:
        if self.distance_m is None:
            return "turn" if self.duration_s > 0 else "none"
        return "move" if self.distance_m > 0 and self.duration_s > 0 else "hold"

    @property
    def speed_mps(self) -> float:
        return self.distance_m / self.duration_s if self.kind == "move" else 0.0


def _history_note(cmd: _HistCmd, issued: float, next_issued: float,
                  is_last: bool) -> str:
    """Of the form ``[done, 2.5s, 1.2m/s]``: run time is the monotonic issue-time difference, speed is the plan value."""
    kind = cmd.kind
    if kind == "none":
        return "[no move]"
    if kind == "hold":
        return "[hold, 0.0m/s]"
    if is_last:
        return "[in progress]"
    elapsed = max(next_issued - issued, 0.0)
    ran = min(elapsed, cmd.duration_s)
    label = "done" if elapsed >= cmd.duration_s else "interrupted"
    if kind == "turn":
        return f"[{label}, {ran:.1f}s]"
    return f"[{label}, {ran:.1f}s, {cmd.speed_mps:.1f}m/s]"


def history_notes(cmds: Sequence[_HistCmd], issued: Sequence[float], now: float) -> list[str]:
    """Per-item bracket notes. The next item's issue time bounds this item's end; the last is bounded by ``now``."""
    nxt = list(issued[1:]) + [now]
    last = len(cmds) - 1
    return [_history_note(c, t, n, i == last)
            for i, (c, t, n) in enumerate(zip(cmds, issued, nxt))]


def summarize_action(action: Action, camera: PinholeCamera,
                     scale: float = POINT_SCALE) -> str:
    """History summary string (§P3 table).

    The point goes back in the coordinates the PROMPT defines -- 0-1000 on both
    axes -- not in pixels. Writing pixels here made the history speak a different
    unit from the one the model was told to use, and a model that copies a
    coordinate out of its own history then had it rescaled by width/1000 and
    height/1000 on every round trip. On a 1280x720 frame that is x * 1.28 and
    y * 0.72 per step, so a copied x leaves the frame within about three steps.
    ``camera`` is required rather than defaulted for exactly that reason: the
    conversion must not be forgettable.
    """
    t = action.kind
    if t == ACTION_GO:
        depth = action.depth_m if action.depth_m is not None else 0
        # Round, not truncate: the pixel came from a truncation on the way in, and
        # truncating again on the way out loses another unit each round trip. With
        # rounding x is exact (1280 > 1000, so the pixel identifies the coordinate);
        # y can still land one unit off because 720 < 1000 is not injective.
        x = round(action.pixel_u / camera.width * scale)
        y = round(action.pixel_v / camera.height * scale)
        return f"GO x={x} y={y} depth={depth:g}m"
    if t == ACTION_ROTATE:
        return f"ROTATE angle={action.angle_deg:g}deg"
    if t == ACTION_THINK:
        return "THINK"
    if t == ACTION_FINISHED:
        return "FINISHED"
    return str(t)


def overlay_banner_text(action: Action) -> str:
    """Banner string drawn top-left of decision_*.jpg (§P21.1 table)."""
    t = action.kind
    if t == ACTION_ROTATE:
        return f"ROTATE {action.angle_deg:+.0f} deg"
    if t == ACTION_FINISHED:
        return "FINISHED - ARRIVAL DECLARED"
    if t == ACTION_THINK:
        return "THINK - REASON NEXT"
    if action.depth_m is None:
        return "GO"
    return f"GO depth={action.depth_m:.1f}m"


# ---------------------------------------------------------------------------
# dict -> Action (§P11–P13) — ONE implementation shared by the single and team policies
# Moved verbatim out of VlmPolicy.parse_response; behaviour unchanged.
# ---------------------------------------------------------------------------


def _drone_tail(vehicle: str) -> str:
    # The vehicle goes at the END of the [ACTION] line, never after the tag,
    # so the runner's ``^\[ACTION\]\s+([A-Z][A-Z_]*)`` still reads the action name. Empty
    # vehicle (single-drone) adds nothing — the single-path output is byte-identical.
    return f" drone={vehicle}" if vehicle else ""


def _center_action(camera: PinholeCamera, kind: str, *, angle_deg: float = 0.0, reason: str = "") -> Action:
    cx, cy = camera.center
    return Action(kind, pixel_u=cx, pixel_v=cy, angle_deg=angle_deg, reason=reason)


def action_from_dict(
    data: Mapping[str, Any],
    camera: PinholeCamera,
    point_order: str,
    *,
    vehicle: str = "",
    point_scale: float = POINT_SCALE,
) -> Action:
    """One action dict -> :class:`Action`. Raises on unusable input.

    Emits the ``[ACTION]`` / ``[PARSE]`` stdout lines at the moment the Action
    is created (before any movement can be issued). ``vehicle`` (multi-drone) appends the
    `` drone=<name>`` tail to the ``[ACTION]`` lines and nothing else.
    """
    if not isinstance(data, Mapping):
        raise TypeError(f"action JSON must be an object, got {type(data).__name__}")
    order = "xy" if str(point_order).lower() == "xy" else "yx"
    tail = _drone_tail(vehicle)

    action_name = str(data.get("action", "")).lower()
    reason = str(data.get("reason", ""))

    if action_name == ACTION_FINISHED:
        _out(f"[ACTION] FINISHED — arrival declared. reason={reason[:60]}{tail}")
        return _center_action(camera, ACTION_FINISHED, reason=reason)
    if action_name == ACTION_THINK:
        _out(f"[ACTION] THINK — hold & reason next. reason={reason[:60]}{tail}")
        return _center_action(camera, ACTION_THINK, reason=reason)
    if action_name == ACTION_ROTATE:
        raw_angle = float(data.get("angle_deg", 0.0))
        angle = max(-ROTATE_MAX_DEG, min(ROTATE_MAX_DEG, raw_angle))
        if angle != raw_angle:
            _out(f"[ACTION] ROTATE {raw_angle:+.1f}° → {angle:+.1f}° "
                 f"(clamped to ±{ROTATE_MAX_DEG:g}°) — reason={reason[:60]}{tail}")
        else:
            _out(f"[ACTION] ROTATE {angle:+.1f}° — reason={reason[:60]}{tail}")
        return _center_action(camera, ACTION_ROTATE, angle_deg=angle, reason=reason)

    # Everything else (go / missing / unknown) takes the GO path.
    if "point" not in data:
        _out(f"[ACTION] no point → default +45° scan rotation fallback{tail}")
        return _center_action(
            camera, ACTION_ROTATE, angle_deg=FALLBACK_ROTATE_DEG, reason=FALLBACK_ROTATE_REASON
        )
    point = data["point"]
    if order == "xy":
        u, v = point
    else:
        v, u = point
    scale = 1.0 if max(u, v) <= 1.0 else float(point_scale)
    if scale == 1.0:
        _out(f"[PARSE] normalized coordinates detected {point} — interpreting as 0-1 scale")
    px, py = camera.normalized_to_pixel(u, v, scale)
    # A point outside the frame is not a place the model can have seen, so it is
    # clamped for the same reason rotate is: what gets executed stays inside the
    # action space the prompt describes. Clamping in pixel space also closes the
    # boundary, where the prompt's own maximum coordinate mapped one column past
    # the frame. The original goes in the log so violations stay countable.
    cpx = min(max(px, 0), camera.width - 1)
    cpy = min(max(py, 0), camera.height - 1)
    if (cpx, cpy) != (px, py):
        _out(f"[PARSE] point {point} → pixel ({px},{py}) is outside the frame "
             f"— clamped to ({cpx},{cpy}){tail}")
    px, py = cpx, cpy
    depth = float(data.get("depth_m", data.get("depth", DEFAULT_DEPTH_M)))
    depth = min(DEPTH_MAX_M, max(DEPTH_MIN_M, depth))
    forward, right, down = camera.pixel_to_offset(px, py, depth)
    action = Action(
        ACTION_GO,
        forward=forward,
        right=right,
        down=down,
        pixel_u=px,
        pixel_v=py,
        depth_m=depth,
        reason=reason,
    )
    _out()
    _out(f"[ACTION] GO point=({px},{py}) depth_m={depth:.2f}{tail}")
    return action


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------


class VlmPolicy:
    """One VLM decision per call. Backend contract: ``generate(prompt, image_bgr, think, response_schema)``."""

    def __init__(
        self,
        backend: Any,
        camera: PinholeCamera,
        *,
        point_order: str = "yx",
        point_scale: float = POINT_SCALE,
        history_len: int = HISTORY_LEN,
        api_provider: str = "gemini",
        model_name: str = "",
        intrinsics_in_prompt: bool = False,
        canonical_depth: bool = False,
        plan_schema: bool = False,
        mm_history: bool | None = None,
        mm_history_scale: float | None = None,
        mode: str = "airsim",
        save_raw_frames: bool | None = None,
        burn_overlay: bool | None = None,
        think_action: bool = True,
    ) -> None:
        self.backend = backend
        self.camera = camera
        self.point_order = "xy" if str(point_order).lower() == "xy" else "yx"
        # The range the model writes its point in. Every model on the roster uses
        # 0-1000, which is what the prompt asks for, but the knob sits beside
        # point_order because it is the same kind of per-model convention and the
        # history has to be rendered back in whichever one the model uses.
        self.point_scale = float(point_scale) if float(point_scale) > 0 else POINT_SCALE
        self.history_len = int(history_len)
        # Adds the camera-intrinsics line to the GO bullet. Default off keeps the
        # golden prompt byte-exact.
        self.intrinsics_in_prompt = bool(intrinsics_in_prompt)
        # Require depth_m as canonical depth (config: canonical_depth).
        # Default off keeps the golden prompt byte-exact.
        self.canonical_depth = bool(canonical_depth)
        # On think steps, instead of relaxing the schema, force the extended schema and keep native reasoning
        # off. Reasoning comes out as two fields, not free text.
        self.plan_schema = bool(plan_schema)
        # Whether THINK is in the action list (config: think_action, default true = golden prompt).
        # false: prompt and schema become 3 actions (GO/ROTATE/FINISHED) and the plan field rule is dropped too.
        # If the model still emits think, execute it as hold (in place) and log it; the next step is not made a plan step.
        self.think_action = bool(think_action)
        if not self.think_action:
            _out("[POLICY] think_action OFF — 3-action prompt (GO/ROTATE/FINISHED), plan field rules excluded")
        self.api_provider = str(api_provider)
        self.model_name = resolve_model_name(self.api_provider, model_name)
        self.mode = str(mode or "airsim")   # record label: airsim / tello
        # Whether to bake the banner/target point into decision_<ts>.jpg. Real drone (tello) does **not** bake: only clean frames are saved,
        # and the markers are drawn on screen by an external viewer from decision_<ts>.json. Baked images can't be undone,
        # so they can't be used as paper figures. AirSim keeps baking since existing tools assume baked images.
        self.burn_overlay = (self.mode != "tello") if burn_overlay is None else bool(burn_overlay)
        # Only meaningful when baking: save one more original next to the baked image (_raw.jpg). Default off.
        self.save_raw_frames = bool(save_raw_frames) and self.burn_overlay
        # The four deques must share maxlen, or the item / issue time / command / view pairing drifts.
        self._history: deque[str] = deque(maxlen=self.history_len)
        self._last_frame_wh: tuple[int, int] | None = None
        self._hist_t: deque[float] = deque(maxlen=self.history_len)
        self._hist_cmd: deque[_HistCmd] = deque(maxlen=self.history_len)
        self._hist_view: deque[Any] = deque(maxlen=self.history_len)
        self.mm_history = bool(mm_history)
        # Past view width = current send width × mm_history_scale (config; 1.0 = same size as current, 0.5 = half).
        # The ratio holds even if the send width (vlm_max_width) changes. Missing or 0 = unset = backend default (same size as current).
        # Negative values are not allowed (not checked: experimenters set this by hand and only use 0.5/1.0).
        self.mm_history_scale = float(mm_history_scale) if mm_history_scale else None
        if self.mm_history and not (getattr(backend, "supports_multi", True)
                                    and hasattr(backend, "generate_multi")):
            _out(f"[MMHIST] backend {type(backend).__name__} cannot carry multiple images, turning it off")
            self.mm_history = False
        if self.mm_history:
            width = f" (past view width = current send width × {self.mm_history_scale:g})" if self.mm_history_scale else ""
            _out(f"[MMHIST] ON — attaching the views of the past {self.history_len} steps with action labels{width}")
        # M5: one informational line; must not resemble any runner marker.
        _out(
            f"[POLICY] {self.api_provider}:{self.model_name} "
            f"point_order={self.point_order} point_scale={self.point_scale:g} "
            f"history_len={self.history_len}"
        )

    @classmethod
    def from_config(cls, config: dict, camera: PinholeCamera, backend: Any,
                    *, mode: str = "airsim") -> "VlmPolicy":
        cfg = config or {}
        from vlmpilot import artifacts as _artifacts  # saved-frame size: config save_frames_at_sent_size (default on)
        _artifacts.set_save_sent_size(cfg.get("save_frames_at_sent_size"))
        return cls(
            backend,
            camera,
            point_order=str(cfg.get("point_order", "yx") or "yx"),
            point_scale=float(cfg.get("point_scale") or POINT_SCALE),
            api_provider=str(cfg.get("api_provider", "gemini") or "gemini"),
            model_name=str(cfg.get("model_name", "") or ""),
            intrinsics_in_prompt=bool(cfg.get("intrinsics_in_prompt", False) or False),
            canonical_depth=bool(cfg.get("canonical_depth", False) or False),
            plan_schema=bool(cfg.get("plan_schema", True) or False),  # ON if the key is missing from config. All 47 deployed configs are true and no config
            # uses false. A config that omitted it executed THINK via native reasoning instead of the schema and
            # the real drone crashed (router measured 35-37 s vs Tello's 15 s no-command auto-land).
            # The base/A reproduction arms set plan_schema: false explicitly.
            history_len=int(cfg.get("history_len") or HISTORY_LEN),
            mm_history=cfg.get("mm_history"),
            mm_history_scale=cfg.get("mm_history_scale"),
            mode=mode,
            save_raw_frames=cfg.get("save_raw_frames"),
            burn_overlay=cfg.get("burn_overlay"),
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

    def _note_frame(self, image_bgr) -> None:
        """Remember this step's actual frame size. The prompt states that value."""
        try:
            h, w = int(image_bgr.shape[0]), int(image_bgr.shape[1])
        except Exception:
            return
        if w > 0 and h > 0:
            self._last_frame_wh = (w, h)
    def history_items(self) -> list[str]:
        """``summary  [execution note]`` items, oldest first."""
        items = list(self._history)
        if len(self._hist_t) != len(items) or len(self._hist_cmd) != len(items):
            return items
        notes = history_notes(list(self._hist_cmd), list(self._hist_t), time.monotonic())
        return [f"{s}  {n}" for s, n in zip(items, notes)]

    def history_labels(self) -> list[str]:
        """``step i of n: summary  [execution note]``: text lines and view labels use the same sentence."""
        items = self.history_items()
        n = len(items)
        labels = [f"step {i} of {n}: {s}" for i, s in enumerate(items, start=1)]
        if labels:
            labels[-1] += "  <- latest"
        return labels

    def history_lines(self) -> str:
        if not self._history:
            return EMPTY_HISTORY_LINE
        return "\n".join(f"  {lab}" for lab in self.history_labels())

    def _mm_history_pairs(self) -> list[tuple[Any, str]]:
        """(past view, label) pairs, oldest first. Empty list when the gate is off."""
        if not self.mm_history or not self._hist_view:
            return []
        return list(zip(self._hist_view, self.history_labels()))

    def _remember_view(self, image_bgr: Any) -> None:
        if not self.mm_history or image_bgr is None:
            return
        try:
            self._hist_view.append(image_bgr)
        except Exception as exc:  # noqa: BLE001 — losing one image shifts every later pair, but the step survives
            _out(f"[MMHIST] failed to keep the past view (ignored): {exc}")

    def _mm_past_width(self) -> int | None:
        """Width cap in px for one past view = current send width × mm_history_scale. None (backend default) if scale is unset."""
        if not self.mm_history_scale:
            return None
        return max(1, int(round(self.frame_fields()["frame_w"] * self.mm_history_scale)))

    def _mm_max_widths(self, n_past: int) -> list[int | None] | None:
        """Width caps for n_past past images + 1 current. None (backend default) if unset."""
        past_w = self._mm_past_width()
        if not past_w or n_past <= 0:
            return None
        return [past_w] * n_past + [None]

    def note_command(self, duration_s: float, distance_m: float | None = None) -> None:
        """Length / travel distance of the command issued by the last action (called by the controller after plan). No-op if history is empty."""
        if self._hist_cmd:
            self._hist_cmd[-1] = _HistCmd(
                float(duration_s or 0.0),
                None if distance_m is None else float(distance_m),
            )

    def _with_intrinsics(self, template: str) -> str:
        if not self.intrinsics_in_prompt:
            return template
        line = _INTRINSICS_LINE.format(
            w=self.camera.width, h=self.camera.height,
            hfov=self.camera.fov_horizontal_deg, f=self.camera.focal_px,
        )
        return template.replace(_INTRINSICS_ANCHOR, _INTRINSICS_ANCHOR + line, 1)

    def _with_canonical_depth(self, template: str) -> str:
        # Swap only the definition sentence; the action JSON and FAR/CLOSE lines are the same in both wordings.
        if not self.canonical_depth:
            return template
        return template.replace(_DEPTH_BULLET_ACTUAL, _DEPTH_BULLET_CANONICAL, 1)

    def _with_mm_history(self, template: str) -> str:
        # The first step has no past view to attach, so the golden prompt goes out as-is.
        if not self.mm_history or not self._history:
            return template
        if _ONE_IMAGE_ANCHOR in template:
            template = template.replace(_ONE_IMAGE_ANCHOR, _MM_IMAGES_INTRO, 1)
        # History lines go out once per view label, so the block would duplicate the same content.
        return template.replace(_HISTORY_BLOCK, "", 1)

    def _with_think_action(self, template: str) -> str:
        if self.think_action:
            return template
        for anchor, repl in _NOTHINK_EDITS:
            if anchor not in template:          # the system/user split templates contain only some anchors
                continue
            template = template.replace(anchor, repl, 1)
        return template

    def _template(self, base: str) -> str:
        return self._with_mm_history(
            self._with_intrinsics(self._with_canonical_depth(self._with_think_action(base))))

    def build_prompt(self, instruction: str) -> str:
        return self._template(_PROMPT_TEMPLATE).format(
            instruction=instruction,
            point_fmt=self.point_fmt,
            history_len=self.history_len,
            history=self.history_lines(),
            **self.frame_fields(),
        )

    def build_system_prompt(self) -> str:
        """system part: identical every step (no command, no history); the span the prefix cache covers."""
        template = self._template(_SPLIT_SYSTEM_TEMPLATE)
        if self.plan_schema and self.think_action:
            template = template.rstrip("\n") + "\n\n" + _PLAN_FIELDS_RULE.rstrip("\n")
        return template.format(point_fmt=self.point_fmt, **self.frame_fields())

    def build_user_prompt(self, instruction: str) -> str:
        """user part: ``Command:`` + history + closing line. Same on plan steps:
        the schema enforces both fields as required, so no per-step instruction is needed."""
        return self._with_mm_history(_SPLIT_USER_TEMPLATE).format(
            instruction=instruction,
            history_len=self.history_len,
            history=self.history_lines(),
        )

    # -- parsing --------------------------------------------------------------

    @staticmethod
    def _load_json(text: str) -> Any:
        try:
            return json.loads(text)
        except Exception as first_err:
            m = re.search(r"\{.*\}", text, re.DOTALL)
            if not m:
                raise first_err
            return json.loads(m.group(0))

    def _read_plan_fields(self, text: str) -> dict | None:
        """Read the observation and plan fields from a plan step's response. None on failure."""
        return read_plan_fields(text)

    def parse_response(self, text: str) -> Action:
        """Model raw text -> Action (§P9–P13). Raises on unusable input.

        Also emits the ``[ACTION]`` / ``[PARSE]`` stdout lines of §5 at the moment the
        Action is created (which is before any movement can be issued).
        """
        cleaned = _extract_action_json(text)
        data = self._load_json(cleaned)
        if isinstance(data, list):
            data = data[0] if data else {}
        if not isinstance(data, dict):
            raise TypeError(f"action JSON must be an object, got {type(data).__name__}")

        return action_from_dict(data, self.camera, self.point_order,
                                point_scale=self.point_scale)

    # -- think toggle -----------------------------------------------------------

    def _think_flag(self) -> bool:
        # P16: only when the *last* history entry starts with THINK.
        if not self.think_action:
            return False
        return bool(self._history) and self._history[-1].startswith("THINK")

    # -- record -------------------------------------------------------------------

    def _make_overlay(self, image_bgr: np.ndarray, action: Action) -> np.ndarray:
        canvas = np.array(image_bgr, copy=True)
        cv2.putText(
            canvas,
            overlay_banner_text(action),
            _BANNER_ORG,
            _BANNER_FONT,
            _BANNER_SCALE,
            _BANNER_COLOR_BGR,
            _BANNER_THICKNESS,
            cv2.LINE_AA,
        )
        # Target point and displacement are baked only on **steps that actually move (go)**.
        # For rotate/think/finished, pixel_u/v is not a model value but the camera principal point filled by _center_action,
        # and R/F/U are always 0; left on the frame it reads as "the model picked here and moves 0".
        # Kept in the record (decision_*.json), removed only from the image. The banner stays: the step type must be visible.
        if action.kind != ACTION_GO:
            return canvas
        cx = int(round(float(action.pixel_u)))
        cy = int(round(float(action.pixel_v)))
        cv2.circle(canvas, (cx, cy), _MARK_RADIUS, _MARK_COLOR_BGR, _MARK_THICKNESS, cv2.LINE_AA)
        label = f"R{action.right:+.1f} F{action.forward:+.1f} U{action.up:+.1f}"
        cv2.putText(
            canvas,
            label,
            (cx + _MARK_RADIUS + 6, cy + 6),
            _LABEL_FONT,
            _LABEL_SCALE,
            _LABEL_COLOR_BGR,
            _LABEL_THICKNESS,
            cv2.LINE_AA,
        )
        return canvas

    def _make_payload(self, timestamp: str, instruction: str, action: Action,
                      think: bool = False, thoughts: str | None = None,
                      vlm_ms: float | None = None, plan: dict | None = None) -> dict:
        return {
            "timestamp": timestamp,
            "mode": self.mode,
            "frame_clean": not self.burn_overlay,      # True means decision_<ts>.jpg has no markers
            "api_provider": self.api_provider,
            "model": self.model_name,
            "instruction": instruction,
            # ``think`` is what the scaffold asked for; ``reasoning`` is what came
            # back. Only the second says whether the endpoint actually spent
            # tokens, and nothing else in the run records it. It goes in the step
            # record rather than the episode log: reasoning content runs to
            # thousands of lines and would bury the log it was added to.
            "think": bool(think),
            "reasoning_chars": len(thoughts) if thoughts else 0,
            "reasoning": thoughts,
            # Wall time of the backend call alone. Together with the controller's
            # timing_ms (sim RPC time) this separates "the model was slow" from
            # "the simulator was overloaded" — the two are indistinguishable in
            # steps-per-second and that number decides how many decisions fit in
            # an episode cap.
            "vlm_ms": vlm_ms,
            # The plan step's observation and plan fields. Unlike free reasoning, the schema enforces them.
            "plan": plan,
            "actions": [dataclasses.asdict(action)],
        }

    # -- main entry -------------------------------------------------------------

    def call_backend(self, instruction: str, image_bgr, *, think: bool,
                     schema: Mapping[str, Any] | None) -> tuple[Any, str | None]:
        """One VLM call → (raw, thoughts). If the backend can carry multiple images, send split
        system/user + past views; otherwise a single prompt. The body in head/body mode also
        calls this method, so its prompt and schema match the centralized arm."""
        # Part order: [prompt, past label, past view, …, CURRENT label, current view]
        past = self._mm_history_pairs()
        imgs = [v for v, _ in past] + [image_bgr]
        labels = [lab for _, lab in past] + [MM_CURRENT_LABEL if self.mm_history else ""]
        widths = self._mm_max_widths(len(past))
        extra = {} if widths is None else {"max_widths": widths}
        # Every backend has the generate_multi name; branch on supports_multi.
        can_multi = (getattr(self.backend, "supports_multi", True)
                     and (hasattr(self.backend, "generate_multi")
                          or hasattr(self.backend, "generate_multi_detailed")))
        if can_multi:
            multi_kwargs = dict(think=think, response_schema=schema,
                                system=self.build_system_prompt(), **extra)
            prompt = self.build_user_prompt(instruction)
            if hasattr(self.backend, "generate_multi_detailed"):
                result = self.backend.generate_multi_detailed(prompt, imgs, labels, **multi_kwargs)
                return result.text, result.thoughts
            return self.backend.generate_multi(prompt, imgs, labels, **multi_kwargs), None
        if hasattr(self.backend, "generate_detailed"):
            result = self.backend.generate_detailed(self.build_prompt(instruction), image_bgr,
                                                    think=think, response_schema=schema)
            return result.text, result.thoughts
        return self.backend.generate(self.build_prompt(instruction), image_bgr,
                                     think=think, response_schema=schema), None

    def remember(self, action: Action, image_bgr) -> str:
        """Record this step in history and return its summary line: summary, issue time, command slot, (if on) view."""
        summary = summarize_action(action, self.camera, self.point_scale)
        self._history.append(summary)
        self._hist_t.append(time.monotonic())
        self._hist_cmd.append(_HistCmd())   # filled by the controller via note_command after plan
        self._remember_view(image_bgr)
        return summary

    def decide(self, image_bgr: np.ndarray | None, instruction: str) -> tuple[Action | None, DecisionRecord | None]:
        """One step. ``(Action, DecisionRecord)`` on success, ``(None, None)`` when unusable.

        Never raises (P14).
        """
        if image_bgr is None:
            _out("[POLICY] input frame is None — skipping this step (no backend call)")
            return None, None

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        raw: Any = None
        self._note_frame(image_bgr)
        try:
            think = self._think_flag()
            if think:
                _out("[THINK] previous THINK → thinking ON this step")
            # The detailed variants also return the reply's reasoning text, which
            # is the only evidence of deliberation a run keeps; fall back to the
            # bare contract for a custom backend that implements only that.
            thoughts: str | None = None
            # plan_schema: take this step's reasoning as schema fields, not free text.
            # Native reasoning stays off (think=False to the backend): if on, the grammar binds from the first token
            # so the reasoning span never gets generated (local), or reasoning eats the response budget (router).
            plan_step = bool(think and self.plan_schema)
            schema = PLAN_RESPONSE_SCHEMA if plan_step else action_response_schema(self.think_action)
            want_think = False if plan_step else think
            vlm_t0 = time.monotonic()
            raw, thoughts = self.call_backend(instruction, image_bgr,
                                              think=want_think, schema=schema)
            vlm_ms = round((time.monotonic() - vlm_t0) * 1000.0, 1)
            if raw is None:
                _out("[POLICY] no backend reply (None) — no decision this step")
                return None, None
            # Episode log is the only place the raw reply is kept. Each line is prefixed so a
            # model that echoes marker-like text can never satisfy a runner regex.
            for line in str(raw).splitlines() or [""]:
                _out(f"[RAW] {line}")

            plan = self._read_plan_fields(str(raw)) if plan_step else None
            action = self.parse_response(str(raw))
            if not self.think_action and action.kind == ACTION_THINK:
                _out("[ACTION] THINK is not in this arm's action list — executed as hold (think_action OFF)")
            self.remember(action, image_bgr)

            overlay = self._make_overlay(image_bgr, action) if self.burn_overlay else np.array(image_bgr, copy=True)
            payload = self._make_payload(timestamp, instruction, action, think, thoughts,
                                         vlm_ms=vlm_ms, plan=plan)
            raw = np.array(image_bgr, copy=True) if self.save_raw_frames else None
            return action, DecisionRecord(timestamp=timestamp, overlay_image_bgr=overlay, payload=payload,
                                          raw_image_bgr=raw)
        except Exception as exc:  # noqa: BLE001 — P14: swallow everything
            _out(f"[POLICY] decision failed ({type(exc).__name__}: {exc}) — no decision this step")
            if raw is not None:
                _out(f"[POLICY] failed raw text: {str(raw)[:500]!r}")
            return None, None
