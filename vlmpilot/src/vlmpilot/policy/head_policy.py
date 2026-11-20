# DroneCATS
# Copyright (c) 2026-present NAVER Cloud Corp.
# Apache-2.0

"""head_policy.py — the orchestrator of ``agent_mode: head_body``.

The head is a big model that never touches a control surface: it grabs every drone's view
**silently** (no ``[FRAME]`` marker — those belong to the bodies, and the runner reads a pose at
each one) and rewrites their standing orders. A body is stateless, so every instruction has to
be self-contained. Every call and skipped wake, with the turn and frames it sent, goes to
``head_log.jsonl`` (``VLMPILOT_HEAD_TRACE=0`` drops the frames)."""

from __future__ import annotations

import json
import os
import re
import threading
import time
from typing import Any, Callable, Mapping, Sequence

import cv2
import numpy as np

from vlmpilot.policy.body_policy import say
from vlmpilot.policy.camera import PinholeCamera
from vlmpilot.policy import head_map
from vlmpilot.controller.airsim.fleet import grab_scene_bgr
from vlmpilot.policy.policy import _extract_action_json, resolve_model_name

MAP_TRAIL_POINTS = 12          # recent path drawn on the overview map (≈ the last 12 head calls)

__all__ = ["HeadAgent", "HEAD_LOG_NAME", "HEAD_FRAMES_DIR", "HEAD_TRACE_ENV", "HOLD_KEYWORD",
           "STATE_SCHEMA_MARKER", "is_hold", "HeadPromptError", "resolve_prompt", "PROMPT_MAP",
           "PROMPT_DEFAULT", "PROMPT_ONESHOT"]

HEAD_LOG_NAME = "head_log.jsonl"
HEAD_FRAMES_DIR = "head_frames"
HEAD_TRACE_ENV = "VLMPILOT_HEAD_TRACE"   # default ON; "0" turns it off
HEAD_TRACE_JPEG_QUALITY = 75      # for archiving: storage quality, not the quality sent to the model
TRACE_MISSING = "missing"         # drone with no frame this turn (a black placeholder was used instead)

DEFAULT_POLL_INTERVAL_S = 12.0
DEFAULT_MAX_CALLS = 200        # continuous-loop budget (poll_interval_s: 0)
CONTINUOUS_FLOOR_S = 2.0       # floor in the continuous loop, enough to never see the same frame twice
DEFAULT_HISTORY_TURNS = 4
DEFAULT_ORDER_TTL_STEPS = 4     # step budget a body executes per order
MIN_CALL_GAP_S = 5.0
INSTRUCTION_ECHO_CHARS = 120
# Cap when feeding the ledger notes (``state``) back into the prompt. Only the latest turn is sent in full up to this cap.
LEDGER_ECHO_CHARS = 300

# Scheduling keyword, not an order. It is consumed by :meth:`HeadAgent._hold` and NEVER handed
# to a body: the Command: slot would carry a harness word instead of an instruction (arm
# invariant). To a parked drone it means "stay parked", to a flying one "carry on".
HOLD_KEYWORD = "HOLD"

# A template asking for ``{"Drone1": {"state": …, "order": …}}`` says so with this phrase, and
# ``_example()`` then renders the two-field shape — a template that asks for two fields while
# the example shows one contradicts itself, and the model answers in whichever shape it believes.
STATE_SCHEMA_MARKER = '"state": your ledger note'

_TRIGGER_POLL = "poll"
_TRIGGER_DECLARATION = "declaration"
_TRIGGER_INITIAL = "initial"

# Rendered with a single-pass substitution so the JSON example's braces need no escaping.
# This is the prompt the 2nd experiment actually used; byte-identical to head_variants/head_policy.v22_diet3.py.
# Change a single character and the table's numbers no longer reproduce from this code.
_SYSTEM_TEMPLATE = """You are the COMMANDER of {n} autonomous search drones in a simulator. STRONG BODY: each drone
is a capable pilot that flies itself — you orchestrate, you do not pilot.

1. YOUR DRONES. One forward-camera view per drone, in roster order. Each drone is its own
   vision model: it sees ONLY its current frame, YOUR latest order and its own last few actions
   (no map, no memory of this conversation). Given a destination described as a THING it flies
   there on its own, can read text on the way, and outputs FINISHED when it is at it. FINISHED
   makes it STOP and WAIT for your next order (its report says "WAITING"). To keep a WAITING
   drone parked, its "order" is exactly HOLD; for a flying drone HOLD means "carry on with your
   current order". Any other text restarts it under that text. EVERY ORDER MUST BE CONVERGENT: it re-reads your order
   on every step with no memory of progress, so name a thing to reach — never a fixed angle or
   a fixed distance, and never a step size.

2. YOUR JOB IS THE SEARCH, NOT THE FLYING. Keep a ledger of candidates: UNCHECKED, RULED OUT,
   CONFIRMED. From the first turn send each drone to a DIFFERENT unchecked candidate or sector
   and keep it there until that candidate is decided. Never send two drones to the same
   candidate. The moment a drone is WAITING or its candidate is ruled out, give it the nearest
   unchecked one. Only ONE drone has to reach the target.

3. EVERY ORDER HAS TWO PARTS: a) NARRATE what that drone's camera is showing you right now,
   plainly; b) its destination and what to do there, in terms of THINGS and their relations:
   "Your view: three tall structures ahead. Go to the marker at the foot of the leftmost one and
   read it. If it is the target — {mission} — stay right at its base and output FINISHED. If it
   is not, do not stop: fly on to the marker between the first and second structures and read
   that one." Include the mission text in every order so the drone can match it by itself.
   FINISHED is for the confirmed target ONLY — a drone must never stop at a non-target; on a
   ruled-out candidate it goes straight on to the next unchecked one you name.

4. FORBIDDEN: NEVER write pixel coordinates — no [x,y], no numbers in brackets (the drone copies
   them back on later frames where they point at nothing). NEVER anchor to the momentary picture
   ("the blob on the left of your view"). NEVER name an angle or a turn ("turn around", "180
   degrees") — a turn order makes the drone spin in place for many steps; name the thing to go
   to instead. DO reference objects and their relations: counting, size, colour, what a thing
   stands between, behind or past. No step sizes, no "slowly".

5. YOU ARE THE ONLY READER of the whole picture: drones do not see each other's frames. Share
   what matters in each order ("the marker between the two tall structures is ruled out"), and
   use what a drone reads to update the ledger for everyone. A candidate counts as READ only
   when every letter is fully inside the frame and legible — if any letter touches the frame
   edge or is blurred, it is NOT read; send the drone closer instead of deciding. A drone's
   FINISHED is not evidence about the marker: only the letters are. Never invent candidates:
   the ledger holds only markers you have actually seen in a drone's frame.

6. WHEN A DRONE REPORTS FINISHED: look at its frame. Target confirmed and it stands at the
   target -> "You have arrived at the target. FINISHED." If it stopped short (target still a
   few meters ahead, the mission has not ended) -> "Move right up to the base of the <target>
   and output FINISHED." If it is at a non-target -> ruled out; send it to the next candidate.

7. STUCK DRONES. If a drone's report says it moved less than a meter since your last order
   while it was told to fly, it is caught on a structure: do NOT repeat the order — send it to
   the END of the structure in front of it ("go around the near end of the tall structure on
   your left, then …") and re-issue the destination after it has moved. Keep orders short: one
   or two sentences per drone, no rules repeated to the drone.

An order expires after {ttl} steps — a drone with an expired order HOLDS in place until you
speak to it again. Refresh every drone's order on every turn; silence stops that drone.

## Drone roster (image order = this order)
{roster}

## The mission
{mission}

## Output — ONE JSON object; for each drone an object with two fields:
  "state": your ledger note for that drone (what it is checking, what is ruled out/confirmed,
           stuck or waiting) — this is for you, the drone never sees it;
  "order": the sentence(s) the drone receives — or exactly HOLD.
Shape (fill every drone):
{example}
Output ONLY the JSON object. No text outside it.
"""


# The one-shot arm's prompt (`head.prompt: "oneshot"`): the commander speaks once, at t0, and is
# never woken again. Byte-identical to head_variants/head_policy.v22_oneshot.py.
_SYSTEM_TEMPLATE_ONESHOT = """You are the COMMANDER of {n} autonomous search drones in a simulator. STRONG BODY: each drone
is a capable pilot that flies itself — you orchestrate, you do not pilot.

THIS IS A ONE-SHOT COMMAND. You speak to your drones EXACTLY ONCE, right now. There is no
second turn: no reports come back, you cannot re-task, correct, or HOLD anyone. Whatever you
write now is the whole plan each drone will follow for the entire mission, so every order must
be COMPLETE and SELF-SUFFICIENT — a plan the drone can finish alone.

1. YOUR DRONES. One forward-camera view per drone, in roster order. Each drone is its own
   vision model: it sees ONLY its current frame, YOUR order and its own last few actions (no
   map, no memory of this conversation). Given destinations described as THINGS it flies to
   them on its own and can read text when it is close. It re-reads your order on every step, so
   everything it must do for the whole mission has to be in the order: where to go, in what
   sequence, what to read, when to stop, and what to do when a marker is not the target.

2. YOUR JOB IS COVERAGE. Split the scene into {n} DIFFERENT search sectors or candidate chains
   so that together the drones cover every marker you can see or infer, with no two drones on
   the same chain. Give each drone an ORDERED route through the markers in its sector, first the
   nearest, then onward. Only ONE drone has to reach the target, so spread out.

3. EVERY ORDER IS A COMPLETE MISSION SCRIPT, in this shape: a) one sentence on what that
   drone's camera shows now; b) the mission text verbatim — {mission} — so the drone can match
   it by itself; c) its route as a sequence of THINGS with their relations ("first the marker at
   the foot of the tall structure ahead, then the marker between the two lower structures behind
   it, then whatever markers stand beyond them"); d) the standing rule, spelled out: at each
   marker fly right up to its base and read it; if the letters match the mission, stay at its
   base and output FINISHED; if they do not, do NOT stop and do NOT output FINISHED — fly on to
   the next marker in the route; when the route is exhausted, keep flying toward structures it
   has not passed yet and read every marker it meets. FINISHED is for the confirmed target ONLY.

4. FORBIDDEN: NEVER write pixel coordinates — no [x,y], no numbers in brackets. NEVER anchor to
   the momentary picture ("the blob on the left of your view"). NEVER name an angle, a turn or a
   distance ("turn around", "180 degrees", "fly 20 meters") — name the thing to go to instead.
   NEVER write HOLD or "wait for further orders": nobody will speak to this drone again. DO
   reference objects and their relations: counting, size, colour, what a thing stands between,
   behind or past. No step sizes, no "slowly".

5. YOU ARE THE ONLY READER of the whole picture: drones do not see each other's frames. If one
   frame already shows legible letters that match the mission, send the nearest drone straight
   to that marker and still give every other drone a full route (the reading may be wrong).
   Never invent markers you have not seen; where you see none, assign a direction of structures
   to explore and the rule of reading every marker on the way.

## Drone roster (image order = this order)
{roster}

## The mission
{mission}

## Output — ONE JSON object; for each drone an object with two fields:
  "state": your ledger note for that drone (its sector, its route, why) — this is for you,
           the drone never sees it;
  "order": the complete mission script the drone receives (3–6 sentences). Never HOLD.
Shape (fill every drone):
{example}
Output ONLY the JSON object. No text outside it.
"""

_INITIAL_USER = """This is the start of the mission. The {n} images are the drones' current views,
in roster order.

Assign EVERY drone a task now, spread across different directions. Output the JSON object with
all {n} drones."""

_POLL_USER = """## Earlier exchanges
{transcript}

## Interval report — woken by {trigger} at t+{elapsed:.0f}s
{report}

{images_line} For EVERY drone output its state
and its order. Output ONLY the JSON object."""

_IMAGES_LINE = "The {n} images are the drones' CURRENT views, in roster order."
_IMAGES_LINE_MAP = ("The first {n} images are the drones' CURRENT views, in roster order; the LAST image is the "
                    "overview map.")

_PLACEHOLDERS = re.compile(r"\{(n|roster|mission|example|ttl)\}")

PROMPT_DEFAULT = "default"
PROMPT_ONESHOT = "oneshot"
PROMPT_MAP = "map"
# The control-tower arm (``head.prompt: map`` + ``head.map: true``): the default prompt plus one
# section on the overview map. Derived from the default at call time so the seven rules stay
# byte-equal and a spliced variant file still wins.
_MAP_SECTION = """
8. THE OVERVIEW MAP. Your LAST image is a top-down radar map of the whole area (north is up; the
   bar at the bottom-left is 10 m). It is drawn from the simulator, not by a camera: grey bars are
   tall structures, coloured shapes are coloured markers drawn with the footprint of the thing
   they mark, so a long one shows which way that thing lies (their COLOUR and OUTLINE only —
   NO text or number is drawn, so the map can never tell you what a marker says), hollow circles
   are the start slots, and each drone is a numbered dot with an arrow showing where its camera points and a
   line showing its recent path. USE THE MAP TO ORCHESTRATE: send drones to DIFFERENT unchecked
   markers, never two to one already read; tell a drone which way its destination lies relative
   to what its camera shows (its left/right is the arrow's left/right); notice a drone that has
   not moved, sits in the wrong lane, or wandered off toward distant scenery, and bring it back.
   BEFORE you let a drone commit to a destination it must enter (a lane, a passage, an opening),
   check on the map that the drone is lined up with THAT one and not its neighbour — a drone
   that has gone one lane over looks the same from inside; only the map tells them apart. If it
   is in the wrong one, say so and send it back out and into the right one.
   The map is NOT evidence of what a marker says — only a drone's own frame can read it. Never
   send map positions or metres to a drone: it has no map and no compass.
"""


def _map_template() -> str:
    """The default template *as currently bound* (a variant file may have spliced it) plus §8."""
    anchor = "\nAn order expires after {ttl} steps"
    base = _SYSTEM_TEMPLATE
    return base.replace(anchor, _MAP_SECTION + anchor, 1) if anchor in base else base + _MAP_SECTION


_SYSTEM_TEMPLATE_MAP = _map_template()      # for inspection; the registry re-derives it per call
# Each entry is a *lambda*, not the string: a variant file from infra-multiagent/head_variants/
# is spliced over this module by rebinding the module-level template, and a mapping that captured
# the value at import time would keep serving the old prose for the rest of the episode.
_TEMPLATES: Mapping[str, Any] = {
    PROMPT_DEFAULT: lambda: _SYSTEM_TEMPLATE,
    PROMPT_ONESHOT: lambda: _SYSTEM_TEMPLATE_ONESHOT,
    PROMPT_MAP: _map_template,
}


class HeadPromptError(ValueError):
    """`head.prompt:` names a template that does not exist."""


def resolve_prompt(name: Any) -> str:
    """Validate a ``head.prompt:`` value, or raise with the choices in the message."""
    key = str(name or PROMPT_DEFAULT).strip().lower()
    if key not in _TEMPLATES:
        choices = ", ".join(sorted(_TEMPLATES))
        raise HeadPromptError(f"head.prompt={name!r} is not a known prompt — valid values: {choices}")
    return key


# A prompt that makes every order narrate the view cannot produce the bare word — the head
# writes "Your view: the pink marker ahead. HOLD." — so HOLD also counts as the answer's LAST
# token. A "hold" mid-sentence ("hold your course toward the gate") stays an ordinary order.
_HOLD_RE = re.compile(r"(?:^|[.!?\s])" + HOLD_KEYWORD + r"[.!]?\s*$", re.IGNORECASE)

# Under the two-field schema the narration lives in ``state``, so a sentence that merely ENDS
# in HOLD is a real order (the relaxed rule ate real stall-escapes) — it must BE the keyword.
_HOLD_STRICT_RE = re.compile(r"^" + HOLD_KEYWORD + r"[.!]?$", re.IGNORECASE)


def is_hold(text: Any, *, strict: bool = False) -> bool:
    """``HOLD`` alone, or (unless ``strict``) as the trimmed answer's last token."""
    stripped = str(text).strip()
    pattern = _HOLD_STRICT_RE if strict else _HOLD_RE
    return pattern.search(stripped) is not None


def _fold(text: Any, limit: int = INSTRUCTION_ECHO_CHARS) -> str:
    """One line, bounded — head text reaches stdout, which the runner scans line by line."""
    folded = " ".join(str(text).split())
    return folded if len(folded) <= limit else folded[: limit - 1] + "…"


class HeadAgent:
    """Multi-turn orchestrator over N :class:`~vlmpilot.policy.body_policy.BodyAgent` s."""

    def __init__(
        self,
        *,
        backend: Any,
        bodies: Sequence[Any],
        clients: Sequence[Any],
        camera: PinholeCamera,
        camera_name: str,
        mission_text: str,
        run_dir: str | None = None,
        poll_interval_s: float = DEFAULT_POLL_INTERVAL_S,
        max_calls: int = DEFAULT_MAX_CALLS,
        history_turns: int = DEFAULT_HISTORY_TURNS,
        order_ttl_steps: int = DEFAULT_ORDER_TTL_STEPS,
        min_call_gap_s: float = MIN_CALL_GAP_S,
        api_provider: str = "gemini",
        model_name: str = "",
        prompt: str = PROMPT_DEFAULT,
        frame_source: Callable[[int], np.ndarray | None] | None = None,
        map_on: bool = False,
        scene: Mapping[str, Any] | None = None,
    ) -> None:
        self.backend = backend
        # Control-tower arm: a text-free top-down map as the last image (head_map). Off = the
        # shipped head, byte-for-byte; the scene geometry comes from the runner's episode config.
        self.map_on = bool(map_on)
        self.scene: dict = dict(scene or {})
        self._trails: dict[str, list[list[float]]] = {}
        self._last_map: np.ndarray | None = None
        self.bodies = list(bodies)
        self.drone_names = [b.name for b in self.bodies]
        # One per drone: driving one client from two threads wedges the msgpack-rpc IOLoop permanently.
        self.clients = list(clients)
        self.camera = camera
        self.camera_name = str(camera_name)
        self.mission_text = str(mission_text)
        self.run_dir = run_dir
        self.poll_interval_s = float(poll_interval_s)
        self.max_calls = int(max_calls)
        self.history_turns = max(1, int(history_turns))
        self.order_ttl_steps = int(order_ttl_steps)
        self.min_call_gap_s = float(min_call_gap_s)
        self.api_provider = str(api_provider)
        # Injection point for setups that get frames from somewhere other than AirSim RPC (e.g. an external viewer's live step).
        self.frame_source = frame_source
        self.model_name = resolve_model_name(self.api_provider, model_name)
        self.prompt = resolve_prompt(prompt)

        self.calls = 0
        self.turns: list[dict] = []
        self._t0 = time.time()
        self._last_call_t = 0.0
        self._blind: set[str] = set()  # drones that got no frame this turn (stated in the report)
        self._last_report_xy: dict[str, tuple[float, float]] = {}   # reference point for "how far did it move"
        self._last_states: dict[str, str] = {}
        self._schema_names: set[str] = set()
        say(f"[HEAD] {self.api_provider}:{self.model_name} drones={','.join(self.drone_names)} "
            f"prompt={self.prompt} poll={self.poll_interval_s:g}s max_calls={self.max_calls} "
            f"history_turns={self.history_turns} map={'on' if self.map_on else 'off'}")
        if self.map_on and not self.scene:
            say("[HEAD] map=on but there is no scene geometry — drawing drone positions only (no structures or signs)")

    def roster_lines(self) -> str:
        return "\n".join(f"  - DRONE {i} = {name}" for i, name in enumerate(self.drone_names, start=1))

    def system_template(self) -> str:
        """This head's template, read fresh on every call so a spliced variant still wins."""
        return _TEMPLATES[self.prompt]()

    def _example(self) -> str:
        """The output shape, in whichever schema THIS head's template asks for."""
        if STATE_SCHEMA_MARKER in self.system_template():
            body = ",\n".join(f'  "{name}": {{"state": "<ledger note for {name}>",'
                              f' "order": "<order for {name} or HOLD>"}}'
                              for name in self.drone_names)
        else:
            body = ",\n".join(f'  "{name}": "<order for {name}>"' for name in self.drone_names)
        return "{\n" + body + "\n}"

    def build_system_prompt(self) -> str:
        values = {
            "n": str(len(self.drone_names)),
            "roster": self.roster_lines(),
            "mission": self.mission_text,
            "example": self._example(),
            "ttl": str(self.order_ttl_steps) if self.order_ttl_steps > 0 else "several",
        }
        return _PLACEHOLDERS.sub(lambda m: values[m.group(1)], self.system_template())

    def image_labels(self) -> list[str]:
        labels = [f"DRONE {i + 1} ({name}) view:" for i, name in enumerate(self.drone_names)]
        if self.map_on:
            labels.append(head_map.MAP_LABEL)
        return labels

    def images_line(self) -> str:
        """The sentence that tells the head how many images follow and what the last one is —
        with the map on there are N+1, and saying "N" would call the map a drone view."""
        return (_IMAGES_LINE_MAP if self.map_on else _IMAGES_LINE).format(n=len(self.drone_names))

    def render_map(self) -> np.ndarray | None:
        """The overview map for this call: live GLOBAL positions (simGetObjectPose on the head's
        own clients) over the scene geometry, plus each drone's recent path. None when map is off."""
        if not self.map_on:
            return None
        positions = head_map.global_positions(self.clients, self.drone_names)
        for name, d in positions.items():
            if d and d.get("xy"):
                trail = self._trails.setdefault(name, [])
                trail.append(list(d["xy"]))
                del trail[:-MAP_TRAIL_POINTS]
        try:
            self._last_map = head_map.render(self.scene, positions, self._trails)
        except Exception as exc:  # noqa: BLE001 — a drawing bug must not cost a head call
            say(f"[HEAD] map_render_failed err={type(exc).__name__}: {exc}")
        return self._last_map

    def kept_turns(self) -> list[dict]:
        """Turn 0 (mission + first assignment) always, then a window of the most recent turns."""
        if len(self.turns) <= self.history_turns:
            return list(self.turns)
        # history_turns == 1 means turn 0 alone: turns[-0:] is the WHOLE list, not the empty tail.
        recent = self.turns[-(self.history_turns - 1):] if self.history_turns > 1 else []
        return [self.turns[0]] + recent

    def transcript(self) -> str:
        """The kept turns as prompt lines: what you ordered, and what your ledger said. Only
        the newest ledger is echoed whole — unread, it decays into an activity log in 5 calls."""
        lines: list[str] = []
        kept = self.kept_turns()
        for turn in kept:
            ordered = "; ".join(f'{k}="{_fold(v)}"' for k, v in (turn.get("instructions") or {}).items())
            line = f"  [t+{turn['elapsed']:.0f}s, {turn['trigger']}] you ordered: {ordered or '(nothing)'}"
            states = turn.get("states") or {}
            if states:
                limit = LEDGER_ECHO_CHARS if turn is kept[-1] else INSTRUCTION_ECHO_CHARS
                line += " | ledger: " + "; ".join(f'{k}="{_fold(v, limit)}"' for k, v in states.items())
            lines.append(line)
        return "\n".join(lines) or "  (none — this is your first update)"

    def interval_report(self) -> str:
        lines: list[str] = []
        for body in self.bodies:
            status = body.status()
            parts = ["; ".join(status["last_actions"]) or "(no action yet)"]
            if status["name"] in self._blind:
                # Reading a black placeholder as "nothing there" would turn back a drone that is approaching.
                parts.append("(camera unavailable this turn — the image for this drone is blank)")
            pos = status.get("last_pos")
            if pos:
                parts.append(f"pos x={pos.get('x')} y={pos.get('y')} alt={pos.get('alt_m')}m yaw={pos.get('yaw_deg')}°")
                moved = self._moved_since_last_report(status["name"], pos)
                if moved is not None:
                    # The only evidence for stall diagnosis: the report has no previous pos, so the commander can't compare.
                    parts.append(f"moved {moved:.1f} m since last report")
                # How many times the commander's order was actually re-executed; data, not a rule.
            parts.append(f"executed your last order {status.get('order_steps', 0)} steps"
                         f" over {float(status.get('order_age_s', 0.0)):.0f} s")
            if status.get("waiting"):
                # When it stopped, in mission time: elapsed now − time standing = declaration time.
                since = max(0.0, (time.time() - self._t0) - float(status.get("waiting_age_s", 0.0)))
                parts.append(f"WAITING since declaration at t+{since:.0f}s"
                             " — send a new order to move it, or HOLD to keep it parked")
            elif status["declared"]:
                parts.append("DECLARED FINISHED (declared at dist unknown — verify from its frame)")
            if status.get("order_expired") and not status.get("waiting"):
                # A THINK from a drone with an exhausted budget is a freeze, not obedience. Unless told, the commander
                # reads it as "holding nicely" and issues HOLD again (self-reinforcing deadlock).
                parts.append("ORDER EXPIRED — holding until you speak")
            lines.append(f"  {status['name']}: " + " | ".join(parts))
        return "\n".join(lines)

    def _moved_since_last_report(self, name: str, pos: Mapping) -> float | None:
        """Planar distance from the previous report's position, or None the first time. Planar
        on purpose: altitude jitter would hide a horizontal stall behind a fake metre."""
        try:
            here = (float(pos.get("x")), float(pos.get("y")))
        except (TypeError, ValueError):
            return None
        there = self._last_report_xy.get(name)
        self._last_report_xy[name] = here
        if there is None:
            return None
        return float(np.hypot(here[0] - there[0], here[1] - there[1]))

    def grab_frames(self) -> list[np.ndarray | None]:
        """Every drone's view at once, on the head's OWN clients, without a ``[FRAME]`` line —
        the bodies own that marker."""
        n = len(self.drone_names)
        frames: list[np.ndarray | None] = [None] * n

        def _one(i: int) -> None:
            name = self.drone_names[i]
            try:
                frames[i] = (self.frame_source(i) if self.frame_source is not None
                             else grab_scene_bgr(self.clients[i], self.camera_name, name))
            except Exception as exc:  # noqa: BLE001 — one blind drone must not blind the head
                say(f"[HEAD] frame_failed drone={name} err={type(exc).__name__}")

        threads = [threading.Thread(target=_one, args=(i,), daemon=True,
                                    name=f"vlmpilot-head-capture-{i}") for i in range(n)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self._blind = {self.drone_names[i] for i in range(n) if frames[i] is None}
        return frames

    def _generate(self, user_prompt: str,
                  frames: Sequence[np.ndarray]) -> tuple[str | None, Any, str | None]:
        """(text, usage, failure reason). The third value is set only when the backend swallowed an exception."""
        args = (user_prompt, frames, self.image_labels())
        kwargs = {"think": False, "system": self.build_system_prompt()}
        detailed = getattr(self.backend, "generate_multi_detailed", None)
        if not callable(detailed):
            return self.backend.generate_multi(*args, **kwargs), None, None
        result = detailed(*args, **kwargs)
        return (getattr(result, "text", None), getattr(result, "usage", None),
                getattr(result, "error", None))

    def parse_instructions(self, raw: str) -> dict[str, str]:
        """``{"Drone1": "<order>", …}`` → orders keyed by real vehicle name; raises if unusable.
        A value may also be ``{"state": …, "order": …}``, and only the order reaches a body."""
        data = json.loads(_extract_action_json(raw))
        if isinstance(data, list):  # extract_action_json promotes a lone object to [obj]
            data = next((d for d in data if isinstance(d, Mapping)), None)
        if not isinstance(data, Mapping):
            raise TypeError(f"head response must be a JSON object, got {type(data).__name__}")
        orders: dict[str, str] = {}
        states: dict[str, str] = {}
        schema: set[str] = set()
        for key, value in data.items():
            name = self._resolve_name(key)
            if not name:
                continue
            plain = isinstance(value, str)
            text, state = (value, "") if plain else self._instruction_parts(value)
            if not text:
                continue
            orders[name] = text
            if not plain:
                # The narration moved to state, so the relaxation rule is not applied to this drone's order.
                schema.add(name)
            if state:
                states[name] = state
        if not orders:
            raise ValueError(f"head response named no known drone: {sorted(map(str, data))}")
        self._last_states = states
        self._schema_names = schema
        return orders

    def _resolve_name(self, key: Any) -> str | None:
        text = str(key).strip()
        for name in self.drone_names:
            if text.lower() == name.lower():
                return name
        digits = re.findall(r"\d+", text)  # "drone 2", "2", "DRONE_2"
        if digits:
            index = int(digits[-1]) - 1
            if 0 <= index < len(self.drone_names):
                return self.drone_names[index]
        return None

    @staticmethod
    def _instruction_parts(value: Any) -> tuple[str, str]:
        """``(order, state)`` out of one drone's object value, under the aliases a model
        reaches for on its own; ``("", "")`` out of anything else."""
        if not isinstance(value, Mapping):
            return "", ""

        def first(*keys: str) -> str:
            found = (value.get(key) for key in keys)
            return next((t.strip() for t in found if isinstance(t, str) and t.strip()), "")

        return (first("order", "instruction", "task", "text"),
                first("state", "status", "ledger", "note"))

    def _ask(self, trigger: str, user_prompt: str | None = None) -> dict[str, str]:
        """One head call (one reparse retry), applied to the bodies. Never raises.
        ``user_prompt=None`` is a poll turn, built AFTER the grab so the report can name the
        drones whose image is blank."""
        frames = self.grab_frames()
        if all(f is None for f in frames):
            say("[HEAD] frame failed for all drones — skipping this call")
            self._log({"trigger": trigger, "skipped": "no_frames"})
            return {}
        blank = np.zeros((self.camera.height, self.camera.width, 3), np.uint8)
        filled = [f if f is not None else blank for f in frames]
        map_img = self.render_map()
        if map_img is not None:
            filled = filled + [map_img]          # N+1th image; the label list grows with it
        if user_prompt is None:
            user_prompt = _POLL_USER.format(
                transcript=self.transcript(), trigger=trigger, elapsed=time.time() - self._t0,
                report=self.interval_report(), n=len(self.drone_names), images_line=self.images_line())
        # A black placeholder must be told apart from an empty scene, or the head re-plans on
        # false evidence ("Drone2 sees nothing → send it elsewhere").
        missing = [n for n, f in zip(self.drone_names, frames) if f is None]
        if missing:
            user_prompt += ("\nNOTE (camera unavailable this turn): the image for "
                            + ", ".join(missing) + " is a black placeholder, not an empty scene.")

        orders: dict[str, str] = {}
        usage: Any = None
        started = time.time()
        for attempt in (1, 2):
            # Re-parse retries also consume the call budget; without this re-check one cap check lets two calls
            # go out and self.calls ends at max_calls+1.
            if self.calls >= self.max_calls:
                break
            raw, usage, err = self._generate(user_prompt, filled)
            self.calls += 1
            self._last_call_t = time.time()
            if not raw:   # None and an empty body are both "no response"; an empty string used to masquerade as parse_failed
                # Without the reason on this line, an arm flies entirely without command and the log keeps only
                # this line (e.g. a model that rejects thinking_budget=0 on every call).
                say(f"[HEAD] no reply (attempt {attempt})" + (f" — {err}" if err else ""))
                continue
            try:
                orders = self.parse_instructions(str(raw))
                break
            except Exception as exc:  # noqa: BLE001 — malformed plan, try once more then broadcast
                say(f"[HEAD] parse_failed attempt={attempt} err={type(exc).__name__}: {exc}")
        latency = time.time() - started
        # Only now, with the answer in hand: a trace must never sit in front of a head call.
        trace = self._trace(user_prompt, list(frames) + ([map_img] if map_img is not None else []))

        if not orders:
            if trigger != _TRIGGER_INITIAL:
                # Broadcasting the mission text mid-episode would throw away orders built up over
                # several turns for one malformed reply. The turn is not recorded either — the
                # transcript only shows orders actually issued.
                say("[HEAD] order parse failed — keeping the previous order")
                self._log({"trigger": trigger, "skipped": "parse_failed", "call": self.calls,
                           "latency_s": round(latency, 2), **trace})
                return {}
            # t0 is different: nothing has orders yet, so the raw mission text beats silence.
            say("[HEAD] could not read orders — sending the original mission text to every drone as is")
            orders = {name: self.mission_text for name in self.drone_names}
            self._last_states, self._schema_names = {}, set()

        states = dict(self._last_states)
        self.apply(orders)
        elapsed = time.time() - self._t0
        turn = {"trigger": trigger, "elapsed": elapsed, "instructions": dict(orders)}
        if states:
            turn["states"] = states
        self.turns.append(turn)
        say(f"[HEAD] call={self.calls} trigger={trigger} latency={latency:.1f}s changed={len(orders)}")
        for name, text in orders.items():
            say(f"[HEAD] {name} <- {_fold(text)}")
        self._log({
            "trigger": trigger, "elapsed_s": round(elapsed, 2), "call": self.calls,
            "latency_s": round(latency, 2), "instructions": orders,
            # The head's ledger note. Never sent to bodies; the only record for reading its judgement afterwards.
            **({"states": states} if states else {}), "usage": usage, **trace,
        })
        return orders

    def apply(self, orders: Mapping[str, str]) -> None:
        """Hand each order to its body, consuming ``HOLD``. Must follow
        :meth:`parse_instructions`, which records how strictly to read the keyword per drone."""
        by_name = {body.name: body for body in self.bodies}
        for name, text in orders.items():
            body = by_name.get(name)
            if body is None:
                continue
            # HOLD is a keyword on every arm: gated on wait_after_finished it reached the
            # ``Command:`` slot of a body that cannot park (the flag's default), and a harness
            # word there is not an instruction. Where the wait is off ``_hold`` refreshes the
            # budget instead of parking, which is what "carry on" means to a flying drone.
            if is_hold(text, strict=name in self._schema_names):
                self._hold(body)
                continue
            if str(text).strip().upper().startswith(HOLD_KEYWORD):
                # A HOLD that only grazed the strict check becomes the order as-is; must not pass silently.
                say(f"[HEAD] hold_unmatched drone={name} — starts with HOLD but is not exactly HOLD, "
                    f"so it is sent as an instruction: {_fold(text)}")
            body.set_orders(text)

    def _hold(self, body: Any) -> None:
        """Park a declared drone; to any other one HOLD means "carry on". A flying drone keeps
        its order's words but gets a fresh step budget — without that refresh it hung on
        ``THINK(awaiting fresh orders)`` for the rest of the episode."""
        if bool(body.status().get("waiting")):
            say(f"[HEAD] hold drone={body.name} — keeps waiting where it declared")
            return
        body.refresh_order_ttl()
        say(f"[HEAD] hold_continue drone={body.name} — same command, only the budget is renewed")

    def assign_initial(self) -> dict[str, str]:
        """t0: the only call that must reach every drone — nothing flies until it lands."""
        orders = self._ask(_TRIGGER_INITIAL, _INITIAL_USER.format(n=len(self.drone_names)))
        missing = [b for b in self.bodies if not b.started.is_set()]
        if missing:
            say(f"[HEAD] drones without an initial order {', '.join(b.name for b in missing)} — starting them on the original mission text")
            for body in missing:
                body.set_orders(self.mission_text)
        return orders

    def poll_loop(self, stop_event: Any, declaration_event: Any) -> None:
        """Re-plan until ``stop_event``: continuously (``poll_interval_s == 0``), on a timer, or
        never (``< 0``, the one-shot ablation). The head's turn rate IS the control rate."""
        if self.poll_interval_s < 0:
            say("[HEAD] poll_interval_s<0 — issuing initial orders only, no watch loop (one-shot)")
            stop_event.wait()
            return
        continuous = self.poll_interval_s == 0
        if continuous:
            say(f"[HEAD] continuous loop — the next call starts right after each call ends (frame refresh floor {CONTINUOUS_FLOOR_S:g}s)")
        while not stop_event.is_set():
            woke = declaration_event.wait(timeout=CONTINUOUS_FLOOR_S if continuous else self.poll_interval_s)
            if stop_event.is_set():
                break
            trigger = _TRIGGER_DECLARATION if woke else _TRIGGER_POLL
            if self.calls >= self.max_calls:
                # Only the head clears the TTL; left alone, within order_ttl_steps steps after the cap is reached
                # every drone freezes in THINK(awaiting fresh orders) (that is not "holding").
                for body in self.bodies:
                    body.order_ttl_steps = 0
                say(f"[HEAD] call cap {self.max_calls} reached — releasing the order TTL and keeping the current orders")
                self._log({"trigger": trigger, "skipped": "max_calls"})
                stop_event.wait()
                return
            gap = time.time() - self._last_call_t
            # A declaration is never throttled: the min gap exists to stop the poll timer from
            # burning the call budget, and a body that just declared is the one moment the head
            # must judge at once. The continuous loop has no such guard; the floor above is the only spacing.
            if not continuous and trigger != _TRIGGER_DECLARATION and gap < self.min_call_gap_s:
                self._log({"trigger": trigger, "skipped": "min_interval", "gap_s": round(gap, 2)})
                stop_event.wait(self.min_call_gap_s - gap)
                continue
            # Only now, once this wake is certain to become a call — clearing above the guards
            # dropped declarations that landed inside the min-gap window.
            declaration_event.clear()  # coalesce a burst of declarations into one call
            self._ask(trigger)  # the poll prompt is built inside, after the frames are in

    def _trace(self, user_prompt: str, frames: Sequence[np.ndarray | None]) -> dict:
        """What this call sent the model, as fields for its ``head_log`` row — nothing else
        records it. ``{}`` with no run dir or the trace off; a failure to write costs no flight."""
        if not self.run_dir or str(os.environ.get(HEAD_TRACE_ENV, "1")).strip() == "0":
            return {}
        try:
            return {"user_prompt": user_prompt, "frames": self._save_frames(frames)}
        except Exception as exc:  # noqa: BLE001 — tracing is cheaper than flying
            say(f"[HEAD] head_trace write failed (ignored): {type(exc).__name__}: {exc}")
            return {}

    def _save_frames(self, frames: Sequence[np.ndarray | None]) -> list[str | None]:
        """``head_frames/call_<NNN>_<DroneName>.jpg`` per drone, in roster order: the file name,
        ``"missing"`` for one blind this turn, or ``None`` if the write failed."""
        out_dir = os.path.join(self.run_dir, HEAD_FRAMES_DIR)
        os.makedirs(out_dir, exist_ok=True)
        saved: list[str | None] = []
        names = list(self.drone_names) + ([head_map.MAP_FRAME_NAME] if len(frames) > len(self.drone_names) else [])
        for name, frame in zip(names, frames):
            if frame is None:
                saved.append(TRACE_MISSING)
                continue
            filename = f"call_{self.calls:03d}_{name}.jpg"
            try:
                ok = cv2.imwrite(os.path.join(out_dir, filename), frame,
                                 [cv2.IMWRITE_JPEG_QUALITY, HEAD_TRACE_JPEG_QUALITY])
            except Exception as exc:  # noqa: BLE001 — one failed image must not block the rest
                say(f"[HEAD] head_frame save failed (ignored) drone={name} err={type(exc).__name__}")
                ok = False
            saved.append(filename if ok else None)
        return saved

    def _log(self, row: dict) -> None:
        if not self.run_dir:
            return
        row = {"t": round(time.time() - self._t0, 2), **row}
        try:
            with open(os.path.join(self.run_dir, HEAD_LOG_NAME), "a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        except Exception as exc:  # noqa: BLE001 — logging never costs a flight
            say(f"[HEAD] head_log write failed (ignored): {type(exc).__name__}: {exc}")
