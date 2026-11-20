# DroneCATS
# Copyright (c) 2026-present NAVER Cloud Corp.
# Apache-2.0

"""body_policy.py — one autonomous body agent per drone (``agent_mode: head_body``).

A body is a *single-drone pilot*: its own thread, its own VLM endpoint, its own vehicle, and a
:class:`vlmpilot.policy.policy.VlmPolicy` of its own — the prompt, response schema, history and
multi-image rules are the centralised arm's by construction, not by copy. The only thing that
makes it a body is where its ``Command:`` line comes from — the head rewrites it on every re-plan.

It also owns the mode's stdout serialisation: five threads print into one pipe the bench runner
parses line by line, so every line must arrive whole. :func:`say` takes :data:`PRINT_LOCK`, and
:func:`line_atomic_stdout` puts that lock under the ``print`` calls of the modules a body
drives, which know nothing of it."""

from __future__ import annotations

import contextlib
import dataclasses
import sys
import threading
import time
from collections import deque
from typing import Any, Callable, Mapping

import numpy as np

from vlmpilot.controller.airsim import mission
from vlmpilot.policy.action import ACTION_FINISHED, ACTION_ROTATE, ACTION_THINK, Action
from vlmpilot.policy.policy import (
    ACTION_RESPONSE_SCHEMA,
    PLAN_RESPONSE_SCHEMA,
    DecisionRecord,
    VlmPolicy,
    _extract_action_json,
    action_from_dict,
)
from vlmpilot.policy.team_policy import render_tile

__all__ = [
    "PRINT_LOCK",
    "BodyAgent",
    "line_atomic_stdout",
    "say",
]

# Same length as the single/team streak, but a body only warns — the head ends the episode.
ROTATE_WARN_STEPS = mission.SEARCH_GIVE_UP_STEPS
STATUS_ACTIONS = 4  # recent actions handed to the head in a status snapshot
STEP_RETRY_DELAY_S = 0.5
# Capture failures are logged only as streaks: one line per step would flood the log with 1 s retries.
CAPTURE_LOG_STREAKS = (1, 5, 10)
CAPTURE_LOG_EVERY = 10

# Step budget per order: if the commander goes silent, a stale command drifts forever. Once spent, skip the VLM
# and wait in place with THINK (keeps runner pairing). <= 0 means unlimited.
DEFAULT_ORDER_TTL_STEPS = 4
# That waiting THINK is a harness bookkeeping action, not a model decision (_think_flag).
HOLD_REASON = "awaiting fresh orders"
# Interval for only checking release while waiting after declaring (``body.wait_after_finished``, default off).
WAIT_POLL_S = 0.5

# --------------------------------------------------------------- stdout (log contract)

PRINT_LOCK = threading.RLock()


def say(line: str = "") -> None:
    """Print one whole line. Never call this while holding an RPC or a VLM call."""
    with PRINT_LOCK:
        print(line, flush=True)


class _LineAtomicStdout:
    """``sys.stdout`` proxy so one thread's half-written line can never split another's:
    ``print(x)`` is two writes, and the runner would eventually read
    ``[ACTION] … drone=Drone1[MOVEMENT] …`` and measure the wrong vehicle."""

    def __init__(self, stream: Any) -> None:
        self._stream = stream
        self._partial: dict[int, str] = {}

    def write(self, text: str) -> int:
        ident = threading.get_ident()
        buf = self._partial.pop(ident, "") + text
        head, sep, tail = buf.rpartition("\n")
        if sep:
            with PRINT_LOCK:
                self._stream.write(head + sep)
                self._stream.flush()
        if tail:
            self._partial[ident] = tail
        return len(text)

    def flush(self) -> None:
        with PRINT_LOCK:
            self._stream.flush()

    def drain(self) -> None:
        with PRINT_LOCK:
            for text in list(self._partial.values()):
                self._stream.write(text)
            self._partial.clear()
            self._stream.flush()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)


@contextlib.contextmanager
def line_atomic_stdout():
    """Serialise ``sys.stdout`` for the duration of the block."""
    original = sys.stdout
    proxy = _LineAtomicStdout(original)
    sys.stdout = proxy
    try:
        yield
    finally:
        proxy.drain()
        sys.stdout = original


# ------------------------------------------------------------------------------ body agent


class BodyAgent:
    """One drone's pilot loop. ``run()`` is the thread body; every other method is the head's."""

    def __init__(
        self,
        *,
        name: str,
        index: int,
        controller: Any,
        policy: VlmPolicy,
        make_sensor_client: Callable[[], Any],
        grab_frame: Callable[[Any], np.ndarray | None],
        stop_event: threading.Event,
        declaration_event: threading.Event,
        order_ttl_steps: int = DEFAULT_ORDER_TTL_STEPS,
        arrival_frames: int = 1,
        wait_after_finished: bool = False,
        on_decision: Callable[[Any], None] | None = None,
        on_declared: Callable[["BodyAgent"], None] | None = None,
    ) -> None:
        self.name = str(name)
        self.index = int(index)
        self.controller = controller
        self.policy = policy
        # Built in run(), on this body's own thread: sensing must never share the command client.
        self.make_sensor_client = make_sensor_client
        self.sensor_client: Any = None
        self.grab_frame = grab_frame
        self.stop_event = stop_event
        self.declaration_event = declaration_event
        self.order_ttl_steps = int(order_ttl_steps)
        self.arrival_frames = max(1, int(arrival_frames))
        self.wait_after_finished = bool(wait_after_finished)
        self.on_decision = on_decision
        # Runs inside the declaration's print critical section — it may only set flags.
        self.on_declared = on_declared

        self.started = threading.Event()  # the head's first orders release the body
        self.steps = 0

        self._lock = threading.Lock()  # guards everything the head reads
        self._orders = ""
        self._recent: deque[str] = deque(maxlen=STATUS_ACTIONS)
        self._declared = False
        self._order_steps = 0
        self._order_set_t = time.monotonic()
        self._order_expired_said = False
        self._last_pose: dict | None = None
        self._arrival_streak = 0
        self._rotate_streak = 0
        self._rotate_warned = False
        self._think_pending = False   # one inference call per model-emitted THINK (_think_flag)
        self._capture_streak = 0
        self._capture_streak_t0 = 0.0
        self._waiting = False
        self._waiting_since = 0.0
        self._waiting_said = False
        self._run_t0 = time.monotonic()   # run() re-stamps it; the waiting log dates from here

        say(f"[BODY] ready drone={self.name} {self.api_provider}:{self.model_name} "
            f"point_order={self.point_order} history_len={self.history_len}")

    def set_orders(self, text: str) -> None:
        """Replace this body's standing orders whole and release it if it is still waiting.

        Any order releases a parked body, even one repeating the last order — a re-issued order
        is the head saying "again", which is a decision, while silence is not."""
        with self._lock:
            self._orders = str(text or "")
            self._order_steps = 0        # new order = new budget
            self._order_set_t = time.monotonic()
            self._order_expired_said = False
            self._waiting = False        # new command = release the wait (even if the text is the same)
            self._waiting_said = False
        self.started.set()

    def refresh_order_ttl(self) -> bool:
        """Give the STANDING order a fresh step budget without changing a word of it.

        "Carry on" needs its own verb: while the TTL only ever reset in :meth:`set_orders`, a
        head answering HOLD to a flying drone silently froze it on ``THINK(awaiting fresh
        orders)`` for the rest of the episode. It also un-freezes an already-expired order.

        Returns False for a drone parked by a declaration — only a real order releases that
        one, and the head parks it by staying silent."""
        with self._lock:
            if self._waiting:
                return False
            self._order_steps = 0
            self._order_set_t = time.monotonic()
            self._order_expired_said = False
        return True

    def status(self) -> dict:
        """Snapshot for the head's interval report. Cheap: no RPC, no VLM."""
        with self._lock:
            snapshot: dict[str, Any] = {
                "name": self.name,
                "last_actions": list(self._recent),
                "declared": self._declared,
                # How many times the memoryless body has re-read the same order; a plain fact.
                "order_steps": self._order_steps,
                "order_age_s": max(0.0, time.monotonic() - self._order_set_t),
                # Same test as _step (single source). This drone's THINK is not obedience.
                "order_expired": 0 < self.order_ttl_steps <= self._order_steps,
                # Declaration time = now - waiting_age_s.
                "waiting": self._waiting,
                "waiting_age_s": (max(0.0, time.monotonic() - self._waiting_since)
                                  if self._waiting else 0.0),
            }
            if self._last_pose is not None:
                snapshot["last_pos"] = dict(self._last_pose)
            return snapshot

    # The policy answers these; if the body kept its own copy the two would diverge.
    backend = property(lambda self: self.policy.backend)
    camera = property(lambda self: self.policy.camera)
    api_provider = property(lambda self: self.policy.api_provider)
    model_name = property(lambda self: self.policy.model_name)
    point_order = property(lambda self: self.policy.point_order)
    history_len = property(lambda self: self.policy.history_len)

    def build_prompt(self, orders: str) -> str:
        """The golden single-drone prompt with the head's latest orders in the ``Command:`` slot
        — byte-for-byte the centralised arm's otherwise. Nothing may be appended: a second
        difference between the arms would make the outcome unattributable."""
        return self.policy.build_prompt(orders)

    def _note_capture_empty(self) -> None:
        """Log a None capture on a streak schedule. Records only — the retry wait is untouched."""
        self._capture_streak += 1
        if self._capture_streak == 1:
            self._capture_streak_t0 = time.monotonic()
        if not (self._capture_streak in CAPTURE_LOG_STREAKS
                or self._capture_streak % CAPTURE_LOG_EVERY == 0):
            return
        say(f"[BODY] capture_empty drone={self.name} streak={self._capture_streak}")

    def _note_capture_ok(self) -> None:
        """Close out a capture-failure streak, turning the gap into a duration."""
        if not self._capture_streak:
            return
        waited = time.monotonic() - self._capture_streak_t0
        say(f"[BODY] capture_recovered drone={self.name} "
            f"after={self._capture_streak} ({waited:.1f} s)")
        self._capture_streak = 0

    def _think_flag(self) -> bool:
        """Does this step get reasoning? One call per THINK the MODEL chose — no more.

        Two traps: the intent is consumed by the ATTEMPT, not by the answer (read off
        ``_history[-1]`` it latched forever on the first timed-out call, which writes no
        history); and the TTL-expiry hold's synthetic THINK is bookkeeping for the runner's
        frame/action pairing, not a decision, so it never arms the intent."""
        with self._lock:
            think, self._think_pending = self._think_pending, False
            return think

    def run(self) -> None:
        """Wait for the head's first orders, then fly until told to stop. Never raises."""
        self._run_t0 = time.monotonic()
        if not self._open_sensor_client():
            return
        while not self.stop_event.is_set():
            if self.started.wait(timeout=0.2):
                break
        while not self.stop_event.is_set():
            try:
                self._step()
            except Exception as exc:  # noqa: BLE001 — one bad step never ends the body
                say(f"[BODY] step_failed drone={self.name} err={type(exc).__name__}: {exc}")
                self.stop_event.wait(STEP_RETRY_DELAY_S)
        say(f"[BODY] stopped drone={self.name} steps={self.steps}")

    def _open_sensor_client(self) -> bool:
        """Open this thread's own sensing connection. False = this body cannot fly safely."""
        try:
            self.sensor_client = self.make_sensor_client()
            self.controller.use_sensor_client(self.sensor_client)
            return True
        except Exception as exc:  # noqa: BLE001
            # Falling back to the command client is not an option: that is the wedge this whole
            # split exists to prevent, and one wedged client takes the entire episode down.
            say(f"[BODY] sensor_client_failed drone={self.name} err={type(exc).__name__}: {exc} — this drone will not fly")
            return False

    def _parked(self) -> bool:
        """Declared and waiting for a NEW order: skip the step, sense nothing, decide nothing.

        Without ``wait_after_finished`` a body that says FINISHED rests one step, re-reads the
        SAME order and flies on — one order then produced hundreds of declarations. Parked, it
        captures no frame and prints one line for the whole wait, so the runner never sees a
        ``[FRAME]`` without its ``[ACTION]``. The wait outranks the order TTL: only the head may
        un-park a drone."""
        with self._lock:
            if not self._waiting:
                return False
            first = not self._waiting_said
            self._waiting_said = True
            since = self._waiting_since
        if first:
            say(f"[BODY] waiting drone={self.name} since={max(0.0, since - self._run_t0):.1f}s")
        self.stop_event.wait(WAIT_POLL_S)
        return True

    def _step(self) -> None:
        if self._parked():
            return
        ctrl = self.controller
        # Same waits as the single-drone loop — a body IS that loop.
        ctrl.wait_until_idle(timeout=mission.IDLE_WAIT_TIMEOUT_S, stop_event=self.stop_event)
        if self.stop_event.is_set():
            return

        frame = self.grab_frame(self.sensor_client)
        if frame is None:
            self._note_capture_empty()
            self.stop_event.wait(mission.CAPTURE_RETRY_DELAY_S)
            return
        self._note_capture_ok()
        # The runner reads this vehicle's pose at this line — nothing may sit between the grab
        # and the marker (mission.fly prints it in the same place for one drone).
        say(f"{mission.FRAME_MARKER} drone={self.name}")

        depth = ctrl.capture_depth()
        pose = ctrl.pose_snapshot()
        heading = ctrl.current_yaw_rad()
        with self._lock:
            self._last_pose = pose
            orders = self._orders
            expired = 0 < self.order_ttl_steps <= self._order_steps
            first_expiry = expired and not self._order_expired_said
            if first_expiry:
                self._order_expired_said = True
            spent = self._order_steps

        if expired:
            # Stay in place. No VLM call, but THINK keeps the [FRAME]↔[ACTION] pairing.
            if first_expiry:
                say(f"[BODY] order_expired drone={self.name} steps={spent}")
            with PRINT_LOCK:
                action = action_from_dict({"action": "think", "reason": HOLD_REASON},
                                          self.camera, self.point_order, vehicle=self.name,
                                          point_scale=self.policy.point_scale)
        else:
            raw = self._call_backend(orders, frame)
            # Teardown can begin mid-call, and past here the step would print onto a stdout
            # whose lock is gone and drive RPCs on a client Fleet.shutdown() is using.
            if raw is None or self.stop_event.is_set():
                return
            action = self._parse(raw)
            if action is None or self.stop_event.is_set():
                return

        self.steps += 1
        with self._lock:
            self._order_steps += 1
        self._note(action, frame, model_decided=not expired)
        response, _, _ = ctrl.apply_action(action, pose=pose, depth=depth, heading_ref_rad=heading)
        say(f"[BODY] drone={self.name} {response}")
        self._emit_decision(frame, action, orders)

    def _call_backend(self, orders: str, frame: np.ndarray) -> str | None:
        """One VLM call through this body's policy — same prompts and schema as the centralised
        arm. Returns the raw text, or None after printing the contract line."""
        think = self._think_flag()
        # plan steps take reasoning as a schema field with native reasoning off (same rule as VlmPolicy.decide).
        plan_step = bool(think and self.policy.plan_schema)
        try:
            raw, _ = self.policy.call_backend(
                orders, frame, think=False if plan_step else think,
                schema=PLAN_RESPONSE_SCHEMA if plan_step else ACTION_RESPONSE_SCHEMA)
        except Exception as exc:  # noqa: BLE001 — a provider that raises past the backend
            say(f"[BODY] call_failed drone={self.name} err={type(exc).__name__}")
            return None
        if raw is None:  # the backend swallowed the provider error and reported it as None
            say(f"[BODY] call_failed drone={self.name} err=NoResponse")
            return None
        with PRINT_LOCK:
            for line in str(raw).splitlines() or [""]:
                print(f"[RAW] {line}", flush=True)
        return str(raw)

    def _parse(self, raw: str) -> Action | None:
        """Raw text → Action, printing the ``[ACTION]`` line (and the arrival pair) atomically."""
        try:
            data = VlmPolicy._load_json(_extract_action_json(raw))
            if isinstance(data, list):
                data = data[0] if data else {}
            if not isinstance(data, Mapping):
                raise TypeError(f"action JSON must be an object, got {type(data).__name__}")
            with PRINT_LOCK:
                action = action_from_dict(data, self.camera, self.point_order, vehicle=self.name,
                                          point_scale=self.policy.point_scale)
                if action.kind == ACTION_FINISHED:
                    self._declare()
                else:
                    self._arrival_streak = 0
            return action
        except Exception as exc:  # noqa: BLE001 — this drone holds; the others fly on
            say(f"[BODY] parse_failed drone={self.name} err={type(exc).__name__}: {exc}")
            return None

    def _declare(self) -> None:
        """Arrival line, printed inside the caller's critical section so it stays glued to the
        ``[ACTION] FINISHED`` above it — the runner pairs them positionally."""
        self._arrival_streak += 1
        print(f"[ARRIVAL] Target-reached signal (VLM) {self._arrival_streak}/{self.arrival_frames} "
              f"drone={self.name}", flush=True)
        if self._arrival_streak < self.arrival_frames:
            return
        # The runner only counts a line whose n == N: left to climb the streak would print
        # 2/1, 3/1 … and every declaration after the first would vanish from SR.
        self._arrival_streak = 0
        with self._lock:
            self._declared = True
            if self.wait_after_finished:
                # This step finishes normally; stopping starts from the next step (keeps scoring log and wake).
                self._waiting = True
                self._waiting_since = time.monotonic()
                self._waiting_said = False
        self.declaration_event.set()  # wakes the head; the body waits (or flies on) after this
        if self.on_declared is not None:
            try:
                self.on_declared(self)
            except Exception as exc:  # noqa: BLE001
                print(f"[BODY] declared_hook_failed drone={self.name} err={type(exc).__name__}: {exc}", flush=True)

    def _note(self, action: Action, frame: np.ndarray | None = None, *,
              model_decided: bool = True) -> None:
        with self._lock:
            self._think_pending = model_decided and action.kind == ACTION_THINK
        # History coordinates go back to the prompt-defined 0-1000 (not pixels); the policy records them.
        summary = self.policy.remember(action, frame)
        with self._lock:
            self._recent.append(f"t={time.strftime('%H:%M:%S')} {summary}")
        if action.kind == ACTION_ROTATE:
            self._rotate_streak += 1
            if self._rotate_streak >= ROTATE_WARN_STEPS and not self._rotate_warned:
                self._rotate_warned = True
                say(f"[BODY] rotate_only drone={self.name} {self._rotate_streak} consecutive rotations — "
                    "not finding it from this spot (still flying)")
        elif action.kind != ACTION_THINK:
            self._rotate_streak = 0
            self._rotate_warned = False

    def _emit_decision(self, frame: np.ndarray, action: Action, orders: str) -> None:
        if self.on_decision is None:
            return
        try:
            record = DecisionRecord(
                timestamp=time.strftime("%Y%m%d_%H%M%S"),
                overlay_image_bgr=render_tile(frame, self.index, self.name, action),
                payload={
                    "timestamp": time.strftime("%Y%m%d_%H%M%S"), "mode": "airsim_agents",
                    "api_provider": self.api_provider, "model": self.model_name,
                    "instruction": orders, "drone": self.index + 1, "vehicle": self.name,
                    "actions": [dataclasses.asdict(action)],
                },
            )
            self.on_decision(record)
        except Exception as exc:  # noqa: BLE001 — a debug image must never cost a step
            say(f"[BODY] decision_record_failed drone={self.name} err={type(exc).__name__}: {exc}")
