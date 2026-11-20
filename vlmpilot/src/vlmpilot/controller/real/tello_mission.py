# DroneCATS
# Copyright (c) 2026-present NAVER Cloud Corp.
# Apache-2.0

"""Tello real-drone mission loop: reproduces mission.py's (AirSim) closed loop on a real drone.

Clean-room implementation (sources: DJI public SDK + djitellopy docs). The AirSim path is not touched
by a single line: this module shares only mission.py's **contract** (instruction input → takeoff →
[no queue wait] capture → inference → command → arrival/search handling → wrap-up), and the
implementation is built fresh on top of TelloController.

Same artifacts as the AirSim version (for offline evaluation): under run_dir, decision_*.json/jpg
(decision records produced by the policy), steps.jsonl (telemetry trajectory), arrival frames.
The stdout contract ([FRAME] / [ACTION] / Action Response: / [TIMING]) is also identical, so
runners and analysis tools later use the same parser regardless of backend.

Control mode (VLMPILOT_CONTROL_STDIN=1), for an external flight client that drives the pilot over stdin:
  · On an arrival declaration (FINISHED) the drone does not land but **hovers in place**, waiting while the judge looks at
    the drone and frame and measures inside-circle / visible. Tello lands by itself after 15 s without commands, so
    keepalives are sent while paused.
  · Takes one-line commands on stdin: ``continue`` (declaration below the bar; keep flying the same run),
    ``land`` (end the run, land), ``order <instruction>`` (replace the instruction; the commanding side re-orders mid-run).
    A ``land`` received in flight lands at the next step boundary. An ``order`` in flight uses the new instruction
    from the next step (arrival/search counters reset).
  · **An ``order`` while paused does not resume**: it only swaps the instruction and keeps waiting. Only a human
    verdict (``continue``/``land``) releases the pause. An ``order`` sent in flight that happens to coincide with an arrival
    declaration is not consumed as a verdict. (head/body mode inverts this one line; see below.)
  · Markers: ``[PAUSE] arrival — waiting for continue|land`` / ``[RESUME] continue`` / ``[LAND] requested`` /
    ``[ORDER] <instruction>`` / ``[ORDER] held — still waiting for continue|land``.

An arrival declaration (FINISHED) is **not a landing signal**. Judging measures inside-circle and visible separately, and below
the bar the same run must keep flying, so the drone must stay where it declared. In control mode it hovers waiting for the verdict;
without control it hovers for ``finished_hover_s`` then lands (``land_on_finished: true`` restores the old immediate landing).

head/body mode (VLMPILOT_HEADBODY=1, control mode only): **the commander picks the next move** instead of a human.
One switch turns on two things together (they always go together: there is no human in the loop):
  · **Wait for the first order**: only take off (+climb) and hover until the first ``order`` arrives. Before that, the VLM
    is neither called nor are decisions recorded (so a few seconds of the body deciding alone don't mix with 'what the commander decided').
    Only the first order releases it: ``continue`` and an empty ``order`` are ignored, and ``land`` lands as-is.
    While waiting, ``wait_<ts>.jpg`` is saved every keepalive period (not a decision but the 'current view'; the commander's first
    order is decided from this image). Markers: ``[WAIT] awaiting first order`` / ``[WAIT] released``.
  · **Arrival doesn't wait for a human verdict**: the hover after FINISHED is the same, but what it waits for is not a verdict but
    the commander's next ``order`` (``continue`` and ``land`` are accepted as before). When an ``order`` arrives it flies on
    immediately with the new instruction. Same place as the sim's ``wait_after_finished: true``. Inside-circle / visible is measured **afterwards**
    on the recorded declaration frames.
    Marker: ``[PAUSE] arrival — waiting for order|land`` (distinct from normal mode's ``…continue|land``).
"""

from __future__ import annotations

import json
import math
import os
import queue
import sys
import threading
import time
from typing import Any

from vlmpilot import artifacts
from vlmpilot.policy import vlm_backends
from vlmpilot.policy.camera import PinholeCamera
from vlmpilot.policy.policy import ACTION_RESPONSE_SCHEMA, ACTION_SCHEMA_NAME as SCHEMA_NAME, VlmPolicy
from vlmpilot.controller.real.tello_controller import TelloController

# ── stdout/behavior contract values identical to mission.py ─────────────────────────────────
# mission.py imports airsim at module load (absent on the real-drone laptop), so they can't be imported
# directly: the values are duplicated, and test_tello checks them against mission.
DEFAULT_ARRIVAL_FRAMES = 1
ENV_CONTINUE_AFTER_ARRIVAL = "VLMPILOT_CONTINUE_AFTER_ARRIVAL"
ENV_CONTROL_STDIN = "VLMPILOT_CONTROL_STDIN"
ENV_HEADBODY = "VLMPILOT_HEADBODY"           # head/body mode (meaningful only in control mode)
ENV_WAIT_TIMEOUT_S = "VLMPILOT_WAIT_TIMEOUT_S"
ENV_HOLD_TIMEOUT_S = "VLMPILOT_HOLD_TIMEOUT_S"
# A pause nobody is watching has a deadline: if the commander dies or goes silent, the drone stays up until the battery
# runs out (that span is excluded from cap too, so auto-landing never kicks in). Judge pauses watched by a human are exempt.
DEFAULT_WAIT_TIMEOUT_S = 120.0               # limit for waiting for the first order; 4 drones' connection/video readiness can skew and delay the commander's first call (60s landed the drone that took off first)
DEFAULT_HOLD_TIMEOUT_S = 300.0               # limit for waiting for the next order after arrival. A drone the commander parks with HOLD (waiting in front of the target, waiting for the sum to settle)
                                             # is a legitimate state, so allow the run length (cap 300s); 90s landed what the commander meant as parking.
WAIT_MARKER = "[WAIT] awaiting first order"
WAIT_RELEASED_MARKER = "[WAIT] released"
WAIT_TIMEOUT_MARKER = "[WAIT] timeout"       # followed by " — no order within <n>s, landing"
HOLD_TIMEOUT_MARKER = "[PAUSE] timeout"      # followed by " — landing"
PAUSE_MARKER = "[PAUSE] arrival — waiting for continue|land"
HEADBODY_PAUSE_MARKER = "[PAUSE] arrival — waiting for order|land"
ENV_LIVE_FRAME_S = "VLMPILOT_LIVE_FRAME_S"   # (for tests) override the live frame period
LIVE_FRAME_S = 2.0                           # in head/body mode, one 'current view' every 2s throughout the flight.
                                             # The commander runs a continuous loop (2s floor), so match that beat;
                                             # one per step (7~8s) would have it called on the same image several times.
RESUME_MARKER = "[RESUME] continue"
LAND_MARKER = "[LAND] requested"
HOVER_MARKER = "[HOVER] finished — holding position"
PAUSE_KEEPALIVE_S = 4.0      # Tello auto-lands after 15 s without commands; keep poking it before that
# Time to hold in place after an arrival declaration when flying without a judge. 0 = forever until Ctrl+C.
DEFAULT_FINISHED_HOVER_S = 120.0
CMD_CONTINUE, CMD_LAND = "continue", "land"
CMD_ORDER = "order"          # order <instruction>: re-order mid-run
CMD_TIMEOUT = "timeout"      # not a control command: _hold's return value meaning the pause exceeded its deadline
ORDER_MARKER = "[ORDER] "
ORDER_HELD_MARKER = "[ORDER] held — still waiting for continue|land"
FRAME_MARKER = "[FRAME] captured"
INSTRUCTION_PROMPT = "[INPUT] Enter this flight's goal on one line (press Enter to confirm):"
SEARCH_GIVE_UP_STEPS = 24
STEP_REPORT_PREFIX = "Action Response: "


class StdinControl:
    """Read stdin in the background and queue control commands (continue|land|order <instruction>).

    Must start **after** read_instruction has consumed the single instruction line. Other lines are ignored.
    """

    def __init__(self, stream=None):
        self.q: "queue.Queue[str]" = queue.Queue()
        self._stream = stream or sys.stdin
        self._th = threading.Thread(target=self._pump, daemon=True)
        self._th.start()

    def _pump(self) -> None:
        for line in self._stream:
            raw = line.rstrip("\r\n").strip()
            cmd = raw.lower()
            if cmd in (CMD_CONTINUE, CMD_LAND):
                self.q.put(cmd)
            elif cmd.startswith(CMD_ORDER + " "):
                # Lowercase only the keyword; pass the instruction through exactly as written
                self.q.put(f"{CMD_ORDER} {raw[len(CMD_ORDER) + 1:]}")
        # Stream ended = the flight client died or the pipe closed. Nobody can release the pause any more,
        # so land; better than leaving a drone that can't receive commands in the air.
        self.q.put(CMD_LAND)

    def get(self, timeout: float | None = None) -> str | None:
        try:
            return self.q.get(timeout=timeout)
        except queue.Empty:
            return None

    def poll(self) -> str | None:
        return self.get(timeout=0)


def _timeout_env(name: str, default: float) -> float:
    """Pause-deadline env. Non-numeric or <= 0 → default (never opens a path to waiting forever).

    Also blocks ``inf``/``nan``: ``float("inf")`` is positive so it passes the gate but never exceeds the deadline,
    and with ``nan`` every comparison is False so it never exceeds either; both mean 'no deadline'.
    """
    try:
        value = float(os.environ.get(name) or default)
    except ValueError:
        return default
    return value if math.isfinite(value) and value > 0 else default


def parse_order(cmd: str | None) -> str | None:
    """For ``order <instruction>``, that instruction ("" if empty); for other commands, None."""
    head, _, text = (cmd or "").partition(" ")
    return text.strip() if head.lower() == CMD_ORDER else None


def read_instruction(prompt: str) -> str:
    """Same contract as mission.read_instruction: one line of input, "" on EOF."""
    _say(prompt)
    try:
        line = input()
    except EOFError:
        return ""
    return line.strip()

# Fixed resolution of the Tello stream (DJI spec: 960x720). Can be overridden via config.
DEFAULT_FRAME_WIDTH = 960
DEFAULT_FRAME_HEIGHT = 720
# Tello camera horizontal FOV, from a checkerboard calibration (numbers in the configs/real/tello_gemini38_or.yaml comments). Can be overridden via config.
DEFAULT_TELLO_HFOV_DEG = 56.4   # horizontal FOV (fx≈896px @960x720). The spec's 82.6° is for the photo sensor and doesn't fit the 720p stream

DEFAULT_CONFIG_PATH = "configs/real/tello_gemini38_or.yaml"   # relative to cwd (vlmpilot/), like the airsim default

CAPTURE_RETRY_DELAY_S = 0.5


def _say(msg: str = "") -> None:
    print(msg, flush=True)


ENV_CONFIG_OVERRIDE = "VLMPILOT_CONFIG_OVERRIDE"


def _apply_override(config: dict) -> dict:
    """Override config keys with `VLMPILOT_CONFIG_OVERRIDE` (JSON).

    An external flight client can pass an environment profile's flight parameters (step length, step time, speed, takeoff altitude, approach bands)
    through this env var, so there is no need for a separate config file per environment.
    A malformed value is ignored and the config is used as-is (doesn't block the flight).
    """
    raw = os.environ.get(ENV_CONFIG_OVERRIDE, "").strip()
    if not raw:
        return config
    try:
        over = json.loads(raw)
        if not isinstance(over, dict):
            raise ValueError(f"not a mapping: {type(over).__name__}")
    except Exception as exc:  # noqa: BLE001 — a bad override must not block the flight
        _say(f"[CONFIG] could not read {ENV_CONFIG_OVERRIDE}, ignoring it ({type(exc).__name__}: {exc})")
        return config
    changed = {k: v for k, v in over.items() if config.get(k) != v}
    config.update(over)
    scene = os.environ.get("VLMPILOT_SCENE", "")
    _say(f"[CONFIG] environment override{f' ({scene})' if scene else ''} — " + (
        ", ".join(f"{k}={v}" for k, v in changed.items()) if changed else "same as config"))
    return config


def _load_config(path: str | None) -> dict:
    import yaml

    p = path or DEFAULT_CONFIG_PATH
    with open(p, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"config is not a mapping: {p}")
    return _apply_override(data)


def _opt_float(v: Any) -> float | None:
    return None if v is None else float(v)


def _camera_from_config(config: dict) -> PinholeCamera:
    """Build a pinhole camera from frame size, horizontal FOV, and (optional) calibrated principal point camera_cx/camera_cy."""
    width = int(config.get("tello_frame_width", DEFAULT_FRAME_WIDTH))
    height = int(config.get("tello_frame_height", DEFAULT_FRAME_HEIGHT))
    cx, cy = _opt_float(config.get("camera_cx")), _opt_float(config.get("camera_cy"))
    if str(config.get("tello_video_resolution", "")).lower() == "low":
        # 480p (640x480) stream: the calibration (principal point, fx≈896px) is for 960x720, so scale it by the same ratio. The FOV is unchanged,
        # so PinholeCamera recomputes the focal length from the width. Used to reduce 2.4GHz load in multi-drone runs.
        k = 640.0 / width
        width, height = 640, 480
        cx = None if cx is None else cx * k
        cy = None if cy is None else cy * k
    return PinholeCamera(width, height,
                         float(config.get("fov_horizontal", DEFAULT_TELLO_HFOV_DEG)), cx=cx, cy=cy)


def _assemble(config: dict, run_dir: str, tello: Any = None
              ) -> tuple[Any, TelloController]:
    """PinholeCamera → backend → VlmPolicy → TelloController (counterpart of mission._assemble)."""
    camera = _camera_from_config(config)
    backend = vlm_backends.build_backend(
        config, response_schema=ACTION_RESPONSE_SCHEMA, schema_name=SCHEMA_NAME
    )
    policy = VlmPolicy.from_config(config, camera, backend, mode="tello")
    controller = TelloController(
        config=config,
        run_dir=run_dir,
        policy=policy,
        camera=camera,
        on_decision=lambda record: artifacts.save_decision(run_dir, record),
        tello=tello,
    )
    return backend, controller


def fly(args: Any, tello: Any = None, control: Any = None) -> int:
    """Single Tello episode. Returns the process exit code (same as the cli contract).

    ``tello`` injection is for tests; None attaches to a real djitellopy drone.
    ``control`` is the control-command source (``get(timeout)``/``poll()``); if None and VLMPILOT_CONTROL_STDIN=1,
    a StdinControl is created after reading the instruction. If neither, the old behavior (arrival → landing).
    """
    config = _load_config(getattr(args, "config", None))
    output_root = str(config.get("output_root", "outputs"))
    run_dir = artifacts.create_run_dir(output_root)
    _say(f"[RUN] output folder: {run_dir}")

    backend, controller = _assemble(config, run_dir, tello=tello)

    instruction = read_instruction(INSTRUCTION_PROMPT)
    if not instruction:
        _say("[INPUT] instruction is empty or input was closed — exiting without takeoff (exit 2)")
        return 2

    loop_delay = float(config.get("command_loop_delay", 0) or 0)
    arrival_frames = int(config.get("arrival_frames", DEFAULT_ARRIVAL_FRAMES)
                         or DEFAULT_ARRIVAL_FRAMES)
    continue_after_arrival = os.environ.get(ENV_CONTINUE_AFTER_ARRIVAL) == "1"
    land_on_finished = str(config.get("land_on_finished", False)).lower() in ("1", "true", "yes")
    hover_s = float(config.get("finished_hover_s", DEFAULT_FINISHED_HOVER_S))
    if control is None and os.environ.get(ENV_CONTROL_STDIN) == "1":
        control = StdinControl()
    _say(f"[CONFIG] tello loop_delay={loop_delay:g}s arrival_frames={arrival_frames} "
         f"continue_after_arrival={continue_after_arrival} land_on_finished={land_on_finished} "
         f"finished_hover_s={hover_s:g} control={'stdin' if control is not None else 'off'}")
    # head/body mode: wait for the first order + don't wait for a human verdict on arrival (the commander makes the next move).
    # Outside control mode there is no channel to feed orders, so it is ignored.
    headbody = control is not None and os.environ.get(ENV_HEADBODY) == "1"
    wait_timeout_s = _timeout_env(ENV_WAIT_TIMEOUT_S, DEFAULT_WAIT_TIMEOUT_S)
    hold_timeout_s = _timeout_env(ENV_HOLD_TIMEOUT_S, DEFAULT_HOLD_TIMEOUT_S)
    if headbody:
        _say(f"[CONFIG] headbody=on — waiting for the first order (up to {wait_timeout_s:g}s) · after arrival it also waits for the commander's "
             f"next order (up to {hold_timeout_s:g}s)")

    def _hold(marker: str, *, order_releases: bool, continue_releases: bool,
              timeout_s: float | None = None) -> str:
        """One hover-pause spot: wait while sending keepalives and return **what released it**.

        ``CMD_ORDER`` (released by a new order) · ``CMD_CONTINUE`` (continue with the same instruction) · ``CMD_LAND`` (land).
        Instruction replacement and its echo (``[ORDER] …``) are finished here. Three places share this loop:
        the judge wait (order is held), head/body's arrival wait (order releases), and the first-order wait.
        **Unknown commands land**: rather than ignore
        words it doesn't understand and stay up, it lands (only ``continue`` during the first-order wait is ignored).
        With ``timeout_s``, emits ``CMD_TIMEOUT`` after that time: a deadline for pauses nobody is watching.
        """
        nonlocal instruction
        deadline = None if timeout_s is None else time.monotonic() + timeout_s
        _say(marker)
        while True:
            cmd = control.get(timeout=PAUSE_KEEPALIVE_S)
            text = parse_order(cmd)
            if text is not None:
                if text:
                    instruction = text
                    _say(f"{ORDER_MARKER}{text}")
                    if order_releases:
                        return CMD_ORDER
                else:
                    _say(f"{ORDER_MARKER}ignored — empty")
                if not order_releases:
                    _say(ORDER_HELD_MARKER)
            elif cmd == CMD_LAND:
                return CMD_LAND
            elif cmd == CMD_CONTINUE:
                if continue_releases:
                    return CMD_CONTINUE
                # continue means nothing during the first-order wait; only the first order can release it
            elif cmd:
                # Unknown commands land on the safe side (the old judge-wait contract): rather than keep hovering
                # without knowing what was intended, it is better for a human to take off again.
                _say(f"[LAND] unknown command {cmd!r} — landing")
                return CMD_LAND
            if deadline is not None and time.monotonic() >= deadline:
                return CMD_TIMEOUT
            # If orders keep coming, get() never times out; block the 15 s auto-landing here
            controller.keepalive()

    live_stop = threading.Event()

    def _live_frames() -> None:
        """One ``live_<ts>.jpg`` every LIVE_FRAME_S throughout the flight. **The commander's eyes.**

        Not a decision: only a jpg is written to keep fake actions out of the decision stream (a run viewer's step
        list and offline evaluation read only ``decision_*``). artifacts keeps only the latest few.
        Closed-loop steps take 7~8s, so one per step would show the commander, running every 2s, the same image many times;
        hence a thread independent of steps, uploading at the same beat in flight, waiting, or paused alike.
        """
        every = _timeout_env(ENV_LIVE_FRAME_S, LIVE_FRAME_S)
        while not live_stop.is_set():
            try:
                frame = controller.frame()
                if frame is not None:
                    artifacts.save_view_frame(run_dir, frame)
            except Exception as exc:  # noqa: BLE001 — one view image must not block the flight
                _say(f"[VIEW] frame failed (ignored): {type(exc).__name__}: {exc}")
            live_stop.wait(every)

    def _await_first_order() -> str:
        """Hover after takeoff until the first order; only the first order releases it (``continue`` is ignored).

        Neither calls the VLM nor records decisions. The view the commander sees is uploaded by the live frame thread.
        """
        released = _hold(WAIT_MARKER, order_releases=True, continue_releases=False,
                         timeout_s=wait_timeout_s)
        if released == CMD_TIMEOUT:
            # The commander may never come; the first-order wait is excluded from cap too, so auto-landing never kicks in
            _say(f"{WAIT_TIMEOUT_MARKER} — no order within {wait_timeout_s:g}s, landing")
            released = CMD_LAND
        _say(LAND_MARKER if released == CMD_LAND else WAIT_RELEASED_MARKER)
        return released

    def _hover_in_place(limit_s: float) -> None:
        """Hold in place after an arrival declaration, instead of landing. limit_s<=0 means until Ctrl+C."""
        _say(HOVER_MARKER)
        _say("[HOVER] staying in place without landing — press Ctrl+C after the verdict to land"
             + (f" (up to {limit_s:g}s)" if limit_s > 0 else ""))
        t0 = time.monotonic()
        while limit_s <= 0 or time.monotonic() - t0 < limit_s:
            controller.keepalive()
            time.sleep(min(PAUSE_KEEPALIVE_S, limit_s) if limit_s > 0 else PAUSE_KEEPALIVE_S)
        _say(f"[HOVER] {limit_s:g}s elapsed — landing")

    airborne = False
    reached = False
    try:
        if headbody and hasattr(controller, "frame_warmup_s"):
            # Waiting for the first video frame skewed takeoff by 20~70s per drone (Wi-Fi interference among 4 drones).
            # head/body runs hover after takeoff until the first order, so video only needs to come up in between; cut the wait short.
            controller.frame_warmup_s = min(float(controller.frame_warmup_s), 3.0)
        controller.connect()
        _say(f"[START] instruction: {instruction!r} — taking off")
        controller.takeoff()
        airborne = True
        if headbody:
            # The commander looks every 2s, so the view is every 2s too; uploaded by one thread regardless of waiting/flying/paused
            threading.Thread(target=_live_frames, daemon=True, name="tello-live-frames").start()
        landed_in_wait = headbody and _await_first_order() == CMD_LAND
        if not landed_in_wait:
            _say("[LOOP] closed loop started: capture → infer → command")
            _say("[LOOP] a single Ctrl+C runs the wrap-up (landing)")

        arrival_count = 0
        search_streak = 0
        while not landed_in_wait:              # if land came during the wait, don't enter the closed loop (doesn't change inside the loop)
            landing = False
            while control is not None:
                # Consume all control commands queued at the step boundary: land lands, order replaces the instruction,
                # continue in flight means nothing and is ignored (old behavior).
                cmd = control.poll()
                if cmd is None:
                    break
                if cmd == CMD_LAND:
                    _say(LAND_MARKER)
                    landing = True
                    break
                text = parse_order(cmd)
                if text is None:
                    continue
                if not text:
                    _say(f"{ORDER_MARKER}ignored — empty")
                    continue
                instruction = text
                arrival_count = 0
                search_streak = 0
                _say(f"{ORDER_MARKER}{text}")
            if landing:
                break
            _cap_t0 = time.monotonic()
            frame = controller.frame()
            controller.last_frame_ms = round((time.monotonic() - _cap_t0) * 1000.0, 1)
            if frame is None:
                _say(f"[CAPTURE] could not get a frame — retrying in {CAPTURE_RETRY_DELAY_S:g}s")
                time.sleep(CAPTURE_RETRY_DELAY_S)
                continue
            _say(FRAME_MARKER)

            response_text, arrived, searching = controller.run_step(frame, instruction)
            _say(f"{STEP_REPORT_PREFIX}{response_text}")

            if searching and not headbody:      # in head/body runs giving up is the commander's call; the body doesn't land by itself
                search_streak += 1
                if search_streak >= SEARCH_GIVE_UP_STEPS:
                    _say(f"[SEARCH] {SEARCH_GIVE_UP_STEPS} consecutive misses — target not found, ending.")
                    break
            else:
                search_streak = 0

            if arrived is True:
                arrival_count += 1
                _say(f"[ARRIVAL] Target-reached signal (VLM) {arrival_count}/{arrival_frames}")
                if arrival_count >= arrival_frames:
                    _say("[ARRIVAL] FINISHED (arrival declared)")
                    if control is not None:
                        # Normal mode waits for the **judge's** verdict (continue|land), head/body mode for the **commander's** next
                        # order (order|continue|land). Unless land, keep flying the same run with the instruction (possibly
                        # changed meanwhile): the declaration is already recorded, and head/body judging is done afterwards.
                        released = _hold(HEADBODY_PAUSE_MARKER if headbody else PAUSE_MARKER,
                                         order_releases=headbody, continue_releases=True,
                                         # Judge pauses watched by a human have no deadline (as before);
                                         # the deadline applies only to head/body pauses with nobody to wait for
                                         timeout_s=hold_timeout_s if headbody else None)
                        if released == CMD_TIMEOUT:
                            _say(f"{HOLD_TIMEOUT_MARKER} — landing")
                            released = CMD_LAND
                        if released != CMD_LAND:
                            _say(RESUME_MARKER)
                            arrival_count = 0
                            search_streak = 0
                        else:
                            _say(LAND_MARKER)
                            reached = True
                            try:
                                artifacts.save_arrival_frame(run_dir, frame)
                            except Exception:  # noqa: BLE001 — souvenir frame is best effort
                                pass
                            break
                    elif continue_after_arrival:
                        arrival_count = 0
                    else:
                        reached = True
                        try:
                            artifacts.save_arrival_frame(run_dir, frame)
                        except Exception:  # noqa: BLE001 — souvenir frame is best effort
                            pass
                        if land_on_finished:
                            _say("[ARRIVAL] Target reached — landing and ending sequence.")
                        else:
                            # Declaration ≠ arrival. Judging (inside-circle, visible) remains, so hold the declared spot.
                            _say("[ARRIVAL] Target reached — holding position (no landing).")
                            _hover_in_place(hover_s)
                        break
            elif arrived is False:
                arrival_count = 0
            # arrived is None → no decision; keep counters (same as mission.py R-316)

            if loop_delay > 0:
                time.sleep(loop_delay)
    except KeyboardInterrupt:
        _say("[LOOP] interrupt received — moving to landing")
    except Exception as exc:  # noqa: BLE001 — on errors in flight, landing comes first
        if not airborne:
            raise
        _say(f"[LOOP] unexpected error, ending the episode: {type(exc).__name__}: {exc}")
    finally:
        live_stop.set()
        if airborne:
            try:
                controller.land()
                _say("[TELLO] landed")
            except Exception as exc:  # noqa: BLE001 — report the landing failure and exit
                _say(f"[TELLO] landing failed — manual intervention needed: {exc!r}")
        _say(f"[DONE] reached={reached} run_dir={run_dir}")
    return 0
