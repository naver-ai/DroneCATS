# DroneCATS
# Copyright (c) 2026-present NAVER Cloud Corp.
# Apache-2.0

"""pegasus_mission.py — lifecycle of one episode + file-based frame source.

The loop body (arrival counter, search streak, finally ordering) follows `mission.py`
and its rules are not restated here.

Two differences from the AirSim path:

- **The camera is a file.** Process A (`scripts/isaacsim/pegasus_sim.py`) atomically replaces
  the frame and this side reads that path. No RPC queries.
- **Control is MAVLink.** `PegasusLink` attaches to PX4 directly.

Helpers that define the CLI contract are reused from `mission.py`. That pulls in
`airsim`, but duplicating `read_instruction` (EOF handling) and
`confirm_low_resolution` (`isatty` branch) risks the two paths diverging.
"""

from __future__ import annotations

import glob
import json
import os
import shutil
import sys
import time
import traceback
from typing import Any

import cv2
import numpy as np

from vlmpilot import artifacts
from vlmpilot.controller.airsim import mission
from vlmpilot.policy import vlm_backends
from vlmpilot.policy.camera import PinholeCamera
from vlmpilot.controller.isaacsim.pegasus_controller import DEFAULT_MAVSDK_ADDRESS, PegasusFlightController
from vlmpilot.policy.policy import ACTION_RESPONSE_SCHEMA, VlmPolicy

# ---------------------------------------------------------------------------
# Constants. Values shared with mission.py are imported; restating them would let
# them drift apart.
# ---------------------------------------------------------------------------

# Control parameters recorded in run.json. Single and team paths must use **the same
# list**: comparing solo vs multi requires proof that both runs had the same tuning,
# and diverging lists would silently halve that proof. So there is one copy, here.
TUNING_KEYS = (
    "max_step_distance_m",
    "step_duration_s",
    "base_yaw_rate",
    "yaw_deadband_deg",
    "vertical_deadband",
    "adaptive_depth",
    "adaptive_depth_bands",
    "continuous_control",
    "fov_horizontal",
    "point_order",
)

# No default config: every Isaac run passes one (verifier/isaacsim/batch_run.py writes a per-episode copy of an
# vlmpilot/configs/isaacsim/single-drone/*.yaml and calls `vlmpilot pegasus --config <that copy>`).
CONFIG_REQUIRED_MSG = ("pegasus needs --config <yaml>, e.g. vlmpilot/configs/isaacsim/single-drone/eval_px4_gemini37-or.yaml "
                       "(verifier/isaacsim/batch_run.py passes one per episode)")


def require_config_path(args: Any) -> str:
    """The --config path; raises ValueError with a clear message if it is missing or not a file.

    A missing file must not fall through to `mission.load_mission_config`, which returns {} and would
    fly on built-in defaults (a different model and control setup) without saying so.
    """
    path = getattr(args, "config", None)
    if not path:
        raise ValueError(CONFIG_REQUIRED_MSG)
    if not os.path.isfile(path):
        raise ValueError(f"pegasus config not found: {path}")
    return path

DEFAULT_FRAME_PATH = "/tmp/vlmpilot_pegasus/latest.jpg"
DEFAULT_SIM_INFO_PATH = "/tmp/vlmpilot_pegasus/sim_info.json"
DEFAULT_FRAME_MAX_AGE_S = 5.0
FRAME_WAIT_S = 30.0
FOV_MISMATCH_TOLERANCE_DEG = 1.0

SCHEMA_NAME = mission.SCHEMA_NAME
STEP_REPORT_PREFIX = mission.STEP_REPORT_PREFIX
FRAME_MARKER = mission.FRAME_MARKER


def _say(line: str = "") -> None:
    print(line, flush=True)


# ---------------------------------------------------------------------------
# Frame source: a file
# ---------------------------------------------------------------------------


def read_sim_info(path: str = DEFAULT_SIM_INFO_PATH) -> dict:
    """Sim conditions written by A. `{}` if missing or broken; the run proceeds anyway."""
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except Exception:  # noqa: BLE001 — missing is a normal path
        return {}


def check_fov(config: dict, sim_info: dict) -> None:
    """Compare config `fov_horizontal` with the FOV A reported.

    A reports two values: `hfov_deg` (measured) and `fov_deg` (requested). **Measured
    wins**: a 90° request produced 91.7°, and back-projection sees the measured value.
    Without a measured value (older process A) the requested one is compared and that is
    stated; then we can only tell whether config and A's request disagree.

    Does not abort: an FOV mismatch only skews back-projection **magnitude**, direction
    stays correct, so flight works. But measurements get silently contaminated, so it
    must be visible.
    """
    cam = sim_info.get("camera")
    cam = cam if isinstance(cam, dict) else {}
    actual, kind = cam.get("hfov_deg"), "measured"
    if actual is None:
        actual, kind = cam.get("fov_deg"), "requested (no measured value)"
    if actual is None:
        actual, kind = sim_info.get("hfov_deg"), "measured"
    if actual is None:
        return
    want = float(config.get("fov_horizontal", mission.DEFAULT_FOV_HORIZONTAL_DEG))
    try:
        gap = abs(float(actual) - want)
    except (TypeError, ValueError):
        return
    if gap > FOV_MISMATCH_TOLERANCE_DEG:
        _say(
            f"[FRAME] ⚠️ FOV mismatch: config fov_horizontal={want:g}° vs A {kind} "
            f"{float(actual):.2f}° (diff {gap:.2f}°) — back-projected distances are off by that ratio"
        )
    else:
        _say(f"[FRAME] FOV ok: config {want:g}° ≈ A {kind} {float(actual):.2f}°")


def probe_frame_source(frame_path: str, wait_s: float = FRAME_WAIT_S) -> tuple[int, int, bool]:
    """Returns `(width, height, needs_attention)`.

    **No fallback resolution**, same as the AirSim probe: if the file cannot be read,
    flight is impossible. `(0, 0, True)` beats taking off with a fake resolution and
    silently wrong back-projection.
    """
    deadline = time.monotonic() + float(wait_s)
    while not os.path.exists(frame_path):
        if time.monotonic() >= deadline:
            _say(f"[FRAME] frame file not found: {frame_path}")
            _say("[FRAME] Check that process A (Isaac Sim + Pegasus) is running and writing to that path")
            _say("[FRAME] (start it with `bash scripts/isaacsim/pegasus_sim.sh`).")
            _say("[FRAME] A's --frame-path must match frame_path in the config.")
            return (0, 0, True)
        _say(f"[FRAME] waiting for frame... ({frame_path})")
        time.sleep(1.0)

    try:
        img = cv2.imread(frame_path)
    except Exception as exc:  # noqa: BLE001
        _say(f"[FRAME] error reading file: {exc!r}")
        return (0, 0, True)
    if img is None or getattr(img, "size", 0) == 0:
        _say(f"[FRAME] could not read file (being written or corrupt): {frame_path}")
        return (0, 0, True)

    height, width = img.shape[:2]
    _say(f"[FRAME] frame source: {frame_path} — {width}x{height}")
    if mission.capture_too_small(width, height):
        _say(
            f"[FRAME] ⚠️ resolution below recommended (short side {mission.MIN_CAPTURE_SHORT_SIDE_PX}px) "
            f"— 2D grounding gets harder."
        )
        _say("[FRAME] Raise A's --camera-width / --camera-height.")
        return (width, height, True)
    return (width, height, False)


def grab_frame(
    frame_path: str, max_age_s: float | None = None
) -> tuple[np.ndarray | None, float | None]:
    """Returns `(frame, age_s)`.

    A stale frame is **returned with only a warning**: B cannot tell a slow A from a
    stalled one (RTF 0.5 is normal), and that beats stopping the loop. The age is kept
    as `frame_age_s` in `steps.jsonl` for later analysis.
    """
    age: float | None = None
    mtime: float | None = None
    try:
        mtime = os.path.getmtime(frame_path)
        age = max(0.0, time.time() - mtime)
    except OSError:
        age = None
    try:
        img = cv2.imread(frame_path)
    except Exception as exc:  # noqa: BLE001
        _say(f"[FRAME] capture error: {exc!r}")
        return (None, None)
    if img is None or getattr(img, "size", 0) == 0:
        _say(f"[FRAME] could not read frame: {frame_path}")
        return (None, None)
    if max_age_s is not None and age is not None and age > float(max_age_s):
        _say(f"[FRAME] warning: frame is {age:.1f}s old (limit {float(max_age_s):g}s) — is A stalled?")
    grab_frame.last_mtime = mtime          # join key; `_record_step` records it as-is
    return (img, age)


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


def _connect_link(config: dict) -> Any:
    """There is a single link: created here and injected into the controller.

    `mavsdk_server` holds UDP 14540, so a second connect would fail.
    """
    from vlmpilot.controller.isaacsim.pegasus_link import PegasusLink  # mavsdk is only in the pegasus extra

    address = str(config.get("mavsdk_address", DEFAULT_MAVSDK_ADDRESS))
    # **Always pass `grpc_port`.** Otherwise `PegasusLink`'s default 50051 is used and
    # slot isolation **breaks on gRPC only**: MAVSDK UDP differs per slot, but every
    # slot's mavsdk_server grabs 50051.
    #
    # Measured: a slot-5 batch had `grpc_port: 50056` in its config, yet all 138 gRPC
    # addresses in the logs were `127.0.0.1:50051`. Each episode's new B starts a server;
    # if the previous one has not died yet, the new server cannot bind and exits at once,
    # and the first RPC (`arm`) fails with `AioRpcError: Stream removed (Socket closed)`.
    # In a 20-episode batch, 7 (35%) came out empty that way. That is also why raising
    # the settle time 60 -> 90 -> 150 s did not help (the race is at B startup, not PX4
    # readiness).
    grpc_port = int(config.get("grpc_port", 50051))
    # Diagnostics: if set, attach to an already-running server (see `PegasusLink.server_address`)
    server_address = config.get("mavsdk_server_address") or None
    _say(f"[LINK] connecting to PX4 ({address} · gRPC {grpc_port}"
         + (f" · attach {server_address}" if server_address else "") + ")")
    link = PegasusLink(address=address, grpc_port=grpc_port,
                       server_address=server_address)
    link.connect()
    _say("[LINK] connected")
    return link


def _assemble(
    config: dict, width: int, height: int, run_dir: str, link: Any
) -> tuple[Any, PegasusFlightController]:
    """PinholeCamera -> backend -> VlmPolicy -> PegasusFlightController (link injected)."""
    camera = PinholeCamera(
        width, height, float(config.get("fov_horizontal", mission.DEFAULT_FOV_HORIZONTAL_DEG))
    )
    backend = vlm_backends.build_backend(
        config, response_schema=ACTION_RESPONSE_SCHEMA, schema_name=SCHEMA_NAME
    )  # VlmConfigError propagates -> exit code 1
    policy = VlmPolicy.from_config(config, camera, backend)
    controller = PegasusFlightController(
        config=config,
        image_width=width,
        image_height=height,
        run_dir=run_dir,
        policy=policy,
        camera=camera,
        on_decision=lambda record: artifacts.save_decision(run_dir, record),
        link=link,
    )
    return backend, controller


# ---------------------------------------------------------------------------
# Loop and finally
# ---------------------------------------------------------------------------


class _EpisodeState:
    """Mutable state shared by the loop and finally. Pegasus version of `mission._EpisodeState`."""

    def __init__(self) -> None:
        self.instruction = ""
        self.run_dir: str | None = None
        self.link: Any = None
        self.backend: Any = None
        self.controller: PegasusFlightController | None = None
        self.reached = False
        self.declared_once = False
        self.airborne_phase = False
        self.outcome = "error: unknown"
        self.steps = 0
        self.sim_info: dict = {}
        self.started_at = time.time()
        # Cutoff the recording assembly uses to pick only this run. Re-stamped right before takeoff.
        self.record_from_ns: int | None = None


def _safe_hover(state: _EpisodeState, *, wait_idle: bool) -> None:
    try:
        if wait_idle and state.controller is not None:
            state.controller.wait_until_idle(timeout=mission.ARRIVAL_IDLE_TIMEOUT_S)
        if state.link is not None:
            state.link.hover()
    except Exception as exc:  # noqa: BLE001 — KeyboardInterrupt deliberately passes through
        _say(f"[LINK] hover failed (ignored): {exc}")


def _request_gt_hold(path: str | None) -> None:
    """Tell A "GT pixels are needed now" by touching a file.

    A's measurement mode attaches the annotator regardless of distance while this file's
    mtime is fresh. **Failure never blocks flight**: A also has an altitude latch, and if
    that misses too, the GT join reports the episode as not measured. Losing one
    layer never yields a **silently** wrong result.
    """
    if not path:
        return
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(str(time.time()))
        _say(f"[GT] asked A to hold GT: {path}")
    except OSError as exc:
        _say(f"[GT] could not write GT hold request (continuing): {exc!r}")


def _save_final_frame(state: _EpisodeState, frame_path: str) -> None:
    """After arrival settles, read and save one more frame. Failures are ignored."""
    try:
        time.sleep(mission.ARRIVAL_SETTLE_S)
        final, _age = grab_frame(frame_path)
        if final is not None and state.run_dir:
            jpg_path, _ = artifacts.save_arrival_frame(state.run_dir, final)
            _say(f"[ARRIVAL] saved final arrival frame: {jpg_path}")
    except Exception as exc:  # noqa: BLE001
        _say(f"[ARRIVAL] failed to save final frame (ignored): {exc}")


def _write_run_json(state: _EpisodeState, config: dict) -> None:
    """Not required by the CLI contract; offline tools read this file.

    `depth_source` is **the raw config value**. If a fallback happened,
    `depth_source_effective` is recorded too, so contamination can be detected later.
    """
    if not state.run_dir:
        return
    ctrl = state.controller
    row: dict[str, Any] = {
        "backend": "pegasus",
        "outcome": state.outcome,
        "instruction": state.instruction,
        "steps": state.steps,
        "wall_s": round(time.time() - state.started_at, 1),
        "api_provider": config.get("api_provider"),
        "model": getattr(state.backend, "model", None) if state.backend is not None else None,
        "depth_source": config.get("depth_source", "model"),
        "arrival_frames": int(config.get("arrival_frames", mission.DEFAULT_ARRIVAL_FRAMES) or 1),
        "tuning": {key: config.get(key) for key in TUNING_KEYS},
        # Copy A's test conditions (scene, camera, FOV, RTF, livestream, wind) verbatim
        "sim": state.sim_info or None,
    }
    if ctrl is not None and ctrl.depth_source_effective:
        row["depth_source_effective"] = ctrl.depth_source_effective
    try:
        with open(os.path.join(state.run_dir, "run.json"), "w", encoding="utf-8") as fh:
            json.dump(row, fh, ensure_ascii=False, indent=2)
        _say(f"[RUN] run.json written: {os.path.join(state.run_dir, 'run.json')} (outcome={state.outcome})")
    except Exception as exc:  # noqa: BLE001
        _say(f"[RUN] failed to write run.json (ignored): {exc}")


def _record_root(state: _EpisodeState) -> str | None:
    """Whether A has recording on. `camera.record_root` in `sim_info` is the only source."""
    cam = state.sim_info.get("camera")
    root = cam.get("record_root") if isinstance(cam, dict) else None
    return str(root) if root else None


def _encode_run_video(state: _EpisodeState) -> None:
    """Assemble an mp4 from **only this run's span** of the frames A has accumulated.

    A keeps appending for its whole session, so `artifacts.snapshot_recording_dirs`-style
    "new folder" detection does not work (A's folder exists before B starts). Instead we
    cut the run span by the nanosecond timestamps in filenames, build a symlink tree,
    and call the existing `artifacts.encode_flight_videos` on it **unchanged**.

    Symlinks avoid copying gigabytes; ffmpeg reads via glob, so symlinks suffice.
    """
    root = _record_root(state)
    if not root or not state.run_dir or state.record_from_ns is None:
        return
    images = os.path.join(root, "images")
    try:
        frames = sorted(glob.glob(os.path.join(images, "img_*_0_0_*.png")))
    except OSError as exc:
        _say(f"[RECORD] could not read recording folder (skipping assembly): {exc}")
        return
    picked = []
    for f in frames:
        try:
            ts = int(os.path.splitext(os.path.basename(f))[0].rsplit("_", 1)[-1])
        except ValueError:
            continue
        if ts >= state.record_from_ns:
            picked.append(f)
    if len(picked) < 2:
        _say(f"[RECORD] only {len(picked)} frame(s) in this run's span — skipping assembly")
        return

    stage_root = os.path.join(state.run_dir, "rec")
    stage = os.path.join(stage_root, "run", "images")
    try:
        os.makedirs(stage, exist_ok=True)
        for f in picked:
            link = os.path.join(stage, os.path.basename(f))
            if not os.path.lexists(link):
                os.symlink(os.path.abspath(f), link)
        _say(f"[RECORD] assembling {len(picked)} frame(s) from this run's span")
        artifacts.encode_flight_videos(stage_root, set(), state.run_dir)
    except Exception as exc:  # noqa: BLE001 — assembly failure must not block the rest of finalize
        _say(f"[RECORD] assembly error (ignored): {exc!r}")
    finally:
        # The symlink tree is not an output; keep only the mp4.
        try:
            shutil.rmtree(stage_root, ignore_errors=True)
        except Exception:  # noqa: BLE001
            pass


def _finalize(state: _EpisodeState, config: dict) -> None:
    """Each stage is isolated so a second Ctrl+C loses only that stage. No `return` here.

    Order: mp4 assembly -> grid -> run.json -> shutdown -> SEQUENCE FINISHED.
    mp4 first follows `mission.py`; run.json comes after the grid so neither is caught
    by the other's failure.
    """
    try:
        _encode_run_video(state)
    except (Exception, KeyboardInterrupt) as exc:
        _say(f"[RECORD] assembly stage aborted: {exc!r}")
    if state.run_dir:
        try:
            model = getattr(state.backend, "model", None) if state.backend is not None else None
            artifacts.build_recognition_grid(state.run_dir, instruction=state.instruction, model=model)
        except (Exception, KeyboardInterrupt) as exc:
            _say(f"[GRID] generation failed (ignored): {exc}")
    try:
        _write_run_json(state, config)
    except (Exception, KeyboardInterrupt) as exc:
        _say(f"[RUN] run.json stage aborted: {exc!r}")
    if state.controller is not None:
        try:
            state.controller.shutdown(land=not (state.reached or state.declared_once))
        except (Exception, KeyboardInterrupt) as exc:
            _say(f"[CTRL] error during finalize (ignored): {exc!r}")
    if state.reached:
        _say()
        _say("=== SEQUENCE FINISHED: target reached ===")


def fly(args: Any) -> int:
    """Pegasus mode entry point. Returns the process exit code."""
    config_path = require_config_path(args)
    debug = bool(getattr(args, "debug", False))
    if getattr(args, "port", None):
        # Ignoring silently would make users think they changed the port
        _say(f"[CONFIG] --port {args.port} is AirSim-only — ignored; pegasus connects over MAVLink")
    _say(f"[CONFIG] config file: {config_path}")
    config = mission.load_mission_config(config_path)

    frame_path = str(config.get("frame_path", DEFAULT_FRAME_PATH))
    sim_info_path = str(config.get("sim_info_path", DEFAULT_SIM_INFO_PATH))
    frame_max_age_s = config.get("frame_max_age_s", DEFAULT_FRAME_MAX_AGE_S)
    loop_delay = float(config.get("command_loop_delay", 0) or 0)
    arrival_frames = int(
        config.get("arrival_frames", mission.DEFAULT_ARRIVAL_FRAMES) or mission.DEFAULT_ARRIVAL_FRAMES
    )
    output_root = str(config.get("output_root", mission.DEFAULT_OUTPUT_ROOT))
    # **No step cap in the flight loop.** The cap **belongs to judging**: flight ends on
    # an arrival declaration or search give-up (`SEARCH_GIVE_UP_STEPS`), and the 60-step
    # cut is applied only at re-judging. AirSim has the same structure, which is what
    # makes the two backends comparable.
    #
    # B -> A signal path. A in measurement mode attaches the annotator on step 1 when it sees this.
    gt_hold_path = config.get("gt_hold_path")
    continue_after_arrival = os.environ.get(mission.ENV_CONTINUE_AFTER_ARRIVAL) == "1"
    _say(
        f"[CONFIG] frame_path={frame_path!r} max_age={frame_max_age_s}s "
        f"loop_delay={loop_delay:g}s arrival_frames={arrival_frames} "
        f"continue_after_arrival={continue_after_arrival}"
    )

    state = _EpisodeState()
    try:
        state.sim_info = read_sim_info(sim_info_path)
        if state.sim_info:
            _say(f"[SIM] A's conditions: {json.dumps(state.sim_info, ensure_ascii=False)}")
            if state.sim_info.get("livestream"):
                _say("[SIM] ⚠️ livestream is on — RTF drops and wall-clock time grows")
        else:
            _say(f"[SIM] could not read sim_info.json ({sim_info_path}) — proceeding without sim conditions")

        width, height, needs_attention = probe_frame_source(frame_path)
        if width == 0 or height == 0:
            # Without the resolution back-projection is impossible; exit without asking
            _say("[FRAME] frame resolution unknown — exiting without takeoff")
            return 1
        if needs_attention and not mission.confirm_low_resolution():
            return 1
        check_fov(config, state.sim_info)

        state.link = _connect_link(config)
        state.run_dir = artifacts.create_run_dir(output_root)
        _say(f"[RUN] output folder: {state.run_dir}")
        state.backend, state.controller = _assemble(config, width, height, state.run_dir, state.link)

        state.instruction = mission.read_instruction(mission.INSTRUCTION_PROMPT)
        if not state.instruction:
            _say("[INPUT] instruction empty or input closed — exiting without takeoff (exit 2)")
            state.outcome = "error: no instruction"
            return 2
        state.airborne_phase = True
        _say(f"[START] instruction: {state.instruction!r} — taking off in {mission.START_DELAY_S:g}s")
        time.sleep(mission.START_DELAY_S)
        # Start of the recording span. Stamped right **before** takeoff so the wait for
        # instruction input is not in the video. If A is not recording, assembly is skipped.
        state.record_from_ns = time.time_ns()
        if _record_root(state):
            _say(f"[RECORD] A is recording — this run's span will be assembled in finally")
        state.controller.takeoff()
        _say("[LOOP] closed loop started: wait queue -> capture -> infer -> command")
        _say("[LOOP] a single Ctrl+C runs finalize (grid, run record, landing)")

        arrival_count = 0
        search_streak = 0
        state.outcome = "interrupted"
        while True:
            if debug:
                _say("[LOOP] waiting for previous command to finish...")
                state.controller.wait_until_idle(timeout=mission.IDLE_WAIT_TIMEOUT_S, verbose=True)
                _say("[LOOP] wait done — capturing frame")
            else:
                state.controller.wait_until_idle(timeout=mission.IDLE_WAIT_TIMEOUT_S)

            if state.steps == 0:
                # **Tell A "GT pixels are needed now", right before capture.**
                #
                # A's measurement mode only attaches the annotator near the target, but
                # step 1 of an episode is usually more than 15 m away. Then
                # `seen_at_start` silently becomes all False and the judge's searching
                # validity gate never fires. A does not know B's step number (the sidecar
                # is write-only), so **B telling it** is the only observable step-1 moment.
                #
                # Why here and not outside the loop: the request's validity window is
                # wall-clock (A and B are separate processes), and the preceding
                # `wait_until_idle` grows at low RTF. Keeping request and capture
                # adjacent allows a short window.
                #
                # If A is absent or older, nobody reads this file and A's altitude latch
                # covers instead; either alone avoids a **silently** wrong result.
                _request_gt_hold(gt_hold_path)

            _cap_t0 = time.monotonic()
            frame, age = grab_frame(frame_path, max_age_s=frame_max_age_s)
            if frame is None:
                _say(f"[CAPTURE] no frame — retrying in {mission.CAPTURE_RETRY_DELAY_S:g}s")
                time.sleep(mission.CAPTURE_RETRY_DELAY_S)
                continue
            # The controller does not read the frame source itself, so this is the only path
            state.controller.last_frame_ms = round((time.monotonic() - _cap_t0) * 1000.0, 1)
            state.controller.last_frame_age_s = None if age is None else round(age, 3)
            # **Do not derive frame time by subtraction.** `t − frame_age_s` is "row build
            # time − age measured at grab", not frame time, and that gap made joins drop
            # rows at random (`no_gt`).
            state.controller.last_frame_t = getattr(grab_frame, "last_mtime", None)
            _say(FRAME_MARKER)
            yaw0 = state.controller.current_yaw_rad()

            response_text, arrived, searching = state.controller.run_step(
                frame, state.instruction, heading_ref_rad=yaw0
            )
            _say(f"{STEP_REPORT_PREFIX}{response_text}")
            state.steps += 1

            if searching:
                search_streak += 1
                if search_streak >= mission.SEARCH_GIVE_UP_STEPS:
                    _say(
                        f"[SEARCH] target not found for {mission.SEARCH_GIVE_UP_STEPS} consecutive steps — "
                        "ending."
                    )
                    state.outcome = "search_giveup"
                    break
            else:
                search_streak = 0

            if arrived is True:
                arrival_count += 1
                _say(f"[ARRIVAL] Target-reached signal (VLM) {arrival_count}/{arrival_frames}")
                _safe_hover(state, wait_idle=True)
                if arrival_count >= arrival_frames:
                    _say("[ARRIVAL] FINISHED (arrival declared)")
                    if continue_after_arrival:
                        state.declared_once = True
                        arrival_count = 0
                    else:
                        _say("[ARRIVAL] Target reached — hovering and ending sequence.")
                        state.reached = True
                        state.outcome = "arrived"
                        _safe_hover(state, wait_idle=False)
                        _save_final_frame(state, frame_path)
                        break
            elif arrived is False:
                arrival_count = 0
            # arrived is None -> no VLM response. Leave the counter alone.

            if loop_delay > 0:
                time.sleep(loop_delay)
    except KeyboardInterrupt:
        _say("[LOOP] interrupt received — moving on to finalize")
        state.outcome = "interrupted"
    except Exception as exc:  # noqa: BLE001
        # Exceptions before takeoff propagate (exit code 1). Exceptions in flight are
        # absorbed, the vehicle lands, and we exit 0 (same rule as mission.py).
        state.outcome = f"error: {type(exc).__name__}"
        if not state.airborne_phase:
            raise
        _say(f"[LOOP] ending episode on unexpected error: {type(exc).__name__}: {exc}")
        traceback.print_exc(file=sys.stdout)
        sys.stdout.flush()
    finally:
        _finalize(state, config)
    return 0
