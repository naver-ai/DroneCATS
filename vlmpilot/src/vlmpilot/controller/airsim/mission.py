# DroneCATS
# Copyright (c) 2026-present NAVER Cloud Corp.
# Apache-2.0

"""mission.py — one episode's lifecycle.

config → single RPC connection → camera check → run dir → object assembly →
instruction → takeoff → main loop → finally. This module is the only assembly
point (§2.5) and the source of the ``[ARRIVAL]``/``[SEARCH]``/``[RECORD] recording started``
markers; ``[ACTION]`` lines come from ``policy``, ``[MOVEMENT]``/``[YAW]``/``[STEP]``
from ``control``.
"""

from __future__ import annotations

import os
import sys
import time
import traceback
from typing import Any

import airsim
import cv2
import numpy as np
import yaml

from vlmpilot import artifacts
from vlmpilot.policy import vlm_backends
from vlmpilot.policy.camera import PinholeCamera
from vlmpilot.controller.airsim.flight_controller import FlightController
from vlmpilot.policy.policy import ACTION_RESPONSE_SCHEMA, VlmPolicy

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_CONFIG_PATH = "configs/airsim/single-drone/airsim_gemini.yaml"
DEFAULT_RPC_PORT = 41451
DEFAULT_CAMERA_NAME = "0"
DEFAULT_RECORDING_ROOT = "/home/airsim/Documents/AirSim"
DEFAULT_OUTPUT_ROOT = "outputs"
DEFAULT_ARRIVAL_FRAMES = 1
DEFAULT_FOV_HORIZONTAL_DEG = 90.0

# Below this short side the point the VLM picks gets too coarse to steer by.
MIN_CAPTURE_SHORT_SIDE_PX = 480


def capture_too_small(width: int, height: int) -> bool:
    return min(width, height) < MIN_CAPTURE_SHORT_SIDE_PX

START_DELAY_S = 3.0  # never shortened (R-36)
CAPTURE_RETRY_DELAY_S = 1.0
IDLE_WAIT_TIMEOUT_S = 30.0
ARRIVAL_IDLE_TIMEOUT_S = 10.0
ARRIVAL_SETTLE_S = 1.5
SEARCH_GIVE_UP_STEPS = 24

ENV_CONTINUE_AFTER_ARRIVAL = "VLMPILOT_CONTINUE_AFTER_ARRIVAL"
ENV_NO_RECORD = "VLMPILOT_NO_RECORD"
ENV_PORT = "MY_PORT"   # optional environment default for the AirSim RPC port

SCHEMA_NAME = "vlmpilot_action"
TEST_IMAGE_PATH = "frame_airsim_test.jpg"
TEST_INSTRUCTION = "approach the most prominent building in view and stop in front of it"

INSTRUCTION_PROMPT = "[INPUT] Enter this flight's goal on one line (press Enter to confirm):"
LOW_RES_PROMPT = "[CAMERA] Resolution is below the recommended value. Take off anyway? [y/N] "

# SPEC-GAP: `Action Response:` is not among the documented markers, but the
# verifier counts steps with `out.count("Action Response:")`
# (verifier/airsim/run_testset.py), and the line belongs to the
# mission loop. We print it exactly once per run_step so the runner's step count works.
STEP_REPORT_PREFIX = "Action Response: "
FRAME_MARKER = "[FRAME] captured"   # contract §3.4: time the frame is fixed


def _say(line: str = "") -> None:
    print(line, flush=True)


# ---------------------------------------------------------------------------
# §3.2 config + port
# ---------------------------------------------------------------------------


def config_flag(config: Any, key: str, env_name: str) -> bool:
    """One behavior-changing toggle. config key > env > off.

    env remains only for backward compatibility; the runner shells (verifier/) use these names.
    For env, only exactly ``"1"`` is true (R-313).
    """
    value = (config or {}).get(key)
    if value is None:
        return os.environ.get(env_name) == "1"
    return bool(value)


def load_mission_config(path: str) -> dict:
    """Load the YAML config; on any failure print one line and return ``{}`` (R-11)."""
    try:
        with open(path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
    except Exception as exc:  # noqa: BLE001 — config trouble must not kill the process here
        _say(f"[CONFIG] could not read the config — continuing with defaults ({type(exc).__name__}: {exc})")
        return {}
    if data is None:
        return {}
    if not isinstance(data, dict):
        _say(f"[CONFIG] config top level is not a mapping ({type(data).__name__}) — continuing with defaults")
        return {}
    return data


def _port_from_env() -> int:
    raw = os.environ.get(ENV_PORT, "")
    try:
        return int(raw)
    except (TypeError, ValueError):
        return 0


def resolve_port(cli_port: int | None, config: dict) -> int:
    """``--port > $MY_PORT > config["airsim_rpc_port"] > 41451`` (R-12).

    An override is written back into ``config`` so the controller (which reads the
    config) connects to the same port as the mission.
    """
    override = cli_port or _port_from_env() or None
    if override:
        config["airsim_rpc_port"] = int(override)
        return int(override)
    try:
        return int(config.get("airsim_rpc_port", DEFAULT_RPC_PORT) or DEFAULT_RPC_PORT)
    except (TypeError, ValueError):
        return DEFAULT_RPC_PORT


# ---------------------------------------------------------------------------
# §3.3 camera check · §3.4 frame capture
# ---------------------------------------------------------------------------


def _scene_request(camera_name: str) -> Any:
    return airsim.ImageRequest(camera_name, airsim.ImageType.Scene, False, False)


def probe_camera(client: Any, camera_name: str) -> tuple[int, int, bool] | None:
    """Query camera info + one Scene frame; return ``(width, height, needs_attention)`` (R-20…R-23).

    ``None`` when no frame comes back: back-projection needs the real capture size,
    so the mission stops before takeoff instead of guessing one.
    """
    try:
        info = client.simGetCameraInfo(camera_name)
        _say(f"[CAMERA] name={camera_name!r} fov={getattr(info, 'fov', None)} pose={getattr(info, 'pose', None)}")
        responses = client.simGetImages([_scene_request(camera_name)])
        if not responses or int(getattr(responses[0], "width", 0) or 0) <= 0:
            _say("[CAMERA] no frame from the camera — check CaptureSettings in AirSim settings.json")
            return None
        width = int(responses[0].width)
        height = int(responses[0].height)
        if not capture_too_small(width, height):
            _say(f"[CAMERA] resolution {width}x{height} — OK")
            return width, height, False
        _say(
            f"[CAMERA] warning: resolution {width}x{height} is below the recommended minimum (short side {MIN_CAPTURE_SHORT_SIDE_PX}px). "
            "The default AirSim camera is small, so raise Width/Height under CaptureSettings "
            "in AirSim settings.json."
        )
        return width, height, True
    except Exception as exc:  # noqa: BLE001
        _say(f"[CAMERA] error while checking the camera ({type(exc).__name__}: {exc})")
        return None


def grab_scene_bgr(client: Any, camera_name: str) -> np.ndarray | None:
    """One uncompressed Scene frame as a BGR ``uint8`` array, or ``None`` (R-30…R-32)."""
    try:
        responses = client.simGetImages([_scene_request(camera_name)])
        if not responses:
            _say("[CAPTURE] empty response — no frame")
            return None
        first = responses[0]
        height, width = int(first.height), int(first.width)
        if height == 0 or width == 0:
            _say(f"[CAPTURE] invalid frame size {width}x{height}")
            return None
        buf = np.frombuffer(first.image_data_uint8, dtype=np.uint8)
        return buf.reshape(height, width, 3)  # AirSim's raw Scene buffer is already BGR
    except Exception as exc:  # noqa: BLE001
        _say(f"[CAPTURE] frame decode failed ({type(exc).__name__}: {exc})")
        return None


def _current_yaw_rad(client: Any) -> float | None:
    """Heading at capture time (R-302). ``None`` (= no correction) if the state read fails."""
    try:
        orientation = client.getMultirotorState().kinematics_estimated.orientation
        return float(airsim.to_eularian_angles(orientation)[2])
    except Exception as exc:  # noqa: BLE001
        _say(f"[STATE] heading read failed — no yaw correction this step ({exc})")
        return None


# ---------------------------------------------------------------------------
# §3.6 stdin
# ---------------------------------------------------------------------------


def read_instruction(prompt: str) -> str:
    """Print ``prompt`` and read one stripped line from stdin; ``""`` on EOF (R-40…R-42)."""
    _say(prompt)
    try:
        line = input()
    except EOFError:
        return ""
    return line.strip()


def confirm_low_resolution() -> bool:
    """Ask y/n only on a real terminal; a piped stdin always proceeds (R-24, E-1)."""
    if not sys.stdin.isatty():
        _say("[CAMERA] non-interactive input — skipping confirmation and continuing")
        return True
    print(LOW_RES_PROMPT, end="", flush=True)
    try:
        answer = input()
    except EOFError:
        return True
    if answer.strip().lower() == "y":
        return True
    _say("[CAMERA] user cancelled — exiting without takeoff")
    return False


# ---------------------------------------------------------------------------
# §2.5 assembly
# ---------------------------------------------------------------------------


def _assemble(config: dict, width: int, height: int, run_dir: str, client: Any) -> tuple[Any, FlightController]:
    """PinholeCamera → backend → VlmPolicy → FlightController (single injected client)."""
    camera = PinholeCamera(width, height, float(config.get("fov_horizontal", DEFAULT_FOV_HORIZONTAL_DEG)))
    backend = vlm_backends.build_backend(
        config, response_schema=ACTION_RESPONSE_SCHEMA, schema_name=SCHEMA_NAME
    )  # VlmConfigError propagates → exit 1
    policy = VlmPolicy.from_config(config, camera, backend)
    controller = FlightController(
        config=config,
        image_width=width,
        image_height=height,
        run_dir=run_dir,
        policy=policy,
        camera=camera,
        on_decision=lambda record: artifacts.save_decision(run_dir, record),
        on_command=policy.note_command,
        client=client,
    )
    return backend, controller


def _connect(port: int) -> Any:
    client = airsim.MultirotorClient(port=port)
    client.confirmConnection()
    _say(f"Connected to AirSim (RPC port {port})")
    return client


# ---------------------------------------------------------------------------
# §3.5–§3.9 flight
# ---------------------------------------------------------------------------


class _EpisodeState:
    """Mutable flags shared between the loop and the finally chain."""

    def __init__(self) -> None:
        self.instruction = ""  # initialised before anything can fail (R-43)
        self.run_dir: str | None = None
        self.client: Any = None
        self.backend: Any = None
        self.controller: FlightController | None = None
        self.recording = False
        self.known_dirs: set[str] = set()
        self.reached = False
        self.declared_once = False
        self.airborne_phase = False  # True once the instruction is accepted (see fly())


def _start_recording(state: _EpisodeState, recording_root: str) -> None:
    """R-60/R-61: start AirSim recording right after takeoff; failure never blocks flight."""
    try:
        state.known_dirs = artifacts.snapshot_recording_dirs(recording_root)
        state.client.startRecording()
        state.recording = True
        _say(f"[RECORD] recording started ({recording_root})")
    except Exception as exc:  # noqa: BLE001
        _say(f"[RECORD] recording start failed (continuing): {exc}")


def _safe_hover(state: _EpisodeState, *, wait_idle: bool) -> None:
    try:
        if wait_idle and state.controller is not None:
            state.controller.wait_until_idle(timeout=ARRIVAL_IDLE_TIMEOUT_S)
        state.client.hoverAsync().join()
    except Exception as exc:  # noqa: BLE001 — KeyboardInterrupt deliberately passes through (E-4)
        _say(f"hover failed (ignored): {exc}")


def _save_final_frame(state: _EpisodeState, camera_name: str) -> None:
    """R-315: settle, grab one more frame, write decision_<ts>_arrived.{jpg,json}."""
    try:
        time.sleep(ARRIVAL_SETTLE_S)
        final = grab_scene_bgr(state.client, camera_name)
        if final is not None and state.run_dir:
            jpg_path, _ = artifacts.save_arrival_frame(state.run_dir, final)
            _say(f"[ARRIVAL] final arrival frame saved: {jpg_path}")
    except Exception as exc:  # noqa: BLE001
        _say(f"[ARRIVAL] final frame save failed (ignored): {exc}")


def _finalize(state: _EpisodeState, recording_root: str) -> None:
    """R-70: stop recording → mp4 → grid → shutdown → (reached) SEQUENCE FINISHED.

    Every stage is isolated so a second Ctrl+C only loses that one stage (E-5).
    Contains no ``return`` — the caller's ``return`` values must survive (R-78).
    """
    if state.recording:
        try:
            state.client.stopRecording()
        except (Exception, KeyboardInterrupt):  # R-72: silent
            pass
        try:
            artifacts.encode_flight_videos(recording_root, state.known_dirs, state.run_dir or "recordings")
        except (Exception, KeyboardInterrupt) as exc:  # encode_flight_videos already isolates; belt and braces
            _say(f"[RECORD] assembly step aborted: {exc!r}")
    if state.run_dir:
        try:
            model = getattr(state.backend, "model", None) if state.backend is not None else None
            artifacts.build_recognition_grid(state.run_dir, instruction=state.instruction, model=model)
        except (Exception, KeyboardInterrupt) as exc:
            _say(f"[GRID] build failed (ignored): {exc}")
    if state.controller is not None:
        try:
            state.controller.shutdown(land=not (state.reached or state.declared_once))
        except (Exception, KeyboardInterrupt) as exc:
            _say(f"[CTRL] error during wrap-up (ignored): {exc!r}")
    if state.reached:
        _say()
        _say("=== SEQUENCE FINISHED: target reached ===")


def fly(args: Any) -> int:
    """Airsim mode body. Returns the process exit code (§3.5–§3.9)."""
    config_path = getattr(args, "config", None) or DEFAULT_CONFIG_PATH
    debug = bool(getattr(args, "debug", False))
    _say(f"[CONFIG] config file: {config_path}")
    config = load_mission_config(config_path)
    port = resolve_port(getattr(args, "port", None), config)

    camera_name = str(config.get("camera_name", DEFAULT_CAMERA_NAME))
    loop_delay = float(config.get("command_loop_delay", 0) or 0)
    arrival_frames = int(config.get("arrival_frames", DEFAULT_ARRIVAL_FRAMES) or DEFAULT_ARRIVAL_FRAMES)
    recording_root = str(config.get("recording_root", DEFAULT_RECORDING_ROOT))
    output_root = str(config.get("output_root", DEFAULT_OUTPUT_ROOT))
    continue_after_arrival = config_flag(config, "continue_after_arrival", ENV_CONTINUE_AFTER_ARRIVAL)
    # Default False — startRecording() writes a PNG per step
    # (930 GB over 915 runs measured); the multi path reads the same key with the same default.
    # no_record is a veto over auto_record: if both are true, don't record.
    record_wanted = (bool(config.get("auto_record", False))
                     and not config_flag(config, "no_record", ENV_NO_RECORD))
    _say(f"[CONFIG] camera={camera_name!r} loop_delay={loop_delay:g}s arrival_frames={arrival_frames} "
         f"continue_after_arrival={continue_after_arrival} record={record_wanted}")

    state = _EpisodeState()
    try:
        state.client = _connect(port)
        probe = probe_camera(state.client, camera_name)
        if probe is None:
            _say("[CAMERA] camera unavailable — exiting without takeoff")
            return 1
        width, height, needs_attention = probe
        if needs_attention and not confirm_low_resolution():
            return 1  # inside try: finally runs, value survives (R-78)

        state.run_dir = artifacts.create_run_dir(output_root)
        _say(f"[RUN] output folder: {state.run_dir}")
        state.backend, state.controller = _assemble(config, width, height, state.run_dir, state.client)

        state.instruction = read_instruction(INSTRUCTION_PROMPT)
        if not state.instruction:
            _say("[INPUT] instruction is empty or input was closed — exiting without takeoff (exit 2)")
            return 2
        state.airborne_phase = True
        _say(f"[START] instruction: {state.instruction!r} — taking off in {START_DELAY_S:g}s")
        time.sleep(START_DELAY_S)
        state.controller.takeoff()
        if record_wanted:
            _start_recording(state, recording_root)
        _say("[LOOP] closed loop started: wait for queue → capture → infer → command")
        _say("[LOOP] a single Ctrl+C runs the wrap-up (grid · video · landing)")

        arrival_count = 0
        search_streak = 0
        while True:
            if debug:
                _say("[LOOP] waiting for the previous command to finish…")
                state.controller.wait_until_idle(timeout=IDLE_WAIT_TIMEOUT_S, verbose=True)
                _say("[LOOP] wait over — capturing a frame")
            else:
                state.controller.wait_until_idle(timeout=IDLE_WAIT_TIMEOUT_S)

            _cap_t0 = time.monotonic()
            frame = grab_scene_bgr(state.client, camera_name)
            state.controller.last_frame_ms = round((time.monotonic() - _cap_t0) * 1000.0, 1)
            if frame is None:
                _say(f"[CAPTURE] could not get a frame — retrying in {CAPTURE_RETRY_DELAY_S:g}s")
                time.sleep(CAPTURE_RETRY_DELAY_S)
                continue
            _say(FRAME_MARKER)
            yaw0 = _current_yaw_rad(state.client)

            response_text, arrived, searching = state.controller.run_step(
                frame, state.instruction, heading_ref_rad=yaw0
            )
            _say(f"{STEP_REPORT_PREFIX}{response_text}")

            if searching:
                search_streak += 1
                if search_streak >= SEARCH_GIVE_UP_STEPS:
                    _say(f"[SEARCH] {SEARCH_GIVE_UP_STEPS} consecutive misses — target not found, ending.")
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
                        _safe_hover(state, wait_idle=False)
                        _save_final_frame(state, camera_name)
                        break
            elif arrived is False:
                arrival_count = 0
            # arrived is None → VLM gave nothing; counter untouched (R-316)

            if loop_delay > 0:
                time.sleep(loop_delay)
    except KeyboardInterrupt:
        _say("[LOOP] interrupt received — moving to wrap-up")
    except Exception as exc:  # noqa: BLE001
        # SPEC-GAP: §3.9 catches every Exception → exit 0, but R-15 / §2.5 / E-12
        # require connection, VlmConfigError and run-dir failures to exit non-zero.
        # Resolution: setup-phase exceptions (before the instruction is accepted) propagate to
        # cli → exit 1 — the finally chain still runs (E-6). In-flight exceptions are absorbed
        # (traceback to stdout) so the vehicle is landed and exit is 0, as §3.9 describes.
        if not state.airborne_phase:
            raise
        _say(f"[LOOP] unexpected error, ending the episode: {type(exc).__name__}: {exc}")
        traceback.print_exc(file=sys.stdout)
        sys.stdout.flush()
    finally:
        _finalize(state, recording_root)
    return 0


# ---------------------------------------------------------------------------
# §3.10 --test
# ---------------------------------------------------------------------------


def single_shot(args: Any) -> int:
    """One inference on ``frame_airsim_test.jpg`` in cwd; no takeoff/recording/grid (R-80…R-84)."""
    if not os.path.isfile(TEST_IMAGE_PATH):
        _say(f"[TEST] static image {TEST_IMAGE_PATH} is not in the current folder")
        return 1
    image = cv2.imread(TEST_IMAGE_PATH)
    if image is None:
        _say(f"[TEST] could not decode {TEST_IMAGE_PATH}")
        return 1
    height, width = image.shape[:2]
    _say(f"[TEST] running one inference on the {width}x{height} image")

    config_path = getattr(args, "config", None) or DEFAULT_CONFIG_PATH
    _say(f"[CONFIG] config file: {config_path}")
    config = load_mission_config(config_path)
    port = resolve_port(getattr(args, "port", None), config)
    output_root = str(config.get("output_root", DEFAULT_OUTPUT_ROOT))

    run_dir = artifacts.create_run_dir(output_root)
    _say(f"[RUN] output folder: {run_dir}")
    client = _connect(port)
    _, controller = _assemble(config, width, height, run_dir, client)
    response_text, _, _ = controller.run_step(image, TEST_INSTRUCTION, heading_ref_rad=None)
    _say(f"[TEST] result: {response_text}")
    controller.shutdown(land=True)
    return 0
