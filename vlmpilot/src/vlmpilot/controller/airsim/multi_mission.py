# DroneCATS
# Copyright (c) 2026-present NAVER Cloud Corp.
# Apache-2.0

"""multi_mission.py — one multi-drone episode's lifecycle.

Same shape as ``mission.fly``: config → N RPC connections → per-vehicle
camera check → run dir → object assembly → instruction → (start poses) → takeoff →
team loop → finally. Reached from ``cli.main`` only when the config carries a non-empty
``drone_names``; the CLI itself is unchanged.

This module is the source of the multi ``[ARRIVAL]`` lines (with the ``drone=<name>`` tail),
``[TEAM]``, ``[SEARCH]``, ``[CAMERA]`` and ``[RECORD] recording started``. ``[ACTION]`` lines come
from ``team_policy`` (via ``policy.action_from_dict``), ``[MOVEMENT]``/``[YAW]``/``[STEP]``
from ``flight_controller``.
"""

from __future__ import annotations

import math
import os
import sys
import time
import traceback
from collections.abc import Mapping, Sequence
from typing import Any

import airsim

from vlmpilot import artifacts
from vlmpilot.controller.airsim import mission
from vlmpilot.policy import vlm_backends
from vlmpilot.policy.camera import PinholeCamera
from vlmpilot.controller.airsim.fleet import Fleet

# Constants shared with the single path — the same values on purpose.
SEARCH_GIVE_UP_STEPS = mission.SEARCH_GIVE_UP_STEPS
START_DELAY_S = mission.START_DELAY_S
IDLE_WAIT_TIMEOUT_S = mission.IDLE_WAIT_TIMEOUT_S
ARRIVAL_IDLE_TIMEOUT_S = mission.ARRIVAL_IDLE_TIMEOUT_S
ARRIVAL_SETTLE_S = mission.ARRIVAL_SETTLE_S
DEFAULT_END_ON = "all"
CAMERA_PITCH_WARN_DEG = 5.0

TEAM_SCHEMA_NAME = "vlmpilot_team_action"
MOSAIC_COLS = 2  # §4.7-M17 — the final arrival frame uses the same mosaic as decision_*.jpg


def _say(line: str = "") -> None:
    print(line, flush=True)


# ---------------------------------------------------------------------------
# §6.2 per-vehicle camera check
# ---------------------------------------------------------------------------


def probe_cameras(
    clients_by_vehicle: Mapping[str, Any], camera_name: str, drone_names: Sequence[str]
) -> tuple[int, int, bool] | None:
    """Check **every** vehicle's camera. Returns ``(w, h, needs_confirm)``.

    Resolution = the first vehicle that yields a frame. ``needs_confirm`` when any vehicle is
    below the recommended capture size. If no vehicle yields a frame: one warning line and
    ``None`` (the caller stops before takeoff) — never raises. Every vehicle leaves a
    ``[CAMERA] {veh}:`` line.
    """
    width: int | None = None
    height: int | None = None
    needs_confirm = False
    for veh in drone_names:
        client = clients_by_vehicle[veh]
        try:
            info = client.simGetCameraInfo(camera_name, veh)
            try:
                pitch = math.degrees(airsim.to_eularian_angles(info.pose.orientation)[0])
            except Exception:  # noqa: BLE001 — pitch is diagnostic only
                pitch = float("nan")
            try:
                fov = float(getattr(info, "fov", float("nan")))
            except (TypeError, ValueError):
                fov = float("nan")
            responses = client.simGetImages([mission._scene_request(camera_name)], vehicle_name=veh)
            if not responses or int(getattr(responses[0], "width", 0) or 0) <= 0:
                _say(f"[CAMERA] {veh}: got no frame — check the settings")
                continue
            w = int(responses[0].width)
            h = int(responses[0].height)
            _say(f"[CAMERA] {veh}: {w}x{h}  FOV {fov:.0f}°  pitch {pitch:+.1f}°")
            if not math.isnan(pitch) and abs(pitch) > CAMERA_PITCH_WARN_DEG:
                _say(
                    f"[CAMERA] ⚠ {veh} camera is tilted {pitch:+.1f}° — the bench needs it level (0°). "
                    "As is, it only sees the floor"
                )
            if mission.capture_too_small(w, h):
                needs_confirm = True
            if width is None:
                width, height = w, h
        except Exception as exc:  # noqa: BLE001 — L5: skip, but leave the name behind
            _say(f"[CAMERA] {veh}: check failed — {exc}")
    if width is None or height is None:
        _say("[CAMERA] got no frame from any drone — exiting without takeoff")
        return None
    return width, height, needs_confirm


# ---------------------------------------------------------------------------
# §6.3 assembly
# ---------------------------------------------------------------------------


def _assemble(
    config: dict, width: int, height: int, run_dir: str, drone_names: Sequence[str], clients: Sequence[Any]
) -> tuple[Any, Fleet]:
    """PinholeCamera → backend (team schema) → provider gate → TeamPolicy → Fleet (§6.3)."""
    from vlmpilot.policy import team_policy  # written by the team-policy implementer; imported here so single-path imports never depend on it

    camera = PinholeCamera(width, height, float(config.get("fov_horizontal", mission.DEFAULT_FOV_HORIZONTAL_DEG)))
    backend = vlm_backends.build_backend(
        config,
        response_schema=team_policy.team_action_response_schema(len(drone_names)),
        schema_name=TEAM_SCHEMA_NAME,
    )  # VlmConfigError propagates → exit 1
    if not vlm_backends.supports_multi_image(config):  # §6.4-L9
        raise vlm_backends.VlmConfigError(
            f"api_provider={config.get('api_provider')!r} cannot send multiple images, so it cannot be used for multi-drone"
        )
    policy = team_policy.TeamPolicy.from_config(config, camera, backend, drone_names)
    fleet = Fleet(
        config=config,
        drone_names=drone_names,
        image_width=width,
        image_height=height,
        run_dir=run_dir,
        team_policy=policy,
        camera=camera,
        on_decision=lambda record: artifacts.save_team_decision(run_dir, record),
        clients=clients,
    )
    return backend, fleet


def _connect_all(port: int, n: int) -> list[Any]:
    """§6.1 step 4: one ``MultirotorClient`` per vehicle (§5.1-R5); one connection line."""
    clients = []
    for _ in range(n):
        client = airsim.MultirotorClient(port=port)
        client.confirmConnection()
        clients.append(client)
    _say(f"Connected to AirSim (RPC port {port})")
    return clients


# ---------------------------------------------------------------------------
# §6.5–§6.6 flight
# ---------------------------------------------------------------------------


class _TeamState:
    """Mutable flags shared between the loop and the finally chain (initialised before anything can fail — L3)."""

    def __init__(self, drone_names: Sequence[str]) -> None:
        self.drone_names = list(drone_names)
        self.instruction = ""
        self.run_dir: str | None = None
        self.clients: list[Any] = []
        self.backend: Any = None
        self.fleet: Fleet | None = None
        self.recording = False
        self.known_dirs: set[str] = set()
        self.reached = False
        self.declared_once = False
        self.airborne_phase = False
        self.arrived_flags: list[bool] = [False] * len(self.drone_names)


def _start_recording(state: _TeamState, recording_root: str) -> None:
    """R-60/R-61 on the fleet's first client."""
    try:
        state.known_dirs = artifacts.snapshot_recording_dirs(recording_root)
        state.clients[0].startRecording()
        state.recording = True
        _say(f"[RECORD] recording started ({recording_root})")
    except Exception as exc:  # noqa: BLE001
        _say(f"[RECORD] recording start failed (continuing): {exc}")


def _stop_one(state: _TeamState, i: int) -> None:
    """§6.5-L14 step 6: drain + hover **that vehicle only**."""
    try:
        ctrl = state.fleet.controllers[i]
        ctrl.wait_until_idle(timeout=ARRIVAL_IDLE_TIMEOUT_S)
        ctrl.hover()
    except Exception as exc:  # noqa: BLE001 — KeyboardInterrupt passes through
        _say(f"[ARRIVAL] D{i + 1} hover failed (ignored): {exc}")


def _save_final_frame(state: _TeamState, drone_index: int) -> None:
    """§6.5-L14b step 5: settle, capture all vehicles, write ``decision_<ts>_arrived.{jpg,json}``.

    The image is the §4.7-M17 mosaic of every vehicle's frame; the JSON gains
    ``"drone": <declaring vehicle>`` (artifacts.save_arrival_frame, §11.1-C13).
    """
    try:
        time.sleep(ARRIVAL_SETTLE_S)
        frames, _, _ = state.fleet.capture_all()
        tiles = [f for f in frames if f is not None]
        if not tiles or not state.run_dir:
            return
        image = artifacts.build_mosaic(tiles, cols=MOSAIC_COLS)
        jpg_path, _ = artifacts.save_arrival_frame(state.run_dir, image, drone=state.drone_names[drone_index])
        _say(f"[ARRIVAL] final arrival frame saved: {jpg_path}")
    except Exception as exc:  # noqa: BLE001
        _say(f"[ARRIVAL] final frame save failed (ignored): {exc}")


def _finalize(state: _TeamState, recording_root: str, model: str | None = None) -> None:
    """§6.6: stop recording → mp4 → grid (cols=3) → fleet shutdown → (reached) banner.

    Every stage is isolated (L25); contains no ``return`` (L26).
    """
    if state.recording:
        try:
            state.clients[0].stopRecording()
        except (Exception, KeyboardInterrupt):
            pass
        try:
            artifacts.encode_flight_videos(recording_root, state.known_dirs, state.run_dir or "recordings")
        except (Exception, KeyboardInterrupt) as exc:
            _say(f"[RECORD] assembly step aborted: {exc!r}")
    if state.run_dir:
        try:
            # ``model`` overrides the backend's own name: in head/body mode ``state.backend`` is
            # the commander, but the grid shows the frames the BODIES flew, so it must say theirs.
            if model is None and state.backend is not None:
                model = getattr(state.backend, "model", None)
            artifacts.build_recognition_grid(state.run_dir, cols=3, instruction=state.instruction, model=model)  # L22
        except (Exception, KeyboardInterrupt) as exc:
            _say(f"[GRID] build failed (ignored): {exc}")
    if state.fleet is not None:
        try:
            state.fleet.shutdown(land=not (state.reached or state.declared_once))  # L24
        except (Exception, KeyboardInterrupt) as exc:
            _say(f"[FLEET] error during wrap-up (ignored): {exc!r}")
    if state.reached:  # §11.1-C6: never in continue mode (reached is never set there)
        _say()
        _say("=== SEQUENCE FINISHED: target reached ===")


def fly_team(args: Any) -> int:
    """Multi-drone airsim body. Returns the process exit code."""
    config_path = getattr(args, "config", None) or mission.DEFAULT_CONFIG_PATH
    debug = bool(getattr(args, "debug", False))
    _say(f"[CONFIG] config file: {config_path}")
    config = mission.load_mission_config(config_path)
    port = mission.resolve_port(getattr(args, "port", None), config)

    drone_names = [str(n) for n in (config.get("drone_names") or [])]
    if not drone_names:
        raise RuntimeError("multi path but the config has no drone_names (cli.is_multi_config should have filtered this out)")
    n = len(drone_names)
    _say(f"[CONFIG] Drones: {', '.join(drone_names)}")

    camera_name = str(config.get("camera_name", mission.DEFAULT_CAMERA_NAME))
    loop_delay = float(config.get("command_loop_delay", 0) or 0)
    arrival_frames = int(config.get("arrival_frames", mission.DEFAULT_ARRIVAL_FRAMES) or mission.DEFAULT_ARRIVAL_FRAMES)
    recording_root = str(config.get("recording_root", mission.DEFAULT_RECORDING_ROOT))
    output_root = str(config.get("output_root", mission.DEFAULT_OUTPUT_ROOT))
    end_on = str(config.get("end_on", DEFAULT_END_ON) or DEFAULT_END_ON).lower()
    start_poses = config.get("start_poses") or {}
    continue_after_arrival = mission.config_flag(
        config, "continue_after_arrival", mission.ENV_CONTINUE_AFTER_ARRIVAL)
    record_wanted = (bool(config.get("auto_record", False))                    # §3.2-S1
                     and not mission.config_flag(config, "no_record", mission.ENV_NO_RECORD))
    _say(
        f"[CONFIG] camera={camera_name!r} loop_delay={loop_delay:g}s arrival_frames={arrival_frames} "
        f"end_on={end_on} continue_after_arrival={continue_after_arrival} record={record_wanted} "
        f"start_poses={'yes' if start_poses else 'no'}"
    )

    state = _TeamState(drone_names)
    try:
        state.clients = _connect_all(port, n)
        probe = probe_cameras(dict(zip(drone_names, state.clients)), camera_name, drone_names)
        if probe is None:
            return 1
        width, height, needs_confirm = probe
        if needs_confirm and not mission.confirm_low_resolution():
            return 1

        state.run_dir = artifacts.create_run_dir(output_root)  # L1: before the controllers exist
        _say(f"[RUN] output folder: {state.run_dir}")
        state.backend, state.fleet = _assemble(config, width, height, state.run_dir, drone_names, state.clients)

        state.instruction = mission.read_instruction(mission.INSTRUCTION_PROMPT)
        if not state.instruction:
            _say("[INPUT] instruction is empty or input was closed — exiting without takeoff (exit 2)")
            return 2
        state.airborne_phase = True
        _say(f"[START] instruction: {state.instruction!r} — taking off in {START_DELAY_S:g}s")
        time.sleep(START_DELAY_S)
        if start_poses:
            state.fleet.apply_start_poses(start_poses)  # L2: before takeoff
        state.fleet.takeoff_all()
        if start_poses:
            state.fleet.climb_to_start_alt(start_poses)  # L2: after takeoff (R13)
        if record_wanted:
            _start_recording(state, recording_root)
        _say("[LOOP] team closed loop started: wait for queue → simultaneous capture → one inference → per-drone commands")
        _say("[LOOP] a single Ctrl+C runs the wrap-up (grid · video · landing)")

        step = 0
        arrived_streak = [0] * n
        team_search_streak = 0
        while True:
            step += 1
            state.fleet.wait_all_idle(timeout=IDLE_WAIT_TIMEOUT_S, verbose=debug)
            responses, arrived, searching = state.fleet.run_team_step(state.instruction, state.arrived_flags)

            _say()
            _say(f"===== STEP {step} =====")  # L11: no denominator
            for i, name in enumerate(drone_names):
                _say(f"[D{i + 1} {name}] {responses[i]}")  # L12

            ended = False
            for i in range(n):
                if state.arrived_flags[i]:  # L18
                    continue
                if arrived[i] is True:  # L14
                    arrived_streak[i] += 1
                    _say(f"[ARRIVAL] Target-reached signal (VLM) {arrived_streak[i]}/{arrival_frames} drone={drone_names[i]}")
                    if arrived_streak[i] < arrival_frames:
                        continue
                    state.arrived_flags[i] = True
                    _say("[ARRIVAL] FINISHED (arrival declared)")  # §11.1-C4
                    _stop_one(state, i)
                    _say(f"[ARRIVAL] D{i + 1} {drone_names[i]} arrival confirmed")
                    if continue_after_arrival:  # L15: unfreeze immediately
                        state.declared_once = True
                        state.arrived_flags[i] = False
                        arrived_streak[i] = 0
                    else:  # addendum (b): first confirmed arrival ends the episode, like the single path
                        _say("[ARRIVAL] Target reached — hovering and ending sequence.")
                        state.reached = True
                        try:
                            state.fleet.hover_all()
                        except Exception as exc:  # noqa: BLE001
                            _say(f"[ARRIVAL] hover failed (ignored): {exc}")
                        _save_final_frame(state, i)
                        ended = True
                        break
                elif arrived[i] is None:  # L16: no evidence either way
                    pass
                else:  # L17
                    arrived_streak[i] = 0
            if ended:
                break

            k = sum(1 for f in state.arrived_flags if f)
            s = sum(1 for x in searching if x)
            _say(f"[TEAM] arrived {k}/{n} (end_on={end_on}), searching={s}")  # L19

            if all(searching):  # L21: every vehicle still searching
                team_search_streak += 1
                if team_search_streak >= SEARCH_GIVE_UP_STEPS:
                    _say(f"[SEARCH] {SEARCH_GIVE_UP_STEPS} consecutive misses — target not found, ending.")
                    break
            else:
                team_search_streak = 0

            if loop_delay > 0:
                time.sleep(loop_delay)
    except KeyboardInterrupt:
        _say("[LOOP] interrupt received — moving to wrap-up")
    except Exception as exc:  # noqa: BLE001 — same two-phase rule as mission.fly
        if not state.airborne_phase:
            raise
        _say(f"[LOOP] unexpected error, ending the episode: {type(exc).__name__}: {exc}")
        traceback.print_exc(file=sys.stdout)
        sys.stdout.flush()
    finally:
        _finalize(state, recording_root)
    return 0
