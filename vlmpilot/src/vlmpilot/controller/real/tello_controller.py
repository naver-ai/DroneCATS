# DroneCATS
# Copyright (c) 2026-present NAVER Cloud Corp.
# Apache-2.0

"""DJI Tello real-drone backend: runs the vlmpilot closed loop on a real drone.

Clean-room implementation: no code from the old runtime was referenced. The only sources are the DJI Tello SDK
public spec (UDP text commands) and the MIT-licensed djitellopy docs.

Design principle: keep **the same contract** as the AirSim path (FlightController):
  · run_step(frame, instruction) -> (response_text, arrived, searching)
  · All decisions are made by VlmPolicy and all geometry by StepPlanner;
    this module only maps the planner's TURN/MOVE/HOLD to Tello SDK commands.
  · Writes [TIMING] vlm_ms/capture_ms/rpc_ms/think lines, decision records, and steps.jsonl
    in the same format as AirSim. There is no online judging (no sim GT), but
    the whole episode is recorded so it can be evaluated offline later.

Tello command mapping:
  KIND_TURN {angle_deg}            -> rotate_clockwise / rotate_counter_clockwise
  KIND_MOVE {vx,vy,vz(NED),duration} -> converted to displacement (cm), go_xyz_speed
  KIND_HOLD                         -> send_rc_control(0,0,0,0) (hover)

Tello SDK's go takes -500~500cm per axis and is rejected if |value|<20cm on all three axes;
smaller steps are replaced by a hover (keeps the real drone from stopping on an error response).
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import math
import os
import threading
import time
from collections.abc import Mapping
from typing import Any

import cv2

from vlmpilot.policy.action import ACTION_FINISHED, ACTION_ROTATE, ACTION_THINK
from vlmpilot.policy.camera import canonical_depth_scale
from vlmpilot.policy.step_planner import KIND_HOLD, KIND_MOVE, KIND_TURN, StepPlanner

# Same response-wording contract as the AirSim controller; shared by the runner/log parsers.
_RESPONSE_NO_ACTION = "No valid actions identified"
_RESPONSE_ERROR = "Error processing command"
_RESPONSE_NO_POLICY = "No policy attached"

# Tello SDK hard limits (DJI public spec).
_GO_MIN_CM = 20
_GO_MAX_CM = 500
_SPEED_MIN = 10
_SPEED_MAX = 100
_ROTATE_MIN_DEG = 1
_ROTATE_MAX_DEG = 360

DEFAULT_MIN_BATTERY = 15  # %: refuse takeoff below this (prevents real-drone crashes)

# Dummy that djitellopy's BackgroundFrameRead fills in before decoding. If the worker thread
# dies the frame stays here forever, and since it is an ndarray it passes a `frame is None`
# check; the VLM would run on a black image, so treat it as None.
_PLACEHOLDER_HW = (300, 400)


class ResilientFrameRead:
    """Replacement for djitellopy's BackgroundFrameRead that recovers from decode failures by itself.

    djitellopy 2.5.0's update_frame catches only ``av.error.ExitError``. Tello's
    H.264 UDP stream often throws ``InvalidDataError`` when joined before a keyframe,
    which ends the ``container.decode()`` generator, kills the worker thread, and leaves the frame
    stuck on the 300x400 black placeholder forever (real-drone measurement: 0 frames). The mission loop
    got an ndarray, so it assumes all is well and runs the VLM on a black image.

    Re-entering alone was not enough: after InvalidDataError the container soon ends with ExitError
    (measured: ends after 8 frames). So the container is reopened.
    """

    def __init__(self, address: str, *, open_timeout: float = 5.0) -> None:
        self.address = address
        self._open_timeout = float(open_timeout)
        self._lock = threading.Lock()
        self._frame: Any = None
        self.frames_decoded = 0
        self.frames_skipped = 0
        self.reopens = 0
        self.decode_errors = 0
        self._stopped = threading.Event()
        self._worker = threading.Thread(target=self._pump, daemon=True)

    def start(self) -> "ResilientFrameRead":
        self._worker.start()
        return self

    def stop(self) -> None:
        self._stopped.set()

    @property
    def frame(self) -> Any:
        with self._lock:
            return self._frame

    def wait_for_frame(self, timeout: float) -> Any:
        """Wait for the first good frame. None if none arrives."""
        deadline = time.monotonic() + float(timeout)
        while time.monotonic() < deadline:
            frame = self.frame
            if frame is not None:
                return frame
            time.sleep(0.1)
        return self.frame

    def _pump(self) -> None:
        import av  # noqa: PLC0415 — needed only on the real-drone path (tests/CI don't import it)
        import numpy as np  # noqa: PLC0415

        while not self._stopped.is_set():
            try:
                # FFmpeg's udp protocol overrides SO_RCVBUF with its own default (64KB).
                # Tello's IDRs arrive in bursts, so that much truncates the tail.
                container = av.open(
                    self.address,
                    options={
                        "buffer_size": "8388608",
                        "fifo_size": "1000000",
                        "overrun_nonfatal": "1",
                    },
                    timeout=(self._open_timeout, None),
                )
            except Exception:  # noqa: BLE001 — open failures are absorbed by retrying
                self.reopens += 1
                time.sleep(0.3)
                continue
            # Keyframe gate. Joining mid-stream decodes from P frames with no IDR to
            # reference, giving green/grey corrupted images (real-drone measurement). A 12 s observation
            # had IDR 1 · SPS 12 · PPS 11 · P 24, so drop everything until the first keyframe.
            #
            # Tried dropping to demux + packet.decode(), but unless it is the stream's own decoder
            # it only yields empty frames with all-zero YUV (solid green RGB(3,125,3));
            # use container.decode() and reopen from outside when the generator dies.
            keyed = False
            try:
                for frame in container.decode(video=0):
                    if self._stopped.is_set():
                        break
                    if not keyed:
                        if not getattr(frame, "key_frame", False):
                            self.frames_skipped += 1
                            continue
                        keyed = True
                    with self._lock:
                        self._frame = np.array(frame.to_image())
                    self.frames_decoded += 1
            except Exception:  # noqa: BLE001 — InvalidDataError etc. recovered by reopening
                self.decode_errors += 1
            finally:
                try:
                    container.close()
                except Exception:  # noqa: BLE001
                    pass
            self.reopens += 1


def _say(msg: str) -> None:
    print(msg, flush=True)


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


class TelloController:
    """One Tello + VlmPolicy. Follows FlightController's single-drone contract."""

    def __init__(
        self,
        *,
        config: Mapping[str, Any] | None,
        run_dir: str | None,
        policy: Any,
        camera: Any | None = None,
        on_decision: Any = None,
        tello: Any = None,
    ) -> None:
        cfg: Mapping[str, Any] = config if config is not None else {}
        self.run_dir = run_dir
        self.policy = policy
        self.camera = camera
        self.on_decision = on_decision
        self.planner = StepPlanner.from_config(cfg)
        self.min_battery = int(cfg.get("tello_min_battery", DEFAULT_MIN_BATTERY) or 0)
        self.speed_cms = int(_clamp(float(cfg.get("tello_speed_cms", 40) or 40),
                                    _SPEED_MIN, _SPEED_MAX))
        # Same adaptive bands as AirSim (shorter when close); depth is model-estimated only.
        self.adaptive_depth = bool(cfg.get("adaptive_depth", True))
        raw_bands = cfg.get("adaptive_depth_bands", [[2.0, 0.0, True], [5.0, 0.6, False]])
        self.adaptive_depth_bands = sorted((list(b) for b in (raw_bands or [])),
                                           key=lambda b: float(b[0]))
        # djitellopy's frame channel order. 2.5.0 builds frames with `np.array(frame.to_image())`
        # and to_image() is PIL (=RGB), so it comes out RGB (measured). Everything downstream
        # assumes BGR (cv2.imencode/imwrite, AirSim path), so left as-is, images with R/B
        # swapped go to the VLM and decision_*.jpg. Set false for a build that delivers BGR.
        self.frame_is_rgb = bool(cfg.get("tello_frame_rgb", True))
        # Time connect() waits for the first good frame (= after a keyframe). The measured IDR
        # interval while stationary on the ground is ~12 s, so allow comfortably more.
        self.frame_warmup_s = float(cfg.get("tello_frame_warmup_s", 20.0))
        # Video settings (SDK set_video_*): resolution high|low(480p), bitrate 0(auto)~5(Mbps), fps high|middle|low. None = drone default.
        self.video_resolution = cfg.get("tello_video_resolution")
        self.video_bitrate = cfg.get("tello_video_bitrate")
        self.video_fps = cfg.get("tello_video_fps")
        # Interval (s) for keeping the drone awake while inference runs. Tello lands by itself after 15 s
        # without commands, and the mission loop sends the next command only after policy.decide() returns. If that gap
        # exceeds 15 s the drone sets down on the spot; a structural problem unrelated to a slow model
        # (vlm_timeout is 30 s, so the system allows up to 30 s). 0 means don't send.
        self.keepalive_interval_s = float(cfg.get("tello_keepalive_interval_s", 5.0) or 0.0)
        # Target altitude after takeoff (cm). SDK takeoff hovers at about 70~100cm (measured),
        # but hitting person-height targets needs more. 0/None keeps the SDK default height.
        self.takeoff_height_cm = int(cfg.get("tello_takeoff_height_cm", 0) or 0)
        # Fixed downward tilt of the front camera (°). The Tello camera looks down, not straight ahead
        # (measured 12.6° with tools/tello/tello_pitch_facing.py; cross-checked against board attitude 11.9°). The pinhole model (camera.py)
        # assumes optical axis = forward, so (forward, up/down) in camera coordinates is mapped back to body coordinates here.
        # 0 means no correction (previous behavior). Not applicable to the sim, whose camera faces forward.
        self.camera_pitch_deg = float(cfg.get("tello_camera_pitch_deg", 0.0) or 0.0)
        # Receiver-side canonical depth conversion: same convention as the AirSim version (flight_controller).
        # If the real camera's hFOV is wrong, it becomes a constant scale error as-is.
        self.canonical_depth = bool(cfg.get("canonical_depth", False) or False)
        _convert = cfg.get("canonical_depth_convert", None)
        self.canonical_depth_convert = (
            self.canonical_depth if _convert is None else bool(_convert)
        )
        self.depth_scale = canonical_depth_scale(camera, cfg.get("fov_horizontal"))
        if self.canonical_depth_convert:
            if self.depth_scale is None:
                _say("[DEPTH] canonical_depth=on but there is neither a camera nor fov_horizontal, so "
                     "the conversion factor cannot be derived — using canonical values as is")
            else:
                _hfov = (getattr(camera, "fov_horizontal_deg", None)
                         if camera is not None else cfg.get("fov_horizontal"))
                _hfov_s = f"{float(_hfov):.1f}°" if _hfov is not None else "?"
                _say(f"[DEPTH] canonical->metric x{self.depth_scale:.3f} (hFOV {_hfov_s}, "
                     f"prompt={'canonical' if self.canonical_depth else 'metric'}, convert=on)")
        # Filled by the mission loop right after frame capture; capture_ms in [TIMING].
        self.last_frame_ms: float | None = None
        self._step_count = 0

        if tello is None:
            # Lazy import so this module imports without a real drone (tests/CI).
            from djitellopy import Tello  # noqa: PLC0415

            tello = Tello()
        self.tello = tello
        self._frame_reader: Any = None

    # ------------------------------------------------------------------ session

    def connect(self) -> None:
        self.tello.connect()
        battery = self._safe_query("get_battery")
        _say(f"[TELLO] connected — battery={battery}%")
        multi = os.environ.get("VLMPILOT_HEADBODY") == "1" or os.environ.get("VLMPILOT_TELLO_VIDEO_LOW") == "1"
        # Several drones share 2.4GHz in one room (with 4 simultaneous streams 'command'/'up'/'land'
        # failed even after 4 retries). Multi-drone default is 5fps/1Mbps; config's tello_video_* takes precedence if set.
        # Resolution low (480p) only via config, since the mission must scale the camera calibration by the same ratio (_camera_from_config).
        settings = [("set_video_fps", self.video_fps if self.video_fps is not None else ("low" if multi else None)),
                    ("set_video_bitrate", self.video_bitrate if self.video_bitrate is not None else (1 if multi else None)),
                    ("set_video_resolution", self.video_resolution)]
        applied = []
        for name, arg in settings:
            if arg is None:
                continue
            fn = getattr(self.tello, name, None)
            if callable(fn):
                try:
                    fn(int(arg) if name == "set_video_bitrate" else str(arg))
                    applied.append(f"{name[10:]}={arg}")
                except Exception as exc:  # noqa: BLE001 — a failed video setting must not block the flight
                    _say(f"[TELLO] {name}({arg}) failed, ignored: {exc}")
        if applied:
            _say("[TELLO] video " + " · ".join(applied) + (" (multi-drone interference mitigation)" if multi else ""))
        self.tello.streamon()
        self._frame_reader = self._open_frame_reader()

    def _open_frame_reader(self) -> Any:
        """ResilientFrameRead for a real drone, otherwise (FakeTello etc.) the SDK default reader.

        The djitellopy default reader's worker dies on the first InvalidDataError; swap it only on the
        real drone, and let injected fake drones keep the existing contract.
        """
        addr_fn = getattr(self.tello, "get_udp_video_address", None)
        if not callable(addr_fn):
            return self.tello.get_frame_read()
        reader = ResilientFrameRead(addr_fn()).start()
        first = reader.wait_for_frame(self.frame_warmup_s)
        if first is None:
            _say(f"[TELLO] warning — no video frame within {self.frame_warmup_s:g}s "
                 "(check Wi-Fi signal/battery). The mission loop keeps retrying.")
        else:
            _say(f"[TELLO] video ready — {getattr(first, 'shape', None)} "
                 f"({reader.frames_decoded} frames after keyframe, "
                 f"{reader.frames_skipped} pre-keyframe skipped, "
                 f"{reader.decode_errors} decode errors)")
        return reader

    def takeoff(self) -> None:
        battery = self._safe_query("get_battery")
        if isinstance(battery, (int, float)) and battery < self.min_battery:
            # A low-voltage takeoff leads to a forced landing mid-flight (= episode contamination + risk of
            # damaging the drone); cutting it off here is the real-drone version of the guardrail.
            raise RuntimeError(f"battery {battery}% < min {self.min_battery}% — takeoff refused")
        _say("[TELLO] takeoff…")
        self.tello.takeoff()
        _say("[TELLO] takeoff complete")
        self._climb_to_takeoff_height()

    def _climb_to_takeoff_height(self) -> None:
        """Climb to tello_takeoff_height_cm right after takeoff. ToF (distance to floor) first, else barometric altitude.

        SDK `up` only accepts 20~500cm, so values outside are clipped and differences under 20cm are ignored.
        A climb failure is only logged and the flight continues (altitude is not a mission safety condition).
        """
        target = self.takeoff_height_cm
        if target <= 0:
            return
        current = self._safe_query("get_distance_tof")
        if not isinstance(current, (int, float)) or current <= 0:
            current = self._safe_query("get_height")
        if not isinstance(current, (int, float)):
            _say("[TELLO] takeoff height — could not read the current altitude, skipping the climb")
            return
        delta = int(_clamp(target - float(current), 0, 500))
        if delta < 20:
            _say(f"[TELLO] takeoff height — already {current}cm (target {target}cm), skipping the climb")
            return
        _say(f"[TELLO] climbing +{delta}cm → target {target}cm (now {current}cm)")
        try:
            self.tello.move_up(delta)
        except Exception as exc:  # noqa: BLE001 — a failed climb must not break the mission
            _say(f"[TELLO] climb failed, continuing at current height: {exc!r}")
            return
        after = self._safe_query("get_distance_tof")
        _say(f"[TELLO] takeoff height reached — tof={after}cm")

    def land(self) -> None:
        try:
            self.tello.land()
        finally:
            stop = getattr(self._frame_reader, "stop", None)
            if callable(stop):
                try:
                    stop()
                except Exception:  # noqa: BLE001 — shutdown cleanup is best effort
                    pass
            try:
                self.tello.streamoff()
            except Exception:  # noqa: BLE001 — shutdown cleanup is best effort
                pass

    def keepalive(self) -> None:
        """Prevent auto-landing (15 s without commands) while hovering: SDK keepalive if available, else a state query."""
        fn = getattr(self.tello, "send_keepalive", None)
        try:
            if callable(fn):
                fn()
            else:
                self._safe_query("get_battery")
        except Exception as exc:  # noqa: BLE001 — a failed keepalive must not break the flight
            _say(f"[TELLO] keepalive failed, ignoring: {exc!r}")

    def frame(self) -> Any:
        """Current frame (BGR ndarray). None if the stream isn't started.

        djitellopy 2.5.0 delivers RGB (to_image() is PIL). Downstream assumes BGR, so
        flip it once here; can be turned off with config `tello_frame_rgb: false`.
        """
        if self._frame_reader is None:
            return None
        frame = self._frame_reader.frame
        if frame is None:
            return None
        if getattr(frame, "ndim", 0) != 3 or frame.shape[2] != 3:
            return frame
        # djitellopy's black dummy is an ndarray, so `is None` misses it; it must be filtered here
        # so the mission loop doesn't run the VLM on a black image.
        if frame.shape[:2] == _PLACEHOLDER_HW and not frame.any():
            return None
        if not self.frame_is_rgb:
            return frame
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # --------------------------------------------------------------- telemetry

    def telemetry(self) -> dict:
        """State snapshot for offline evaluation: the real-drone counterpart of sim GT.

        Tello has no global position. All that can be kept: battery, altitude (baro/ToF),
        attitude, velocity. Failed items are left None, but the step record is never cut.
        """
        out: dict[str, Any] = {}
        for key, getter in (
            ("battery", "get_battery"),
            ("height_cm", "get_height"),
            ("tof_cm", "get_distance_tof"),
            ("baro_cm", "get_barometer"),
            ("pitch_deg", "get_pitch"),
            ("roll_deg", "get_roll"),
            ("yaw_deg", "get_yaw"),
            ("speed_x", "get_speed_x"),
            ("speed_y", "get_speed_y"),
            ("speed_z", "get_speed_z"),
        ):
            out[key] = self._safe_query(getter)
        return out

    def _safe_query(self, getter: str) -> Any:
        try:
            return getattr(self.tello, getter)()
        except Exception:  # noqa: BLE001 — missing telemetry must not stop the flight
            return None

    # --------------------------------------------------------------------- step

    @contextlib.contextmanager
    def _keepalive_during(self):
        """Send keepalives on a separate thread while the block runs.

        Use only around inference: meanwhile the main thread only waits for the HTTP response and doesn't talk
        to the drone, so only here can another thread use the SDK socket without crossing commands/responses.
        Wrapping apply_action too would overlap move commands with keepalives, so it isn't wrapped.
        """
        if self.keepalive_interval_s <= 0:
            yield
            return
        stop = threading.Event()

        def pump() -> None:
            while not stop.wait(self.keepalive_interval_s):
                self.keepalive()   # keepalive() swallows its own exceptions

        worker = threading.Thread(target=pump, daemon=True)
        worker.start()
        try:
            yield
        finally:
            stop.set()
            worker.join(timeout=1.0)

    def run_step(self, frame: Any, instruction: str) -> tuple[str, bool | None, bool]:
        """Same contract as FlightController.run_step: observe → decide → act."""
        if self.policy is None:
            _say("[CTRL] run_step called without a policy — nothing to decide")
            return (_RESPONSE_NO_POLICY, None, False)
        try:
            t0 = time.monotonic()
            telem = self.telemetry()  # state right before deciding; the step record's reference point
            rpc_ms = round((time.monotonic() - t0) * 1000.0, 1)

            with self._keepalive_during():
                action, record = self.policy.decide(frame, instruction)
            if action is None:
                return (_RESPONSE_NO_ACTION, None, False)

            if record is not None:
                record.payload["timing_ms"] = {
                    "capture_time": self.last_frame_ms,
                    "telemetry_time": rpc_ms,
                }
                # Same [TIMING] format as AirSim, so guardrail/offline analysis tools
                # use one parser regardless of backend. rpc here is telemetry.
                _say(f"[TIMING] vlm_ms={record.payload.get('vlm_ms')} "
                     f"capture_ms={self.last_frame_ms} "
                     f"rpc_ms={rpc_ms} "
                     f"think={1 if record.payload.get('think') else 0}")
            self._emit_decision(record)
            try:
                return self.apply_action(action, telemetry=telem)
            except Exception as exc:  # noqa: BLE001 — rejected steps are recorded too
                _say(f"[CTRL] action failed: {exc!r}")
                self._record_step(action, telem, [], error=repr(exc))
                return (_RESPONSE_ERROR, None, False)
        except Exception as exc:  # noqa: BLE001 — a step failure must not kill the process
            _say(f"[CTRL] step failed: {exc!r}")
            return (_RESPONSE_ERROR, None, False)

    def apply_action(self, action: Any, *, telemetry: dict | None = None
                     ) -> tuple[str, bool | None, bool]:
        """Execute one Action. Same response wording and arrived/searching meaning as the AirSim version."""
        telem = telemetry if telemetry is not None else self.telemetry()
        kind = action.kind

        if kind == ACTION_FINISHED:
            text = f"→ FINISHED (arrival declared): {action.reason}"
            self._hover()
            self._record_step(action, telem, [])
            return (text, True, False)

        if kind == ACTION_THINK:
            text = f"→ THINK (hold & reason next): {action.reason}"
            self._hover()
            self._record_step(action, telem, [])
            return (text, False, False)

        if kind == ACTION_ROTATE:
            angle = float(action.angle_deg)
            self._exec_turn(angle)
            text = f"→ ROTATE {angle:+.1f}° ({action.reason})"
            self._record_step(action, telem, [(KIND_TURN, {"angle_deg": angle})])
            return (text, False, True)

        # GO: depth is model-estimated only (no oracle on the real drone).
        action = self._camera_to_body(action)
        eff = action.depth_m
        # canonical -> metric (non-mutating). k == 1.0 returns the same object.
        k = 1.0
        if self.canonical_depth_convert and self.depth_scale is not None:
            k = float(self.depth_scale)
        if k != 1.0 and eff is not None:
            eff = eff * k
        exec_action = action if k == 1.0 else dataclasses.replace(
            action, forward=action.forward * k, right=action.right * k, down=action.down * k, depth_m=eff
        )
        mult, brake = self._band_lookup(eff) if self.adaptive_depth else (1.0, False)
        text = (f"→ GO R{exec_action.right:+.2f} F{exec_action.forward:+.2f} U{exec_action.up:+.2f} "
                f"depth[model]={eff} speed×{mult:g}" + (" brake" if brake else ""))
        commands = self.planner.plan(exec_action, distance_scale=mult, precise_stop=brake)
        for kind_, params in commands:
            self._exec(kind_, params)
        self._record_step(action, telem, commands,
                          depth_scale=k if self.canonical_depth_convert else None)
        return (text, False, False)

    def _camera_to_body(self, action: Any) -> Any:
        """Camera-frame offset → body frame. If the camera looks down by θ,
        the optical axis is (cosθ forward, −sinθ up) in the body frame, and camera-up is (sinθ forward, cosθ up).
        θ=0 returns the input unchanged. action is treated as immutable; a copy is made (§6-E7).
        """
        th = math.radians(self.camera_pitch_deg)
        if abs(th) < 1e-9:
            return action
        cam_fwd, cam_up = float(action.forward), action.up
        fwd = cam_fwd * math.cos(th) + cam_up * math.sin(th)
        up = -cam_fwd * math.sin(th) + cam_up * math.cos(th)
        return dataclasses.replace(action, forward=fwd, down=0.0 - up)

    def _band_lookup(self, depth_m: float | None) -> tuple[float, bool]:
        """Same rule as AirSim §3.3-R9: (scale, brake) of the first band below depth."""
        if depth_m is None:
            return (1.0, False)
        for band in self.adaptive_depth_bands:
            if depth_m < float(band[0]):
                return (float(band[1]), bool(band[2]) if len(band) > 2 else False)
        return (1.0, False)

    # ---------------------------------------------------------------- commands

    def _exec(self, kind: str, params: dict) -> None:
        if kind == KIND_TURN:
            self._exec_turn(float(params.get("angle_deg", 0.0)))
        elif kind == KIND_MOVE:
            self._exec_move(params)
        else:  # KIND_HOLD
            self._hover()

    def _exec_turn(self, angle_deg: float) -> None:
        # The planner's + is clockwise (NED yaw), same as Tello rotate_clockwise.
        deg = int(round(_clamp(abs(angle_deg), 0, _ROTATE_MAX_DEG)))
        if deg < _ROTATE_MIN_DEG:
            return
        _say(f"[MOVEMENT] rotate {angle_deg:+.1f}°")
        if angle_deg >= 0:
            self.tello.rotate_clockwise(deg)
        else:
            self.tello.rotate_counter_clockwise(deg)

    def _exec_move(self, params: dict) -> None:
        # The planner gives body-frame velocity (NED: vz positive down) → converted to displacement (cm).
        duration = float(params.get("duration", 0.0))
        fwd_cm = int(round(float(params.get("vx", 0.0)) * duration * 100.0))
        left_cm = int(round(-float(params.get("vy", 0.0)) * duration * 100.0))  # NED y=right → go y=left
        up_cm = int(round(-float(params.get("vz", 0.0)) * duration * 100.0))   # NED z=down → go z=up
        fwd_cm = int(_clamp(fwd_cm, -_GO_MAX_CM, _GO_MAX_CM))
        left_cm = int(_clamp(left_cm, -_GO_MAX_CM, _GO_MAX_CM))
        up_cm = int(_clamp(up_cm, -_GO_MAX_CM, _GO_MAX_CM))
        if max(abs(fwd_cm), abs(left_cm), abs(up_cm)) < _GO_MIN_CM:
            # Tiny moves the SDK rejects; hovering is safer than stopping on an error.
            _say(f"[MOVEMENT] go skipped — displacement {fwd_cm}/{left_cm}/{up_cm}cm < {_GO_MIN_CM}cm")
            self._hover()
            return
        _say(f"[MOVEMENT] go fwd={fwd_cm}cm left={left_cm}cm up={up_cm}cm speed={self.speed_cms}")
        self.tello.go_xyz_speed(fwd_cm, left_cm, up_cm, self.speed_cms)

    def _hover(self) -> None:
        try:
            self.tello.send_rc_control(0, 0, 0, 0)
        except Exception as exc:  # noqa: BLE001
            _say(f"[CTRL] hover failed, continuing: {exc!r}")

    # ----------------------------------------------------------------- records

    def _emit_decision(self, record: Any) -> None:
        if self.on_decision is None or record is None:
            return
        try:
            self.on_decision(record)
        except Exception as exc:  # noqa: BLE001
            _say(f"[CTRL] on_decision failed, continuing: {exc!r}")

    def _record_step(self, action: Any, telem: dict, commands: list,
                     error: str | None = None, depth_scale: float | None = None) -> None:
        """One steps.jsonl line: the only trajectory material for offline evaluation.

        There is no sim GT (true distance/pose), so keep everything available: time, action,
        telemetry, executed command. A later evaluator reconstructs the episode from this file + the decision records
        (frames, model responses).
        """
        if self.run_dir is None:
            return
        self._step_count += 1
        row = {
            "t": round(time.time(), 2),
            "step": self._step_count,
            "action": getattr(action, "kind", None),
            "reason": getattr(action, "reason", "")[:120],
            "depth_m": getattr(action, "depth_m", None),
            "commands": [[k, {kk: vv for kk, vv in p.items()
                              if isinstance(vv, (int, float, str, bool))}]
                         for k, p in commands],
            "telemetry": telem,
            **({"depth_scale": depth_scale} if depth_scale is not None else {}),
            **({"error": error} if error else {}),
        }
        try:
            with open(os.path.join(self.run_dir, "steps.jsonl"), "a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        except Exception as exc:  # noqa: BLE001
            _say(f"[CTRL] steps.jsonl write failed, ignoring: {exc!r}")
