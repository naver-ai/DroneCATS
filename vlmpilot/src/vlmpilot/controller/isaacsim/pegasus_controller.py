# DroneCATS
# Copyright (c) 2026-present NAVER Cloud Corp.
# Apache-2.0

"""PX4 session, command queue/worker, yaw correction, GT depth, step logging
for the Isaac Sim + Pegasus backend.

Where the rules match the AirSim path, this follows `flight_controller.py` and does
not restate them here.

Same **public surface** as `flight_controller.FlightController`. The only differences:
`link=` takes the place of `client=`, and GT depth comes from a sidecar file written by
A instead of an RPC.

This module never imports `airsim`, directly or indirectly: it depends only on
`step_planner` and `action`, both backend-agnostic. `mavsdk` is imported lazily inside
a function only when no `link` is injected (this module must import without the
`pegasus` extra).
"""

from __future__ import annotations

import json
import math
import os
import queue
import threading
import time
import traceback
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

from vlmpilot.policy.action import ACTION_FINISHED, ACTION_ROTATE, ACTION_THINK
from vlmpilot.policy.step_planner import KIND_HOLD, KIND_MOVE, KIND_TURN, Command, StepPlanner

WORKER_SHUTDOWN_TIMEOUT_S = 15.0  # shutdown(): max wait for the worker thread

OnDecision = Callable[[Any], None]

# Must match flight_controller **verbatim**. Not a parsed contract string, but logs of
# the two backends must diff cleanly.
_RESPONSE_NO_ACTION = "No valid actions identified"  # team-authored (2b6db839)
_RESPONSE_ERROR = "Error processing command"  # team-authored (2b6db839)
_RESPONSE_NO_POLICY = "No policy attached"

_DEFAULT_BANDS: list[list[Any]] = [[2.0, 0.0, True], [5.0, 0.6, False]]
_WORKER_POLL_S = 0.1
_IDLE_POLL_S = 0.1

DEFAULT_MAVSDK_ADDRESS = "udpin://0.0.0.0:14540"
DEFAULT_DEPTH_PATH = "/tmp/vlmpilot_pegasus/latest_depth.npy"
DEFAULT_TAKEOFF_ALTITUDE_M = 2.5
# Landing wait cap (s). The link default is 300 s, which is silently used up when the
# vehicle is stuck on a ceiling or shelf; in practice it ran over 4 minutes at 8.9 m.
# If the runner hits that on every task, measurement stalls, so cap it lower and log the
# wait. The next task starts with a reset (PX4 restart) anyway, so no need to see it end.
DEFAULT_LAND_WAIT_S = 90.0


def _say(msg: str) -> None:
    print(msg, flush=True)


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


class PegasusFlightController:
    """Drop-in for `FlightController`, driven by `PegasusLink`."""

    def __init__(
        self,
        *,
        config: Mapping[str, Any] | None,
        image_width: int,
        image_height: int,
        run_dir: str | None,
        policy: Any,
        camera: Any | None = None,
        on_decision: OnDecision | None = None,
        link: Any | None = None,
        vehicle_name: str = "",
    ) -> None:
        cfg: Mapping[str, Any] = config if config is not None else {}
        self.image_width = int(image_width)
        self.image_height = int(image_height)
        self.run_dir = run_dir
        self.policy = policy
        self.camera = camera
        self.on_decision = on_decision
        # In team runs, records which vehicle a steps.jsonl row belongs to. Empty for the
        # single-vehicle path so rows stay unchanged; same rule as airsim's
        # `FlightController.vehicle_name`.
        self.vehicle_name = str(vehicle_name or "")

        # Control constants live in the planner and config; none are defined here.
        self.planner = StepPlanner.from_config(cfg)
        self.depth_source = str(cfg.get("depth_source", "model"))
        self._depth_source_key = self.depth_source.strip().lower()
        self.adaptive_depth = bool(cfg.get("adaptive_depth", True))
        raw_bands = cfg.get("adaptive_depth_bands", _DEFAULT_BANDS)
        self.adaptive_depth_bands: list[list[Any]] = sorted(
            (list(b) for b in (raw_bands or [])), key=lambda b: float(b[0])
        )
        self.base_yaw_rate = float(cfg.get("base_yaw_rate", 60.0))
        self.continuous_control = bool(cfg.get("continuous_control", True))
        self.takeoff_altitude_m = float(
            cfg.get("takeoff_altitude_m", DEFAULT_TAKEOFF_ALTITUDE_M) or DEFAULT_TAKEOFF_ALTITUDE_M
        )
        self.land_wait_s = float(cfg.get("land_wait_s", DEFAULT_LAND_WAIT_S) or DEFAULT_LAND_WAIT_S)

        # GT depth. capture_depth() returns None if there is no sidecar.
        self.depth_path = str(cfg.get("depth_path", DEFAULT_DEPTH_PATH))
        self.gt_depth_log = bool(cfg.get("gt_depth_log", False))
        if not self.gt_depth_log and self._depth_source_key == "oracle":
            _say("[GT-DEPTH] gt_depth_log=false ignored because depth_source=oracle; always capturing")
            self.gt_depth_log = True
        # If oracle and the first step gets no depth, fall back to model. Sporadic failures
        # later are not a fallback: "A does not write depth" (structural) must be told
        # apart from "this frame failed to read" (transient), and the former shows up on
        # the first step.
        self.depth_source_effective: str | None = None
        self._depth_probed = False

        # Filled by the mission loop right after capture
        self.last_frame_ms: float | None = None
        self.last_frame_age_s: float | None = None
        self.last_frame_t: float | None = None      # frame file write time = join key

        if link is None:
            # mavsdk is only in the pegasus extra; needed only when nothing is injected.
            from vlmpilot.controller.isaacsim.pegasus_link import PegasusLink

            # Omitting `grpc_port` breaks slot isolation on gRPC; see the same spot in
            # `pegasus_mission` for why.
            link = PegasusLink(
                address=str(cfg.get("mavsdk_address", DEFAULT_MAVSDK_ADDRESS)),
                grpc_port=int(cfg.get("grpc_port", 50051)))
            link.connect()
        self.link = link

        # Only set up the queue and worker. takeoff() arms right before takeoff.
        self._queue: queue.Queue[Command] = queue.Queue()
        self.running = True
        self._worker = threading.Thread(
            target=self._worker_loop, name="vlmpilot-pegasus-cmd-worker", daemon=True
        )
        self._worker.start()
        self._step_count = 0
        _say(
            f"[CTRL] controller ready — depth_source={self.depth_source} "
            f"adaptive={self.adaptive_depth} bands={self.adaptive_depth_bands} "
            f"gt_depth_log={self.gt_depth_log} continuous={self.continuous_control}"
        )

    # ------------------------------------------------------------------ session

    def takeoff(self) -> None:
        """Arms **inside here**: PX4 auto-disarms 10 s after arming, and between the
        constructor and takeoff there is instruction input (typed by a human) and a 3 s wait."""
        _say(f"[LINK] arm -> takeoff (target {self.takeoff_altitude_m:g}m)")
        self.link.arm()
        self.link.takeoff()
        _say("[LINK] takeoff done — entering offboard")

    def shutdown(self, land: bool = True) -> None:
        """Pending queue items are dropped.

        Each stage is isolated so `close()` is reached even if an earlier one fails.
        Skipping it leaves `mavsdk_server` holding UDP 14540, and the next run's connect
        "looks successful, then blows up on the next call".
        """
        self.running = False
        if self._worker.is_alive():
            self._worker.join(timeout=WORKER_SHUTDOWN_TIMEOUT_S)
            if self._worker.is_alive():
                _say(
                    f"[CTRL] command worker still busy after {WORKER_SHUTDOWN_TIMEOUT_S:g}s — "
                    "giving up and moving on to landing"
                )
        try:
            self.link.cancel()
            self.link.hover()
        except Exception as exc:  # noqa: BLE001
            _say(f"[LINK] hover failed (ignored): {exc!r}")
        if land:
            alt = self._altitude_m()
            # State the cap up front. This wait once silently took 4 minutes, and from
            # the log alone it was indistinguishable from a hang.
            _say(
                f"[LINK] landing... (altitude {alt if alt is None else f'{alt:.2f}m'}, "
                f"waiting up to {self.land_wait_s:g}s)"
            )
            t0 = time.monotonic()
            for step, fn in (
                ("offboard stop", self.link.stop_offboard),
                ("land", lambda: self.link.land(wait=True, wait_timeout=self.land_wait_s)),
                ("disarm", self.link.disarm),
            ):
                try:
                    fn()
                except Exception as exc:  # noqa: BLE001
                    _say(f"[LINK] {step} failed (ignored): {exc!r}")
            alt2 = self._altitude_m()
            _say(
                f"[LINK] landing stage done — {time.monotonic() - t0:.1f}s, "
                f"altitude {alt2 if alt2 is None else f'{alt2:.2f}m'}"
            )
        else:
            _say("[LINK] exiting without landing — keeping offboard, left hovering")
        try:
            self.link.close()
        except Exception as exc:  # noqa: BLE001 — E11
            _say(f"[LINK] cleanup failed (ignored): {exc!r}")

    def _altitude_m(self) -> float | None:
        """Query failures are ignored: the value only decorates logs and does not affect flight."""
        try:
            return -float(self.link.position_ned().position.down_m)
        except Exception:  # noqa: BLE001
            return None

    # -------------------------------------------------------------------- queue

    def enqueue(self, command: Command) -> None:
        self._queue.put(command)

    def wait_until_idle(self, timeout: float = 30.0, verbose: bool = False) -> bool:
        """Wait until all issued commands have finished, not merely until the queue is empty.

        Uses the unfinished-task count. `queue.empty()` turns true the moment the worker
        dequeues, so the next frame would be captured while a turn is still running.
        """
        deadline = time.monotonic() + float(timeout)
        if verbose:
            _say(f"[QUEUE] waiting: unfinished={self._queue.unfinished_tasks} timeout={timeout:g}s")
        while self._queue.unfinished_tasks > 0:
            if time.monotonic() >= deadline:
                _say(
                    f"[QUEUE] warning: {self._queue.unfinished_tasks} command(s) still running "
                    f"after {timeout:g}s — continuing"
                )
                return False
            if verbose:
                _say(f"[QUEUE] … unfinished={self._queue.unfinished_tasks}")
            time.sleep(_IDLE_POLL_S)
        if verbose:
            _say("[QUEUE] idle")
        return True

    def _worker_loop(self) -> None:
        """The 0.1 s get timeout is what lets the loop notice shutdown."""
        while self.running:
            try:
                item = self._queue.get(timeout=_WORKER_POLL_S)
            except queue.Empty:
                continue
            try:
                if item:
                    self._execute(item)
            except Exception as exc:  # noqa: BLE001 — the flight thread never dies
                _say(f"[CTRL] worker: command failed and was dropped: {exc!r}")
            finally:
                self._queue.task_done()

    # ------------------------------------------------------------- command execution

    def _corrected_angle(self, angle_deg: float, heading_ref_rad: float | None) -> float:
        """Same as the AirSim path, with `link.yaw_rad()` in place of `to_eularian_angles`."""
        if heading_ref_rad is None:
            return angle_deg
        yaw_now = self.link.yaw_rad()
        if yaw_now is None:
            return angle_deg
        target_world = math.degrees(heading_ref_rad) + angle_deg
        err = (target_world - math.degrees(yaw_now) + 180.0) % 360.0 - 180.0
        drift = angle_deg - err
        if abs(drift) > 1.0:
            _say(f"[YAW] rotated {drift:+.1f}° since observation -> target {angle_deg:+.1f}° corrected to {err:+.1f}°")
        return err

    def _execute(self, command: Command) -> None:
        """Dispatch a command. Unknown kinds are silently ignored."""
        kind, params = command
        try:
            if kind == KIND_TURN:
                self._exec_turn(params)
            elif kind == KIND_MOVE:
                self._exec_move(params)
            elif kind == KIND_HOLD:
                self._exec_hold()
        except Exception as exc:  # noqa: BLE001 — give up on this command only
            _say(f"[CTRL] command {kind!r} failed: {exc!r}")
            traceback.print_exc()

    def _exec_turn(self, params: dict) -> None:
        """Rotate in place."""
        angle = self._corrected_angle(float(params["angle_deg"]), params.get("heading_ref_rad"))
        yaw_rate = float(params["yaw_rate_dps"])  # required key; a KeyError exposes a caller bug
        duration = abs(angle) / yaw_rate
        rate = yaw_rate if angle > 0 else -yaw_rate
        self.link.rotate_yaw_rate(rate, duration)
        _say(f"[CTRL] turned {angle:+.1f}° at {rate:+g}°/s over {duration:.2f}s")
        # Hover is decided by **checking for remaining items**. With unfinished_tasks,
        # this running turn itself counts as unfinished and we would never hover.
        if self._queue.empty():
            self.link.hover()

    def _exec_move(self, params: dict) -> None:
        """Body-frame move. No altitude clamp (that is a multi-vehicle-only extension)."""
        vx, vy, vz, duration = params["vx"], params["vy"], params["vz"], params["duration"]
        yaw_rate = 0.0  # local initial value; not read from params
        brake = bool(params.get("brake", False))
        continuous = (not brake) and self.continuous_control
        _say(
            f"[MOVEMENT] body-frame vel (fwd {vx:.2f}, lat {vy:.2f}, down {vz:+.2f}) m/s · {duration:g}s "
            f"yaw_rate={yaw_rate:g} step={params.get('step_distance_m', 0.0):.2f}m"
            + ("" if continuous else " (brake)")
        )
        want_yaw = params.get("yaw_angle") or 0.0
        # Skip correction when want_yaw == 0: correction "returns to the heading at
        # observation time" and would undo an in-place turn that already finished.
        yaw_angle = (
            self._corrected_angle(float(want_yaw), params.get("heading_ref_rad")) if want_yaw else 0.0
        )
        if yaw_angle:
            yaw_rate = _clamp(yaw_angle / duration, -self.base_yaw_rate, self.base_yaw_rate)
            _say(f"[YAW] in-motion turn {yaw_angle:+.1f} deg @ {yaw_rate:+.1f} deg/s")
        # PX4 holds the last setpoint forever, so something must reset it to zero after
        # the duration; nonblocking=True schedules the hold coroutine on the link's loop.
        self.link.move_velocity_body(
            vx, vy, vz, duration, yawspeed=yaw_rate, nonblocking=continuous
        )

    def _exec_hold(self) -> None:
        """Dropping move commands does not stop the vehicle; the running one must be cancelled."""
        _say("[MOVEMENT] hold — cancelling running command and hovering")
        self.link.cancel()
        self.link.hover()

    # ------------------------------------------------------------------ step

    def run_step(
        self,
        frame: Any,
        instruction: str,
        heading_ref_rad: float | None = None,
    ) -> tuple[str, bool | None, bool]:
        """Same order as the AirSim path: (1) and (2) happen before inference."""
        if self.policy is None:
            _say("[CTRL] run_step called without a policy — nothing to decide")
            return (_RESPONSE_NO_POLICY, None, False)
        try:
            t0 = time.monotonic()
            depth = self.capture_depth()  # (1) before inference
            pose = self.pose_snapshot()  # (2) before inference
            t1 = time.monotonic()
            action, record = self.policy.decide(frame, instruction)  # (3) [ACTION] is printed here
            if action is None:
                return (_RESPONSE_NO_ACTION, None, False)
            if record is not None:
                record.payload["timing_ms"] = {
                    "capture_time": self.last_frame_ms,
                    "sim_info_time": round((t1 - t0) * 1000.0, 1),
                    "decide": round((time.monotonic() - t1) * 1000.0, 1),
                }
                _say(
                    f"[TIMING] vlm_ms={record.payload.get('vlm_ms')} "
                    f"capture_ms={record.payload['timing_ms']['capture_time']} "
                    f"rpc_ms={record.payload['timing_ms']['sim_info_time']} "
                    f"think={1 if record.payload.get('think') else 0}"
                )
            self._emit_decision(record)  # (4)
            return self.apply_action(action, pose=pose, depth=depth, heading_ref_rad=heading_ref_rad)
        except Exception as exc:  # noqa: BLE001
            _say(f"[CTRL] step failed: {exc!r}")
            return (_RESPONSE_ERROR, None, False)

    def apply_action(
        self,
        action: Any,
        *,
        pose: dict | None = None,
        depth: np.ndarray | None = None,
        heading_ref_rad: float | None = None,
    ) -> tuple[str, bool | None, bool]:
        """Same as the AirSim path; only the `think` hover becomes a link call."""
        if pose is None:
            pose = self.pose_snapshot()

        kind = action.kind
        if kind == ACTION_FINISHED:
            text = f"→ FINISHED (arrival declared): {action.reason}"
            self._record_step(action, pose, [], depth_used=None, speed_mult=None)
            return (text, True, False)

        if kind == ACTION_THINK:
            try:
                self.wait_until_idle(timeout=10)
                self.link.hover()
            except Exception as exc:  # noqa: BLE001
                _say(f"[CTRL] think: hover failed, continuing: {exc!r}")
            text = f"→ THINK (hold & reason next): {action.reason}"
            self._record_step(action, pose, [], depth_used=None, speed_mult=None)
            return (text, False, False)

        if kind == ACTION_ROTATE:
            angle = float(action.angle_deg)
            self.enqueue(
                (
                    KIND_TURN,
                    {
                        "angle_deg": angle,
                        "yaw_rate_dps": self.base_yaw_rate,
                        "heading_ref_rad": heading_ref_rad,
                    },
                )
            )
            text = f"→ ROTATE {angle:+.1f}° ({action.reason})"
            self._record_step(
                action, pose, [(KIND_TURN, {"angle_deg": angle})], depth_used=None, speed_mult=None
            )
            return (text, False, True)

        # Everything else takes the GO path (the enum is not re-validated).
        self._sample_gt_depth(depth, action)
        oracle = self._depth_source_key == "oracle"
        eff = action.gt_depth_m if oracle else action.depth_m
        if oracle and eff is not None:
            if self.camera is not None:
                # Direction comes from the pixel and stays; **only the magnitude** becomes the true distance.
                action.forward, action.right, action.down = self.camera.pixel_to_offset(
                    action.pixel_u, action.pixel_v, eff
                )
            else:
                _say("[GT-DEPTH] warning: depth_source=oracle but no camera — keeping the model depth vector")
        mult, brake = self.band_lookup(eff) if self.adaptive_depth else (1.0, False)
        text = (
            f"→ GO R{action.right:+.2f} F{action.forward:+.2f} U{action.up:+.2f} "
            f"depth[{self.depth_source}]={eff} speed×{mult:g}" + (" brake" if brake else "")
        )
        commands = self.planner.plan(action, distance_scale=mult, precise_stop=brake)
        for kind_, params in commands:
            queued = dict(params)
            queued["heading_ref_rad"] = heading_ref_rad
            self.enqueue((kind_, queued))
        self._record_step(action, pose, commands, depth_used=eff, speed_mult=mult)
        return (text, False, False)

    def _emit_decision(self, record: Any) -> None:
        """File logging never blocks flight."""
        if self.on_decision is None or record is None:
            return
        try:
            self.on_decision(record)
        except Exception as exc:  # noqa: BLE001
            _say(f"[CTRL] on_decision failed, continuing: {exc!r}")

    def band_lookup(self, depth_m: float | None) -> tuple[float, bool]:
        """First band whose upper bound is **strictly greater** than depth."""
        if depth_m is None:
            return (1.0, False)
        for band in self.adaptive_depth_bands:
            if depth_m < float(band[0]):
                brake = bool(band[2]) if len(band) > 2 else False
                return (float(band[1]), brake)
        return (1.0, False)

    # ------------------------------------------------------- GT depth · pose

    def capture_depth(self) -> np.ndarray | None:
        """Read GT depth from the sidecar file that A replaces atomically (not an RPC).

        Time alignment with RGB is **assumed**: A writes both from the same render, so
        they match, and B has no way to verify it.
        """
        if not self.gt_depth_log:
            return None
        try:
            if not os.path.exists(self.depth_path):
                self._note_depth_missing("sidecar file not found")
                return None
            depth = np.load(self.depth_path)
            arr = np.asarray(depth, dtype=np.float32)
            if arr.ndim != 2:
                self._note_depth_missing(f"not 2-D (shape={arr.shape})")
                return None
            if arr.shape != (self.image_height, self.image_width):
                self._note_depth_missing(
                    f"resolution differs from RGB (depth={arr.shape}, "
                    f"rgb={(self.image_height, self.image_width)})"
                )
                return None
            self._depth_probed = True
            return arr
        except Exception as exc:  # noqa: BLE001
            self._note_depth_missing(f"read failed: {exc!r}")
            return None

    def _note_depth_missing(self, why: str) -> None:
        """If the first step gets no depth, fall back to model. Later misses are not a fallback."""
        if self._depth_probed:
            _say(f"[GT-DEPTH] skipping depth for this step — {why}")
            return
        self._depth_probed = True
        if self._depth_source_key == "oracle":
            self.depth_source_effective = "model"
            self._depth_source_key = "model"
            _say(
                f"[GT-DEPTH] depth_source=oracle but no depth available ({why}) "
                "-> falling back to model and continuing"
            )
        else:
            _say(f"[GT-DEPTH] skipping depth logging — {why}")

    def _sample_gt_depth(self, depth: np.ndarray | None, action: Any) -> None:
        """Sample the chosen pixel and append it to gt_depth.jsonl."""
        if depth is None:
            return
        try:
            h, w = depth.shape
            ix = int(_clamp(int(action.pixel_u / self.image_width * w), 0, w - 1))
            iy = int(_clamp(int(action.pixel_v / self.image_height * h), 0, h - 1))
            gt = float(depth[iy, ix])
            # 40% of sidecar values are non-finite: sky and beyond render range come as
            # `inf` (measured). Used as a distance, back-projection yields inf, the planner
            # computes `inf * 0.0 = nan`, and **a NaN velocity setpoint reaches PX4**
            # (confirmed). Non-finite means "nothing at that pixel", so treat it as no GT;
            # the existing `eff is None` path already covers that (skip reprojection and
            # keep flying the model's vector).
            if not math.isfinite(gt):
                _say(
                    f"[GT-DEPTH] point=({action.pixel_u},{action.pixel_v}) sim={gt} "
                    "— non-finite, treating as no GT (continuing with model depth)"
                )
                return
            action.gt_depth_m = gt
            _say(f"[GT-DEPTH] point=({action.pixel_u},{action.pixel_v}) model={action.depth_m} sim={gt:.2f}m")
        except Exception as exc:  # noqa: BLE001
            _say(f"[GT-DEPTH] sampling failed, skipping: {exc!r}")
            return
        if self.run_dir is None:
            return
        try:
            row = {
                "pixel_u": action.pixel_u,
                "pixel_v": action.pixel_v,
                "model_depth_m": action.depth_m,
                "gt_depth_m": gt,
            }
            with open(os.path.join(self.run_dir, "gt_depth.jsonl"), "a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        except Exception as exc:  # noqa: BLE001
            _say(f"[GT-DEPTH] failed to write gt_depth.jsonl, ignoring: {exc!r}")

    def pose_snapshot(self) -> dict | None:
        """Pose snapshot. `z` is raw NED (down +), `alt_m` is `-z`.

        The origin is **that run's spawn position** (PX4 local NED).
        """
        try:
            pos = self.link.position_ned()
            north = float(pos.position.north_m)
            east = float(pos.position.east_m)
            down = float(pos.position.down_m)
            yaw_deg = float(self.link.attitude_euler().yaw_deg)
            return {
                "x": round(north, 3),
                "y": round(east, 3),
                "z": round(down, 3),
                "alt_m": round(-down, 3),
                "yaw_deg": round(yaw_deg, 2),
            }
        except Exception as exc:  # noqa: BLE001
            _say(f"[STEP] pose snapshot failed, recording null: {exc!r}")
            return None

    def current_yaw_rad(self) -> float | None:
        """Heading read by the mission loop at the start of a step. `None` on failure (that step runs uncorrected)."""
        try:
            return float(self.link.yaw_rad())
        except Exception as exc:  # noqa: BLE001
            _say(f"[STATE] heading read failed — this step runs without yaw correction ({exc})")
            return None

    def sim_time_s(self) -> float | None:
        """`sim_time_s` for step records. `None` if the link does not know it."""
        try:
            getter = getattr(self.link, "sim_time_s", None)
            if getter is None:
                return None
            value = getter()
            return None if value is None else float(value)
        except Exception:  # noqa: BLE001 — logging only, give up quietly
            return None

    def _record_step(
        self,
        action: Any,
        pose: dict | None,
        commands: list[Command],
        *,
        depth_used: float | None,
        speed_mult: float | None,
    ) -> None:
        """Same row as the AirSim path, plus pegasus-specific fields."""
        self._step_count += 1
        if self.run_dir is None:
            return
        try:
            row: dict[str, Any] = {
                "step": self._step_count,
                "t": time.time(),
                "kind": action.kind,
                "pose": pose,
                "pixel_u": action.pixel_u,
                "pixel_v": action.pixel_v,
                "angle_deg": action.angle_deg,
                "depth_model_m": action.depth_m,
                "depth_gt_m": action.gt_depth_m,
                "depth_source": self.depth_source,
                "depth_used_m": depth_used,
                "speed_mult": speed_mult,
                "speed_mult_on": self.adaptive_depth,
                "reason": action.reason,
                # Whether A is alive, and how long capture took
                "frame_age_s": self.last_frame_age_s,
                # Joins use this value **first**; a key built from `t − frame_age_s`
                # includes the time it took to build the row
                "frame_t": self.last_frame_t,
                "frame_ms": self.last_frame_ms,
                "sim_time_s": self.sim_time_s(),
            }
            for kind, params in commands:
                if kind == KIND_MOVE:
                    duration = params["duration"]
                    row["move"] = {
                        "vx": round(params["vx"], 3),
                        "vy": round(params["vy"], 3),
                        "vz": round(params["vz"], 3),
                        "duration": round(duration, 2) if duration else duration,
                        "step_distance_m": params["step_distance_m"],
                        "brake": bool(params.get("brake", False)),
                        "yaw_angle": params.get("yaw_angle", 0.0),
                    }
                elif kind == KIND_TURN:
                    row["rotate_in_place_deg"] = params["angle_deg"]
                elif kind == KIND_HOLD:
                    row["move"] = {"step_distance_m": 0.0, "brake": True}
            if self.vehicle_name:
                # Team runs write one row per vehicle per step. Without the vehicle name
                # those rows cannot be told apart afterwards (a real flight once left 6
                # anonymous rows for 3 steps). The single-vehicle path adds nothing.
                row["vehicle"] = self.vehicle_name
            with open(os.path.join(self.run_dir, "steps.jsonl"), "a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        except Exception as exc:  # noqa: BLE001
            _say(f"[STEP] failed to write steps.jsonl, ignoring: {exc!r}")
