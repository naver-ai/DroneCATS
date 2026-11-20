# DroneCATS
# Copyright (c) 2026-present NAVER Cloud Corp.
# Apache-2.0

"""AirSim session, command queue/worker, yaw correction, GT depth, step logging.

``airsim`` is imported lazily (see ``_api``) so unit tests can inject a fake
module via ``flight_controller.airsim = <fake>`` without the real package.

Every AirSim RPC this class issues carries ``vehicle_name``. The
default ``""`` is what AirSim resolves to the first vehicle, so the single-drone path
behaves exactly as before.
"""

from __future__ import annotations

import dataclasses
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
from vlmpilot.policy.camera import canonical_depth_scale
from vlmpilot.policy.step_planner import KIND_HOLD, KIND_MOVE, KIND_TURN, Command, StepPlanner

WORKER_SHUTDOWN_TIMEOUT_S = 15.0  # shutdown(): max wait for the command worker thread

# Resolved on first use. Tests may assign a fake module here before constructing
# a controller; the real ``airsim`` is imported only when this is still None.
# SPEC-GAP: the spec only says "import airsim inside the controller module, guarded";
# the injection point (module attribute ``airsim``) is this implementation's choice.
airsim: Any = None

OnDecision = Callable[[Any], None]
# (duration_s, distance_m) of the command just issued; distance None = not a move (ROTATE).
OnCommand = Callable[[float, float | None], None]

_RESPONSE_NO_ACTION = "No valid actions identified"  # team-authored (2b6db839)
_RESPONSE_ERROR = "Error processing command"  # team-authored (2b6db839)
_RESPONSE_NO_POLICY = "No policy attached"

_DEFAULT_BANDS: list[list[Any]] = [[2.0, 0.0, True], [5.0, 0.6, False]]
_WORKER_POLL_S = 0.1
# agent mode: 4 body threads append concurrently to the same run_dir's steps.jsonl / gt_depth.jsonl.
_RUN_DIR_APPEND_LOCK = threading.Lock()
_IDLE_POLL_S = 0.1

# Altitude clamp defaults
DEFAULT_MIN_ALTITUDE_M = 0.0  # 0 = clamp off
DEFAULT_MAX_VERTICAL_STEP_M = 3.0


def _api() -> Any:
    global airsim
    if airsim is None:
        import airsim as _mod  # real package, only when nothing was injected

        airsim = _mod
    return airsim


def _say(msg: str) -> None:
    print(msg, flush=True)


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


class FlightController:
    """One AirSim multirotor session + asynchronous command execution (§2.3)."""

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
        on_command: OnCommand | None = None,
        client: Any | None = None,
        vehicle_name: str = "",
        own_wind: bool = True,
        # Opens a replacement command connection during teardown; see shutdown(). Default None
        # means "a real MultirotorClient on this controller's port".
        make_command_client: Callable[[], Any] | None = None,
    ) -> None:
        cfg: Mapping[str, Any] = config if config is not None else {}
        self.image_width = int(image_width)
        self.image_height = int(image_height)
        self.run_dir = run_dir
        self.policy = policy
        self.camera = camera
        self.on_decision = on_decision
        self.on_command = on_command
        self.vehicle_name = str(vehicle_name or "")
        self.own_wind = bool(own_wind)
        # The global-pose name cache MUST be initialised here. An
        # AttributeError on first access would be swallowed by the clamp's try/except and
        # silently disable it.
        self._global_name: str | None = None

        # §4.1 / §4.2 constants
        self.planner = StepPlanner.from_config(cfg)
        self.camera_name = str(cfg.get("camera_name", "0"))
        # SPEC-GAP: §3.4-R6 says steps.jsonl carries the config string "as is", §4.2 says
        # compare it lower-cased. We keep the raw string for logging/response_text and use
        # the normalised key only for the model/oracle branch.
        self.depth_source = str(cfg.get("depth_source", "model"))
        self._depth_source_key = self.depth_source.strip().lower()
        self.adaptive_depth = bool(cfg.get("adaptive_depth", True))
        raw_bands = cfg.get("adaptive_depth_bands", _DEFAULT_BANDS)
        self.adaptive_depth_bands: list[list[Any]] = sorted(
            (list(b) for b in (raw_bands or [])), key=lambda b: float(b[0])
        )
        # Receiver-side canonical depth conversion. The single ``canonical_depth`` flag turns on both the
        # prompt wording (policy) and this conversion; if they disagree, either way it is silently 2x off.
        # ``canonical_depth_convert`` is an experimental escape hatch overriding only the controller (None = follow).
        self.canonical_depth = bool(cfg.get("canonical_depth", False) or False)
        _convert = cfg.get("canonical_depth_convert", None)
        self.canonical_depth_convert = (
            self.canonical_depth if _convert is None else bool(_convert)
        )
        # k = fx/W = 1/(2*tan(hFOV/2)). With a camera, use the same f as back-projection.
        self.depth_scale = canonical_depth_scale(camera, cfg.get("fov_horizontal"))
        if self.canonical_depth_convert:
            if self.depth_scale is None:
                # Silently falling back to k=1 would hide a 2x error; always log it.
                _say("[DEPTH] canonical_depth=on but there is neither a camera nor fov_horizontal, so "
                     "the conversion factor cannot be derived — using canonical values as is")
            else:
                _hfov = (getattr(camera, "fov_horizontal_deg", None)
                         if camera is not None else cfg.get("fov_horizontal"))
                _hfov_s = f"{float(_hfov):.1f}°" if _hfov is not None else "?"
                _say(f"[DEPTH] canonical->metric x{self.depth_scale:.3f} (hFOV {_hfov_s}, "
                     f"prompt={'canonical' if self.canonical_depth else 'metric'}, convert=on)")
        self.base_yaw_rate = float(cfg.get("base_yaw_rate", 60.0))
        port = int(cfg.get("airsim_rpc_port", 41451))
        self._rpc_port = port
        # Only used by shutdown() when the command worker cannot be joined — see there.
        self._make_command_client = make_command_client

        self.last_frame_ms: float | None = None   # filled by the mission loop right after capture

        # GT depth capture switch (record-only unless depth_source=oracle).
        # Default off: capture costs 255 ms per step, which eats into the episode cap by that much.
        self.gt_depth_log = bool(cfg.get("gt_depth_log", False))
        if not self.gt_depth_log and self._depth_source_key == "oracle":
            _say("[GT-DEPTH] gt_depth_log=false but depth_source=oracle, so ignoring it and always capturing")
            self.gt_depth_log = True

        # Altitude clamps, applied in _exec_move right before the RPC.
        self.min_altitude_m = float(cfg.get("min_altitude_m", DEFAULT_MIN_ALTITUDE_M) or 0.0)
        self.max_vertical_step_m = float(
            cfg.get("max_vertical_step_m", DEFAULT_MAX_VERTICAL_STEP_M) or 0.0
        )
        # Altitude to settle at after takeoff (relative to the takeoff point, m). 0 keeps AirSim's default takeoff altitude.
        # Comes from the per-episode config (verifier.airsim.run_testset PROFILES: 1.5 m both indoor and outdoor).
        self.start_altitude_m = float(cfg.get("start_altitude_m", 0.0) or 0.0)

        # §3.2-R1 session establishment, in order.
        api = _api()
        if client is None:
            client = api.MultirotorClient(port=port)
        self.client = client
        # Read-only RPCs may be moved to a second connection (see use_sensor_client). By default
        # they share the command client, so every existing mode behaves exactly as before.
        self.sensor_client = client
        self.client.confirmConnection()
        self.client.enableApiControl(True, self.vehicle_name)
        self.client.armDisarm(True, self.vehicle_name)
        if config is not None and self.own_wind:
            # simSetWind is world-global — the fleet sets it once.
            wx = float(cfg.get("wind_x", 0.0) or 0.0)
            wy = float(cfg.get("wind_y", 0.0) or 0.0)
            wz = float(cfg.get("wind_z", 0.0) or 0.0)
            self.client.simSetWind(api.Vector3r(wx, wy, wz))
            _say(f"[CTRL] wind set (NED m/s): x={wx:g} y={wy:g} z={wz:g}")

        self._queue: queue.Queue[Command] = queue.Queue()
        self.running = True
        self._worker = threading.Thread(target=self._worker_loop, name="vlmpilot-cmd-worker", daemon=True)
        self._worker.start()
        self._step_count = 0
        _say(
            f"[CTRL] controller ready — port={port} vehicle={self.vehicle_name!r} camera={self.camera_name!r} "
            f"depth_source={self.depth_source} adaptive={self.adaptive_depth} bands={self.adaptive_depth_bands} "
            f"gt_depth_log={self.gt_depth_log} min_alt={self.min_altitude_m:g} max_vstep={self.max_vertical_step_m:g}"
        )

    # ------------------------------------------------------------------ session

    def takeoff(self) -> None:
        """§3.2-R2."""
        _say(f"[CTRL] takeoff…{self._veh_tag()}")
        z0 = None
        if self.start_altitude_m > 0:
            # Build the altitude from the takeoff point's z. AirSim's takeoff altitude is fixed, not configurable,
            # so the target altitude is set here. If the state query fails, keep the default takeoff altitude.
            try:
                z0 = float(self.client.getMultirotorState(vehicle_name=self.vehicle_name)
                           .kinematics_estimated.position.z_val)
            except Exception:
                z0 = None
        self.takeoff_async().join()
        if z0 is not None:
            self.client.moveToZAsync(z0 - self.start_altitude_m, 1.0, vehicle_name=self.vehicle_name).join()
        _say(f"[CTRL] takeoff complete{self._veh_tag()}")

    def takeoff_async(self) -> Any:
        """Issue takeoff, return the future (the fleet joins all at once)."""
        return self.client.takeoffAsync(vehicle_name=self.vehicle_name)

    def hover(self) -> None:
        """Drain the queue, then hover this vehicle."""
        self.wait_until_idle()
        self.client.hoverAsync(vehicle_name=self.vehicle_name).join()

    def shutdown(self, land: bool = True) -> None:
        """§3.2-R3. Pending queue items are discarded.

        The landing RPCs go through the command client ONLY when the command worker was
        actually joined. An abandoned worker is still sitting inside an RPC on that client, and
        a msgpack-rpc client owns one tornado IOLoop that exactly one thread may drive (see
        :meth:`use_sensor_client`) — so driving it from here raised
        ``RuntimeError('IOLoop is already running')`` and the vehicle never hovered, never
        landed and never released API control. Seen on four of four vehicles in one episode, each
        preceded by the abandon line.
        """
        self.running = False
        abandoned = False
        if self._worker.is_alive():
            # Bounded: a worker stuck inside an AirSim RPC .join() (sim stall) must not block
            # landing forever — signals only reach the main thread, so this is the only backstop
            # before the verifier's watchdog SIGKILLs the process (W5 review, CRITICAL).
            self._worker.join(timeout=WORKER_SHUTDOWN_TIMEOUT_S)
            abandoned = self._worker.is_alive()
            if abandoned:
                _say(f"[CTRL] command worker still busy after {WORKER_SHUTDOWN_TIMEOUT_S:g}s — abandoning it and proceeding to land")
        # ``abandoned`` only knows about the command worker. A body thread abandoned by
        # ``agent_mission._stop_bodies`` can be driving this very connection with no flag to
        # show it, and then the first RPC below raises the same
        # ``RuntimeError('IOLoop is already running')`` — so a wedge discovered there falls back
        # to a connection of this thread's own and reruns the sequence on it, once.
        client = None if abandoned else self.client
        for _ in (1, 2):
            if client is None:
                # A connection of this thread's own, so the other thread's IOLoop is irrelevant.
                client = self._open_teardown_client()
                if client is None:
                    _say(f"[CTRL] could not open a landing connection — this drone will not be landed"
                         f"{self._veh_tag()}")
                    return
            try:
                client.hoverAsync(vehicle_name=self.vehicle_name).join()
                break
            except RuntimeError as exc:
                if client is not self.client or "IOLoop" not in str(exc):
                    raise
                _say(f"[CTRL] another thread holds the existing connection ({exc}) — retrying with a landing connection"
                     f"{self._veh_tag()}")
                client = None
        if land:
            _say(f"[CTRL] landing…{self._veh_tag()}")
            client.landAsync(vehicle_name=self.vehicle_name).join()
            client.armDisarm(False, self.vehicle_name)
            client.enableApiControl(False, self.vehicle_name)
            _say(f"[CTRL] landed, API control released{self._veh_tag()}")
        else:
            _say(f"[CTRL] shutdown without landing — holding hover{self._veh_tag()}")

    def _open_teardown_client(self) -> Any | None:
        """A fresh command connection, opened on the calling thread. None = could not open.

        Never touches ``self.client``: that one belongs to the worker this method exists to work
        around, and stealing it back is the wedge itself.
        """
        try:
            factory = self._make_command_client
            client = factory() if factory is not None else _api().MultirotorClient(port=self._rpc_port)
            client.confirmConnection()
            _say(f"[CTRL] opened a new landing connection (an abandoned worker holds the existing one)"
                 f"{self._veh_tag()}")
            return client
        except Exception as exc:  # noqa: BLE001 — teardown must not raise into the fleet
            _say(f"[CTRL] landing connection failed: {type(exc).__name__}: {exc}{self._veh_tag()}")
            return None

    def _veh_tag(self) -> str:
        return f" ({self.vehicle_name})" if self.vehicle_name else ""

    def use_sensor_client(self, client: Any) -> None:
        """Send this controller's read-only RPCs to ``client`` instead of the command client.

        A msgpack-rpc client owns one tornado IOLoop and may only ever be driven by one thread:
        a second thread calling in while the first is inside ``loop.start()`` raises "IOLoop is
        already running" and leaves the connection permanently wedged (measured in agent
        mode: drone step 3.2s→20s, head stalled, even landing failed). The command client belongs to the
        command worker thread, so a caller that senses from its own thread — a head/body mode
        body — hands in a connection of its own here. Call it from the thread that will use it.
        """
        self.sensor_client = client

    # -------------------------------------------------------------------- queue

    def enqueue(self, command: Command) -> None:
        self._queue.put(command)

    def wait_until_idle(self, timeout: float = 30.0, verbose: bool = False, stop_event: Any = None) -> bool:
        """§3.2-R5: wait until every issued command finished executing (unfinished_tasks == 0).

        ``stop_event`` (head/body mode) makes the wait abandonable: a body parked here while the
        episode is being torn down returns at once instead of holding the queue for the full
        timeout and then issuing more RPCs on a client the fleet is already shutting down. It
        returns False without the timeout warning — nothing timed out, the caller is leaving.
        """
        deadline = time.monotonic() + float(timeout)
        if verbose:
            _say(f"[QUEUE] waiting: unfinished={self._queue.unfinished_tasks} timeout={timeout:g}s")
        while self._queue.unfinished_tasks > 0:
            if stop_event is not None and stop_event.is_set():
                return False
            if time.monotonic() >= deadline:
                _say(f"[QUEUE] warning: {self._queue.unfinished_tasks} command(s) still running after {timeout:g}s — continuing")
                return False
            if verbose:
                _say(f"[QUEUE] … unfinished={self._queue.unfinished_tasks}")
            time.sleep(_IDLE_POLL_S)
        if verbose:
            _say("[QUEUE] idle")
        return True

    def _worker_loop(self) -> None:
        """§3.2-R4."""
        while self.running:
            try:
                item = self._queue.get(timeout=_WORKER_POLL_S)
            except queue.Empty:
                continue
            try:
                if item:
                    self._execute(item)
            except Exception as exc:  # R4-3: never let the flight thread die
                _say(f"[CTRL] worker: command failed and was dropped: {exc!r}")
            finally:
                self._queue.task_done()  # R4-4 / E9

    # ------------------------------------------------------- command execution

    def _corrected_angle(self, angle_deg: float, heading_ref_rad: float | None) -> float:
        """§3.2-R6: re-express an observation-time relative angle against the current heading."""
        if heading_ref_rad is None:
            return angle_deg
        state = self.client.getMultirotorState(vehicle_name=self.vehicle_name)
        yaw_now = _api().to_eularian_angles(state.kinematics_estimated.orientation)[2]
        target_world = math.degrees(heading_ref_rad) + angle_deg
        err = (target_world - math.degrees(yaw_now) + 180.0) % 360.0 - 180.0
        drift = angle_deg - err
        if abs(drift) > 1.0:
            _say(f"[YAW] rotated {drift:+.1f}° since observation → correcting target {angle_deg:+.1f}° to {err:+.1f}°")
        return err

    def _execute(self, command: Command) -> None:
        """§3.2-R7..R10. Unknown kinds are ignored silently (R9.5)."""
        kind, params = command
        try:
            if kind == KIND_TURN:
                self._exec_turn(params)
            elif kind == KIND_MOVE:
                self._exec_move(params)
            elif kind == KIND_HOLD:
                self._exec_hold()
        except Exception as exc:  # R10: isolate this command only
            _say(f"[CTRL] command {kind!r} failed: {exc!r}")
            traceback.print_exc()

    def _exec_turn(self, params: dict) -> None:
        angle = self._corrected_angle(float(params["angle_deg"]), params.get("heading_ref_rad"))
        yaw_rate = float(params["yaw_rate_dps"])  # required key — KeyError surfaces a caller bug
        duration = abs(angle) / yaw_rate
        rate = yaw_rate if angle > 0 else -yaw_rate
        self.client.rotateByYawRateAsync(rate, duration, vehicle_name=self.vehicle_name).join()
        _say(f"[CTRL] turned {angle:+.1f}° at {rate:+g}°/s over {duration:.2f}s")
        if self._queue.empty():
            self.client.hoverAsync(vehicle_name=self.vehicle_name).join()

    def _global_pose_name(self) -> str:
        """Name accepted by ``simGetObjectPose`` for this vehicle.

        Multi passes the vehicle name through; single (``""``) resolves ``listVehicles()[0]``
        once and caches it. The cache field is initialised in ``__init__``.
        """
        if self.vehicle_name:
            return self.vehicle_name
        if self._global_name is None:
            names = list(self.client.listVehicles() or [])
            if not names:
                raise RuntimeError("listVehicles() returned no vehicles")
            self._global_name = str(names[0])
        return self._global_name

    def _clamp_vertical(self, vz: float, duration: float) -> float:
        """① vertical step cap, ② minimum altitude — independent of each other.

        NED: ``z`` grows downward, so a positive ``dz_cmd`` descends. Lookup failures are
        swallowed with one line and the clamp is skipped — safety rails never ground a flight.
        """
        dz_cmd = float(vz) * float(duration)
        # ① vertical step cap — whenever max_vertical_step_m > 0, regardless of min_altitude_m
        if self.max_vertical_step_m > 0 and abs(dz_cmd) > self.max_vertical_step_m:
            capped = math.copysign(self.max_vertical_step_m, dz_cmd)
            _say(
                f"[ALT] vertical move {dz_cmd:+.2f}m → {capped:+.2f}m clamped "
                f"(max_vertical_step_m={self.max_vertical_step_m:g}){self._veh_tag()}"
            )
            dz_cmd = capped
        # ② minimum altitude — only when enabled
        if self.min_altitude_m > 0:
            try:
                cur_z = float(self.client.simGetObjectPose(self._global_pose_name()).position.z_val)
                if math.isnan(cur_z):
                    raise ValueError("simGetObjectPose returned NaN z")
                want_z = cur_z + dz_cmd
                target_z = min(want_z, -self.min_altitude_m)
                if target_z != want_z:
                    _say(
                        f"[ALT] min altitude {self.min_altitude_m:g}m clamp: target z {want_z:+.2f} → {target_z:+.2f} "
                        f"(current z {cur_z:+.2f}){self._veh_tag()}"
                    )
                dz_cmd = target_z - cur_z
            except Exception as exc:  # lookup failure never blocks the move
                _say(f"[ALT] altitude query failed — skipping min altitude clamp: {exc!r}{self._veh_tag()}")
        return dz_cmd / duration if duration else 0.0

    def _exec_move(self, params: dict) -> None:
        api = _api()
        vx, vy, vz, duration = params["vx"], params["vy"], params["vz"], params["duration"]
        yaw_rate = 0.0  # local start value; never read from params (§2.1)
        continuous = not params.get("brake", False)  # §4-N1: continuous mode always on
        _say(
            f"[MOVEMENT] body-frame vel (fwd {vx:.2f}, lat {vy:.2f}, down {vz:+.2f}) m/s · {duration:g}s "
            f"yaw_rate={yaw_rate:g} step={params.get('step_distance_m', 0.0):.2f}m"
            + ("" if continuous else " (brake)")
            + self._veh_tag()
        )
        want_yaw = params.get("yaw_angle") or 0.0
        yaw_angle = self._corrected_angle(float(want_yaw), params.get("heading_ref_rad")) if want_yaw else 0.0
        if yaw_angle:
            limit = self.base_yaw_rate
            yaw_rate = _clamp(yaw_angle / duration, -limit, limit)
            _say(f"[YAW] in-motion turn {yaw_angle:+.1f} deg @ {yaw_rate:+.1f} deg/s")
        vz = self._clamp_vertical(vz, duration)  # right before the RPC
        yaw_mode = api.YawMode(is_rate=True, yaw_or_rate=yaw_rate)
        future = self.client.moveByVelocityBodyFrameAsync(
            vx, vy, vz, duration, api.DrivetrainType.MaxDegreeOfFreedom, yaw_mode, self.vehicle_name
        )
        if not continuous:
            future.join()
            self.client.hoverAsync(vehicle_name=self.vehicle_name).join()

    def _exec_hold(self) -> None:
        _say(f"[MOVEMENT] hold — cancel in-flight command and hover{self._veh_tag()}")
        self.client.cancelLastTask(vehicle_name=self.vehicle_name)
        self.client.hoverAsync(vehicle_name=self.vehicle_name).join()

    # --------------------------------------------------------------------- step

    def run_step(
        self,
        frame: Any,
        instruction: str,
        heading_ref_rad: float | None = None,
    ) -> tuple[str, bool | None, bool]:
        """§3.3: one decision step. Returns ``(response_text, arrived, searching)``.

        Observe → decide → ``apply_action``. Behaviour is unchanged from the
        pre-split implementation.
        """
        if self.policy is None:
            _say(f"[CTRL] run_step called without a policy — nothing to decide{self._veh_tag()}")
            return (_RESPONSE_NO_POLICY, None, False)
        try:
            t0 = time.monotonic()
            depth = self.capture_depth()  # R1-1: before inference
            pose = self.pose_snapshot()  # R1-2: before inference
            t1 = time.monotonic()
            action, record = self.policy.decide(frame, instruction)  # R1-3: [ACTION] printed here
            if action is None:
                return (_RESPONSE_NO_ACTION, None, False)  # R4
            # capture_time=frame capture (sim load), sim_info_time=gt depth+pose RPC,
            # decide=whole decision (vlm_ms is the backend call within it).
            if record is not None:
                record.payload["timing_ms"] = {
                    "capture_time": self.last_frame_ms,
                    "sim_info_time": round((t1 - t0) * 1000.0, 1),
                    "decide": round((time.monotonic() - t1) * 1000.0, 1),
                }
                # For the runner's integrity check (verifier/airsim/guardrail/integrity.py); emits direct
                # measurements only: vlm (model inference) / capture (image capture) / rpc (pose/
                # depth RPC). The rest of the step time (sim_ms = interval − vlm) and the residual
                # (residual_ms = sim − capture − rpc) can't be known by the pilot at this point
                # (before the next frame), so the runner derives them. Drone motion is
                # continuous flight, parallel to the next step, so it shows up in no term (only the waits
                # of brake/hold/hover steps remain in residual). For multi-drone support it carries
                # the same ` drone=<name>` suffix as [ACTION].
                _tail = f" drone={self.vehicle_name}" if self.vehicle_name else ""
                # think=1 means this step's vlm call went out with reasoning on; it is legitimately
                # long, so it must be excluded from slow_vlm stats (median/baseline) to avoid false alarms.
                _say(f"[TIMING] vlm_ms={record.payload.get('vlm_ms')} "
                     f"capture_ms={record.payload['timing_ms']['capture_time']} "
                     f"rpc_ms={record.payload['timing_ms']['sim_info_time']} "
                     f"think={1 if record.payload.get('think') else 0}{_tail}")
            self._emit_decision(record)  # R3.5
            return self.apply_action(action, pose=pose, depth=depth, heading_ref_rad=heading_ref_rad)
        except Exception as exc:  # R2
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
        """§3.3-R5..R8 for one already-decided ``Action``.

        ``pose=None`` → snapshot taken here (R32). Every branch records a steps.jsonl line (R34).
        Raises on unexpected failure — ``run_step`` (single) and ``Fleet`` (multi) isolate it.
        """
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
                self.client.hoverAsync(vehicle_name=self.vehicle_name).join()
            except Exception as exc:  # E8
                _say(f"[CTRL] think: hover failed, continuing: {exc!r}")
            text = f"→ THINK (hold & reason next): {action.reason}"
            self._record_step(action, pose, [], depth_used=None, speed_mult=None)
            return (text, False, False)

        if kind == ACTION_ROTATE:
            angle = float(action.angle_deg)
            self.enqueue(
                (
                    KIND_TURN,
                    {"angle_deg": angle, "yaw_rate_dps": self.base_yaw_rate, "heading_ref_rad": heading_ref_rad},
                )
            )
            # ROTATE doesn't go through the planner, so it has no duration payload; derive it as |angle|/angular rate.
            rate = float(self.base_yaw_rate or 0.0)
            self._note_command(abs(angle) / rate if rate > 0 else 0.0, None)
            text = f"→ ROTATE {angle:+.1f}° ({action.reason})"
            self._record_step(
                action, pose, [(KIND_TURN, {"angle_deg": angle})], depth_used=None, speed_mult=None
            )
            return (text, False, True)

        # R8: everything else is the GO path (no enum re-validation), steps 1..9.
        self._sample_gt_depth(depth, action)
        oracle = self._depth_source_key == "oracle"
        eff = action.gt_depth_m if oracle else action.depth_m
        if oracle and eff is not None:
            if self.camera is not None:
                action.forward, action.right, action.down = self.camera.pixel_to_offset(
                    action.pixel_u, action.pixel_v, eff
                )
            else:
                _say("[GT-DEPTH] warning: depth_source=oracle but no camera injected — keeping model-depth vector")
        # distance_scale + precise_stop MUST reach the planner (adaptive bands).
        # canonical -> metric. oracle's gt_depth_m is already metric, so k=1.
        k = 1.0
        if not oracle and self.canonical_depth_convert and self.depth_scale is not None:
            k = float(self.depth_scale)
        if k != 1.0 and eff is not None:
            eff = eff * k
        # k == 1.0 returns the same object: the off path even preserves object identity.
        # The conversion is non-mutating: the original ``action`` stays canonical for history and decision payload.
        exec_action = action if k == 1.0 else dataclasses.replace(
            action, forward=action.forward * k, right=action.right * k, down=action.down * k, depth_m=eff
        )
        mult, brake = self.band_lookup(eff) if self.adaptive_depth else (1.0, False)
        text = (
            f"→ GO R{exec_action.right:+.2f} F{exec_action.forward:+.2f} U{exec_action.up:+.2f} "
            f"depth[{self.depth_source}]={eff} speed×{mult:g}" + (" brake" if brake else "")
        )
        commands = self.planner.plan(exec_action, distance_scale=mult, precise_stop=brake)
        # A plan with only hold left (R1 no displacement, R6 band stop) reports distance 0.
        mv = next((p for k, p in commands if k == KIND_MOVE), None)
        self._note_command(float(mv.get("duration", 0.0)) if mv else 0.0,
                           float(mv.get("step_distance_m", 0.0)) if mv else 0.0)
        for kind_, params in commands:
            queued = dict(params)
            queued["heading_ref_rad"] = heading_ref_rad
            self.enqueue((kind_, queued))
        self._record_step(action, pose, commands, depth_used=eff, speed_mult=mult,
                          depth_scale=k if self.canonical_depth_convert else None)
        return (text, False, False)

    def _note_command(self, duration_s: float, distance_m: float | None) -> None:
        if self.on_command is not None:
            self.on_command(duration_s, distance_m)

    def _emit_decision(self, record: Any) -> None:
        if self.on_decision is None or record is None:
            return
        try:
            self.on_decision(record)
        except Exception as exc:
            _say(f"[CTRL] on_decision failed, continuing: {exc!r}")

    def band_lookup(self, depth_m: float | None) -> tuple[float, bool]:
        """§3.3-R9: first band whose (sorted) upper bound is strictly greater than depth."""
        if depth_m is None:
            return (1.0, False)
        for band in self.adaptive_depth_bands:
            if depth_m < float(band[0]):
                brake = bool(band[2]) if len(band) > 2 else False
                return (float(band[1]), brake)
        return (1.0, False)

    # ------------------------------------------------------ GT depth / pose / io

    def capture_depth(self) -> np.ndarray | None:
        """§3.4-R1. Skipped entirely (→ None) when ``gt_depth_log`` is off."""
        if not self.gt_depth_log:
            return None
        try:
            api = _api()
            request = api.ImageRequest(self.camera_name, api.ImageType.DepthPlanar, pixels_as_float=True, compress=False)
            responses = self.sensor_client.simGetImages([request], vehicle_name=self.vehicle_name)
            if not responses or responses[0].height == 0:
                return None
            r = responses[0]
            return np.array(r.image_data_float, dtype=np.float32).reshape(r.height, r.width)
        except Exception as exc:
            _say(f"[GT-DEPTH] capture failed, skipping: {exc!r}")
            return None

    # Backwards-compatible private aliases (names from before the multi-drone split).
    _capture_depth = capture_depth

    def _sample_gt_depth(self, depth: np.ndarray | None, action: Any) -> None:
        """§3.4-R2/R3: sample the chosen pixel, fill ``action.gt_depth_m``, append gt_depth.jsonl."""
        if depth is None:
            return
        try:
            h, w = depth.shape
            ix = int(_clamp(int(action.pixel_u / self.image_width * w), 0, w - 1))
            iy = int(_clamp(int(action.pixel_v / self.image_height * h), 0, h - 1))
            gt = float(depth[iy, ix])
            action.gt_depth_m = gt
            _say(f"[GT-DEPTH] point=({action.pixel_u},{action.pixel_v}) model={action.depth_m} sim={gt:.2f}m")
        except Exception as exc:
            _say(f"[GT-DEPTH] sample failed, skipping: {exc!r}")
            return
        if self.run_dir is None:
            return
        try:
            row: dict[str, Any] = {
                "pixel_u": action.pixel_u,
                "pixel_v": action.pixel_v,
                "model_depth_m": action.depth_m,
                "gt_depth_m": gt,
            }
            if self.vehicle_name:
                row["vehicle"] = self.vehicle_name
            with _RUN_DIR_APPEND_LOCK, open(os.path.join(self.run_dir, "gt_depth.jsonl"), "a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        except Exception as exc:  # E4
            _say(f"[GT-DEPTH] gt_depth.jsonl write failed, ignoring: {exc!r}")

    def pose_snapshot(self) -> dict | None:
        """§3.4-R4. Always this vehicle's state."""
        try:
            kin = self.sensor_client.getMultirotorState(vehicle_name=self.vehicle_name).kinematics_estimated
            p = kin.position
            yaw = _api().to_eularian_angles(kin.orientation)[2]
            return {
                "x": round(float(p.x_val), 3),
                "y": round(float(p.y_val), 3),
                "z": round(float(p.z_val), 3),
                "alt_m": round(-float(p.z_val), 3),
                "yaw_deg": round(math.degrees(yaw), 2),
            }
        except Exception as exc:  # E3
            _say(f"[STEP] pose snapshot failed, logging null: {exc!r}")
            return None

    _pose_snapshot = pose_snapshot

    def current_yaw_rad(self) -> float | None:
        """Heading of this vehicle now (R-302 equivalent). ``None`` on failure."""
        try:
            orientation = self.sensor_client.getMultirotorState(vehicle_name=self.vehicle_name).kinematics_estimated.orientation
            return float(_api().to_eularian_angles(orientation)[2])
        except Exception as exc:
            _say(f"[STATE] heading read failed — no yaw correction this step ({exc}){self._veh_tag()}")
            return None

    def _record_step(
        self,
        action: Any,
        pose: dict | None,
        commands: list[Command],
        *,
        depth_used: float | None,
        speed_mult: float | None,
        depth_scale: float | None = None,
    ) -> None:
        """§3.4-R5/R6: one steps.jsonl line per valid-action step (+ ``vehicle`` in multi, R35)."""
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
            }
            # Added only when the flag is on: off rows keep the same key set and order as before.
            if depth_scale is not None:
                row["depth_scale"] = depth_scale
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
                row["vehicle"] = self.vehicle_name  # single path adds nothing
            # In agent mode 4 body threads append to the same run_dir's steps.jsonl; the
            # only remaining unguarded shared file write, so serialize with a lock (single/team modes don't contend).
            with _RUN_DIR_APPEND_LOCK, open(os.path.join(self.run_dir, "steps.jsonl"), "a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        except Exception as exc:  # E4
            _say(f"[STEP] steps.jsonl write failed, ignoring: {exc!r}")
