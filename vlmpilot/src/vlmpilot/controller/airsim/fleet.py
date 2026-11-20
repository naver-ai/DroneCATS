# DroneCATS
# Copyright (c) 2026-present NAVER Cloud Corp.
# Apache-2.0

"""fleet.py — N ``FlightController`` s driven as one team.

One step: capture every vehicle's (RGB, depth, yaw) **at the same instant** on N threads
(§5.5), ask the team policy once (§4), then ``apply_action`` per vehicle (§5.6). The fleet
owns no run folder (§5.7-R36) and never prints a line that could satisfy a runner regex
(§8.3).

Everything vehicle-specific goes through ``FlightController`` so the single and multi
paths execute the same code (§1.1).
"""

from __future__ import annotations

import functools
import math
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from vlmpilot.controller.airsim import flight_controller as _fc
from vlmpilot.controller.airsim import mission
from vlmpilot.controller.airsim import vehicle_frame
from vlmpilot.controller.airsim.flight_controller import FlightController

# §5.3-R12 verification polling
START_POSE_VERIFY_TRIES = 15
START_POSE_VERIFY_INTERVAL_S = 0.2
START_POSE_POS_TOL_M = 1.0
START_POSE_YAW_TOL_DEG = 3.0
CLIMB_VELOCITY_MPS = 2.0  # §5.3-R13 moveToZAsync speed

# Responses for vehicles that execute nothing this step (team-authored wording).
RESPONSE_FROZEN = "(already arrived — holding)"
RESPONSE_NO_ACTION = "(no action — holding)"
RESPONSE_CAPTURE_FAILED = "frame capture failed"

OnTeamDecision = Callable[[Any], None]


def _say(msg: str) -> None:
    print(msg, flush=True)


def _api() -> Any:
    return _fc._api()


def _yaw_err_deg(yaw_deg: float, want_deg: float) -> float:
    return abs((yaw_deg - want_deg + 180.0) % 360.0 - 180.0)


def grab_scene_bgr(client: Any, camera_name: str, vehicle_name: str) -> np.ndarray | None:
    """One uncompressed Scene frame for ``vehicle_name`` as BGR uint8, or None.

    Takes the client explicitly: the fleet's own capture uses the vehicle's client, while the
    head/body mode's commander must grab through a client no body thread touches (§5.1-R5).
    """
    api = _api()
    request = api.ImageRequest(camera_name, api.ImageType.Scene, False, False)
    responses = client.simGetImages([request], vehicle_name=vehicle_name)
    if not responses:
        return None
    first = responses[0]
    height, width = int(first.height), int(first.width)
    if height == 0 or width == 0:
        return None
    buf = np.frombuffer(first.image_data_uint8, dtype=np.uint8)
    return buf.reshape(height, width, 3)  # AirSim's raw Scene buffer is already BGR


class Fleet:
    """Team-level session, observation and one-step execution over N controllers."""

    def __init__(
        self,
        *,
        config: Mapping[str, Any] | None,
        drone_names: Sequence[str],
        image_width: int,
        image_height: int,
        run_dir: str | None,
        team_policy: Any,
        camera: Any | None = None,
        on_decision: OnTeamDecision | None = None,
        on_command: Callable[..., None] | None = None,
        clients: Sequence[Any] | None = None,
    ) -> None:
        cfg: Mapping[str, Any] = config if config is not None else {}
        self.config = cfg
        self.drone_names: list[str] = [str(n) for n in drone_names]
        if not self.drone_names:
            raise ValueError("Fleet needs at least one drone name")
        self.n_drones = len(self.drone_names)
        self.image_width = int(image_width)
        self.image_height = int(image_height)
        self.run_dir = run_dir
        self.team_policy = team_policy
        self.camera = camera
        self.on_decision = on_decision
        self.camera_name = str(cfg.get("camera_name", "0"))
        self.applied_start_poses: dict | None = None

        port = int(cfg.get("airsim_rpc_port", 41451))
        if clients is None:
            # §5.1-R5: one RPC connection per vehicle (sequential calls on one client are 3× slower).
            api = _api()
            clients = []
            for _ in self.drone_names:
                c = api.MultirotorClient(port=port)
                c.confirmConnection()
                clients.append(c)
        clients = list(clients)
        if len(clients) != self.n_drones:
            raise ValueError(f"clients ({len(clients)}) must match drone_names ({self.n_drones})")
        self.clients: list[Any] = clients

        # Who records command length / travel distance in the history: the team policy if present; in agent mode,
        # the per-drone policies (injected). Otherwise nobody does. Signature: (i, duration_s, distance_m).
        note_command = on_command if on_command is not None else (
            None if team_policy is None else team_policy.note_command)
        # §5.1-R3: no per-vehicle policy — the team policy decides. §5.1-R4: wind set once.
        self.controllers: list[FlightController] = [
            FlightController(
                config=config,
                image_width=self.image_width,
                image_height=self.image_height,
                run_dir=run_dir,
                policy=None,
                camera=camera,
                on_decision=None,
                on_command=(None if note_command is None else functools.partial(note_command, i)),
                client=clients[i],
                vehicle_name=name,
                own_wind=(i == 0),
            )
            for i, name in enumerate(self.drone_names)
        ]
        _say(f"[FLEET] {self.n_drones} controllers ready: {', '.join(self.drone_names)}")

    # ------------------------------------------------------------------ session

    def takeoff_all(self) -> None:
        """§5.4-R14: issue every takeoff first, then join in order."""
        futures: list[tuple[str, Any]] = []
        for name, ctrl in zip(self.drone_names, self.controllers):
            try:
                futures.append((name, ctrl.takeoff_async()))
            except Exception as exc:
                _say(f"[FLEET] {name} takeoff issue failed (ignored): {exc!r}")
        for name, fut in futures:
            try:
                fut.join()
            except Exception as exc:
                _say(f"[FLEET] {name} takeoff join failed (ignored): {exc!r}")
        _say("All drones airborne.")

    def hover_all(self) -> None:
        """§5.4-R15."""
        for name, ctrl in zip(self.drone_names, self.controllers):
            try:
                ctrl.client.hoverAsync(vehicle_name=name).join()
            except Exception as exc:
                _say(f"[FLEET] {name} hover failed (ignored): {exc!r}")

    def shutdown(self, land: bool = True) -> None:
        """§5.4-R16: one vehicle's failure never stops the others."""
        for name, ctrl in zip(self.drone_names, self.controllers):
            try:
                ctrl.shutdown(land=land)
            except Exception as exc:
                _say(f"[FLEET] {name} shutdown failed (ignored): {exc!r}")

    def wait_all_idle(self, timeout: float = 30.0, verbose: bool = False) -> None:
        """§5.4-R17."""
        for ctrl in self.controllers:
            ctrl.wait_until_idle(timeout=timeout, verbose=verbose)

    # -------------------------------------------------------------- start poses

    def _index(self, vehicle: str) -> int:
        return self.drone_names.index(vehicle)

    def apply_start_poses(self, spec: Mapping[str, Mapping[str, Any]]) -> None:
        """§5.3-R10..R12: global start poses → relative frame → simSetVehiclePose → verify."""
        if not spec:
            return
        api = _api()
        client0 = self.clients[0]
        try:
            offsets = vehicle_frame.measure_offsets(client0, self.drone_names)
            _say(f"[START] drone spawn offsets: {vehicle_frame.offset_report(offsets)}")
        except Exception as exc:
            offsets = {}
            _say(f"[START] spawn offset measurement failed — continuing without correction: {exc}")

        wanted: dict[str, tuple[float, float, float]] = {}
        for veh, entry in spec.items():
            if veh not in self.drone_names:  # R11-3: untouched
                continue
            client = self.clients[self._index(veh)]
            try:
                cur = client.simGetVehiclePose(vehicle_name=veh)
                gx = float(entry.get("x", cur.position.x_val))
                gy = float(entry.get("y", cur.position.y_val))
                lx, ly = vehicle_frame.to_local(offsets, veh, gx, gy)
                z = float(entry["z"]) if entry.get("z") is not None else float(cur.position.z_val)  # R13: as given
                yaw_deg = float(entry.get("yaw_deg", 0.0) or 0.0)
                _say(f"[START] {veh} → global({gx:.2f},{gy:.2f}) = local({lx:.2f},{ly:.2f}) z={z:.2f} yaw={yaw_deg:.1f}°")
                pose = api.Pose(api.Vector3r(lx, ly, z), api.to_quaternion(0.0, 0.0, math.radians(yaw_deg)))
                client.simSetVehiclePose(pose, True, veh)
                wanted[veh] = (gx, gy, yaw_deg)
            except Exception as exc:
                _say(f"[START] {veh} start pose apply failed (ignored): {exc}")
        self.applied_start_poses = self._verify_start_poses(wanted)

    def _verify_start_poses(self, wanted: Mapping[str, tuple[float, float, float]]) -> dict:
        """§5.3-R12: poll the **global** pose until it settles, record per-vehicle result."""
        api = _api()
        result: dict[str, dict[str, Any]] = {}
        for veh, (want_x, want_y, want_yaw) in wanted.items():
            client = self.clients[self._index(veh)]
            rec: dict[str, Any] = {
                "x": None, "y": None, "yaw_deg": None,
                "want_x": want_x, "want_y": want_y, "want_yaw_deg": want_yaw,
                "err_pos_m": None, "err_yaw_deg": None, "ok": False,
            }
            for attempt in range(START_POSE_VERIFY_TRIES):
                try:
                    pose = client.simGetObjectPose(veh)
                    x, y = float(pose.position.x_val), float(pose.position.y_val)
                    yaw = math.degrees(api.to_eularian_angles(pose.orientation)[2])
                    err_pos = math.hypot(x - want_x, y - want_y)
                    err_yaw = _yaw_err_deg(yaw, want_yaw)
                    rec.update(x=x, y=y, yaw_deg=yaw, err_pos_m=err_pos, err_yaw_deg=err_yaw)
                    if err_pos <= START_POSE_POS_TOL_M and err_yaw <= START_POSE_YAW_TOL_DEG:
                        rec["ok"] = True
                        break
                except Exception as exc:
                    _say(f"[START] {veh} placement check failed (retrying): {exc}")
                if attempt < START_POSE_VERIFY_TRIES - 1:
                    time.sleep(START_POSE_VERIFY_INTERVAL_S)
            if not rec["ok"]:
                d = rec["err_pos_m"] if rec["err_pos_m"] is not None else float("nan")
                a = rec["err_yaw_deg"] if rec["err_yaw_deg"] is not None else float("nan")
                _say(f"[START] warning — {veh} placement mismatch: position off by {d:.2f} m, heading off by {a:.1f}°")
            result[veh] = rec
        ok = sum(1 for r in result.values() if r["ok"])
        bad = [v for v, r in result.items() if not r["ok"]]
        _say(f"[START] placement verification: {ok}/{len(result)} match" + (f" (mismatch: {', '.join(bad)})" if bad else ""))
        return result

    def climb_to_start_alt(self, spec: Mapping[str, Mapping[str, Any]]) -> None:
        """§5.3-R13: after takeoff, raise only the vehicles whose spec names ``z``."""
        if not spec:
            return
        futures: list[tuple[str, Any]] = []
        for veh, entry in spec.items():
            if veh not in self.drone_names or entry.get("z") is None:
                continue
            client = self.clients[self._index(veh)]
            try:
                futures.append((veh, client.moveToZAsync(float(entry["z"]), CLIMB_VELOCITY_MPS, vehicle_name=veh)))
            except Exception as exc:
                _say(f"[START] {veh} start altitude command failed (ignored): {exc}")
        if not futures:
            return
        for veh, fut in futures:
            try:
                fut.join()
            except Exception as exc:
                _say(f"[START] {veh} start altitude join failed (ignored): {exc}")
        parts = []
        for veh, _ in futures:
            try:
                _, _, z = vehicle_frame.global_of(self.clients[self._index(veh)], veh)
                parts.append(f"{veh} {z:.2f}")
            except Exception:
                parts.append(f"{veh} ?")
        _say(f"[START] start altitude applied — {', '.join(parts)}")

    # ---------------------------------------------------------------- observe

    def _grab_rgb(self, i: int) -> np.ndarray | None:
        """One uncompressed Scene frame for vehicle ``i`` as BGR uint8, or None."""
        return grab_scene_bgr(self.clients[i], self.camera_name, self.drone_names[i])

    def capture_all(self) -> tuple[list, list, list]:
        """§5.5-R18..R24: (frames, depths, yaws), each of length n_drones, captured in parallel.

        Slots are pre-allocated and written **by index** — never appended — so slot ``i``
        is always vehicle ``i`` no matter which thread finishes first (R20).
        """
        n = self.n_drones
        frames: list[np.ndarray | None] = [None] * n
        depths: list[np.ndarray | None] = [None] * n
        yaws: list[float | None] = [None] * n

        def _one(i: int) -> None:
            veh = self.drone_names[i]
            ctrl = self.controllers[i]
            try:
                frames[i] = self._grab_rgb(i)
            except Exception as exc:  # R22: isolate per vehicle, per quantity
                _say(f"[MULTI] {veh} RGB capture failed (ignored): {exc}")
            try:
                depths[i] = ctrl.capture_depth()  # R24: honours gt_depth_log, own vehicle_name
            except Exception as exc:
                _say(f"[MULTI] {veh} depth capture failed (ignored): {exc}")
            try:
                yaws[i] = ctrl.current_yaw_rad()  # R23: own heading reference
            except Exception as exc:
                _say(f"[MULTI] {veh} yaw query failed (ignored): {exc}")

        threads = [threading.Thread(target=_one, args=(i,), name=f"vlmpilot-capture-{i}", daemon=True) for i in range(n)]
        for t in threads:
            t.start()
        for t in threads:  # R21: join every thread before anything is read
            t.join()
        # The moment every drone's view is fixed; the consumer (runner) samples per-drone poses here.
        # Same marker as single mission.fly. Without it the runner can't tell capture time and SR is all 0.
        _say(mission.FRAME_MARKER)
        return frames, depths, yaws

    # --------------------------------------------------------------- one step

    def run_team_step(
        self, instruction: str, frozen: Sequence[bool]
    ) -> tuple[list[str], list[bool | None], list[bool]]:
        """§5.6: capture → one VLM call → per-vehicle apply → deferred decision record."""
        n = self.n_drones
        _t0 = time.monotonic()
        frames, depths, yaws = self.capture_all()
        _capture_ms = round((time.monotonic() - _t0) * 1000.0, 1)

        if all(f is None for f in frames):  # R25
            _say("[MULTI] frame capture failed for all drones — skipping this step")
            return ([RESPONSE_CAPTURE_FAILED] * n, [None] * n, [False] * n)
        for i in range(n):  # R26: keep image order == drone order
            if frames[i] is None:
                _say(f"[MULTI] DRONE {i + 1} no frame — using a blank placeholder frame")
                frames[i] = np.zeros((self.image_height, self.image_width, 3), np.uint8)

        _t1 = time.monotonic()
        actions, record = self.team_policy.decide_team(frames, instruction)  # [ACTION] lines here
        _vlm_ms = round((time.monotonic() - _t1) * 1000.0, 1)
        # For the runner's integrity check (verifier/airsim/guardrail/integrity.py). A team step makes one VLM
        # call, so [TIMING] is also one line per step (no drone suffix), and the runner attaches this
        # value to every drone event of that step. capture is the wall clock of the all-drone capture+depth+
        # yaw bundle (per-part breakdown later); rpc is included in it, so it is set
        # to 0. think marking doesn't exist in the multi team policy yet, fixed at 0.
        _say(f"[TIMING] vlm_ms={_vlm_ms} capture_ms={_capture_ms} rpc_ms=0.0 think=0")
        actions = list(actions or [])

        responses: list[str] = [""] * n
        arrived: list[bool | None] = [None] * n
        searching: list[bool] = [False] * n
        for i in range(n):
            if i < len(frozen) and frozen[i]:  # R28
                responses[i], arrived[i], searching[i] = RESPONSE_FROZEN, True, False
                continue
            action = actions[i] if i < len(actions) else None  # R27
            if action is None:
                responses[i], arrived[i], searching[i] = RESPONSE_NO_ACTION, None, False
                continue
            try:
                responses[i], arrived[i], searching[i] = self.controllers[i].apply_action(
                    action, depth=depths[i], heading_ref_rad=yaws[i]
                )  # R30/R32: pose snapshot happens inside, per vehicle, right before execution
            except Exception as exc:  # R29
                _say(f"[MULTI] DRONE {i + 1} execution failed: {exc}")
                responses[i], arrived[i], searching[i] = f"execution failed: {exc}", None, False

        if record is not None and self.on_decision is not None:  # M16: after execution
            try:
                self.on_decision(record)
            except Exception as exc:
                _say(f"[MULTI] decision write failed (ignored): {exc}")
        return responses, arrived, searching
