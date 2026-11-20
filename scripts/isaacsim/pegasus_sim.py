#!/usr/bin/env python
# Based on examples/1_px4_single_vehicle.py from https://github.com/PegasusSimulator/PegasusSimulator
# (BSD-3-Clause, Copyright (c) 2023, Marcelo Jacinto).
"""Process A: Isaac Sim + Pegasus + PX4 SITL.

Loads the scene, spawns the vehicle(s) and attaches PX4 SITL. It does no control: the runner
(process B) connects to PX4 directly over MAVSDK. This is the "sim half" of a two-process split
that keeps VLM inference from stalling the physics.

Use the wrapper (it carries the GPU pinning arguments):

    bash scripts/isaacsim/pegasus_sim.sh

To run directly, always pin a single GPU. Otherwise Isaac Sim tries to grab every GPU and
collides with anything else running on the same machine:

    $ISAACSIM_PYTHON scripts/isaacsim/pegasus_sim.py \\
        --/renderer/multiGpu/enabled=false --/renderer/activeGpu=3 --/physics/cudaDevice=3

All ports are shifted by `vehicle_id`. With vehicle_id=0, MAVSDK is on 14540.
This does not overlap the AirSim port range (41451-41599 / 42451-42599).

Based on the Pegasus example `1_px4_single_vehicle.py`, with three differences:
  - uses `isaacsim.core.api.world` (the example's `omni.isaac.core.world` is the old path;
    this matches what Pegasus uses internally)
  - headless by default; `--gui` enables the streaming viewport
  - prints what process B needs (vehicle prim path, ports) to stdout
"""

import argparse
import collections
import sys
import time

# ── Isaac Sim boot ─────────────────────────────────────────────────────────
# SimulationApp must be created right after import. Extensions load here, so importing
# other isaacsim/omni modules first crashes.
_parser = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
_parser.add_argument("--env", default="Curved Gridroom",
                     help="Pegasus SIMULATION_ENVIRONMENTS key. "
                          "Heavy scenes (Warehouse family) hang for minutes without logs "
                          "on first load while shaders compile")
_parser.add_argument("--slot-wait", type=float, default=30.0, metavar="S",
                     help="Max seconds to **wait for release** when the slot is taken (default 30). "
                          "If the previous A was just stopped, Isaac may still be shutting down "
                          "and holding the slot; refusing immediately would be confusing. "
                          "If another simulator really holds it, refuse after this time")
_parser.add_argument("--slot", type=int, default=None, metavar="N",
                     help="Slot number. Separates **ports and file paths together** so several "
                          "simulators on one pod don't collide (one simulator per slot/port). "
                          "Without slots they **silently overwrite** each "
                          "other's `latest.jpg`. Slot 0 uses the default paths/ports. "
                          "Explicitly given arguments override slot values")
_parser.add_argument("--vehicle-id", type=int, default=0,
                     help="PX4 instance ID. All ports shift by this amount. "
                          "⚠️ IDs >= 10 share offboard port 14549. "
                          "With --slot it comes from the slot's allocation")
_parser.add_argument("--altitude", type=float, default=0.07,
                     help="Spawn altitude (m). Leaves the vehicle landed on the ground")
_parser.add_argument("--gui", action="store_true",
                     help="Open a local window. **Requires DISPLAY**; without one it falls "
                          "back to headless. For remote viewing use --livestream")
_parser.add_argument("--livestream", nargs="?", const="", default=None,
                     metavar="ADDR",
                     help="Enable WebRTC streaming. View with the Isaac Sim WebRTC Streaming "
                          "Client. If ADDR is given it is advertised as the public endpoint "
                          "(e.g. --livestream <public-ip>)")
_parser.add_argument("--no-render", action="store_true",
                     help="Run physics only and skip rendering entirely (same as --render-hz 0). "
                          "For control checks that need no camera")
_parser.add_argument("--render-hz", type=float, default=5.0,
                     help="Renders per second (default 5). Physics runs at 250 Hz and only "
                          "rendering uses this rate; the camera doesn't need 60 fps, so this "
                          "recovers RTF. Measured: 5 Hz gives RTF≈1.0, render-every-step 0.5")
_parser.add_argument("--physics-device", choices=["cpu", "cuda"], default=None,
                     help="Physics device. Prefer the Pegasus default (cpu): "
                          "measured RTF with cuda dropped to 0.42. A single quadrotor can't "
                          "amortize GPU-physics kernel launch overhead, and the GPU may be "
                          "shared with other workloads")
_parser.add_argument("--physics-hz", type=float, default=None,
                     help="Physics step rate (Hz). Pegasus default is 250")
_parser.add_argument("--rtf-every", type=float, default=10.0,
                     help="Print RTF (real-time factor) every N seconds. 0 disables")
_parser.add_argument("--info-file", default="/tmp/vlmpilot_pegasus/sim_info.json",
                     help="Write sim conditions (scene, camera, physics, RTF) here. Process B "
                          "copies them into the run record so each run notes its conditions")
_parser.add_argument("--reset-file", default="/tmp/vlmpilot_pegasus/reset.request",
                     help="Vehicle reset request file. When it appears the vehicle returns to "
                          "its spawn pose; counterpart of AirSim's scripts/airsim/reset_drone.sh. "
                          "Send requests with scripts/isaacsim/pegasus_reset.py")
_parser.add_argument("--no-camera", action="store_true",
                     help="Don't attach the vehicle camera. For control checks only")
_parser.add_argument("--frame-path", default="/tmp/vlmpilot_pegasus/latest.jpg",
                     help="Path to atomically replace with each frame. Must match frame_path "
                          "in the runner config")
_parser.add_argument("--frame-hz", type=float, default=5.0,
                     help="Frames saved per second. Exceeding the render rate is pointless; "
                          "it just re-saves the same image")
_parser.add_argument("--camera-width", type=int, default=1920,
                     help="AirSim's default 256x144 is far too small for 2D grounding. "
                          "Don't skimp here")
_parser.add_argument("--camera-height", type=int, default=1080)
_parser.add_argument("--camera-fov", type=float, default=90.0,
                     help="Horizontal FOV (deg). Must match fov_horizontal in the runner "
                          "config; the 2D->3D back-projection uses it")
_parser.add_argument("--vehicles", type=int, default=1, metavar="N",
                     help="Number of vehicles. Pegasus starts a PX4 instance per vehicle_id and "
                          "separates ports as `baseport + id`. Vehicle 0 keeps the default "
                          "output paths; from vehicle 1 a suffix is added "
                          "(latest.jpg -> latest_1.jpg). ⚠️ RTF drops further per vehicle")
_parser.add_argument("--wind", default="0,0,0", metavar="X,Y,Z",
                     help="Wind (ENU m/s: x east, y north, z up). Enters Pegasus's linear drag "
                          "as relative velocity. ⚠️ Not numerically equivalent to AirSim's "
                          "simSetWind; cross-sim absolute comparison is invalid, only "
                          "robustness vs. zero wind within the same sim")
_parser.add_argument("--depth", action="store_true",
                     help="Also write a GT depth sidecar. "
                          "The runner's depth_source=oracle reads it to reproject with the "
                          "true distance of the pixel the VLM pointed at, separating "
                          "'pointing' from 'distance estimation'. **Off by default**: it adds "
                          "render load and RTF is already below 1.0 (AirSim's gt_depth_log is off for the same reason)")
_parser.add_argument("--depth-path", default="/tmp/vlmpilot_pegasus/latest_depth.npy",
                     help="Depth sidecar path. Must match depth_path in the runner config. "
                          "float32 2D .npy at the RGB resolution")
_parser.add_argument("--record", default=None, metavar="DIR",
                     help="Drop rendered frames as PNG into this folder. The runner "
                          "assembles its own run window into mp4 in `finally`. **Off by default**: "
                          "1920x1080 PNG at 5 Hz is several MB/s (measured on AirSim: "
                          "915 runs = 930 GB)")
_parser.add_argument("--record-keep", type=int, default=6000, metavar="N",
                     help="Max frames kept in the record folder; oldest are deleted first. "
                          "6000 frames at 5 Hz is about 20 min. Unbounded growth fills the disk")
_parser.add_argument("--camera-offset", default="0.0,0.0,0.0",
                     help="Camera position offset x,y,z (m) in the vehicle body frame")
_parser.add_argument("--camera-pitch", type=float, default=0.0,
                     help="Camera tilt down (deg, positive = down). 0 is level forward. "
                          "⚠️ Tilting changes the 2D->3D assumption in action_projector.py")
# Kit consumes --/... arguments itself, so pass unknown arguments through
_args, _unknown = _parser.parse_known_args()

from isaacsim import SimulationApp  # noqa: E402

# Streaming boots headless and then enables the extension (no window is created).
simulation_app = SimulationApp({"headless": not _args.gui or _args.livestream is not None})

if _args.livestream is not None:
    # `SimulationApp` has no livestream setting key; enable the extension after boot.
    # (--gui won't do it: that is a local window, falling back to headless without DISPLAY)
    import carb.settings  # noqa: E402
    from isaacsim.core.utils.extensions import enable_extension  # noqa: E402

    if _args.livestream:
        carb.settings.get_settings().set(
            "/app/livestream/publicEndpointAddress", _args.livestream)
    ok = enable_extension("omni.kit.livestream.webrtc")
    print(f"[STREAM] WebRTC extension {'enabled' if ok else 'failed'}"
          + (f", public address {_args.livestream}" if _args.livestream else ""))
    print("[STREAM] Connect with the Isaac Sim WebRTC Streaming Client (default port 49100).")
    print("[STREAM] ⚠️ Reconnecting after a dropped session does not work; fully quit the "
          "client and reopen it. A black screen is usually this, not the network.")
    simulation_app.update()

# ── Actual script starts here ──────────────────────────────────────────────
import carb  # noqa: E402
import omni.timeline  # noqa: E402
from isaacsim.core.api.world import World  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

from pegasus.simulator.params import ROBOTS, SIMULATION_ENVIRONMENTS  # noqa: E402
from pegasus.simulator.logic.backends.px4_mavlink_backend import (  # noqa: E402
    PX4MavlinkBackend,
    PX4MavlinkBackendConfig,
)
from pegasus.simulator.logic.vehicles.multirotor import (  # noqa: E402
    Multirotor,
    MultirotorConfig,
)
from pegasus.simulator.logic.dynamics import LinearDrag  # noqa: E402
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface  # noqa: E402
from pegasus.simulator.logic.vehicle_manager import VehicleManager  # noqa: E402

import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402

import numpy as np  # noqa: E402

# Pure-computation modules. `pegasus_wind` is A-only and lives in scripts/;
# `pegasus_layout` is shared by A and B so it lives in the package.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                "vlmpilot", "src"))
import pegasus_wind  # noqa: E402
from vlmpilot.controller.isaacsim import pegasus_layout  # noqa: E402
from PIL import Image  # noqa: E402
from isaacsim.core.utils.prims import get_prim_at_path  # noqa: E402
from isaacsim.core.utils.rotations import euler_angles_to_quat  # noqa: E402
from isaacsim.sensors.camera import Camera  # noqa: E402
from pxr import UsdPhysics  # noqa: E402

# ── Slot resolution ────────────────────────────────────────────────────────
# **Explicit arguments win.** Only values equal to the argparse default are derived
# from the slot; never overwrite a path the user passed.
_SLOT_ENTRY = None
if _args.slot is not None:
    from vlmpilot.controller.isaacsim import pegasus_slots  # noqa: E402

    # The previous A may still be shutting down; only that case is waited for (measured:
    # Isaac takes 10+ s to exit, so an immediate restart is rightly refused by the guard).
    # If it is still alive after the timeout it really belongs to another simulator, so refuse.
    _t0 = time.time()
    while True:
        try:
            _SLOT_ENTRY = pegasus_slots.attach(_args.slot, _args.vehicles,
                                               note=f"env={_args.env}")
            break
        except ValueError as _exc:
            if time.time() - _t0 >= _args.slot_wait:
                print(f"[SLOT] ❌ {_exc}", flush=True)
                raise
            if _t0 == _t0 and time.time() - _t0 < 1.5:
                print(f"[SLOT] slot {_args.slot} is taken; waiting up to "
                      f"{_args.slot_wait:.0f}s for release", flush=True)
            time.sleep(2.0)
    _res = pegasus_slots.resolve(_args.slot, _args.vehicles)
    if _args.vehicle_id == _parser.get_default("vehicle_id"):
        _args.vehicle_id = _res["vehicle_ids"][0]
    for _opt, _key in (("info_file", "info_file"), ("reset_file", "reset_file"),
                       ("frame_path", "frame_path"), ("depth_path", "depth_path")):
        if getattr(_args, _opt) == _parser.get_default(_opt):
            setattr(_args, _opt, _res["paths"][_key])
    if _args.record == _parser.get_default("record"):
        pass            # recording defaults to off (None); the slot doesn't force it on
    # Children (PX4, mavsdk_server) inherit the tag. A's own /proc/environ is fixed at
    # exec time and won't change; A's ownership is determined by the PID in the registry.
    os.environ[pegasus_layout.SLOT_ENV] = str(int(_args.slot))
    print(f"[SLOT] {_args.slot} · vehicle_id {_args.vehicle_id} · "
          f"owner {_SLOT_ENTRY.get('owner')!r} · root {_res['paths']['root']}",
          flush=True)
    if pegasus_layout.slot_of_pid(os.getpid()) is None:
        print(f"[SLOT] ⚠️ this process has no {pegasus_layout.SLOT_ENV} tag; "
              f"exporting it in the wrapper lets cleanup tools identify A too "
              f"(scripts/isaacsim/pegasus_sim.sh does)", flush=True)

# Multi-vehicle names, paths and layout live as pure functions in `pegasus_layout`;
# this file can't be imported (it boots Isaac Sim), so it can't be tested directly.
VEHICLE_PRIM = pegasus_layout.VEHICLE_PRIM
VEHICLE_SPACING_M = pegasus_layout.VEHICLE_SPACING_M
vehicle_prim_for = pegasus_layout.vehicle_prim_for
per_vehicle_path = pegasus_layout.per_vehicle_path


def find_body_prim(vehicle_prim):
    """Find the prim that **actually moves**, to attach the camera to.

    The vehicle root (`/World/quadrotor`) is a container made by `define_prim(..., "Xform")`.
    The RigidBody link below it is what moves in flight; a camera on the root stays put
    while the vehicle flies away, with no error and a frozen image.

    Uses the first descendant with RigidBodyAPI. The structure is logged so that if the
    vehicle asset changes, it is obvious what was picked.
    """
    root = get_prim_at_path(vehicle_prim)
    if not root.IsValid():
        raise RuntimeError(f"vehicle prim not found: {vehicle_prim}")

    rigid, tree = None, []
    for prim in Usd_iter(root):
        path = str(prim.GetPath())
        has_rb = bool(prim.HasAPI(UsdPhysics.RigidBodyAPI))
        depth = path.count("/") - vehicle_prim.count("/")
        if depth <= 2:
            tree.append(f"[CAM]   {'  ' * depth}{path.rpartition('/')[-1]}"
                        f"{'  ← RigidBody' if has_rb else ''}")
        if has_rb and rigid is None:
            rigid = path

    print("[CAM] vehicle prim structure:")
    for line in tree[:20]:
        print(line)

    if rigid is None:
        print(f"[CAM] ⚠️ no RigidBody found; attaching to root ({vehicle_prim}). "
              "Make sure the camera moves with the vehicle")
        return vehicle_prim
    return rigid


def Usd_iter(prim):
    """Iterate a prim and its descendants."""
    yield prim
    for child in prim.GetChildren():
        yield from Usd_iter(child)


class WindDrag(LinearDrag):
    """`LinearDrag` with wind applied as relative velocity.

        f = -diag(c) · (v_body - R_wb · v_wind_enu)

    **With zero wind it yields exactly the same value as `LinearDrag`**: the `else` branch
    below is literally the parent's formula. Leaving the existing dynamics unchanged with
    wind off is the regression guard, pinned by tests on the pure functions in `pegasus_wind`.

    The math lives in `scripts/isaacsim/pegasus_wind.py`; this file can't be imported (it boots
    Isaac Sim), so it can't be tested directly.
    """

    def __init__(self, drag_coefficients, wind_enu=(0.0, 0.0, 0.0)):
        super().__init__(drag_coefficients)
        self._coeff_vec = np.asarray(drag_coefficients, dtype=float)
        self._wind_enu = np.asarray(wind_enu, dtype=float)

    def update(self, state, dt):
        self._drag_force = pegasus_wind.drag_force(
            self._coeff_vec, state.linear_body_velocity, state.attitude, self._wind_enu
        )
        return self._drag_force


class VehicleRig:
    """Per-vehicle camera, frame, depth, recording and spawn state.

    Before multi-vehicle support these were all single fields on `PegasusSimApp`. With N
    vehicles we must know which vehicle a frame belongs to, so they are grouped here.

    **Behavior with N=1 must be unchanged**: the single-vehicle path is validated in real
    flights, so path names and logs stay as they were. `tag` being an empty string
    for vehicle 0 is what ensures that.
    """

    def __init__(self, index, vehicle_id, prim, phys_hz, spawn_pos, spawn_quat_wxyz):
        self.index = int(index)
        self.vehicle_id = int(vehicle_id)
        self.prim = prim
        self.phys_hz = float(phys_hz)
        self.spawn_pos = list(spawn_pos)
        self.spawn_quat_wxyz = list(spawn_quat_wxyz)
        # vehicle 0 has no suffix; single-vehicle logs must stay unchanged
        self.tag = "" if self.index == 0 else f"[{self.index}]"

        self.cam = None
        self.frame_path = None
        self.frame_tmp = None
        self.steps_per_frame = 0
        self.depth_path = None
        self.depth_tmp = None
        self.record_dir = None
        self.record_keep = 0
        self._recorded = collections.deque()
        self._frames_written = 0
        self._frame_warned = False
        self._write_warned = False
        self._depth_warned = False
        self._record_warned = False
        # Measured FOV. Differs from the request: a 90° request gave 91.7°.
        # The runner's back-projection needs the measured value, so keep it per vehicle.
        self.hfov_deg = None
        self.vfov_deg = None

    def info(self):
        """This vehicle's conditions for `sim_info`."""
        return {
            "index": self.index,
            "vehicle_id": self.vehicle_id,
            "prim": self.prim,
            "spawn": {"pos": list(self.spawn_pos), "quat_wxyz": list(self.spawn_quat_wxyz)},
            "frame_path": self.frame_path,
            "depth_path": self.depth_path,
            "hfov_deg": self.hfov_deg,
            "vfov_deg": self.vfov_deg,
            "px4_sim_tcp": 4560 + self.vehicle_id,
            "mavsdk_udp": 14540 + self.vehicle_id,
        }

    def setup_camera(self, frame_path, frame_hz, width, height, fov_deg,
                      offset, pitch_deg, depth=False, depth_path=None,
                      record_root=None, record_keep=6000):
        """Attach a first-person camera to the vehicle and prepare frame saving.

        Orientation: an Isaac camera with identity orientation looks along the parent's **+X**.
        The body frame's +X is forward, so as an identity child of the body link it is
        already a forward FPV view. `--camera-pitch` tilts it down.

        FOV: `HFOV = 2·atan(aperture / (2·focal))`, so
        `focal = aperture / (2·tan(HFOV/2))`. Since aperture is read back, Isaac's length-unit
        convention cancels out in the ratio; nothing is hard-coded.
        """
        body = find_body_prim(self.prim)
        cam_path = body + "/fpv_cam"
        print(f"[CAM]{self.tag} mount point: {cam_path}")

        self.cam = Camera(
            prim_path=cam_path,
            translation=np.array(offset, dtype=float),
            orientation=euler_angles_to_quat(
                np.array([0.0, pitch_deg, 0.0]), degrees=True),
            resolution=(width, height),
        )
        self.cam.initialize()

        # Set the focal length to get the requested HFOV.
        #
        # ⚠️ First **match the vertical aperture to the resolution aspect**. The camera is
        # created with a default aperture (about 4:3) but we render 16:9. Fixing only the
        # focal length makes the renderer widen the horizontal FOV to fill the aspect:
        # 90° on paper measured 111.6° on screen. set_horizontal_aperture with
        # maintain_square_pixels syncs the vertical aperture.
        aperture = float(self.cam.get_horizontal_aperture())
        self.cam.set_horizontal_aperture(aperture, maintain_square_pixels=True)
        aperture = float(self.cam.get_horizontal_aperture())
        v_aperture = float(self.cam.get_vertical_aperture())

        focal = aperture / (2.0 * math.tan(math.radians(fov_deg) / 2.0))
        self.cam.set_focal_length(focal)

        hfov = math.degrees(self.cam.get_horizontal_fov())
        vfov = math.degrees(self.cam.get_vertical_fov())
        print(f"[CAM]{self.tag} {width}x{height}, HFOV {hfov:.2f}° / VFOV {vfov:.2f}° "
              f"(requested {fov_deg:.1f}°)")
        # Requested (fov_deg) and measured differ: a 90° request gave 91.7°.
        # B's back-projection needs the measured value, so sim_info keeps both. With only
        # the request, B can't tell it is off and measurements are silently corrupted.
        self.hfov_deg = round(hfov, 3)
        self.vfov_deg = round(vfov, 3)
        print(f"[CAM]{self.tag} aperture H {aperture:.4f} / V {v_aperture:.4f} "
              f"(ratio {aperture / v_aperture:.4f}, screen {width / height:.4f}) "
              f"→ focal {focal:.4f}")
        print(f"[CAM]{self.tag} ⚠️ set fov_horizontal in the runner config to {fov_deg:.0f}. "
              "Don't trust the math; compare with the measured HFOV above")
        if pitch_deg:
            print(f"[CAM]{self.tag} tilted down {pitch_deg:.1f}°; action_projector's "
                  "2D->3D assumption changes")

        # GT depth sidecar. `distance_to_image_plane` is **distance along the optical axis**,
        # the same definition as AirSim's DepthPlanar. `distance_to_camera` (radial distance)
        # would break the runner's pinhole back-projection and silently drift at the edges.
        self.depth_path = depth_path if depth else None
        self.depth_tmp = (self.depth_path + ".tmp") if self.depth_path else None
        self._depth_warned = False
        if self.depth_path:
            self.cam.add_distance_to_image_plane_to_frame()
            os.makedirs(os.path.dirname(self.depth_path) or ".", exist_ok=True)
            print(f"[CAM]{self.tag} GT depth sidecar: {self.depth_path} "
                  "(distance_to_image_plane = optical-axis distance)")

        # Continuous recording. File names follow the convention `artifacts.encode_flight_videos`
        # relies on: it extracts the camera token (`split("_")[2]`) and measures fps from timestamps:
        #   img_<vehicle>_<camera>_<imagetype>_<ns>.png
        # The runner picks only frames from its own run window, so A just keeps appending.
        self.record_dir = None
        self.record_keep = int(record_keep or 0)
        self._recorded = collections.deque()
        self._record_warned = False
        if record_root:
            self.record_dir = os.path.join(record_root, "images")
            os.makedirs(self.record_dir, exist_ok=True)
            print(f"[CAM]{self.tag} continuous recording: {self.record_dir} (keeping at most {self.record_keep} frames)")

        self.frame_path = frame_path
        os.makedirs(os.path.dirname(frame_path) or ".", exist_ok=True)
        self.frame_tmp = frame_path + ".tmp"
        phys_hz = self.phys_hz
        self.steps_per_frame = max(1, int(round(phys_hz / frame_hz)))
        print(f"[CAM]{self.tag} frame output: {frame_path} @ {frame_hz:.1f} Hz "
              f"(every {self.steps_per_frame} steps)")
        self._frames_written = 0
        self._frame_warned = False
        self._write_warned = False

    def write_frame(self):
        """Atomically replace the frame file with the latest render.

        Write to tmp and move with os.replace. replace is atomic on the same filesystem, so
        the reader (process B) always sees a complete file, never a half-written JPEG.
        (The tmp name doesn't end in .jpg, so format is explicit; otherwise PIL fails
        trying to infer it from the extension.)

        **A frame write failure must not kill the sim.** If process A dies, PX4 goes down
        with it and the flight is cut. Failures are reported once and the loop continues.
        """
        try:
            rgba = self.cam.get_rgba()
            if rgba is None or not getattr(rgba, "size", 0):
                if not self._frame_warned:
                    print(f"[CAM]{self.tag} frame still empty (renderer warming up)", flush=True)
                    self._frame_warned = True
                return
            Image.fromarray(rgba[:, :, :3].astype("uint8")).save(
                self.frame_tmp, format="JPEG", quality=90)
            os.replace(self.frame_tmp, self.frame_path)
            # depth is extracted **from the same render as RGB** and replaced the same way,
            # so the runner can assume both share the same viewpoint.
            # Keep RGB first: the runner counts steps by RGB, and a depth one frame late
            # safely takes the `eff is None` path.
            if self.depth_path:
                self.write_depth(rgba.shape[:2])
            if self.record_dir:
                self.record_frame(rgba)
            self._frames_written += 1
            if self._frames_written == 1:
                h, w = rgba.shape[:2]
                print(f"[CAM]{self.tag} first frame saved ({w}x{h}) → {self.frame_path}",
                      flush=True)
        except Exception as e:
            if not self._write_warned:
                print(f"[CAM]{self.tag} frame save failed (continuing): {e}", flush=True)
                self._write_warned = True

    def record_frame(self, rgba):
        """Save one rendered frame as PNG into the record folder.

        Failures must not kill the sim (same principle as `_write_frame`). Without a
        recording the runner skips mp4 assembly and still produces everything else.
        """
        try:
            ts = time.time_ns()
            name = f"img_quadrotor_0_0_{ts}.png"
            path = os.path.join(self.record_dir, name)
            Image.fromarray(rgba[:, :, :3].astype("uint8")).save(path, format="PNG")
            self._recorded.append(path)
            # Delete oldest first; unbounded growth fills the disk (AirSim measured 915 runs = 930 GB).
            while self.record_keep and len(self._recorded) > self.record_keep:
                old = self._recorded.popleft()
                try:
                    os.remove(old)
                except OSError:
                    pass
        except Exception as e:
            if not self._record_warned:
                print(f"[CAM]{self.tag} recording save failed (continuing): {e}", flush=True)
                self._record_warned = True

    def write_depth(self, rgb_hw):
        """Atomically replace GT depth as a float32 2D .npy.

        Failures must not kill the sim (same principle as `_write_frame`). Without depth the
        runner downgrades `depth_source=oracle` to model and keeps flying.
        """
        try:
            depth = self.cam.get_depth()
            if depth is None:
                if not self._depth_warned:
                    print(f"[CAM]{self.tag} depth annotator still empty (renderer warming up)", flush=True)
                    self._depth_warned = True
                return
            arr = np.asarray(depth, dtype=np.float32)
            # The annotator may return (h, w, 1); the runner expects 2D.
            if arr.ndim == 3 and arr.shape[2] == 1:
                arr = arr[:, :, 0]
            if arr.shape != tuple(rgb_hw):
                if not self._depth_warned:
                    print(f"[CAM]{self.tag} ⚠️ depth resolution differs from RGB "
                          f"(depth={arr.shape}, rgb={tuple(rgb_hw)}); not writing sidecar",
                          flush=True)
                    self._depth_warned = True
                return
            np.save(self.depth_tmp, arr)
            # np.save appends .npy when there is no extension; the tmp name already ends in
            # .npy.tmp so it is kept as is. Check the path actually written and move it.
            written = self.depth_tmp if os.path.exists(self.depth_tmp) else self.depth_tmp + ".npy"
            os.replace(written, self.depth_path)
            if self._frames_written == 0:
                print(f"[CAM]{self.tag} first depth saved {arr.shape} → {self.depth_path}", flush=True)
        except Exception as e:
            if not self._depth_warned:
                print(f"[CAM]{self.tag} depth save failed (continuing): {e}", flush=True)
                self._depth_warned = True


def _isaac6_teleport(v, pos, quat_wxyz):
    """isaac6-port: Pegasus 6 Vehicle is not an Isaac Robot (no set_world_pose). Move every rigid body
    of the vehicle (body + rotors) rigidly so the joints stay intact, then zero velocities."""
    import omni.usd
    from pxr import UsdPhysics
    from scipy.spatial.transform import Rotation as _R
    from isaacsim.core.experimental.prims import RigidPrim as _RP
    stage = omni.usd.get_context().get_stage()
    root = stage.GetPrimAtPath(v._stage_prefix)
    paths, stack = [], [root]
    while stack:
        p = stack.pop(); stack.extend(p.GetChildren())
        if p.HasAPI(UsdPhysics.RigidBodyAPI):
            paths.append(str(p.GetPath()))
    body = v._stage_prefix + "/body"
    paths.sort(key=lambda x: x != body)
    rp = _RP(paths=paths)
    P, Q = rp.get_world_poses()
    P = P.numpy().astype(float); Q = Q.numpy().astype(float)
    Rb = _R.from_quat([Q[0,1], Q[0,2], Q[0,3], Q[0,0]])
    w = quat_wxyz
    Rn = _R.from_quat([w[1], w[2], w[3], w[0]])
    Rd = Rn * Rb.inv()
    newP = np.asarray(pos, dtype=float) + Rd.apply(P - P[0])
    newQ = []
    for q in Q:
        x = (Rd * _R.from_quat([q[1], q[2], q[3], q[0]])).as_quat()
        newQ.append([x[3], x[0], x[1], x[2]])
    rp.set_world_poses(positions=newP.astype(np.float32), orientations=np.array(newQ, dtype=np.float32))
    z = np.zeros((len(paths), 3), dtype=np.float32)
    rp.set_velocities(linear_velocities=z, angular_velocities=z)
    print(f"[RESET] isaac6 teleport: {len(paths)} rigid bodies -> {list(pos)}", flush=True)

class PegasusSimApp:
    def __init__(self, env_name, vehicle_id, altitude,
                 render=True, render_hz=None, physics_device=None, physics_hz=None,
                 rtf_every=10.0, camera=None, reset_file=None, info_file=None,
                 livestream=None, wind_enu=(0.0, 0.0, 0.0), n_vehicles=1):
        self.timeline = omni.timeline.get_timeline_interface()
        self.render = render
        self.rtf_every = float(rtf_every)

        self.pg = PegasusInterface()
        # Pegasus defaults: physics_dt=1/250, rendering_dt=1/60, device="cpu".
        # When RTF < 1.0, duration-based commands get proportionally shorter.
        if physics_device is not None:
            self.pg.set_world_settings(device=physics_device)
        if physics_hz is not None:
            self.pg.set_world_settings(physics_dt=1.0 / physics_hz)
        # Render cadence is counted in the step loop rather than left to Isaac's
        # rendering_dt; "render every N steps" is explicit, so its link to RTF is clear.
        ws = self.pg._world_settings
        phys_hz = 1.0 / ws["physics_dt"]
        if not render or (render_hz is not None and render_hz <= 0):
            self.steps_per_render = 0            # no rendering
        elif render_hz is None:
            self.steps_per_render = 1            # every step: RTF 0.5. Not recommended
        else:
            self.steps_per_render = max(1, int(round(phys_hz / render_hz)))
            self.pg.set_world_settings(rendering_dt=1.0 / render_hz)
            ws = self.pg._world_settings

        if self.steps_per_render == 0:
            render_desc = "none"
        else:
            render_desc = (f"{phys_hz / self.steps_per_render:.1f} Hz "
                           f"(once every {self.steps_per_render} steps)")
        print(f"[SIM] physics: {phys_hz:.0f} Hz, device={ws.get('device')} | "
              f"render: {render_desc}")
        self.pg._world = World(**self.pg._world_settings)
        self.world = self.pg._world  # isaac6-port: Pegasus 6 has no .world property

        if env_name not in SIMULATION_ENVIRONMENTS:
            print(f"[SIM] unknown environment '{env_name}'. Available:")
            for k in SIMULATION_ENVIRONMENTS:
                print(f"        {k}")
            simulation_app.close()
            sys.exit(2)

        print(f"[SIM] loading environment: {env_name}")
        self.pg.load_environment(SIMULATION_ENVIRONMENTS[env_name])

        # Pegasus launches PX4 SITL itself (px4_autolaunch). No need to run make px4_sitl
        # in a separate terminal, and it is cleaned up when the sim exits.
        mavlink_config = PX4MavlinkBackendConfig({
            "vehicle_id": vehicle_id,
            "px4_autolaunch": True,
            "px4_dir": self.pg.px4_path,
            "px4_vehicle_model": self.pg.px4_default_airframe,
        })
        config_multirotor = MultirotorConfig()
        config_multirotor.backends = [PX4MavlinkBackend(mavlink_config)]
        # Wind. At 0 it equals the default LinearDrag, so leaving it would be harmless,
        # but it is always swapped in so the coefficients get logged.
        coeffs = np.diag(config_multirotor.drag._drag_coefficients).copy()
        config_multirotor.drag = WindDrag(coeffs, wind_enu)
        self.wind_enu = [float(v) for v in wind_enu]
        if any(self.wind_enu):
            print(f"[WIND] ENU {self.wind_enu} m/s, drag coefficients {list(coeffs)}")
            print("[WIND] ⚠️ not numerically equivalent to AirSim's simSetWind; no cross-sim "
                  "absolute comparison. Only valid vs. zero wind within the same sim")
            if pegasus_wind.vertical_is_ineffective(coeffs, self.wind_enu):
                print("[WIND] ⚠️ vertical wind given but the z drag coefficient is 0, so it has **no effect** "
                      f"(coefficients {list(coeffs)}); only horizontal components apply")
        else:
            print("[WIND] none (same as default LinearDrag)")

        # N vehicles. With N=1 everything is exactly as before: same prim name and spawn.
        # From vehicle 1 a suffix is added and vehicles are spread along x (overlapping
        # bodies would push each other in physics).
        self.n_vehicles = max(1, int(n_vehicles))
        phys_hz_now = 1.0 / self.pg._world_settings["physics_dt"]
        self.rigs = []
        for i in range(self.n_vehicles):
            prim_i = vehicle_prim_for(i)
            vid = vehicle_id + i
            spawn_i = pegasus_layout.spawn_for([0.0, 0.0, altitude], i)
            if i > 0:
                # Each vehicle gets its own PX4 instance; ports split as baseport + id.
                cfg_i = PX4MavlinkBackendConfig({
                    "vehicle_id": vid,
                    "px4_autolaunch": True,
                    "px4_dir": self.pg.px4_path,
                    "px4_vehicle_model": self.pg.px4_default_airframe,
                })
                conf_i = MultirotorConfig()
                conf_i.backends = [PX4MavlinkBackend(cfg_i)]
                conf_i.drag = WindDrag(coeffs, wind_enu)
            else:
                conf_i = config_multirotor
            Multirotor(
                prim_i,
                ROBOTS["Iris"],
                vid,
                spawn_i,
                Rotation.from_euler("XYZ", [0.0, 0.0, 0.0], degrees=True).as_quat(),
                config=conf_i,
            )
            self.rigs.append(VehicleRig(i, vid, prim_i, phys_hz_now,
                                        spawn_i, [1.0, 0.0, 0.0, 0.0]))

        # articulation init; without it the vehicle doesn't take part in physics
        self.world.reset()
        self.stop_sim = False

        # Remember the spawn pose for resets (the target to restore)
        self.spawn_pos = [0.0, 0.0, altitude]
        self.spawn_quat_wxyz = [1.0, 0.0, 0.0, 0.0]      # level, heading 0
        self.vehicle_id = vehicle_id
        self.reset_file = reset_file
        if reset_file:
            os.makedirs(os.path.dirname(reset_file) or ".", exist_ok=True)
            # A leftover request from a previous session would trigger an immediate reset
            for f in (reset_file, reset_file + ".done"):
                try:
                    os.remove(f)
                except OSError:
                    pass
            print(f"[RESET] watching request file: {reset_file}")

        # Write sim conditions to a file; process B puts them in the run record.
        # RTF changes during the run, so it is rewritten periodically.
        self.info_file = info_file
        self.env_name = env_name
        self.camera_cfg = dict(camera) if camera else None
        # Streaming adds heavy render load and halves RTF. Recording RTF without the cause
        # would leave "why was this run 2x slower" unanswerable.
        self.livestream = livestream
        self.last_rtf = None
        self._write_info()

        # ── Vehicle cameras ──────────────────────────────────────────
        # One per vehicle. Only vehicle 0 keeps the default output paths; from vehicle 1 a
        # suffix is added, so the runner config defaults still work for a single vehicle.
        if camera:
            for rig in self.rigs:
                per = dict(camera)
                per["frame_path"] = per_vehicle_path(camera["frame_path"], rig.index)
                if per.get("depth_path"):
                    per["depth_path"] = per_vehicle_path(per["depth_path"], rig.index)
                if per.get("record_root"):
                    # Recording is split into per-vehicle folders; mixed in one folder the
                    # encoder can't separate them (that slot in the file name is the camera number).
                    per["record_root"] = pegasus_layout.record_root_for(
                        per["record_root"], rig.index)
                rig.setup_camera(**per)
            # The existing `camera.hfov_deg` key keeps **vehicle 0's value**; the runner's
            # FOV cross-check reads it and single-vehicle behavior must not change.
            if self.camera_cfg is not None and self.rigs[0].hfov_deg is not None:
                self.camera_cfg["hfov_deg"] = self.rigs[0].hfov_deg
                self.camera_cfg["vfov_deg"] = self.rigs[0].vfov_deg
            self._write_info()   # rewrite now that measured HFOV is filled in

        # What process B needs to know. This is also the prim path the camera attaches to.
        print("=" * 60)
        for rig in self.rigs:
            print(f"[SIM] vehicle {rig.index}       : {rig.prim} (vehicle_id {rig.vehicle_id})")
            print(f"[SIM]   PX4 sim link : tcp {4560 + rig.vehicle_id}")
            print(f"[SIM]   MAVSDK connect: udpin://0.0.0.0:{14540 + rig.vehicle_id}")
            if rig.frame_path:
                print(f"[SIM]   frames       : {rig.frame_path}")
        print(f"[SIM] px4_dir        : {self.pg.px4_path}")
        print(f"[SIM] airframe       : {self.pg.px4_default_airframe}")
        print("=" * 60)
        print("[SIM] Ready: you can start process B (the runner).", flush=True)

    def _write_info(self):
        """Write sim conditions as JSON ("record and compare").

        Process B can't know which scene, camera or RTF it ran under because the processes
        are separate. Without those conditions two runs can't be compared, so A writes them
        here and B copies them into the run record.
        """
        if not self.info_file:
            return
        ws = self.pg._world_settings
        info = {
            "env": self.env_name,
            "vehicle_prim": VEHICLE_PRIM,
            "vehicle_id": self.vehicle_id,
            "spawn": {"pos": list(self.spawn_pos),
                      "quat_wxyz": list(self.spawn_quat_wxyz)},
            "physics": {"hz": round(1.0 / ws["physics_dt"], 1),
                        "device": ws.get("device")},
            "render": {"steps_per_render": self.steps_per_render,
                       "hz": (round(1.0 / ws["physics_dt"] / self.steps_per_render, 2)
                              if self.steps_per_render else 0)},
            "camera": self.camera_cfg,
            # Vehicle list. The single-vehicle legacy keys (`vehicle_prim`, `spawn`,
            # `ports`, `camera`) are **kept as is**; the runner reads them.
            "vehicles": [rig.info() for rig in self.rigs],
            "n_vehicles": len(self.rigs),
            "rtf": self.last_rtf,
            "livestream": self.livestream,
            # Measurement condition; the runner doesn't interpret it, just copies it into run.json.
            # The axis convention is in the name so it isn't confused with AirSim's NED.
            "wind_enu": self.wind_enu,
            "ports": {"px4_sim_tcp": 4560 + self.vehicle_id,
                      "mavsdk_udp": 14540 + self.vehicle_id},
        }
        try:
            os.makedirs(os.path.dirname(self.info_file) or ".", exist_ok=True)
            tmp = self.info_file + ".tmp"
            with open(tmp, "w") as f:
                json.dump(info, f, ensure_ascii=False, indent=1)
            os.replace(tmp, self.info_file)
        except Exception as e:
            print(f"[SIM] info file write failed (ignored): {e}")

    def _reset_vehicle(self, restart_px4=True):
        """Return the vehicle to its spawn pose; counterpart of AirSim's `client.reset()`.

        **Why**: if the vehicle flips after scraping the ground, PX4 reports
        `Preflight Fail: Attitude failure (pitch)` / `Compass needs calibration` /
        `Failsafe activated` and `is_armable` then stays False, so process B hangs forever
        at connect. Without this, process A had to be fully restarted each time during
        development (~100 s including scene load).

        **Why not via MAVSDK**: restoring the pose is a sim (USD) job, and PX4's EKF and
        failsafe state must be reset too. Only process A can do both.

        PX4 restart uses the Pegasus backend's `stop()`/`start()`, which already pairs
        closing the mavlink link with killing/relaunching the PX4 process.
        """
        print("[RESET] vehicle reset started", flush=True)
        self._dump_scene("before reset")
        # Reset every vehicle. With N=1 this is the same as before.
        mgr = VehicleManager.get_vehicle_manager()
        found = []
        for rig in self.rigs:
            v = mgr.get_vehicle(rig.prim)
            if v is None:
                print(f"[RESET] vehicle not found: {rig.prim}")
                return False
            found.append((rig, v))

        all_backends = [(rig, list(getattr(v, "_backends", []))) for rig, v in found]

        # (1) Stop PX4 (the mavlink link closes with it)
        if restart_px4:
            for rig, backends in all_backends:
                for b in backends:
                    try:
                        b.stop()
                    except Exception as e:
                        print(f"[RESET]{rig.tag} backend.stop failed (continuing): {e}")

        # (2) Teleport each vehicle to its spawn pose and zero its velocity. Without clearing
        #     velocity, leftover momentum tips it over again after the restore.
        for rig, v in found:
            try:
                if hasattr(v, "set_world_pose"):
                    v.set_world_pose(position=np.array(rig.spawn_pos, dtype=float),
                                     orientation=np.array(rig.spawn_quat_wxyz, dtype=float))
                    v.set_linear_velocity(np.zeros(3))
                    v.set_angular_velocity(np.zeros(3))
                else:
                    _isaac6_teleport(v, rig.spawn_pos, rig.spawn_quat_wxyz)
            except Exception as e:
                print(f"[RESET]{rig.tag} pose restore failed: {e}")
                return False

        # (3) Step physics a few times so contact and rest settle
        for _ in range(50):
            self.world.step(render=False)

        # (4) Relaunch PX4. EKF and failsafe start clean in the new process.
        if restart_px4:
            for rig, backends in all_backends:
                for b in backends:
                    try:
                        b.start()
                    except Exception as e:
                        print(f"[RESET]{rig.tag} backend.start failed: {e}")
            print("[RESET] PX4 restarted; process B must reconnect", flush=True)

        self._dump_scene("after reset")
        print(f"[RESET] done, spawn {[rig.spawn_pos for rig in self.rigs]}", flush=True)
        return True

    def _dump_scene(self, tag):
        """Print /World children and the world position of each vehicle RigidBody.

        Needed to diagnose "after reset there appear to be two vehicles": whether there
        really are two prims, or one vehicle whose body and rotors came apart.
        """
        try:
            from pxr import UsdGeom
            import omni.usd
            stage = omni.usd.get_context().get_stage()
            world = stage.GetPrimAtPath("/World")
            kids = [c.GetName() for c in world.GetChildren()] if world else []
            print(f"[DUMP:{tag}] /World children: {kids}")

            for name in kids:
                if "quadrotor" not in name.lower():
                    continue
                v = stage.GetPrimAtPath(f"/World/{name}")
                rows = []
                stack = [v]
                while stack:
                    prim = stack.pop()
                    stack.extend(prim.GetChildren())
                    if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                        xf = UsdGeom.Xformable(prim)
                        m = xf.ComputeLocalToWorldTransform(0)
                        t = m.ExtractTranslation()
                        rows.append(f"{prim.GetName()}=({t[0]:+.2f},{t[1]:+.2f},{t[2]:+.2f})")
                print(f"[DUMP:{tag}] /World/{name} RigidBody: {' '.join(sorted(rows))}")
        except Exception as e:
            print(f"[DUMP:{tag}] failed: {e}")

    def _check_reset_request(self):
        """Check for and handle a reset request file.

        A file is used for the same reason as frame passing: zero new infrastructure to
        cross the process boundary, and the requester can be shell or Python.
        The request is JSON (plain `nopx4` is also accepted for backward compatibility):
            {"px4": true, "pos": [x, y, z], "yaw_deg": 0.0}
        If `pos`/`yaw_deg` are given, **that pose is remembered as the new spawn point**;
        needed for fixed per-task spawns.
        """
        if not self.reset_file or not os.path.exists(self.reset_file):
            return
        try:
            with open(self.reset_file) as f:
                body = f.read()
        except OSError:
            body = ""
        try:
            os.remove(self.reset_file)
        except OSError:
            pass

        req = {}
        try:
            req = json.loads(body) if body.strip().startswith("{") else {}
        except ValueError:
            req = {}
        restart_px4 = bool(req.get("px4", "nopx4" not in body))

        # ── Per-vehicle start poses ──────────────────────────────────────────
        # Benchmark episodes give a start pose per vehicle. AirSim set them one by one with
        # `simSetVehiclePose`, but Pegasus has no such RPC, so **this request is the only path**.
        # If `vehicles` is present it wins. Vehicles not mentioned are left untouched
        # (same rule as "untouched" in the multi-drone runner).
        vehicles_req = req.get("vehicles")
        if isinstance(vehicles_req, list) and vehicles_req:
            by_index = {rig.index: rig for rig in self.rigs}
            applied, unknown = [], []
            for entry in vehicles_req:
                try:
                    idx = int(entry["index"])
                    pos = [float(v) for v in entry["pos"]]
                except (KeyError, TypeError, ValueError) as e:
                    print(f"[RESET] could not read vehicle pose entry ({e}): {entry!r}")
                    continue
                rig = by_index.get(idx)
                if rig is None:
                    unknown.append(idx)
                    continue
                rig.spawn_pos = pos
                if entry.get("yaw_deg") is not None:
                    q = Rotation.from_euler(
                        "XYZ", [0.0, 0.0, float(entry["yaw_deg"])], degrees=True).as_quat()
                    rig.spawn_quat_wxyz = [q[3], q[0], q[1], q[2]]   # Isaac is w-first
                applied.append(f"{idx}→{pos}"
                               + (f"@{entry['yaw_deg']}°" if entry.get("yaw_deg") is not None else ""))
                if idx == 0:
                    # Vehicle 0 also updates the single-vehicle compat key (`spawn`); it has readers
                    self.spawn_pos = list(rig.spawn_pos)
                    self.spawn_quat_wxyz = list(rig.spawn_quat_wxyz)
            if unknown:
                print(f"[RESET] ⚠️ ignored unknown vehicle indices: {unknown} "
                      f"({len(self.rigs)} vehicles)")
            if applied:
                print(f"[RESET] per-vehicle start poses ({len(applied)}): {', '.join(applied)}")
            untouched = [i for i in by_index if i not in
                         {int(e.get("index", -1)) for e in vehicles_req if isinstance(e, dict)}]
            if untouched:
                print(f"[RESET] leaving unspecified vehicles as is: {sorted(untouched)}")

        # Spawn change **relative to vehicle 0**; other vehicles keep their relative layout.
        # **Skipped** when `vehicles` was given: the base pose would overwrite the per-vehicle
        # poses and silently discard them.
        base_pos = req.get("pos")
        base_yaw = req.get("yaw_deg")
        if vehicles_req and (base_pos is not None or base_yaw is not None):
            print("[RESET] ⚠️ both `vehicles` and `pos`/`yaw_deg` given; "
                  "using per-vehicle poses and ignoring the base pose")
        elif base_pos is not None or base_yaw is not None:
            if base_pos is not None:
                base = [float(v) for v in base_pos]
                self.spawn_pos = base
                for rig in self.rigs:
                    rig.spawn_pos = pegasus_layout.spawn_for(base, rig.index)
            if base_yaw is not None:
                q = Rotation.from_euler("XYZ", [0.0, 0.0, float(base_yaw)],
                                        degrees=True).as_quat()   # x,y,z,w
                self.spawn_quat_wxyz = [q[3], q[0], q[1], q[2]]    # Isaac is w-first
                for rig in self.rigs:
                    rig.spawn_quat_wxyz = list(self.spawn_quat_wxyz)
            print(f"[RESET] new spawn pose: pos={self.spawn_pos} "
                  f"yaw={base_yaw if base_yaw is not None else 'unchanged'}"
                  + (f" ({len(self.rigs)} vehicles, spacing {VEHICLE_SPACING_M:g}m)"
                     if len(self.rigs) > 1 else ""))

        ok = self._reset_vehicle(restart_px4=restart_px4)
        self._write_info()      # update the record if the spawn changed

        # Leave a marker so the requester knows it finished
        try:
            with open(self.reset_file + ".done", "w") as f:
                f.write("ok\n" if ok else "fail\n")
        except OSError as e:
            print(f"[RESET] done marker failed (ignored): {e}")

    def run(self):
        """Step loop + RTF monitoring.

        RTF (real-time factor) = sim time / wall-clock time.

        `link.py`'s hold measures duration in **sim time**, so commands still execute fully
        at low RTF. It originally used wall-clock time, so at RTF 0.5 turns and forward moves
        came out half as long.

        RTF is still printed for two reasons. **(1) Throughput**: at RTF 0.15 the same flight
        takes 7x longer in wall time. **(2) Diagnostics**: wall-clock timeouts (EKF convergence,
        takeoff and landing waits) must scale with RTF, and this number explains why.
        """
        self.timeline.play()

        t_wall0 = time.monotonic()
        t_sim0 = self.world.current_time
        t_last = t_wall0
        i = 0

        while simulation_app.is_running() and not self.stop_sim:
            i += 1
            do_render = (self.steps_per_render > 0
                         and i % self.steps_per_render == 0)
            self.world.step(render=do_render)

            # Save only on rendered steps; buffers on non-rendered steps hold the same image
            first = self.rigs[0]
            if (first.cam is not None and do_render
                    and first.steps_per_frame
                    and i % first.steps_per_frame == 0):
                for rig in self.rigs:
                    rig.write_frame()

            # Watch for reset requests (every 0.1 s; a file stat is cheap)
            if i % 25 == 0:
                self._check_reset_request()

            if self.rtf_every > 0:
                now = time.monotonic()
                if now - t_last >= self.rtf_every:
                    d_wall = now - t_wall0
                    d_sim = self.world.current_time - t_sim0
                    rtf = d_sim / d_wall if d_wall > 0 else 0.0
                    print(f"[SIM] RTF={rtf:.3f}  (sim {d_sim:.1f}s / wall "
                          f"{d_wall:.1f}s)"
                          + ("" if rtf > 0.9 else
                             f"  ⚠️ {1.0 / rtf:.1f}x slower than wall clock"),
                          flush=True)
                    t_last = now
                    t_wall0, t_sim0 = now, self.world.current_time
                    self.last_rtf = round(rtf, 3)
                    self._write_info()

        carb.log_warn("PegasusSimApp closing.")
        self.timeline.stop()
        simulation_app.close()


def main():
    camera = None
    if not _args.no_camera and not _args.no_render:
        camera = {
            "frame_path": _args.frame_path,
            "frame_hz": _args.frame_hz,
            "width": _args.camera_width,
            "height": _args.camera_height,
            "fov_deg": _args.camera_fov,
            "offset": [float(v) for v in _args.camera_offset.split(",")],
            "pitch_deg": _args.camera_pitch,
            "depth": _args.depth,
            "depth_path": _args.depth_path if _args.depth else None,
            "record_root": _args.record,
            "record_keep": _args.record_keep if _args.record else None,
        }
    elif not _args.no_camera:
        print("[CAM] --no-render, so no camera is attached "
              "(no frames without rendering)")

    app = PegasusSimApp(_args.env, _args.vehicle_id, _args.altitude,
                        render=not _args.no_render,
                        render_hz=_args.render_hz,
                        physics_device=_args.physics_device,
                        physics_hz=_args.physics_hz,
                        rtf_every=_args.rtf_every,
                        camera=camera,
                        reset_file=_args.reset_file,
                        info_file=_args.info_file,
                        wind_enu=pegasus_wind.parse_wind(_args.wind),
                        n_vehicles=_args.vehicles,
                        livestream=(_args.livestream if _args.livestream is not None
                                    else False) or (_args.livestream is not None))
    try:
        app.run()
    except KeyboardInterrupt:
        print("\n[SIM] interrupted, exiting")
        app.timeline.stop()
        simulation_app.close()


if __name__ == "__main__":
    main()
