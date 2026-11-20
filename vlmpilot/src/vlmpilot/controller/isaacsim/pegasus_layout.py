#!/usr/bin/env python
# DroneCATS
# Copyright (c) 2026-present NAVER Cloud Corp.
# Apache-2.0

"""Names, paths and placement for multiple vehicles. Pure functions.

**Lives in the package because both A (process A) and B (the runner) use it.** It is
kept out of `pegasus_sim.py` for the same reason as `pegasus_wind.py`: that file boots
Isaac Sim at import time, so it cannot be imported or tested.

Standard library only, so it imports fine under A's bundled Isaac Sim Python (3.11):
A puts `<repo>/vlmpilot/src` on `sys.path` and imports `vlmpilot.controller.isaacsim.pegasus_layout`
(`vlmpilot/__init__.py` is empty, so nothing heavy comes along).

**Core rule: "index 0 stays as is".** Single-vehicle paths are validated in real
flight, and runner config defaults (`frame_path` etc.) point at them. Renaming vehicle
0 for multi-vehicle support would invalidate that and break readers. So only vehicle 0
is special; suffixes start at 1.
"""

from __future__ import annotations

import os

VEHICLE_PRIM = "/World/quadrotor"
# Spacing between vehicles (m). Overlapping spawns push each other and tip over.
VEHICLE_SPACING_M = 3.0


def vehicle_prim_for(index: int, base: str = VEHICLE_PRIM) -> str:
    """Prim path of vehicle `index`. **Index 0 is `base` unchanged.**"""
    return base if index == 0 else f"{base}{index}"


def per_vehicle_path(base: str, index: int) -> str:
    """Per-vehicle output path. Index 0 is `base`, others `<stem>_<index><ext>`.

    Keeping the extension matters: the recording encoder's glob fixes `.png`, and the
    runner's frame source decodes by extension via `cv2.imread`.
    """
    if index == 0:
        return base
    stem, ext = os.path.splitext(base)
    return f"{stem}_{index}{ext}"


def record_root_for(base: str, index: int) -> str:
    """Per-vehicle recording folder. Index 0 is `base`, others `<base>/v<index>`.

    Must not share a folder: the encoder splits streams by the **camera token** in the
    filename (3rd field of `img_<vehicle>_<camera>_…`), and A always writes `0` there.
    So vehicles are separated by folder.
    """
    return base if index == 0 else os.path.join(base, f"v{index}")


def spawn_for(base_pos, index: int, spacing: float = VEHICLE_SPACING_M) -> list[float]:
    """Spawn position of vehicle `index`, offset along x from vehicle 0.

    Spawn change requests (`pegasus_reset.py --pos`) are **relative to vehicle 0** only.
    Carrying N positions in a request complicates runners and task files; tasks only need
    "the same layout every time", so a fixed relative layout is enough.
    """
    x, y, z = (float(v) for v in base_pos)
    return [x + float(index) * float(spacing), y, z]


def mavsdk_port(vehicle_id: int, base: int = 14540) -> int:
    """MAVSDK UDP port of that vehicle. Pegasus splits by `baseport + vehicle_id`."""
    return int(base) + int(vehicle_id)


def px4_sim_port(vehicle_id: int, base: int = 4560) -> int:
    """PX4 sim-link TCP port of that vehicle (A<->PX4)."""
    return int(base) + int(vehicle_id)


def grpc_port(vehicle_id: int, base: int = 50051) -> int:
    """MAVSDK **gRPC** port of that vehicle (`mavsdk_server` listens here).

    Must differ per link. `System()` starts the server on 50051 by default; if a second
    link uses the same port it **attaches to the first server and sees the same
    vehicle**. That is why two links reported the same position in multi-vehicle tests.
    Single vehicle (`vehicle_id=0`) matches the MAVSDK default.
    """
    return int(base) + int(vehicle_id)


# ---------------------------------------------------------------------------
# Per-vehicle start poses
# ---------------------------------------------------------------------------
#
# Benchmark tasks give a start pose per vehicle (what `fleet.apply_start_poses` set one
# by one via `simSetVehiclePose` in AirSim). Pegasus has no such RPC, so **A's reset
# request takes per-vehicle poses**; it is the only way to place a vehicle at a specific
# spot.


def parse_vehicle_pose(text: str) -> dict:
    """`"1:3,0,0.07@90"` -> `{"index": 1, "pos": [3.0, 0.0, 0.07], "yaw_deg": 90.0}`.

    `@yaw` is optional (keeps that vehicle's heading). A malformed spec raises
    `ValueError`: silently falling back to defaults would **falsify the test conditions**
    (same policy as the wind arguments, which also fail hard on bad format).
    """
    raw = str(text).strip()
    head, _, yaw_part = raw.partition("@")
    idx_part, sep, pos_part = head.partition(":")
    if not sep:
        raise ValueError(f"missing vehicle index; format is `<index>:x,y,z[@yaw]` (got: {text!r})")
    try:
        index = int(idx_part.strip())
    except ValueError:
        raise ValueError(f"vehicle index is not an integer: {idx_part!r}") from None
    if index < 0:
        raise ValueError(f"vehicle index is negative: {index}")
    parts = [p.strip() for p in pos_part.split(",")]
    if len(parts) != 3:
        raise ValueError(f"position must be three values x,y,z (got: {pos_part!r})")
    entry = {"index": index, "pos": [float(p) for p in parts]}
    if yaw_part.strip():
        entry["yaw_deg"] = float(yaw_part)
    return entry


def build_pose_request(specs) -> list[dict]:
    """Turn several `--vehicle` arguments into the reset request's `vehicles` list.

    Giving the same vehicle twice raises `ValueError`; silently picking a winner leads
    to "I set it explicitly, why is it over there?".
    """
    out: list[dict] = []
    seen: set[int] = set()
    for spec in specs or ():
        entry = parse_vehicle_pose(spec)
        if entry["index"] in seen:
            raise ValueError(f"vehicle {entry['index']} specified twice")
        seen.add(entry["index"])
        out.append(entry)
    return sorted(out, key=lambda e: e["index"])


# ---------------------------------------------------------------------------
# Heading frame conversion
# ---------------------------------------------------------------------------
#
# The Isaac world is **ENU** (x east, y north, z up); PX4 is **NED** (x north, y east,
# z down). Both the reference axis and rotation direction differ, so yaw does not carry
# over directly:
#
#     yaw_ned = 90° - yaw_enu
#
# Verified by measurement: setting per-vehicle start poses of ENU +45° and −90° gave
# PX4 telemetry of +43.2° and +178.7° (predicted +45° / −180°; the <2° error is
# post-reset settling).
#
# **B must apply this:** `fleet.apply_start_poses` verifies poses after placing them
# (`_verify_start_poses`); comparing the task yaw with telemetry yaw directly is off by
# 90° and reports failure.


def wrap180(deg: float) -> float:
    """Normalize an angle to `[-180, 180)`."""
    return (float(deg) + 180.0) % 360.0 - 180.0


def yaw_enu_to_ned(yaw_enu_deg: float) -> float:
    """Isaac world (ENU) heading -> PX4 (NED) heading. Related by `90° - θ`."""
    return wrap180(90.0 - float(yaw_enu_deg))


# ---------------------------------------------------------------------------
# Slots: keep several people / sims on one machine from colliding
# ---------------------------------------------------------------------------
#
# **Why "slot" rather than "user".** The team shares one OS account (`isaac`), so the OS
# cannot tell "my processes" from "someone else's". So we **introduce one explicit
# separating value** and derive ports, paths, outputs and ownership from it.
#
# What collides without it (counted by measurement):
#
#     ports  4560+id(A<->PX4) · 14540+id(MAVSDK) · 14580+id(PX4 offboard)
#            18570+id(GCS) · 50051+id(gRPC)
#     files  latest.jpg · latest_gt.json · sim_info.json · reset.request
#            latest_depth.npy · recording folder
#
# **Files are the more dangerous part.** A port clash fails loudly at bind (the port
# guard), but files are silently overwritten: the runner reads someone else's frames and
# flies on normally.
#
# **The "index 0 stays as is" rule applies to slots too.** Slot 0 is **identical** to the
# paths and ports validated in real flight, so those validations, config defaults and
# documented command lines stay valid. Suffixes start at slot 1.
#
# **The cap is 10.** Upstream PX4 folds the offboard port to a single 14549 for instances
# 10 and above (`px4-rc.mavlink:6`). That is where MAVSDK attaches, so an 11th vehicle
# cannot be addressed. Slots x vehicles across the machine cannot exceed 10.

SLOT_ENV = "VLMPILOT_SLOT"          # ownership tag set on processes
SLOT_ROOT = "/tmp/vlmpilot_pegasus"  # slot 0 output root (the original path)
SLOT_REGISTRY = "/tmp/vlmpilot_slots"  # who uses which slot (for humans)
MAX_VEHICLES = 10                   # ceiling from px4-rc.mavlink:6


def slot_root(slot: int, base: str = SLOT_ROOT) -> str:
    """Output root of the slot. **Slot 0 is `base` unchanged.**

    Separated by directory rather than per-file suffixes: a slot writes six kinds of
    files whose names are baked into readers (`cv2.imread` decodes `latest.jpg` by
    extension), so splitting directories leaves reader code untouched.
    """
    slot = int(slot)
    return base if slot == 0 else f"{base}_s{slot}"


def default_vehicle_base(slot: int) -> int:
    """**Default** first `vehicle_id` when the registry gives no allocation: the slot number.

    **Not multiplied by the slot number.** With `slot × vehicles`, the `vehicles` value
    shifts slot boundaries: one person using slot 1 with 1 vehicle (id 1) and another
    using slot 0 with 4 vehicles (ids 0-3) **collide**. So multi-vehicle allocations come
    from the registry (`pegasus_slots.claim`), not arithmetic. This is only the default
    for the common "1 vehicle = slot number" case.
    """
    slot = int(slot)
    if slot < 0:
        raise ValueError(f"slot {slot}: must be >= 0")
    return slot


def check_vehicle_range(base: int, vehicles: int = 1,
                        max_vehicles: int = MAX_VEHICLES) -> list:
    """Check that `vehicles` ids starting at `base` are usable and return them.

    **`max_vehicles` is not a constant but a result of the current port policy.** The
    default 10 comes from upstream PX4 (`px4-rc.mavlink:6` folds the offboard port to a
    single 14549 for instances 10+, so MAVSDK cannot address an 11th vehicle).

    That ceiling **can be lifted**. The file is sourced by relative path inside the
    rootfs PX4 is given (`. px4-rc.mavlink` in `rcS`), so a per-slot rootfs copy with a
    modified port formula is enough. Then only this argument needs raising; the slot
    model stays.
    """
    base, vehicles = int(base), int(vehicles)
    if base < 0 or vehicles < 1:
        raise ValueError(f"base {base} · vehicles {vehicles}: base must be >= 0, vehicles >= 1")
    if base + vehicles > int(max_vehicles):
        raise ValueError(
            f"vehicle_id {base}~{base + vehicles - 1} exceeds the ceiling {max_vehicles}. "
            f"Upstream PX4 (px4-rc.mavlink:6) folds the offboard port to 14549 for "
            f"instances 10+, so they cannot be addressed. To lift it, put a px4-rc.mavlink "
            f"with a modified port formula in a per-slot rootfs copy and raise max_vehicles")
    return list(range(base, base + vehicles))


def slot_paths(slot: int, base: str = SLOT_ROOT) -> dict:
    """All file paths used by a slot. Keys match each tool's argument names.

    `frame` and `depth` are split again per vehicle (`per_vehicle_path`); a slot can
    hold several vehicles, so that branch stays.
    """
    root = slot_root(slot, base)
    return {
        "root": root,
        "frame_path": os.path.join(root, "latest.jpg"),
        "depth_path": os.path.join(root, "latest_depth.npy"),
        "info_file": os.path.join(root, "sim_info.json"),
        "reset_file": os.path.join(root, "reset.request"),
        "gt_live": os.path.join(root, "latest_gt.json"),
        # B -> A signal: "GT pixels needed now". A's measurement mode attaches the
        # annotator when it sees this. A does not know B's step number (the sidecar is
        # write-only), so this file is the **only observable** first-step moment.
        "gt_hold": os.path.join(root, "gt_hold.request"),
    }


def slot_ports(slot: int, vehicles: int = 1, base: int | None = None,
               max_vehicles: int = MAX_VEHICLES) -> dict:
    """All ports used by a slot, listed for `vehicles` vehicles.

    With `base`, use that allocation (set by the registry); otherwise the default
    (`default_vehicle_base`). The port guard uses these to check whether someone else
    holds them.
    """
    b = default_vehicle_base(slot) if base is None else int(base)
    ids = check_vehicle_range(b, vehicles, max_vehicles)
    return {
        "vehicle_ids": ids,
        "px4_sim_tcp": [px4_sim_port(i) for i in ids],
        "mavsdk_udp": [mavsdk_port(i) for i in ids],
        "grpc_tcp": [grpc_port(i) for i in ids],
        "px4_offboard_udp": [14580 + i for i in ids],
        "px4_gcs_udp": [18570 + i for i in ids],
    }


def slot_env(slot: int, env: dict | None = None) -> dict:
    """Copy of the environment with the `SLOT_ENV` tag set. Child processes inherit it.

    **This is the only way to tell ownership apart under a shared account.** Pegasus
    launches PX4 with `env=os.environ` (px4_launch_tool), so the tag propagates.
    """
    out = dict(os.environ if env is None else env)
    out[SLOT_ENV] = str(int(slot))
    return out


def slot_of_pid(pid: int) -> int | None:
    """Which slot that process belongs to. `None` if untagged.

    Reads `/proc/<pid>/environ`, always readable under the same account (one place where
    sharing the account helps). Untagged processes are **treated as someone else's**:
    it is safer for cleanup tools not to kill what they do not know.
    """
    try:
        with open(f"/proc/{int(pid)}/environ", "rb") as fh:
            raw = fh.read()
    except OSError:
        return None
    want = (SLOT_ENV + "=").encode()
    for item in raw.split(b"\0"):
        if item.startswith(want):
            try:
                return int(item[len(want):].decode())
            except ValueError:
                return None
    return None


def ancestor_pids(pid: int | None = None) -> set:
    """Set of PIDs from this process up to init.

    Needed so cleanup tools **do not kill their own ancestors**. Excluding only the
    parent still self-kills when a shell further up contains the same pattern (seen 5
    times in this harness).
    """
    pid = os.getpid() if pid is None else int(pid)
    seen: set = set()
    while pid and pid not in seen:
        seen.add(pid)
        try:
            with open(f"/proc/{pid}/stat", encoding="utf-8") as fh:
                # comm may contain spaces or parens, so read after the last ')'
                pid = int(fh.read().rsplit(")", 1)[1].split()[1])
        except (OSError, IndexError, ValueError):
            break
    return seen


def slot_pids(slot: int, exclude_self: bool = True) -> list:
    """PIDs of processes tagged with that slot, excluding ancestors.

    This is why `pkill -f <pattern>` is not used: the pattern appears in our own shell's
    cmdline and kills it, and above all it **kills other slots too**.
    """
    mine = ancestor_pids() if exclude_self else set()
    out = []
    for name in os.listdir("/proc"):
        if not name.isdigit():
            continue
        pid = int(name)
        if pid in mine:
            continue
        if slot_of_pid(pid) == int(slot):
            out.append(pid)
    return sorted(out)
