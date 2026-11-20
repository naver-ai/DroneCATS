# DroneCATS
# Copyright (c) 2026-present NAVER Cloud Corp.
# Apache-2.0

"""vehicle_frame.py — spawn-offset correction for multi-vehicle AirSim.

AirSim exposes two coordinate frames per vehicle:

* ``simGetObjectPose(name)``            — **global** (world) frame
* ``simGetVehiclePose(vehicle_name)``   — **relative** to that vehicle's spawn origin
  (``simSetVehiclePose`` interprets its argument in the same relative frame)

With one vehicle at the origin both agree; with a multi-vehicle ``settings.json`` every
vehicle but the first is shifted by its spawn offset. Targets are spawned in the global
frame, so anything that places a vehicle or measures a distance must convert (§5.2-R6).

Pure functions plus four thin RPC wrappers. No stdout here except nothing — callers log.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SpawnOffset:
    """``global − relative`` for one vehicle (m, NED). ``measured=False`` = could not be
    determined (the scene has no object by that vehicle name → NaN pose) → zero offset."""

    dx: float
    dy: float
    dz: float
    measured: bool

    @property
    def horizontal_m(self) -> float:
        return math.hypot(self.dx, self.dy)


ZERO_OFFSET = SpawnOffset(0.0, 0.0, 0.0, measured=False)


def _xyz(position: Any) -> tuple[float, float, float]:
    return (float(position.x_val), float(position.y_val), float(position.z_val))


def _has_nan(values: Sequence[float]) -> bool:
    return any(math.isnan(v) for v in values)


def measure_offsets(client: Any, vehicles: Sequence[str]) -> dict[str, SpawnOffset]:
    """§5.2-R7: read both frames at the same instant and subtract — once per vehicle.

    A NaN component in the global pose means the scene has no object for that name;
    that vehicle gets ``SpawnOffset(0, 0, 0, measured=False)`` so the caller can tell.
    RPC exceptions propagate — the caller decides how to log and whether to continue.
    """
    out: dict[str, SpawnOffset] = {}
    for veh in vehicles:
        g = _xyz(client.simGetObjectPose(veh).position)
        r = _xyz(client.simGetVehiclePose(vehicle_name=veh).position)
        if _has_nan(g) or _has_nan(r):
            out[veh] = ZERO_OFFSET
            continue
        out[veh] = SpawnOffset(g[0] - r[0], g[1] - r[1], g[2] - r[2], measured=True)
    return out


def to_local(
    offsets: Mapping[str, SpawnOffset],
    vehicle: str,
    gx: float,
    gy: float,
    gz: float | None = None,
) -> tuple[float, ...]:
    """Global → that vehicle's relative frame. Unknown vehicle → zero offset (no-op).

    Returns ``(x, y)`` when ``gz`` is None, else ``(x, y, z)``.
    """
    off = offsets.get(vehicle, ZERO_OFFSET)
    lx = float(gx) - off.dx
    ly = float(gy) - off.dy
    if gz is None:
        return (lx, ly)
    return (lx, ly, float(gz) - off.dz)


def global_of(client: Any, vehicle: str) -> tuple[float, float, float]:
    """§5.2-R8: the vehicle's position in the global frame (``simGetObjectPose``)."""
    return _xyz(client.simGetObjectPose(vehicle).position)


def offset_report(offsets: Mapping[str, SpawnOffset]) -> str:
    """One-line summary for the ``[START] drone spawn offsets:`` log — ``Drone1 +0.00m, Drone2 +8.00m``.

    Unmeasured vehicles are marked so a silent zero is never mistaken for a real zero.
    """
    parts = []
    for name, off in offsets.items():
        text = f"{name} {off.horizontal_m:+.2f}m"
        if not off.measured:
            text += "(not measured)"
        parts.append(text)
    return ", ".join(parts)
