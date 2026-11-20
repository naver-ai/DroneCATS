# DroneCATS
# Copyright (c) 2026-present NAVER Cloud Corp.
# Apache-2.0

"""Pure planning layer: one ``Action`` -> list of low-level commands.

This module knows nothing about AirSim, threads, files or time. The only
side effect is the ``[YAW]`` / ``[VERT]`` / ``[STEP]`` stdout lines (§5.1).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - type-only import, keeps this module dependency-free
    from vlmpilot.policy.action import Action

# --- command representation (§2.1) -------------------------------------------

Command = tuple[str, dict]

KIND_TURN = "turn_in_place"  # yaw in place
KIND_MOVE = "body_velocity"  # body-frame velocity move
KIND_HOLD = "hold"  # cancel in-flight command + stop in place

# Thresholds fixed by §3.1 (not config keys).
_NO_MOTION_NORM_M = 0.01  # R1
_MIN_HORIZONTAL_M = 0.01  # R2
_ZERO_STEP_M = 1e-3  # R6


def _say(msg: str) -> None:
    print(msg, flush=True)


@dataclass(frozen=True)
class StepPlanner:
    """Turns a body-frame displacement into ``[turn?] + (move | hold)`` (§3.1)."""

    max_step_distance_m: float = 3.0
    step_duration_s: float = 2.5
    base_yaw_rate: float = 60.0
    yaw_deadband_deg: float = 8.0
    vertical_deadband: float = 0.15

    @classmethod
    def from_config(cls, config: Mapping[str, Any] | None) -> "StepPlanner":
        """Read the §4.1 keys; unknown keys are ignored, missing keys keep defaults."""
        if not config:
            return cls()
        kwargs: dict[str, float] = {}
        for f in fields(cls):
            value = config.get(f.name)
            if value is not None:
                kwargs[f.name] = float(value)
        return cls(**kwargs)

    def plan(
        self,
        action: "Action",
        distance_scale: float = 1.0,
        precise_stop: bool = False,
    ) -> list[Command]:
        """Apply rules R1..R9 of §3.1 in order. Never mutates ``action`` (§6-E7)."""
        fwd = float(action.forward)
        right = float(action.right)
        up = 0.0 - float(action.down)  # planned as climb(+); NED sign goes back in R7

        # R1: no motion -> active stop, nothing else evaluated or printed.
        if math.sqrt(right * right + fwd * fwd + up * up) < _NO_MOTION_NORM_M:
            return [(KIND_HOLD, {})]

        commands: list[Command] = []

        # R2: horizontal distance.
        d_xy = math.hypot(right, fwd)
        vertical = up

        if d_xy > _MIN_HORIZONTAL_M:
            # R3: yaw first, with strict-> deadband.
            theta = math.degrees(math.atan2(right, fwd))
            if abs(theta) > self.yaw_deadband_deg:
                commands.append(
                    (KIND_TURN, {"angle_deg": theta, "yaw_rate_dps": float(self.base_yaw_rate)})
                )
                _say(f"[YAW] in-place rotation {theta:+.1f}° (deadband {self.yaw_deadband_deg:g}°, rate {self.base_yaw_rate:g}°/s)")
            else:
                _say(f"[YAW] bearing error {theta:+.1f}° ≤ deadband {self.yaw_deadband_deg:g}° → rotation skipped")

            # R4: vertical ratio deadband, strict-<.
            frac = abs(up) / max(d_xy, 1e-6)
            if frac < self.vertical_deadband:
                if up != 0.0:
                    _say(f"[VERT] |up|/d_xy={frac:.3f} < {self.vertical_deadband:g} → altitude held (up {up:+.2f}m ignored)")
                vertical = 0.0

        # R5: clamp + proportional shrink (direction preserved).
        forward = d_xy
        raw = math.hypot(forward, vertical)
        want = raw * distance_scale
        step = min(want, self.max_step_distance_m)
        if raw > 1e-6:
            k = step / raw
            forward *= k
            vertical *= k

        # R6: zero-distance step -> active hold (a preceding turn survives).
        if step < _ZERO_STEP_M:
            _say(f"[STEP] travel distance {step:.3f}m (raw {raw:.2f}m × {distance_scale:g}) → stop (hold)")
            commands.append((KIND_HOLD, {}))
            return commands

        # R7: velocities. NED vertical sign flip.
        duration = float(self.step_duration_s)
        vx = forward / duration
        vy = 0.0
        vz = -vertical / duration
        _say(
            f"[STEP] raw {raw:.2f}m × {distance_scale:g} → step {step:.2f}m "
            f"(cap {self.max_step_distance_m:g}m) / {duration:g}s → vx={vx:.2f} vz={vz:+.2f}"
            + (" brake" if precise_stop else "")
        )

        # R8: the move command.
        commands.append(
            (
                KIND_MOVE,
                {
                    "vx": vx,
                    "vy": vy,
                    "vz": vz,
                    "duration": duration,
                    "brake": bool(precise_stop),
                    "step_distance_m": step,
                    "yaw_angle": 0.0,
                },
            )
        )
        return commands
