# DroneCATS
# Copyright (c) 2026-present NAVER Cloud Corp.
# Apache-2.0

"""Per-step decision data type.

Pure value holder — no validation, no normalisation, no dependencies.
The field names are the ``decision_*.json`` keys.
"""

from __future__ import annotations

from dataclasses import dataclass

ACTION_GO = "go"
ACTION_ROTATE = "rotate"
ACTION_THINK = "think"
ACTION_FINISHED = "finished"
ACTION_ARRIVED = "arrived"  # never produced by the policy; mission's final frame only


@dataclass
class Action:
    """One step's decision.

    Body-frame offset in FRD (the AirSim/PX4 body frame): ``forward``(+),
    ``right``(+), ``down``(+). ``pixel_u``/``pixel_v`` are the chosen image point
    in capture pixels (u right, v down).
    """

    kind: str
    forward: float = 0.0
    right: float = 0.0
    down: float = 0.0
    pixel_u: float = 0.0
    pixel_v: float = 0.0
    angle_deg: float = 0.0
    depth_m: float | None = None
    gt_depth_m: float | None = None  # always None from the policy; controller fills it later (R5)
    reason: str = ""

    @property
    def up(self) -> float:
        """Height gain (m), for log lines. ``0.0 - down`` so a zero prints as +0.0."""
        return 0.0 - self.down

    def __str__(self) -> str:
        # Human-readable one-liner (R7). Not a contract; must not start with "[" so it
        # can never satisfy any runner regex.
        return (
            f"action={self.kind} "
            f"R{self.right:+.1f} F{self.forward:+.1f} U{self.up:+.1f}"
        )
