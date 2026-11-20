# DroneCATS
# Copyright (c) 2026-present NAVER Cloud Corp.
# Apache-2.0

"""Pinhole camera model.

Pixel <-> body-frame 3D offset conversion. Stateless, frozen, pure ``math``.
Shared by the policy (model depth) and the controller (oracle GT depth re-projection).

Conventions (fixed):
  * pixel origin top-left, ``px`` grows right, ``py`` grows down
  * offsets are FRD: ``forward``(+) == depth along the optical axis, ``right``(+), ``down``(+)
  * principal point defaults to the exact (float) image centre; a calibrated (cx, cy) may be given
  * one focal length derived from the *horizontal* FOV, reused vertically (square pixels)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

DEFAULT_FOV_HORIZONTAL_DEG = 90.0


@dataclass(frozen=True)
class PinholeCamera:
    width: int
    height: int
    fov_horizontal_deg: float = DEFAULT_FOV_HORIZONTAL_DEG
    cx: float | None = None   # principal point (optical axis) in pixels. None → width/2 (sim camera)
    cy: float | None = None   # None → height/2

    def __post_init__(self) -> None:
        if not self.width > 0:
            raise ValueError(f"width must be > 0, got {self.width!r}")
        if not self.height > 0:
            raise ValueError(f"height must be > 0, got {self.height!r}")
        if not 0.0 < float(self.fov_horizontal_deg) < 180.0:
            raise ValueError(
                f"fov_horizontal_deg must be in (0, 180), got {self.fov_horizontal_deg!r}"
            )
        if self.cx is not None and not 0.0 < float(self.cx) < self.width:
            raise ValueError(f"cx must be in (0, width), got {self.cx!r}")
        if self.cy is not None and not 0.0 < float(self.cy) < self.height:
            raise ValueError(f"cy must be in (0, height), got {self.cy!r}")

    @property
    def focal_px(self) -> float:
        # C2: f = width / (2 * tan(fov_h / 2)); horizontal FOV only.
        return self.width / (2.0 * math.tan(self.fov_horizontal_deg * math.pi / 360.0))

    @property
    def center(self) -> tuple[float, float]:
        # C1: real-valued centre, no integer division. Uses the calibrated principal point when given.
        cx = self.width / 2.0 if self.cx is None else float(self.cx)
        cy = self.height / 2.0 if self.cy is None else float(self.cy)
        return (cx, cy)

    def pixel_to_offset(self, px: float, py: float, depth_m: float) -> tuple[float, float, float]:
        """Back-project a pixel at ``depth_m`` (optical-axis distance) to (forward, right, down).

        C3–C5: no clamping of depth or pixel coordinates; out-of-frame pixels extrapolate.
        """
        f = self.focal_px
        cx, cy = self.center
        right = (px - cx) / f * depth_m
        down = (py - cy) / f * depth_m
        return (depth_m, right, down)

    def offset_to_pixel(self, forward: float, right: float, down: float) -> tuple[float, float]:
        """Exact inverse of :meth:`pixel_to_offset` (C6). ``forward <= 0`` -> ValueError."""
        if forward <= 0:
            raise ValueError(f"forward distance must be > 0, got {forward!r}")
        f = self.focal_px
        cx, cy = self.center
        px = cx + right * f / forward
        py = cy + down * f / forward
        return (px, py)

    def normalized_to_pixel(self, u: float, v: float, scale: float) -> tuple[int, int]:
        """Model coordinates in ``[0, scale]`` -> pixel, truncated toward zero (C7). No clamping."""
        px = int(u / scale * self.width)
        py = int(v / scale * self.height)
        return (px, py)


def canonical_depth_scale(
    camera: object | None = None, fov_horizontal_deg: float | None = None
) -> float | None:
    """canonical depth → metric conversion factor ``k = fx / W = 1 / (2*tan(hFOV/2))``.

    Canonical depth in the training labels is mapped back with ``metric = canonical * k`` (hFOV 90° → 0.5).
    If ``camera`` is given, derive it from that camera's focal length / width (self-consistent, since it uses
    the same f as back-projection). Otherwise fall back to ``fov_horizontal_deg``; if neither, ``None``.
    """
    if camera is not None:
        f = getattr(camera, "focal_px", None)
        w = getattr(camera, "width", None)
        if f is not None and w:
            return float(f) / float(w)
    if fov_horizontal_deg is None:
        return None
    fov = float(fov_horizontal_deg)
    if not 0.0 < fov < 180.0:
        return None
    return 1.0 / (2.0 * math.tan(math.radians(fov) / 2.0))
