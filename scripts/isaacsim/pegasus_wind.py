#!/usr/bin/env python
# DroneCATS
# Copyright (c) 2026-present NAVER Cloud Corp.
# Apache-2.0

"""Wind drag computation as pure functions.

The math used by process A (`pegasus_sim.py`) lives here separately for **testability**:
`pegasus_sim.py` boots Isaac Sim at the top level so it cannot be imported, and Pegasus's
`State`/`LinearDrag` pull in scipy (absent from the runner venv). This module uses only
numpy, so the runner tests can verify it directly.

**Numbers do not match AirSim.** AirSim injects wind into its own aerodynamics via
`simSetWind`; here it enters Pegasus's linear drag as relative velocity, so the same
"3 m/s" is not the same disturbance. Only **robustness vs. zero wind within the same sim**
is a valid comparison.

Frames (Pegasus convention):
  - inertial frame is **ENU** (x east, y north, z up)
  - vehicle body frame is **FLU** (forward-left-up)
  - `attitude` is a quaternion `[qx, qy, qz, qw]`: FLU body expressed in ENU
"""

from __future__ import annotations

import numpy as np


def body_from_enu(attitude_xyzw, v_enu) -> np.ndarray:
    """Rotate an ENU inertial vector into the FLU body frame (world -> body).

    Same as `Rotation.from_quat(attitude).inv().apply(v_enu)`, but done by hand
    because the runner venv has no scipy.

    body->world is `R`, so world->body is `Rᵀ`, i.e. rotating by the conjugate
    `q* = (-x, -y, -z, w)`.
    """
    q = np.asarray(attitude_xyzw, dtype=float)
    v = np.asarray(v_enu, dtype=float)
    u = -q[:3]  # conjugate = inverse rotation
    w = q[3]
    # Rodrigues form: v' = v + 2w(u×v) + 2u×(u×v)
    t = np.cross(u, v)
    return v + 2.0 * w * t + 2.0 * np.cross(u, t)


def relative_body_velocity(body_vel, attitude_xyzw, wind_enu) -> np.ndarray:
    """Velocity relative to the air mass (body frame).

    With zero wind the body velocity is returned **unchanged**, so the result matches
    `LinearDrag` exactly (regression guard).
    """
    v_body = np.asarray(body_vel, dtype=float)
    wind = np.asarray(wind_enu, dtype=float)
    if not wind.any():
        return v_body
    return v_body - body_from_enu(attitude_xyzw, wind)


def drag_force(drag_coefficients, body_vel, attitude_xyzw, wind_enu) -> np.ndarray:
    """Drag force (N) in the body frame: the `Pegasus LinearDrag` formula with relative velocity.

        f = -diag(c) · (v_body - R_wb · v_wind_enu)

    With `wind_enu == 0` this is `f = -diag(c) · v_body`, identical to `LinearDrag`.
    """
    c = np.asarray(drag_coefficients, dtype=float)
    rel = relative_body_velocity(body_vel, attitude_xyzw, wind_enu)
    return -(c * rel)


def parse_wind(text: str) -> list[float]:
    """`"x,y,z"` → `[x, y, z]` (ENU m/s). Raises `ValueError` on bad format."""
    parts = [p.strip() for p in str(text).split(",")]
    if len(parts) != 3:
        raise ValueError(f"wind must be three values x,y,z (got: {text!r})")
    return [float(p) for p in parts]


def vertical_is_ineffective(drag_coefficients, wind_enu, tol: float = 1e-9) -> bool:
    """True if vertical wind was requested but the z drag coefficient is 0, so it has no effect.

    The Iris default is `[0.50, 0.30, 0.0]`, i.e. **z is 0**: `--wind 0,0,5` does nothing
    to the vehicle. Rather than silently ignoring it, the caller checks this and warns.
    """
    c = np.asarray(drag_coefficients, dtype=float)
    w = np.asarray(wind_enu, dtype=float)
    return bool(abs(w[2]) > tol and abs(c[2]) <= tol)
