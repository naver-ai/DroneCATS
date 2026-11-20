#!/usr/bin/env python3
# DroneCATS
# Copyright (c) 2026-present NAVER Cloud Corp.
# Apache-2.0

"""Measure the fixed tilt (pitch) of the Tello front camera — no takeoff.

Principle: with the vehicle on a flat floor the lens is ~3cm above the floor. The line where a wall 2m ahead meets the floor
is atan(0.03/2) ≈ 0.9° below the optical horizon, so that line's pixel row ≈ the horizon row.
The more the camera looks down, the higher the horizon sits in the image:
down_tilt = atan((cy - y_horizon) / f),  f = W / (2 tan(HFOV/2)).  Positive = looking down.
The boundary is atan(lens_h/dist) below the horizon, so that amount is added to convert it to the horizon.

Usage:  uv run python tools/tello/tello_camera_tilt.py [--hfov 56.4] [--row Y] [--dist-m 2.0]
  Vehicle on the floor, a plain wall in front (about 2m), with the wall-floor boundary spanning the frame horizontally.
  With --row, that row is used as the boundary instead of auto-detection (set manually by looking at the saved image).
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))


def detect_junction_row_profile(bgr: np.ndarray) -> int | None:
    """Row where the per-row median brightness drops the most — bright wall → dark floor transition. Works on textured floors too."""
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(float)
    w = g.shape[1]
    prof = np.median(g[:, w // 4: w * 3 // 4], axis=1)
    prof = cv2.GaussianBlur(prof.reshape(-1, 1), (1, 9), 0).ravel()
    d = np.diff(prof)
    y = int(np.argmin(d))
    return y if d[y] < -2.0 else None


def detect_junction_row(bgr: np.ndarray) -> tuple[int | None, np.ndarray]:
    """Center row of the strongest 'near-horizontal' line. Falls back to the brightness profile, then None."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 40, 120)
    lines = cv2.HoughLinesP(edges, 1, math.pi / 180, threshold=120,
                            minLineLength=bgr.shape[1] // 3, maxLineGap=40)
    if lines is None:
        return detect_junction_row_profile(bgr), edges
    best = None
    for x1, y1, x2, y2 in np.asarray(lines).reshape(-1, 4):
        ang = abs(math.degrees(math.atan2(y2 - y1, x2 - x1)))
        if ang > 3.0:          # exclude lines tilted more than 3° from horizontal
            continue
        length = math.hypot(x2 - x1, y2 - y1)
        if best is None or length > best[0]:
            best = (length, (y1 + y2) / 2.0, (x1, y1, x2, y2))
    if best:
        return int(round(best[1])), edges
    return detect_junction_row_profile(bgr), edges


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hfov", type=float, default=56.4, help="horizontal FOV (measured by calibration)")
    ap.add_argument("--row", type=int, default=None, help="wall-floor boundary row (manual)")
    ap.add_argument("--dist-m", type=float, default=2.0, help="lens-to-wall distance (for correction)")
    ap.add_argument("--lens-h-m", type=float, default=0.03, help="lens height above the floor")
    ap.add_argument("--warmup", type=float, default=6.0)
    ap.add_argument("--image", default=None, help="compute from a saved frame instead of the drone")
    args = ap.parse_args()

    if args.image:
        frame = cv2.imread(args.image)
    else:
        from djitellopy import Tello
        from vlmpilot.controller.real.tello_controller import ResilientFrameRead

        t = Tello()
        t.connect()
        print(f"[TELLO] battery={t.get_battery()}%  pitch={t.get_pitch()}°  temp={t.get_temperature()}")
        t.streamon()
        reader = ResilientFrameRead(t.get_udp_video_address()).start()
        frame = reader.wait_for_frame(20.0)
        time.sleep(args.warmup)                   # stable frame after the keyframe
        frame = reader.frame if getattr(reader, "frame", None) is not None else frame
        reader.stop()
        t.streamoff()
        if frame is None:
            print("no frame"); return 1
        frame = cv2.cvtColor(np.asarray(frame), cv2.COLOR_RGB2BGR)

    os.makedirs("outputs/tello", exist_ok=True)
    cv2.imwrite("outputs/tello/camera_tilt_raw.jpg", frame)       # keep the raw image even if detection fails
    h, w = frame.shape[:2]
    cy = h / 2.0
    f = w / (2.0 * math.tan(math.radians(args.hfov) / 2.0))
    row, edges = (args.row, None) if args.row is not None else detect_junction_row(frame)

    os.makedirs("outputs/tello", exist_ok=True)
    vis = frame.copy()
    cv2.line(vis, (0, int(cy)), (w, int(cy)), (0, 255, 255), 1)       # frame center
    if row is not None:
        cv2.line(vis, (0, row), (w, row), (0, 0, 255), 2)             # detected boundary
    cv2.imwrite("outputs/tello/camera_tilt.jpg", vis)
    if edges is not None:
        cv2.imwrite("outputs/tello/camera_tilt_edges.jpg", edges)

    print(f"frame {w}x{h}, f={f:.1f}px (HFOV {args.hfov}°, VFOV {math.degrees(2*math.atan(h/(2*f))):.1f}°)")
    if row is None:
        print("wall-floor boundary auto-detection failed — look at outputs/tello/camera_tilt.jpg and set it with --row Y")
        return 2
    raw = math.degrees(math.atan((cy - row) / f))            # boundary above center → camera looks down
    corr = math.degrees(math.atan(args.lens_h_m / args.dist_m))
    print(f"boundary row y={row}  (center {cy:.0f}, {cy-row:+.0f}px above center)")
    print(f"RESULT: camera downward tilt ≈ {raw + corr:+.1f}° (from the boundary line {raw:+.1f}° + lens height correction {corr:.1f}°)")
    for hf in (56.4, 70.3, 82.6):
        ff = w / (2.0 * math.tan(math.radians(hf) / 2.0))
        print(f"        assuming HFOV {hf}°: {math.degrees(math.atan((cy - row) / ff)) + corr:+.1f}°")
    print("        saved outputs/tello/camera_tilt.jpg")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
