#!/usr/bin/env python3
# DroneCATS
# Copyright (c) 2026-present NAVER Cloud Corp.
# Apache-2.0

"""Measure front camera tilt (pitch) and left-right misalignment (yaw) using a facing Tello — no takeoff.

Principle: two units of the same model on the same floor have their lenses at the same height → the direction to the other lens is the true horizon.
If the other lens is Δy px above the image center (precisely, the principal point cx, cy), the camera looks down by atan(Δy / fy).
No distance or lens-height correction is needed, and floor unevenness matters less at 3~4m apart.

Usage:
  uv run python tools/tello/tello_pitch_facing.py --intrinsics outputs/tello/calib/<folder>/intrinsics.json
  uv run python tools/tello/tello_pitch_facing.py --image saved.jpg --hfov 56.4        # without intrinsics
  Optional: --tape-mm 1000 --dist-m 3.2  →  independently cross-check f using two tape points on the wall behind the other unit

Click order: ① center of the other Tello's lens  (② ③ the two tape points, if --tape-mm is given)
  left click = place point, u = undo last point, Enter = compute, q = quit. If the lens is small, aim using the zoom window (bottom right).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))


def grab_live_frame(warmup: float) -> np.ndarray:
    from djitellopy import Tello
    from vlmpilot.controller.real.tello_controller import ResilientFrameRead

    t = Tello()
    t.connect()
    print(f"[TELLO] battery={t.get_battery()}%")
    t.streamon()
    reader = ResilientFrameRead(t.get_udp_video_address()).start()
    frame = reader.wait_for_frame(20.0)
    time.sleep(warmup)
    frame = reader.frame if reader.frame is not None else frame
    reader.stop(); t.streamoff()
    if frame is None:
        raise SystemExit("no frame")
    return cv2.cvtColor(np.asarray(frame), cv2.COLOR_RGB2BGR)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", default=None, help="saved frame. if omitted, one frame is taken from the drone")
    ap.add_argument("--intrinsics", default=None, help="intrinsics.json from tello_calib_solve.py")
    ap.add_argument("--hfov", type=float, default=56.4, help="horizontal FOV when there are no intrinsics (measured by calibration)")
    ap.add_argument("--tape-mm", type=float, default=None, help="spacing between the two wall tape points (mm)")
    ap.add_argument("--dist-m", type=float, default=None, help="distance from my lens to that wall (m)")
    ap.add_argument("--warmup", type=float, default=6.0)
    ap.add_argument("--out", default="outputs/tello/pitch_facing.jpg")
    args = ap.parse_args()

    frame = cv2.imread(args.image) if args.image else grab_live_frame(args.warmup)
    if frame is None:
        print("could not read the image"); return 1
    h, w = frame.shape[:2]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    cv2.imwrite(args.out.replace(".jpg", "_raw.jpg"), frame)

    if args.intrinsics:
        K = json.load(open(args.intrinsics, encoding="utf-8"))
        if (K["width"], K["height"]) != (w, h):
            print(f"[WARN] intrinsics {K['width']}x{K['height']} ≠ frame {w}x{h}")
        fx, fy, cx, cy = K["fx"], K["fy"], K["cx"], K["cy"]
        dist = np.array(K["dist"], dtype=np.float64)
        src = f"intrinsics {os.path.basename(os.path.dirname(args.intrinsics))} (fx {fx:.0f} fy {fy:.0f} cx {cx:.0f} cy {cy:.0f})"
    else:
        fx = fy = w / (2 * math.tan(math.radians(args.hfov) / 2))
        cx, cy, dist = w / 2, h / 2, np.zeros(5)
        src = f"assuming HFOV {args.hfov}° (f {fx:.0f}px, center = image center)  ← provisional value before calibration"
    Kmat = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

    need = 3 if args.tape_mm else 1
    pts: list[tuple[int, int]] = []
    labels = ["other Tello lens center", "tape point 1", "tape point 2"]
    zoom = 6

    def on_mouse(ev, x, y, flags, _):
        if ev == cv2.EVENT_LBUTTONDOWN and len(pts) < need:
            pts.append((x, y))

    win = "pitch facing  [click / u=undo / Enter=calc / q=quit]"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)
    mouse = [w // 2, h // 2]

    def on_move(ev, x, y, flags, _):
        on_mouse(ev, x, y, flags, _)
        mouse[0], mouse[1] = x, y
    cv2.setMouseCallback(win, on_move)

    while True:
        vis = frame.copy()
        cv2.line(vis, (0, int(round(cy))), (w, int(round(cy))), (0, 255, 255), 1)
        cv2.line(vis, (int(round(cx)), 0), (int(round(cx)), h), (0, 255, 255), 1)
        for i, (x, y) in enumerate(pts):
            cv2.drawMarker(vis, (x, y), (0, 0, 255), cv2.MARKER_CROSS, 18, 1)
            cv2.putText(vis, labels[i], (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        nxt = labels[len(pts)] if len(pts) < need else "Enter to compute"
        cv2.putText(vis, f"next click: {nxt}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        # zoom window (around the mouse, zoom x)
        r = 30
        mx, my = int(np.clip(mouse[0], r, w - r - 1)), int(np.clip(mouse[1], r, h - r - 1))
        patch = cv2.resize(frame[my - r:my + r, mx - r:mx + r], (2 * r * zoom, 2 * r * zoom), interpolation=cv2.INTER_NEAREST)
        cv2.drawMarker(patch, (r * zoom, r * zoom), (0, 255, 0), cv2.MARKER_CROSS, 40, 1)
        ph, pw = patch.shape[:2]
        vis[h - ph - 10:h - 10, w - pw - 10:w - 10] = patch
        cv2.imshow(win, vis)
        k = cv2.waitKey(20) & 0xFF
        if k in (ord("q"), 27):
            return 1
        if k == ord("u") and pts:
            pts.pop()
        if k in (13, 10) and len(pts) >= need:
            break
    cv2.imwrite(args.out, vis)
    cv2.destroyAllWindows()

    # normalized coordinates after undistortion (x', y') = (X/Z, Y/Z). Up = y' < 0.
    p = np.array([[pts[0]]], dtype=np.float64)
    xn, yn = cv2.undistortPoints(p, Kmat, dist).reshape(2)
    pitch_down = math.degrees(math.atan(-yn))
    yaw_right = math.degrees(math.atan(xn))
    print(f"basis: {src}")
    print(f"other lens pixel ({pts[0][0]}, {pts[0][1]})  vs principal point Δx={pts[0][0]-cx:+.1f} Δy={cy-pts[0][1]:+.1f} px (up is +)")
    print(f"RESULT: camera downward tilt = {pitch_down:+.2f}°   (config tello_camera_pitch_deg)")
    print(f"        camera left/right skew = {yaw_right:+.2f}°  (+ = other lens is right of image center = optical axis points left of the drone's front)")
    if not args.intrinsics:
        for hf in (55.0, 56.4, 60.0, 70.3):
            f2 = w / (2 * math.tan(math.radians(hf) / 2))
            print(f"        assuming HFOV {hf}°: tilt {math.degrees(math.atan((cy - pts[0][1]) / f2)):+.2f}°")

    if args.tape_mm and args.dist_m:
        (x1, y1), (x2, y2) = pts[1], pts[2]
        q = cv2.undistortPoints(np.array([[[x1, y1]], [[x2, y2]]], dtype=np.float64), Kmat, dist).reshape(2, 2)
        # treat each point as a ray; their spacing on the wall plane (optical-axis distance dist_m) → cross-check scale error
        span_px = math.hypot(x2 - x1, y2 - y1)
        f_tape = span_px * args.dist_m / (args.tape_mm / 1000.0)
        hfov_tape = 2 * math.degrees(math.atan(w / (2 * f_tape)))
        print(f"tape cross-check: two points {span_px:.1f}px, D={args.dist_m}m, L={args.tape_mm}mm → f≈{f_tape:.0f}px, HFOV≈{hfov_tape:.1f}°"
              f"  ({100*(f_tape/fx-1):+.1f}% vs fx {fx:.0f}px in use)")
        print(f"        (normalized spacing {math.hypot(q[1][0]-q[0][0], q[1][1]-q[0][1]):.4f} × D = {math.hypot(q[1][0]-q[0][0], q[1][1]-q[0][1])*args.dist_m*1000:.0f} mm, measured {args.tape_mm:.0f} mm)")
    print(f"saved {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
