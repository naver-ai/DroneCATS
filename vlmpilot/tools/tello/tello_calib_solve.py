#!/usr/bin/env python3
# DroneCATS
# Copyright (c) 2026-present NAVER Cloud Corp.
# Apache-2.0

"""Checkerboard frame folder → Tello camera intrinsics (fx, fy, cx, cy, distortion) + field of view.

Usage:  uv run python tools/tello/tello_calib_solve.py outputs/tello/calib/<folder> [--square-mm 25]
  Results go to the terminal and <folder>/intrinsics.json. If reprojection RMS exceeds 0.5px, drop blurry frames with --drop and rerun.

The pinhole horizontal FOV (the value to use as-is for config fov_horizontal) is computed as 2·atan(W / 2fx).
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import time

import cv2
import numpy as np

PATTERN = (9, 6)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder")
    ap.add_argument("--square-mm", type=float, default=25.0, help="measured size of one printed square (mm), checked with a ruler")
    ap.add_argument("--drop", default="", help="image numbers to exclude, e.g. 3,7")
    ap.add_argument("--fix-aspect", action="store_true", help="force fx=fy (assume square pixels)")
    ap.add_argument("--free-k3", action="store_true", help="also estimate the 3rd-order distortion term k3 (fixed by default — avoids local minima)")
    args = ap.parse_args()

    drop = {int(x) for x in args.drop.split(",") if x.strip()}
    paths = sorted(glob.glob(os.path.join(args.folder, "img_*.jpg")))
    paths = [p for p in paths if int(os.path.basename(p)[4:6]) not in drop]
    if len(paths) < 6:
        print(f"{len(paths)} images — minimum 6, 15+ recommended"); return 1

    objp = np.zeros((PATTERN[0] * PATTERN[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:PATTERN[0], 0:PATTERN[1]].T.reshape(-1, 2) * (args.square_mm / 1000.0)  # in meters

    obj_pts, img_pts, used, size = [], [], [], None
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            continue
        h, w = img.shape[:2]
        if size is None:
            size = (w, h)
        elif size != (w, h):
            print(f"[SKIP] {p}: size {w}x{h} ≠ {size}"); continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ok, c = cv2.findChessboardCornersSB(gray, PATTERN, flags=cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY)
        if not ok:
            print(f"[SKIP] {p}: corner detection failed"); continue
        obj_pts.append(objp); img_pts.append(c.reshape(-1, 1, 2).astype(np.float32)); used.append(p)
    if len(used) < 6:
        print(f"{len(used)} images detected — not enough"); return 1

    # k3 is fixed by default: freeing k3 on an ordinary lens falls into a bad local minimum where f and k2/k3 compensate each other
    # (measured on unit 2: k3 free → fx 1300, k3 -3.6, RMS 0.51 / k3 fixed → fx 897, RMS 0.31).
    flags = 0 if args.free_k3 else cv2.CALIB_FIX_K3
    if args.fix_aspect:
        flags |= cv2.CALIB_FIX_ASPECT_RATIO
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(obj_pts, img_pts, size, None, None, flags=flags)
    # Safeguard: if f differs by 5% or more from the distortion-free model, the distortion terms dragged f — warn
    _, K_nd, _, _, _ = cv2.calibrateCamera(
        obj_pts, img_pts, size, None, None,
        flags=cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3 | cv2.CALIB_ZERO_TANGENT_DIST)
    f_ratio = float(K[0, 0] / K_nd[0, 0])

    per = []
    for i, (o, im, r, t) in enumerate(zip(obj_pts, img_pts, rvecs, tvecs)):
        proj, _ = cv2.projectPoints(o, r, t, K, dist)
        per.append(float(np.sqrt(np.mean(np.sum((proj - im) ** 2, axis=2)))))

    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    W, H = size
    hfov = 2 * math.degrees(math.atan(W / (2 * fx)))
    vfov = 2 * math.degrees(math.atan(H / (2 * fy)))
    dfov = 2 * math.degrees(math.atan(math.hypot(W / (2 * fx), H / (2 * fy))))

    print(f"{len(used)} images, {W}x{H}, square {args.square_mm} mm")
    print(f"reprojection RMS {rms:.3f} px   (per image: " + " ".join(f"{e:.2f}" for e in per) + ")")
    print(f"fx={fx:.1f}  fy={fy:.1f}  (fy/fx={fy/fx:.4f})   cx={cx:.1f}  cy={cy:.1f}  (vs center {cx-W/2:+.1f}, {cy-H/2:+.1f} px)")
    print(f"distortion k1={dist[0][0]:+.4f} k2={dist[0][1]:+.4f} p1={dist[0][2]:+.5f} p2={dist[0][3]:+.5f} k3={dist[0][4]:+.4f}")
    print(f"FOV  horizontal {hfov:.1f}°  vertical {vfov:.1f}°  diagonal {dfov:.1f}°")
    print(f"cross-check  distortion-free model fx={K_nd[0,0]:.1f} (ratio {f_ratio:.3f})" + ("  [WARN] differs by 5% or more — too little pose variety or a local minimum" if abs(f_ratio - 1) > 0.05 else "  OK"))
    print(f"→ config fov_horizontal: {hfov:.1f}   (for the current code path)")
    worst = int(np.argmax(per)) + 1
    if rms > 0.5:
        print(f"[WARN] RMS above 0.5px — try dropping the worst image #{int(os.path.basename(used[worst-1])[4:6])} ({per[worst-1]:.2f}px) with --drop")

    meta_path = os.path.join(args.folder, "meta.json")
    meta = json.load(open(meta_path, encoding="utf-8")) if os.path.exists(meta_path) else {}
    result = {
        "serial": meta.get("serial", "unknown"), "capture_meta": meta,
        "date": time.strftime("%Y-%m-%d %H:%M"), "width": W, "height": H, "n_images": len(used),
        "square_mm": args.square_mm, "rms_px": round(float(rms), 4),
        "fx": round(float(fx), 2), "fy": round(float(fy), 2), "cx": round(float(cx), 2), "cy": round(float(cy), 2),
        "dist": [round(float(x), 6) for x in dist.ravel()],
        "hfov_deg": round(hfov, 2), "vfov_deg": round(vfov, 2), "dfov_deg": round(dfov, 2),
        "fov_horizontal_for_config": round(hfov, 1), "images": [os.path.basename(p) for p in used],
        "flags": "fix_k3" + ("+fix_aspect" if args.fix_aspect else "") if not args.free_k3 else "free_k3",
        "fx_no_distortion_model": round(float(K_nd[0, 0]), 2),
    }
    out = os.path.join(args.folder, "intrinsics.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
