#!/usr/bin/env python3
# DroneCATS
# Copyright (c) 2026-present NAVER Cloud Corp.
# Apache-2.0

"""Collect checkerboard frames for Tello camera calibration — no takeoff.

Receives frames through the same path as the flight code (djitellopy streamon → ResilientFrameRead, 960x720 RGB)
and saves automatically when the checkerboard (9x6 inner corners, tools/tello/tello_chessboard_9x6_25mm_A4.pdf) is visible
and the vehicle is still. Saved frames are solved with tools/tello/tello_calib_solve.py.

Usage:  uv run python tools/tello/tello_calib_capture.py [--target 20] [--out outputs/tello/calib]
  Hold the vehicle in hand (propellers stopped) and view the board in many poses: frontal / tilted left-right / tilted up-down /
  board in each of the four screen corners / near and far. Hold each pose for about 1 second and it is saved.
  Keys: space = force save, q/esc = quit. Where no window can open (headless OpenCV), proceeds with terminal output only.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))

PATTERN = (9, 6)          # inner corners (cols, rows)
EXPECT_WH = (960, 720)    # stream size used in flight. Warn if different (calibration values are tied to resolution)


def find_corners(gray: np.ndarray) -> np.ndarray | None:
    """SB detector (sub-pixel accurate, robust to distortion and tilt). (N,2) float32 or None."""
    flags = cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
    ok, corners = cv2.findChessboardCornersSB(gray, PATTERN, flags=flags)
    if not ok or corners is None:
        return None
    return corners.reshape(-1, 2).astype(np.float32)


def mean_shift(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.linalg.norm(a - b, axis=1)))


def open_tello():
    from djitellopy import Tello
    from vlmpilot.controller.real.tello_controller import ResilientFrameRead

    t = Tello()
    t.connect()
    try:
        raw = str(t.query_serial_number()).strip()
    except Exception:  # noqa: BLE001 — old firmware does not support sn?
        raw = ""
    sn = raw if re.fullmatch(r"[0-9A-Za-z]{8,}", raw) else "unknown"   # unsupported firmware returns 'ok'/'unknown command'
    meta = {"serial": sn, "battery": t.get_battery(), "temp_c": t.get_temperature(),
            "sdk": None, "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    try:
        sdk = str(t.query_sdk_version()).strip()
        meta["sdk"] = sdk if re.fullmatch(r"\d+", sdk) else None
    except Exception:  # noqa: BLE001
        pass
    print(f"[TELLO] serial={sn}  battery={meta['battery']}%  temp={meta['temp_c']}  sdk={meta['sdk']}")
    t.streamon()
    reader = ResilientFrameRead(t.get_udp_video_address()).start()
    first = reader.wait_for_frame(20.0)
    if first is None:
        raise SystemExit("no frame — check Wi-Fi/drone power/whether an app holds the stream")
    return t, reader, meta


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=None, help="output folder. default outputs/tello/calib/<time>")
    ap.add_argument("--target", type=int, default=20, help="target number of images (15+ recommended)")
    ap.add_argument("--min-move", type=float, default=40.0, help="save a new image only if the mean corner shift (px) from every saved pose exceeds this")
    ap.add_argument("--still", type=float, default=1.5, help="'still' when the mean corner shift (px) between consecutive frames is below this")
    ap.add_argument("--still-frames", type=int, default=5, help="number of consecutive frames for the still verdict")
    args = ap.parse_args()

    out = args.out or os.path.join("outputs", "tello", "calib", time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(out, exist_ok=True)

    t, reader, meta = open_tello()
    with open(os.path.join(out, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    show = True
    saved: list[np.ndarray] = []
    prev: np.ndarray | None = None
    still = 0
    warned_size = False
    flash_until = 0.0
    try:
        while len(saved) < args.target:
            rgb = reader.frame
            if rgb is None:
                time.sleep(0.02)
                continue
            frame = cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR)
            h, w = frame.shape[:2]
            if (w, h) != EXPECT_WH and not warned_size:
                print(f"[WARN] frame {w}x{h} — differs from the flight stream {EXPECT_WH}. A calibration at this size is unusable in flight.")
                warned_size = True
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners = find_corners(gray)

            status = "no board"
            if corners is not None:
                still = still + 1 if (prev is not None and mean_shift(corners, prev) < args.still) else 0
                novel = all(mean_shift(corners, s) > args.min_move for s in saved)
                status = f"board OK  still {min(still, args.still_frames)}/{args.still_frames}  {'new pose' if novel else 'pose already captured — move it'}"
                if still >= args.still_frames and novel:
                    n = len(saved) + 1
                    cv2.imwrite(os.path.join(out, f"img_{n:02d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
                    np.save(os.path.join(out, f"img_{n:02d}_corners.npy"), corners)
                    saved.append(corners)
                    still = 0
                    flash_until = time.time() + 0.4
                    print(f"[SAVE] {n}/{args.target}  {out}/img_{n:02d}.jpg")
            else:
                still = 0
            prev = corners

            if show:
                vis = frame.copy()
                if corners is not None:
                    cv2.drawChessboardCorners(vis, PATTERN, corners.reshape(-1, 1, 2), True)
                for s in saved:                                       # area covered so far
                    cv2.polylines(vis, [cv2.convexHull(s).astype(np.int32)], True, (80, 80, 80), 1)
                if time.time() < flash_until:
                    cv2.rectangle(vis, (0, 0), (w - 1, h - 1), (0, 255, 0), 12)
                cv2.putText(vis, f"saved {len(saved)}/{args.target}  {w}x{h}", (12, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.putText(vis, status, (12, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                try:
                    cv2.imshow("tello calib  [space=save  q=quit]", vis)
                    k = cv2.waitKey(1) & 0xFF
                except cv2.error:
                    show = False
                    print("[INFO] cannot show a window (headless OpenCV) — auto-save only")
                    continue
                if k in (ord("q"), 27):
                    break
                if k == ord(" ") and corners is not None:
                    n = len(saved) + 1
                    cv2.imwrite(os.path.join(out, f"img_{n:02d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
                    np.save(os.path.join(out, f"img_{n:02d}_corners.npy"), corners)
                    saved.append(corners)
                    flash_until = time.time() + 0.4
                    print(f"[SAVE:manual] {n}/{args.target}")
            else:
                time.sleep(0.03)
    finally:
        try:
            reader.stop()
            t.streamoff()
        except Exception:
            pass
        if show:
            cv2.destroyAllWindows()

    if saved:
        cov = np.zeros((EXPECT_WH[1], EXPECT_WH[0], 3), np.uint8)
        for s in saved:
            cv2.polylines(cov, [cv2.convexHull(s).astype(np.int32)], True, (0, 255, 0), 1)
        cv2.imwrite(os.path.join(out, "coverage.png"), cov)
    print(f"[DONE] {len(saved)} images → {out}")
    print(f"       next: uv run python tools/tello/tello_calib_solve.py {out}")
    return 0 if len(saved) >= 10 else 2


if __name__ == "__main__":
    raise SystemExit(main())
