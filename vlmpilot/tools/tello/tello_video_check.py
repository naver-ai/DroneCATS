#!/usr/bin/env python3
# DroneCATS
# Copyright (c) 2026-present NAVER Cloud Corp.
# Apache-2.0

"""Check numerically whether Tello SDK video arrives intact. No takeoff.

  uv run python tools/tello/tello_video_check.py                 # default 30 s
  uv run python tools/tello/tello_video_check.py --seconds 20 --config configs/real/tello_gemini38_or.yaml

Receives frames through the real flight code path (TelloController.connect → ResilientFrameRead → frame())
and measures what % of the 720 vertical rows contain 'real content'. A row whose mean Laplacian
exceeds 1.0 counts as having content (flat green/gray areas are near 0).

Verdict
  80% or more  → normal. OK to fly.
  10~30%    → the tail of the keyframe does not arrive: large UDP packets are lost on the Wi-Fi link
              (seen on a Mac). Not a code problem.
  no frames → the stream does not attach at all. Check Wi-Fi / vehicle power / phone app holding the connection.

Saves the most intact frame to outputs/tello/video_check.jpg.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import cv2
import numpy as np
import yaml
from dotenv import load_dotenv

# so the vlmpilot package is found even when run from tools/
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))

from vlmpilot.controller.real.tello_controller import TelloController  # noqa: E402

FRAME_H = 720
DETAIL_THRESHOLD = 1.0


def live_rows(bgr: np.ndarray) -> tuple[int, int]:
    """(number of rows with content, y where content cuts off)."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    detail = np.abs(cv2.Laplacian(gray, cv2.CV_32F)).mean(axis=1)
    live = detail > DETAIL_THRESHOLD
    last = int(np.max(np.where(live)[0])) if live.any() else -1
    return int(live.sum()), last


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/real/tello_gemini38_or.yaml")
    ap.add_argument("--seconds", type=float, default=30.0, help="time spent collecting frames")
    ap.add_argument("--out", default="outputs/tello/video_check.jpg")
    args = ap.parse_args()

    load_dotenv()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    ctrl = TelloController(config=cfg, run_dir=os.path.dirname(args.out) or ".", policy=object())
    ctrl.connect()
    reader = ctrl._frame_reader

    best, best_rows, best_cut = None, -1, -1
    t0 = time.monotonic()
    while time.monotonic() - t0 < args.seconds:
        frame = ctrl.frame()
        if frame is not None:
            rows, cut = live_rows(frame)
            if rows > best_rows:
                best, best_rows, best_cut = frame.copy(), rows, cut
        time.sleep(0.3)

    print(f"decoded={getattr(reader, 'frames_decoded', '?')} "
          f"skipped={getattr(reader, 'frames_skipped', '?')} "
          f"reopens={getattr(reader, 'reopens', '?')} "
          f"errors={getattr(reader, 'decode_errors', '?')}")

    if best is None:
        print("RESULT: no frame")
        code = 2
    else:
        pct = best_rows / best.shape[0] * 100
        print(f"RESULT: {best_rows}px of {best.shape[0]}px rows have content = {pct:.0f}%")
        print(f"        content cuts off at y={best_cut} ({best_cut / best.shape[0] * 100:.0f}%)")
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        cv2.imwrite(args.out, best)
        print(f"        saved {args.out}")
        code = 0 if pct >= 80 else 1

    print(f"battery={ctrl._safe_query('get_battery')}% temp={ctrl._safe_query('get_temperature')}")
    stop = getattr(reader, "stop", None)
    if callable(stop):
        stop()
    try:
        ctrl.tello.streamoff()
    except Exception:  # noqa: BLE001 — shutdown cleanup is best-effort
        pass
    return code


if __name__ == "__main__":
    sys.exit(main())
