#!/usr/bin/env python3
# DroneCATS
# Copyright (c) 2026-present NAVER Cloud Corp.
# Apache-2.0

"""VLM connectivity check — runs a single inference on one image, without the simulator, and checks the response.

Verifies that the keys/endpoints in .env work through the same path as the flight pipeline
(build_backend → VlmPolicy prompt). The prompt is built exactly as the policy module builds it for real flights.

Usage (from vlmpilot/):
    uv run python tools/probe_vlm.py --config configs/airsim/single-drone/airsim_gemini.yaml
    uv run python tools/probe_vlm.py --config configs/airsim/single-drone/airsim_gemini38_or.yaml --image frame.png \
        --instruction "fly to the red car"

Without --image, a synthetic frame (1280x720) with a red rectangle on a gray background is used.

Exit codes: 0 OK · 1 argument/config error · 3 request failed (no response) · 4 response received but JSON parsing failed
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from vlmpilot.policy.camera import PinholeCamera  # noqa: E402
from vlmpilot.policy.policy import VlmPolicy  # noqa: E402
from vlmpilot.policy.vlm_backends import VlmConfigError, build_backend, extract_action_json  # noqa: E402

DEFAULT_INSTRUCTION = "fly to the red car"


def synthetic_frame(width: int = 1280, height: int = 720) -> np.ndarray:
    """Gray ground + sky, with one red box right of center."""
    img = np.full((height, width, 3), 120, dtype=np.uint8)
    img[: height // 2] = (200, 170, 120)  # sky (BGR)
    cv2.rectangle(img, (int(width * 0.62), int(height * 0.55)), (int(width * 0.72), int(height * 0.68)),
                  (40, 40, 220), thickness=-1)
    return img


def main() -> int:
    ap = argparse.ArgumentParser(description="vlmpilot VLM connection check (one inference)")
    ap.add_argument("--config", default="configs/airsim/single-drone/airsim_gemini.yaml", help="YAML to read provider/model from")
    ap.add_argument("--image", default=None, help="input frame (synthetic frame if omitted)")
    ap.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    ap.add_argument("--think", action="store_true", help="send one request in think mode")
    args = ap.parse_args()

    try:
        with open(args.config, encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
    except OSError as e:
        print(f"could not read config: {e}", file=sys.stderr)
        return 1

    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            print(f"could not read image: {args.image}", file=sys.stderr)
            return 1
    else:
        frame = synthetic_frame()
    h, w = frame.shape[:2]

    try:
        backend = build_backend(config)
    except VlmConfigError as e:
        print(f"backend config error: {e}", file=sys.stderr)
        return 1
    camera = PinholeCamera(w, h, float(config.get("fov_horizontal", 90)))
    policy = VlmPolicy.from_config(config, camera, backend)
    prompt = policy.build_prompt(args.instruction)

    print(f"provider={backend.provider} model={backend.model} image={w}x{h} think={args.think}")
    raw = backend.generate(prompt, frame, think=args.think)
    if raw is None:
        print("request failed — no reply (check key/endpoint/timeout)", file=sys.stderr)
        return 3
    print("--- raw ---")
    print(raw)
    cleaned = extract_action_json(raw)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as e:
        print(f"--- JSON parse failed: {e}", file=sys.stderr)
        return 4
    print("--- parsed ---")
    print(json.dumps(parsed, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
