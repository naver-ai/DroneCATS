# DroneCATS
# Copyright (c) 2026-present NAVER Cloud Corp.
# Apache-2.0

"""Send one drone frame and a command to a running serve.sh server and print the action.

    python example_request.py --image frame.jpg --instruction "go to the egg" --hfov 56.4

The camera intrinsics reach the model as the JPEG EXIF focal length (the plugin turns it into the
`<|camera_aux|>` JSON that the transformers example passes as `camera_aux`). The prompt files are the
ones in the model repository; use them as they are, the model was trained on exactly this wording.
"""
import argparse
import base64
import io
import json
import math
import urllib.request
from fractions import Fraction

from huggingface_hub import hf_hub_download
from PIL import Image
from PIL.TiffImagePlugin import IFDRational

REPO = "naver-ai/DroneCATS-Cosmos3-Edge-2B"
MAX_WIDTH, JPEG_QUALITY = 640, 85
ACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["go", "rotate", "think", "finished"]},
        "point": {"type": "array", "items": {"type": "integer", "minimum": 0, "maximum": 1000},
                  "minItems": 2, "maxItems": 2},
        "depth_m": {"type": "number", "minimum": 0, "maximum": 200},
        "angle_deg": {"type": "number", "minimum": -90, "maximum": 90},
    },
    "required": ["action"],
}


def encode_frame(image: Image.Image, k: float) -> tuple[bytes, tuple[int, int]]:
    """JPEG (at most 640 px wide, focal length k = fx / width in EXIF) and its (width, height)."""
    image = image.convert("RGB")
    if image.width > MAX_WIDTH:
        image = image.resize((MAX_WIDTH, int(image.height * MAX_WIDTH / image.width)), Image.BOX)
    kf = Fraction(float(k)).limit_denominator(10**6)
    exif = Image.Exif()
    ifd = exif.get_ifd(0x8769)
    ifd[0x920A] = IFDRational(36 * kf.numerator, kf.denominator)  # FocalLength = 36 * k mm
    res = IFDRational(10 * image.width, 36)                         # FocalPlane{X,Y}Resolution, px/cm
    ifd[0xA20E] = res
    ifd[0xA20F] = res
    ifd[0xA210] = 3                                                 # unit = cm
    buf = io.BytesIO()
    image.save(buf, "JPEG", quality=JPEG_QUALITY, exif=exif)
    return buf.getvalue(), image.size


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:18000/v1")
    ap.add_argument("--model", default="drone-vlm-2b-cosmos3edge")
    ap.add_argument("--image", required=True)
    ap.add_argument("--instruction", required=True)
    ap.add_argument("--hfov", type=float, default=56.4, help="horizontal FOV in degrees (DJI Tello: 56.4)")
    ap.add_argument("--temperature", type=float, default=0.4)
    args = ap.parse_args()

    k = 1 / (2 * math.tan(math.radians(args.hfov) / 2))
    jpeg, (w, h) = encode_frame(Image.open(args.image), k)
    system = open(hf_hub_download(REPO, "prompts/system_prompt.txt"), encoding="utf-8").read()
    user = open(hf_hub_download(REPO, "prompts/user_prompt.txt"), encoding="utf-8").read()
    body = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": system.format(frame_w=w, frame_h=h, quarter_x=w // 4, quarter_y=h // 4)},
            {"role": "user", "content": [
                {"type": "text", "text": user.format(instruction=args.instruction,
                                                     history="(none yet — this is the first step)")},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()}},
            ]},
        ],
        "response_format": {"type": "json_schema", "json_schema": {"name": "vlmpilot_action", "schema": ACTION_SCHEMA}},
        "chat_template_kwargs": {"enable_thinking": False, "skip_reasoning": True},
        "max_tokens": 8192,
        "temperature": args.temperature,
    }
    req = urllib.request.Request(args.url + "/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    answer = json.load(urllib.request.urlopen(req, timeout=60))["choices"][0]["message"]["content"]
    print(answer)


if __name__ == "__main__":
    main()
