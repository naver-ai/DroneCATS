# DroneCATS
# Copyright (c) 2026-present NAVER Cloud Corp.
# Apache-2.0

"""Image operations; byte identity depends on these exact calls, so do not "improve" them."""
from __future__ import annotations

from io import BytesIO

import cv2
import numpy as np
from PIL import Image, JpegImagePlugin

ACTION_MAX_WIDTH = 640
ACTION_JPEG_QUALITY = 85
NYU_PNG_COMPRESS_LEVEL = 1


def decode_bgr(data: bytes) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("OpenCV could not decode the image")
    return img


def crop_x(img: np.ndarray, x0: int, x1: int) -> np.ndarray:
    """Applied before downscaling, never after."""
    if not (0 <= x0 < x1 <= img.shape[1]):
        raise ValueError(f"crop window {(x0, x1)} outside image width {img.shape[1]}")
    return img[:, x0:x1]


def fit_width_jpeg(img: np.ndarray, max_width: int = ACTION_MAX_WIDTH,
                   quality: int = ACTION_JPEG_QUALITY) -> bytes:
    h, w = img.shape[:2]
    if w > max_width:
        s = max_width / w
        img = cv2.resize(img, (max_width, max(1, round(h * s))), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("OpenCV JPEG encoding failed")
    return buf.tobytes()


def resize_lanczos(data: bytes, width: int, height: int) -> bytes:
    """JPEG reuses the source qtables, subsampling and progressive flag."""
    with Image.open(BytesIO(data)) as im:
        fmt = (im.format or "").upper()
        im.load()
        out = im.resize((width, height), Image.LANCZOS)
        buf = BytesIO()
        if fmt == "JPEG":
            kw: dict = {}
            q = getattr(im, "quantization", None)
            if q:
                kw["qtables"] = q
            try:
                ss = JpegImagePlugin.get_sampling(im)
                if ss in (0, 1, 2):
                    kw["subsampling"] = ss
            except Exception:  # noqa: BLE001 - mirror the original: sampling is best effort
                pass
            if im.info.get("progressive"):
                kw["progressive"] = True
            if out.mode not in ("RGB", "L", "CMYK"):
                out = out.convert("RGB")
            out.save(buf, format="JPEG", **kw)
        elif fmt == "PNG":
            out.save(buf, format="PNG", optimize=True)
        else:
            raise ValueError(f"unsupported source format for resize: {fmt!r}")
        return buf.getvalue()


def nyu_rgb_to_png(rgb: np.ndarray) -> bytes:
    buf = BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG", compress_level=NYU_PNG_COMPRESS_LEVEL)
    return buf.getvalue()


def pixel_digest(data: bytes) -> str:
    import hashlib

    with Image.open(BytesIO(data)) as im:
        rgb = im.convert("RGB")
        h = hashlib.sha256(f"{rgb.width}x{rgb.height}:".encode())
        h.update(rgb.tobytes())
    return h.hexdigest()
