# DroneCATS
# Copyright (c) 2026-present NAVER Cloud Corp.
# Apache-2.0

from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path

CHUNK = 1 << 22


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(CHUNK):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def environment_info() -> dict:
    """Library versions that decide whether encoders reproduce the released bytes."""
    import cv2
    import numpy
    import PIL
    from PIL import features

    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "pillow": PIL.__version__,
        "pillow_libjpeg": features.version("jpg"),
        "pillow_zlib": features.version("zlib"),
        "opencv": cv2.__version__,
        "numpy": numpy.__version__,
    }
    build = cv2.getBuildInformation().splitlines()
    info["opencv_codecs"] = [ln.strip() for ln in build
                             if ln.strip().startswith(("JPEG:", "PNG:", "ZLib:"))]
    try:
        import h5py

        info["h5py"] = h5py.__version__
    except ImportError:
        info["h5py"] = None
    return info
