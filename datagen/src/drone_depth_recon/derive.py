# DroneCATS
# Copyright (c) 2026-present NAVER Cloud Corp.
# Apache-2.0

"""Run ledger operation lists per source frame, memoising shared operation prefixes."""
from __future__ import annotations

import json
from typing import Callable

import numpy as np

from . import ops as imgops
from .exif_camera import add_camera_exif


def _key(prefix: list[dict]) -> str:
    return json.dumps(prefix, sort_keys=True)


def run_ops(op_list: list[dict], source: Callable[[], bytes],
            cache: dict[str, object] | None = None) -> bytes:
    cache = {} if cache is None else cache
    state: object = None
    for i, op in enumerate(op_list):
        key = _key(op_list[: i + 1])
        if key in cache:
            state = cache[key]
            continue
        name = op["op"]
        if name == "source":
            if i != 0:
                raise ValueError("'source' must be the first operation")
            state = source()
        elif name == "crop_x":
            state = imgops.crop_x(_as_array(state, cache, op_list[:i]), op["x0"], op["x1"])
        elif name == "fit_width_jpeg":
            state = imgops.fit_width_jpeg(_as_array(state, cache, op_list[:i]),
                                          op["max_width"], op["quality"])
        elif name == "resize_lanczos":
            state = imgops.resize_lanczos(_as_bytes(state), op["width"], op["height"])
        elif name == "camera_exif":
            state = add_camera_exif(_as_bytes(state), op["fx"], op["fy"], op["width"], op["height"])
        else:
            raise ValueError(f"unknown operation {name!r}")
        cache[key] = state
    return _as_bytes(state)


def _as_array(state: object, cache: dict, prefix: list[dict]) -> np.ndarray:
    if isinstance(state, np.ndarray):
        return state
    key = _key(prefix) + "#decoded"
    if key not in cache:
        cache[key] = imgops.decode_bgr(_as_bytes(state))
    return cache[key]  # type: ignore[return-value]


def _as_bytes(state: object) -> bytes:
    if not isinstance(state, (bytes, bytearray)):
        raise ValueError("operation expects encoded image bytes; the op list is malformed")
    return bytes(state)


def derive_frame(records: list[dict], source: Callable[[], bytes]) -> dict[str, bytes]:
    cache: dict[str, object] = {}
    src_cache: list[bytes] = []

    def once() -> bytes:
        if not src_cache:
            src_cache.append(source())
        return src_cache[0]

    return {r["path"]: run_ops(r["ops"], once, cache) for r in records}
