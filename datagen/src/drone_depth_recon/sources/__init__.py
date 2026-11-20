# DroneCATS
# Copyright (c) 2026-present NAVER Cloud Corp.
# Apache-2.0

"""Source frame readers; they keep file handles open, so create one per process."""
from __future__ import annotations

from pathlib import Path

from .nyuv2 import NyuReader
from .sunrgbd import SunRgbdReader
from .tartanair import TartanAirReader


def make_reader(source: str, path: Path):
    return {"nyuv2": NyuReader, "sunrgbd": SunRgbdReader, "tartanair": TartanAirReader}[source](path)


__all__ = ["NyuReader", "SunRgbdReader", "TartanAirReader", "make_reader"]
