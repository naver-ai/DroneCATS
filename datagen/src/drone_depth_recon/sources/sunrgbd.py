# DroneCATS
# Copyright (c) 2026-present NAVER Cloud Corp.
# Apache-2.0

"""SUN RGB-D (`SUNRGBD.zip`). Frames are the archive's JPEG members, byte for byte."""
from __future__ import annotations

import zipfile
from pathlib import Path

ARCHIVE_NAME = "SUNRGBD.zip"


class SunRgbdReader:
    def __init__(self, zip_path: Path | str):
        self._zf = zipfile.ZipFile(str(zip_path))

    def source_bytes(self, frame: dict) -> bytes:
        # Member path from the ledger: in some subsets (kv2) it cannot be derived from the id.
        return self._zf.read(frame["archive_member"])

    def close(self) -> None:
        self._zf.close()
