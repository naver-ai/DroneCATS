# DroneCATS
# Copyright (c) 2026-present NAVER Cloud Corp.
# Apache-2.0

"""TartanAir v2 front-camera frames, from the official archives or an extracted tree."""
from __future__ import annotations

import zipfile
from pathlib import Path


class TartanAirReader:
    def __init__(self, root: Path | str):
        self._root = Path(root)
        self._zips: dict[str, zipfile.ZipFile] = {}

    def source_bytes(self, frame: dict) -> bytes:
        extracted = self._root / frame["archive_member"]
        if extracted.is_file():
            return extracted.read_bytes()
        archive = frame["archive"]
        zf = self._zips.get(archive)
        if zf is None:
            zf = self._zips[archive] = zipfile.ZipFile(self._root / archive)
        return zf.read(frame["archive_member"])

    def close(self) -> None:
        for zf in self._zips.values():
            zf.close()
