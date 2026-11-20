# DroneCATS
# Copyright (c) 2026-present NAVER Cloud Corp.
# Apache-2.0

"""NYU Depth v2 labeled set (`nyu_depth_v2_labeled.mat`, MATLAB v7.3 / HDF5)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..ops import nyu_rgb_to_png

ARCHIVE_NAME = "nyu_depth_v2_labeled.mat"


class NyuReader:
    def __init__(self, mat_path: Path | str):
        import h5py

        self._f = h5py.File(str(mat_path), "r")
        self._images = self._f["images"]

    def rgb(self, mat_index: int) -> np.ndarray:
        # HDF5 stores MATLAB arrays transposed: (3, W, H) -> (H, W, 3).
        return np.transpose(self._images[mat_index], (2, 1, 0))

    def source_bytes(self, frame: dict) -> bytes:
        return nyu_rgb_to_png(self.rgb(int(frame["mat_index"])))

    def close(self) -> None:
        self._f.close()
