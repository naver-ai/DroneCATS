# DroneCATS
# Copyright (c) 2026-present NAVER Cloud Corp.
# Apache-2.0

"""Derivation ledger: derivation path parser, operation lists and package loader."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

SOURCES = ("nyuv2", "sunrgbd", "tartanair")
KINDS = ("orig", "action_base", "crop", "rs")
RS_BASES = ("images", "images_action_base", "images_crop")

# images/<source>/<kind>/[<W>x<H>/<base>/]<frame>[__<cell>__<x0>_<x1>]__<camera>.<ext>,
# camera = fx<fx>_fy<fy>_<W>x<H> | nofx. fx, fy stay strings: the EXIF writer used that spelling.
_NAME = re.compile(
    r"^(?P<frame>.+?)"
    r"(__(?P<cell>[a-z_]+)__(?P<x0>\d+)_(?P<x1>\d+))?"
    r"__(fx(?P<fx>[0-9.]+)_fy(?P<fy>[0-9.]+)_(?P<w>\d+)x(?P<h>\d+)|(?P<nofx>nofx))$"
)


@dataclass(frozen=True)
class DerivationPath:
    path: str
    source: str
    kind: str
    frame: str
    ext: str
    rs_size: tuple[int, int] | None
    rs_base: str | None
    cell: str | None
    crop: tuple[int, int] | None
    camera: tuple[str, str, int, int] | None  # (fx, fy, W, H)


def parse_path(path: str) -> DerivationPath:
    parts = path.split("/")
    if len(parts) not in (4, 6) or parts[0] != "images":
        raise ValueError(f"unexpected derivation path layout: {path}")
    source, kind = parts[1], parts[2]
    if source not in SOURCES or kind not in KINDS:
        raise ValueError(f"unknown source/kind in {path}")
    rs_size = rs_base = None
    if kind == "rs":
        if len(parts) != 6:
            raise ValueError(f"rs path needs <WxH>/<base>/: {path}")
        w, h = parts[3].split("x")
        rs_size, rs_base = (int(w), int(h)), parts[4]
        if rs_base not in RS_BASES:
            raise ValueError(f"unknown rs base in {path}")
    elif len(parts) != 4:
        raise ValueError(f"only rs paths have subdirectories: {path}")
    stem, _, ext = parts[-1].rpartition(".")
    m = _NAME.match(stem)
    if not m:
        raise ValueError(f"unparseable file name: {path}")
    crop = (int(m["x0"]), int(m["x1"])) if m["cell"] else None
    has_crop = kind == "crop" or rs_base == "images_crop"
    if has_crop != (crop is not None):
        raise ValueError(f"crop window present/absent inconsistently: {path}")
    camera = None if m["nofx"] else (m["fx"], m["fy"], int(m["w"]), int(m["h"]))
    if kind == "rs" and (camera is None or (camera[2], camera[3]) != rs_size):
        raise ValueError(f"rs file must carry camera EXIF of its own size: {path}")
    return DerivationPath(path=path, source=source, kind=kind, frame=m["frame"], ext=ext,
                          rs_size=rs_size, rs_base=rs_base, cell=m["cell"], crop=crop,
                          camera=camera)


def ops_for_path(path: str) -> list[dict]:
    d =parse_path(path)
    ops: list[dict] = [{"op": "source"}]
    base = d.kind if d.kind != "rs" else {"images": "orig", "images_action_base": "action_base",
                                          "images_crop": "crop"}[d.rs_base]
    if base == "crop":
        ops.append({"op": "crop_x", "cell": d.cell, "x0": d.crop[0], "x1": d.crop[1]})
    if base in ("action_base", "crop"):
        ops.append({"op": "fit_width_jpeg", "max_width": 640, "quality": 85})
    if d.kind == "rs":
        ops.append({"op": "resize_lanczos", "width": d.rs_size[0], "height": d.rs_size[1]})
    if d.camera is not None:
        fx, fy, w, h = d.camera
        ops.append({"op": "camera_exif", "fx": fx, "fy": fy, "width": w, "height": h})
    return ops


# ---- package files ---------------------------------------------------------------------

LEDGER_DIR = "ledger"
ROWS_DIR = "rows"
REFERENCE_DIR = "reference"
DERIVATIONS = f"{LEDGER_DIR}/derivations.jsonl"
ZIP_LAYOUT = f"{LEDGER_DIR}/zip_layout.jsonl"
ARCHIVES = f"{LEDGER_DIR}/archives.json"
SHA256SUMS = f"{REFERENCE_DIR}/SHA256SUMS"

# TartanAir archives are keyed by their path under the TartanAir v2 root instead.
ARCHIVE_NAMES = {"nyuv2": "nyu_depth_v2_labeled.mat", "sunrgbd": "SUNRGBD.zip"}


def archive_path(source: str, name: str, given: Path | str) -> Path:
    return Path(given) / name if source == "tartanair" else Path(given)


def archive_source(name: str, meta: dict) -> str:
    for source, fixed in ARCHIVE_NAMES.items():
        if name == fixed:
            return source
    return meta.get("source", "tartanair")


def frames_file(source: str) -> str:
    return f"{LEDGER_DIR}/frames_{source}.jsonl"


def read_jsonl(path: Path) -> Iterator[dict]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


@dataclass
class Package:
    root: Path
    derivations: dict[str, dict]         # derivation path -> record
    zips: list[dict]                     # zip layout records, in file order
    frames: dict[str, dict[str, dict]]   # source -> frame id -> record
    archives: dict
    row_files: list[str]

    @classmethod
    def load(cls, root: Path | str) -> "Package":
        root = Path(root)
        derivations = {r["path"]: r for r in read_jsonl(root / DERIVATIONS)}
        zips = list(read_jsonl(root / ZIP_LAYOUT))
        frames = {s: {r["frame"]: r for r in read_jsonl(root / frames_file(s))} for s in SOURCES}
        archives = json.loads((root / ARCHIVES).read_text())
        row_files = sorted(p.name for p in (root / ROWS_DIR).iterdir() if p.is_file())
        return cls(root, derivations, zips, frames, archives, row_files)

    def required_archives(self, source: str) -> list[str]:
        frames = {r["frame"] for r in self.derivations.values() if r["source"] == source}
        if not frames:
            return []
        if source in ARCHIVE_NAMES:
            return [ARCHIVE_NAMES[source]]
        return sorted({self.frames[source][f]["archive"] for f in frames})

    def reference_hashes(self) -> dict[str, str]:
        out = {}
        for line in (self.root / SHA256SUMS).read_text().splitlines():
            if line.strip():
                digest, name = line.split(None, 1)
                out[name.lstrip("*")] = digest
        return out
