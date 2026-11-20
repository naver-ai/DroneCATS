# DroneCATS
# Copyright (c) 2026-present NAVER Cloud Corp.
# Apache-2.0

"""Write image zips as in `ledger/zip_layout.jsonl` (same order and attributes, same bytes)."""
from __future__ import annotations

import os
import zipfile
from pathlib import Path
from typing import Callable

# Attributes shared by every member of the released zips.
MEMBER_DEFAULTS = {
    "compress_type": zipfile.ZIP_STORED,
    "external_attr": 0o600 << 16,
    "internal_attr": 0,
    "create_system": 3,
    "create_version": 20,
    "extract_version": 20,
    "flag_bits": 0,
}


def member_info(name: str, date_time: list[int] | tuple[int, ...]) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=tuple(date_time))
    info.compress_type = MEMBER_DEFAULTS["compress_type"]
    info.external_attr = MEMBER_DEFAULTS["external_attr"]
    info.internal_attr = MEMBER_DEFAULTS["internal_attr"]
    info.create_system = MEMBER_DEFAULTS["create_system"]
    info.create_version = MEMBER_DEFAULTS["create_version"]
    info.extract_version = MEMBER_DEFAULTS["extract_version"]
    return info


def write_zip(out_path: Path, members: list[dict], get_bytes: Callable[[str], bytes]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(out_path.name + ".partial")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_STORED, allowZip64=False) as zf:
        for m in members:
            zf.writestr(member_info(m["name"], m["date_time"]), get_bytes(m["path"]))
    os.replace(tmp, out_path)


def describe_member(info: zipfile.ZipInfo) -> dict:
    return {k: getattr(info, k) for k in MEMBER_DEFAULTS} | {"extra": info.extra,
                                                             "comment": info.comment}
