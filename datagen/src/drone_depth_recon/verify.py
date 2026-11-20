# DroneCATS
# Copyright (c) 2026-present NAVER Cloud Corp.
# Apache-2.0

"""Check a reconstructed dataset directory against the package's reference hashes."""
from __future__ import annotations

import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .exif_camera import camera_exif_tags, expected_camera_exif_tags
from .ledger import Package
from .ops import pixel_digest
from .util import environment_info, sha256_bytes, sha256_file


def _pixel_match(data: bytes, rec: dict) -> bool:
    try:
        if pixel_digest(data) != rec["pixel_sha256"]:
            return False
        cam = next((op for op in rec["ops"] if op["op"] == "camera_exif"), None)
        tags = camera_exif_tags(data)
        if cam is None:
            return not tags
        return tags == expected_camera_exif_tags(cam["fx"], cam["fy"], cam["width"], cam["height"])
    except Exception:  # noqa: BLE001 - undecodable data is simply a failure
        return False


def _check_zip(zrec: dict, path: Path, ref: dict[str, str], pkg: Package, mode: str) -> dict:
    zip_key = f"images/{zrec['zip']}"
    res: dict = {"zip": zrec["zip"], "status": Counter(), "failed": [], "missing": [],
                 "checked": {zip_key}, "order_ok": False, "zip_identical": False}
    expected_names = [m["name"] for m in zrec["members"]]
    member_keys = [f"{zip_key}#{n}" for n in expected_names]
    res["checked"].update(member_keys)
    if not path.is_file():
        res["missing"] = member_keys
        res["status"]["missing"] += len(member_keys)
        return res
    res["zip_identical"] = sha256_file(path) == ref.get(zip_key, zrec["sha256"])
    with zipfile.ZipFile(path) as zf:
        infos = zf.infolist()
        present = {i.filename: i for i in infos}
        res["order_ok"] = [i.filename for i in infos] == expected_names
        extra = sorted(set(present) - set(expected_names))
        if extra:
            res["unexpected_members"] = extra
        for m, key in zip(zrec["members"], member_keys):
            if m["name"] not in present:
                res["missing"].append(key)
                res["status"]["missing"] += 1
                continue
            data = zf.read(present[m["name"]])
            expected = ref.get(key)
            if expected is not None and sha256_bytes(data) == expected:
                res["status"]["byte_exact"] += 1
            elif (expected is not None and mode == "pixel"
                  and _pixel_match(data, pkg.derivations[m["path"]])):
                res["status"]["pixel_exact_only"] += 1
            else:
                res["status"]["failed"] += 1
                res["failed"].append(key if expected is not None else f"{key} (no reference hash)")
    return res


def verify(package: Path, dataset: Path, *, mode: str = "byte", jobs: int = 8,
           allow_unexpected_files: bool = False) -> dict:
    """Pixel mode accepts a byte mismatch only with equal pixels and camera EXIF."""
    if mode not in ("byte", "pixel"):
        raise ValueError(mode)
    pkg = Package.load(package)
    ref = pkg.reference_hashes()
    dataset = Path(dataset)
    problems: list[str] = []
    if not ref:
        problems.append("reference/SHA256SUMS is empty")
    if not pkg.row_files or not pkg.zips:
        problems.append("package lists no row files or no zips")
    if not dataset.is_dir():
        problems.append(f"dataset directory not found: {dataset}")

    checked: set[str] = set()
    top = {"byte_exact": 0, "failed": [], "missing": []}
    for name in pkg.row_files:
        checked.add(name)
        p = dataset / name
        if not p.is_file():
            top["missing"].append(name)
        elif name in ref and sha256_file(p) == ref[name]:
            top["byte_exact"] += 1
        else:
            top["failed"].append(name)

    with ThreadPoolExecutor(jobs) as ex:
        results = list(ex.map(
            lambda z: _check_zip(z, dataset / "images" / z["zip"], ref, pkg, mode), pkg.zips))
    members = Counter()
    for r in results:
        members.update(r["status"])
        checked |= r["checked"]
    unchecked = sorted(set(ref) - checked)
    if unchecked:
        problems.append(f"{len(unchecked)} reference entries are not covered by the package "
                        "layout")
    expected_files = set(pkg.row_files) | {f"images/{z['zip']}" for z in pkg.zips}
    actual = ({p.relative_to(dataset).as_posix() for p in dataset.rglob("*") if p.is_file()}
              if dataset.is_dir() else set())
    report = {
        "mode": mode,
        "top_level_files": {"expected": len(pkg.row_files), **top},
        "zip_members": {"expected": sum(len(z["members"]) for z in pkg.zips),
                        "byte_exact": members["byte_exact"],
                        "pixel_exact_only": members["pixel_exact_only"],
                        "failed": members["failed"], "missing": members["missing"]},
        "zips": {"expected": len(pkg.zips),
                 "byte_identical": sum(1 for r in results if r["zip_identical"]),
                 "member_order_ok": sum(1 for r in results if r["order_ok"])},
        "reference_entries": {"total": len(ref), "checked": len(set(ref) & checked),
                              "unchecked": unchecked[:100]},
        "failed_members": sorted(k for r in results for k in r["failed"]),
        "missing_members": sorted(k for r in results for k in r["missing"]),
        "unexpected_members": {r["zip"]: r["unexpected_members"]
                               for r in results if r.get("unexpected_members")},
        "unexpected_files": sorted(actual - expected_files),
        "problems": problems,
        "environment": environment_info(),
    }
    order_bad = [r["zip"] for r in results if r["order_ok"] is False
                 and (dataset / "images" / r["zip"]).is_file()]
    report["passed"] = (
        not problems
        and not top["failed"] and not top["missing"]
        and members["failed"] == 0 and members["missing"] == 0
        and not report["unexpected_members"]
        and not order_bad
        and (allow_unexpected_files or not report["unexpected_files"])
    )
    return report
