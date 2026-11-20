# DroneCATS
# Copyright (c) 2026-present NAVER Cloud Corp.
# Apache-2.0

"""Rebuild the released dataset directory from the package and the official archives."""
from __future__ import annotations

import os
import shutil
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

from .derive import derive_frame
from .ledger import ROWS_DIR, Package, archive_path
from .pack import write_zip
from .sources import make_reader
from .util import environment_info, sha256_bytes, sha256_file, write_json

_READERS: dict = {}


def _init_worker(source_paths: dict[str, str]) -> None:
    for source, path in source_paths.items():
        _READERS[source] = make_reader(source, Path(path))


def _derive_task(source: str, frame: dict, records: list[dict], staging: str) -> list[tuple]:
    reader = _READERS[source]
    results = []
    try:
        outputs = derive_frame(records, lambda: reader.source_bytes(frame))
    except Exception as exc:  # noqa: BLE001 - report per frame, keep going
        return [(r["path"], "error", repr(exc)) for r in records]
    expected = {r["path"]: r["sha256"] for r in records}
    for path, data in outputs.items():
        dst = Path(staging) / path
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_name(dst.name + ".partial")
        tmp.write_bytes(data)
        os.replace(tmp, dst)
        ok = sha256_bytes(data) == expected[path]
        results.append((path, "ok" if ok else "mismatch", None))
    return results


def _cpu_count() -> int:
    """CPUs this process may use: the affinity mask, capped by a cgroup CPU quota (containers).

    os.cpu_count() reports every CPU of the host, which in a container can be far more than its quota.
    """
    try:
        n = len(os.sched_getaffinity(0))
    except AttributeError:
        n = os.cpu_count() or 1
    for quota_file, period_file in (("/sys/fs/cgroup/cpu.max", None),
                                    ("/sys/fs/cgroup/cpu/cpu.cfs_quota_us", "/sys/fs/cgroup/cpu/cpu.cfs_period_us")):
        try:
            fields = Path(quota_file).read_text().split()
            quota = fields[0]
            period = fields[1] if period_file is None else Path(period_file).read_text().strip()
            if quota not in ("max", "-1"):
                n = min(n, max(1, int(quota) // int(period)))
            break
        except (OSError, ValueError, IndexError):
            continue
    return max(1, n)


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


def reconstruct(package: Path, out: Path, *, nyu_mat: Path | None, sunrgbd_zip: Path | None,
                tartanair_root: Path | None = None, work: Path | None = None,
                jobs: int = 0, check_archives: bool = True) -> dict:
    pkg = Package.load(package)
    out = Path(out)
    work = Path(work) if work else out.with_name(out.name + ".work")
    staging = work / "staging"
    jobs = jobs or _cpu_count()
    report: dict = {"environment": environment_info(), "package": str(package)}

    by_source: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for rec in pkg.derivations.values():
        by_source[rec["source"]][rec["frame"]].append(rec)

    source_paths = {"nyuv2": nyu_mat, "sunrgbd": sunrgbd_zip, "tartanair": tartanair_root}
    derive_sources = [s for s in ("nyuv2", "sunrgbd", "tartanair") if by_source.get(s)]
    for s in derive_sources:
        if not (source_paths[s] and Path(source_paths[s]).exists()):
            raise FileNotFoundError(f"{s}: original data not found at {source_paths[s]!r}")
    # A different archive version would silently yield different images; refuse it early.
    for s in derive_sources:
        for name in pkg.required_archives(s):
            _check_archive(pkg, s, name, Path(source_paths[s]), check_sha=check_archives)

    # Staged files whose hash already matches are not derived again.
    tasks = []
    skipped = 0
    for s in derive_sources:
        for frame_id, recs in sorted(by_source.get(s, {}).items()):
            todo = [r for r in recs if not _staged_ok(staging / r["path"], r["sha256"])]
            skipped += len(recs) - len(todo)
            if todo:
                tasks.append((s, pkg.frames[s][frame_id], todo))
    _log(f"derive: {sum(len(t[2]) for t in tasks)} files from {len(tasks)} frames "
         f"({skipped} already staged), {jobs} workers")
    status: dict[str, list] = defaultdict(list)
    t0 = time.time()
    if tasks:
        init_paths = {s: str(source_paths[s]) for s in {t[0] for t in tasks}}
        with ProcessPoolExecutor(jobs, initializer=_init_worker, initargs=(init_paths,)) as ex:
            futs = [ex.submit(_derive_task, s, fr, recs, str(staging)) for s, fr, recs in tasks]
            for n, fut in enumerate(as_completed(futs), 1):
                for path, st, err in fut.result():
                    status[st].append(path if err is None else f"{path}: {err}")
                if n % 500 == 0:
                    _log(f"derive: {n}/{len(futs)} frames")
    report["derive"] = {"derived_ok": len(status["ok"]), "already_staged": skipped,
                        "hash_mismatch": status["mismatch"], "errors": status["error"],
                        "seconds": round(time.time() - t0, 1)}
    _log(f"derive: ok {len(status['ok'])}, mismatch {len(status['mismatch'])}, "
         f"errors {len(status['error'])}")

    out.mkdir(parents=True, exist_ok=True)
    for name in pkg.row_files:
        shutil.copyfile(pkg.root / ROWS_DIR / name, out / name)

    missing: list[str] = []
    failed_zips: list[str] = []

    def get_bytes(path: str) -> bytes:
        staged = staging / path
        if staged.is_file():
            return staged.read_bytes()
        missing.append(path)
        raise FileNotFoundError(path)

    def pack_one(z: dict) -> None:
        # A zip with a missing member is abandoned as `<zip>.partial`, never completed.
        try:
            write_zip(out / "images" / z["zip"], z["members"], get_bytes)
        except FileNotFoundError:
            failed_zips.append(z["zip"])

    t0 = time.time()
    with ThreadPoolExecutor(min(jobs, 8)) as ex:
        list(ex.map(pack_one, pkg.zips))
    report["pack"] = {"zips": len(pkg.zips) - len(failed_zips),
                      "failed_zips": sorted(failed_zips),
                      "missing_members": sorted(set(missing)),
                      "seconds": round(time.time() - t0, 1)}
    _log(f"pack: {len(pkg.zips) - len(failed_zips)} zips written, {len(failed_zips)} abandoned"
         f" ({len(set(missing))} missing member files)")
    write_json(work / "reconstruct_report.json", report)
    return report


def _check_archive(pkg: Package, source: str, name: str, given: Path, *, check_sha: bool) -> None:
    meta = pkg.archives.get(name)
    if meta is None:
        raise ValueError(f"package has no reference for {name} (ledger/archives.json)")
    path = archive_path(source, name, given)
    if not path.is_file():
        # An extracted TartanAir tree cannot be checked against the archive digests.
        if source == "tartanair" and not check_sha:
            _log(f"{name}: archive not found under {given}, reading extracted files unchecked")
            return
        raise FileNotFoundError(f"{path} not found (official URL: {meta.get('url')}); "
                                "an extracted TartanAir tree needs --skip-archive-check")
    if path.stat().st_size != meta["bytes"]:
        raise ValueError(f"{path} has {path.stat().st_size} bytes, expected {meta['bytes']}")
    if check_sha:
        _log(f"checking SHA-256 of {name}")
        if sha256_file(path) != meta["sha256"]:
            raise ValueError(f"{path}: SHA-256 differs from ledger/archives.json "
                             "(different download?); use --skip-archive-check to override")


def _staged_ok(path: Path, sha: str) -> bool:
    return path.is_file() and sha256_file(path) == sha
