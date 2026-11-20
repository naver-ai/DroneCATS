# DroneCATS
# Copyright (c) 2026-present NAVER Cloud Corp.
# Apache-2.0

"""Command line interface: `ddr fetch-check | reconstruct | verify`."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .ledger import Package, archive_path, archive_source
from .util import sha256_file, write_json


def _fetch_check(args) -> int:
    pkg = Package.load(args.package)
    given = {"nyuv2": args.nyu, "sunrgbd": args.sunrgbd, "tartanair": args.tartanair_root}
    if not any(given.values()):
        print("nothing to check: give --nyu, --sunrgbd and/or --tartanair-root", file=sys.stderr)
        return 2
    ok = True
    for name, meta in sorted(pkg.archives.items()):
        source = archive_source(name, meta)
        if given.get(source) is None:
            print(f"SKIP  {name}: not given (official URL: {meta['url']})")
            continue
        path = archive_path(source, name, given[source])
        if not path.is_file():
            print(f"MISS  {name}: {path} not found (official URL: {meta['url']})")
            ok = False
            continue
        size = path.stat().st_size
        if size != meta["bytes"]:
            print(f"FAIL  {name}: {size} bytes, expected {meta['bytes']}")
            ok = False
            continue
        digest = sha256_file(path)
        good = digest == meta["sha256"]
        ok &= good
        print(f"{'OK   ' if good else 'FAIL '} {name}: sha256 {digest}")
    return 0 if ok else 1


def _reconstruct(args) -> int:
    from .reconstruct import reconstruct

    report = reconstruct(Path(args.package), Path(args.out), nyu_mat=args.nyu,
                         sunrgbd_zip=args.sunrgbd, tartanair_root=args.tartanair_root,
                         work=args.work, jobs=args.jobs,
                         check_archives=not args.skip_archive_check)
    bad = (len(report["derive"]["hash_mismatch"]) + len(report["derive"]["errors"])
           + len(report["pack"]["missing_members"]) + len(report["pack"]["failed_zips"]))
    print(json.dumps({"derive": {k: (len(v) if isinstance(v, list) else v)
                                 for k, v in report["derive"].items()},
                      "pack": {k: (len(v) if isinstance(v, list) else v)
                               for k, v in report["pack"].items()}}, indent=2))
    if bad:
        print("reconstruction finished with problems; run `ddr verify` for details",
              file=sys.stderr)
    return 1 if bad else 0


def _verify(args) -> int:
    from .verify import verify

    report = verify(Path(args.package), Path(args.dataset), mode=args.mode, jobs=args.jobs,
                    allow_unexpected_files=args.allow_unexpected_files)
    if args.report:
        write_json(Path(args.report), report)
    summary = {k: report[k] for k in ("mode", "zip_members", "zips", "problems", "passed")}
    summary["reference_entries"] = {k: (len(v) if isinstance(v, list) else v)
                                    for k, v in report["reference_entries"].items()}
    summary["top_level_files"] = {k: (len(v) if isinstance(v, list) else v)
                                  for k, v in report["top_level_files"].items()}
    summary["unexpected_files"] = len(report["unexpected_files"])
    print(json.dumps(summary, indent=2))
    print("PASS" if report["passed"] else "FAIL")
    return 0 if report["passed"] else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="ddr", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("fetch-check", help="check the official archives against the package")
    p.add_argument("--package", required=True)
    p.add_argument("--nyu", help="path to nyu_depth_v2_labeled.mat")
    p.add_argument("--sunrgbd", help="path to SUNRGBD.zip")
    p.add_argument("--tartanair-root",
                   help="directory holding the TartanAir v2 archives as "
                        "<env>/<Data_diff>/image_lcam_front.zip")
    p.set_defaults(func=_fetch_check)

    p = sub.add_parser("reconstruct", help="rebuild the dataset directory")
    p.add_argument("--package", required=True)
    p.add_argument("--nyu", required=True, help="path to nyu_depth_v2_labeled.mat")
    p.add_argument("--sunrgbd", required=True, help="path to SUNRGBD.zip")
    p.add_argument("--out", required=True, help="output dataset directory")
    p.add_argument("--tartanair-root", required=True,
                   help="directory holding the official TartanAir v2 archives as "
                        "<env>/<Data_diff>/image_lcam_front.zip")
    p.add_argument("--work", default=None,
                   help="scratch directory for derived files (default: <out>.work)")
    p.add_argument("--jobs", type=int, default=0, help="worker processes (default: CPUs available to this process, cgroup quota included)")
    p.add_argument("--skip-archive-check", action="store_true",
                   help="do not verify archive SHA-256 before deriving (size is still checked; "
                        "also allows an extracted TartanAir tree)")
    p.set_defaults(func=_reconstruct)

    p = sub.add_parser("verify", help="compare a dataset directory with the reference hashes")
    p.add_argument("--package", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--mode", choices=("byte", "pixel"), default="byte")
    p.add_argument("--report", default=None, help="write the full JSON report here")
    p.add_argument("--allow-unexpected-files", action="store_true",
                   help="do not fail on files in the dataset directory beyond the release")
    p.add_argument("--jobs", type=int, default=8)
    p.set_defaults(func=_verify)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
