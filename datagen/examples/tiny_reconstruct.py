# DroneCATS
# Copyright (c) 2026-present NAVER Cloud Corp.
# Apache-2.0

"""Smallest run of `ddr reconstruct` + `ddr verify`, without any download.

    python examples/tiny_reconstruct.py /tmp/ddr_tiny

Writes into a new folder: a two-frame stand-in for SUNRGBD.zip, an annotation package for it (rows, ledger,
reference hashes) and the reconstructed dataset. The images are synthetic; the package layout and the two
commands are the real ones.
"""
from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path
import sys
import zipfile

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from drone_depth_recon.cli import main as ddr  # noqa: E402
from drone_depth_recon.derive import derive_frame  # noqa: E402
from drone_depth_recon.ledger import SOURCES, ops_for_path  # noqa: E402
from drone_depth_recon.pack import write_zip  # noqa: E402
from drone_depth_recon.util import sha256_bytes, sha256_file  # noqa: E402

W, H, FX = 730, 530, "518.86"   # the size and focal length of a kv1/NYUdata frame
DATE = [2026, 1, 1, 0, 0, 0]


def jpeg(number: int) -> bytes:
    image = Image.new("RGB", (W, H), (70, 80, 90))
    ImageDraw.Draw(image).rectangle([200 + 40 * number, 180, 420 + 40 * number, 360], fill=(190, 60, 50))
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    return buffer.getvalue()


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))


def main(out: Path) -> int:
    pkg, archive, dataset = out / "PACKAGE", out / "SUNRGBD.zip", out / "dataset"
    out.mkdir(parents=True)
    # 1. the "official" archive: two JPEG members, as SUN RGB-D ships them
    frames = []
    with zipfile.ZipFile(archive, "w") as zf:
        for n in range(2):
            member = f"SUNRGBD/kv1/NYUdata/NYU{n:04d}/image/NYU{n:04d}.jpg"
            zf.writestr(member, jpeg(n))
            frames.append({"frame": f"sunrgbd_kv1.NYUdata.NYU{n:04d}", "archive_member": member,
                           "width": W, "height": H})
    # 2. the ledger: per frame the original with camera EXIF and the 640-wide action image without it.
    #    Each derivation path fixes its operations (ledger.ops_for_path); the hash is of the result.
    sources = {f["frame"]: zipfile.ZipFile(archive).read(f["archive_member"]) for f in frames}
    derivations = []
    for f in frames:
        paths = [f"images/sunrgbd/orig/{f['frame']}__fx{FX}_fy{FX}_{W}x{H}.jpg",
                 f"images/sunrgbd/action_base/{f['frame']}__nofx.jpg"]
        records = [{"path": p, "source": "sunrgbd", "frame": f["frame"], "ops": ops_for_path(p)} for p in paths]
        for r, data in zip(records, derive_frame(records, lambda: sources[f["frame"]]).values()):
            derivations.append({**r, "sha256": sha256_bytes(data), "bytes": len(data)})
    members = [{"name": f"{d['sha256']}.jpg", "path": d["path"], "date_time": DATE} for d in derivations]
    staged = {d["path"]: derive_frame([d], lambda d=d: sources[d["frame"]])[d["path"]] for d in derivations}
    write_zip(out / "expected.zip", members, staged.__getitem__)
    zip_sha = sha256_file(out / "expected.zip")
    (out / "expected.zip").unlink()
    write_jsonl(pkg / "ledger/derivations.jsonl", derivations)
    write_jsonl(pkg / "ledger/zip_layout.jsonl", [{"zip": "shard-00000.zip", "sha256": zip_sha, "members": members}])
    write_jsonl(pkg / "ledger/frames_sunrgbd.jsonl", frames)
    for source in SOURCES:
        if source != "sunrgbd":
            write_jsonl(pkg / f"ledger/frames_{source}.jsonl", [])
    (pkg / "ledger/archives.json").write_text(json.dumps({"SUNRGBD.zip": {
        "bytes": archive.stat().st_size, "sha256": sha256_file(archive), "source": "sunrgbd",
        "url": "https://rgbd.cs.princeton.edu/data/SUNRGBD.zip"}}, indent=2))
    # 3. one row per frame, pointing at its action image inside the zip
    rows = [{"meta": {"source": "sunrgbd", "oid": f"{f['frame']}-365x270"}, "tools": [], "messages": [
        {"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"images/shard-00000.zip#{m['name']}"}},
                                     {"type": "text", "text": "How far is the red box?"}]},
        {"role": "assistant", "content": "\\boxed{2.0m}"}]}
        for f, m in zip(frames, members[1::2])]
    write_jsonl(pkg / "rows/train_depth.jsonl", rows)
    sums = {"train_depth.jsonl": sha256_file(pkg / "rows/train_depth.jsonl"), "images/shard-00000.zip": zip_sha}
    sums |= {f"images/shard-00000.zip#{m['name']}": m["name"].split(".")[0] for m in members}
    (pkg / "reference").mkdir()
    (pkg / "reference/SHA256SUMS").write_text("".join(f"{v}  {k}\n" for k, v in sorted(sums.items())))
    # 4. the two commands a user runs (NYU and TartanAir are not used by this package; any existing path will do)
    args = ["--package", str(pkg), "--nyu", str(archive), "--sunrgbd", str(archive), "--tartanair-root", str(out)]
    if ddr(["reconstruct", *args, "--out", str(dataset), "--jobs", "2"]):
        return 1
    return ddr(["verify", "--package", str(pkg), "--dataset", str(dataset)])


if __name__ == "__main__":
    if len(sys.argv) != 2 or Path(sys.argv[1]).exists():
        sys.exit("usage: tiny_reconstruct.py <new output folder>")
    raise SystemExit(main(Path(sys.argv[1])))
