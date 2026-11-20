# DroneCATS
# Copyright (c) 2026-present NAVER Cloud Corp.
# Apache-2.0

"""Check that both paths of GTStream.occlusion_from_annotator (row loop, default; numpy, with
GT_OCCLUSION_VECTORIZED=1) return what the original per-row loop did.

    python3 scripts/isaacsim/sim/test_gt_occlusion.py      # or: pytest scripts/isaacsim/sim/test_gt_occlusion.py

Needs only numpy (no Isaac Sim).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gt_sidecar import GTStream  # noqa: E402


def _loop_reference(d, imap, path):
    """The original per-row loop, kept verbatim as the reference."""
    oc = d.get("occlusion")
    arr = oc.get("data") if isinstance(oc, dict) else oc
    if not isinstance(arr, np.ndarray) or arr.dtype.names is None:
        return None, "occlusion not ready"
    if "instanceId" not in arr.dtype.names or "occlusionRatio" not in arr.dtype.names:
        return None, f"missing fields {arr.dtype.names}"
    iids = imap.get(path)
    if not iids:
        return None, "no instance id"
    best = None
    for row in arr:
        if int(row["instanceId"]) not in {int(x) for x in iids}:
            continue
        v = float(row["occlusionRatio"])
        if v < 0.0:
            continue
        best = v if best is None else min(best, v)
    if best is not None:
        return min(max(best, 0.0), 1.0), "annotator"
    for row in arr:
        if int(row["instanceId"]) not in {int(x) for x in iids}:
            continue
        v = float(row["occlusionRatio"])
        if v < 0.0:
            return None, "off screen (-1)"
        return min(max(v, 0.0), 1.0), "annotator"
    return None, "no target row"


DT = np.dtype([("instanceId", np.uint32), ("semanticId", np.uint32), ("occlusionRatio", np.float32)])


def _cases(n=500, seed=0):
    rng = np.random.default_rng(seed)
    for _ in range(n):
        rows = rng.integers(0, 12)
        arr = np.zeros(rows, dtype=DT)
        arr["instanceId"] = rng.integers(1, 20, rows)
        r = rng.random(rows).astype(np.float32)
        r[rng.random(rows) < 0.3] = -1.0
        arr["occlusionRatio"] = r
        iids = [int(x) for x in rng.integers(1, 20, rng.integers(0, 4))]
        yield {"occlusion": {"data": arr}}, {"/t": iids}, "/t"
    yield {"occlusion": None}, {"/t": [1]}, "/t"
    yield {"occlusion": np.zeros(2, dtype=[("x", np.float32)])}, {"/t": [1]}, "/t"
    yield {"occlusion": np.zeros(0, dtype=DT)}, {"/t": [1]}, "/t"
    yield {"occlusion": np.zeros(1, dtype=DT)}, {}, "/t"


def test_matches_loop():
    for d, imap, path in _cases():
        ref = _loop_reference(d, imap, path)
        for vec in (False, True):
            got = GTStream.occlusion_from_annotator(None, d, imap, path, vectorized=vec)
            assert got == ref, (vec, got, ref, d, imap)


if __name__ == "__main__":
    test_matches_loop()
    print("occlusion_from_annotator (loop and numpy) matches the loop reference")
