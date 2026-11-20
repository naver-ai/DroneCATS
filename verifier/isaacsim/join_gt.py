#!/usr/bin/env python3
# DroneCATS
# Copyright (c) 2026-present NAVER Cloud Corp.
# Apache-2.0

"""Join `gt_stream.jsonl` (A) onto `steps.jsonl` (B) to produce `gt.jsonl`.

No Isaac needed: pure Python, runs anywhere after the run ends.

Key
  Wall clock. B writes `t = time.time()` in `_record_step` and reports via
  `frame_age_s` how many seconds ago the frame was captured. So the target time is
  `t - frame_age_s` (the actual capture moment), and the nearest tick in the A
  stream is picked. Without `frame_age_s`, `t` is used and the row records that.

  Sim time is not the key: B uses PX4 odometry and A the Isaac world clock, so their
  origins can differ. Instead the difference over matched pairs is measured and its
  offset and jitter reported. That is the only audit of whether both processes see
  the same time.

Output (`<run_dir>/gt.jsonl`, one line per steps.jsonl line)
  {"step":1,"vehicle":"","join":{"dt_s":0.021,"gt_i":37,"key":"t-frame_age_s"},
   "camera_xyz_m":[...],"yaw_deg":..,"camera":{..},"targets":{...},"frames":{...}}

Usage
  python3 verifier/isaacsim/join_gt.py --run-dir <run_dir> [--gt-dir <A output>] [--max-dt 0.5]
"""
import argparse
import bisect
import json
import os
import statistics
import sys


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                # A crash may have truncated the last line; drop it and report
                print(f"[JOIN] {os.path.basename(path)}:{n} dropping broken line")
    return rows


def nearest(times, order, target):
    """Return the index (original order) nearest to `target` in sorted `times`."""
    j = bisect.bisect_left(times, target)
    best, bestd = None, None
    for k in (j - 1, j, j + 1):
        if 0 <= k < len(times):
            d = abs(times[k] - target)
            if bestd is None or d < bestd:
                best, bestd = k, d
    return order[best], bestd


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True, help="B's run directory (containing steps.jsonl)")
    ap.add_argument("--gt-dir", default=None,
                    help="A's GT output directory (default: same as --run-dir)")
    ap.add_argument("--max-dt", default="auto",
                    help="do not join pairs farther apart than this (s). Default 'auto' = half "
                         "the GT tick interval (min 0.5 s), the same yardstick as pair ambiguity. "
                         "A fixed 500 ms drops valid pairs (measured: 6/31 lost at 2.65 s ticks)")
    ap.add_argument("--out", default=None, help="default: <run_dir>/gt.jsonl")
    a = ap.parse_args(argv)

    gt_dir = a.gt_dir or a.run_dir
    steps_p = os.path.join(a.run_dir, "steps.jsonl")
    gt_p = os.path.join(gt_dir, "gt_stream.jsonl")
    for p in (steps_p, gt_p):
        if not os.path.exists(p):
            print(f"[JOIN] missing: {p}")
            return 2

    steps = load_jsonl(steps_p)
    gts = load_jsonl(gt_p)
    if not steps or not gts:
        print(f"[JOIN] empty input: steps {len(steps)} · gt {len(gts)}")
        return 2

    # Target meta (color, kind, moving) is per episode, so not in stream rows. The join
    # output must be **self-contained** so the judge runs without gt_dir; embed it per row.
    tmeta = {}
    cam_meta = {}          # camera spec: only in the header, not in rows
    mp = os.path.join(gt_dir, "gt_meta.json")
    if os.path.exists(mp):
        try:
            _m = json.load(open(mp, encoding="utf-8")) or {}
            tmeta = _m.get("targets_meta") or {}
            cam_meta = _m.get("camera") or {}
        except Exception:  # noqa: BLE001
            tmeta = {}

    order = sorted(range(len(gts)), key=lambda i: gts[i].get("t") or 0.0)
    times = [gts[i].get("t") or 0.0 for i in order]

    # The cap is set by "is the nearest tick unambiguously right": half the tick interval.
    gaps = [b - x for x, b in zip(times, times[1:]) if b > x]
    half = (statistics.median(gaps) / 2.0) if gaps else None
    if str(a.max_dt).lower() == "auto":
        max_dt = max(0.5, half) if half else 0.5
        print(f"[JOIN] cap auto → {max_dt * 1000:.0f} ms "
              f"(half of median GT tick interval {(half or 0) * 2 * 1000:.0f} ms)")
    else:
        max_dt = float(a.max_dt)

    out_p = a.out or os.path.join(a.run_dir, "gt.jsonl")
    dts, offs, matched, skipped = [], [], 0, 0
    with open(out_p, "w", encoding="utf-8") as out:
        for row in steps:
            t = row.get("t")
            if t is None:
                skipped += 1
                continue
            # **Use the frame timestamp if present.** Derive it only otherwise.
            #
            # `t − frame_age_s` is not the frame time: `t` is when the row is **built**
            # after the decision, while `frame_age_s` is measured at **capture**, so the
            # difference includes the time spent in between. Measured: `dt_s` grew with
            # `frame_age_s` and hit the cap (half the median tick interval = 1.805 s),
            # randomly dropping steps in wide-interval stretches (`no_gt`).
            # Raising the cap is not the fix: moving targets would be judged against GT
            # from another moment.
            ft = row.get("frame_t")
            age = row.get("frame_age_s")
            if isinstance(ft, (int, float)) and not isinstance(ft, bool) and ft > 0:
                key, target = "frame_t", float(ft)
            elif isinstance(age, (int, float)):
                key, target = "t-frame_age_s", t - age
            else:
                key, target = "t", t
            gi, dt = nearest(times, order, target)
            if dt is None or dt > max_dt:
                skipped += 1
                continue
            g = gts[gi]
            joined = {
                "step": row.get("step"),
                "vehicle": row.get("vehicle", ""),
                "join": {"dt_s": round(dt, 4), "gt_i": g.get("i"), "key": key,
                         "max_dt_s": round(max_dt, 4)},
                "camera_xyz_m": g.get("camera_xyz_m"),
                # **Carry heading and camera spec too.** Otherwise flight-run trajectory
                # plots lack the drone FOV wedge: the angle comes from `camera.hfov_deg`
                # and the direction from `yaw_deg`, both only in A's GT rows. The GIF tool
                # uses heading too.
                "yaw_deg": g.get("yaw_deg"),
                # If absent in the row, use the header (`gt_meta.json`): A writes the
                # camera spec only once per stream.
                "camera": g.get("camera") or cam_meta or None,
                "targets": g.get("targets"),
                "n_visible": g.get("n_visible"),
                "think_decoy": g.get("think_decoy"),
                "meta": tmeta,
                "frames": g.get("frames"),
                "sim_time_a_s": g.get("sim_time_s"),
                "sim_time_b_s": row.get("sim_time_s"),
            }
            out.write(json.dumps(joined, ensure_ascii=False) + "\n")
            dts.append(dt)
            matched += 1
            if isinstance(g.get("sim_time_s"), (int, float)) and \
               isinstance(row.get("sim_time_s"), (int, float)):
                offs.append(row["sim_time_s"] - g["sim_time_s"])

    print(f"[JOIN] steps {len(steps)} · gt ticks {len(gts)} → matched {matched} · dropped {skipped}")
    # GT tick interval: the yardstick for pair ambiguity
    if gaps:
        print(f"[JOIN] GT tick interval median {statistics.median(gaps) * 1000:.0f} ms "
              f"(min {min(gaps) * 1000:.0f} · max {max(gaps) * 1000:.0f})")
    if dts:
        print(f"[JOIN] dt median {statistics.median(dts) * 1000:.1f} ms · "
              f"max {max(dts) * 1000:.1f} ms  (cap {max_dt * 1000:.0f} ms)")
        print("       dt is B's capture-to-record latency, not alignment error (see file header)")
    if half is not None and dts:
        amb = sum(1 for d in dts if d > half)
        print(f"[JOIN] ambiguous pairs {amb}/{matched} "
              f"(dt > half tick interval {half * 1000:.0f} ms)"
              + ("  ← use frame alignment (--gt-on-frame) or raise render rate" if amb else "  ← all unambiguous"))
    if offs:
        spread = (max(offs) - min(offs))
        print(f"[JOIN] sim clock diff B−A  median {statistics.median(offs):+.3f} s · "
              f"jitter {spread:.3f} s")
        print("       (a constant offset is normal since origins differ; large jitter means "
              "frame correspondence between the processes is suspect)")
    print(f"[JOIN] {out_p}")
    return 0 if matched else 1


if __name__ == "__main__":
    sys.exit(main())
