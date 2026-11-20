#!/usr/bin/env python3
# DroneCATS
# Copyright (c) 2026-present NAVER Cloud Corp.
# Apache-2.0

"""Success rates from a run_all.sh output directory, with steps per episode and per-step model latency.

    python3 scripts/runpod/airsim/summarize.py                         # newest run_all.sh run in /workspace/results
    python3 scripts/runpod/airsim/summarize.py /workspace/results/<run>
"""
import glob, json, os, statistics, sys
from collections import defaultdict

PARTS = ("single", "multi_self", "multi_agents")   # the part folders run_all.sh writes under $OUT


def is_run_all_dir(d):
    """A run_all.sh output dir: has at least one part folder. Skips the Isaac results dir (isaac/) and anything else."""
    return (os.path.isdir(d) and os.path.basename(d) != "isaac"
            and any(os.path.isdir(os.path.join(d, p)) for p in PARTS))


if len(sys.argv) > 1:
    root = sys.argv[1]
else:
    runs = [d for d in glob.glob("/workspace/results/*") if is_run_all_dir(d)]
    if not runs:
        sys.exit("no run_all.sh runs in /workspace/results")
    root = max(runs, key=os.path.getmtime)
print(root)
LEVEL = {"nh_multi-drone": "license plate", "blocks_multi-drone-hard": "sign", "blocks_multi-drone-sum_main": "three-sign sum"}


def rate(xs):
    return f"{sum(xs)}/{len(xs)} ({100 * sum(xs) / len(xs):.0f}%)" if xs else "-"


def med(xs, fmt):
    return fmt.format(statistics.median(xs)) if xs else "-"


def vlm_ms(chunk_dir):
    """Backend call time of every step (decision_<ts>.json "vlm_ms"; arrival records have none)."""
    out = []
    for f in glob.glob(os.path.join(chunk_dir, "*", "runs", "**", "decision_*.json"), recursive=True):
        try:
            v = json.load(open(f)).get("vlm_ms")
        except (OSError, ValueError):
            continue
        if isinstance(v, (int, float)):
            out.append(float(v))
    return out


single, steps, lat = defaultdict(list), defaultdict(list), defaultdict(list)
for s in glob.glob(os.path.join(root, "single", "*", "*", "summary.json")):
    name = s.split(os.sep + "single" + os.sep)[1].split(os.sep)[0].rsplit("__", 1)[0]
    for r in json.load(open(s)):
        single[name].append(bool(r.get("success")))
        if isinstance(r.get("steps"), int):
            steps[name].append(r["steps"])
for d in glob.glob(os.path.join(root, "single", "*__*")):
    if os.path.isdir(d):
        lat[os.path.basename(d).rsplit("__", 1)[0]] += vlm_ms(d)
if single:
    print(f"single drone{'':16s} {'success':14s} {'steps/ep (median)':>18s} {'step latency (median)':>22s}")
    row = lambda k, ok, st, ms: print(f"  {k:26s} {rate(ok):14s} {med(st, '{:.0f}'):>18s} {med(ms, '{:.0f} ms'):>22s}")
    for k in sorted(single):
        row(k, single[k], steps[k], lat[k])
    row("all", [x for v in single.values() for x in v], [x for v in steps.values() for x in v],
        [x for v in lat.values() for x in v])

for part in ("multi_self", "multi_agents"):
    lv = defaultdict(dict)
    for v in glob.glob(os.path.join(root, part, "*", "*", "run1", "verdict.json")):
        set_ = v.split(os.sep + part + os.sep)[1].split(os.sep)[0].split("__")[0]
        d = json.load(open(v)); lv[set_][d.get("task")] = bool(d.get("success"))
    if lv:
        print(part.replace("_", " "))
        for k in LEVEL:
            print(f"  {LEVEL[k]:26s} {rate(list(lv[k].values()))}")
        print(f"  {'all':26s} {rate([x for d in lv.values() for x in d.values()])}")
