#!/usr/bin/env python3
# DroneCATS
# Copyright (c) 2026-present NAVER Cloud Corp.
# Apache-2.0

"""Compare a run with a reference run, per episode — prints the table, the RTF and the hardware caveat.

    python3 scripts/runpod/isaacsim/compare.py --results /workspace/results/isaac/<name> [more dirs…] [--md out.md]
    (default reference: episodes/isaacsim/single-drone/reference_gemini37_rtf010.jsonl, Gemini 3.7 Flash at RTF 0.10)

A results dir is run.sh's $OUT_ROOT/<name> (out_*/batch.jsonl, A.*.log). Pass several dirs if one model was split
over pods. graded and no_finish count as scored; anything else is a measurement failure and is left out of SR.
"""
import argparse, glob, json, os, re, statistics, sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
CAUTION = ("> ⚠️ **Isaac Sim results depend on the hardware.** The simulation keeps running while the model thinks, so "
           "the sim time that passes per decision depends on the machine's speed (RTF). Moving targets walk or drive "
           "during that time, so moving-target episodes change with RTF (e.g. pedestrians 8/8 at RTF 0.10 vs 0/7 at "
           "RTF 0.48 with the same model). Always report the RTF with the results.")


def plan_index():
    idx = {}
    for f in glob.glob(os.path.join(REPO, "episodes", "isaacsim", "single-drone", "plan.*.jsonl")):
        for l in open(f, encoding="utf-8"):
            q = json.loads(l)
            idx[(q["instruction"], q["spawn"], float(q.get("yaw") or 0))] = q
    return idx


def group(q):
    """static · vehicle · pedestrian, from the episode's target kind."""
    meta = q.get("meta") or q.get("plan_meta") or {}
    if not (meta.get("moving") or q.get("target_move")):
        return "static target"
    kind = (meta.get("target_kind") or "")
    return "moving target — vehicle" if kind in ("car", "drone") or kind.startswith("car") else "moving target — pedestrian"


def load_results(dirs, idx):
    rows, rtfs = {}, []
    for d in dirs:
        for p in glob.glob(os.path.join(d, "out_*", "batch.jsonl")):
            for l in open(p, encoding="utf-8"):
                r = json.loads(l)
                q = idx.get((r.get("instruction"), r.get("spawn"), float(r.get("yaw") or 0)), {})
                rid = r.get("id") or q.get("id")
                if rid:
                    rows[rid] = r
        for a in glob.glob(os.path.join(d, "A.*.log")):
            rtfs += [float(x) for x in re.findall(r"RTF=([0-9.]+)", open(a, errors="ignore").read())]
    return rows, rtfs


def scored(r): return r is not None and r.get("reason") in ("graded", "no_finish")
def ok(r): return r is not None and r.get("reason") == "graded" and bool(r.get("success"))
def label(r):
    if r is None: return "—"
    if r.get("reason") == "graded":
        return "success" if r.get("success") else f"fail({(r.get('verdict') or {}).get('reason')})"
    return "fail(no_finish)" if r.get("reason") == "no_finish" else f"not measured({r.get('reason')})"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline", default=os.path.join(REPO, "episodes", "isaacsim", "single-drone", "reference_gemini37_rtf010.jsonl"))
    ap.add_argument("--baseline-rtf", type=float, default=0.10)
    ap.add_argument("--baseline-name", default="reference (RTF 0.10)")
    ap.add_argument("--results", nargs="+", required=True)
    ap.add_argument("--name", default="this run")
    ap.add_argument("--md", default=None)
    a = ap.parse_args(argv)

    idx = plan_index()
    byid = {q["id"]: q for q in idx.values()}
    base = {}
    for l in open(a.baseline, encoding="utf-8"):
        d = json.loads(l)
        base[d["id"]] = {"reason": d.get("reason"), "success": d.get("success"),
                         "verdict": {"reason": d.get("vreason")}, "seconds": d.get("sec")}
    rows, rtfs = load_results(a.results, idx)
    rtfs.sort()
    rtf_txt = f"{rtfs[len(rtfs)//2]:.2f} (p10 {rtfs[len(rtfs)//10]:.2f}, n={len(rtfs)})" if rtfs else "not logged"

    out = []
    groups = ["static target", "moving target — vehicle", "moving target — pedestrian"]
    out.append(f"| group | {a.baseline_name} | {a.name} | same outcome |")
    out.append("|---|---|---|---|")
    tot = dict(b=0, bn=0, r=0, rn=0, ag=0, n=0)
    for g in groups:
        ids = [i for i in base if i in byid and group(byid[i]) == g]
        bs = [i for i in ids if scored(base[i])]
        rs = [i for i in ids if scored(rows.get(i))]
        both = [i for i in ids if scored(base[i]) and scored(rows.get(i))]
        b, r = sum(ok(base[i]) for i in bs), sum(ok(rows[i]) for i in rs)
        ag = sum(ok(base[i]) == ok(rows[i]) for i in both)
        out.append(f"| {g} ({len(ids)} ep) | {b}/{len(bs)} | {r}/{len(rs)} | {ag}/{len(both)} |")
        for k, v in zip(("b", "bn", "r", "rn", "ag", "n"), (b, len(bs), r, len(rs), ag, len(both))):
            tot[k] += v
    pct = lambda x, n: f"{100*x/n:.0f}%" if n else "—"
    out.append(f"| **overall SR** | **{tot['b']}/{tot['bn']} ({pct(tot['b'], tot['bn'])})** | "
               f"**{tot['r']}/{tot['rn']} ({pct(tot['r'], tot['rn'])})** | {tot['ag']}/{tot['n']} |")
    out.append(f"| **RTF** | **{a.baseline_rtf:.2f}** | **{rtf_txt}** | |")
    fails = [i for i in rows if not scored(rows[i])]
    secs = [r.get("seconds") for r in rows.values() if r.get("seconds")]
    out.append("")
    out.append(f"{len(fails)} episode(s) not measured, left out of SR: " + (", ".join(f"{i}({rows[i].get('reason')})" for i in sorted(fails)) or "none"))
    if secs:
        out.append(f"median wall time per episode {statistics.median(secs):.0f} s (n={len(secs)})")
    out.append("")
    out.append(CAUTION)
    out.append("")
    out.append(f"| id | group | {a.baseline_name} | {a.name} | same |")
    out.append("|---|---|---|---|---|")
    for i in sorted(base, key=lambda k: (groups.index(group(byid[k])) if k in byid else 9, k)):
        r = rows.get(i)
        m = "" if not (scored(base[i]) and scored(r)) else ("✓" if ok(base[i]) == ok(r) else "✗")
        out.append(f"| {i} | {group(byid[i]) if i in byid else '?'} | {label(base[i])} | {label(r)} | {m} |")
    text = "\n".join(out)
    print(text)
    if a.md:
        open(a.md, "w", encoding="utf-8").write(text + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
