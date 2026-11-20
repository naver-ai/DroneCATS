#!/usr/bin/env python3
# DroneCATS
# Copyright (c) 2026-present NAVER Cloud Corp.
# Apache-2.0

"""Judge a run with the **same criterion** as `verifier/airsim/judge.py::_grade`, using the GT sidecar.

Why a separate tool: `verifier/airsim/judge.py` is tied to AirSim (`import airsim`,
`detect_target(client, ...)`), and pegasus runs have no such client. But **the
criterion itself is simulator-independent**:

    near    = dist ≤ delta_m
    success = near AND (visible OR not require_visible)
    reason  = "ok" | "too_far" | "not_visible" | "too_far+not_visible"

So the inputs come from the GT sidecar and the grading rule is copied as is. AirSim's
"detection" is replaced by an instance-mask area threshold; that is visibility in pegasus.

One pegasus-only reason is added:

    "no_finish"   `finished` was never declared before the episode ended (step cap, error)

**Oracle validation is the first use of this tool.** Done means "judging the oracle's own
rollouts gives SR ≈ 100%". Otherwise the oracle and the judge disagree, a bug to catch
**before** mass data generation.

Usage
  python3 verifier/isaacsim/judge_run.py --run-dir outputs/<timestamp> [--gt-dir <A output>]
  python3 verifier/isaacsim/judge_run.py --runs outputs/*/ --gt-dir <A output>   # SR summary
"""
import argparse
import glob
import json
import os
import sys


def load_jsonl(path):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass          # last line truncated by a crash
    return rows


def _kind(step):
    # steps.jsonl from older runs wrote the action under "action_type".
    return step.get("kind", step.get("action_type"))


def grade(dist, visible, delta_m, require_visible=True):
    """Same rule as `verifier/airsim/judge.py::_grade`; only the inputs come from GT."""
    near = dist is not None and dist <= delta_m
    ok = bool(near and (visible or not require_visible))
    if ok:
        reason = "ok"
    elif not near and require_visible and not visible:
        reason = "too_far+not_visible"
    elif not near:
        reason = "too_far"
    else:
        reason = "not_visible"
    parts = []
    if not near:
        parts.append(f"distance {'—' if dist is None else f'{dist:.1f}m'} > δ{delta_m}m")
    if require_visible and not visible:
        parts.append("target not detected (out of view or occluded)")
    return {"success": ok, "reason": reason,
            "dist_m": (None if dist is None else round(float(dist), 3)),
            "visible": bool(visible), "delta_m": delta_m,
            "detail": "; ".join(parts)}


def target_from_run(run_dir):
    """Use the judged target recorded in `run.json` if the run knows it.

    **Why.** With several targets in a scene, the others are decoys. Picking by the default
    rule (moving target first) **judges a decoy**: measured, 2 of 8 episodes flipped to
    "too_far" that way, and with the target given explicitly SR went 6/8 → 8/8. The
    episode records its own target, so we do not guess.

    For B (flight) runs `run.json` is written by vlmpilot and has no `plan_meta`, so
    `plan_meta.json` (left in the run folder by `batch_run`) is also checked. Without it,
    moving episodes would judge a decoy ("254 m · mask 0") while `batch.jsonl` is correct.
    """
    for name in ("run.json", "plan_meta.json"):
        try:
            with open(os.path.join(run_dir, name), encoding="utf-8") as fh:
                doc = json.load(fh)
        except (OSError, ValueError):
            continue
        meta = doc.get("plan_meta") or {}
        prim = meta.get("target_prim") or doc.get("target_prim")
        if prim:
            return prim
    return None


def _pick_target(g, target=None):
    """Judged target prim: same rule as the oracle (moving target first, else the first one)."""
    targets = g.get("targets") or {}
    if target and target in targets:
        return target
    moving = [k for k in sorted(targets)
              if (g.get("meta") or {}).get(k, {}).get("move")]
    return moving[0] if moving else (sorted(targets)[0] if targets else None)


def _events(steps, gt_by_step, delta_m, visible_px, require_visible, target=None):
    """Measurement sample per declaration (`finished`), shaped like `tasks.py::qf_events`.

    **Why scan all.** The spec is `defaults.end_on: "any"`: success if any declaration
    satisfies the criterion (`_judge_declarations`). Looking only at the last one flips an
    episode that arrived, declared, then declared again after overshooting. Samples are
    also kept so the verdict file alone allows **re-judging with a different δ**.
    """
    evs = []
    for st in steps:
        if _kind(st) != "finished":
            continue
        step_no = st.get("step")
        g = gt_by_step.get(step_no)
        if g is None:
            evs.append({"step": step_no, "dist_m": None, "visible": False,
                        "mask_px": None, "no_gt": True})
            continue
        prim = _pick_target(g, target)
        t = ((g.get("targets") or {}).get(prim)) or {}
        dist = t.get("dist_m")
        mask = int(t.get("mask_px") or 0)
        vis = mask >= visible_px
        near = dist is not None and dist <= delta_m
        evs.append({"step": step_no, "target": prim,
                    "dist_m": (None if dist is None else round(float(dist), 3)),
                    "mask_px": mask, "visible": bool(vis),
                    "near": bool(near),
                    "ok": bool(near and (vis or not require_visible)),
                    "occlusion_ratio": t.get("occlusion_ratio")})
    return evs


def _trace_min_dist(gt_rows, target=None):
    """Minimum trajectory distance: **a metric, not a verdict** (same as verifier/). Failure
    analysis needs to tell "got close but never declared" apart after the fact."""
    ds = []
    for g in gt_rows:
        prim = _pick_target(g, target)
        if prim is None:
            continue
        d = ((g.get("targets") or {}).get(prim) or {}).get("dist_m")
        if isinstance(d, (int, float)):
            ds.append(float(d))
    return round(min(ds), 3) if ds else None


def judge_run(run_dir, gt_dir=None, delta_m=5.0, visible_px=50,
              require_visible=True, target=None, task=None,
              seen_at_start=None, end_on="any"):
    """Judge one run: look at **all declarations** and grade with the `end_on` rule.

    `end_on="any"`  success if any declaration satisfies (`defaults.end_on`)
    `end_on="last"` judge by the last declaration only (old behavior, for reproduction)

    With `task`, per-task validity is recorded too. `searching` variants are valid only if
    **the target is not visible at start**; `judge_searching` does not fail such episodes
    but adds a note (success depends only on declaration, distance, visibility). Same here.
    """
    if target is None:
        target = target_from_run(run_dir)      # the episode's recorded target is canonical
    steps = load_jsonl(os.path.join(run_dir, "steps.jsonl"))
    gt_rows = load_jsonl(os.path.join(run_dir, "gt.jsonl"))
    if not gt_rows and gt_dir:
        # join here if not joined yet
        from join_gt import main as join_main
        join_main(["--run-dir", run_dir, "--gt-dir", gt_dir])
        gt_rows = load_jsonl(os.path.join(run_dir, "gt.jsonl"))
    gt_by_step = {r.get("step"): r for r in gt_rows}

    fin = [s for s in steps if _kind(s) == "finished"]
    out = {"run": os.path.basename(run_dir.rstrip("/")), "steps": len(steps),
           "declared": len(fin), "end_on": end_on}
    if task:
        out["task"] = task
    if seen_at_start is not None:
        out["seen_at_start"] = bool(seen_at_start)

    evs = _events(steps, gt_by_step, delta_m, visible_px, require_visible, target)
    out["qf_events"] = evs
    out["min_dist_m"] = _trace_min_dist(gt_rows, target)

    if not fin:
        out.update({"success": False, "reason": "no_finish", "dist_m": None,
                    "visible": False, "delta_m": delta_m,
                    "reason_text": f"arrival not declared ({len(steps)} steps)",
                    "detail": f"no finished declaration in {len(steps)} steps"})
        return out
    if all(e.get("no_gt") for e in evs):
        out.update({"success": False, "reason": "no_gt", "dist_m": None,
                    "visible": False, "delta_m": delta_m,
                    "reason_text": "GT not joined for declaration steps",
                    "detail": f"no GT for any of {len(fin)} declarations"})
        return out

    hit = next((e for e in evs if e.get("ok")), None) if end_on == "any" else \
        (evs[-1] if evs and evs[-1].get("ok") else None)
    graded = hit or (evs[-1] if end_on != "any" else None) or _best(evs)
    j = grade(graded.get("dist_m"), graded.get("visible"), delta_m, require_visible)
    out.update(j)
    out["success"] = bool(hit is not None)
    if hit is None:
        # No declaration satisfied: take the reason from the closest one
        out["reason"] = j["reason"] if j["reason"] != "ok" else "not_visible"
    ds = [e["dist_m"] for e in evs if e.get("dist_m") is not None]
    best = round(min(ds), 2) if ds else None
    out.update({"target": graded.get("target"), "step": graded.get("step"),
                "mask_px": graded.get("mask_px"),
                "occlusion_ratio": graded.get("occlusion_ratio"),
                "qf_declarations": len(evs), "qf_best_dist_m": best,
                "reason_text": _reason_text(evs, hit, best, delta_m,
                                            end_on, graded)})
    if task in ("searching", "search_tracking") and seen_at_start is True:
        # Same as `judge_searching`: not a failure, but flagged. The episode is
        # effectively an approaching case, so it must be filterable in summaries.
        out["task_valid"] = False
        out["reason_text"] += " · note: visible from start (= approaching condition)"
    elif task:
        out["task_valid"] = True
    return out


def _best(evs):
    """Closest declaration: the sample used for the reason when all failed."""
    with_d = [e for e in evs if e.get("dist_m") is not None]
    return min(with_d, key=lambda e: e["dist_m"]) if with_d else evs[-1]


def _reason_text(evs, hit, best, delta_m, end_on="any", graded=None):
    """Human-readable reason, following the wording of `_judge_declarations`."""
    n = len(evs)
    if hit is not None and n <= 1:
        return "ok"
    if hit is not None:
        return f"satisfied in {n} declarations ({hit['dist_m']:.1f}m ≤ δ{delta_m:g}m, visible)"
    if end_on == "last" and graded is not None:
        # `last` does not ask whether any declaration satisfied; only the last one.
        # The reason must show that, to explain why it differs from `any`.
        d = graded.get("dist_m")
        return (f"last declaration: {'—' if d is None else f'{d:.1f}m'}"
                f"{' ≤' if (d is not None and d <= delta_m) else ' >'} δ{delta_m:g}m"
                f" · {'visible' if graded.get('visible') else 'not visible'}"
                f" ({n} declarations · best {best}m)")
    if any(e.get("near") for e in evs):
        return f"{n} declarations: got within δ{delta_m:g}m but target never visible"
    return f"{n} declarations: best {best}m > δ{delta_m:g}m"


def _luma_of(run_dir):
    """Lighting summary recorded in `run.json`. None for older episodes."""
    p = os.path.join(run_dir, "run.json")
    if not os.path.exists(p):
        return None
    try:
        return json.load(open(p, encoding="utf-8")).get("luma")
    except (OSError, json.JSONDecodeError):
        return None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", help="a single run")
    ap.add_argument("--runs", nargs="*", help="multiple runs (SR summary)")
    ap.add_argument("--gt-dir", default=None, help="A's GT output (joined if gt.jsonl is missing)")
    ap.add_argument("--delta-m", type=float, default=5.0, help="arrival radius (default 5)")
    ap.add_argument("--visible-px", type=int, default=50,
                    help="mask area threshold for visibility (default 50)")
    ap.add_argument("--no-require-visible", action="store_true",
                    help="judge by distance only (like require_visible=False)")
    ap.add_argument("--target", default=None, help="judged target prim (default: same rule as the oracle)")
    ap.add_argument("--task", default=None,
                    choices=["approaching", "searching", "tracking", "search_tracking"],
                    help="task label (adds validity notes)")
    ap.add_argument("--end-on", default="any", choices=["any", "last"],
                    help="any (default, spec): success if any declaration satisfies. "
                         "last: judge by the last declaration only (old behavior)")
    ap.add_argument("--json", action="store_true", help="output JSON")
    a = ap.parse_args(argv)

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    runs = list(a.runs or [])
    if a.run_dir:
        runs.insert(0, a.run_dir)
    runs = [r for pat in runs for r in sorted(glob.glob(pat)) or [pat]]
    if not runs:
        print("[JUDGE] no runs to judge")
        return 2

    results = []
    for r in runs:
        j = judge_run(r, a.gt_dir, a.delta_m, a.visible_px,
                      not a.no_require_visible, a.target, task=a.task,
                      end_on=a.end_on)
        j["luma"] = _luma_of(r)          # display only; not part of the verdict
        results.append(j)
    if a.json:
        print(json.dumps(results, ensure_ascii=False, indent=1))
    else:
        print(f"{'run':<22}{'result':<7}{'reason':<20}{'dist':>8}{'mask':>8}"
              f"{'decl':>5}{'luma':>7}")
        for r in results:
            d = "—" if r.get("dist_m") is None else f"{r['dist_m']:.2f}"
            m = "—" if r.get("mask_px") is None else str(r["mask_px"])
            lm = r.get("luma") or {}
            tv = (lm.get("target_visible_steps") or {}).get("median")
            lt = ("—" if tv is None else
                  (f"{tv:.0f}*" if lm.get("low_light") else f"{tv:.0f}"))
            print(f"{r['run'][:21]:<22}{'ok' if r['success'] else 'FAIL':<7}"
                  f"{r['reason']:<20}{d:>8}{m:>8}{r['declared']:>5}{lt:>7}")
    n = len(results)
    ok = sum(1 for r in results if r["success"])
    dark = [r for r in results if (r.get("luma") or {}).get("low_light")]
    print(f"[JUDGE] SR {ok}/{n} = {ok / n * 100:.1f}%  "
          f"(δ {a.delta_m} m · visible {a.visible_px} px · "
          f"require_visible={not a.no_require_visible})")
    if dark:
        # **Lighting is not part of the verdict**: the rule is only δ and visibility.
        # It should not mix in silently though, so report it in one line (difficulty axis).
        dok = sum(1 for r in dark if r["success"])
        print(f"[JUDGE] ⚠️ low-light episodes {len(dark)}/{n} (median target luma < 20, `*` above) "
              f"; {dok}/{len(dark)} succeeded. Not used in the verdict")
    return 0 if ok == n else 1


if __name__ == "__main__":
    sys.exit(main())
