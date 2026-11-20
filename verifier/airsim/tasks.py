# DroneCATS
# Copyright (c) 2026-present NAVER Cloud Corp.
# Apache-2.0

"""Success judging for the 4 DroneCATS tasks.

A 2x2 over two axes:
  - search   : is the target visible on the starting screen (off) / not visible (on)
  - tracking : is the target fixed (off) / moving (on)

           tracking off        tracking on
  search  ┌───────────────┬──────────────────┐
    off   │ approaching   │ tracking         │
    on    │ searching     │ search_tracking  │
          └───────────────┴──────────────────┘

Judging criterion — all four tasks use **one and the same criterion**:

    success = any one of the arrival declarations has  3D distance ≤ δ  AND  target visible

  Declarations can occur multiple times — a single declaration that misses the condition does not end the episode;
  the runner keeps the executor running (VLMPILOT_CONTINUE_AFTER_ARRIVAL). If a satisfying declaration
  appears, it cuts off there with success; otherwise it runs to the cap and fails.

  The difference between tasks is **scene setup, not the judging formula**:
    search   axis: is the target visible at start (off) / not visible (on — search by rotating)
    tracking axis: is the target fixed (off) / moving (on)

  Why multiple declarations instead of a single shot:
    Looking only at the single point of the declaration moment unfairly kills two kinds — inside δ but undetected at that moment,
    and episodes that drift into δ by inertia right after declaring. The next declaration is measured after that,
    so just accepting multiple declarations absorbs both (no separate hold window needed).

  Background on why tracking came to be measured by declaration rather than "holding":
    The old criterion was keeping distance ≤ δ for hold_s (10s) after first approach. When we decided to
    merge into a single metric, it was switched to the declaration criterion. In exchange **the target must move one-way** —
    with a round trip (ping-pong) the target passes by a drone that sits still, so a single declaration at that moment
    passes (even without ever following). That is why the default motion for the tracking family is one-way
    (mover.TargetMover(oneway=True)).
    Hold-time numbers (acquired·held_s·lost_streak_s) are dropped from judging and **only recorded**
    — how long it stayed close after catching up is still useful for diagnosis.

Why the anchor (declaration requirement) must not be removed:
  A criterion that counts success if the trajectory ever entered δ (OSR) inflates SR the shorter the cap
  — measured: of 130 episodes with end=timeout, 31 had entered δ along the trajectory
  (the same pitfall as the Car_58 case in the trace.is_success comment). Judging inputs are limited to "the moment the model itself
  said it had arrived".

Judging inputs (both left by the verifier during the executor's real flight):
  verdict : distance and visibility for each arrival declaration (qf_events list). The runner measures GT distance at the
            capture moment ([FRAME]), and if within δ also attaches a visibility detection on the spot (no hook inside the executor).
            When the FINISHED marker arrives, it judges using that frame's values. The top-level dist_m/visible is the
            "decisive measurement" among them — the satisfying declaration if any, otherwise the last declaration.
            Old verdict files without qf_events are read by reducing them to a single declaration point,
            so old runs are re-judged with exactly the same SR as before.
  trace   : 2Hz time series (distance). The tracking family also records visible, and if present
            judges holding more strictly as distance AND visible.

When judging happens:
  Success only needs the declarations, so it is settled by verdict (qf_events) — when a satisfying declaration appears it cuts off
  right there (online verdict). Declarations without capture-time visibility (ONLINE_VERDICT=0) are filled in by replay after
  the episode ends. The trajectory is for auxiliary metrics (minimum distance, hold time).

Reference values:
  δ (SR distance)          = 5 m       # same as trace.DELTA_M
  hold_s / lost_limit_s = 10s / 3s  # no longer used for judging — only for computing recorded metrics
"""

import json
import os
from dataclasses import dataclass, asdict

from .trace import DELTA_M

TRACK_HOLD_S = 10.0
TRACK_LOST_LIMIT_S = 3.0
# Episode cap (seconds). Samples after the cap are excluded from judging — so the trajectory accumulated during the executor's wrap-up
# after SIGINT does not leak into the score. Uses the same value as the runner's EP_TIMEOUT.
EP_LIMIT_S = float(os.environ.get("EP_TIMEOUT", "300"))

# Canonical names are progressive forms. Short aliases are accepted too.
TASKS = ("approaching", "searching", "tracking", "search_tracking")
ALIAS = {"approach": "approaching", "search": "searching",
         "track": "tracking", "search_track": "search_tracking"}


def canon(task: str) -> str:
    t = ALIAS.get(task, task)
    if t not in TASKS:
        raise ValueError(f"unknown task {task!r} — {list(TASKS)}")
    return t


@dataclass
class TaskVerdict:
    task: str
    success: bool
    reason: str
    ne_m: float = None            # distance at arrival/end = Navigation Error
    min_dist_m: float = None
    end_reason: str = None
    seen_at_start: bool = None    # target visible at start (must be False for the searching family to be valid)
    detected: bool = None         # captured at least once along the path
    qf_declarations: int = None   # number of arrival declarations (0 means it never ended on its own)
    qf_best_dist_m: float = None  # closest distance among the declarations
    acquired: bool = None         # tracking: first capture achieved
    held_s: float = None          # tracking: time visibility was held after capture
    lost_streak_s: float = None   # tracking: longest continuous loss

    def to_dict(self):
        return {k: v for k, v in asdict(self).items() if v is not None}


def _load_trace(path):
    if not path or not os.path.exists(path):
        return []
    out = []
    for line in open(path):
        try:
            out.append(json.loads(line))
        except Exception:
            pass
    return out


# ─────────────────────────── declaration criterion (approaching / searching) ──────────────

def qf_events(verdict) -> list:
    """List of measurement samples, one per arrival declaration.

    The runner attaches the distance and visibility (when within δ) measured at the capture moment to that frame, and when a declaration (FINISHED marker) arrives, appends it to qf_events.
    Old verdict files without this field are reduced to a single declaration point — it is accepted as a declaration only when
    end_reason is qf, so the SR of old runs is reproduced exactly (success is never granted from the
    trajectory minimum distance).
    """
    evs = verdict.get("qf_events")
    if evs is not None:
        return list(evs)
    if verdict.get("end_reason") == "qf" and verdict.get("dist_m") is not None:
        return [dict(t=None, dist_m=verdict["dist_m"],
                     visible=verdict.get("visible"))]
    return []


def _judge_declarations(verdict, trace, delta_m, basis="center"):
    """Success if any one of the declarations has (distance ≤ δ ∧ visible).

    The runner cuts the episode at a satisfying declaration, so a satisfying declaration, if any, is the last
    sample. We still scan all of them — given only the verdict file, it must be possible to re-judge with a different δ,
    and then which declaration satisfies it changes.

    basis="center" uses the distance to the target center, "surface" the distance to the 3D box surface.
    For large objects like buildings, the center distance points inside the object and δ cannot be satisfied
    (measured on campus: surface 2.4m when the backstop center is 10.4m). For such targets, the testset
    specifies delta_basis: "surface". Surface distance requires a detection
    (box3D of the detection result), so declarations without a value count as unmet. Old verdict
    files without qf_events have no per-declaration surface distance, so surface-based re-judging does not work — in that case
    use an aggregation tool that reads the verdict file's dist_surface_m directly.
    """
    md = min((p["dist_m"] for p in trace), default=verdict.get("dist_m"))
    key = "dist_surface_m" if basis == "surface" else "dist_m"
    label = "surface" if basis == "surface" else "best"
    evs = qf_events(verdict)
    hit = next((e for e in evs
                if e.get(key) is not None and e[key] <= delta_m
                and e.get("visible")), None)
    ok = hit is not None
    ds = [e[key] for e in evs if e.get(key) is not None]
    best = round(min(ds), 2) if ds else None
    n = len(evs)

    if ok and n <= 1:
        reason = "ok"
    elif ok:
        reason = f"satisfied on one of {n} declarations ({hit[key]:.1f}m ≤ δ{delta_m:g}m, visible)"
    elif n == 0:
        reason = f"no arrival declared (end={verdict.get('end_reason')})"
    elif basis == "surface" and not ds:
        reason = f"{n} declarations — no surface distance (detection failed)"
    elif any(e.get(key) is not None and e[key] <= delta_m for e in evs):
        reason = f"{n} declarations — got within δ{delta_m:g}m but the target was never visible"
    else:
        reason = f"{n} declarations — {label} {best}m > δ{delta_m:g}m"
    return ok, reason, (round(md, 2) if md is not None else None), n, best


def judge_approaching(verdict, trace_path, delta_m=DELTA_M, basis="center"):
    tr = _load_trace(trace_path)
    ok, reason, md, n, best = _judge_declarations(verdict, tr, delta_m, basis)
    return TaskVerdict(task="approaching", success=ok, reason=reason,
                       ne_m=verdict.get("dist_m"), min_dist_m=md,
                       end_reason=verdict.get("end_reason"),
                       qf_declarations=n, qf_best_dist_m=best)


def judge_searching(verdict, trace_path, delta_m=DELTA_M, basis="center"):
    tr = _load_trace(trace_path)
    ok, reason, md, n, best = _judge_declarations(verdict, tr, delta_m, basis)
    seen0 = verdict.get("seen_at_start")
    detected = (bool(verdict.get("visible")) or verdict.get("detected") is True
                or any(e.get("visible") for e in qf_events(verdict)))
    if seen0 is True:
        reason += " · note: visible from the start (=approaching condition)"
    return TaskVerdict(task="searching", success=ok, reason=reason,
                       ne_m=verdict.get("dist_m"), min_dist_m=md,
                       end_reason=verdict.get("end_reason"),
                       qf_declarations=n, qf_best_dist_m=best,
                       seen_at_start=seen0, detected=detected)


# ────────────────── hold-time metrics (not judging — for the record) ────────────────────

def _judge_hold(verdict, trace, hold_s, lost_limit_s, delta_m):
    """**Metric calculator** measuring how long it stayed close after first approach (distance ≤ δ ∧ visible).

    It was once the success criterion for tracking (holding hold_s = success). It was dropped from judging when metrics
    were unified under the single declaration criterion, and the returned acquired/held_s/lost_streak_s
    are used only for the record — holding ability after catching up is still useful for diagnosis.

    If the trace has visibility, counts distance AND visible; otherwise distance only.
    """
    trace = [p for p in trace if p.get("t") is not None and p["t"] <= EP_LIMIT_S]
    if len(trace) < 2:
        return False, "no trajectory", dict(min_dist_m=None, acquired=False,
                                     held_s=0.0, lost_streak_s=0.0)
    ts = [p["t"] for p in trace]
    ds = [p.get("dist_m") for p in trace]
    has_vis = any("visible" in p for p in trace)
    vis = [bool(p.get("visible")) if has_vis else True for p in trace]
    md = round(min(d for d in ds if d is not None), 2) if any(ds) else None

    def near(i):
        return ds[i] is not None and ds[i] <= delta_m and vis[i]

    acquired, acq_i, held_s = False, None, 0.0
    lost, lost_max, fail_by_lost = 0.0, 0.0, False
    for i in range(len(trace)):
        if not acquired:
            if near(i):
                acquired, acq_i = True, i
        else:
            dt = ts[i] - ts[i - 1]
            if near(i):
                lost = 0.0
                held_s = ts[i] - ts[acq_i]
            else:
                lost += dt
                lost_max = max(lost_max, lost)
                if lost >= lost_limit_s:
                    fail_by_lost = True
                    break

    ok = bool(acquired and held_s >= hold_s and not fail_by_lost)
    tag = f"within {delta_m:g}m" + ("+visible" if has_vis else "")
    if ok:
        reason = f"held {tag} for {held_s:.0f}s after approach"
    elif not acquired:
        reason = f"approach failed (min {md}m > δ{delta_m:g}m)"
    elif fail_by_lost:
        reason = f"lost track for {lost_max:.0f}s in a row (limit {lost_limit_s:.0f}s)"
    else:
        reason = f"held {held_s:.0f}s < {hold_s:.0f}s"
    return ok, reason, dict(min_dist_m=md, acquired=acquired,
                            held_s=round(held_s, 1), lost_streak_s=round(lost_max, 1))


def _hold_metrics(verdict, trace, hold_s, lost_limit_s, delta_m):
    """Extract only the recorded hold-time metrics (min_dist_m is produced separately by the caller)."""
    _, _, extra = _judge_hold(verdict, trace, hold_s, lost_limit_s, delta_m)
    extra.pop("min_dist_m", None)
    return extra


def judge_tracking(verdict, trace_path, delta_m=DELTA_M,
                   hold_s=TRACK_HOLD_S, lost_limit_s=TRACK_LOST_LIMIT_S):
    tr = _load_trace(trace_path)
    ok, reason, md, n, best = _judge_declarations(verdict, tr, delta_m)
    extra = _hold_metrics(verdict, tr, hold_s, lost_limit_s, delta_m)
    return TaskVerdict(task="tracking", success=ok, reason=reason,
                       ne_m=verdict.get("dist_m"), min_dist_m=md,
                       end_reason=verdict.get("end_reason"),
                       qf_declarations=n, qf_best_dist_m=best, **extra)


def judge_search_tracking(verdict, trace_path, **kw):
    tr = _load_trace(trace_path)
    delta_m = kw.get("delta_m", DELTA_M)
    ok, reason, md, n, best = _judge_declarations(verdict, tr, delta_m)
    extra = _hold_metrics(verdict, tr, kw.get("hold_s", TRACK_HOLD_S),
                          kw.get("lost_limit_s", TRACK_LOST_LIMIT_S), delta_m)
    seen0 = verdict.get("seen_at_start")
    if seen0 is True:
        reason += " · note: visible from the start"
    detected = (bool(verdict.get("visible")) or extra.get("acquired") is True
                or any(e.get("visible") for e in qf_events(verdict)))
    return TaskVerdict(task="search_tracking", success=ok, reason=reason,
                       ne_m=verdict.get("dist_m"), min_dist_m=md,
                       end_reason=verdict.get("end_reason"),
                       qf_declarations=n, qf_best_dist_m=best,
                       seen_at_start=seen0, detected=detected, **extra)


JUDGE = {
    "approaching": judge_approaching,
    "searching": judge_searching,
    "tracking": judge_tracking,
    "search_tracking": judge_search_tracking,
}


def judge_task(task, verdict, trace_path, **kw):
    """Pick the judge by task name (progressive form or alias) and run it."""
    return JUDGE[canon(task)](verdict, trace_path, **kw)
