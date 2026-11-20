# DroneCATS
# Copyright (c) 2026-present NAVER Cloud Corp.
# Apache-2.0

"""Episode integrity verdict — the runner asks itself "can this episode go into the results?".

Motivated by three kinds of incidents seen in benchmark runs:
  · episodes silently wasted on a dead VLM server (request failed on every step, plausible score of 0)
  · host load spiked steps from 4~5s → 14~25s, wrecking moving-target episodes
    (same model·same task went 0/10 → 3~5/10 on a low-load reflight)
  · episodes started with the drone stuck from a spawn collision (path traveled < 5m, while other
    models flew normally from the same pose — a sim race)
All three surfaced only in post-hoc forensics. This module measures the same signals right after the episode,
records the verdict in the verdict file, and the runner reflies a contaminated episode once.

Thresholds were chosen from the measured distributions of those runs: healthy cells had step median 1.6~4s ·
spike (>10s) rate ≤10%, load-contaminated cells had median 6~7s · spike rate 23~44%,
dead-server episodes had ~100% failure rate, and stuck episodes had paths under 5m over dozens of steps.
The gaps between the three clusters are wide, so the cutoffs run through the middle of those valleys.

Cadence is measured not on raw step intervals but as **sim_ms = interval − vlm_ms**
(everything except model time — capture·RPC·logging residue. Drone motion runs in parallel so it is not
counted; only the waits of brake·hold·hover steps are included). In offline calibration, raw intervals
massively false-flagged models whose API is inherently slow (claude healthy-cell median 9~10s, spike rate 0.3~0.8 — not
load). With model time subtracted, the healthy band converges to 1.6~4s regardless of model — the
thresholds above were calibrated on raw intervals of local models (infer≈0), so they remain
valid as-is. The runner supplies timing in step_events from the [TIMING] stdout line (vlm_ms/capture_ms/rpc_ms/think —
VLM inference / image capture / pose RPC / inference flag).
If absent, falls back to raw intervals (legacy data — slow API models may be false-flagged, beware).

With multi-drone (1 orchestrator VLM → N vehicles), each step interleaves one event per
vehicle — cadence·path are measured per vehicle sequence (_by_drone), and if even one vehicle
trips, the episode is contaminated. A team step makes one VLM call, so the runner attaches that value
to every vehicle's event (each vehicle's frame interval contains that call once).

Pure module — touches neither the simulator nor the filesystem. The runner collects the signals and passes them in.
"""

from __future__ import annotations

import os
import statistics

# Default thresholds. Each can be overridden with the VLMPILOT_INTEG_<NAME UPPERCASE> environment variable.
DEFAULTS = dict(
    spike_s=10.0,          # steps longer than this = spike (2.5x+ the healthy median of 1.6~4s)
    spike_rate_max=0.25,   # above this spike rate → slow_steps (healthy ≤0.10, contaminated ≥0.23)
    spike_min=4,           # guards against rate false positives on short episodes — minimum absolute spike count
    vlm_fail_rate_max=0.30,  # above this request failure rate → vlm_failures (timeout-type hiccups pass)
    vlm_fail_min=5,        # minimum absolute failure count
    stuck_path_m=5.0,      # path traveled below this → stuck_spawn (based on measured spawn collisions)
    stuck_min_events=5,    # with fewer GOs than this, the motion judgment is deferred
    stuck_progress_m=1.0,  # and only when target distance shrank by less than this — a spawn near the
                           #   target (multi measured: 8.2m→4.8m, path 3.5m) is not stuck
    slow_median_s=5.0,     # overhead median above this → slow_median (the valley between healthy 1.6~4s vs
    slow_median_min=10,    #   contaminated 6~7s). Applied only to vlm_ms-corrected episodes —
                           #   disabled in raw fallback, where it false-flags API models.
    sim_err_min=5,         # sim RPC error ([CTRL] step failed etc.) count above this → sim_errors
    target_stuck_path_m=5.0,  # moving-target episode with target path below this → target_stuck
    slow_vlm_ratio=2.5,    # vlm median above this multiple of the model baseline → slow_vlm
    slow_vlm_extra_s=3.0,  #   and a minimum absolute excess (prevents false positives like 0.3s→0.9s)
)


def thresholds_from_env() -> dict:
    """Apply environment-variable overrides on top of DEFAULTS. Invalid values are silently ignored."""
    th = dict(DEFAULTS)
    for key in th:
        raw = os.environ.get(f"VLMPILOT_INTEG_{key.upper()}")
        if raw is None:
            continue
        try:
            th[key] = type(th[key])(float(raw))
        except (TypeError, ValueError):
            pass
    return th


def _by_drone(step_events) -> dict[str, list]:
    """Split events per vehicle — multi interleaves one event per vehicle in each
    step, so cadence·path must be measured on per-vehicle sequences
    (measuring them mixed, the ≈0s gaps between vehicles within a step contaminate the samples).
    Single (1 vehicle) reduces to a single key ""."""
    groups: dict[str, list] = {}
    for e in step_events:
        groups.setdefault(e.get("drone", "") or "", []).append(e)
    return groups


def _sim_dts(step_events) -> list[float]:
    """Per-step sim_ms (seconds) = interval − vlm. Everything except model time.

    Takes **one vehicle's** event sequence (for multi, call after splitting with _by_drone).
    Event a's vlm_ms is the model time of the decision made from frame a, and that call
    happens within the a→b interval. Drone motion is continuous flight, parallel to the next step —
    it does not show up in the interval (only waits of brake·hold·hover steps remain). Without
    vlm_ms (legacy data), falls back to raw intervals.
    """
    evs = [e for e in step_events if isinstance(e.get("t"), (int, float))]
    dts = []
    for a, b in zip(evs, evs[1:]):
        dt = b["t"] - a["t"]
        # intervals over 600s are treated as clock contamination (restarts etc.) and dropped from the cadence samples.
        if not 0 < dt < 600:
            continue
        # vlm_ms is the current key (also accepts validation-run data that once called it infer_ms).
        vlm = a.get("vlm_ms", a.get("infer_ms"))
        if isinstance(vlm, (int, float)):
            dt = max(0.0, dt - vlm / 1000.0)
        dts.append(dt)
    return dts


def _path_length_m(step_events, key="drone_xyz") -> float | None:
    pts = [e.get(key) for e in step_events if e.get(key)]
    if len(pts) < 2:
        return None
    total = 0.0
    for a, b in zip(pts, pts[1:]):
        total += sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5
    return round(total, 2)


def assess(step_events, *, vlm_failed: int = 0, aborted_api: bool = False,
           sim_restarts: int = 0, host_sims: int | None = None,
           sim_errors: int = 0, moving_target: bool = False,
           sim_dead: bool = False, vlm_baseline_s: float | None = None,
           thresholds: dict | None = None) -> dict:
    """Integrity verdict for one episode.

    Returns: {"valid": bool, "reasons": [...], "metrics": {...}}
    reasons lists the invalidity reasons, and metrics are observations always kept regardless of
    reasons (cadence·failure rate·path) — so thresholds can be redrawn later.
    """
    th = dict(thresholds_from_env() if thresholds is None else
              {**DEFAULTS, **thresholds})
    step_events = step_events or []
    groups = _by_drone(step_events)
    # Cadence is measured per vehicle and the **worst vehicle** is representative — if even one lane is sick,
    # the episode is contaminated. Single (1 vehicle) reduces to the same as before.
    per_dts = {d: _sim_dts(evs) for d, evs in groups.items()}
    dts = max(per_dts.values(),
              key=lambda v: (statistics.median(v) if v else 0.0), default=[])
    spikes = max((sum(1 for x in v if x > th["spike_s"]) for v in per_dts.values()),
                 default=0)
    spike_rate = round(max((sum(1 for x in v if x > th["spike_s"]) / len(v)
                            for v in per_dts.values() if v), default=0.0), 3)
    n_steps = len(step_events)
    fail_rate = round(vlm_failed / max(1, n_steps + vlm_failed), 3)
    # Drone path is also per vehicle — if even one is stuck (path < threshold), the episode is contaminated.
    per_path = {d: _path_length_m(evs) for d, evs in groups.items()}
    _paths = [p for p in per_path.values() if p is not None]
    path_m = min(_paths) if _paths else None

    reasons = []
    if sim_dead:
        # The sim died during/right after the flight — visibility replay is impossible, so no verdict can be made
        # (measured: UE render thread segfault. Running replay on a dead sim
        # stalls the runner for hours with 30s timeouts per step — replay was skipped).
        reasons.append("sim_died")
    if aborted_api:
        # The runner cut the episode after consecutive failures — server dead. Reflying while the server stays down
        # just invalidates it again for the same reason; the point is for the operator to notice.
        reasons.append("api_dead")
    if vlm_failed >= th["vlm_fail_min"] and fail_rate > th["vlm_fail_rate_max"]:
        reasons.append("vlm_failures")
    if spikes >= th["spike_min"] and spike_rate > th["spike_rate_max"]:
        reasons.append("slow_steps")
    # Uniform slowness without spikes (moderate load like 3s→8s on every step) is not caught
    # by spike rate — catch it with the overhead median. Only on episodes carrying vlm_ms:
    # the raw-fallback median mixes in model time (claude 9~10s) and gives false positives.
    vlms_all = [e.get("vlm_ms", e.get("infer_ms")) for e in step_events]
    vlms_all = [x / 1000.0 for x in vlms_all if isinstance(x, (int, float))]
    vlm_timed = len(vlms_all)
    # slow_vlm stats use think-OFF steps only — long inference on think-ON steps (the turn right after THINK)
    # is the model's choice, not contamination. This prevents the median from spiking
    # into false positives on episodes that repeat THINK. (Host time calculation subtracts each step's
    # actual vlm, so it uses all steps regardless of think — _sim_dts.)
    vlms = [e.get("vlm_ms", e.get("infer_ms")) for e in step_events
            if not e.get("think_on")]
    vlms = [x / 1000.0 for x in vlms if isinstance(x, (int, float))]
    vlm_median = round(statistics.median(vlms), 2) if vlms else None
    med = statistics.median(dts) if dts else None
    if (vlm_timed >= max(1, len(step_events) // 2)
            and len(dts) >= th["slow_median_min"]
            and med is not None and med > th["slow_median_s"]
            and "slow_steps" not in reasons):
        reasons.append("slow_median")
    # The real signature of a stuck spawn is "issued GO several times but did not move". Looking only at the path
    # mistakes (a) a vehicle the orchestrator deliberately held in multi (measured: one of 4
    # at 5.4m — right under the threshold) and (b) an episode that explored by rotating only for being stuck —
    # both are the model's choice, not contamination.
    def _go_count(evs):
        return sum(1 for e in evs if str(e.get("action", "")).upper() == "GO")

    def _progress_m(evs):
        """How much the target distance shrank (first valid dist − last valid dist)."""
        ds = [e.get("dist_m") for e in evs]
        ds = [x for x in ds if isinstance(x, (int, float)) and x == x]
        return (ds[0] - ds[-1]) if len(ds) >= 2 else None
    # Stuck vehicle = repeated GO, short path, and **did not get closer to the target**.
    # Looking only at the path mistakes a vehicle spawned near the target (short trip, arrived) for stuck
    # (measured on multi: Drone3 approached 8.2m→4.8m, path 3.5m, 6 GOs).
    def _stuck(d, p):
        if p is None or p >= th["stuck_path_m"]:
            return False
        if _go_count(groups[d]) < th["stuck_min_events"]:
            return False
        prog = _progress_m(groups[d])
        return prog is None or prog < th["stuck_progress_m"]
    if any(_stuck(d, p) for d, p in per_path.items()):
        reasons.append("stuck_spawn")
    # Accumulated sim RPC errors — an episode flown with broken render/pose is not the model's fault.
    if sim_errors >= th["sim_err_min"]:
        reasons.append("sim_errors")
    # Model-side latency blowup — if this episode's infer is much slower than the same model's normal
    # episodes (baseline) due to API congestion etc., the world keeps moving meanwhile (closed loop), distorting
    # results just like load. The baseline is built per model within the run, so there is no fixed-seconds
    # "big models penalized" problem. Without a baseline (the model's first episodes), deferred.
    if (vlm_baseline_s and vlm_median is not None
            and len(vlms) >= th["slow_median_min"]
            and vlm_median > max(th["slow_vlm_ratio"] * vlm_baseline_s,
                                 vlm_baseline_s + th["slow_vlm_extra_s"])):
        reasons.append("slow_vlm")
    # Target distance was never measured on any step — the target is not in the scene (the class where
    # 34 episodes were silently invalidated by the spawn-name suffix incident: when simGetObjectPose returns NaN
    # all verdict inputs collapse while the score stays a plausible 0).
    dists = [e["dist_m"] for e in step_events if "dist_m" in e]
    finite = [d for d in dists if isinstance(d, (int, float)) and d == d]
    if n_steps >= th["stuck_min_events"] and dists and not finite:
        reasons.append("target_missing")
    # Moving-target episode but the target stayed put — mover died·static actor. Whether the score goes up
    # or down, it is contamination that cannot be compared with other episodes.
    target_path_m = (_path_length_m(step_events, key="target_xyz")
                     if moving_target else None)
    if (moving_target and target_path_m is not None
            and n_steps >= th["stuck_min_events"]
            and target_path_m < th["target_stuck_path_m"]):
        reasons.append("target_stuck")

    # Derived residual: residual_i = sim_i − capture − rpc. Interval a→b contains frame b's
    # capture and step a's rpc (capture right before the frame marker, rpc right after).
    evs_t = [e for e in step_events if isinstance(e.get("t"), (int, float))]
    caps, residuals = [], []
    for a, b, sim_i in zip(evs_t, evs_t[1:], dts):
        cap = b.get("capture_ms")
        rpc = a.get("rpc_ms")
        if isinstance(cap, (int, float)):
            caps.append(cap / 1000.0)
            if isinstance(rpc, (int, float)):
                residuals.append(max(0.0, sim_i - cap / 1000.0 - rpc / 1000.0))
    return {
        "valid": not reasons,
        "reasons": reasons,
        "metrics": {
            "steps": n_steps,
            "sim_median_s": round(med, 2) if med is not None else None,
            "sim_p90_s": (round(sorted(dts)[max(0, int(len(dts) * 0.9) - 1)], 2)
                          if dts else None),
            "capture_median_s": (round(statistics.median(caps), 2) if caps else None),
            "residual_median_s": (round(statistics.median(residuals), 2)
                                  if residuals else None),
            "spikes": spikes,
            "spike_rate": spike_rate,
            # number of events corrected with vlm_ms — 0 means raw-interval fallback (legacy data).
            "vlm_timed_steps": vlm_timed,
            "vlm_median_s": vlm_median,
            "vlm_baseline_s": vlm_baseline_s,
            "vlm_failed": vlm_failed,
            "vlm_fail_rate": fail_rate,
            "path_m": path_m,
            "per_drone": ({d: {"sim_median_s": (round(statistics.median(per_dts[d]), 2)
                                                if per_dts[d] else None),
                               "path_m": per_path[d]}
                           for d in groups} if len(groups) > 1 else None),
            "target_path_m": target_path_m,
            "sim_errors": sim_errors,
            "sim_dead": sim_dead,
            "sim_restarts": sim_restarts,
            "host_sims": host_sims,
        },
    }
