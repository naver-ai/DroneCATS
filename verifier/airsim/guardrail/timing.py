# DroneCATS
# Copyright (c) 2026-present NAVER Cloud Corp.
# Apache-2.0

"""Wiring from the executor's [TIMING] lines → step_events.

Each step, the controller prints 3 direct measurements + the think flag on one line:

  [TIMING] vlm_ms=<float|None> capture_ms=<float|None> rpc_ms=<float> [think=0|1] [drone=<name>]

sim_ms (interval − vlm) and residual_ms (sim − capture − rpc) are derived by integrity.
Verdict inputs: vlm_ms → slow_vlm, sim_ms → slow_steps/slow_median.
"""

from __future__ import annotations

import re

TIMING_RE = re.compile(r"^\[TIMING\] vlm_ms=([\d.]+|None) capture_ms=([\d.]+|None) "
                       r"rpc_ms=([\d.]+)(?: think=([01]))?")


def attach_timing(step_evs: list, m_tim, drone: str) -> None:
    """Attach [TIMING] values to step_events.

    Single (drone= tag present / 1 vehicle) attaches to that vehicle's last event; a multi
    team step (no tail — one VLM call per team) attaches the same value to **all vehicles' events
    at the same capture time**. Each vehicle's frame interval contains that shared call
    once, so per-vehicle host time = interval − (same team vlm) holds.
    """
    if not step_evs:
        return
    vlm = None if m_tim.group(1) == "None" else float(m_tim.group(1))
    cap = None if m_tim.group(2) == "None" else float(m_tim.group(2))
    rpc = float(m_tim.group(3))
    think = m_tim.group(4) == "1"
    if drone:
        targets = [next((e for e in reversed(step_evs)
                         if e.get("drone", "") == drone), None)]
    else:
        multi = any(e.get("drone") for e in step_evs)
        if multi:
            t_last = step_evs[-1].get("t")
            targets = [e for e in step_evs if e.get("t") == t_last]
        else:
            targets = [step_evs[-1]]
    for ev in targets:
        if ev is None or "vlm_ms" in ev:
            continue
        ev["vlm_ms"] = vlm
        ev["capture_ms"] = cap
        ev["rpc_ms"] = rpc
        if think:
            ev["think_on"] = True
