# DroneCATS
# Copyright (c) 2026-present NAVER Cloud Corp.
# Apache-2.0

"""Pairer that groups executor stdout into steps — the `[ACTION]` that follows the pose taken at a `[FRAME]` attaches the verb.

Split out of the stdout loop of `run_bench_tasks.run_episode`, so it can be verified by feeding lines only,
without real flight. Pose RPC, clock, and output are all injected — there is no airsim here.

Two kinds of logs are read with the same code:
  · Centralized (lockstep) — per team step, **one line** of `[FRAME] captured`, followed by one
    `[ACTION] … drone=<name>` per drone. One capture covers the whole formation, so each frame rewrites pending
    wholesale. This path must reproduce the verdict numbers of old episodes exactly.
  · Async (head/body) — each drone thread prints `[FRAME] captured drone=<name>` at **its own** capture moment,
    and its own `[ACTION] … drone=<name>` after the VLM round trip (seconds). Lines from 4 drones interleave
    arbitrarily — the output lock guarantees **only line atomicity**, not adjacency. So a tagged
    frame replaces **only that drone's slot**. Clearing others' pending would throw away [ACTION]s that
    have not arrived yet (mass step loss in 4-drone interleaving).
"""
import re

from verifier.airsim.judge import judged_dist
from verifier.airsim.run_testset import (ACTION_RE, FRAME_RE, DeclarationScanner, action_drone,
                                  pick_declared_measurement)

# Drone tag of async frames. FRAME_RE is a prefix match on `^\[FRAME\] captured`, so it still matches
# with a tail — compatible with old logs. The tail format is the same as [ACTION] (it must not be a prefix).
FRAME_DRONE_RE = re.compile(r"\bdrone=(\S+)\s*$")


def frame_drone(line: str):
    """The drone of `[FRAME] captured drone=<name>`. None if untagged (centralized log)."""
    m = FRAME_DRONE_RE.search(line)
    return m.group(1) if m else None


class StepPairer:
    """Feed lines one at a time to accumulate step events — exposes the state the runner used as attributes, unchanged.

    pose_fn(drone, tagged) → capture pose event dict. If it has a `drone_xyz_raw` key, it is treated as a capture
    corrected by coordinate cross-check and counted in n_pose_fix. tagged says whether it is an async frame —
    the runner uses it to decide whether to save RPCs (the RPC pattern of the centralized path is left untouched).
    clock() → seconds elapsed since t0. warn(str) → the runner's print (declaration notices also go here).

    Online verdict is attached by injection — detection RPC belongs to the runner and there is no airsim here:
    see_fn(drone, ev) → attaches capture-time visibility to ev and returns True/False, None if outside δ or failed.
    on_verdict(ev) → the declaration was satisfied by capture-time visibility (the runner stops early).
    If neither is given, everything is judged by post-hoc replay as before (ONLINE_VERDICT=0).
    """

    def __init__(self, fleet, pose_fn, clock, warn, see_fn=None, on_verdict=None):
        self.fleet = list(fleet)
        self._pose, self._clock, self._warn = pose_fn, clock, warn
        self._see, self._on_verdict = see_fn, on_verdict
        self.online_checks = 0      # number of frames where capture-time detection actually ran (for logging)
        self.decls = DeclarationScanner()
        self.pending = {}       # drone → pose taken at [FRAME]. That drone's [ACTION] confirms it
        self.step_evs, self.qf_evs = [], []
        self.n_frames, self.n_actions, self.n_pose_fix = 0, 0, 0
        self.steps_per_drone = {}   # frame count per drone — in async mode each drone has its own step stream

    def feed(self, line):
        """One line subject to instrumentation. Lines after the cap (EP_TIMEOUT) are never fed in by the runner."""
        if FRAME_RE.match(line):
            self._frame(line)
        m_act = ACTION_RE.match(line)
        if m_act:
            self._action(line, m_act.group(1))
        if self.decls.feed(line):
            self._declaration()

    # ------------------------------------------------------------------ frame
    def _frame(self, line):
        d = frame_drone(line)
        if d is None:
            if self.pending:
                left = ",".join(sorted(k or "single" for k in self.pending))
                self._warn(f"  ! frame without [ACTION] ({left}) — dropped")
            self._stamp(self.fleet, tagged=False)
            return
        # tagged frame: only when the same drone printed again without an [ACTION], discard only that drone's record.
        if d in self.pending:
            self._warn(f"  ! frame without [ACTION] ({d}) — dropped")
        self._stamp([d], tagged=True)

    def _stamp(self, drones, tagged):
        try:
            t_cap = round(self._clock(), 2)
            self.n_frames += 1
            self._clear(drones, tagged)
            for d in drones:
                ev = self._pose(d, tagged)
                if "drone_xyz_raw" in ev:       # capture corrected by cross-check
                    self.n_pose_fix += 1
                ev.update(t=t_cap, drone=d, step=self.n_frames)
                self.pending[d] = ev
                self.steps_per_drone[d] = self.steps_per_drone.get(d, 0) + 1
                if self._see is not None and self._see(d, ev) is not None:
                    self.online_checks += 1
        except Exception as e:
            self._warn(f"  ! capture-time pose failed {e}")
            self._clear(drones, tagged)         # discard half-filled records — must not pair with a stale pose

    def _clear(self, drones, tagged):
        if not tagged:
            self.pending = {}
            return
        for d in drones:
            self.pending.pop(d, None)

    # ------------------------------------------------------------------ action·declaration
    def _action(self, line, verb):
        self.n_actions += 1
        d = action_drone(line)
        ev = self.pending.pop(d, None)
        if ev is None:
            self._warn(f"  ! [ACTION] drone={d!r} has no capture record — dropped")
            return
        ev["action"] = verb
        self.step_evs.append(ev)

    def _declaration(self):
        qv = pick_declared_measurement(self.step_evs, self.decls.vehicle)
        if qv is None:
            self._warn(f"  ! no FINISHED measurement right before the declaration (drone={self.decls.vehicle!r}) — excluded from aggregation")
            return
        qv["declared"] = True                   # marked directly on the element — subject to judging (capture-time value or replay)
        self.qf_evs.append(qv)
        who, n = qv.get("drone") or "(single)", len(self.qf_evs)
        # the distance printed in the notice is **the distance used for judging** — if plate episodes printed target center,
        # the log alone would not tell why it is inside/outside δ (judge.judged_dist).
        d = judged_dist(qv)
        d = float("nan") if d is None else d
        if self._on_verdict is None:
            self._warn(f"  declaration {n}: {who} {d:.1f}m (visibility replayed after the episode)")
        elif qv.get("visibility_src") != "capture":
            self._warn(f"  declaration {n}: {who} {d:.1f}m (outside δ — no detection)")
        elif not qv["visible"]:
            self._warn(f"  declaration {n}: {who} {d:.1f}m — 0 capture-time detections, still flying")
        else:
            self._on_verdict(qv)
