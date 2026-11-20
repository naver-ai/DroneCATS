# DroneCATS
# Copyright (c) 2026-present NAVER Cloud Corp.
# Apache-2.0

"""Episode trajectory sampler — fills in metrics that judging only at the end cannot produce.

The verdict (distance and visibility at the moment of the arrival declaration) is measured **only once, at the very moment**
the executor declares arrival. So minimum approach distance, path length, and rotation amount were empty. This sampler fills those three.

Design decision — do not rely on executor internals:
  The runner once relied on the executor's stdout marker (`=== SINGLE ACTION MODE ===`) and an environment variable
  (`(old think-suppression env)`), and when the executor changed both, the step count became 0 and
  evaluation ran with thinking turned off. Both were silent breakages.
  Pose polling references none of the executor's internal structure, so it cannot break
  the same way. The cost is one extra RPC connection and polling load, but pose queries
  do not go through rendering, so at 2Hz it is negligible.

Fixed targets (approaching/searching) are read once at the start and reused.
The tracking family has moving targets, so with moving_target=True they are re-read every sample.
"""

import json
import math
import os
import threading
import time


class TraceSampler:
    """Poll vehicle poses during an episode and collect trajectory metrics.

    with TraceSampler(port, "Car_58") as tr:
        subprocess.run(...)          # executor real flight
    tr.result()  -> dict

    Segments where the pose cannot be read (sim restart, etc.) are skipped silently — not killing
    the evaluation itself comes first, and reliability can be judged from the sample count (`samples`).
    """

    # 2Hz. Since frame capture moved to screen recording, there is no reason to go faster —
    # RPC contends with the render thread, so it is not called more often than needed.
    def __init__(self, port: int, target_name: str, hz: float = 2.0,
                 vehicle: str = "", trace_path: str = None,
                 moving_target: bool = False, track_visibility: bool = False,
                 spawn_off=(0.0, 0.0, 0.0)):
        # spawn_off: this vehicle's spawn offset (global − rel). simGetVehiclePose is spawn-relative,
        # simGetObjectPose (target) is global, so measuring distance without the offset overestimates D2~D4
        # (measured gemini37 E-13: min_dist_m {D2 8.48, D3 13.12, D4 15.98} vs actual {6.77, 5.61, 5.82}).
        self.port = port
        self.target = target_name
        self.period = 1.0 / max(hz, 0.1)
        self.vehicle = vehicle
        self.trace_path = trace_path
        # in tracking tasks the target moves — re-read the target pose every sample
        self.moving_target = moving_target
        # so the tracking family can judge holding by 'distance AND visible', measure visibility every sample
        # only on request (it blocks rendering a little, but that is the essence of tracking).
        self.track_visibility = track_visibility
        self.spawn_off = tuple(spawn_off or (0.0, 0.0, 0.0))
        self._tgt = []            # target position per sample (all identical for a fixed target)
        self._vis = []            # visibility per sample (only with track_visibility)

        self._stop = threading.Event()
        self._thread = None
        self._pts = []            # [(t, x, y, z, dist, yaw_deg)]
        self._target_xyz = None
        self._errors = 0
        self.wall_t0 = None       # wall-clock time corresponding to sample t=0

    # ---------- lifecycle ----------

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()
        return False

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 5.0):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=timeout)
        if self.trace_path and self._pts:
            try:
                with open(self.trace_path, "w") as f:
                    for i, (t, x, y, z, d, yaw) in enumerate(self._pts):
                        rec = dict(t=round(t, 2),
                                   xyz=[round(x, 2), round(y, 2), round(z, 2)],
                                   dist_m=round(d, 2), yaw_deg=round(yaw, 1))
                        if i < len(self._tgt):
                            g = self._tgt[i]
                            rec["target_xyz"] = [round(g[0], 2), round(g[1], 2), round(g[2], 2)]
                        if self.track_visibility and i < len(self._vis):
                            rec["visible"] = self._vis[i]
                        f.write(json.dumps(rec) + "\n")
            except Exception:
                pass

    # ---------- sampling ----------

    def _run(self):
        import airsim
        for _attempt in range(3):
            # right after a restart the connection is refused for a while — giving up after one failure
            # leaves that vehicle's trajectory entirely empty (measured: 3 of 4 missing in E-10).
            try:
                # timeout_value defaults to 3600 seconds — if one RPC blocks, it just stalls and the early
                # samples are entirely empty (measured: 105.6-second gap). With 8 seconds, even if one is missed it
                # continues at the next sample.
                c = airsim.MultirotorClient(port=self.port, timeout_value=8)
                c.confirmConnection()
                p = c.simGetObjectPose(self.target).position
                if any(math.isnan(v) for v in (p.x_val, p.y_val, p.z_val)):
                    raise ValueError(f"simGetObjectPose({self.target!r}) NaN")
                self._target_xyz = (p.x_val, p.y_val, p.z_val)
                break
            except Exception:
                self._errors += 1
                if self._stop.wait(2.0):
                    return
        else:
            return                      # if the target cannot be found, give up on trajectory metrics

        t0 = time.time()
        self.wall_t0 = t0
        while not self._stop.is_set():
            try:
                # in tasks where the target moves (tracking), it must be re-read every time.
                # for a stationary target it just returns the same value, so there is no loss.
                if self.moving_target:
                    tp = c.simGetObjectPose(self.target).position
                    if not any(math.isnan(v) for v in (tp.x_val, tp.y_val, tp.z_val)):
                        self._target_xyz = (tp.x_val, tp.y_val, tp.z_val)
                    if self.track_visibility:
                        try:
                            from .judge import detect_target
                            vis, _ = detect_target(c, self.target, camera="0",
                                                   vehicle_name=self.vehicle)
                        except Exception:
                            vis = None
                        self._vis.append(vis)
                pose = c.simGetVehiclePose(vehicle_name=self.vehicle)
                q = pose.position
                xyz = (q.x_val, q.y_val, q.z_val)
                # yaw is measured too. One of the two actions is in-place rotation, so looking only at
                # position, "spinning around" and "completely stopped" look identical
                # (observed: an episode that repeated rotate 90° 24 times was misdiagnosed as 'vehicle stopped').
                _, _, yaw = airsim.to_eularian_angles(pose.orientation)
                if not any(math.isnan(v) for v in xyz):
                    gxyz = (xyz[0] + self.spawn_off[0], xyz[1] + self.spawn_off[1],
                            xyz[2] + self.spawn_off[2])
                    # xyz is recorded as relative coordinates as is (the viewer computes the offset separately
                    # from step_events and adds it) — only distance is measured globally.
                    self._pts.append((time.time() - t0, *xyz,
                                      math.dist(gxyz, self._target_xyz),
                                      math.degrees(yaw)))
                    self._tgt.append(self._target_xyz)
            except Exception:
                self._errors += 1
            self._stop.wait(self.period)

    # ---------- metrics ----------

    def result(self) -> dict:
        """Trajectory metrics. With no samples, values are None and samples=0.

        Why yaw_total_deg is measured too: one of the two actions is in-place rotation, so
        looking only at position, "spinning around" and "staying still" cannot be told apart.
        (Observed: an episode that repeated rotate 90° 24 times was misdiagnosed as 'vehicle stopped'.)

        These values are observations only, not judgments — no episode is excluded from the score.
        Failing to move is also a result of the whole system. They are material for looking into failures.
        """
        if not self._pts:
            return dict(min_dist_m=None, path_length_m=None, yaw_total_deg=None,
                        samples=0, trace_errors=self._errors)

        dists = [p[4] for p in self._pts]
        path = 0.0
        for a, b in zip(self._pts, self._pts[1:]):
            path += math.dist(a[1:4], b[1:4])
        yaw_total = 0.0
        for a, b in zip(self._pts, self._pts[1:]):
            yaw_total += abs((b[5] - a[5] + 180) % 360 - 180)   # wraparound handling
        return dict(
            min_dist_m=round(min(dists), 2),
            start_dist_m=round(dists[0], 2),   # for computing approach ratio = (start-min)/start
            path_length_m=round(path, 1),
            yaw_total_deg=round(yaw_total),
            samples=len(self._pts),
            trace_errors=self._errors,
        )


DELTA_M = 5.0      # success distance threshold. Fixed to a single team reference value.


def step_min(step_events, declared=False, visible=False):
    """Minimum distance over step measurements. declared=only confirmed-declaration steps · visible=only visible ones."""
    ds = [e["dist_m"] for e in (step_events or [])
          if e.get("dist_m") is not None
          and (e.get("declared") if declared else True)
          and (e.get("visible") if visible else True)]
    return round(min(ds), 2) if ds else None


def row_delta(r: dict, delta_m: float = DELTA_M, per_row: bool = True) -> float:
    """The δ recorded on the row (delta_m — episode profile: outdoor 5 m / indoor 2.5 m) if present, otherwise the argument δ."""
    return float(r["delta_m"]) if per_row and r.get("delta_m") is not None else delta_m


def is_success(r: dict, delta_m: float = DELTA_M, per_row: bool = True) -> bool:
    """Success judgment — any one of the confirmed-declaration steps with (distance ≤ δ AND visible).

    Requiring a declaration is the key. Without it, an episode cut off while passing by becomes a success because it was
    "by chance within 5m at that moment" (observed Car_58: end=timeout yet ok).
    The input is sr_min_m (step_min above) — None with 0 declarations.
    """
    return _within(r.get("sr_min_m"), row_delta(r, delta_m, per_row))


def _within(x, d) -> bool:
    return x is not None and not math.isnan(x) and x <= d


def sr_at(rows: list, deltas=(DELTA_M,), per_row: bool = True) -> dict:
    """SR/OSR at δ.

    Both look only at the single measurement per model output; the filters differ:
      SR  = among confirmed-declaration steps, (distance ≤ δ AND visible)
      OSR = among all steps, (distance ≤ δ) — regardless of declaration/visibility, "did it get inside δ at all"
    SR's steps also satisfy the OSR condition, so OSR ≥ SR holds by definition.
    Distance itself is kept per row, so even if δ changes later it can be recomputed without re-flying.
    """
    out = {}
    for d in deltas:
        n = len(rows) or 1
        # per_row: if delta_m is baked into the row (episode profile), count with that δ — keys keep the default δ name
        sr = sum(1 for r in rows if is_success(r, d, per_row)) / n
        osr = sum(1 for r in rows if _within(r.get("osr_min_m"), row_delta(r, d, per_row))) / n
        out[f"{d:g}m"] = dict(SR=round(sr, 3), OSR=round(osr, 3))
    return out
