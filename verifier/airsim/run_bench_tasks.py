# DroneCATS
# Copyright (c) 2026-present NAVER Cloud Corp.
# Apache-2.0

"""Run bench episodes (editor-v2, `tasks` list) in real flight and judge them by verifier conventions — shared by single and multi.

Does the three things `run_testset.py` cannot read:
  ① building the scene — spawning targets, objects, and walls; painting; license plates (`bench_scene.py`, same rules as run_bench.py)
  ② per-drone start poses — the episode's `start_poses` go into the episode config and the executor places them (including spawn-offset correction)
  ③ per-episode radius — `task.arrival_radius_m` is δ

Judging follows the same convention as `run_testset.py` (online verdict + early stop):
  - at the `[FRAME] captured` moment, only drone and target **poses** are taken (a few RPCs, ms). No per-step detection (1 s or more).
  - the next `[ACTION] … drone=<name>` attaches a verb to that drone's capture record and confirms the step.
  - a confirmed declaration (`Target-reached signal (VLM) n/N drone=…`) marks that drone's last FINISHED step as the declaration.
  - **Online verdict**: for a drone within δ at `[FRAME]`, a visibility detection is attached on the spot (detection only, no render, ≈8ms).
    When that frame's answer is confirmed as finished, judge immediately with the attached visibility — if visible, stop instrumentation and SIGINT (early stop).
    With ONLINE_VERDICT=0, everything is replayed post hoc as before.
  - Success = declaration ∧ dist ≤ δ ∧ visible (at capture time, or by replay if missing). Also leaves a per-drone 2 Hz trajectory (trace_<drone>.jsonl) — material for the aerial-view GIF.

Usage (repo root):
    PORT=64002 EP_TIMEOUT=300 vlmpilot/.venv/bin/python -m verifier.airsim.run_bench_tasks \\
        ~/tasks/bench_multi-drone-block.json --config configs/airsim/multi-drone/airsim_multi_gemini37flash.yaml --ids E-01 --runs 2

    Launch the sim with the episode's `map` (nh/blocks), and for multi with settings that have as many Vehicles as drones:
    SETTINGS_BASE=vlmpilot/settings.multidrone.json.example bash scripts/airsim/launch_airsim.sh <map> <port> <gpu>

Each episode records `api_failures` (head/body calls, failures, 429 count) in the verdict — a clue for cases where the episode stays valid
while only SR silently drops (guardrail/api_failures.py).

env: PORT · EP_TIMEOUT(300) · CLEANUP_GRACE(120) · FORCE_SETUP=1 (rebuild scene) · OUT_DIR
     REPLAY_BUDGET_S(120) — time cap for the whole post-hoc visibility replay (safety net for when the sim dies)
Outputs: <OUT_DIR>/<task>/run<k>/{pilot.log, verdict.json, cfg.yaml, trace_<drone>.jsonl, runs/…}, <OUT_DIR>/summary.json
"""
import argparse
import glob as glob_mod
import json
import math
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time

from verifier.airsim import bench_scene as SC
from verifier.airsim import judge as J
from verifier.airsim.guardrail import count_api_failures_file
from verifier.airsim.judge import _box_area, _get_detections, vehicle_global_position
from verifier.airsim.pair_stream import StepPairer
from verifier.airsim.run_testset import (PILOT, PY, _hold_still, _pose_now, capture_visibility, place)
from verifier.airsim.trace import TraceSampler

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_doc(path):
    doc = json.load(open(path, encoding="utf-8"))
    if doc.get("version") != "editor-v2":
        raise SystemExit(f"not an editor-v2 document: version={doc.get('version')!r}")
    if not doc.get("tasks"):
        raise SystemExit(f"{path}: tasks is empty")
    return doc


def start_z(sp, ground_z):
    """Drone start z (NED). The episode's `z` if given, otherwise ground − 1.5 m.

    (2.0 was a case of tipping over after the drop (camera pitch warning).) Putting z in the episode removes the dependency on ground measurement."""
    if sp.get("z") is not None:
        return round(float(sp["z"]), 2)
    return round(float(ground_z) - 1.5, 2)


def episode_config(base_cfg_path, task, defaults, output_root, out_path, ground_z=0.0):
    """Base config + this episode's output_root·start_poses. Judging-related keys (radius, oracle) are not included — judging belongs to the runner.

    ground_z: measured scene ground z (NED). Start z is set to **terrain − 2 m** — if z is relative to a flat
    plane (0), in high-terrain areas (E-20, ground −1.5) the drone starts underground
    (measured: it only rose above the surface at step 55). Executor R13 uses z as is."""
    import yaml
    cfg = yaml.safe_load(open(base_cfg_path, encoding="utf-8")) or {}
    cfg["output_root"] = output_root
    names = list(cfg.get("drone_names") or [])
    poses = task.get("start_poses") or {}
    if names and poses:
        missing = [n for n in names if n not in poses]
        if missing:
            print(f"  ! drones {missing} are not in start_poses — they start from the settings spawn spots", flush=True)
        cfg["start_poses"] = {k: dict(v, z=start_z(v, ground_z)) for k, v in poses.items() if k in names}
    # the single executor does not read start_poses — for single the runner places it directly with place() (run_episode).
    # scene geometry for the control head (head.map: true) — only text-free walls, color markers, and start slots. Episodes without it do not use the key.
    try:
        from vlmpilot.policy.head_map import scene_of_task
        scene = scene_of_task(task)
        if scene:
            cfg["scene"] = scene
    except Exception as exc:  # noqa: BLE001 — the map is extra info; the episode runs without it
        print(f"  ! scene geometry build failed (continuing without a map): {exc}", flush=True)
    want = int(task.get("drones", defaults.get("drones", len(names) or 1)))
    if names and want != len(names):
        print(f"  ! episode drones={want} ≠ config drone_names={len(names)} — going with the config", flush=True)
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    return cfg


def reconnect_client(port):
    """Create a fresh RPC client for replay — a msgpack client that timed out cannot be reused."""
    import airsim
    c = airsim.MultirotorClient(port=port, timeout_value=30)
    c.confirmConnection()
    return c


_SIM_DEAD_MARKERS = ("ECONNREFUSED", "Connection refused", "ConnectionRefusedError",
                     "Retry connection over the limit")


def _looks_sim_dead(exc):
    """Does the reconnect failure message mean 'the sim process is gone'?

    When UE aborts, tornado repeats `Connect error on fd 42: ECONNREFUSED` and
    ends with `Retry connection over the limit` — a signal that waiting longer will not bring it back.
    """
    s = str(exc).lower()
    return any(m.lower() in s for m in _SIM_DEAD_MARKERS)


def port_listening(port, timeout_s=1.0, host="127.0.0.1"):
    """Check within 1 s whether there is a listener on the RPC port. True if the port is unknown — do not give up prematurely."""
    if not port:
        return True
    try:
        with socket.create_connection((host, int(port)), timeout=timeout_s):
            return True
    except OSError:
        return False


def replay_budget_s(default=120.0):
    """Time cap for the whole replay (env REPLAY_BUDGET_S, default 120 s, disabled if ≤ 0)."""
    try:
        return float(os.environ.get("REPLAY_BUDGET_S") or default)
    except (TypeError, ValueError):
        return default


def replay_todo(step_evs, delta_m, crosscheck=False):
    """Declaration steps that need to be rewound and re-measured — inside δ and not judged at capture time (online verdict).

    With crosscheck=True, those judged at capture time are also re-measured and written separately to visible_replay.
    """
    return [e for e in step_evs
            if e.get("declared") and J.judged_dist(e) is not None and J.judged_dist(e) <= delta_m
            and ("visible" not in e or (crosscheck and e.get("visibility_src") == "capture"))]


def replay_visibility_multi(mc, tgt_sim, step_evs, delta_m, mesh_pattern, camera="0",
                            port=None, attempts=3, reconnect=None, budget_s=None,
                            is_listening=None, clock=None, crosscheck=False):
    """Rewind to the recorded poses and measure visibility (after the episode ends) — the multi version of run_testset.replay_visibility.

    Two differences: ① rewinds the drone by name (the single version is fixed to drone 1) — poses are **relative** to the spawn origin (rel_xyz).
    ② spawned objects are detected with a mesh (asset) wildcard and only the target is kept by actor name (judge._get_detections).
    Declarations already judged at capture time are skipped (online verdict); with crosscheck=True they are written separately to visible_replay.

    Each item is retried, and on failure the client is recreated. Rationale (measured):
    the child exited dirty (`command worker still busy` ·`IOLoop is already running` ×4),
    replay started with the sim RPC blocked, **Request timed out on the first item** → whole run aborted without retry →
    3 in-δ declarations (including 1.47 m) were left without visible and SR hardened into failure.

    **But if the sim is dead (UE abort), retrying is waste.** Repeating 30 s × attempts per item
    multiplies by the number of declarations and stalls for 23 minutes (measured — manually stopped by the watchdog).
    So when a death signal is seen, the remaining items are **dropped immediately without being attempted**:
      ① reconnect fails with ECONNREFUSED/`Retry connection over the limit` (or no listener on the port at that time)
      ② no listener on the RPC port right after one attempt fails (socket connect 1 s) — regardless of the exception
         message (`Request timed out` is caught here). The same check after 2 consecutive item failures backs it up
      ③ the overall time cap REPLAY_BUDGET_S (default 120 s) is exceeded — a safety net for cases the two above miss
    Even when dropping, **the judging rule does not change**: unmeasured declarations stay without visible (they do not turn into successes),
    and only the reason goes errors → `verdict.visibility_error` as "sim dead during replay …".

    Returns: (number of successful measurements, final failure list). The old implementation swallowed exceptions and still returned len(todo),
    so "visibility replay 3 items" was printed on the line right after "replay aborted" — it was 0 but looked like 3.
    """
    todo = replay_todo(step_evs, delta_m, crosscheck)
    if not todo:
        return 0, []
    import airsim
    ident = [1.0, 0.0, 0.0, 0.0]
    if reconnect is None and port is not None:
        reconnect = reconnect_client
    if is_listening is None:
        is_listening = port_listening
    if budget_s is None:
        budget_s = replay_budget_s()
    clock = clock or time.monotonic
    t_start = clock()
    done, errors, aborted, consec_fail = 0, [], None, 0
    dead = False        # once death is confirmed, not even the final simPause(False) is issued (it would burn another 30 s)
    try:
        for e in todo:
            if aborted:
                break                      # the sim is dead — the remaining items are not even attempted
            last = None
            for k in range(max(1, attempts)):
                if budget_s > 0 and clock() - t_start >= budget_s:
                    aborted = f"replay aborted: budget {budget_s:g}s exceeded"
                    break
                try:
                    mc.simPause(True)
                    dx, dy, dz = e.get("rel_xyz") or e["drone_xyz"]
                    qw, qx, qy, qz = e["drone_quat"]
                    mc.simSetVehiclePose(
                        airsim.Pose(airsim.Vector3r(dx, dy, dz), airsim.Quaternionr(qx, qy, qz, qw)),
                        True, e.get("drone") or "")
                    _hold_still(mc, airsim, (dx, dy, dz), (qw, qx, qy, qz), e.get("drone") or "")
                    tx, ty, tz = e["target_xyz"]
                    tw, txx, tyy, tzz = e.get("target_quat") or ident
                    mc.simSetObjectPose(
                        tgt_sim,
                        airsim.Pose(airsim.Vector3r(tx, ty, tz),
                                    airsim.Quaternionr(txx, tyy, tzz, tw)), True)
                    mc.simContinueForFrames(2)   # turn the render — without it the previous frame comes back
                    dets = _get_detections(mc, tgt_sim, camera=camera,
                                           vehicle_name=e.get("drone") or "",
                                           mesh_pattern=mesh_pattern)
                    vis = bool(dets)
                    area = round(max((_box_area(x.box2D) for x in dets), default=0.0), 1)
                    if e.get("visibility_src") == "capture":
                        e["visible_replay"], e["bbox_px_replay"] = vis, area   # cross-check record only
                    else:
                        e["visible"], e["bbox_px"] = vis, area
                    done += 1
                    last = None
                    consec_fail = 0
                    break
                except Exception as exc:
                    last = f"t={e.get('t')} {type(exc).__name__}: {exc}"
                    print(f"  ! replay retry {k + 1}/{attempts} {last}", flush=True)
                    if _looks_sim_dead(exc) and not is_listening(port):
                        aborted = f"sim dead during replay: RPC refused ({exc})"
                        dead = True
                        break
                    if not is_listening(port):
                        # regardless of the marker, attach a 1 s socket check to every failed attempt.
                        # at the time of the incident the message was `Request timed out`, so ① did not catch it, and ② waited for
                        # 2 items, burning 30 s × attempts multiplied.
                        aborted = f"sim dead during replay: no listener on RPC port {port}"
                        dead = True
                        break
                    if reconnect is not None and k + 1 < max(1, attempts):
                        try:
                            mc = reconnect(port)
                        except Exception as rexc:
                            print(f"  ! replay reconnect failed: {rexc}", flush=True)
                            if _looks_sim_dead(rexc) or not is_listening(port):
                                aborted = f"sim dead during replay: reconnect failed ({rexc})"
                                dead = True
                                break
            if last:
                errors.append(last)
                consec_fail += 1
                if not aborted and consec_fail >= 2 and not is_listening(port):
                    aborted = (f"sim dead during replay: no listener on RPC port {port} "
                               f"after {consec_fail} failed items")
                    dead = True
    finally:
        if not dead:                       # an unpause on a dead sim burns one more timeout
            try:
                mc.simPause(False)
            except Exception:
                pass
    if aborted:
        n_skip = len(todo) - done - len(errors)
        note = f"{aborted} — {n_skip}/{len(todo)} declarations left unmeasured"
        print(f"  ! replay aborted: {note}", flush=True)
        errors.append(note)
    return done, errors


def resolve_anchors(mc, task, port, ground_z=0.0, warn=print):
    """Global positions of the judgment anchors (`judge.judge_anchors`) — measured in the scene, falling back to episode coordinates if unreadable.

    For episodes whose anchor is the target itself (sign hard·sum), returns an empty dict — that position comes from
    the `target_xyz` taken every frame (so a moving target is followed as well).
    License plates are stationary objects, so they are read **only once at episode start** — no extra RPCs per frame.
    """
    want = J.anchor_xyz_from_task(task, ground_z)     # name → episode coordinates (both the anchor list and the fallback)
    out = {}
    for name, fallback in want.items():
        sim_name = SC.prop_name_for(task, name, port)
        try:
            p = mc.simGetObjectPose(sim_name).position
            xyz = [round(p.x_val, 3), round(p.y_val, 3), round(p.z_val, 3)]
            if any(v != v for v in xyz):
                raise ValueError("NaN")
            out[name] = xyz
        except Exception as exc:  # noqa: BLE001 — judging must still stand even if one anchor cannot be read
            out[name] = [round(v, 3) for v in fallback]
            warn(f"  ! anchor '{sim_name}' measurement failed ({type(exc).__name__}: {str(exc)[:60]}) "
                 f"— judging with the episode coordinates {out[name]}")
    return out


def validate_episode(task, verdict, ground_z, drone_names):
    """Hard invariants for episode validity — returns a list of violation reasons (empty list = valid).

    Covers every measured contamination type: wrong start (pre-placement/START lost), sinking underground or falling into the abyss
    (spawn before terrain loading, sim collapse), stale trace, initialization hang (step 0).
    """
    reasons = []
    evs = verdict.get("step_events") or []
    if not evs:
        return ["step 0 (initialization row)"]
    poses = task.get("start_poses") or {}
    first = {}
    for e in evs:
        first.setdefault(e.get("drone"), e)
    for n, e in first.items():
        sp = poses.get(n)
        if sp:
            d = math.hypot(e["drone_xyz"][0] - float(sp["x"]), e["drone_xyz"][1] - float(sp["y"]))
            if d > 3.0:
                reasons.append(f"{n} drifted from start by {d:.1f} m (first capture ({e['drone_xyz'][0]:.1f},{e['drone_xyz'][1]:.1f}))")
    # NED: sinking (+) allows only 1 m (strict), above (−) 60 m — climbing to scout is valid behavior
    # (measured false positive: in opus5 E-02 a 20.4 m climb was judged as departure, causing an unnecessary retry)
    lo, hi = float(ground_z) - 60.0, float(ground_z) + 1.0
    for e in evs:
        z = e["drone_xyz"][2]
        if z > hi or z < lo:
            reasons.append(f"{e.get('drone')} altitude drift z={z:+.1f} (ground {ground_z:+.1f}, step {e.get('step')})")
            break
    return reasons


class _VehiclePoseCache:
    """Thin client wrapper that reuses simGetVehiclePose responses for the duration of one frame — async path only.

    The same relative pose of the same drone was being asked twice per frame (once by `_pose_now` to take the attitude,
    once by the runner for coordinate cross-check). In async mode, where each drone thread captures its own frames, this round trip
    comes back as capture latency — the later the pose is taken, the farther the recorded spot is from where the shot was actually taken.
    The centralized path keeps the old RPCs as is (behavior settled by measurement is left untouched).
    """

    def __init__(self, mc):
        self._mc, self._pose = mc, {}

    def __getattr__(self, name):
        return getattr(self._mc, name)

    def simGetVehiclePose(self, vehicle_name=""):
        if vehicle_name not in self._pose:
            self._pose[vehicle_name] = self._mc.simGetVehiclePose(vehicle_name=vehicle_name)
        return self._pose[vehicle_name]


def settle_after_build(reused, verbose=True):
    """Only for episodes that **freshly built** the scene, wait for the sim to settle before t0. Reuse episodes do not wait.

    Rationale (measured): in FORCE_SETUP episodes, right after the first step 3 of 4 drones produced no steps for ~50 s
    (not seen in scene-reuse episodes) — because the sim stays busy for a while with spawning, textures, and physics settling.
    ensure_scene comes before run_episode, so this wait is **outside the episode budget (EP_TIMEOUT)**.
    That is, it does not eat into flight time. Adjust with env SCENE_SETTLE_S (0 means no wait).

    Returns: the seconds actually waited.
    """
    if reused:
        return 0.0
    try:
        secs = float(os.environ.get("SCENE_SETTLE_S", "20"))
    except ValueError:
        secs = 20.0
    if secs <= 0:
        return 0.0
    if verbose:
        print(f"  [SETTLE] built scene → waiting {secs:g}s", flush=True)
    time.sleep(secs)
    return secs


def run_episode(task, defaults, base_cfg, port, ep_dir, ep_timeout, grace, target, drone_names,
                ground_z=0.0):
    """One episode. Returns a verdict dict."""
    import airsim
    os.makedirs(ep_dir, exist_ok=True)
    cfg_path = os.path.join(ep_dir, "cfg.yaml")
    episode_config(base_cfg, task, defaults, os.path.join(ep_dir, "runs"), cfg_path,
                   ground_z=ground_z)
    delta = float(task.get("arrival_radius_m", defaults.get("arrival_radius_m", 5.0)))

    mc = airsim.MultirotorClient(port=port, timeout_value=30)
    mc.confirmConnection()
    poses = task.get("start_poses") or {}
    if not drone_names and poses:
        # single: the runner places the first drone's start pose (z is ground 0 — takeoff sets the altitude)
        sp = next(iter(poses.values()))
        place(port, float(sp["x"]), float(sp["y"]), float(sp.get("z", 0.0)), float(sp.get("yaw_deg", 0.0)))
    fleet = drone_names or [""]
    # spawn offset — at replay time simSetVehiclePose takes coordinates relative to the spawn origin.
    # right after a sim restart, simGetObjectPose z comes back contaminated by tens of m (measured E-04 +28.7)
    # — used as is, dist_m is inflated for the whole episode and the verdict numbers are ruined. Retry for a few seconds.
    spawn_off = {}
    for n in fleet:
        for attempt in range(8):
            g = vehicle_global_position(mc, n)
            r = mc.simGetVehiclePose(vehicle_name=n).position
            off = (g.x_val - r.x_val, g.y_val - r.y_val, g.z_val - r.z_val)
            if abs(off[2]) < 5.0:
                break
            time.sleep(1.0)
        else:
            print(f"  ! {n}: abnormal spawn z offset ({off[2]:+.1f} m) — distance values may be contaminated", flush=True)
        spawn_off[n] = off
    # judgment anchors — for vehicle episodes, the target car's 2 license plates; otherwise the target itself (empty dict)
    anchor_xyz = resolve_anchors(mc, task, port, ground_z)
    if anchor_xyz:
        print(f"  verdict anchors: {len(anchor_xyz)} ({', '.join(anchor_xyz)}) — δ{delta:g}m is "
              f"measured as the minimum distance to these (target-center distance is still recorded in dist_m)", flush=True)

    # remove trace leftovers from the previous attempt — if the sampler fails, stale files get mixed into the plot (measured E-04 D3)
    for p in glob_mod.glob(os.path.join(ep_dir, "trace_*.jsonl")):
        os.remove(p)
    if drone_names and poses:
        # move the drone above the start pose before launching the executor — if this scene's walls spawn at the position
        # left by the previous episode, the drone is trapped inside a wall and the takeoff join never finishes
        # (measured 2 hangs on E-04: frame 0 · step 0, stalled right after the CONFIG output)
        for n in drone_names:
            sp2 = poses.get(n)
            if not sp2:
                continue
            off = spawn_off.get(n) or (0.0, 0.0, 0.0)
            mc.simSetVehiclePose(
                airsim.Pose(airsim.Vector3r(float(sp2["x"]) - off[0], float(sp2["y"]) - off[1],
                                            start_z(sp2, ground_z) - off[2]),
                            airsim.to_quaternion(0, 0, math.radians(float(sp2.get("yaw_deg", 0))))),
                True, n)
        time.sleep(1.5)     # wait for landing and attitude to stabilize — if too short, the executor starts while tilted
        # confirm it was applied — prevents cases where RPC loss made just one drone start from the previous episode's position (E-18 D1)
        for n in drone_names:
            sp2 = poses.get(n)
            if not sp2:
                continue
            off = spawn_off.get(n) or (0.0, 0.0, 0.0)
            for _try in range(2):
                r = mc.simGetVehiclePose(vehicle_name=n).position
                if math.hypot(r.x_val - (float(sp2["x"]) - off[0]),
                              r.y_val - (float(sp2["y"]) - off[1])) <= 2.5:
                    break
                print(f"  ! {n} pre-placement not applied (current rel ({r.x_val:.1f},{r.y_val:.1f})) — retrying", flush=True)
                mc.simSetVehiclePose(
                    airsim.Pose(airsim.Vector3r(float(sp2["x"]) - off[0], float(sp2["y"]) - off[1],
                                                start_z(sp2, ground_z) - off[2]),
                                airsim.to_quaternion(0, 0, math.radians(float(sp2.get("yaw_deg", 0))))),
                    True, n)
                time.sleep(1.2)
    traces = {n: TraceSampler(port, target, hz=2.0, vehicle=n,
                              trace_path=os.path.join(ep_dir, f"trace_{n or 'single'}.jsonl"),
                              spawn_off=(spawn_off.get(n) if drone_names else None))
              for n in fleet}
    env = dict(os.environ, PYTHONUNBUFFERED="1", VLMPILOT_CONTINUE_AFTER_ARRIVAL="1")
    log_path = os.path.join(ep_dir, "pilot.log")
    t0 = time.time()
    proc = subprocess.Popen([PY, "-W", "ignore", "-m", "vlmpilot", "airsim", "--config", cfg_path,
                             "--port", str(port)],
                            cwd=PILOT, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    proc.stdin.write(SC.mission_text(task) + "\n")
    proc.stdin.flush()
    proc.stdin.close()
    for tr in traces.values():
        tr.start()

    timed_out, sigint_sent = {"v": False}, {"v": False}

    def _pose_ev(_d, tagged):
        """The drone's pose at the capture moment — including cross-check and relative coordinates. Called by StepPairer every frame."""
        c = _VehiclePoseCache(mc) if tagged else mc
        ev = _pose_now(c, target, _d)
        off = spawn_off.get(_d) or (0.0, 0.0, 0.0)
        # coordinate cross-check — cases where simGetObjectPose responses get mixed up and all 4 drones
        # receive a bogus global (E-06, z=+3.1 band) → corrected with relative coordinates + offset
        rp = c.simGetVehiclePose(vehicle_name=_d).position
        corr = (rp.x_val + off[0], rp.y_val + off[1], rp.z_val + off[2])
        g = ev["drone_xyz"]
        if math.dist(g, corr) > 1.5:
            ev["drone_xyz_raw"] = g            # this key is itself the correction marker — StepPairer counts it
            ev["drone_xyz"] = [round(v, 3) for v in corr]
            t2 = ev["target_xyz"]
            ev["dist_m"] = round(math.dist(corr, t2), 3)
        ev["rel_xyz"] = [round(ev["drone_xyz"][0] - off[0], 3),
                         round(ev["drone_xyz"][1] - off[1], 3),
                         round(ev["drone_xyz"][2] - off[2], 3)]
        # judged distance — the minimum anchor distance. If the anchor is the target itself, it equals that frame's target center.
        pts = anchor_xyz or {J.TARGET_ANCHOR: ev["target_xyz"]}
        d_anchor, which = J.anchor_min_dist(ev["drone_xyz"], pts)
        ev["dist_anchor_m"], ev["anchor"] = round(d_anchor, 3), which
        return ev

    def _sigint():
        if not sigint_sent["v"] and proc.poll() is None:
            sigint_sent["v"] = True
            try:
                proc.send_signal(signal.SIGINT)
            except Exception:
                pass

    def _interrupt_pilot():
        """Early stop: stop the trajectory → SIGINT → forced kill after grace (the watchdog deadline is absolute, so it is set separately)."""
        _sigint()
        for tr in traces.values():
            tr.stop(timeout=0.1)

        def _abort_kill(p=proc, g=grace):
            try:
                p.wait(timeout=g)
            except Exception:
                try:
                    p.kill()
                except Exception:
                    pass
        threading.Thread(target=_abort_kill, daemon=True).start()

    # the detection filter is a mesh (asset) name wildcard — spawned objects are not caught by actor name (see judge)
    mesh_pat = (task.get("target", {}).get("asset") or defaults.get("spawn_asset") or "*").rstrip("*") + "*"

    # online verdict — the runner runs capture-time detection (RPC), the pairer only reads the result
    online = os.environ.get("ONLINE_VERDICT", "1") == "1"
    online_crosscheck = os.environ.get("ONLINE_VERDICT_CROSSCHECK") == "1"
    early_stop = {"v": False}

    def _see(_d, ev):
        """If within δ, attach only a detection with that drone's camera (≈8ms). Even on failure the episode goes on — replay decides."""
        try:
            return capture_visibility(mc, target, ev, delta, camera="0", vehicle_name=_d,
                                      mesh_pattern=mesh_pat)
        except Exception as e:
            print(f"  ! capture-time detection failed {e} — replay decides", flush=True)
            return None

    def _satisfied(qv):
        early_stop["v"] = True
        where = qv.get("anchor") or J.TARGET_ANCHOR
        print(f"  · declaration #{len(pairer.qf_evs)} meets the condition: {qv.get('drone') or '(single)'} "
              f"{J.judged_dist(qv):.1f}m({where}) ≤ δ{delta:g}m, visible t={qv.get('t')}s — early stop "
              f"({pairer.online_checks} capture-time detection frames)", flush=True)
        _interrupt_pilot()

    pairer = StepPairer(fleet, _pose_ev, lambda: time.time() - t0,
                        lambda s: print(s, flush=True),
                        see_fn=_see if online else None,
                        on_verdict=_satisfied if online else None)

    def _watchdog():
        deadline = t0 + ep_timeout
        while proc.poll() is None:
            if time.time() >= deadline:
                if not timed_out["v"]:
                    timed_out["v"] = True
                    # stop instrumentation the moment the cap is reached — so the wrap-up flight after SIGINT does not leak into judging
                    for tr in traces.values():
                        tr.stop(timeout=0.1)
                    print(f"  ! {task['id']}: exceeded {ep_timeout}s — prompting wrap-up with SIGINT", flush=True)
                    _sigint()
                if time.time() >= deadline + grace:
                    print(f"  ! {task['id']}: wrap-up exceeded {grace}s — force kill", flush=True)
                    proc.kill()
                    return
            time.sleep(0.5)

    threading.Thread(target=_watchdog, daemon=True).start()
    with open(log_path, "w", encoding="utf-8") as lf:
        for line in proc.stdout:
            lf.write(line)
            # lines after the cap or early stop are only recorded — so the wrap-up flight does not leak into judging
            if not timed_out["v"] and not early_stop["v"]:
                pairer.feed(line)
    proc.wait()
    step_evs, qf_evs = pairer.step_evs, pairer.qf_evs
    for tr in traces.values():
        tr.stop()
    wall = round(time.time() - t0, 1)

    # post-hoc replay — rewind only steps with declaration ∧ dist ≤ δ to fill in visibility
    # the list replay attempts differs from the number of declarations that need a verdict — crosscheck also re-measures
    # those already judged at capture time (and those do not lose their verdict even if it fails).
    n_attempt = len(replay_todo(step_evs, delta, online_crosscheck))
    n_need = len(replay_todo(step_evs, delta))
    n_rep, vis_errs = replay_visibility_multi(mc, target, step_evs, delta, mesh_pat, port=port,
                                              crosscheck=online_crosscheck)
    if n_rep:
        print(f"  · visibility replay {n_rep}/{n_attempt}", flush=True)
    if vis_errs:
        # unmeasured = declarations still without visible after replay — instead of counting, apply the same filter again
        n_miss = len(replay_todo(step_evs, delta))
        print(f"  ! visibility replay failed {n_miss}/{n_need} — those declarations cannot be judged and "
              f"are counted as failures: {vis_errs[0]}", flush=True)
    if online_crosscheck:
        for e in step_evs:
            if e.get("visibility_src") == "capture" and "visible_replay" in e:
                same = bool(e["visible"]) == bool(e["visible_replay"])
                print(f"  {'·' if same else '!!'} capture/replay cross-check {'match' if same else 'mismatch'} "
                      f"({e.get('drone') or '(single)'} capture={e['visible']} bbox={e.get('bbox_px')} / "
                      f"replay={e['visible_replay']} bbox={e.get('bbox_px_replay')}) t={e.get('t')}s",
                      flush=True)
    if pairer.n_pose_fix:
        print(f"  ! coordinate cross-check corrected {pairer.n_pose_fix} (simGetObjectPose reply contaminated)", flush=True)
    hit = next((e for e in step_evs
                if e.get("declared") and e.get("visible")
                and J.judged_dist(e) is not None and J.judged_dist(e) <= delta), None)
    end = "qf" if hit else ("timeout" if timed_out["v"] else "eof")
    tr_res = {n or "single": traces[n].result() for n in fleet}
    for n, res_ in tr_res.items():
        if not res_.get("samples"):
            print(f"  ! trace {n}: 0 samples — this drone will be missing from the trajectory plot", flush=True)
    min_dist = {n or "single": (tr_res[n or "single"].get("min_dist_m")) for n in fleet}
    verdict = dict(task=task["id"], prompt=SC.mission_text(task), target=target, delta_m=delta,
                   # what distance the verdict compared against δ — the first question of anyone reading the episode later
                   judge_anchor=("plates" if anchor_xyz else "target"), anchors=(anchor_xyz or None),
                   success=bool(hit), end=end, wall_s=wall, drone=(hit or {}).get("drone"),
                   dist_m=(hit or {}).get("dist_m"),
                   dist_anchor_m=(hit or {}).get("dist_anchor_m"), t_success=(hit or {}).get("t"),
                   declarations=len(qf_evs), steps=pairer.n_frames, actions=pairer.n_actions,
                   early_stop=early_stop["v"], online_checks=pairer.online_checks,
                   # if replay fails, in-δ declarations stay without visible and SR hardens into failure — must not be left silent
                   visibility_expected=n_need, visibility_replayed=n_rep,
                   visibility_error=(vis_errs or None),
                   # in async (head/body), each drone has its own step stream — a single team step cannot read it
                   steps_per_drone={(n or "single"): c for n, c in pairer.steps_per_drone.items()},
                   min_dist_m=min_dist, trace=tr_res, ground_z=round(float(ground_z), 3),
                   qf_events=[{k: v for k, v in e.items() if k != "step_events"} for e in qf_evs],
                   step_events=step_evs, exit_code=proc.returncode)
    inv = validate_episode(task, verdict, ground_z, drone_names)
    verdict["valid"] = not inv
    if inv:
        verdict["invalid_reasons"] = inv
        print(f"  ⚠ episode judged invalid: {'; '.join(inv[:3])}", flush=True)
    # per-arm API calls and failures — if head starves, the formation keeps flying on the previous instruction; if body starves, only that
    # drone stands still. Either way the episode stays valid and only SR drops.
    api_fail = count_api_failures_file(log_path)
    verdict["api_failures"] = api_fail
    if api_fail["head_failed"] or api_fail["body_failed"] or api_fail["rate_limited"]:
        print(f"  ! API failures: head {api_fail['head_failed']}/{api_fail['head_calls']} · "
              f"body {api_fail['body_failed']}/{api_fail['body_calls']} · "
              f"429 {api_fail['rate_limited']}", flush=True)
    json.dump(verdict, open(os.path.join(ep_dir, "verdict.json"), "w"), ensure_ascii=False, indent=1)
    return verdict


def main(argv=None):
    ap = argparse.ArgumentParser(description="bench episode flight runner (single and multi)")
    ap.add_argument("tasks", help="editor-v2 episode JSON")
    ap.add_argument("--config", required=True, help="base executor config (path relative to vlmpilot/)")
    ap.add_argument("--ids", default="", help="comma-separated episode ids (default: all)")
    ap.add_argument("--runs", type=int, default=1, help="repetitions per episode")
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", "41451")))
    ap.add_argument("--out", default=os.environ.get("OUT_DIR") or os.path.join(
        PILOT, "outputs", "bench", time.strftime("%Y%m%d_%H%M%S")))
    a = ap.parse_args(argv)

    import airsim
    import yaml
    doc = load_doc(a.tasks)
    defaults = doc.get("defaults") or {}
    ids = [s.strip() for s in a.ids.split(",") if s.strip()]
    tasks = [t for t in doc["tasks"] if not ids or t.get("id") in ids]
    if not tasks:
        raise SystemExit(f"no episode with those ids: {ids}")
    base_cfg = a.config if os.path.isabs(a.config) else os.path.join(PILOT, a.config)
    if not os.path.exists(base_cfg):
        raise SystemExit(f"config not found: {base_cfg}")
    drone_names = list((yaml.safe_load(open(base_cfg)) or {}).get("drone_names") or [])
    ep_timeout = int(os.environ.get("EP_TIMEOUT", "300"))
    grace = int(os.environ.get("CLEANUP_GRACE", "120"))
    os.makedirs(a.out, exist_ok=True)
    print(f"=== bench {os.path.basename(a.tasks)} map={doc.get('map')} episodes {[t['id'] for t in tasks]} × {a.runs} "
          f"· config={os.path.relpath(base_cfg, PILOT)} · drones={drone_names or 'single'} · port={a.port} · δ=per episode · cap {ep_timeout}s",
          flush=True)
    print(f"    ⚠ the sim must be running map='{doc.get('map')}' (the runner does not check the map)", flush=True)

    c = airsim.MultirotorClient(port=a.port, timeout_value=30)
    c.confirmConnection()
    have = c.listVehicles()
    if drone_names and any(n not in have for n in drone_names):
        raise SystemExit(f"sim drones {have} do not include the config drones {drone_names} — was the sim launched with settings.multidrone?")

    # drone health check — right after a restart, drones spawned before terrain loading fall through the floor
    # (measured: 64003 Drone4 stayed fixed 100 m underground while the lane ran with 3 drones).
    for _try in range(3):
        bad = []
        for n in (drone_names or [""]):
            try:
                z = c.simGetVehiclePose(vehicle_name=n).position.z_val
            except Exception:
                z = float("nan")
            if z != z or abs(z) > 10.0:
                bad.append(f"{n}(z={z:+.1f})")
        if not bad:
            break
        print(f"  ! drone anomaly {bad} — trying a respawn via reset {_try + 1}/3", flush=True)
        try:
            c.reset()
            time.sleep(6.0)
        except Exception as e:
            print(f"  ! reset failed {e}", flush=True)
            time.sleep(4.0)
    else:
        raise SystemExit(f"drones did not recover from abnormal positions: {bad} — restart the simulator")

    rows = []
    for task in tasks:
        try:
            target, reused = SC.ensure_scene(c, task, defaults, a.port,
                                             force=os.environ.get("FORCE_SETUP") == "1")
        except Exception as e:
            # if a sim crash (RPC timeout) kills the whole lane, all remaining episodes are lost
            # (measured: E-05·E-20 not run in v2r1). Record per episode and move on to the next.
            print(f"  ! {task['id']}: scene setup failed — {type(e).__name__}: {e}", flush=True)
            for k in range(1, a.runs + 1):
                rows.append(dict(run=k, task=task["id"], success=False, end="scene_error",
                                 error=str(e), wall_s=None, drone=None, dist_m=None,
                                 declarations=0, steps=0, steps_per_drone={}, min_dist_m={}))
            try:
                c = __import__("airsim").MultirotorClient(port=a.port, timeout_value=30)
                c.confirmConnection()
            except Exception:
                pass
            continue
        settle_after_build(reused)          # only freshly built scenes — before t0, so outside the budget
        # ground z: the episode's recorded value if present (valid even when the scene is reused), otherwise setup_target's measured value
        gz_task = SC.ground_z_of(task, defaults)
        ground_z = gz_task if gz_task is not None else getattr(SC, "LAST_GROUND_Z", 0.0)
        for k in range(1, a.runs + 1):
            ep_dir = os.path.join(a.out, task["id"], f"run{k}")
            print(f"\n[{task['id']} run{k}/{a.runs}] {SC.mission_text(task)}  (target={target})", flush=True)
            try:
                v = run_episode(task, defaults, base_cfg, a.port, ep_dir, ep_timeout, grace, target, drone_names,
                                ground_z=ground_z)
                if not v.get("valid", True):
                    # invalid episode (initialization hang, start departure, sinking underground, etc.) — retry once on the spot
                    print(f"  ! {task['id']}: invalid episode — retrying once", flush=True)
                    v = run_episode(task, defaults, base_cfg, a.port, ep_dir, ep_timeout, grace, target,
                                    drone_names, ground_z=ground_z)
                    if not v.get("valid", True):
                        print(f"  ⚠ {task['id']}: the retry is also invalid — needs a manual check", flush=True)
            except Exception as e:
                print(f"  ! run failed: {type(e).__name__}: {e}", flush=True)
                v = dict(task=task["id"], success=False, end="error", error=str(e), wall_s=None, drone=None,
                         dist_m=None, declarations=0, steps=0, steps_per_drone={}, min_dist_m={})
            rows.append(dict(run=k, **{kk: vv for kk, vv in v.items() if kk != "step_events"}))
            md = v.get("min_dist_m") or {}
            best = min([x for x in md.values() if x is not None] or [float("nan")])
            # in centralized mode the formation is captured together in one frame, so every per-drone count equals steps — only then "team steps".
            # in async mode each drone captures separately and steps is the per-drone sum, so reading it as team steps looks 4x larger.
            spd = v.get("steps_per_drone") or {}
            steps_txt = (f"team steps {v['steps']}" if all(c == v["steps"] for c in spd.values()) else
                         f"frames {v['steps']} (per drone {'/'.join(str(spd[n]) for n in sorted(spd))})")
            print(f"[{task['id']} run{k}] {'✅' if v['success'] else '❌'} end={v['end']}"
                  f"{'(early)' if v.get('early_stop') else ''} "
                  f"{'dist=%.1fm by %s t=%ss' % (v.get('dist_anchor_m') or v['dist_m'], v['drone'] or 'single', v.get('t_success')) if v['success'] else ''} "
                  f"declarations {v['declarations']} · {steps_txt} · "
                  # the nearest approach comes from the trajectory sampler (target center) — it may differ from the judged distance (anchors), so say so
                  f"closest{'(center)' if v.get('judge_anchor') == 'plates' else ''} {best:.1f}m · {v['wall_s']}s",
                  flush=True)
            time.sleep(2)

    anchored = sorted({r.get("judge_anchor") or "target" for r in rows})
    print(f"\n=== per-episode performance (flight · δ = episode arrival_radius_m · verdict distance = {'/'.join(anchored)} "
          f"· visibility = capture time, replay fallback) ===", flush=True)
    print(f"{'episode':8} {'SR':>7} {'decl':>5} {'t_succ':>10} {'closest med':>12}", flush=True)
    for task in tasks:
        rs = [r for r in rows if r["task"] == task["id"]]
        ok = sum(1 for r in rs if r["success"])
        ts = [r["t_success"] for r in rs if r.get("t_success") is not None]
        bests = sorted(min([x for x in (r.get("min_dist_m") or {}).values() if x is not None] or [math.inf]) for r in rs)
        med = bests[len(bests) // 2] if bests else math.inf
        print(f"{task['id']:8} {ok}/{len(rs):>5} {sum(r['declarations'] for r in rs):>5} "
              f"{(sum(ts) / len(ts)) if ts else 0:>9.0f}s {med:>11.1f}m", flush=True)
    json.dump(dict(tasks_file=os.path.abspath(a.tasks), config=os.path.relpath(base_cfg, PILOT),
                   drone_names=drone_names, port=a.port, ep_timeout=ep_timeout, rows=rows),
              open(os.path.join(a.out, "summary.json"), "w"), ensure_ascii=False, indent=1)
    print(f"summary -> {os.path.join(a.out, 'summary.json')}", flush=True)
    return 0


if __name__ == "__main__":
    if signal.getsignal(signal.SIGINT) == signal.SIG_IGN:      # launched with nohup &, SIGINT is ignored (same reason as run_testset)
        signal.signal(signal.SIGINT, signal.default_int_handler)
    sys.exit(main())
