#!/usr/bin/env python3
# DroneCATS
# Copyright (c) 2026-present NAVER Cloud Corp.
# Apache-2.0

"""Generate N episodes automatically: reset → B → join → judge.

**Why.** All pipeline pieces existed but were chained by hand: per episode one reset,
one port cleanup, one B run, one join, one judge. Not feasible for hundreds of episodes.

**A stays up for the whole batch.** Isaac takes ~100 s to load Brownstone, so restarting
per episode would dominate throughput. Instead only the vehicle is reset with
`scripts/isaacsim/pegasus_reset.py`, which also restarts PX4 and so clears link contention.

    # A (another terminal), kept up for the whole batch
    $ISAACSIM/python.sh scripts/isaacsim/sim/launch_brownstone.py --variant Morning \
        --spawn-xy=-14,-36 --targets 'trashcan:orange' \
        --gt --gt-dir /tmp/batch_gt --gt-live --vehicles 1 --render-hz 10

    # B (this tool)
    python3 verifier/isaacsim/batch_run.py --episodes 10 --gt-dir /tmp/batch_gt \
        --config vlmpilot/configs/isaacsim/single-drone/pegasus_oracle.yaml \
        --instruction "go to the orange trash can" --out outputs/batch01

B can be started immediately: it waits for A to come up (`--wait-a-ready-s`, 900 s by
default, polled every 2 s) instead of needing a fixed sleep between the two commands.

For a different instruction per episode, pass JSONL via `--plan` (the plan generator's format):

    {"instruction": "go to the orange trash can", "spawn": "-14,-36,0.07", "yaw": 0}

The plans in `episodes/isaacsim/single-drone/` use this format for the 4 DroneCATS tasks. Task items have
three extra fields:

    task              approaching | searching | tracking | search_tracking
    target_move       toggles target motion at reset (static vs moving without restarting A)
    yaw_candidates    yaw candidates for searching tasks. **"0 detections" is checked on live
                      GT**; on mismatch the next candidate is tried. Nothing is assumed

**Failures are classified.** Measured: link failures (`AioRpcError`) occur in 1 of 3-4 runs.
Counting them as failed episodes would measure infra stability, not oracle quality.
So reasons are split and **only transient failures** are retried:

    link      AioRpcError / Stream removed / connect failure   → retry
    timeout   B did not finish within the cap                  → retry
    a_dead    A died (live GT is stale)                         → stop batch
    no_finish ended without a finished declaration              → policy outcome, counted
    graded    judge produced a grade (ok/too_far/not_visible)   → policy outcome

Outputs
    <out>/batch.jsonl          one line per episode (with attempt history)
    <out>/batch_summary.json   SR, per-reason counts, attempt distribution, wall clock
    <out>/ep_NN/B.log          B log for that episode (look here first on failure)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import signal
import statistics
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)                       # judge_run · join_gt

# Interpreter for B, relative to the repo root (B runs with cwd=REPO)
DEFAULT_B_PYTHON = "vlmpilot/.venv/bin/python"

# Link-failure fingerprints, as documented in `pegasus_link.py` comments
LINK_PAT = re.compile(
    r"AioRpcError|Stream removed|Connection refused|failed to connect|"
    r"could not connect to the drone|Timeout waiting for drone", re.I)
RUNDIR_PAT = re.compile(r"run\.json written:\s*(\S+?)/run\.json\s*\(outcome=([^)]+)\)")

# Ports: slot 0 defaults. `--slot` replaces them with that slot's values
B_PORTS = ((14540, "udp"), (50051, "tcp"))
# Cleanup targets. Patterns live **in this file**; as shell args pkill would match its own shell
B_PROCS = ("mavsdk_server", "run_scripted_episode")
# With `--keep-mavsdk` the server is excluded: never kill the long-lived server we started
B_PROCS_KEEP_SRV = ("run_scripted_episode",)

sys.path.insert(0, os.path.join(REPO, "vlmpilot", "src"))


# ── processes / ports ─────────────────────────────────────────────────────
def _proc_cmdlines():
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as fh:
                cmd = fh.read().replace(b"\0", b" ").decode("utf-8", "replace")
        except OSError:
            continue
        yield int(pid), cmd


def _ancestors(pid=None):
    """Set of PIDs from self up to init."""
    pid = os.getpid() if pid is None else pid
    seen = set()
    while pid and pid not in seen:
        seen.add(pid)
        try:
            with open(f"/proc/{pid}/stat", encoding="utf-8") as fh:
                # comm may contain spaces/parens, so read after the last ')'
                fields = fh.read().rsplit(")", 1)[1].split()
            pid = int(fields[1])           # PPid
        except (OSError, IndexError, ValueError):
            break
    return seen


def _slot_of_pid(pid):
    """Slot tag of the process. 0 if untagged (a single user not using slots)."""
    try:
        from vlmpilot.controller.isaacsim import pegasus_layout as _pl
    except Exception:  # noqa: BLE001 — the batch must run even without vlmpilot
        return 0
    return _pl.slot_of_pid(pid) or 0


def kill_b_leftovers(verbose=True, slot=0, keep_server=False):
    """Kill leftover B processes, **for this slot only.**

    Two reasons not to use `pkill -f`:

    * The pattern **appears in our own shell's cmdline** and kills itself (observed 5
      times). So `/proc` is scanned directly and **the whole ancestor chain is excluded**;
      excluding only the parent still kills a higher shell holding the same string.
    * With several slots on one pod the pattern also matches **other slots' `mavsdk_server`**,
      cutting their flights mid-air. So processes are further filtered by slot tag.
    """
    me = _ancestors()
    pats = B_PROCS_KEEP_SRV if keep_server else B_PROCS
    hits = [(pid, cmd) for pid, cmd in _proc_cmdlines()
            if pid not in me and any(p in cmd for p in pats)
            and _slot_of_pid(pid) == int(slot)]
    for sig in (signal.SIGTERM, signal.SIGKILL):
        alive = []
        for pid, cmd in hits:
            try:
                os.kill(pid, sig)
                alive.append((pid, cmd))
            except OSError:
                pass
        if not alive:
            break
        time.sleep(1.5)
    if verbose and hits:
        print(f"  [CLEAN] slot {slot}: killed {len(hits)} leftovers: "
              + ", ".join(str(p) for p, _ in hits), flush=True)
    return len(hits)


TCP_LISTEN = "0A"          # st column of /proc/net/tcp


def _port_busy(port, proto):
    """Check local port use by reading `/proc/net/*`, with no bind side effects.

    TCP counts **LISTEN only**. Closed sockets lingering in `TIME_WAIT` are not in use;
    counting them made the previous `mavsdk_server` trigger a "50051 still busy after
    45 s" warning every time (observed).
    """
    want = f"{port:04X}"
    for fam in (proto, proto + "6"):
        try:
            with open(f"/proc/net/{fam}", encoding="utf-8") as fh:
                next(fh, None)
                for line in fh:
                    cols = line.split()
                    if len(cols) < 4 or cols[1].rsplit(":", 1)[-1] != want:
                        continue
                    if proto == "tcp" and cols[3] != TCP_LISTEN:
                        continue
                    return True
        except (OSError, StopIteration):
            continue
    return False


def wait_ports_free(timeout=45.0, verbose=True, ports=None, slot=0):
    """Wait **until the ports are actually free**.

    Same failure as documented in `pegasus_link.py`: `kill()` only sends a signal, so the
    dying previous `mavsdk_server` answers the connect query, the connection looks fine,
    and the next call is refused on 50051. Sending the signal is not enough.
    """
    ports = B_PORTS if ports is None else ports
    t0 = time.time()
    while time.time() - t0 < timeout:
        busy = [f"{p}/{proto}" for p, proto in ports if _port_busy(p, proto)]
        mine = _ancestors()
        alive = any(pid not in mine and any(x in cmd for x in B_PROCS)
                    and _slot_of_pid(pid) == int(slot)
                    for pid, cmd in _proc_cmdlines())
        if not busy and not alive:
            if verbose:
                print(f"  [CLEAN] ports free ({time.time() - t0:.0f} s)", flush=True)
            return True
        time.sleep(1.0)
    busy = [f"{p}/{proto}" for p, proto in ports if _port_busy(p, proto)]
    print(f"  [CLEAN] ⚠️ still busy after {timeout:.0f} s: {busy or 'processes left'}", flush=True)
    return False


PX4_PORT = (14580, "udp")     # slot 0 PX4 onboard MAVLink; restart-ready signal


def wait_px4_ready(timeout=60.0, settle_s=20.0, verbose=True, port=None,
                   a_log=None, hb_before=None):
    """Wait until PX4 is back up, then add a settle margin.

    **Why.** The reset tool only guarantees "vehicle at spawn, landed"; PX4's MAVLink
    stream may not be ready yet. Starting B then passes the connect query and drops
    **on the first RPC**:

        [LINK] MAVSDK connected → AioRpcError: Stream removed (Connection reset by peer)

    Measured: 3/3 failures in this race. `14580/udp` being bound again is used as the
    ready signal, plus a short margin (where a manual `sleep 4` used to go).

    **The port alone is not enough** (measured). Even with the PX4 process up and the
    port bound, **the link to the sim (A) may not exist yet.** Starting B then attaches
    MAVSDK but PX4 has no position estimate:

        [LINK] wait failed: local_position_ok=False armable=False home_ok=False

    In one episode A's log printed `Received first hearbeat` **3 min 30 s after** the
    reset, and B had given up 8 s earlier. So with `a_log`, we wait until **A receives a
    new heartbeat**, the only direct signal that PX4 is really attached to the sim.
    """
    port = PX4_PORT if port is None else port
    t0 = time.time()
    while time.time() - t0 < timeout:
        if _port_busy(*port):
            if a_log and not wait_sim_link(a_log, hb_before, timeout - (time.time() - t0),
                                           verbose=verbose):
                return False
            time.sleep(settle_s)
            if verbose:
                print(f"  [PX4] ready ({time.time() - t0:.0f} s, settle {settle_s:.0f} s)",
                      flush=True)
            return True
        time.sleep(1.0)
    print(f"  [PX4] ⚠️ after {timeout:.0f} s, {port[0]} still not bound", flush=True)
    return False


HEARTBEAT_PAT = re.compile(r"Received first hearbeat")   # keeps pegasus's typo verbatim


def count_heartbeats(a_log):
    """Heartbeat count in A's log. Comparing before/after reset tells if PX4 re-attached."""
    if not a_log:
        return None
    try:
        with open(a_log, "rb") as fh:
            return len(HEARTBEAT_PAT.findall(fh.read().decode("utf-8", "replace")))
    except OSError:
        return None


def a_path_gate(a_log, gt_dir, wait_s=60.0, verbose=True):
    """`None` to pass, otherwise **the reason for blocking**.

    `a_alive()` checks the slot sidecar (`/tmp/vlmpilot_pegasus_s5/latest_gt.json`), so it
    **passes even when paths mismatch.** Through that hole, 9 hours of 8 models produced 0
    episodes: A wrote `gt.sr_X` / `A.sr_X.log` while B was given `gt.X` / `A.X.log` (the
    launcher prefixes names with `sr_`). B watched an empty log, timed out with "A got no
    heartbeat within 599 s", then died joining nonexistent GT with `FileNotFoundError`.
    No message said "wrong path".

    So before the batch we check **that A is actually writing to the paths B received**.
    A may have just started, so we wait up to `wait_s`.
    """
    t0 = time.time()
    while True:
        bad = []
        if a_log:
            try:
                n = os.path.getsize(a_log)
            except OSError:
                n = -1
            if n <= 0:
                bad.append(f"--a-log {a_log} "
                           + ("does not exist" if n < 0 else "is 0 bytes")
                           + "; A is writing to another file")
        if gt_dir and not os.path.isdir(gt_dir):
            bad.append(f"--gt-dir {gt_dir} does not exist; A is writing GT elsewhere")
        if not bad:
            return None
        if time.time() - t0 >= wait_s:
            return ("[BATCH] ⚠️ A is not writing to **the paths B received** "
                    f"(waited {wait_s:.0f} s):\n"
                    + "".join(f"   · {b}\n" for b in bad)
                    + "   Compare with the paths A's launch command creates. A launcher\n"
                      "   that prefixes names breaks this. Starting now would yield **0 episodes**.")
        if verbose and time.time() - t0 < 3.0:
            print(f"  [BATCH] checking A paths: {bad[0]}", flush=True)
        time.sleep(2.0)

def wait_sim_link(a_log, before, timeout, verbose=True):
    """Wait until A receives a **new** heartbeat."""
    if before is None:
        return True
    t0 = time.time()
    while time.time() - t0 < timeout:
        now = count_heartbeats(a_log)
        if now is not None and now > before:
            if verbose:
                print(f"  [PX4] sim link up ({time.time() - t0:.0f} s): "
                      f"A received a heartbeat", flush=True)
            return True
        time.sleep(1.0)
    print(f"  [PX4] ⚠️ A got no heartbeat within {timeout:.0f} s: "
          f"PX4 did not attach to the sim", flush=True)
    return False


# ── A monitoring ──────────────────────────────────────────────────────────
def read_live_gt(path):
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None


def start_state(gt_live_path, spawn, delta_m, visible_px=50):
    """Read the start state **before** launching B; a bad start makes the data unusable.

    Checks:

    **1. Is the vehicle really at spawn.** The reset tool restores the USD pose, but with
    `--no-px4` keeping PX4 alive **the EKF still trusts the previous position.** Taking off
    then drags the vehicle there; measured, one episode started at 4.52 m and ended in 1 step.

    **2. Is the target already within the arrival radius.** Then the answer is `finished`
    from step one and there is nothing to navigate. Counting it as success inflates SR.

    **3. Is the target visible at start.** This is the DroneCATS search axis
    (`TaskVerdict.seen_at_start`). **The value measured here is advisory.** Right after
    reset the vehicle is landed (z ≈ 0.07 m) with the camera at ground level, while
    `seen_at_start` is defined on **the first frame after takeoff**. Measured: approaching
    episodes showed "not visible" here; a 0.8 m box 10 m ahead is out of view from the ground.

    So this value is only recorded as `seen_at_start_pregate`; **the canonical value comes
    from the first step of the joined GT** (= first capture after takeoff, `first_step_seen`).
    Yaw is chosen by **geometry**, not mask: if the yaw offset exceeds half the FOV, the
    target is out of view by construction.
    """
    d = read_live_gt(gt_live_path)
    if not d:
        return {"ok": False, "why": "could not read live GT"}
    cam = d.get("camera_xyz_m") or []
    tgts = d.get("targets") or {}
    dists = [t.get("dist_m") for t in tgts.values()
             if isinstance(t.get("dist_m"), (int, float))]
    near = min(dists) if dists else None
    # Moving target first: must match the judge/oracle rule to look at the same target
    meta = d.get("meta") or {}
    moving = [k for k in sorted(tgts) if (meta.get(k) or {}).get("move")]
    prim = moving[0] if moving else (sorted(tgts)[0] if tgts else None)
    mask = int((tgts.get(prim) or {}).get("mask_px") or 0) if prim else 0
    out = {"ok": True, "camera_xyz_m": [round(v, 3) for v in cam],
           "start_dist_m": round(near, 3) if near is not None else None,
           "target": prim, "start_mask_px": mask,
           "seen_at_start_pregate": bool(mask >= int(visible_px))}
    if spawn and len(cam) >= 2:
        try:
            sx, sy = (float(v) for v in spawn.split(",")[:2])
        except ValueError:
            sx = sy = None
        if sx is not None:
            off = ((cam[0] - sx) ** 2 + (cam[1] - sy) ** 2) ** 0.5
            out["spawn_offset_m"] = round(off, 3)
    if near is not None and near <= delta_m:
        out.update({"ok": False, "why": f"within arrival radius at start ({near:.2f} m ≤ {delta_m} m)"})
    return out


def a_alive(gt_live_path, max_age_s):
    """Is A alive, judged by the age of the live GT file."""
    try:
        age = time.time() - os.path.getmtime(gt_live_path)
    except OSError:
        return False, None
    return age <= max_age_s, age


def wait_a_ready(gt_live_path, max_age_s, wait_s, verbose=True, poll_s=2.0):
    """Wait until A is ready. Returns `(alive, age, waited_s)`.

    A's startup takes time (measured 132 s on Brownstone: extension loading 26 · scene load 30 ·
    material rebinding and colliders 32 · target promotion and first GT tick 44). Checking
    `a_alive()` only once made callers add a fixed `sleep 150~210`, which is wrong both ways:
    too short and the whole batch stops with `a_dead` (no retry), too long and the difference
    is wasted. Polling uses only the time it takes. Progress is printed every 10 s so a wait
    is not mistaken for a hang, and the actual wait is reported at the end.
    """
    t0 = time.time()
    live, age = a_alive(gt_live_path, max_age_s)
    if live or wait_s <= 0:
        return live, age, 0.0
    if verbose:
        print(f"[BATCH] waiting for A: {gt_live_path} (up to {wait_s:.0f} s)", flush=True)
    last_note = 0.0
    while True:
        waited = time.time() - t0
        if waited >= wait_s:
            return False, age, waited
        time.sleep(poll_s)
        live, age = a_alive(gt_live_path, max_age_s)
        waited = time.time() - t0
        if live:
            if verbose:
                print(f"[BATCH] A ready after {waited:.0f} s (live GT age {age:.1f} s)", flush=True)
            return True, age, waited
        if verbose and waited - last_note >= 10.0:
            last_note = waited
            state = "missing" if age is None else f"{age:.0f} s old"
            print(f"[BATCH]   … {waited:.0f} s: live GT {state}", flush=True)


PARKING_FAR_M = 1000.0
"""Distance that tells whether a target has left PARKING (`targets.PARKING` = 5000,5000).

Same threshold as A's `launch_brownstone._count_placed(parking_far_m=1000.0)`; if the two
differ, A can say "placed" while B says "not placed".
"""


def placed_count(gt_live_path, far_m=PARKING_FAR_M):
    """Number of targets that **left PARKING** in the live GT, and the total. `(None, None)` if unreadable.

    All targets spawn at `PARKING` and are moved to their places after play, so one distance
    tells placement. Do not require *all* of them: decoys this batch does not use stay at
    PARKING (measured p28: 7 of 57 were 7 km away).
    """
    d = read_live_gt(gt_live_path)
    if not d:
        return None, None
    tg = d.get("targets") or {}
    dists = [t.get("dist_m") for t in tg.values()
             if isinstance(t.get("dist_m"), (int, float))]
    if not dists:
        return None, None
    return sum(1 for x in dists if x < float(far_m)), len(dists)


def wait_targets_placed(gt_live_path, wait_s=300.0, far_m=PARKING_FAR_M,
                        stable_n=3, verbose=True, poll_s=2.0):
    """Wait until the targets are **in place**. `None` means pass, otherwise the reason it blocked.

    **Fresh GT does not mean the scene is ready.** A opens the GT sidecar and says B may start
    **before** it places the targets (measured A log: `[SIM] ready` line 12278 · `[GT] sidecar
    ready` 12298 · `[DG] places set for 57 targets` 13441). A B started in that window sees every
    target at PARKING: episode 1 started 7,230 m / 7,122 m away (the distance to (5000,5000)).
    The start-distance check only rejects starts that are too *close*, so the episode runs,
    chases a target it can never find and ends `no_finish`, which is recorded as a model failure.

    A fixed `sleep 150~210` used to cover this window by accident; waiting for A by polling
    removes that cover, so it is checked here directly. Placement fills up over several ticks,
    so "non-zero" is not enough: it passes when the count is the same `stable_n` times in a row.
    """
    t0 = time.time()
    if wait_s <= 0:
        return None
    seen, last_note = [], 0.0
    while True:
        n, total = placed_count(gt_live_path, far_m)
        if n:
            seen.append(n)
            if len(seen) >= stable_n and len(set(seen[-stable_n:])) == 1:
                if verbose:
                    print(f"[BATCH] targets placed: {n}/{total} in place "
                          f"({time.time() - t0:.0f} s)", flush=True)
                return None
        else:
            seen = []
        waited = time.time() - t0
        if waited >= wait_s:
            return (f"[BATCH] **targets are not in place yet**: after {wait_s:.0f} s, "
                    f"{n if n is not None else 'unreadable'}/{total if total is not None else '?'} "
                    f"are placed. A says B may start *before* placing targets, so fresh GT is "
                    f"not enough. Running now would fly with targets at PARKING (5000,5000) and "
                    f"give `no_finish` for every episode. Check A's log for the target places line")
        if verbose and waited - last_note >= 10.0:
            last_note = waited
            print(f"[BATCH]   … {waited:.0f} s: placed targets "
                  f"{n if n is not None else '?'}/{total if total is not None else '?'}",
                  flush=True)
        time.sleep(poll_s)


def active_targets_of(item):
    """Target prims this episode actually uses: 1 target + 0-3 decoys.

    The scene holds props for 28 episodes at once (32 pairs + 38 decoys = 70). Only those
    this episode sees or is judged on are returned. Without `target_prim`, returns
    **`None`** meaning "all": when unknown, keeping the old behavior is safer.
    """
    meta = item.get("meta") or {}
    prim = meta.get("target_prim")
    if not prim:
        return None
    out = [str(prim)]
    for d in (meta.get("decoys") or []):
        dp = (d or {}).get("prim")
        if dp:
            out.append(str(dp))
    return out


# ── one episode ───────────────────────────────────────────────────────────
def run_reset(spawn, yaw, timeout, log_fh, restart_px4=False, slot=0,
              target_move=None, active_targets=None):
    """Return the vehicle to the spawn point.

    `restart_px4=False` (default) resets **pose only**. With PX4 kept alive, B simply
    reconnects and there is no restart race at all; throughput hinges on this. PX4 is
    restarted only on failed attempts (when EKF/failsafe got stuck).
    """
    # Use the `--pos=-14,...` form: with a space, negatives parse as options (observed)
    cmd = [sys.executable, os.path.join(REPO, "scripts", "isaacsim", "pegasus_reset.py")]
    if int(slot):
        cmd.append(f"--slot={int(slot)}")   # without it, slot 0's vehicle is reset
    if not restart_px4:
        cmd.append("--no-px4")
    if spawn:
        cmd.append(f"--pos={spawn}")
    if yaw is not None:
        cmd.append(f"--yaw={yaw}")
    if target_move is not None:
        # In DroneCATS (approaching, searching) are static and (tracking,
        # search_tracking) moving. Switched per episode without restarting A.
        cmd.append(f"--target-move={'on' if target_move else 'off'}")
    if active_targets is not None:
        # Stop movers of targets this episode does not use. Only with `--mover-active`.
        cmd.append(f"--active-targets={','.join(active_targets)}")
    log_fh.write(f"$ {' '.join(cmd)}\n")
    log_fh.flush()
    try:
        p = subprocess.run(cmd, cwd=REPO, timeout=timeout,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        log_fh.write(p.stdout.decode("utf-8", "replace"))
        log_fh.flush()
        return p.returncode == 0
    except subprocess.TimeoutExpired:
        log_fh.write(f"[RESET] exceeded {timeout} s\n")
        return False


def run_b(b_python, config, instruction, timeout, log_path, slot=0,
          keep_server=False):
    """Run B and return (exit code, timed_out, log body, run_dir, outcome).

    **Only the part written by this attempt is read.** The log is appended per attempt;
    previously the whole file was read and `RUNDIR_PAT.search` took the **first** match,
    so a retry inherited the failed first attempt's result. Measured on a PX4 moving-target
    batch: episodes 1 and 2 **actually arrived** on retry (12 and 25 steps) but both were
    recorded as `error: AioRpcError`. Successful data was silently discarded.

    For the same reason `body` (used by the caller to classify link failures) must be from
    this attempt, or the previous attempt's error fingerprint keeps following.
    """
    cmd = [b_python, os.path.join(HERE, "run_scripted_episode.py"),
           "--config", config]
    if instruction:
        cmd += ["--instruction", instruction]
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    # Set the slot tag: B and its child (mavsdk_server) inherit it so cleanup can pick
    # only this slot's processes. The child is exec'd, so it stays in /proc/environ.
    env["VLMPILOT_SLOT"] = str(int(slot))
    # File size **before** this attempt writes; this attempt starts here
    start = os.path.getsize(log_path) if os.path.exists(log_path) else 0
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write(f"$ {' '.join(cmd)}\n")
        fh.flush()
        try:
            p = subprocess.run(cmd, cwd=REPO, env=env, timeout=timeout,
                               stdout=fh, stderr=subprocess.STDOUT)
            rc, timed_out = p.returncode, False
        except subprocess.TimeoutExpired:
            rc, timed_out = None, True
            fh.write(f"\n[BATCH] B exceeded {timeout} s, killed\n")
    kill_b_leftovers(verbose=False, slot=slot, keep_server=keep_server)
    # Read as bytes and slice; `errors="replace"` handles a cut mid-character
    with open(log_path, "rb") as fh:
        fh.seek(start)
        body = fh.read().decode("utf-8", errors="replace")
    m = RUNDIR_PAT.search(body)
    run_dir = m.group(1) if m else None
    outcome = m.group(2) if m else None
    if run_dir and not os.path.isabs(run_dir):
        run_dir = os.path.join(REPO, run_dir)
    return (rc, timed_out, body, run_dir, outcome)


def _write_plan_meta(run_dir, item):
    """Write `plan_meta.json` into the run folder so the judge tool knows the target."""
    try:
        doc = {k: item[k] for k in ("id", "task", "target_move", "instruction")
               if item.get(k) is not None}
        doc["plan_meta"] = item.get("meta") or {}
        with open(os.path.join(run_dir, "plan_meta.json"), "w",
                  encoding="utf-8") as fh:
            json.dump(doc, fh, ensure_ascii=False, indent=1)
    except OSError as exc:
        print(f"  [WARN] could not write plan_meta.json: {exc!r}", flush=True)


def first_step_dist(run_dir):
    """Target distance at the first joined step; catches "within radius at start" afterwards.

    With `--px4-reset never/auto` PX4 stays alive, only the USD pose is restored, and
    **PX4's EKF trusts the previous position.** The vehicle is dragged there right after
    takeoff, while the pre-gate passes (the stage pose is correct). Measured: such an
    episode started at 4.52 m and ended in 1 step; counting it as success inflates SR.
    """
    from join_gt import load_jsonl
    rows = load_jsonl(os.path.join(run_dir, "gt.jsonl"))
    if not rows:
        return None
    first = min(rows, key=lambda r: r.get("step", 0))
    dists = [t.get("dist_m") for t in (first.get("targets") or {}).values()
             if isinstance(t.get("dist_m"), (int, float))]
    return round(min(dists), 3) if dists else None


def first_step_seen(run_dir, visible_px=50, target=None, gt_dir=None):
    """Was the target visible at the first step (= first capture after takeoff): **canonical `seen_at_start`.**

    The verifier runner also measures on the first episode frame. The pre-gate value
    differs because the camera is at landed height.
    """
    from join_gt import load_jsonl
    path = os.path.join(run_dir, "gt.jsonl")
    if not os.path.exists(path) and gt_dir:
        # Join here if not yet joined, since this is called **before** the judge
        from join_gt import main as join_main
        join_main(["--run-dir", run_dir, "--gt-dir", gt_dir])
    if not os.path.exists(path):
        return None, None
    rows = load_jsonl(path)
    if not rows:
        return None, None
    first = min(rows, key=lambda r: r.get("step", 0))
    tgts = first.get("targets") or {}
    meta = first.get("meta") or {}
    if target and target in tgts:
        prim = target
    else:
        moving = [t for t in sorted(tgts) if (meta.get(t) or {}).get("move")]
        prim = moving[0] if moving else (sorted(tgts)[0] if tgts else None)
    if prim is None:
        return None, None
    mask = int((tgts.get(prim) or {}).get("mask_px") or 0)
    return bool(mask >= int(visible_px)), mask


def join_stats(run_dir):
    """Join quality; without it we cannot later answer whether the data is trustworthy."""
    from join_gt import load_jsonl
    rows = load_jsonl(os.path.join(run_dir, "gt.jsonl"))
    dts = [abs(r["join"]["dt_s"]) for r in rows
           if isinstance((r.get("join") or {}).get("dt_s"), (int, float))]
    steps = load_jsonl(os.path.join(run_dir, "steps.jsonl"))
    return {"steps": len(steps), "matched": len(rows),
            "dt_median_ms": round(statistics.median(dts) * 1000, 1) if dts else None,
            "dt_max_ms": round(max(dts) * 1000, 1) if dts else None}


def _reset_and_check(k, item, args, cfg, log_path, restart_px4, yaw):
    """Reset → cleanup → settle → start state."""
    hb_before = count_heartbeats(args.a_log) if restart_px4 else None
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write(f"\n===== episode {k} · {time.strftime('%H:%M:%S')} "
                 f"· PX4 restart {restart_px4} · yaw {yaw} =====\n")
        ok_reset = run_reset(item.get("spawn", args.spawn), yaw,
                             args.reset_timeout, fh, restart_px4=restart_px4,
                             slot=args.slot, target_move=item.get("target_move"),
                             active_targets=(active_targets_of(item)
                                             if args.mover_active else None))
    if not ok_reset:
        print("  [RESET] failed; continuing anyway (A may have missed the request)", flush=True)

    kill_b_leftovers(slot=args.slot, keep_server=bool(args.keep_mavsdk))
    wait_ports_free(args.port_timeout, ports=args.b_ports_check, slot=args.slot)
    if restart_px4:
        wait_px4_ready(args.px4_timeout, args.settle_s, port=args.px4_port,
                       a_log=args.a_log, hb_before=hb_before)
    elif args.settle_s:
        time.sleep(args.settle_s)

    live, age = a_alive(cfg["gt_live_path"], args.a_max_age_s)
    if not live:
        return None, {"reason": "a_dead", "success": False,
                      "detail": f"live GT age "
                                f"{age if age is None else round(age, 1)} s "
                                f"> {args.a_max_age_s} s; A stalled or died"}

    st = start_state(cfg["gt_live_path"], item.get("spawn", args.spawn),
                     args.delta_m, visible_px=args.visible_px)
    off = st.get("spawn_offset_m")
    if off is not None and off > args.spawn_tol_m:
        st.update({"ok": False,
                   "why": f"{off:.2f} m away from spawn (> {args.spawn_tol_m} m)"})
    return st, None


def _pick_yaw(item, default_yaw, fov_deg):
    """Start yaw: for searching tasks, pick a candidate **clearly beyond half the FOV**.

    A geometric version of `scenario_spec`'s "the one of ±90/±135/180 with 0 detections".
    With 90° FOV the half-angle is 45°, so a 90° offset is out of view with 45° margin. If
    no candidate exceeds the half-angle, use the first and warn: it must not silently
    become an approaching episode.
    """
    yaw = item.get("yaw", default_yaw)
    cands = item.get("yaw_candidates")
    if not cands or item.get("expect_seen_at_start") is not False:
        return yaw
    base = (item.get("meta") or {}).get("yaw_to_target_deg")
    if base is None:
        return yaw
    half = float(fov_deg) / 2.0

    def off(c):
        return abs(((float(c) - float(base) + 180.0) % 360.0) - 180.0)

    ok = [c for c in cands if off(c) > half]
    if not ok:
        print(f"  [START] ⚠️ yaw candidates {cands} all within FOV half-angle {half:.0f}°; "
              f"target stays in view. Proceeding with first candidate {cands[0]}°", flush=True)
        return cands[0]
    if abs(float(ok[0]) - float(yaw)) > 1e-6:
        print(f"  [START] yaw {yaw}° → {ok[0]}° (offset {off(ok[0]):.0f}° > half-angle "
              f"{half:.0f}°)", flush=True)
    return ok[0]


def one_episode(k, item, args, cfg, ep_dir, restart_px4=False):
    """Attempt one episode; retries and escalation are up to the caller."""
    os.makedirs(ep_dir, exist_ok=True)
    log_path = os.path.join(ep_dir, "B.log")

    # **Searching tasks are valid only if the target is not visible at start.** Yaw is
    # chosen by **geometry**: beyond half the FOV the target is out of view by construction.
    # Not by mask, because right after reset the vehicle is landed with a ground-level view
    # that differs from the first frame after takeoff (approaching showed "not visible" there).
    want = item.get("expect_seen_at_start")
    yaw = _pick_yaw(item, args.yaw, args.fov_deg)
    item = dict(item, yaw=yaw)
    st, err = _reset_and_check(k, item, args, cfg, log_path, restart_px4, yaw)
    if err:
        return err
    st["yaw_used"] = yaw
    if want is not None:
        st["seen_at_start_expected"] = bool(want)
    if not st["ok"]:
        return {"reason": "bad_start", "success": False, "start": st,
                "detail": st["why"]}

    # **Tell the oracle the episode's target.** Otherwise the config's default rule (moving
    # target first) applies and it **chases a decoy**: measured, the first warehouse episode
    # declared arrival at the wrong target 22.8 m away. Same defect the judge once had.
    cfg_path = args.config
    prim = (item.get("meta") or {}).get("target_prim")
    if prim:
        cfg_path = episode_config(args.config, prim, ep_dir)
    rc, timed_out, body, run_dir, outcome = run_b(
        args.b_python, cfg_path, item.get("instruction", args.instruction),
        args.b_timeout, log_path, slot=args.slot,
        keep_server=bool(args.keep_mavsdk))

    if timed_out:
        return {"reason": "timeout", "success": False, "run_dir": run_dir,
                "outcome": outcome, "detail": f"B exceeded {args.b_timeout} s"}
    if run_dir is None:
        reason = "link" if LINK_PAT.search(body) else "b_error"
        hit = LINK_PAT.search(body)
        return {"reason": reason, "success": False, "rc": rc,
                "detail": (f"link failure: {hit.group(0)}" if hit
                           else f"B exit code {rc}; no run.json written")}

    if outcome and outcome.startswith("error"):
        # run.json exists but the episode ended with an exception; not judged
        hit = LINK_PAT.search(body)
        return {"reason": "link" if hit else "b_error", "success": False,
                "run_dir": run_dir, "outcome": outcome,
                "detail": f"outcome={outcome}"}

    # Judge; judge_run also joins if gt.jsonl is missing
    from judge_run import judge_run
    # **Canonical `seen_at_start`**: first capture after takeoff. May differ from the pre-gate.
    seen0, seen0_px = first_step_seen(run_dir, args.visible_px, target=prim,
                                      gt_dir=args.gt_dir)
    st["seen_at_start"] = seen0
    st["seen_at_start_mask_px"] = seen0_px
    want0 = item.get("expect_seen_at_start")
    if want0 is not None and seen0 is not None and bool(seen0) != bool(want0):
        st["task_valid"] = False       # not the intended task anymore; filtered in summaries
        print(f"  [START] ⚠️ canonical seen_at_start={seen0} (mask {seen0_px} px) differs from"
              f" expected ({want0}); counted with a validity flag", flush=True)
    verdict = judge_run(run_dir, gt_dir=args.gt_dir, delta_m=args.delta_m,
                        visible_px=args.visible_px,
                        require_visible=not args.no_require_visible,
                        target=prim,           # judge on the episode's target too
                        task=item.get("task"),
                        seen_at_start=seen0,
                        end_on=args.end_on)
    d0 = first_step_dist(run_dir)
    if d0 is not None and d0 <= args.delta_m:
        return {"reason": "bad_start", "success": False, "run_dir": run_dir,
                "outcome": outcome, "start": st, "start_dist_step1_m": d0,
                "detail": f"first step already within arrival radius ({d0} m ≤ {args.delta_m} m)"}

    out = {"reason": "no_finish" if verdict.get("reason") == "no_finish" else "graded",
           "success": bool(verdict.get("success")), "run_dir": run_dir,
           "outcome": outcome, "verdict": verdict, "join": join_stats(run_dir),
           "start": st, "start_dist_step1_m": d0}
    # Item identity; without it we cannot later tell which task the episode was
    for key in ("id", "task", "target_move"):
        if item.get(key) is not None:
            out[key] = item[key]
    if item.get("meta"):
        out["plan_meta"] = item["meta"]
    # **Also store it in the run folder.** `batch.jsonl` exists only after the batch and
    # does not travel with a detached run. `judge_run.py` reads the target from
    # `plan_meta`, which the `run.json` of a B (flight) run does not have; without this file,
    # re-judging an archived run would measure the wrong target.
    _write_plan_meta(run_dir, item)
    if LINK_PAT.search(body):
        out["link_warning"] = True      # completed, but with link noise
    return out


# ── batch ─────────────────────────────────────────────────────────────────
def load_plan(path, n, args):
    if not path:
        return [{} for _ in range(n)]
    items = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    if not items:
        sys.exit(f"[BATCH] plan is empty: {path}")
    if n and n != len(items):
        print(f"[BATCH] --plan {len(items)} items · --episodes {n} → running {min(n, len(items))}",
              flush=True)
        items = items[:n]
    return items


def episode_config(path, target_prim, ep_dir):
    """**Derived config** pinning this episode's target; the original is untouched."""
    import yaml

    with open(path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    cfg = dict(cfg)
    cfg["oracle_target"] = target_prim
    os.makedirs(ep_dir, exist_ok=True)
    out = os.path.join(ep_dir, "config.yaml")
    with open(out, "w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh, allow_unicode=True, sort_keys=False)
    return out


def derive_slot_config(path, slot, out_dir, server_address=None):
    """Create a **derived config** with paths/ports replaced by the slot's values.

    The original is not edited: it is hand-maintained, while the slot varies per run, and
    two users may share a config on different slots.

    For slot 0 nothing changes and the original path is returned ("slot 0 is unchanged"),
    keeping conditions byte-identical to earlier runs.
    """
    import yaml

    if int(slot) == 0:
        return path
    from vlmpilot.controller.isaacsim import pegasus_slots as PS

    res = PS.resolve(slot)
    with open(path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    p = res["paths"]
    cfg["frame_path"] = p["frame_path"]
    cfg["depth_path"] = p["depth_path"]
    cfg["sim_info_path"] = p["info_file"]
    cfg["gt_live_path"] = p["gt_live"]
    cfg["gt_hold_path"] = p["gt_hold"]
    cfg["mavsdk_address"] = f"udpin://0.0.0.0:{res['ports']['mavsdk_udp'][0]}"
    cfg["grpc_port"] = res["ports"]["grpc_tcp"][0]
    if server_address:
        # B attaches to an already running server; see `PegasusLink.server_address`
        cfg["mavsdk_server_address"] = str(server_address)
    cfg["output_root"] = os.path.join(str(cfg.get("output_root") or "outputs"),
                                      f"slot{int(slot)}")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"config_slot{int(slot)}.yaml")
    with open(out, "w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh, allow_unicode=True, sort_keys=False)
    return out


def read_config(path):
    import yaml
    with open(path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    cfg.setdefault("gt_live_path", "/tmp/vlmpilot_pegasus/latest_gt.json")
    return cfg


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episodes", type=int, default=10, help="number of episodes")
    ap.add_argument("--mover-active", action="store_true",
                    help="per episode, move **only the targets that episode uses** (1 target + "
                         "0-3 decoys). Off by default: all 70 movers in the scene run as before. "
                         "⚠️ Changes observations: other episodes' targets stand at their start "
                         "points instead of walking, so frames and GT targets differ. Do not use "
                         "for batches compared against earlier measurements")
    ap.add_argument("--config", default="vlmpilot/configs/isaacsim/single-drone/pegasus_oracle.yaml")
    ap.add_argument("--slot", type=int, default=0, metavar="N",
                    help="slot number. Separates ports, file paths and cleanup scope at once. "
                         "**Must match A.** Slot 0 (default) behaves exactly as before. "
                         "Config paths/ports are overridden in a derived config; the original "
                         "config is untouched")
    ap.add_argument("--gt-dir", default=None, help="A's --gt-dir (where gt_stream.jsonl is). "
                                                  "Default: gt/ under the slot root")
    ap.add_argument("--out", default=None, help="batch output directory (default outputs/batch_<timestamp>)")
    ap.add_argument("--instruction", default=None, help="same instruction for all episodes")
    ap.add_argument("--plan", default=None, help="per-episode instructions (JSONL)")
    ap.add_argument("--spawn", default=None, help="reset position x,y,z (default: keep A's spawn)")
    ap.add_argument("--yaw", type=float, default=None)
    ap.add_argument("--retries", type=int, default=2, help="retries for transient failures")
    ap.add_argument("--mavsdk-server-address", default=None, metavar="HOST",
                    help="if set, B **attaches to an already running mavsdk_server** (MAVSDK "
                         "does not spawn one). The caller must start the server. "
                         "For diagnosis: tells whether spawning a server per episode relates "
                         "to `arm` failures")
    ap.add_argument("--keep-mavsdk", action="store_true",
                    help="do **not kill** `mavsdk_server` in between-episode cleanup. "
                         "Use with `--mavsdk-server-address`, otherwise the batch kills "
                         "the server we started")
    ap.add_argument("--b-python", default=DEFAULT_B_PYTHON)
    ap.add_argument("--b-timeout", type=float, default=600.0,
                    help="**Cap for one episode (B), seconds.** Exceeding it kills B mid-flight. "
                         "⚠️ Not to be confused with `--px4-timeout` (PX4 readiness wait). "
                         "Measured: episodes took 300-550 s at RTF 0.11 and 757 s at RTF 0.08 "
                         "(62 targets). The default 600 cuts all of the latter, and the abort "
                         "leaves the vehicle airborne so the next attempt is blocked with "
                         "armable=False: one overrun becomes two failures. Raise it based on RTF")
    ap.add_argument("--reset-timeout", type=float, default=200.0)
    ap.add_argument("--port-timeout", type=float, default=45.0)
    ap.add_argument("--px4-timeout", type=float, default=300.0,
                    help="cap for 14580 being bound after a PX4 restart and (with --a-log) A "
                         "receiving a new heartbeat. Measured: one sim link came up 3 min 30 s "
                         "after reset, so 60 s is not enough")
    ap.add_argument("--settle-s", type=float, default=20.0,
                    help="margin before launching B (s). This is **not the whole wait** but a "
                         "margin on top of it: `wait_px4_ready` already waits for 14580/udp to be "
                         "bound again and for **a new heartbeat in A's log** (the only direct sign "
                         "that PX4 is attached to the sim); measured, that signal arrives in 1 s. "
                         "Neither reset completion nor the port guarantees **armability**, hence "
                         "the margin. **The default 20 is enough**: A/B with 20 · 30 · 150 s, 15 "
                         "runs each, all succeeded with the same connect-to-arm time (15.6-16.3 s). "
                         "An older note that arm() died with 6 s predates the heartbeat fix, so "
                         "values below 20 are untested. Keep `--mover-takeoff-gate-m` on A when "
                         "lowering it, or the settle time becomes start-distance error for movers")
    ap.add_argument("--px4-reset", choices=["auto", "always", "never"], default="always",
                    help="always=restart every time (default). **Restoring only the pose leaves "
                         "PX4's EKF trusting the old position, dragging the vehicle there**: "
                         "measured, one episode started at 4.52 m and ended in 1 step. "
                         "auto=pose only on first try, restart on retries · never=never")
    ap.add_argument("--spawn-tol-m", type=float, default=1.5,
                    help="if the start position is farther than this from spawn, the reset failed")
    ap.add_argument("--a-max-age-s", type=float, default=30.0,
                    help="if live GT is older than this, A is considered dead")
    ap.add_argument("--wait-a-ready-s", type=float, default=900.0, metavar="S",
                    help="before the batch, **wait until A is ready**. This is a give-up limit, "
                         "not a sleep: it polls every 2 s and starts as soon as A is ready. "
                         "Default 900: a warm Brownstone start measured 118-132 s, a cold start "
                         "with an empty shader cache took ~540 s. 0 = do not wait (old behavior)")
    ap.add_argument("--wait-placed-s", type=float, default=300.0, metavar="S",
                    help="before the batch, **wait until the targets are in place**. A starts "
                         "writing GT before placing targets, so `--wait-a-ready-s` alone is not "
                         "enough; starting in that window flies with targets at PARKING "
                         "(5000,5000), 7 km away, and ends in `no_finish`. 0 = no check")
    ap.add_argument("--delta-m", type=float, default=5.0, help="judge distance threshold")
    ap.add_argument("--visible-px", type=int, default=50)
    ap.add_argument("--a-path-wait-s", type=float, default=60.0, metavar="S",
                    help="how long to wait for A to **actually write** to --a-log and --gt-dir "
                         "(default 60 s). Blocks the batch if missing or 0 bytes; the slot "
                         "sidecar check passes even with mismatched paths (measured: 9 hours "
                         "of 8 models yielded 0 episodes through that hole)")
    ap.add_argument("--a-log", default=None,
                    help="A's log file. If set, after a PX4 restart wait **until A receives a "
                         "new heartbeat**; watching only the port launches B before PX4 attaches "
                         "to the sim and fails with `local_position_ok=False` (measured 3 min 30 s delay)")
    ap.add_argument("--fov-deg", type=float, default=90.0,
                    help="camera horizontal FOV (deg). Used to pick searching yaw by geometry: "
                         "the offset must exceed the half-angle for the target to be out of view. "
                         "**Must match A's actual FOV**")
    ap.add_argument("--end-on", default="any", choices=["any", "last"],
                    help="any (default, `defaults.end_on`): success if any declaration "
                         "satisfies. last: only the last declaration (old behavior of this tool)")
    ap.add_argument("--no-require-visible", action="store_true")
    ap.add_argument("--stop-on-a-dead", action="store_true", default=True)
    args = ap.parse_args(argv)

    if not os.path.isabs(args.config):
        args.config = os.path.join(REPO, args.config)
    # ── slot ─────────────────────────────────────────────────────────────────
    from vlmpilot.controller.isaacsim import pegasus_slots as PS

    slot_res = PS.resolve(args.slot)
    args.b_ports = ((slot_res["ports"]["mavsdk_udp"][0], "udp"),
                    (slot_res["ports"]["grpc_tcp"][0], "tcp"))
    # Do **not empty** `b_ports` itself; it is indexed below for reporting
    # (`args.b_ports[0][0]`). A separate list is used only for the "must be free" check:
    # a server we started legitimately holds both the gRPC and MAVSDK UDP ports.
    args.b_ports_check = tuple() if args.mavsdk_server_address else args.b_ports
    args.px4_port = (slot_res["ports"]["px4_offboard_udp"][0], "udp")
    if args.gt_dir is None:
        args.gt_dir = os.path.join(slot_res["paths"]["root"], "gt")

    cfg = read_config(args.config)
    out_dir = args.out or os.path.join(REPO, "outputs",
                                       "batch_" + time.strftime("%Y%m%d_%H%M%S"))
    if not os.path.isabs(out_dir):
        out_dir = os.path.join(REPO, out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # Derived config; the original is untouched
    args.config = derive_slot_config(args.config, args.slot, out_dir,
                                     args.mavsdk_server_address)
    cfg = read_config(args.config)

    items = load_plan(args.plan, args.episodes, args)
    live, age, waited = wait_a_ready(cfg["gt_live_path"], args.a_max_age_s,
                                     args.wait_a_ready_s)
    print(f"[BATCH] {len(items)} episodes · slot {args.slot} · out {out_dir}", flush=True)
    print(f"[BATCH] ports MAVSDK {args.b_ports[0][0]} · gRPC {args.b_ports[1][0]} · "
          f"PX4 {args.px4_port[0]} · GT {args.gt_dir}"
          + (f" · owner {slot_res['owner']!r}" if slot_res.get("owner") else ""),
          flush=True)
    print(f"[BATCH] A live GT {cfg['gt_live_path']}: "
          + (f"fresh ({age:.1f} s)" if live else "missing or stale")
          + (f" · **waited {waited:.0f} s**" if waited >= 1.0 else ""), flush=True)
    if not live:
        print(f"[BATCH] A must be running first (waited {args.wait_a_ready_s:.0f} s). "
              "See the A command in the file header.", flush=True)
        return 2

    placed = wait_targets_placed(cfg["gt_live_path"], wait_s=args.wait_placed_s)
    if placed:
        print(placed, flush=True)
        return 5

    gate = a_path_gate(args.a_log, args.gt_dir, wait_s=args.a_path_wait_s)
    if gate:
        print(gate, flush=True)
        return 5

    rows_p = os.path.join(out_dir, "batch.jsonl")
    t_batch = time.time()
    rows = []
    for k, item in enumerate(items, 1):
        ep_dir = os.path.join(out_dir, f"ep_{k:02d}")
        instr = item.get("instruction", args.instruction)
        print(f"\n[{k}/{len(items)}] {instr!r}", flush=True)
        attempts = []
        res = None
        for attempt in range(1, args.retries + 2):
            t0 = time.time()
            restart_px4 = (args.px4_reset == "always"
                           or (args.px4_reset == "auto" and attempt > 1))
            res = one_episode(k, item, args, cfg, ep_dir, restart_px4=restart_px4)
            res["px4_restart"] = restart_px4
            res["attempt"] = attempt
            res["seconds"] = round(time.time() - t0, 1)
            attempts.append({"attempt": attempt, "reason": res["reason"],
                             "seconds": res["seconds"],
                             "detail": res.get("detail")})
            tag = "success" if res["success"] else res["reason"]
            print(f"  attempt {attempt}: {tag} · {res['seconds']:.0f} s"
                  + (f" · {res.get('detail')}" if res.get("detail") else ""), flush=True)
            if res["reason"] == "a_dead":
                break
            if res["success"] or res["reason"] in ("graded", "no_finish"):
                break                       # policy outcomes are not retried
            # link · timeout · bad_start · b_error are retried
        row = {"episode": k, "instruction": instr,
               "spawn": item.get("spawn", args.spawn), "yaw": item.get("yaw", args.yaw),
               "attempts": attempts, **res}
        rows.append(row)
        with open(rows_p, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        if res["reason"] == "a_dead" and args.stop_on_a_dead:
            print("[BATCH] A died; stopping the batch. Run the rest after restarting A.",
                  flush=True)
            break

    # ── summary ──
    graded = [r for r in rows if r["reason"] in ("graded", "no_finish")]
    ok = [r for r in graded if r["success"]]
    reasons = {}
    for r in rows:
        key = "success" if r["success"] else (
            (r.get("verdict") or {}).get("reason") or r["reason"])
        reasons[key] = reasons.get(key, 0) + 1
    tries = [len(r["attempts"]) for r in rows]
    # **Per-task SR**: DroneCATS reports the 4 tasks separately. A single overall SR
    # hides things like "searching is notably worse".
    by_task = {}
    for r in rows:
        t = r.get("task")
        if not t:
            continue
        b = by_task.setdefault(t, {"episodes": 0, "graded": 0, "success": 0,
                                   "invalid": 0})
        b["episodes"] += 1
        if r["reason"] in ("graded", "no_finish"):
            b["graded"] += 1
        if r["success"]:
            b["success"] += 1
        if ((r.get("verdict") or {}).get("task_valid") is False
                or (r.get("start") or {}).get("task_valid") is False):
            b["invalid"] += 1
    for b in by_task.values():
        b["sr"] = round(b["success"] / b["graded"] * 100, 1) if b["graded"] else None
    summary = {
        "episodes": len(rows), "graded": len(graded), "success": len(ok),
        "sr": round(len(ok) / len(graded) * 100, 1) if graded else None,
        "reasons": reasons, "by_task": by_task, "end_on": args.end_on,
        "attempts_total": sum(tries),
        "attempts_mean": round(sum(tries) / len(tries), 2) if tries else None,
        "link_retries": sum(1 for r in rows for a in r["attempts"]
                            if a["reason"] == "link"),
        "wall_s": round(time.time() - t_batch, 1),
        "per_episode_s": round((time.time() - t_batch) / max(len(rows), 1), 1),
        "config": args.config, "gt_dir": args.gt_dir, "out": out_dir,
    }
    with open(os.path.join(out_dir, "batch_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=1)

    print("\n[BATCH] ── summary ──", flush=True)
    print(f"  episodes {summary['episodes']} · graded {summary['graded']} · "
          f"success {summary['success']}"
          + (f" · SR {summary['sr']}%" if summary["sr"] is not None else ""), flush=True)
    for key, c in sorted(reasons.items(), key=lambda kv: -kv[1]):
        print(f"    {c:3d}  {key}", flush=True)
    if by_task:
        print("  per task: episodes · graded · success · SR · validity flags", flush=True)
        for t in ("approaching", "searching", "tracking", "search_tracking"):
            b = by_task.get(t)
            if not b:
                continue
            print(f"    {t:<16} {b['episodes']:>3} · {b['graded']:>3} · "
                  f"{b['success']:>3} · "
                  + (f"{b['sr']:>5.1f}%" if b["sr"] is not None else "    —")
                  + (f" · ⚠️ {b['invalid']}" if b["invalid"] else ""), flush=True)
    print(f"  mean attempts {summary['attempts_mean']} · link retries {summary['link_retries']}",
          flush=True)
    print(f"  wall clock {summary['wall_s']:.0f} s · per episode {summary['per_episode_s']:.0f} s",
          flush=True)
    print(f"  {rows_p}", flush=True)
    return 0 if graded and len(ok) == len(graded) else 1


if __name__ == "__main__":
    raise SystemExit(main())
