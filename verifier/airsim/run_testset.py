# DroneCATS
# Copyright (c) 2026-present NAVER Cloud Corp.
# Apache-2.0

"""Run the testset with real executor flights and judge with the verifier.

What is measured is not the model alone but **the whole executor system** — prompt, VLM client,
coordinate conventions, back-projection, and control all follow the executor's real-flight path as is, and QF is the executor's arrival declaration.

The verifier does the judging **by itself** (no hooks planted inside the executor). The executor
prints a FINISHED marker to stdout when the model declares arrival — the moment the runner sees that line,
it measures distance (3D GT) and visibility (engine detection) over its own RPC connection. So evaluation runs
with plain main alone.

Arrival declarations are accepted **multiple times** (the same criterion for all four tasks). Ending the episode
on a single declaration that misses the condition kills episodes for "declaring one step too early", so the executor
is launched with VLMPILOT_CONTINUE_AFTER_ARRIVAL=1 to keep flying after a declaration, and each declaration is measured.
If mixed with an old executor that does not know this environment variable, the executor ends at the first declaration, so it silently
falls back to the old single-declaration behavior — the judging code reads that case as is too.

Online verdict + early stop (ONLINE_VERDICT=1, on by default):
  When measuring distance at [FRAME], if the distance is within δ, a visibility detection is attached on the spot as well — the sim keeps
  rendering, so calling detection alone takes ≈8ms on the current frame (no render request). When that frame's answer is confirmed
  as finished, judge immediately with the visibility already attached: if visible, success is confirmed → cut the episode. If not visible, keep flying.
  Distance and visibility come from the same capture pose, so this is by definition the same as post-hoc replay, and no replay/teleport is needed.
  Multi is the same (the success rule is "any one drone", so the verdict does not change after the first confirmation). With ONLINE_VERDICT=0,
  the old behavior (everything replayed post hoc). run_bench_tasks.py follows the same convention.

Measurement guardrail (GUARDRAIL=off|audit|enforce, default off — verifier/airsim/guardrail package):
  For each episode, judges "may this episode go into the results?" (contamination that is not model skill: dead server, host load, stuck spawn,
  sim death, etc.). off does not judge and aggregates every episode (release default). audit only records the verdict
  in the verdict file and report. enforce moves contaminated episodes to _invalid/ and re-flies the same episode (3 times total),
  and for a dead server it cuts the episode and stops that model after 2 consecutive episodes — team measurements run with GUARDRAIL=enforce.
  guardrail_mode is left on rows and verdicts, so even if models run in different modes are mixed in one OUT_DIR, each is filtered
  by its own rule.

If the old control-improvement branch is also present, trajectory quality
improves, but it is not required to run the evaluation.

  PORT=41455 MODELS=gemini37,gpt6-luna-or,claude \
    vlmpilot/.venv/bin/python -m verifier.airsim.run_testset episodes/airsim/single-drone/nh_approaching.json

Multi-port parallel — launch several sims and give each model one port, and they run concurrently.
List **the port numbers you will use yourself** (any free ports, as long as they do not clash with the README slot table):

  PORTS=(51001 51002 51003)                        # ← ports you chose
  MODELS_LIST=(gemini37 gpt6-luna-or claude)                   # one model per port
  RUN=outputs/verifier/$(date +%Y%m%d_%H%M%S)
  for i in ${!PORTS[@]}; do
    sudo bash scripts/airsim/launch_airsim.sh nh ${PORTS[$i]} $((i % 4)) &
  done; wait
  for i in ${!PORTS[@]}; do
    PORT=${PORTS[$i]} MODELS=${MODELS_LIST[$i]} OUT_DIR=$RUN \
      vlmpilot/.venv/bin/python -m verifier.airsim.run_testset episodes/airsim/single-drone/nh_approaching.json &
  done; wait
  # Giving the same OUT_DIR is safe — episode outputs split into <model>/ subfolders.

Per episode: reset drone → teleport to the testset's start pose → spawn target (+paint)
→ run executor (instruction via stdin) → record captures and declarations → judge after it ends.

Output layout (default when OUT_DIR is not given):
  vlmpilot/outputs/verifier/<run time>/
    summary.json · report.json
    <model>/verdict_*.json · trace_*.jsonl · pilot_*.log
    <model>/runs/<target>/<executor time>/   ← decision·grid·FPV video (isolated per episode)
"""

import json
import math
from collections import deque
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time

from verifier.airsim.trace import DELTA_M, TraceSampler, sr_at, step_min
from verifier.airsim.tasks import canon, judge_task
from verifier.airsim.mover import TargetMover
from verifier.airsim.spawn_unique import UnknownPaintColor
from verifier.airsim.judge import _box_area, _get_detections, judged_dist
from verifier.airsim.guardrail import (TIMING_RE, attach_timing, is_excluded, make_session,
                                summarize_rows)

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PILOT = os.path.join(REPO, "vlmpilot")
PY = os.path.join(PILOT, ".venv", "bin", "python")

# Model → executor config (the very config the executor uses).
# ⚠️ Name them so "whose model, how many B" is evident.
CONFIGS = {
    "gemini":        "configs/airsim/single-drone/airsim_gemini.yaml",        # Gemini 2.5 Flash (Google API)
    "gemini35":      "configs/airsim/single-drone/airsim_gemini35.yaml",      # Gemini 3.5 Flash (Google API)
    "gemini37":      "configs/airsim/single-drone/airsim_gemini37.yaml",      # Gemini 3.7 Flash (Google API)
    "gemini-robotics": "configs/airsim/single-drone/airsim_gemini_robotics.yaml",  # Gemini Robotics-ER 2 preview (specialized for spatial reasoning)
    "gpt":           "configs/airsim/single-drone/airsim_gpt.yaml",           # OpenAI GPT-5 (direct)
    "gpt6-luna":     "configs/airsim/single-drone/airsim_gpt6luna.yaml",      # OpenAI GPT-6 Luna (direct)
    "gpt6-luna-or":  "configs/airsim/single-drone/airsim_gpt6luna_or.yaml",   # GPT-6 Luna (OpenRouter)
    "gemini38":      "configs/airsim/single-drone/airsim_gemini38_or.yaml",   # Gemini 3.8 Flash (OpenRouter)
    "claude":        "configs/airsim/single-drone/airsim_claude.yaml",        # Anthropic Claude Opus 5 (OpenRouter)
    "qwen35-9b":     "configs/airsim/single-drone/airsim_qwen35_9b.yaml",     # Qwen3.5-9B (OpenRouter)
    "qwen35-27b":    "configs/airsim/single-drone/airsim_qwen35_27b.yaml",    # Qwen3.5-27B (OpenRouter)
    "dronecats-2b":  "configs/airsim/single-drone/airsim_dronecats2b.yaml",   # DroneCATS-Cosmos3-Edge-2B (local vLLM); trained on one image, so not in MULTI_CONFIGS
}

# Only aliases are kept so scripts/notes that used the old names do not break.
ALIASES = {"luna": "gpt6-luna"}

# **Multi-drone** configs of the same models. The naming rule does not match single
# (gemini35 → airsim_multi_gemini35flash.yaml), so it cannot be derived by rule and is listed explicitly.
# Selecting with MULTI=1 makes the runner use these — command, judging, and testset are identical to single; only the config changes.
MULTI_CONFIGS = {
    "gemini":          "configs/airsim/multi-drone/airsim_multi_gemini.yaml",
    "gemini35":        "configs/airsim/multi-drone/airsim_multi_gemini35flash.yaml",
    "gemini37":        "configs/airsim/multi-drone/airsim_multi_gemini37flash.yaml",
    "gemini-robotics": "configs/airsim/multi-drone/airsim_multi_geminirobotics.yaml",
    "gpt":             "configs/airsim/multi-drone/airsim_multi_gpt.yaml",
    "gpt6-luna":       "configs/airsim/multi-drone/airsim_multi_gpt6luna.yaml",
    "gpt6-luna-or":    "configs/airsim/multi-drone/airsim_multi_gpt6luna_or.yaml",
    "gemini38":        "configs/airsim/multi-drone/airsim_multi_gemini38_or.yaml",
    "claude":          "configs/airsim/multi-drone/airsim_multi_claude.yaml",
    # Qwen3.5 roster — keeps the roster in line with the single CONFIGS (a model missing here dies with "unknown model" under MULTI=1).
    "qwen35-9b":       "configs/airsim/multi-drone/airsim_multi_qwen35_9b.yaml",
    "qwen35-27b":      "configs/airsim/multi-drone/airsim_multi_qwen35_27b.yaml",
}


# Episode weather — the testset `weather` field (preset name or {parameter: value}). If the field is missing, revert to clear
# (so rain from the previous episode does not leak into the next). Preset values were chosen by eye on NH:
# turning on only Rain makes rain fall on dry ground, which looks odd, and Fog 0.05 was too dense.
WEATHER_PRESETS = {
    "clear": {},
    "rain": {"Rain": 0.5, "Roadwetness": 0.9, "Fog": 0.015},
}
WEATHER_PARAMS = ("Rain", "Roadwetness", "Snow", "RoadSnow", "MapleLeaf", "RoadLeaf", "Dust", "Fog")


def resolve_weather(spec):
    """weather field → {parameter: value}. Unknown presets become clear (never returns None)."""
    if spec is None or spec == "":
        return {}
    if isinstance(spec, str):
        if spec not in WEATHER_PRESETS:
            print(f"  ! unknown weather preset {spec!r} — treating it as clear", flush=True)
            return {}
        return dict(WEATHER_PRESETS[spec])
    return {k: float(v) for k, v in dict(spec).items() if k in WEATHER_PARAMS}


def apply_weather(c, spec):
    """Set every parameter explicitly (0 for ones not given). If any is >0 the weather system is on, otherwise off.
    NH has no BP_Sky_Sphere, so simSetTimeOfDay is ignored; time of day is therefore not handled."""
    import airsim as _as
    vals = resolve_weather(spec)
    on = any(v > 0 for v in vals.values())
    if on:
        c.simEnableWeather(True)
    for k in WEATHER_PARAMS:
        c.simSetWeatherParameter(getattr(_as.WeatherParameter, k), float(vals.get(k, 0.0)))
    if not on:
        c.simEnableWeather(False)
    return {k: v for k, v in vals.items() if v > 0}


# Indoor/outdoor profiles — chosen by the episode's `indoor`. Indoors, δ, step size, bands, and drone size are half of outdoors (δ 2.5 m, step 1.5 m,
# bands 1.0/2.5 m, drone 0.5x); altitude 1.5 m, step 2.5 s, and cap 300 s are shared. These values are **the reference for sim (AirSim) measurement**.
#   The indoor reference for the real Tello (δ 1.5 m · cap 150 s: 2.5 m inside a room is too far to count as "arrived")
#   is kept separate from the sim and is not read from here (the Tello pilot takes its flight values from its own config). Putting tello values
#   here would also change the verdicts and caps of the 12 sim indoor episodes.
#   δ (judging radius) and cap (episode cap) are applied by the runner per episode; step size, step time, and start altitude are overwritten in a per-episode config copy.
#   drone_scale: drone scale factor (simSetObjectScale). Indoors 0.5 for passing through door frames — only the visual and collision meshes shrink; physics parameters and
#   camera FOV stay the same (measured: in front of the same door, 1.0 hit the door frame right after takeoff, 0.5 passed through at 3.6 m).
#   With PROFILES=0, every episode runs on the single DELTA/EP_TIMEOUT environment variables as before.
PROFILES = {
    "outdoor": dict(delta_m=5.0, max_step_distance_m=3.0, step_duration_s=2.5, start_altitude_m=1.5, cap_s=300,
                    drone_scale=1.0),
    "indoor":  dict(delta_m=2.5, max_step_distance_m=1.5, step_duration_s=2.5, start_altitude_m=1.5, cap_s=300, drone_scale=0.5,
                   # the approach bands are also half of outdoors (2.0/5.0): stop only at depth ≤ 1.0 m, approach at 60% step for 1.0~2.5 m.
                   # the default (stop at ≤ 2.0 m) combined with depth underestimation on thin objects and stopped 2.2 m short in reality (lamp, bed).
                   adaptive_depth_bands=[[1.0, 0.0, True], [2.5, 0.6, False]]),
}


def episode_profile(ep):
    """Episode → copy of the profile dict. If the episode has the same key, its value takes precedence (per-episode fine-tuning)."""
    p = dict(PROFILES["indoor" if ep.get("indoor") else "outdoor"])
    for k in p:
        if ep.get(k) is not None:
            p[k] = ep[k]
    return p


def _port_up(port, timeout=2.0):
    """Check that the sim RPC port is alive — running RPC against a dead sim stalls the runner for hours."""
    import socket
    s = socket.socket()
    s.settimeout(timeout)
    try:
        s.connect(("127.0.0.1", int(port)))
        return True
    except Exception:
        return False
    finally:
        s.close()


MULTI_SETTINGS_BASE = os.path.join(PILOT, "settings.multidrone.json.example")


def _sim_env(drone_names) -> dict:
    """Environment for sim restart — multi runs must be launched with 4-drone settings.

    If the runner's automatic restart brings the sim up with the default (1-drone) settings, place_fleet fails
    entirely and all remaining episodes are skipped (measured: spawn failure → restart
    → 3 consecutive placement failures). If the operator gave SETTINGS_BASE, it is respected as is.
    """
    env = dict(os.environ)
    if drone_names and not env.get("SETTINGS_BASE"):
        env["SETTINGS_BASE"] = MULTI_SETTINGS_BASE
    return env


def _wait_sim_ready(port, boot_timeout_s=300, settle_s=90) -> bool:
    """Wait until the restarted sim accepts RPC and level streaming settles.

    Waiting only 5 seconds and retrying fails again on a sim that is still booting, and the episode is skipped
    (measured on a multi run: the restart succeeded but place failed 5s later).
    Even with the port open, the first 1~2 minutes are level loading and spawning/placement is shaky — settle.
    """
    deadline = time.time() + boot_timeout_s
    while time.time() < deadline:
        if _port_up(port):
            time.sleep(settle_s)
            return True
        time.sleep(5)
    return False


def _host_sim_count():
    """Number of AirSim sims running on this host — for episode provenance.

    Stamping the number of co-located sims at measurement time into the verdict makes load-contamination forensics free
    (otherwise it can only be back-computed from directory timestamps).
    """
    try:
        out = subprocess.run(["pgrep", "-af", "settings_"], capture_output=True,
                             text=True, timeout=5).stdout
        # one sim is 2~3 processes including wrappers, so -fc is inflated (measured 21 sims→62).
        # the number of unique settings_<port> is the number of sims.
        ports = set(re.findall(r"settings_(\d+)", out))
        return len(ports)
    except Exception:
        return None


def _airsim_window_geom(port=None):
    """The AirSim window on :0 (w, h, x, y). None if we cannot be sure it is ours.

    Window titles have no port, so the owner cannot be told from the title alone. If there are several candidates,
    do not record — not recording beats silently capturing someone else's screen.
    Grab the window region, not the top-left of the screen (the window appears in the middle of the screen).
    """
    try:
        out = subprocess.run(["xwininfo", "-root", "-tree"],
                             env=dict(os.environ, DISPLAY=":0"),
                             capture_output=True, text=True, timeout=10).stdout
    except Exception:
        return None
    hits = []
    for line in out.splitlines():
        if "AirSim" not in line:
            continue
        m = re.search(r"(\d+)x(\d+)\+(-?\d+)\+(-?\d+)\s+\+(-?\d+)\+(-?\d+)", line)
        if m:
            w, h, _, _, ax, ay = (int(g) for g in m.groups())
            if w > 200 and h > 200:
                hits.append((w, h, ax, ay))
    if len(hits) != 1:
        if hits:
            print(f"  ! {len(hits)} AirSim windows on :0 — cannot tell which one is mine, disabling screen recording",
                  flush=True)
        return None
    # check that the only sim launched onscreen is the one on my port.
    if port is not None:
        try:
            ps = subprocess.run(["pgrep", "-af", "settings_"], capture_output=True,
                                text=True, timeout=10).stdout
            on = [l for l in ps.splitlines()
                  if "/opt/airsim_envs/" in l and "-RenderOffScreen" not in l]
            if len(on) != 1 or f"settings_{port}." not in on[0]:
                print(f"  ! the onscreen sim is not on my port ({port}) — disabling screen recording",
                      flush=True)
                return None
        except Exception:
            return None
    return hits[0]


class DeclarationScanner:
    """Detector that counts 'arrival declarations' in executor stdout — exactly 1 per declaration.

    The executor prints several lines for one declaration (measured):
        [ARRIVAL] Target-reached signal (VLM) 1/1     ← confirmation counter
        [ACTION] FINISHED — arrival declared.
        → FINISHED (arrival declared):
        === SEQUENCE FINISHED: target reached ===     ← finally, after mp4 assembly and grid
    Matching the 3 old string patterns as is catches one declaration as 3, and the last one
    comes after wrap-up work, so it is measured 5 seconds late — not "the moment of declaration" (measured Car_58:
    the same declaration 3 times at 19.52 / 20.53 / 25.03s). There used to be a guard that took only the first match,
    which hid this duplication.

    So only the **confirmation counter** line is counted — a line where n==N in n/N is a confirmed declaration and
    is printed exactly once per declaration (even with ARRIVAL_FRAMES debouncing on, the intermediate
    counts 1/3, 2/3 are not declarations).

    The fallback for old executors that do not print the confirmation counter is narrowed to **one finally-banner line**.
    It is printed only once per process, so duplicates are impossible by construction. Putting declaration-time strings
    ("arrival declared", etc.) in the fallback makes them appear before the confirmation counter and the same declaration
    is counted twice (measured). That build ends the process at the first declaration, so one banner
    is one declaration, the same as the old behavior.
    """

    PRIMARY = re.compile(r"Target-reached signal \(VLM\)\s*(\d+)\s*/\s*(\d+)")
    LEGACY = ("SEQUENCE FINISHED",)
    # multi-drone appends **which drone** declared to the same line (`drone=<name>`). Single builds
    # have no such tail, so the match fails → vehicle "" → default drone, i.e. exactly as before.
    VEHICLE = re.compile(r"drone=(\S+)")

    def __init__(self):
        self.seen_primary = False
        self.vehicle = ""      # name of the drone that last made a confirmed declaration (empty string for single)
        self.legacy_done = False

    def feed(self, line: str) -> bool:
        """Is this line a new arrival declaration?"""
        m = self.PRIMARY.search(line)
        if m:
            self.seen_primary = True
            mv = self.VEHICLE.search(line)
            # update the drone only on confirmed declarations — if the drone of an intermediate count (1/3)
            # overwrites the confirmed drone, the wrong drone gets measured.
            if mv and m.group(1) == m.group(2):
                self.vehicle = mv.group(1)
            return m.group(1) == m.group(2)
        if self.seen_primary:
            return False          # a build that uses the confirmation counter — the remaining lines are the same declaration
        if self.legacy_done or not any(k in line for k in self.LEGACY):
            return False
        self.legacy_done = True
        return True


def place(port, x, y, z, yaw_deg, vehicle_name="", client=None, reset=True):
    """Move the drone **exactly** to the start pose written in the testset.

    Same as placing an asset at given coordinates — teleport to the received (x,y,z,yaw) as is.
    No drop, no ground measurement, no conditional branching. The start coordinates were already decided by the builder when it chose the spot
    and baked into the testset, so here it only deterministically stands the drone at that spot.
    """
    import airsim
    # timeout_value defaults to 3600 seconds — if one RPC blocks, everything stalls for that long
    # (measured: 9+ minutes of silence waiting for place after episode cleanup, hung until stallwatch killed it).
    # 30 seconds is ample even on a busy server, and failures are caught by the caller's restart loop.
    c = client or airsim.MultirotorClient(port=port, timeout_value=30)
    if client is None:
        c.confirmConnection()
    # multi calls this per drone via place_fleet, so reset is done only once, on the first drone.
    if reset:
        c.reset()
        time.sleep(1.0)
    q = airsim.to_quaternion(0, 0, math.radians(yaw_deg))
    c.enableApiControl(True, vehicle_name)
    c.armDisarm(True, vehicle_name)
    # ★ simSetVehiclePose/simGetVehiclePose use coordinates **relative to that vehicle's spawn origin**.
    #   Testset coordinates are global, so subtract the spawn offset (global − relative) before setting, and verify
    #   in global as well (simGetObjectPose). For single (offset 0) the values are the same as before.
    #   (multi measurement: the cause of Drone2~4 starting 8.0~11.3 m off)
    ox, oy, oz = _spawn_offset(c, vehicle_name)
    lx, ly, lz = x - ox, y - oy, z - oz
    # the teleport is sometimes silently ignored (measured: 2 episodes started at the origin
    # (0,0) due to an RPC race right after reset, 1 sank below ground to +3.19). Verify it was applied
    # and place again if off — if the start pose is wrong, the whole episode is invalid.
    for _ in range(10):
        c.simSetVehiclePose(airsim.Pose(airsim.Vector3r(lx, ly, lz), q), True, vehicle_name)
        time.sleep(0.5)
        p = _global_position(c, vehicle_name)
        # the z check is for rejecting starts that sank below ground (+0.28m measured). Starts above ground (tracking
        # testset -0.26~-0.37) settling down to the ground (≈0) is normal and allowed — the old |Δz|<0.25 treated
        # this as a mismatch, and episodes went missing depending on the check timing.
        z_ok = (z - 0.25) <= p.z_val <= min(z + 0.5, max(z, 0.0) + 0.25)
        if math.dist((p.x_val, p.y_val), (x, y)) < 1.0 and z_ok:
            break
        time.sleep(1.5)
    else:
        # a freshly booted sim sometimes keeps ignoring teleports until initialization finishes
        # (measured: all 10 attempts stayed at the origin right after boot). Flying from a wrong start invalidates the whole
        # episode, so raise an exception to take the caller's sim-restart path.
        raise RuntimeError(
            f"start mismatch: requested ({x:.1f},{y:.1f},{z:.2f}) ≠ "
            f"actual ({p.x_val:.1f},{p.y_val:.1f},{p.z_val:.2f})")
    who = f" {vehicle_name}" if vehicle_name else ""
    print(f"  start{who}: ({p.x_val:.1f},{p.y_val:.1f},{p.z_val:.2f}) yaw={yaw_deg:.0f}deg",
          flush=True)


def _spawn_offset(c, vehicle_name=""):
    """The vehicle's (global − relative) spawn offset. (0,0,0) if the vehicle object cannot be found."""
    try:
        rel = c.simGetVehiclePose(vehicle_name=vehicle_name).position
        name = vehicle_name or (c.listVehicles() or [""])[0]
        glo = c.simGetObjectPose(name).position
        if any(v != v for v in (glo.x_val, glo.y_val, glo.z_val)):
            return 0.0, 0.0, 0.0
        return glo.x_val - rel.x_val, glo.y_val - rel.y_val, glo.z_val - rel.z_val
    except Exception:
        return 0.0, 0.0, 0.0


def _global_position(c, vehicle_name=""):
    """The vehicle's **global** position. Falls back to the relative pose if simGetObjectPose is NaN (single, offset 0)."""
    try:
        name = vehicle_name or (c.listVehicles() or [""])[0]
        p = c.simGetObjectPose(name).position
        if not any(v != v for v in (p.x_val, p.y_val, p.z_val)):
            return p
    except Exception:
        pass
    return c.simGetVehiclePose(vehicle_name=vehicle_name).position


# Multi-drone start grid — same layout as the spawn grid (8 m) in settings.multidrone.json.example.
# The testset's start pose is for 1 drone, so the i-th drone is placed at (x + FLEET_GRID_M*(i//2), y + FLEET_GRID_M*(i%2)).
# Heading is the same for all drones.
FLEET_GRID_M = 8.0


def fleet_start_offsets(n, grid_m=FLEET_GRID_M):
    """(dx, dy) offset of the i-th drone — Drone1 (0,0), Drone2 (0,g), Drone3 (g,0), Drone4 (g,g) …"""
    return [((i // 2) * grid_m, (i % 2) * grid_m) for i in range(n)]


def place_fleet(port, x, y, z, yaw_deg, drone_names):
    """Multi: place every drone at the testset start pose + grid offset (all in global coordinates)."""
    import airsim
    # this client is passed straight to place(), so the timeout must be set here too —
    # otherwise the default 3600 seconds remains only on the multi path and one RPC stalls the whole run.
    c = airsim.MultirotorClient(port=port, timeout_value=30)
    c.confirmConnection()
    c.reset()
    time.sleep(1.0)
    for name, (dx, dy) in zip(drone_names, fleet_start_offsets(len(drone_names))):
        place(port, x + dx, y + dy, z, yaw_deg, vehicle_name=name, client=c, reset=False)


FRAME_RE = re.compile(r"^\[FRAME\] captured")

# One line per model output. Attaches this verb to the preceding [FRAME] record.
ACTION_RE = re.compile(r"^\[ACTION\]\s+([A-Z][A-Z_]*)")
# Multi-drone appends a `drone=<name>` tail at the **end** of the [ACTION] line (not a prefix — with a prefix the regex above
# reads the action name as "D" and every declaration drops out of SR). Single has no tail → "".
ACTION_DRONE_RE = re.compile(r"\bdrone=(\S+)\s*$")
# Parsing and attaching the controller's step timing ([TIMING] lines) is in verifier.airsim.guardrail.timing (TIMING_RE, attach_timing).
# Number of times the same episode is retried after a sim restart on spawn failure (sim wedge) — a stability measure unrelated to the guardrail.
SPAWN_RETRIES = 2


def action_drone(line: str) -> str:
    m = ACTION_DRONE_RE.search(line)
    return m.group(1) if m else ""


def pick_declared_measurement(step_evs, vehicle: str):
    """The [ACTION] measurement matching a confirmed declaration — **that drone's** last FINISHED measurement.

    In multi there are as many [ACTION]s per step as drones, so step_evs[-1] may be a different drone.
    For single, drone is "" on both sides, so the result is the same as before (whether the last item is FINISHED).
    """
    for ev in reversed(step_evs):
        if ev.get("drone", "") != (vehicle or ""):
            continue
        return ev if ev.get("action") == "FINISHED" else None
    return None


def _pose_now(mc, tgt_sim, vehicle_name=""):
    """Take only the drone and target poses — 2 RPCs. Detection is expensive, so it is not done here.

    Multi calls this per drone — without passing it, someone else's score is given from drone 1's pose (same pitfall as V10).
    Positions must be global (simGetVehiclePose is relative to the spawn origin).
    """
    p = mc.simGetVehiclePose(vehicle_name=vehicle_name)
    d = _global_position(mc, vehicle_name)
    q = p.orientation
    tp = mc.simGetObjectPose(tgt_sim)
    t, tq = tp.position, tp.orientation
    return dict(
        drone_xyz=[round(d.x_val, 3), round(d.y_val, 3), round(d.z_val, 3)],
        drone_quat=[round(q.w_val, 5), round(q.x_val, 5), round(q.y_val, 5), round(q.z_val, 5)],
        target_xyz=[round(t.x_val, 3), round(t.y_val, 3), round(t.z_val, 3)],
        target_quat=[round(tq.w_val, 5), round(tq.x_val, 5), round(tq.y_val, 5), round(tq.z_val, 5)],
        dist_m=round(math.dist((d.x_val, d.y_val, d.z_val), (t.x_val, t.y_val, t.z_val)), 3),
    )


def _hold_still(mc, airsim, xyz, quat_wxyz, vehicle_name=""):
    """Reset velocity to 0 right after a teleport. simSetVehiclePose changes only position and leaves velocity,
    so repeated replays accumulated falling velocity and the drone at detection time was up to 1.2m below the capture pose
    (measured over 30 repetitions). With velocity cleared, the error after advancing 2 frames is 0.02m. Skipped if the old AirSim lacks the API."""
    try:
        st = airsim.KinematicsState()
        st.position = airsim.Vector3r(*xyz)
        qw, qx, qy, qz = quat_wxyz
        st.orientation = airsim.Quaternionr(qx, qy, qz, qw)
        zero = airsim.Vector3r(0.0, 0.0, 0.0)
        st.linear_velocity = st.angular_velocity = zero
        st.linear_acceleration = st.angular_acceleration = zero
        mc.simSetKinematics(st, True, vehicle_name)
    except Exception:
        pass


def replay_visibility(mc, tgt_sim, step_evs, delta_m, camera="0", crosscheck=False,
                      spawn_off=None):
    """Rewind to the recorded poses and measure visibility (after the episode ends).

    SR is `declared ∧ dist ≤ δ ∧ visible`, so filter by distance first.
    Declarations that already have capture-time visibility (visible key present) are skipped. Multi returns the declaring drone to
    spawn-origin-relative coordinates (spawn_off corrected) and measures with that drone's camera — single (drone "") is as before.
    crosscheck=True (for verification): also replay capture-time-judged declarations and write them separately to visible_replay/bbox_px_replay.
    """
    todo = [e for e in step_evs
            if e.get("declared") and judged_dist(e) is not None and judged_dist(e) <= delta_m
            and ("visible" not in e
                 or (crosscheck and e.get("visibility_src") == "capture"))]
    if not todo:
        return 0
    import airsim
    ident = [1.0, 0.0, 0.0, 0.0]
    try:
        for e in todo:
            veh = e.get("drone") or ""
            off = (spawn_off or {}).get(veh) or (0.0, 0.0, 0.0)
            mc.simPause(True)
            dx, dy, dz = e["drone_xyz"]
            dx, dy, dz = dx - off[0], dy - off[1], dz - off[2]
            qw, qx, qy, qz = e["drone_quat"]
            mc.simSetVehiclePose(
                airsim.Pose(airsim.Vector3r(dx, dy, dz), airsim.Quaternionr(qx, qy, qz, qw)),
                True, veh)
            _hold_still(mc, airsim, (dx, dy, dz), (qw, qx, qy, qz), veh)
            tx, ty, tz = e["target_xyz"]
            tw, txx, tyy, tzz = e.get("target_quat") or ident
            mc.simSetObjectPose(
                tgt_sim,
                airsim.Pose(airsim.Vector3r(tx, ty, tz), airsim.Quaternionr(txx, tyy, tzz, tw)), True)
            mc.simContinueForFrames(2)      # turn the render — without it the previous frame comes back
            dets = _get_detections(mc, tgt_sim, camera=camera, vehicle_name=veh)
            vis = bool(dets)
            area = round(max((_box_area(x.box2D) for x in dets), default=0.0), 1)
            if e.get("visibility_src") == "capture":
                e["visible_replay"], e["bbox_px_replay"] = vis, area   # cross-check record only
            else:
                e["visible"], e["bbox_px"] = vis, area
    except Exception as exc:
        # dying here would lose the verdict of an entire 300-second flight — leave the remaining items as not visible.
        print(f"  ! replay aborted t={e.get('t')}: {exc}", flush=True)
    finally:
        mc.simPause(False)
    return len(todo)


def capture_visibility(mc, tgt_sim, ev, delta_m, camera="0", vehicle_name="", mesh_pattern=None):
    """Right after [FRAME], attach on-the-spot visibility to captures whose distance is within δ →
    visible / bbox_px / visibility_src="capture". Returns True/False, or None if short on distance or missing (ev unchanged).

    The sim keeps rendering, so only detections are queried without a render request (≈8ms). It is the same pose as the frame the pilot
    just captured, so distance and visibility come from one moment. mesh_pattern: spawned objects (bench BM_* actors) require a mesh
    wildcard — without it, the result is silently 0 hits. Exceptions are handled by the caller (that frame is judged by replay).
    """
    d = judged_dist(ev)              # if an anchor (license plate) distance is attached, δ is checked against it
    if d is None or d > delta_m:
        return None
    dets = _get_detections(mc, tgt_sim, camera=camera, vehicle_name=vehicle_name,
                           mesh_pattern=mesh_pattern, render=False)
    if not dets:
        # for the tracking family, the trajectory sampler (TraceSampler, a separate thread) also clears and rebuilds the same camera's detection filter
        # every 0.5s — getting caught in between yields 0 hits. Query once more to filter out the momentary clash.
        dets = _get_detections(mc, tgt_sim, camera=camera, vehicle_name=vehicle_name,
                               mesh_pattern=mesh_pattern, render=False)
    ev["visible"] = bool(dets)
    ev["bbox_px"] = round(max((_box_area(x.box2D) for x in dets), default=0.0), 1)
    ev["visibility_src"] = "capture"
    return bool(dets)


def main():
    # when launched in the background (`nohup … &`), SIGINT is inherited as SIG_IGN, and an executor Popen-ed
    # in that state also ignores SIGINT and the episode cannot be cut (measured: force-killed after 5 minutes of hovering after a declaration).
    # if the runner restores the default handler, the child starts with SIG_DFL on exec and behaves normally.
    if signal.getsignal(signal.SIGINT) == signal.SIG_IGN:
        signal.signal(signal.SIGINT, signal.default_int_handler)
    ts_path = sys.argv[1] if len(sys.argv) > 1 else "episodes/airsim/single-drone/nh_approaching.json"
    ts = json.load(open(ts_path))
    # catch paint colors not in the palette before starting — better than launching sims, running for hours,
    # and having episodes silently go empty. With ALLOW_UNKNOWN_PAINT=1, only warn and proceed.
    try:
        from verifier.airsim.spawn_unique import PAINT_BGR as _PB
        _bad = sorted({e["paint_texture"] for e in ts.get("episodes", [])
                       if e.get("paint_texture") and e["paint_texture"] not in _PB})
        if _bad:
            _msg = (f"testset {os.path.basename(ts_path)}: paint colors not in the palette {_bad} "
                    f"(available: {sorted(_PB)})")
            raise SystemExit(f"[ABORT] {_msg}\n"
                             "  The instructions refer to colors, so running without paint is not allowed. "
                             "Fix the testset colors or add the colors to PAINT_BGR.")
    except ImportError:
        pass
    models = [ALIASES.get(m.strip(), m.strip())
              for m in os.environ.get("MODELS", "gemini").split(",") if m.strip()]
    # with MULTI=1, run with the same model's multi config. Testset and judging stay the same.
    cfgs = MULTI_CONFIGS if os.environ.get("MULTI") == "1" else CONFIGS
    unknown = [m for m in models if m not in cfgs]
    if unknown:
        raise SystemExit(f"unknown models: {unknown} — valid values: {list(cfgs)}")
    # check up front that the config files exist — otherwise the executor silently falls back to the default config
    # and produces the wrong result "that model was evaluated" (this really happens when
    # hopping between branches). Check everything at once before starting the batch.
    missing = [m for m in models if not os.path.exists(os.path.join(PILOT, cfgs[m]))]
    if missing:
        raise SystemExit("config files not found: "
                         + ", ".join(f"{m} → {cfgs[m]}" for m in missing))
    port = int(os.environ.get("PORT", ts.get("port", 41455)))
    delta = os.environ.get("DELTA", str(DELTA_M))
    delta_env = delta
    use_profiles = os.environ.get("PROFILES", "1") == "1"
    # online verdict and early stop (module docstring). If you plan to re-judge with a stricter δ, turn it off with 0 and run.
    online_verdict = os.environ.get("ONLINE_VERDICT", "1") == "1"
    # for verification: also re-measure declarations judged at capture time via replay after the episode ends, and record/log whether they agree.
    online_crosscheck = os.environ.get("ONLINE_VERDICT_CROSSCHECK") == "1"
    # default: vlmpilot/outputs/verifier/<run time>/ — one run is closed into one folder.
    # pinned to an absolute path — the executor is launched with cwd=vlmpilot, so passing a relative OUT_DIR
    # as is would fail to find the episode config and every episode would end with 0 steps.
    out_root = os.path.abspath(os.environ.get("OUT_DIR") or os.path.join(
        PILOT, "outputs", "verifier", time.strftime("%Y%m%d_%H%M%S")))
    # episode cap 5 minutes — bench budget ("about 300 episodes, 5 minutes each, within 24 hours").
    # going over 300 seconds makes a 300-episode batch exceed 25 hours and miss the spec.
    ep_timeout = int(os.environ.get("EP_TIMEOUT", "300"))
    ep_timeout_env = ep_timeout
    # slack for the executor to wrap up (verdict, mp4 assembly, grid) after SIGINT is sent on exceeding the cap.
    # a 300-second flight has thousands of frames, so encoding takes time.
    # 180 seconds was not enough — in a claude (OpenRouter) smoke test it was force-killed and the verdict
    # came out entirely empty. Even after SIGINT arrives, the in-flight VLM request must finish before entering finally,
    # and with a remote API that response takes tens of seconds.
    cleanup_grace = int(os.environ.get("CLEANUP_GRACE", "300"))
    os.makedirs(out_root, exist_ok=True)

    # task (judging criterion) and scenario spec. Testset first, otherwise the TASK env, otherwise approaching.
    # testset-diverse format: top-level scenario_spec{moving_target,visible_at_start,...}
    #   + per-episode scenario/moving/flying. The old format (nh_testset) lacks these fields, so
    #   everything is treated as approaching (stationary, visible at start).
    spec = ts.get("scenario_spec", {})
    default_task = ts.get("scenario") or os.environ.get("TASK", "approaching")
    # the map to launch on restart — follows the testset's scene. Hardcoding nh would make campus
    # measurements silently continue on nh after a crash and contaminate the results (measured).
    scene = ts.get("scene", "nh")

    rows = []
    for model in models:
        cfg = cfgs[model]          # multi config when MULTI=1
        # for a multi config, the list of drone names — read up front so the start pose can be placed on every drone.
        try:
            import yaml as _yaml
            drone_names = list((_yaml.safe_load(open(os.path.join(PILOT, cfg))) or {}).get("drone_names") or [])
        except Exception:
            drone_names = []
        # episode outputs go into per-model subfolders (directly under the run folder, only summaries and trajectories)
        mdir = os.path.join(out_root, model)
        os.makedirs(mdir, exist_ok=True)
        # measurement guardrail (module docstring) — one session per model. With enforce, an episode failing integrity is
        # put back at the front of the queue and the same episode is re-flown right away (3 times total; if still not cleared, it is finalized
        # as invalid and logged at the end of the run to flag it for RESUME re-measurement). With off, no action.
        session = make_session(model, mdir, log=lambda s: print(s, flush=True))
        ep_queue = deque(ts["episodes"])
        spawn_attempts = {}    # tag → number of retries after restarting the sim due to spawn failure
        while ep_queue:
            ep = ep_queue.popleft()
            s, tgt = ep["start"], ep["target"]
            task = canon(ep.get("scenario") or default_task)
            moving = bool(ep.get("moving", spec.get("moving_target", False))
                          or task in ("tracking", "search_tracking"))
            tag = f"{model}_{tgt}"
            # episode profile: δ and cap are applied here; step size, step time, and start altitude in the config copy below
            prof = episode_profile(ep) if use_profiles else None
            delta = str(prof["delta_m"]) if prof else delta_env
            ep_timeout = int(prof["cap_s"]) if prof else ep_timeout_env
            if prof:
                print(f"  profile: {'indoor' if ep.get('indoor') else 'outdoor'} δ{prof['delta_m']:g}m "
                      f"step {prof['max_step_distance_m']:g}m/{prof['step_duration_s']:g}s "
                      f"alt {prof['start_altitude_m']:g}m cap {prof['cap_s']}s scale {prof.get('drone_scale', 1.0):g}", flush=True)
            vpath = os.path.join(mdir, f"verdict_{tag}.json")
            # with RESUME=1, episodes that already have a verdict are skipped (resuming after a sim crash).
            if os.path.exists(vpath):
                if os.environ.get("RESUME") == "1":
                    v = json.load(open(vpath))
                    _integ = v.get("integrity") or {}
                    # enforce: episodes recorded as invalid are not carried over — RESUME after the cause of contamination (server, load)
                    # is resolved is the re-flight opportunity (the session moves them to _invalid/).
                    if not session.resume_reflight(tag, vpath, v):
                        # carried-over episodes cannot rebuild their trajectory — metrics are left as None
                        # so "not measured" and "0" do not get mixed up.
                        rows.append(dict(model=model, target=tgt, color=ep.get("color"),
                                         task=task,
                                         success=v.get("success"), reason=v.get("reason"),
                                         end_reason=v.get("end_reason"), dist_m=v.get("dist_m"),
                                         visible=v.get("visible"), bbox_px=v.get("bbox_px"),
                                         qf_declarations=v.get("qf_declarations"),
                                         steps=v.get("steps", 0), wall_s=0, run_dir="",
                                         delta_m=v.get("delta_m"),
                                         min_dist_m=None, path_length_m=None,
                                         samples=0, trace_errors=0,
                                         osr_min_m=step_min(v.get("step_events")),
                                         sr_min_m=step_min(v.get("step_events"),
                                                           declared=True, visible=True),
                                         # follows the mode at the time the episode was recorded — old enforce
                                         # data is filtered by its own rule even when carried over under off now.
                                         guardrail_mode=v.get("guardrail_mode") or session.mode,
                                         integrity_valid=_integ.get("valid"),
                                         integrity_reasons=_integ.get("reasons")))
                        print(f"[{model:9}] {tgt:9} (skip — already judged)", flush=True)
                        continue
                # the RESUME invalid branch may already have moved the verdict to _invalid/.
                if os.path.exists(vpath):
                    os.remove(vpath)
            # unique-target spawn: if the testset has an asset, spawn directly with that (asset, position)
            # — an object not in the map, so reproduction is closed by the testset alone (no separate
            # spawn_unique pre-run needed). Moving targets are also placed anew every episode and so
            # reset to their start position.
            tgt_sim = tgt   # actual object name in the sim (UE may change it on spawn)
            if ep.get("asset"):
                try:
                    import airsim as _as
                    from verifier.airsim.spawn_unique import spawn_one, paint
                    # timeout is mandatory — the default is 3600s, so if UE wedges on the spawn RPC
                    # the runner stalls for a full hour (measured: a re-flight spawn
                    # blocked 25+ minutes in simSpawnObject, confirmed with py-spy).
                    _c = _as.VehicleClient(port=port, timeout_value=60)
                    _c.confirmConnection()
                    # remove the previous episode's target — if left, several of the same kind are visible and
                    # the uniqueness of references like "the white car" breaks. Identifying "what we spawned"
                    # is a startswith match on the target names written in the testset — UE sometimes
                    # appends a suffix to spawn names (below), so exact matching would miss them.
                    # UE also sometimes changes the case (measured: 'suv' → 'Suv'). If case-sensitive,
                    # they cannot be deleted, and UE dies when the next episode spawns the same name.
                    ours = {e["target"].lower() for e in ts["episodes"] if e.get("asset")}
                    # also delete spawned objects left in the same sim by previous runs — targets renamed in the testset (bed→blacksofa, etc.)
                    # were not in the set above and remained; in one run an old bed hid the new sofa and killed the episode.
                    # spawned names are written to a per-port registry file and read on the next run.
                    _reg_path = f"/tmp/vlmpilot_spawned_{port}.json"
                    try:
                        _reg = set(json.load(open(_reg_path)))
                    except Exception:
                        _reg = set()
                    ours |= {n.lower() for n in _reg}
                    for _o in _c.simListSceneObjects():
                        if _o != tgt and any(_o.lower().startswith(u) for u in ours):
                            try:
                                _c.simDestroyObject(_o)
                            except Exception:
                                pass
                    # ⚠️ if the requested name cannot be used (reserved/previously used, etc.), UE spawns with a
                    # suffix appended and returns that name (measured: requested 'cyclist' →
                    # created 'cyclist1'. All verdicts and trajectories that looked up by name became NaN
                    # — 34 episodes silently invalid). The returned name is used as canonical.
                    tgt_sim = spawn_one(_c, tgt, ep["asset"], ep["target_xyz"],
                                        ep.get("target_yaw_deg", 0.0),
                                        float(ep.get("scale", 1.0) or 1.0)) or tgt
                    if tgt_sim != tgt:
                        print(f"  spawn name changed: {tgt} → {tgt_sim}", flush=True)
                    try:
                        _reg |= {tgt, tgt_sim}
                        json.dump(sorted(_reg), open(_reg_path, "w"))
                    except Exception:
                        pass
                    if ep.get("paint_texture"):
                        # for assets without textures — paint a solid color so color references hold
                        paint(_c, tgt_sim, ep["paint_texture"])
                except UnknownPaintColor as e:
                    # a color error is not a sim problem — restarting gives the same result,
                    # and running without color makes color-referencing instructions false. Stop right away.
                    raise SystemExit(f"[ABORT] {tgt}: {e}")
                except Exception as e:
                    # spawn failures are usually sim wedge/death (measured: UE segfault ×2 on
                    # respawn right after an early abort). Left as is, all remaining episodes
                    # are silently skipped — restart the sim, put this episode back in the queue, and
                    # try once more. If the second attempt also fails, skip it then.
                    if spawn_attempts.get(tag, 0) < SPAWN_RETRIES:
                        spawn_attempts[tag] = spawn_attempts.get(tag, 0) + 1
                        print(f"[{model:9}] {tgt:14} (spawn failed {type(e).__name__}) — "
                              f"restarting the sim and retrying", flush=True)
                        subprocess.run(["bash", os.path.join(REPO, "scripts/airsim/launch_airsim.sh"),
                                        scene, str(port), "3"], capture_output=True,
                                       timeout=300, env=_sim_env(drone_names))
                        _wait_sim_ready(port)
                        ep_queue.appendleft(ep)
                    else:
                        print(f"[{model:9}] {tgt:14} (spawn failed {type(e).__name__}) — skipped",
                              flush=True)
                    continue
            # weather — the testset weather field. Applied after spawning and before placement (it is a world setting, so reset does not clear it).
            # the episode continues even on failure: weather is a condition, not a verdict.
            try:
                import airsim as _as
                _wc = _as.VehicleClient(port=port, timeout_value=30)
                _wc.confirmConnection()
                _w = apply_weather(_wc, ep.get("weather"))
                print(f"  weather: {_w if _w else 'clear'}", flush=True)
            except Exception as _e:
                print(f"  ! weather apply failed {type(_e).__name__}: {_e}", flush=True)
            # move exactly to the start pose — z is baked into the testset from measurement
            # (build_testset calibration measures the ground and stores it). Not re-measured on every run.
            sim_restarts = 0
            placed = False
            for attempt in (1, 2):
                try:
                    if drone_names:
                        place_fleet(port, s["x"], s["y"], s["z"], s["yaw_deg"], drone_names)
                    else:
                        place(port, s["x"], s["y"], s["z"], s["yaw_deg"])
                    placed = True
                    break
                except Exception as e:
                    sim_restarts += 1
                    print(f"  ! sim connection failed ({type(e).__name__}) — restart attempt {attempt}", flush=True)
                    subprocess.run(["bash", os.path.join(REPO, "scripts/airsim/launch_airsim.sh"),
                                    scene, str(port), "3"], capture_output=True, timeout=300,
                                   env=_sim_env(drone_names))
                    if not _wait_sim_ready(port):
                        print(f"  ! the restarted sim (:{port}) did not come up within {300}s", flush=True)
            if not placed:
                # could not get the start pose — flying from there would score 0 from a wrong pose.
                # skip without a verdict, leaving it as a RESUME re-flight target.
                print(f"  !! {tag}: start pose placement failed — skipping without creating an episode",
                      flush=True)
                continue

            # the executor turns thinking on by itself (only on steps whose previous action was THINK).
            # so the runner does not force it —
            # forcing it makes the model fill only the skeleton without thinking and declare arrival early.
            # screen recording (only in onscreen mode). AirSim camera capture is bottlenecked on 1280x720 raw
            # transfer, capping out at a measured 1.4fps — it becomes a slideshow.
            # grabbing the X screen directly gives 30fps real-time video.
            scr, scr_path = None, os.path.join(mdir, "screen", f"{tag}.mp4")
            geom = (_airsim_window_geom(port)
                    if (os.environ.get("SCREEN_REC") == "1"
                        and os.path.exists("/tmp/.X11-unix/X0")) else None)

            # VERIFY_* is not passed — judging is done directly by the runner, not by an executor hook.
            # (evaluation runs on the executor's default path alone.)
            # PYTHONUNBUFFERED — streams executor stdout line by line immediately. Otherwise
            # the pipe is block-buffered, the FINISHED marker arrives late, and measurement happens long after
            # rather than at 'the moment of declaration'.
            env = dict(os.environ, PYTHONUNBUFFERED="1")
            # do not end the executor on an arrival declaration — if one declaration missing the condition ends the episode,
            # the verdict "any one of several declarations is satisfied" does not hold.
            env["VLMPILOT_CONTINUE_AFTER_ARRIVAL"] = "1"
            if geom:
                # with screen recording, AirSim Recording only eats into rendering
                # (measured 25.4 → 20.5 fps, stall segments 8 → 30). Turn it off.
                env["VLMPILOT_NO_RECORD"] = "1"

            t0 = time.time()
            scr_t0 = None
            if geom:
                os.makedirs(os.path.dirname(scr_path), exist_ok=True)
                gw, gh, gx, gy = geom
                # record the wall-clock time recording started. When overlaying decisions on the video, without this
                # it has to be estimated from the executor process start time, but recording starts a few seconds
                # earlier, so the banner appears that much earlier and it looks like
                # "FINISHED but still flying".
                scr_t0 = time.time()
                scr = subprocess.Popen(
                    ["ffmpeg", "-y", "-v", "error", "-f", "x11grab",
                     "-framerate", "30", "-video_size", f"{gw}x{gh}",
                     "-i", f":0+{gx},{gy}",
                     "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                     "-pix_fmt", "yuv420p", scr_path],
                    stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL)
            # record poses throughout the flight to get minimum distance, path length, and rotation. It is a separate
            # connection from the executor, so it does not depend on executor internals (markers, env vars).
            # for the tracking family the target moves and judging is 'distance + visibility held', so the target
            # pose and visibility are also measured every sample.
            trk = task in ("tracking", "search_tracking")
            tr = TraceSampler(port, tgt_sim,
                              trace_path=os.path.join(mdir, f"trace_{tag}.jsonl"),
                              moving_target=moving, track_visibility=trk)
            tr.start()
            # target mover — if the target moves, run it back and forth at constant speed.
            # speed and direction follow the testset's motion spec.
            mt = ep.get("motion") or {}
            # one-way is the default — since judging is declaration-based, a round trip is broken by "the target passes by
            # even if you stay still". If an episode really wants a round trip,
            # it can be turned off with motion.oneway: false.
            mv = (TargetMover(port, tgt_sim,
                              speed=mt.get("speed_mps", 1.2),
                              span_m=mt.get("span_m", 20.0),
                              heading_deg=mt.get("heading_deg"),
                              hz=mt.get("hz", 20.0),
                              waypoints=mt.get("waypoints"),
                              # for keys explicitly null in the testset, .get(k, default) returns
                              # None — `or` handles both missing and null
                              yaw_offset_deg=ep.get("asset_yaw_offset_deg") or 0.0,
                              oneway=bool(mt.get("oneway", True)))
                  if moving else None)
            if mv:
                mv.start()
            # ── run executor + runner's own judging ────────────────────────────────
            # separate RPC connection for judging (independent of the executor). Measurements at the FINISHED marker use this.
            import airsim as _airsim
            # the judging client also has a 30-second cap — the default 3600 seconds is the root of hangs (see place above).
            mc = _airsim.MultirotorClient(port=port, timeout_value=30)
            mc.confirmConnection()
            # drone scale (profile drone_scale). Set explicitly on every episode so the previous episode's scale does not leak.
            try:
                _scale = float(prof.get("drone_scale", 1.0)) if prof else 1.0
                for _n in (drone_names or list(mc.listVehicles())[:1]):
                    mc.simSetObjectScale(_n, _airsim.Vector3r(_scale, _scale, _scale))
            except Exception as _e:
                print(f"  ! drone_scale apply failed {type(_e).__name__}: {_e}", flush=True)
            # multi: per-drone spawn offset (global − relative) — used by post-hoc replay when rewinding that drone.
            spawn_off = {}
            for _n in drone_names:
                try:
                    _g = _global_position(mc, _n)
                    _r = mc.simGetVehiclePose(vehicle_name=_n).position
                    spawn_off[_n] = (_g.x_val - _r.x_val, _g.y_val - _r.y_val, _g.z_val - _r.z_val)
                except Exception as _e:
                    print(f"  ! {tag}: {_n} spawn offset failed {_e} — no replay coordinate correction", flush=True)

            # pass a config copy with an episode-specific output_root baked in.
            # executor run folder names are second-resolution timestamps, so in parallel runs runners that start
            # in the same second write into one folder (measured: mixing with 20 shards running concurrently).
            # layout: <OUT_DIR>/runs/<model>/<target>/<executor timestamp>/ — can be located directly
            # by model and time, and running the same model on several ports does not mix them.
            import yaml
            ep_root = os.path.join(mdir, "runs", tgt)
            cfg_d = yaml.safe_load(open(os.path.join(PILOT, cfg)))
            cfg_d["output_root"] = ep_root
            if prof:
                for k in ("max_step_distance_m", "step_duration_s", "start_altitude_m", "adaptive_depth_bands"):
                    if prof.get(k) is not None:      # keys not in the profile (outdoor bands, etc.) keep the config defaults
                        cfg_d[k] = prof[k]
            cfg_path = os.path.join(mdir, f"cfg_{tag}.yaml")
            with open(cfg_path, "w") as cf:
                yaml.safe_dump(cfg_d, cf, allow_unicode=True)

            proc = subprocess.Popen(
                [PY, "-m", "vlmpilot", "airsim", "--config", cfg_path, "--port", str(port)],
                cwd=PILOT, env=env, stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1)
            proc.stdin.write(ep["instruction"] + "\n")
            proc.stdin.flush()
            proc.stdin.close()

            # cap watchdog — a separate thread. When exceeded, SIGINT (=KeyboardInterrupt) triggers the executor's
            # finally (FPV mp4 assembly, recognition_grid). Killing it would lose all of that.
            # if it still does not die, force-kill after cleanup_grace.
            done = threading.Event()
            timed_out = {"v": False}

            # the moment the flight ends (arrival declaration or cap reached), cut the trajectory and mover —
            # because landing during wrap-up contaminates distance and path.
            flight_over = threading.Event()
            flight_wall = [None]

            def _end_flight():
                if not flight_over.is_set():
                    flight_over.set()
                    if mv:
                        mv.stop()
                    flight_wall[0] = time.time() - t0
                    tr.stop()
                    # screen recording is also cut here. Cutting it after proc.wait() would include the wrap-up
                    # time in the video and its length would not match the episode.
                    if scr and scr.poll() is None:
                        try:
                            scr.communicate(input=b"q", timeout=20)
                        except Exception:
                            scr.kill()

            interrupted = {"v": False}

            def _interrupt_pilot():
                """End the flight here (early stop, server death): SIGINT → stop instrumentation → forced kill after grace.
                SIGINT is sent first — _end_flight's trajectory join and recording shutdown can take a few seconds, and this keeps
                the executor from flying further in the meantime. The caller must keep reading stdout (if it stops, the pipe fills up)."""
                if interrupted["v"]:
                    return
                interrupted["v"] = True
                try:
                    proc.send_signal(signal.SIGINT)
                except Exception:
                    pass
                _end_flight()

                def _abort_kill(p=proc, g=cleanup_grace):
                    try:
                        p.wait(timeout=g)
                    except Exception:
                        try:
                            p.kill()
                        except Exception:
                            pass
                threading.Thread(target=_abort_kill, daemon=True).start()

            def _watchdog():
                if not done.wait(ep_timeout):
                    timed_out["v"] = True
                    print(f"  ! {tag}: exceeded {ep_timeout}s — prompting wrap-up with SIGINT", flush=True)
                    # stop instrumentation the moment the cap is reached. While the executor does wrap-up such as mp4 assembly
                    # after SIGINT (up to cleanup_grace), the target keeps moving and
                    # the trajectory keeps accumulating, so the post-cap segment leaked into judging (measured: in 5 episodes
                    # near samples occurred after 300s, 10 seconds' worth in one episode).
                    _end_flight()
                    if not interrupted["v"]:          # no duplicate SIGINT if already sent by early stop or server death
                        interrupted["v"] = True
                        try:
                            proc.send_signal(signal.SIGINT)
                        except Exception:
                            pass
                    if not done.wait(cleanup_grace):
                        print(f"  ! {tag}: wrap-up exceeded {cleanup_grace}s — force kill", flush=True)
                        try:
                            proc.kill()
                        except Exception:
                            pass
            threading.Thread(target=_watchdog, daemon=True).start()

            qf_evs = []          # confirmed-declaration steps (references elements of step_evs)
            step_evs = []        # measurement per model output — the sole input to SR·OSR
            pending = {}         # drone → pose taken at [FRAME]. That drone's [ACTION] confirms it
            decls = DeclarationScanner()
            lines = []
            # VLM request failure and sim error instrumentation (the guardrail session's per-episode meter). Consecutive failures are
            # the signature of a dead server (APIConnectionError on every step yields a plausible score of 0
            # that is otherwise discovered only afterward).
            # with enforce, feed returns True the moment the consecutive limit is hit, and the episode is cut.
            guard = session.begin_episode(tag)
            host_sims = _host_sim_count()
            # online verdict state — shared by single and multi. Multi detects with the declaring drone's camera.
            online = online_verdict
            early_stop = {"v": False}      # a satisfying declaration was confirmed mid-flight and the episode was cut
            online_checks = 0              # number of frames in this episode where capture-time detection ran (for logging)
            for line in proc.stdout:
                lines.append(line)
                # nothing after the cap (timeout) is counted — even after SIGINT, the executor
                # keeps flying for tens of seconds until the in-flight VLM request finishes, and measuring declarations
                # in that window lets activity outside 300s into judging (measured: E-05 was marked success with
                # 7 declarations at t=502~535s — the trace was cut at 300.98s and the minimum distance within it
                # was 16.9m). Same after an early stop — steps after that are inertia and landing during wrap-up.
                live = not timed_out["v"] and not early_stop["v"]
                if guard.feed(line, live):
                    print(f"  !! {tag}: {guard.vlm_fail_streak} consecutive VLM request failures — "
                          f"server presumed dead, cutting the episode", flush=True)
                    _interrupt_pilot()
                # capture moment — take the pose, and if the distance is within δ attach visibility on the spot too (below).
                if live and FRAME_RE.match(line):
                    if pending:
                        # drones that did not get an [ACTION] for the previous frame remain (VLM error, parse failure).
                        _left = ",".join(sorted(k or "single" for k in pending))
                        print(f"  ! {tag}: frame without [ACTION] ({_left}) — dropped", flush=True)
                    try:
                        # taken per drone — in multi, one capture records N drones at once.
                        # single is [""], same as before.
                        t_cap = round(time.time() - t0, 2)
                        pending = {}
                        for _d in (drone_names or [""]):
                            pending[_d] = _pose_now(mc, tgt_sim, _d)
                            pending[_d]["t"] = t_cap
                            pending[_d]["drone"] = _d
                    except Exception as e:
                        guard.sim_error()
                        print(f"  ! {tag}: capture-time pose failed {e}", flush=True)
                        pending = None
                    # capture-time visibility — only frames within δ, detection only with that drone's camera (≈8ms).
                    # the episode survives a failure: that frame is judged by replay after the episode ends.
                    if online and pending:
                        for _d, _ev in pending.items():
                            try:
                                if capture_visibility(mc, tgt_sim, _ev, float(delta), camera="0",
                                                      vehicle_name=_d) is not None:
                                    online_checks += 1
                            except Exception as e:
                                print(f"  ! {tag}: capture-time detection failed {e} — replay decides", flush=True)
                # the answer given for that frame — attach the verb to the preceding [FRAME] record to confirm it.
                # multi tells which drone answered by the `drone=<name>` at the end of the line.
                m_act = ACTION_RE.match(line) if live else None
                if m_act and pending:
                    _d = action_drone(line)
                    ev = pending.pop(_d, None)
                    if ev is None:
                        print(f"  ! {tag}: [ACTION] drone={_d!r} has no capture record — dropped",
                              flush=True)
                    else:
                        ev["action"] = m_act.group(1)
                        step_evs.append(ev)
                        guard.vlm_ok()        # the response is alive — reset consecutive failures
                # step timing that comes right after [ACTION] — for single, attached to that drone's event;
                # for a multi team step (no tail), attached to all drones at the same capture time.
                # ([TIMING] comes before the declaration marker, so it is already attached before an early stop.)
                m_tim = TIMING_RE.match(line) if live else None
                if m_tim and step_evs:
                    attach_timing(step_evs, m_tim, action_drone(line))
                if live and decls.feed(line):
                    # confirmed declaration — the preceding FINISHED measurement is this very moment (measured: the FINISHED
                    # line comes before ARRIVAL n/N, counts match 20/20). Only marks it.
                    qv = pick_declared_measurement(step_evs, decls.vehicle)
                    if qv is None:
                        print(f"  ! {tag}: no FINISHED measurement right before the declaration — dropped from SR",
                              flush=True)
                        continue
                    qv["declared"] = True
                    qf_evs.append(qv)
                    if not online:
                        continue
                    # online verdict — if this declaration's capture-time visibility is already attached, judge immediately with it.
                    # frames short on distance or with failed detection are judged by replay as before.
                    if qv.get("visibility_src") != "capture":
                        continue
                    if not qv["visible"]:
                        print(f"  · {tag}: declaration #{len(qf_evs)} {qv['dist_m']:.1f}m — 0 capture-time detections, "
                              f"still flying", flush=True)
                        continue
                    early_stop["v"] = True
                    print(f"  · {tag}: declaration #{len(qf_evs)} meets the condition "
                          f"({qv['dist_m']:.1f}m ≤ δ{float(delta):g}m, visible) t={qv.get('t')}s "
                          f"— early stop ({online_checks} capture-time detection frames)", flush=True)
                    _interrupt_pilot()
            proc.stdout.close()
            proc.wait()
            done.set()
            _end_flight()
            # post-hoc replay — only declarations without capture-time visibility (ONLINE_VERDICT=0, detection failure)
            # are filled by rewinding to the recorded poses. In cross-check mode, capture-judged declarations are replayed too.
            # check the sim is alive first: running replay on a dead sim makes the runner stall for hours
            # with 30s timeouts per step (measured: 1-hour hang after a UE segfault).
            sim_dead = not _port_up(port)
            if sim_dead:
                print(f"  !! {tag}: the sim (:{port}) is dead — skipping visibility replay"
                      + (" (guardrail: sim_died invalid → re-flight)" if session.enforcing else ""),
                      flush=True)
                n_rep = 0
            else:
                n_rep = replay_visibility(mc, tgt_sim, step_evs, float(delta),
                                          crosscheck=online_crosscheck, spawn_off=spawn_off)
            if n_rep:
                print(f"  · {tag}: visibility replay {n_rep}", flush=True)
            if online_crosscheck:
                for e in step_evs:
                    if e.get("visibility_src") == "capture" and "visible_replay" in e:
                        same = bool(e["visible"]) == bool(e["visible_replay"])
                        print(f"  {'·' if same else '!!'} {tag}: capture/replay cross-check "
                              f"{'match' if same else 'mismatch'} (capture={e['visible']} "
                              f"bbox={e.get('bbox_px')} / replay={e['visible_replay']} "
                              f"bbox={e.get('bbox_px_replay')}) t={e.get('t')}s", flush=True)
            hit = next((e for e in step_evs
                        if e.get("declared") and e.get("visible")
                        and judged_dist(e) is not None and judged_dist(e) <= float(delta)), None)
            # representative verdict = the satisfying declaration (if any), otherwise the last declaration.
            # the report and is_success below look at this one point.
            v = dict(hit) if hit is not None else (dict(qf_evs[-1]) if qf_evs else None)
            wall = flight_wall[0] if flight_wall[0] is not None else (time.time() - t0)
            out = "".join(lines)

            # end_reason: qf now means "ended with a declaration **that satisfied the condition**"
            # (previously any declaration meant qf — trace.is_success uses this value as a necessary
            # condition for success, so narrowing its meaning is safer).
            # if no satisfying declaration appears it runs to the cap, so timeout; everything else (crash,
            # giving up the search) is eof. Whether any declaration was made is seen in qf_declarations.
            if hit is not None:
                end_reason = "qf"
            elif timed_out["v"]:
                end_reason = "timeout"
            else:
                end_reason = "eof"
            if v is None:
                # ended without a declaration — use the last capture record as the representative value (for NE).
                # measuring anew here would catch a position outside the cap. In multi too, drone is attached to that record.
                v = dict(step_evs[-1]) if step_evs else {}
            v["end_reason"] = end_reason
                # early-stopped episodes have no trajectory afterward and cannot be re-judged with a stricter δ — leave a marker.
            v["early_stop"] = early_stop["v"]
            v["online_checks"] = online_checks
            # keep all declarations — when re-judging with a different δ, which declaration satisfies it
            # changes, so the single representative point is not enough.
            v["qf_events"] = qf_evs
            v["qf_declarations"] = len(qf_evs)
            v["step_events"] = step_evs
            # integrity — may this episode go into the results? Signals and thresholds are in guardrail/integrity.py.
            # None with off (no integrity key in the verdict). The mode is left in the verdict so whoever reads the data
            # can tell "whether the numbers went through the clean filter".
            integ = session.assess(guard, step_evs,
                                   sim_restarts=sim_restarts, host_sims=host_sims,
                                   # early-stopped (success-confirmed) episodes are short, so a moving target may not travel 5m —
                                   # target_stuck is only checked on episodes that flew to the cap (measured: 2 false positives on 23s success episodes).
                                   moving_target=moving and not early_stop["v"],
                                   sim_dead=sim_dead)
            v["guardrail_mode"] = session.mode
            if integ is not None:
                v["integrity"] = integ
            # the runner records the verdict itself (previously an executor hook wrote it). RESUME and reports read it as is.
            try:
                json.dump(v, open(vpath, "w"))
            except Exception:
                pass

            trace = tr.result()
            if scr:
                trace["screen_mp4"] = os.path.relpath(scr_path, out_root)
                trace["screen_t0"] = scr_t0

            # keep the executor's raw log — the reasons behind action decisions (anti-spin, giving up search, clamping)
            # are all printed here, and until now it was being thrown away. To diagnose failures after the fact,
            # verdict numbers alone are not enough.
            try:
                with open(os.path.join(mdir, f"pilot_{tag}.log"), "w") as lf:
                    lf.write(out)
            except Exception as e:
                print(f"  ! {tag}: failed to save the executor log {e}", flush=True)

            # the run folder the executor created (where decision/mp4/grid live)
            run_dir = ""
            for line in out.splitlines():
                if "[GRID] saved:" in line or "flight_cam" in line:
                    run_dir = os.path.dirname(line.split("saved: ")[-1].split()[0])
                    break
            if not run_dir:
                # fallback when stdout parsing fails — find the outputs folder created on disk after this episode
                # started. If the grid was not created the marker above was missing and the path was lost
                # entirely (measured: for timed-out episodes not even the output location was recorded).
                oroot = os.path.join(PILOT, "outputs")
                cands = [d for d in os.listdir(oroot)
                         if os.path.isdir(os.path.join(oroot, d))
                         and os.path.getmtime(os.path.join(oroot, d)) >= t0 - 5] \
                    if os.path.isdir(oroot) else []
                if cands:
                    run_dir = os.path.join("outputs", max(cands))
            # also copy recognition_grid into the model folder — browse per-episode decision summaries as
            # <model>/grid_*.jpg without digging into runs/<target>/<time>/.
            # per-step decision_*.json repeat the common meta (model, instruction, etc.) on every step,
            # so they are merged into one steps_<tag>.json as meta once + a steps array.
            # (the full raw exchange with the model is in pilot_<tag>.log in its entirety.)
            try:
                import glob as _glob
                gdir = run_dir if os.path.isabs(run_dir) else os.path.join(PILOT, run_dir)
                gsrc = os.path.join(gdir, "recognition_grid.jpg")
                if run_dir and os.path.exists(gsrc):
                    shutil.copy(gsrc, os.path.join(mdir, f"grid_{tag}.jpg"))
                djs = sorted(_glob.glob(os.path.join(gdir, "decision_*.json"))) if run_dir else []
                if djs:
                    meta, steps_l = {}, []
                    for dj in djs:
                        d = json.load(open(dj))
                        if not meta:
                            meta = {k: d.get(k) for k in
                                    ("model", "api_provider", "mode",
                                     "adaptive_mode", "instruction")}
                        steps_l.append(dict(timestamp=d.get("timestamp"),
                                            actions=d.get("actions")))
                    json.dump(dict(**meta, steps=steps_l),
                              open(os.path.join(mdir, f"steps_{tag}.json"), "w"),
                              ensure_ascii=False, indent=1)
            except Exception:
                pass
            # step count = number of "Action Response:" lines, printed exactly once per loop iteration.
            steps = out.count("Action Response:")
            if not steps and drone_names:
                # multi prints as many [ACTION] lines per step as drones instead of "Action Response:" → converted to team step count.
                n_act = sum(1 for l in out.splitlines() if ACTION_RE.match(l))
                steps = n_act // max(1, len(drone_names))
            rows.append(dict(model=model, target=tgt, color=ep.get("color"),
                             task=task, reason=v.get("reason"),
                             end_reason=v.get("end_reason"), dist_m=v.get("dist_m"),
                             visible=v.get("visible"), bbox_px=v.get("bbox_px"),
                             qf_declarations=v.get("qf_declarations"),
                             early_stop=early_stop["v"],
                             osr_min_m=step_min(step_evs),
                             sr_min_m=step_min(step_evs, declared=True, visible=True),
                             steps=steps, wall_s=round(wall), run_dir=run_dir,
                             delta_m=float(delta), **trace))
            r = rows[-1]
            # the minimum distance cannot exceed the final distance. The trajectory is cut at SIGINT, but
            # the verdict is measured afterward during wrap-up, so approaching further in between inverts them
            # (measured Car_21: trajectory min 13.18 vs verdict 12.76). The verdict distance is a real observation too,
            # so it is included in the minimum.
            if r["dist_m"] is not None:
                r["min_dist_m"] = (r["dist_m"] if r["min_dist_m"] is None
                                   else round(min(r["min_dist_m"], r["dist_m"]), 2))
            # searching validity (not visible at start) trusts the testset spec —
            # start-time detection wobbles with render timing and is unstable as judging input.
            if "seen_at_start" not in v and "visible_at_start" in spec:
                v["seen_at_start"] = spec["visible_at_start"]
            # per-task judging — approaching/searching by distance, the tracking family by visibility held.
            v["delta_m"] = float(delta)   # the δ this episode was judged with — so aggregation (sr_at) and RESUME use the same δ
            tv = judge_task(task, v, os.path.join(mdir, f"trace_{tag}.jsonl"),
                            delta_m=float(delta))
            r["success"] = tv.success
            r["task_reason"] = tv.reason
            r["task_verdict"] = tv.to_dict()
            # guardrail fields (guardrail_mode·integrity_valid·integrity_reasons). Aggregation
            # filters with is_excluded — only enforce's invalid episodes drop out; audit's suspect episodes stay in.
            r.update(session.row_fields(integ))
            # write the verdict back into the verdict file too — RESUME skipping and report regeneration read only this file,
            # so without it, rows carried over after an interruption propagate as success=None.
            v["success"] = tv.success
            v["reason"] = tv.reason
            v["task_verdict"] = tv.to_dict()
            try:
                json.dump(v, open(vpath, "w"))
            except Exception:
                pass
            print(f"[{model:9}] {tgt:9} {task:14} "
                  f"{'✅' if r['success'] else '❌'} {str(tv.reason):26} "
                  f"dist={r['dist_m'] if r['dist_m'] is None else round(r['dist_m'],1)}m "
                  f"min={r['min_dist_m']}m "
                  f"end={r['end_reason']}{'(early)' if early_stop['v'] else ''} "
                  f"steps={steps} {round(wall)}s", flush=True)
            # guardrail action — with enforce, contaminated episodes are removed from the results and the same episode is
            # re-flown immediately (the session moves the verdict to _invalid/ and counts attempts). If contaminated,
            # it is invalid regardless of success/failure — not just SR but step, time, and path metrics are also distorted
            # (clean data first). audit/off always keep.
            if session.after_episode(tag, vpath, integ, r["success"]) == "reflight":
                rows.remove(r)
                ep_queue.appendleft(ep)
            # per-model summary goes in the model folder — even if parallel runners use the same OUT_DIR
            # they do not overwrite each other. The run-level summary/report is scanned and merged at the end.
            # it must be written after integrity handling so the state with invalid episodes removed is what lands on disk.
            json.dump([r for r in rows if r["model"] == model],
                      open(os.path.join(mdir, "summary.json"), "w"), indent=1)
            # server-death circuit breaker (enforce) — do not burn the remaining episodes on a dead server.
            # Skipped episodes have no verdict, so
            # after the server recovers they can be flown as is with RESUME=1.
            if session.model_should_stop():
                print(f"  !!!! {model}: api_dead on {session.api_dead_flights} episodes in a row — "
                      f"treating the server as dead and stopping this model's remaining "
                      f"{len(ep_queue)} episodes (RESUME=1 after recovery)", flush=True)
                ep_queue.clear()

    print(f"\n=== per-model performance (executor flights · success threshold {DELTA_M:g}m) ===")
    print(f"{'model':15} {'SR':>8} {'OSR':>6} {'NE mean':>8} {'min mean':>9} {'approach':>7} "
          f"{'steps':>6} {'time':>7} {'declared':>9}")
    report = {}
    for m in models:
        rs_all = [r for r in rows if r["model"] == m]
        if not rs_all:
            continue
        # enforce's integrity-invalid episodes are not mixed into the score — contaminated episodes have every metric
        # (SR, steps, time) distorted regardless of outcome (success/failure) (clean data first).
        # how many episodes were dropped and why is printed separately below the table. audit's suspect episodes stay in (marked only).
        rs = [r for r in rs_all if not is_excluded(r)]
        inv = [r for r in rs_all if is_excluded(r)]
        flagged = [r for r in rs if r.get("integrity_valid") is False]
        if not rs:
            print(f"{m:15} (no valid episodes — {len(inv)} invalid: "
                  f"{sorted({x for r in inv for x in (r.get('integrity_reasons') or [])})})",
                  flush=True)
            continue
        n = len(rs)
        ne = [r["dist_m"] for r in rs if r["dist_m"] is not None]
        md = [r["min_dist_m"] for r in rs if r.get("min_dist_m") is not None]
        # approach ratio = (start distance - min distance)/start distance — "how much closer it got relative to the start".
        # even without success, whether it approached at all is useful for comparing models.
        prog = [(r["start_dist_m"] - r["min_dist_m"]) / r["start_dist_m"]
                for r in rs if r.get("start_dist_m") and r.get("min_dist_m") is not None]
        report[m] = sr_at(rs)
        print(f"{m:15} {sum(1 for r in rs if r['success'])}/{n:<3} {report[m][f'{DELTA_M:g}m']['SR']*100:3.0f}% "
              f"{report[m][f'{DELTA_M:g}m']['OSR']*100:5.0f}% "
              f"{(sum(ne)/len(ne) if ne else float('nan')):8.1f} "
              f"{(sum(md)/len(md) if md else float('nan')):9.1f} "
              f"{(sum(prog)/len(prog)*100 if prog else float('nan')):6.0f}% "
              f"{sum(r['steps'] for r in rs)/n:6.1f} "
              f"{sum(r['wall_s'] for r in rs)/n:6.0f}s "
              # 'declared arrival' = episodes with at least one declaration. end_reason=='qf' now
              # means only 'satisfying declaration', so it equals SR and the column becomes meaningless.
              f"{sum(1 for r in rs if (r.get('qf_declarations') or 0) > 0)}/{n}")
        if inv:
            print(f"{'':15} ⚠ {len(inv)} invalid episodes excluded: "
                  + ", ".join(f"{r['target']}({','.join(r.get('integrity_reasons') or [])})"
                              for r in inv), flush=True)
        if flagged:
            print(f"{'':15} ⓘ {len(flagged)} audit-suspect episodes (included in aggregation): "
                  + ", ".join(f"{r['target']}({','.join(r.get('integrity_reasons') or [])})"
                              for r in flagged), flush=True)
    print(f"  SR = at least one arrival declaration within {DELTA_M:g}m + target visible on screen "
          f"(tracking family: held after first approach) · NE = mean error at the decisive declaration")
    print("  declared = episodes with at least one arrival declaration (whether or not the condition was met)")
    print(f"  OSR = at least one step measurement within {DELTA_M:g}m (regardless of declaration or visibility)")

    # run-level summary/report — scan all <model>/summary.json on disk and merge them.
    # even if parallel runners share the same OUT_DIR, each rewrites this at the end, so
    # in the end every model that completed is included (idempotent merge, no lock needed).
    all_rows = []
    for md in sorted(os.listdir(out_root)):
        sp = os.path.join(out_root, md, "summary.json")
        if os.path.isdir(os.path.join(out_root, md)) and os.path.exists(sp):
            try:
                all_rows += json.load(open(sp))
            except Exception:
                pass
    valid_rows = [r for r in all_rows if not is_excluded(r)]
    merged_report = {m2: sr_at([r for r in valid_rows if r["model"] == m2])
                     for m2 in sorted({r["model"] for r in valid_rows})}
    # invalid = episodes dropped from aggregation by enforce, audit_flagged = episodes audit only marked as suspect (included in aggregation).
    invalid_summary, flagged_summary = summarize_rows(all_rows)
    json.dump(all_rows, open(os.path.join(out_root, "summary.json"), "w"),
              ensure_ascii=False, indent=1)
    n_invalid = sum(len(v2) for v2 in invalid_summary.values())
    if n_invalid:
        print(f"\n⚠ {n_invalid} final invalid episodes (excluded from aggregation):", flush=True)
        for m2, items in sorted(invalid_summary.items()):
            print(f"  {m2}: " + ", ".join(
                f"{it['target']}({','.join(it['reasons'] or [])})" for it in items),
                flush=True)
        print(f"  → after fixing the contamination cause (load/server), rerun in the same OUT_DIR with GUARDRAIL=enforce RESUME=1 "
              f"and only these episodes are re-flown automatically.", flush=True)
    n_flagged = sum(len(v2) for v2 in flagged_summary.values())
    if n_flagged:
        print(f"\nⓘ {n_flagged} audit-suspect episodes (included in aggregation — would be re-flown under GUARDRAIL=enforce):",
              flush=True)
        for m2, items in sorted(flagged_summary.items()):
            print(f"  {m2}: " + ", ".join(
                f"{it['target']}({','.join(it['reasons'] or [])})" for it in items),
                flush=True)
    modes = sorted({r.get("guardrail_mode") or "enforce" for r in all_rows})
    json.dump(dict(rows=all_rows, sr_at=merged_report, invalid=invalid_summary,
                   audit_flagged=flagged_summary, guardrail_modes=modes),
              open(os.path.join(out_root, "report.json"), "w"), ensure_ascii=False, indent=1)
    print(f"\nsummary -> {os.path.join(out_root, 'summary.json')} ({len(all_rows)} rows merged)")
    print(f"report  -> {os.path.join(out_root, 'report.json')}")


if __name__ == "__main__":
    main()
