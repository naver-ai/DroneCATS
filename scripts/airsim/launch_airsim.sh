#!/usr/bin/env bash
# DroneCATS
# Copyright (c) 2026-present NAVER Cloud Corp.
# Apache-2.0

# AirSim launcher — one simulator per RPC port, so several can run side by side
#
# Usage (as root; on a RunPod pod you already are root):
#   scripts/airsim/launch_airsim.sh <MAP> [RPC_PORT] [GPU_INDEX] [onscreen]
#
#   MAP       : blocks | nh | campus | mountains | africa | park  (aliases, case-insensitive;
#               scripts/runpod/airsim/setup.sh installs nh, campus and blocks)
#   RPC_PORT  : AirSim RPC port (default: the optional environment variable $MY_PORT if set, else 41451).
#               Use a different port per simulator;
#               scripts/runpod/airsim/run_all.sh uses 46001+ (single drone) and 46101+ (multi-drone).
#   GPU_INDEX : -graphicsadapter value (default 0). For spreading across GPUs — adapter order may differ
#               from nvidia-smi indices, so check the actual placement with nvidia-smi.
#   onscreen  : passing 'onscreen' as the 4th argument runs windowed on X display :0. Needs an X server on :0
#               (a headless RunPod pod has none); one onscreen instance at a time.
#
# Default is offscreen (-RenderOffScreen): no X server needed, several instances per GPU possible.
# To watch a simulator, use the web viewer:
#   cd vlmpilot && uv run python ../scripts/airsim/spectate.py --rpc <PORT> --port 89xx --camera chase
#
# e.g.
#   bash scripts/airsim/launch_airsim.sh nh 46001                 # NH on 46001, GPU0, offscreen
#   bash scripts/airsim/launch_airsim.sh campus 46002 1           # campus on 46002, GPU1
#   bash scripts/airsim/launch_airsim.sh nh 46001 0 onscreen      # windowed on :0 (machines with a display only)
#
# UE4 refuses to run as root → runs as the airsim user (this script itself runs as root).
set -euo pipefail

MAP_RAW="${1:?MAP required: blocks|nh|mountains|africa|park|campus}"
# Port precedence: argument > $MY_PORT (optional environment default) > 41451
RPC_PORT="${2:-${MY_PORT:-41451}}"
GPU_IDX="${3:-0}"
MODE="${4:-offscreen}"
RESX=1280; RESY=720

# Map alias → .sh launcher (directory/launcher names differ per map, so mapped explicitly)
MAP="$(echo "$MAP_RAW" | tr '[:upper:]' '[:lower:]')"
case "$MAP" in
  blocks)              LAUNCHER="/opt/airsim_envs/LinuxBlocks1.8.1/LinuxNoEditor/Blocks.sh" ;;
  nh|airsimnh)         LAUNCHER="/opt/airsim_envs/AirSimNH/LinuxNoEditor/AirSimNH.sh" ;;
  mountains|landscape) LAUNCHER="/opt/airsim_envs/LandscapeMountains/LinuxNoEditor/LandscapeMountains.sh" ;;
  africa|savannah)     LAUNCHER="/opt/airsim_envs/Africa_Savannah/LinuxNoEditor/Africa_001.sh" ;;
  park|abandoned)      LAUNCHER="/opt/airsim_envs/AbandonedPark/LinuxNoEditor/AbandonedPark.sh" ;;
  campus|city|msbuild) LAUNCHER="/opt/airsim_envs/MSBuild2018/LinuxNoEditor/MSBuild2018.sh" ;;
  *) echo "unknown map: $MAP_RAW  (blocks|nh|mountains|africa|park|campus)"; exit 1 ;;
esac
[ -x "$LAUNCHER" ] || { echo "launcher missing/not executable: $LAUNCHER"; exit 1; }
[ "$MAP" = "africa" ] && echo "⚠ Africa is an old build, so the drone body is not visible in the spectate (chase) view (flight/FPV are fine)"

# Generate a per-port settings.json — the RPC port can only be changed via ApiServerPort in settings.
# SETTINGS_BASE selects a different base settings file — multi-drone uses vlmpilot/settings.multidrone.json.example
# (4 Vehicles). The default is the single-vehicle base file.
BASE_SETTINGS="${SETTINGS_BASE:-/home/airsim/Documents/AirSim/settings.json}"
SLOT_SETTINGS="/home/airsim/Documents/AirSim/settings_${RPC_PORT}.json"
python3 - "$BASE_SETTINGS" "$SLOT_SETTINGS" "$RPC_PORT" <<'PY'
import json, sys
base, out, port = sys.argv[1], sys.argv[2], int(sys.argv[3])
with open(base) as f:
    cfg = json.load(f)
cfg["ApiServerPort"] = port
with open(out, "w") as f:
    json.dump(cfg, f, indent=4)
print(f"settings generated: {out} (ApiServerPort={port})")
PY
chown airsim:airsim "$SLOT_SETTINGS"

LOG="/tmp/airsim_${MAP}_p${RPC_PORT}.log"
if [ "$MODE" = "onscreen" ]; then
  # Onscreen (:0 windowed mode): renders on the X server at :0, so only :0 is supported and the GPU is the one
  # that X server runs on (GPU_INDEX is not passed here).
  DISPLAY=:0 xhost +SI:localuser:airsim >/dev/null 2>&1 || true
  RUNCMD="DISPLAY=:0 setsid '$LAUNCHER' -windowed -ResX=$RESX -ResY=$RESY -settings='$SLOT_SETTINGS' >'$LOG' 2>&1 &"
  echo "▶ MAP=$MAP  RPC=$RPC_PORT  onscreen(:0/GPU0)  log=$LOG"
else
  # offscreen (default): no X needed, any GPU, multiple instances per GPU possible
  RUNCMD="setsid '$LAUNCHER' -RenderOffScreen -graphicsadapter=$GPU_IDX -windowed -ResX=$RESX -ResY=$RESY -settings='$SLOT_SETTINGS' >'$LOG' 2>&1 &"
  echo "▶ MAP=$MAP  RPC=$RPC_PORT  offscreen(adapter=$GPU_IDX)  log=$LOG"
fi

# Clean up existing instances on the same port — otherwise on restart the old process keeps holding the port
# while a new instance comes up, and several pile up per port (seen in practice: 3 UE processes per port, high
# load, and lost episodes). The runner restarts frequently.
STALE=$(pgrep -f "settings_${RPC_PORT}([^0-9]|$)" | tr '\n' ' ' || true)   # so no-match (exit 1) does not trip set -e
if [ -n "$STALE" ]; then
  echo "  cleaning up existing ${RPC_PORT} instance: $STALE"
  for p in $STALE; do kill -9 "$p" 2>/dev/null; done
  for i in $(seq 1 30); do ss -tln 2>/dev/null | grep -q ":${RPC_PORT} " || break; sleep 1; done
fi
runuser -u airsim -- bash -c "$RUNCMD"

echo "  waiting for boot... (small maps ~40s, large maps ~90s)"
for i in $(seq 1 150); do
  if ss -tln 2>/dev/null | grep -q ":${RPC_PORT} "; then
    echo "✅ RPC $RPC_PORT LISTEN — ready"; exit 0
  fi
  sleep 1
done
echo "⚠ ${RPC_PORT} did not open within 150 s — check the log: $LOG"; exit 1
