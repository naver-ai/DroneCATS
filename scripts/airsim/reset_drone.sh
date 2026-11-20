#!/usr/bin/env bash
# DroneCATS
# Copyright (c) 2026-present NAVER Cloud Corp.
# Apache-2.0

# Drone reset — returns the vehicle to its spawn point (landed) without restarting the sim.
#
# Usage: scripts/airsim/reset_drone.sh [RPC_PORT]   (default: $MY_PORT if set, else 41451)
# e.g.   scripts/airsim/reset_drone.sh 41452
set -euo pipefail
# Port precedence: argument > $MY_PORT (optional environment default) > 41451
PORT="${1:-${MY_PORT:-41451}}"
cd "$(dirname "$0")/../../vlmpilot"
# sudo resets PATH so uv cannot be found; use the venv python directly
PY=".venv/bin/python"; [ -x "$PY" ] || PY="$(command -v uv >/dev/null && echo 'uv run python' || echo python3)"
$PY -c "
import airsim
c = airsim.MultirotorClient(port=$PORT, timeout_value=15)
c.confirmConnection()
c.reset()
print('✅ drone reset done (port $PORT) — at the spawn point, landed')"
