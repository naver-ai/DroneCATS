#!/usr/bin/env bash
# DroneCATS
# Copyright (c) 2026-present NAVER Cloud Corp.
# Apache-2.0

# Spectator viewer wrapper — run directly from the repo root
#
# Usage:
#   scripts/airsim/spectate.sh                 # spectate the sim on $MY_PORT (optional env default, else 41451) in third person
#   scripts/airsim/spectate.sh --camera 0      # first person (model's view)
#
# Browser: http://localhost:<sim port+1000>
set -euo pipefail
cd "$(dirname "$0")/../../vlmpilot"
PY=".venv/bin/python"; [ -x "$PY" ] || PY="uv run python"
exec $PY ../scripts/airsim/spectate.py "$@"
