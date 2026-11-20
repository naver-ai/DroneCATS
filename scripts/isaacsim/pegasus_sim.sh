#!/usr/bin/env bash
# DroneCATS
# Copyright (c) 2026-present NAVER Cloud Corp.
# Apache-2.0

# Process A launcher: adds GPU-pinning arguments (see scripts/runpod/isaacsim/README.md).
#
# Usage: bash scripts/isaacsim/pegasus_sim.sh [PEGASUS_GPU] [-- pegasus_sim.py args...]
#   e.g. bash scripts/isaacsim/pegasus_sim.sh                    # GPU 3, Curved Gridroom, headless
#        bash scripts/isaacsim/pegasus_sim.sh 2 -- --gui         # GPU 2, viewport on
#        bash scripts/isaacsim/pegasus_sim.sh 3 -- --env "Warehouse with Forklifts"
#
# ⚠️ Always pin a single GPU. Otherwise Isaac Sim tries to grab every GPU on the machine
#    and collides with other workloads (e.g. an AirSim benchmark) on the same server.
set -euo pipefail

GPU="${1:-3}"
[ $# -gt 0 ] && shift
[ "${1:-}" = "--" ] && shift

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
: "${ISAACSIM_PYTHON:=$HOME/isaacsim/python.sh}"

if [ ! -x "$ISAACSIM_PYTHON" ]; then
  echo "❌ ISAACSIM_PYTHON not found: $ISAACSIM_PYTHON" >&2
  exit 1
fi

# -- Detect an A that is already running -----------------------------------
# -- Slots --------------------------------------------------------------------
# With several simulators on one pod (one per slot/port), `pgrep` also matches **the other
# slots' processes**. Left unscoped, the two cleanup blocks below would kill their sim and
# PX4, so they are narrowed by slot.
#
# **Untagged processes count as slot 0**, so a single simulator without slots behaves the
# same as slot 0.
SLOT=0
_prev=""
for _a in "$@"; do
  case "$_a" in
    --slot=*) SLOT="${_a#--slot=}" ;;
    *) [ "$_prev" = "--slot" ] && SLOT="$_a" ;;
  esac
  _prev="$_a"
done
# Give A itself the tag too: changing os.environ in Python does not change
# /proc/self/environ (exec-time value). It must be exported here for cleanup tools to identify A.
export VLMPILOT_SLOT="$SLOT"

# Slot tag of that PID. Missing tag reads as 0 (rule above).
_slot_of() {
  local t
  t="$(tr '\0' '\n' < "/proc/$1/environ" 2>/dev/null | sed -n 's/^VLMPILOT_SLOT=//p' | head -1)"
  echo "${t:-0}"
}
# Keep only processes of my slot.
_mine_only() {
  local out="" p
  for p in $1; do
    [ "$(_slot_of "$p")" = "$SLOT" ] && out="$out $p"
  done
  echo "$out" | tr -s ' ' | sed 's/^ //;s/ $//'
}

# Starting two As makes the second fail to bind TCP 4560 (A<->PX4 sim link) and crash
# inside Pegasus; the cause only shows after digging through logs. Meanwhile the first A
# keeps running, so it looks like "it started, why isn't it working". Block it here first.
#   Do not kill another slot's sim casually: refuse and let the operator decide.
#   With FORCE=1, clean up and continue.
RUNNING="$(pgrep -f 'isaacsim/kit/python.*pegasus_sim\.py' | grep -v "^$$\$" || true)"
RUNNING="$(_mine_only "$RUNNING")"   # leave other slots alone
if [ -n "$RUNNING" ]; then
  if [ "${FORCE:-0}" = "1" ]; then
    echo "▶ FORCE=1 - cleaning up only slot $SLOT's A: $(echo "$RUNNING" | tr '\n' ' ')"
    # shellcheck disable=SC2086
    kill -TERM $RUNNING 2>/dev/null || true
    sleep 10
  else
    echo "❌ process A is already running in slot $SLOT: $(echo "$RUNNING" | tr '\n' ' ')" >&2
    echo "   A second one would fail to bind TCP 4560." >&2
    echo "   To work in another slot:  bash scripts/isaacsim/pegasus_sim.sh $GPU -- --slot <N> ..." >&2
    echo "   To stop this slot and restart:  FORCE=1 bash scripts/isaacsim/pegasus_sim.sh $GPU -- --slot $SLOT ..." >&2
    exit 1
  fi
fi

# -- Clean up leftover PX4 ---------------------------------------------------
# If a PX4 from a previous session is alive when A restarts, Pegasus reuses it with
# "PX4 server already running for instance 0". But that PX4's sim link points at the dead
# A and does not attach to the new A's TCP 4560: A hangs forever at "Waiting for first
# heartbeat" and the runner dies on the MAVSDK connect timeout.
# The symptom only looks like "cannot connect", so the cause is hard to find. Remove it here first.
# Zombies (Z) do not go away with kill; they stay in pgrep until the parent reaps them.
# Reporting them as "cleaned up" would be false, so only live ones are selected.
# ⚠️ This block runs **every time**, without FORCE; unscoped, it would silently kill other
# slots' PX4. PX4 started by A carries the slot tag by inheriting `os.environ`.
STALE=""
for _p in $(pgrep -f 'px4_sitl_default/bin/px4' || true); do
  _st="$(awk '{print $3}' "/proc/$_p/stat" 2>/dev/null || echo Z)"
  [ "$_st" = "Z" ] && continue
  [ "$(_slot_of "$_p")" = "$SLOT" ] || continue
  STALE="$STALE $_p"
done
STALE="$(echo "$STALE" | tr -s ' ' | sed 's/^ //;s/ $//')"
if [ -n "$STALE" ]; then
  echo "▶ cleaning up previous-session PX4 in slot $SLOT: $(echo "$STALE" | tr '\n' ' ')"
  # shellcheck disable=SC2086
  kill -TERM $STALE 2>/dev/null || true
  sleep 3
  LEFT="$(_mine_only "$(pgrep -f 'px4_sitl_default/bin/px4' || true)")"
  # shellcheck disable=SC2086
  [ -n "$LEFT" ] && { kill -KILL $LEFT 2>/dev/null || true; sleep 1; }
fi

echo "▶ starting process A on GPU $GPU · slot $SLOT"
exec "$ISAACSIM_PYTHON" "$REPO/scripts/isaacsim/pegasus_sim.py" \
  --/renderer/multiGpu/enabled=false \
  --/renderer/activeGpu="$GPU" \
  --/physics/cudaDevice="$GPU" \
  "$@"
