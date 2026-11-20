#!/bin/bash
# DroneCATS
# Copyright (c) 2026-present NAVER Cloud Corp.
# Apache-2.0

# Run the Isaac Sim (Pegasus + PX4) closed-loop benchmark for one model on one GPU pod.
#
#   bash scripts/runpod/isaacsim/run.sh <name> <config> [batch …]
#   e.g. bash scripts/runpod/isaacsim/run.sh gemini37 vlmpilot/configs/isaacsim/single-drone/eval_px4_gemini37-or.yaml
#        bash scripts/runpod/isaacsim/run.sh gemini37 vlmpilot/configs/isaacsim/single-drone/eval_px4_gemini37-or.yaml keepA   # half, on pod A
#
# The 28 episodes come in three scene batches (keep 17 · topup 7 · p28 4); the simulator is restarted per batch
# with that batch's targets. To split one model over two pods use keepA (14) on one and "keepB topup p28" (14)
# on the other. Results go to $OUT_ROOT/<name>/ (default /workspace/results/isaac) and survive a pod stop.
#
# Environment
#   OPENROUTER_API_KEY / GEMINI_API_KEY — whatever the config's provider needs. If unset in the shell they are
#                        read from the pod's environment (RunPod Secrets). Never write keys to files.
#   GT_MODE (measure)  — ground truth only within 10 m of the target; judging is unchanged, sim ~2x faster
#   SETTLE_S (20)      — margin after each PX4 restart, on top of what batch_run.py already waits for
#                        (14580/udp re-bound, and a new heartbeat in A's log). Measured 2026-10-21
#                        (A/B, 15 resets per value): 20 · 30 · 150 all armed 45/45 and mean
#                        connect-to-arm was 15.6-16.3 s whatever the value (individual runs
#                        13.3-19.2 s), so the earlier 150 bought nothing and cost ~130 s per
#                        episode (~1 h over 28). It does not move a target's start point either:
#                        movers stand still until the drone takes off (--mover-takeoff-gate-m).
#                        The reference runs used 150; set SETTLE_S=150 to match them exactly
#   A_WAIT_S (2400)    — cap for waiting on the simulator to come up; polled every 2 s, so a fast
#                        start costs nothing. batch_run.py polls again (--wait-a-ready-s, 900 s)
#   RETRIES (2)        — retries for transient failures (arm failure right after a PX4 restart, timeouts)
#   AUTOSTOP (1)       — start watchdog.sh: stop the pod when done / stuck for 1 h / at HARD_CAP_UTC.
#                        A stop wipes the container disk: keep $OUT_ROOT on a volume disk, or set AUTOSTOP=0
#   LIMIT              — run only the first LIMIT episodes of each batch (smoke test)
#   GT_OCCLUSION_VECTORIZED (0) — 1 = numpy GT occlusion check: same values, faster sim, but moving-target
#                        outcomes change with RTF, so it is off for the benchmark
#   HARD_CAP_UTC       — absolute stop time for the watchdog (default: now + 12 h)
#   DRY_RUN=1          — start nothing and write nothing, only print what would run
set -u
name=${1:?name}; cfg=${2:?config}; shift 2
BATCHES=("$@"); [ ${#BATCHES[@]} -eq 0 ] && BATCHES=(keep topup p28)
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
C="$REPO/episodes/isaacsim/single-drone"
O="${OUT_ROOT:-/workspace/results/isaac}/$name"
SETTLE_S=${SETTLE_S:-20}; GT_MODE=${GT_MODE:-measure}; RETRIES=${RETRIES:-2}; A_WAIT_S=${A_WAIT_S:-2400}
DRY=${DRY_RUN:-0}; VENV=${VENV:-/root/venv-pegasus}
ASSETS=/root/omniverse-assets
export ISAACSIM_PATH=/root/isaacsim
# Use only the GLX Vulkan ICD the container runtime provides (two ICDs make Isaac warn about duplicate drivers).
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
export DATAGEN_BROWNSTONE=$ASSETS/Brownstone/Demos/AEC/BrownstoneDemo
export DATAGEN_ISAACPEOPLE=$ASSETS/IsaacPeople DATAGEN_VEHICLES=$ASSETS/Vehicles
export DATAGEN_IRIS=/root/PegasusSimulator/extensions/pegasus.simulator/pegasus/simulator/assets/Robots/Iris/iris.usd
if [ -r /proc/1/environ ]; then
  for k in OPENROUTER_API_KEY GEMINI_API_KEY OPENAI_API_KEY OPENAI_DIRECT_API_KEY; do
    [ -n "${!k:-}" ] || export "$k=$(tr '\0' '\n' </proc/1/environ 2>/dev/null | sed -n "s/^$k=//p")"
  done
fi
say(){ echo "[$name $(date +%H:%M:%S)] $*"; }
[ -f "$REPO/$cfg" ] || [ -f "$cfg" ] || { say "config not found: $cfg"; exit 2; }
for b in "${BATCHES[@]}"; do [ -f "$C/plan.$b.jsonl" ] || { say "no episodes: $C/plan.$b.jsonl"; exit 2; }; done
key_env=$(sed -n 's/^api_key_env: *"\{0,1\}\([A-Z_]*\)"\{0,1\}.*/\1/p' "$REPO/$cfg" 2>/dev/null || true)
[ "$(sed -n 's/^api_provider: *"\{0,1\}\([a-z]*\).*/\1/p' "$REPO/$cfg" 2>/dev/null)" = gemini ] && key_env=GEMINI_API_KEY
if [ -n "$key_env" ] && [ -z "${!key_env:-}" ] && [ "$DRY" != 1 ]; then
  say "$key_env is empty — export it (read -s $key_env && export $key_env) or attach a RunPod Secret"; exit 2; fi
[ "$DRY" = 1 ] || mkdir -p "$O"; cd "$REPO"
if [ "${AUTOSTOP:-1}" = 1 ] && [ "$DRY" != 1 ]; then
  HARD_CAP_UTC=${HARD_CAP_UTC:-$(date -u -d '+12 hours' '+%Y-%m-%d %H:%M')} RUN_LOG="$O/run.log" \
    setsid nohup bash "$REPO/scripts/runpod/isaacsim/watchdog.sh" </dev/null >/dev/null 2>&1 &
  say "watchdog on: stops the pod when this run ends (log $O/watchdog.log)"
fi
[ "$DRY" = 1 ] || exec > >(tee -a "$O/run.log") 2>&1
if [ "${AUTOSTOP:-1}" = 1 ] && [ "$DRY" != 1 ] && [ "$(findmnt -n -o TARGET -T "$O" 2>/dev/null)" = / ]; then
  say "WARNING: $O is on the container disk, not a volume. The watchdog's pod stop wipes it, results included."
  say "         Attach a volume at /workspace (or set OUT_ROOT to one), or re-run with AUTOSTOP=0."
fi
run(){ if [ "$DRY" = 1 ]; then echo "  [dry] $*"; else "$@"; fi; }
down(){ [ "$DRY" = 1 ] && return 0
        pkill -f launch_brownstone.py; pkill -f mavsdk_server; pkill -x px4; sleep 5
        pkill -9 -f launch_brownstone.py; pkill -9 -x px4; rm -rf /tmp/vlmpilot_pegasus /tmp/vlmpilot_slots; }
wait_gt(){ [ "$DRY" = 1 ] && return 0
  local t0; t0=$(date +%s); local p=/tmp/vlmpilot_pegasus/latest_gt.json
  while [ $(( $(date +%s) - t0 )) -lt "$A_WAIT_S" ]; do
    [ -f $p ] && [ $(( $(date +%s) - $(stat -c %Y $p) )) -lt 5 ] && { say "simulator ready ($(( $(date +%s) - t0 )) s)"; return 0; }
    sleep 2; done; say "simulator did not come up within ${A_WAIT_S} s"; return 1; }
keep_runs(){ # copy run folders from the container disk next to the results
  [ "$DRY" = 1 ] && return 0
  python3 - "$O/out_$1/batch.jsonl" "$O/runs" <<'PY'
import json, os, shutil, sys
src, dst = sys.argv[1], sys.argv[2]
if os.path.exists(src):
    os.makedirs(dst, exist_ok=True)
    for l in open(src):
        rd = json.loads(l).get("run_dir") or ""
        if rd and os.path.isdir(rd) and not os.path.exists(os.path.join(dst, os.path.basename(rd))):
            shutil.copytree(rd, os.path.join(dst, os.path.basename(rd)))
PY
}
batch(){
  local B=$1 plan=$C/plan.$1.jsonl TGT AT
  case $B in keep*) TGT=$C/keep_targets.txt AT=$C/keep_at.txt ;;
             topup*) TGT=$C/topup_targets.txt AT=$C/topup_at.txt ;;
             p28*) TGT=$C/p28_targets.txt AT=$C/p28_at.txt ;;
             *) say "batch must start with keep/topup/p28: $B"; return 1 ;; esac
  local N; N=$(wc -l < "$plan" | tr -d ' '); local GT=/root/gt.$B ALOG=$O/A.$B.log
  [ -n "${LIMIT:-}" ] && [ "$LIMIT" -lt "$N" ] && N=$LIMIT
  say "▶ $B: $N episodes"; down; [ "$DRY" = 1 ] || rm -rf "$GT"
  if [ "$DRY" = 1 ]; then echo "  [dry] simulator: launch_brownstone.py --gt-mode $GT_MODE --targets <$(basename "$TGT")> > $ALOG"
  else
  setsid nohup /root/isaacsim/python.sh scripts/isaacsim/sim/launch_brownstone.py \
    --variant Morning --spawn-xy=0,0 --render-aa dlaa --ground-desaturate 0.0 \
    --targets "$(cat "$TGT")" --target-at="$(cat "$AT")" \
    --target-oneway --target-speed 0.3 \
    --gt --gt-mode "$GT_MODE" --gt-dir "$GT" --gt-res 640x480 --gt-targets 'DG_target' \
    --keep-alive --mover-takeoff-gate-m 1.5 \
    --vehicles 1 --render-hz 10 --camera-width 640 --camera-height 480 > "$ALOG" 2>&1 </dev/null &
  fi
  wait_gt || return 1
  run timeout 28800 python3 verifier/isaacsim/batch_run.py --plan "$plan" \
      --episodes "$N" --gt-dir "$GT" --config "$cfg" \
      --out "$O/out_$B" --px4-reset always --settle-s "$SETTLE_S" \
      --px4-timeout 600 --b-timeout 3000 --a-log "$ALOG" \
      --b-python "$VENV/bin/python" \
      --retries "$RETRIES" --fov-deg 90 --end-on any
  say "■ $B done"; keep_runs "$B"
}
say "start: config=$cfg batches=${BATCHES[*]} gt-mode=$GT_MODE settle=${SETTLE_S}s retries=$RETRIES out=$O${LIMIT:+ limit=$LIMIT}"
for b in "${BATCHES[@]}"; do batch "$b"; done
down
if [ "$DRY" != 1 ]; then
  grep -ho "RTF=[0-9.]*" "$O"/A.*.log 2>/dev/null | sed 's/RTF=//' | sort -n | \
    awk '{a[NR]=$1} END{if(NR) printf "{\"n\": %d, \"median\": %s, \"p10\": %s}\n", NR, a[int(NR/2)+1], a[int(NR*0.1)+1]}' > "$O/rtf.json"
  nvidia-smi --query-gpu=name,driver_version --format=csv,noheader > "$O/host_gpu.txt" 2>/dev/null; nproc > "$O/host_vcpu.txt"
fi
say "ALL DONE"
