#!/bin/bash
# DroneCATS
# Copyright (c) 2026-present NAVER Cloud Corp.
# Apache-2.0

# Stop the pod on its own when the benchmark ends, so an unattended run does not keep billing.
#   run log shows ALL DONE (or SETUP_FAILED)     → pack results → stop
#   no new file under the results dir for STALL_S → pack results → stop
#   HARD_CAP_UTC reached                          → pack results → stop
# Results live under /workspace, which survives a stop. The stop call uses the pod-scoped RUNPOD_API_KEY that
# RunPod injects into every pod; nothing is written to disk. run.sh starts this for you (AUTOSTOP=1).
#
#   HARD_CAP_UTC="2026-10-20 06:00" RUN_LOG=/workspace/results/isaac/<name>/run.log \
#     setsid nohup bash scripts/runpod/isaacsim/watchdog.sh </dev/null >/dev/null 2>&1 &
#   DRY_RUN=1 runs one check and logs instead of stopping.
set -u
STALL_S=${STALL_S:-3600}; HARD_CAP_UTC=${HARD_CAP_UTC:?e.g. "2026-10-20 06:00"}
RUN_LOG=${RUN_LOG:?path of the run log}; RES=${RES_DIR:-$(dirname "$RUN_LOG")}
LOG=${WATCHDOG_LOG:-$RES/watchdog.log}; DRY=${DRY_RUN:-0}
log(){ echo "[$(date -u +%FT%TZ)] $*" >> "$LOG"; }
env1(){ tr '\0' '\n' </proc/1/environ 2>/dev/null | sed -n "s/^$1=//p"; }
pack(){ tar czf "$RES.tgz" -C "$(dirname "$RES")" "$(basename "$RES")" 2>/dev/null; log "packed $(du -h "$RES.tgz" | cut -f1) → $RES.tgz"; }
stop_pod(){
  log "stopping: $1"
  if [ "$DRY" = 1 ]; then log "[dry] POST https://rest.runpod.io/v1/pods/<RUNPOD_POD_ID>/stop"; exit 0; fi
  local code; code=$(curl -sS -o /dev/null -w '%{http_code}' -X POST \
      -H "Authorization: Bearer $(env1 RUNPOD_API_KEY)" "https://rest.runpod.io/v1/pods/$(env1 RUNPOD_POD_ID)/stop")
  log "REST stop HTTP $code"; [ "${code#2}" != "$code" ] && exit 0
  RUNPOD_API_KEY="$(env1 RUNPOD_API_KEY)" runpodctl pod stop "$(env1 RUNPOD_POD_ID)" >> "$LOG" 2>&1 && exit 0
  log "stop failed — retrying in 5 min"; }
cap=$(date -u -d "$HARD_CAP_UTC" +%s 2>/dev/null || date -j -u -f "%Y-%m-%d %H:%M" "$HARD_CAP_UTC" +%s)
newest(){ find "$RES" -type f -printf '%T@\n' 2>/dev/null | sort -n | tail -1 | cut -d. -f1; }
log "watching $RUN_LOG · stall ${STALL_S}s · hard cap $HARD_CAP_UTC UTC"
while :; do
  now=$(date +%s)
  [ "$now" -ge "$cap" ] && { pack; stop_pod "hard cap"; }
  if grep -qE "ALL DONE|SETUP_FAILED" "$RUN_LOG" 2>/dev/null; then pack; stop_pod "run finished"; fi
  n=$(newest); [ -n "$n" ] && [ $(( now - n )) -ge "$STALL_S" ] && { pack; stop_pod "no progress for ${STALL_S}s"; }
  [ "$DRY" = 1 ] && { log "[dry] one check done"; exit 0; }
  sleep 60
done
