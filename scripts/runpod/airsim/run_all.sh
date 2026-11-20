#!/usr/bin/env bash
# DroneCATS
# Copyright (c) 2026-present NAVER Cloud Corp.
# Apache-2.0

# One model through the whole benchmark on one GPU pod, with simulators in parallel.
#
#   MODEL=gpt6-luna-or bash scripts/runpod/airsim/run_all.sh
#
# Parts (PARTS, default "single multi_self multi_agents"):
#   single        8 single-drone sets (NH + campus, 10 episodes each), SIMS simulators in parallel
#   multi_self    3 multi-drone levels (30 episodes), the model drives all four drones itself
#   multi_agents  same 30 episodes, the model commands four Gemini Robotics-ER sub-agents
#
# Environment (all optional except MODEL):
#   MODEL        MODELS= name for single-drone runs (see verifier/airsim/run_testset.py), e.g. gpt6-luna-or, gemini38
#   MULTI_CFG    multi-drone config   (default: MULTI_CONFIGS[MODEL] in verifier/airsim/run_testset.py)
#   AGENTS_CFG   commander + ER config (required for multi_agents)
#   SIMS         parallel simulators for single (default 10; 8-10 per RTX 4090)
#   MSIMS        parallel simulators for multi  (default 6)
#   SETS         space-separated single-drone sets to run (default: all 8)
#   OUT          output directory (default /workspace/results/<MODEL>-<UTC time>)
#
# Episodes a crashed simulator skipped (ECONNREFUSED) are rerun afterwards, one fresh simulator each.
# Prints the summary at the end; later, `python3 scripts/runpod/airsim/summarize.py` shows the newest run again.
# Also writes $OUT.tar.gz: the results without camera frames (*.jpg), for copying off the pod.
set -u
cd "$(dirname "$0")/../../.."
: "${MODEL:?set MODEL, e.g. MODEL=gpt6-luna-or}"
PARTS=${PARTS:-"single multi_self multi_agents"}
SIMS=${SIMS:-10}; MSIMS=${MSIMS:-6}
SETS=${SETS:-"nh_approaching nh_searching nh_tracking nh_search_tracking campus_approaching campus_searching campus_tracking campus_search_tracking"}
OUT=${OUT:-/workspace/results/$MODEL-$(date -u +%m%d_%H%M)}
PY=${PY:-vlmpilot/.venv/bin/python}
export EP_TIMEOUT=${EP_TIMEOUT:-300}
if [ -z "${MULTI_CFG:-}" ]; then
  # read MULTI_CONFIGS from the runner's source (importing it would start a run)
  MULTI_CFG=$(python3 - "$MODEL" <<'PY'
import re, sys
src = open("verifier/airsim/run_testset.py", encoding="utf-8").read()
block = src[src.index("MULTI_CONFIGS = {"):]; block = block[:block.index("\n}")]
m = re.search(r'"%s"\s*:\s*"([^"]+)"' % re.escape(sys.argv[1]), block)
print(m.group(1) if m else "")
PY
)
fi

# ── API keys set as RunPod Secrets live in the pod's environment (PID 1); a shell opened over SSH does not inherit
# them. Fill every key that is unset here from there and export it, so the preflight below and the runners see it.
# Same approach as setup.sh and scripts/runpod/isaacsim/run.sh. KEYS_ENVIRON overrides the source file (for testing).
load_pod_keys(){
  local src=${KEYS_ENVIRON:-/proc/1/environ} k v
  [ -r "$src" ] || return 0
  for k in "$@"; do
    [ -n "${!k:-}" ] && continue
    v=$(tr '\0' '\n' < "$src" 2>/dev/null | sed -n "s/^$k=//p" | head -n 1)
    [ -n "$v" ] && export "$k=$v"
  done
  return 0
}
load_pod_keys OPENROUTER_API_KEY GEMINI_API_KEY OPENAI_API_KEY OPENAI_DIRECT_API_KEY

# ── preflight: every API key the chosen configs read must be set (otherwise it only shows up mid-run)
pre=$($PY - "$MODEL" "$PARTS" "${MULTI_CFG:-}" "${AGENTS_CFG:-}" <<'PY'
import os, re, sys, yaml
model, parts, multi, agents = sys.argv[1], sys.argv[2].split(), sys.argv[3], sys.argv[4]
src = open("verifier/airsim/run_testset.py", encoding="utf-8").read()
block = src[src.index("CONFIGS = {"):]; block = block[:block.index("\n}")]
m = re.search(r'"%s"\s*:\s*"([^"]+)"' % re.escape(model), block)
files = []
if "single" in parts and m: files.append(m.group(1))
if "multi_self" in parts and multi: files.append(multi)
if "multi_agents" in parts and agents: files.append(agents)

def key(c, default_provider):
    p = c.get("api_provider", default_provider)
    if p == "gemini":
        return "GEMINI_API_KEY"
    if p == "openai" and (c.get("api_key_env") or "openai.com" in str(c.get("openai_base_url", "openai.com"))):
        return c.get("api_key_env") or "OPENAI_API_KEY"
    return None  # local endpoints need no key

need = set()
for f in files:
    c = yaml.safe_load(open(os.path.join("vlmpilot", f), encoding="utf-8"))
    top = {k: v for k, v in c.items() if k not in ("head", "body")}
    if c.get("agent_mode") == "head_body":
        need.add(key({**top, **c.get("head", {})}, "gemini"))
        body = {**top, **c.get("body", {})}
        if c.get("body", {}).get("base_urls"):
            body["openai_base_url"] = c["body"]["base_urls"][0]
        need.add(key(body, "openai"))
    else:
        need.add(key(top, "openai"))
print(" ".join(sorted(k for k in need if k and not os.environ.get(k, "").strip())))  # line 1: missing
print(" ".join(sorted(k for k in need if k)))                                         # line 2: all needed
PY
) || { echo "could not read the configs" >&2; exit 1; }
missing=$(sed -n 1p <<<"$pre"); need_keys=$(sed -n 2p <<<"$pre")
if [ -n "$missing" ]; then
  echo "missing API key(s) for PARTS=\"$PARTS\": $missing" >&2
  echo "set them first (a RunPod Secret in the pod environment, or read -s NAME && export NAME), or drop the part that needs them from PARTS" >&2
  exit 1
fi

# ── preflight: the keys must also work. A revoked key or one with no credit left fails every call, and the runs
# would still go on to their time limits. One free request per key (no model call); SKIP_KEY_CHECK=1 skips it.
if [ "${SKIP_KEY_CHECK:-0}" != 1 ] && [ -n "$need_keys" ]; then
  bad=$($PY - $need_keys <<'PY'
import json, os, sys, urllib.error, urllib.request

def get(url, headers=None):
    req = urllib.request.Request(url, headers={"User-Agent": "dronecats-preflight", **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()
    except Exception as e:                      # no network: do not block the run, only warn
        return None, str(e).encode()

for name in sys.argv[1:]:
    key = os.environ.get(name, "").strip()
    if name == "OPENROUTER_API_KEY":
        code, body = get("https://openrouter.ai/api/v1/key", {"Authorization": f"Bearer {key}"})
        if code == 200:
            left = (json.loads(body).get("data") or {}).get("limit_remaining")
            if left is not None and left <= 0:
                print(f"{name}: the key has no credit left (OpenRouter key limit reached)")
            continue
    elif name == "GEMINI_API_KEY":
        code, body = get(f"https://generativelanguage.googleapis.com/v1beta/models?pageSize=1&key={key}")
    elif name.startswith("OPENAI"):
        code, body = get("https://api.openai.com/v1/models", {"Authorization": f"Bearer {key}"})
    else:
        continue
    if code is None:
        print(f"warning: could not check {name} ({body.decode()[:80]})", file=sys.stderr)
    elif code in (401, 403) or (code == 400 and b"API_KEY_INVALID" in body):
        print(f"{name}: rejected by the provider (HTTP {code})")
    elif code != 200:
        print(f"warning: {name} check returned HTTP {code}", file=sys.stderr)
PY
)
  if [ -n "$bad" ]; then
    echo "$bad" >&2
    echo "fix the key (for a RunPod Secret, change it and start a new pod: a restart keeps the old value), or set SKIP_KEY_CHECK=1" >&2
    exit 1
  fi
fi
mkdir -p "$OUT"; echo "results → $OUT"

waitport(){ for _ in $(seq 1 60); do ss -ltn | grep -q ":$1 " && return 0; sleep 5; done; echo "simulator on $1 did not start" >&2; return 1; }
sim(){ # sim <map> <port> [multi]
  if [ "${3:-}" = multi ]; then SETTINGS_BASE=vlmpilot/settings.multidrone.json.example bash scripts/airsim/launch_airsim.sh "$1" "$2" > "$OUT/sim_$2.log" 2>&1 < /dev/null
  else bash scripts/airsim/launch_airsim.sh "$1" "$2" > "$OUT/sim_$2.log" 2>&1 < /dev/null; fi
  waitport "$2" && sleep 5; }
stop_sims(){ for p in "$@"; do pkill -f "[s]ettings_$p.json" 2>/dev/null; done; sleep 2; }

# ── single drone: each set split in two chunks of 5, chunks dealt round-robin to SIMS workers
if [[ " $PARTS " == *" single "* ]]; then
  mkdir -p "$OUT/single/chunks"; : > "$OUT/single/chunks.txt"
  for s in $SETS; do $PY - "$s" "$OUT/single" >> "$OUT/single/chunks.txt" <<'PY'
import json, sys
s, out = sys.argv[1], sys.argv[2]; d = json.load(open(f"episodes/airsim/single-drone/{s}.json")); eps = d["episodes"]
for k in range(0, len(eps), 5):
    c = dict(d); c["episodes"] = eps[k:k + 5]; p = f"{out}/chunks/{s}__{k}.json"
    json.dump(c, open(p, "w"), ensure_ascii=False); print(d.get("scene", "nh"), p)
PY
  done
  date -u +%T > "$OUT/single/start"
  worker(){ local i=$1 p=$((46001 + $1)) n=0
    while read -r scene path; do
      [ $((n % SIMS)) -eq "$i" ] || { n=$((n + 1)); continue; }; n=$((n + 1))
      sim "$scene" $p || continue
      PORT=$p MODELS=$MODEL OUT_DIR="$OUT/single/$(basename "$path" .json)" $PY -m verifier.airsim.run_testset "$path" \
        > "$OUT/single/log_$(basename "$path" .json).txt" 2>&1 < /dev/null
    done < "$OUT/single/chunks.txt"; stop_sims $p; }
  for i in $(seq 0 $((SIMS - 1))); do worker "$i" & done; wait
  date -u +%T > "$OUT/single/end"
fi

# ── multi-drone: one part = 3 levels × 10 episodes
LEVELS="nh:nh_multi-drone blocks:blocks_multi-drone-hard blocks:blocks_multi-drone-sum_main"
ids(){ $PY -c "import json; print(' '.join(t['id'] for t in json.load(open('episodes/airsim/multi-drone/$1.json'))['tasks']))"; }
multi(){ local part=$1 cfg=$2 i=0 pids=()
  [ -n "$cfg" ] || { echo "skip $part: no config" >&2; return; }
  mkdir -p "$OUT/$part"; date -u +%T > "$OUT/$part/start"
  # first pass: each level in chunks over MSIMS simulators
  local per=$(( (30 + MSIMS - 1) / MSIMS ))
  for lv in $LEVELS; do map=${lv%%:*}; set_=${lv#*:}
    all=($(ids "$set_"))
    for ((k = 0; k < ${#all[@]}; k += per)); do
      chunk=$(IFS=,; echo "${all[*]:k:per}"); p=$((46101 + i)); i=$((i + 1))
      ( sim "$map" $p multi && PORT=$p OUT_DIR="$OUT/$part/${set_}__$k" $PY -m verifier.airsim.run_bench_tasks "episodes/airsim/multi-drone/$set_.json" \
          --config "$cfg" --ids "$chunk" --runs 1 > "$OUT/$part/log_${set_}__$k.txt" 2>&1 < /dev/null; stop_sims $p ) &
    done
  done; wait
  # second pass: episodes a crashed simulator skipped, one fresh simulator per episode
  for lv in $LEVELS; do map=${lv%%:*}; set_=${lv#*:}
    for id in $(ids "$set_"); do
      ls "$OUT/$part"/${set_}__*/"$id"/run1/verdict.json > /dev/null 2>&1 && continue
      echo "rerun $part $set_ $id"; p=46199
      sim "$map" $p multi && PORT=$p OUT_DIR="$OUT/$part/${set_}__re_$id" $PY -m verifier.airsim.run_bench_tasks "episodes/airsim/multi-drone/$set_.json" \
        --config "$cfg" --ids "$id" --runs 1 >> "$OUT/$part/log_rerun.txt" 2>&1 < /dev/null; stop_sims $p
    done
  done
  date -u +%T > "$OUT/$part/end"; }
[[ " $PARTS " == *" multi_self "* ]] && multi multi_self "$MULTI_CFG"
[[ " $PARTS " == *" multi_agents "* ]] && multi multi_agents "${AGENTS_CFG:-}"
echo "done → $OUT"; python3 scripts/runpod/airsim/summarize.py "$OUT"
# results without camera frames (json, logs, trajectory plots; ~2 MB for 80 episodes) in one file next to $OUT,
# so getting them off the pod is one scp: the network volume opens only on pods in its own data center
tar czf "$OUT.tar.gz" --exclude='*.jpg' -C "$(dirname "$OUT")" "$(basename "$OUT")" && echo "results without frames → $OUT.tar.gz"
