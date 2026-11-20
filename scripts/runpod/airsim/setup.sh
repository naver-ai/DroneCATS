#!/usr/bin/env bash
# DroneCATS
# Copyright (c) 2026-present NAVER Cloud Corp.
# Apache-2.0

# One-command RunPod GPU pod setup — AirSimNH (UE4) + vlmpilot + verifier benchmark stack
#
# Usage (inside the pod, as root, from the repo root):
#   bash scripts/runpod/airsim/setup.sh            # full setup (idempotent — safe to rerun any number of times)
#   bash scripts/runpod/airsim/setup.sh --check    # verify only, no install (Vulkan/ICD/GPU + RPC boot)
#
# Environment variables (all optional):
#   RUNPOD_VOLUME : Network Volume mount point (default /workspace). The extracted AirSimNH goes here.
#   SMOKE_PORT    : temporary RPC port for the smoke/boot check (default 46001). Any free TCP port works;
#                   the check stops its simulator again, so only avoid a port a running simulator is using.
#   KEEP_ZIP=1    : keep the downloaded AirSimNH.zip (≈2.0GiB) on the volume after extraction (for reinstalls).
#   SKIP_SMOKE=1  : skip the final smoke test (actual sim launch) — when you only want setup.
#
# e.g.
#   bash scripts/runpod/airsim/setup.sh                          # standard (install + smoke)
#   SKIP_SMOKE=1 bash scripts/runpod/airsim/setup.sh             # install only
#   SMOKE_PORT=46002 bash scripts/runpod/airsim/setup.sh --check # verify only, on a different port
#
# What it does (8 steps, each idempotent — skipped if already done):
#   1) apt dependencies  2) Vulkan/ICD check  3) airsim user  4) AirSimNH 1.8.1 download+extract
#   5) /opt/airsim_envs symlink  6) uv + vlmpilot dependencies  7) base settings.json
#   8) .env guidance + smoke test (boot sim → confirm RPC LISTEN, then exit)
#
# Pod creation prerequisites:
#   - env NVIDIA_DRIVER_CAPABILITIES=all  ← without it the Vulkan ICD is not injected into the
#     container and UE4 cannot find a render device. Exporting it inside an already running pod
#     does not help (injection happens at container start) — add it via pod Edit and restart.
#   - Mount a Network Volume at /workspace (holds the ~5.4GB extracted AirSimNH, survives pod recreation)
#   - The container starts as root. UE4 refuses to run as root, so an airsim user is created and
#     the sim runs via runuser (same as scripts/airsim/launch_airsim.sh).
#   - Passthrough GPU, so no X server needed — only -RenderOffScreen (Vulkan offscreen) has to work.
#
# Benchmark run order after setup (from the repo root):
#   API keys: RunPod Secrets or a shell export — not a file on the volume
#   bash scripts/airsim/launch_airsim.sh nh 46001          # simulator on RPC port 46001
#   PORT=46001 MODELS=gpt6-luna-or vlmpilot/.venv/bin/python -m verifier.airsim.run_testset episodes/airsim/single-drone/nh_approaching.json
#
# Measured basis (checked at release time):
#   - AirSimNH 1.8.1 public binary: AirSimNH.zip from microsoft/AirSim GitHub releases v1.8.1.
#     Confirmed 302 (release-assets.githubusercontent.com) → 200 with curl -sIL,
#     Content-Length 2,110,059,210 bytes (≈2.0GiB, ~5.4GB extracted).
#   - zip layout: first local header inspected via a 1KB range request — entries start with 'AirSimNH/'.
#     → extracting into /workspace/airsim_envs makes launch_airsim.sh work unmodified with just the /opt/airsim_envs symlink.
#   - base settings.json: the /home/airsim/Documents/AirSim/settings.json the benchmark runs used
#     (chase camera + Recording block — the verifier's FPV/spectator video assembly uses these settings).
set -euo pipefail

say()  { echo "▶ $*"; }
ok()   { echo "✅ $*"; }
warn() { echo "⚠ $*"; }
die()  { echo "❌ $*" >&2; exit 1; }

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
VP_DIR="$REPO/vlmpilot"

VOL="${RUNPOD_VOLUME:-/workspace}"
ENVS_DIR="$VOL/airsim_envs"
NH_DIR="$ENVS_DIR/AirSimNH"
LAUNCHER="$NH_DIR/LinuxNoEditor/AirSimNH.sh"
OPT_LINK="/opt/airsim_envs"

NH_URL="https://github.com/microsoft/AirSim/releases/download/v1.8.1/AirSimNH.zip"

if command -v mountpoint > /dev/null && ! mountpoint -q "$VOL"; then
  warn "$VOL is not a network volume — this pod has none attached (a volume only attaches to pods in its own data center)."
  warn "Everything installed below is lost when the pod stops. Deploy the pod from the volume (Storage → volume → Deploy) to keep it."
fi
NH_ZIP_SIZE=2110059210   # measured with curl -sIL at release time. May change if the asset is replaced, so only warn.

# Temporary RPC port for the smoke/boot check — any free port (see SMOKE_PORT above).
SMOKE_PORT="${SMOKE_PORT:-46001}"

SETTINGS_DIR="/home/airsim/Documents/AirSim"
BASE_SETTINGS="$SETTINGS_DIR/settings.json"

# Minimal packages for the UE4.27 offscreen Vulkan runtime (no X server needed, but X client
# libraries are linked, so they are required). iproute2 (ss) and procps (pgrep) are used by launch/smoke.
# git is for cloning the exported bundle and switching branches; ffmpeg is needed to assemble frames → mp4
# for vlmpilot recordings (vlmpilot/src/vlmpilot/artifacts.py) — without it the videos are silently empty.
APT_PKGS=(libvulkan1 vulkan-tools libsm6 libxext6 libxrandr2 libxcursor1 libxi6
          curl ca-certificates unzip git ffmpeg python3 iproute2 procps)

# uv installs into ~/.local/bin — it may be missing from PATH in non-interactive shells, so prepend it
# (otherwise every rerun would reinstall uv).
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

# ── Shared check helpers ─────────────────────────────────────────────────────────

check_driver_caps() {
  # Shells attached via sshd do not inherit the container env (observed) — if empty, read from PID 1's environment
  local caps="${NVIDIA_DRIVER_CAPABILITIES:-}"
  [ -n "$caps" ] || caps="$(tr '\0' '\n' < /proc/1/environ 2>/dev/null | sed -n 's/^NVIDIA_DRIVER_CAPABILITIES=//p')"
  case "$caps" in
    *all*|*graphics*) ok "NVIDIA_DRIVER_CAPABILITIES='$caps'" ;;
    *) warn "NVIDIA_DRIVER_CAPABILITIES='${caps:-<empty>}' — 'all' is required."
       echo "   Set it as an env var when creating/editing the pod and restart, so the Vulkan ICD gets injected." ;;
  esac
}

check_icd() {
  local f found=""
  for f in /usr/share/vulkan/icd.d/nvidia_icd.json /etc/vulkan/icd.d/nvidia_icd.json; do
    [ -f "$f" ] && found="$f" && break
  done
  if [ -n "$found" ]; then
    ok "NVIDIA Vulkan ICD: $found"
  else
    warn "NVIDIA Vulkan ICD (nvidia_icd.json) missing — UE4 cannot find the GPU."
    echo "   Most likely cause: pod env var NVIDIA_DRIVER_CAPABILITIES=all is missing (pod must be recreated)."
    echo "   Rarely, the host driver does not ship the Vulkan component (Community Cloud variance)"
    echo "   — it is faster to drop that pod and start again on another host."
  fi
}

ensure_egl_icd() {
  # The nvidia_icd.json injected by RunPod points at libGLX_nvidia.so.0, but in a container without X
  # it cannot obtain vkCreateInstance and fails with "Found no drivers" (observed on an RTX 4090 pod).
  # Adding a second ICD pointing at the same driver's libEGL_nvidia.so.0 makes it work headless.
  local egl=/usr/share/vulkan/icd.d/nvidia_egl_icd.json
  command -v vulkaninfo >/dev/null 2>&1 || return 0
  vulkaninfo --summary 2>/dev/null | grep -qi 'deviceName.*NVIDIA' && return 0
  ldconfig -p 2>/dev/null | grep -q 'libEGL_nvidia.so.0' || return 0
  [ -f "$egl" ] && return 0
  mkdir -p "$(dirname "$egl")"
  printf '{"file_format_version":"1.0.1","ICD":{"library_path":"libEGL_nvidia.so.0","api_version":"1.3.0"}}\n' > "$egl"
  ok "EGL Vulkan ICD added: $egl (the GLX ICD fails headless)"
}

check_vulkan_gpu() {
  # Check with vulkaninfo that a real render device is found. Seeing only llvmpipe (software) is a failure.
  if ! command -v vulkaninfo >/dev/null 2>&1; then
    warn "vulkaninfo missing — vulkan-tools is not installed, so the GPU cannot be verified."
    echo "   Fix: apt-get update && apt-get install -y vulkan-tools libvulkan1"
    echo "   (the default mode 'bash scripts/runpod/airsim/setup.sh' installs it automatically)"
    return 1
  fi
  local summary
  if ! summary="$(vulkaninfo --summary 2>&1)"; then
    warn "vulkaninfo failed:"
    echo "$summary" | head -5 | sed 's/^/   /'
    echo "   Hint: ICD not injected (item above) or a host driver problem."
    return 1
  fi
  echo "$summary" | grep -i 'deviceName' | sed 's/^[[:space:]]*/   /' || true
  if echo "$summary" | grep -qi 'deviceName.*NVIDIA'; then
    ok "Vulkan sees the NVIDIA GPU"
  elif echo "$summary" | grep -qi 'llvmpipe'; then
    warn "only llvmpipe (software renderer) is visible — NVIDIA ICD not injected. UE4 cannot render."
    return 1
  else
    warn "NVIDIA not found in the Vulkan device list — check the deviceName output above."
    return 1
  fi
}

wait_rpc_listen() {  # $1=port $2=timeout in seconds — returns 0 once LISTEN appears
  local port="$1" limit="$2"
  for _ in $(seq 1 "$limit"); do
    if ss -tln 2>/dev/null | grep -q ":${port} "; then return 0; fi
    sleep 1
  done
  return 1
}

kill_sim() {  # $1=settings filename pattern — kills only UE instances launched with those settings
  pkill -TERM -f "$1" 2>/dev/null || true
  sleep 3
  pkill -KILL -f "$1" 2>/dev/null || true
}

check_boot_rpc() {
  # Boot AirSimNH directly on a temporary port, confirm RPC LISTEN, then shut it down.
  # (a one-command version of the "30-minute manual Vulkan check on the pod" — if this comes up, render/RPC work)
  if [ ! -x "$LAUNCHER" ]; then
    warn "AirSimNH binary missing ($LAUNCHER) — skipping the boot check (the default mode installs it)"
    return 0
  fi
  [ "$(id -u)" = 0 ] || { warn "not root — cannot use runuser, skipping the boot check"; return 0; }
  id -u airsim >/dev/null 2>&1 || { warn "no airsim user — skipping the boot check (the default mode creates it)"; return 0; }
  # Without ss the LISTEN check cannot work and wastes 150 s — bail out early (for standalone --check runs).
  command -v ss >/dev/null 2>&1 \
    || { warn "ss missing (iproute2 not installed) — skipping the boot check. apt-get install -y iproute2"; return 0; }

  local port="$SMOKE_PORT"
  local chk_settings="/tmp/runpod_check_${port}.json"
  local log="/tmp/airsim_check_p${port}.log"
  printf '{\n  "SettingsVersion": 1.2,\n  "SimMode": "Multirotor",\n  "ApiServerPort": %s\n}\n' "$port" > "$chk_settings"
  chmod 644 "$chk_settings"

  say "AirSimNH temporary boot (RPC $port, offscreen)... log=$log"
  runuser -u airsim -- bash -c \
    "setsid '$LAUNCHER' -RenderOffScreen -nosound -windowed -ResX=640 -ResY=360 -settings='$chk_settings' >'$log' 2>&1 &"
  if wait_rpc_listen "$port" 150; then
    ok "RPC $port LISTEN — Vulkan rendering + AirSim RPC OK"
    kill_sim "runpod_check_${port}.json"
    return 0
  fi
  kill_sim "runpod_check_${port}.json"
  warn "RPC $port did not open within 150 s — end of the log:"
  tail -15 "$log" 2>/dev/null | sed 's/^/   /' || true
  echo "   Hint: 'Cannot find a compatible Vulkan...' → ICD/NVIDIA_DRIVER_CAPABILITIES problem."
  echo "         'Refusing to run with the root privileges' → it was not launched as the airsim user."
  return 1
}

# ── --check mode ────────────────────────────────────────────────────────────

if [ "${1:-}" = "--check" ]; then
  say "RunPod environment check (no installation)"
  fail=0
  check_driver_caps
  if command -v nvidia-smi >/dev/null 2>&1; then
    ok "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
  else
    warn "nvidia-smi missing — check that this is a GPU pod"
  fi
  check_icd
  check_vulkan_gpu || fail=1
  if [ "$fail" = 0 ]; then
    check_boot_rpc || fail=1
  else
    warn "failed at the Vulkan stage — skipping the boot check"
  fi
  if [ "$fail" = 0 ]; then ok "check passed"; else die "check failed — see the hints above"; fi
  exit 0
fi

[ -n "${1:-}" ] && die "unknown argument: $1  (usage: scripts/runpod/airsim/setup.sh [--check])"

# ── Default mode: full setup ─────────────────────────────────────────────────────

[ "$(id -u)" = 0 ] || die "must run as root (the RunPod container default is root)"

say "[1/8] installing apt dependencies (${APT_PKGS[*]})"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq || die "apt update failed — check the pod's outbound network"
apt-get install -y -qq --no-install-recommends "${APT_PKGS[@]}" \
  || die "apt install failed — check that the base image is ubuntu-based"
ok "apt done"

say "[2/8] checking Vulkan ICD / driver"
check_driver_caps
check_icd
ensure_egl_icd
check_vulkan_gpu || warn "continuing with the Vulkan GPU unverified — the smoke test is the final judge"

say "[3/8] airsim user (needed because UE4 refuses to run as root)"
if id -u airsim >/dev/null 2>&1; then
  ok "airsim user exists"
else
  useradd -m -s /bin/bash airsim || die "useradd failed"
  ok "airsim user created"
fi

say "[4/8] AirSimNH 1.8.1 binary ($ENVS_DIR)"
if [ -x "$LAUNCHER" ]; then
  ok "already installed — skipping download"
else
  [ -d "$VOL" ] || die "$VOL missing — check that the Network Volume is mounted at /workspace (can be changed with RUNPOD_VOLUME)"
  mkdir -p "$ENVS_DIR"
  ZIP="$VOL/AirSimNH.zip"
  # If the complete zip is already present (rerun after only extraction failed), skip the network.
  if [ -f "$ZIP" ] && [ "$(stat -c %s "$ZIP")" = "$NH_ZIP_SIZE" ]; then
    ok "zip already fully downloaded — skipping download: $ZIP"
  else
    say "downloading (~2.0GiB, resumable): $NH_URL"
    curl -fL --retry 3 -C - -o "$ZIP" "$NH_URL" \
      || die "download failed — check GitHub access/disk space (df -h $VOL). A partial file resumes on rerun."
    actual="$(stat -c %s "$ZIP")"
    [ "$actual" = "$NH_ZIP_SIZE" ] \
      || warn "zip size $actual ≠ measured $NH_ZIP_SIZE — the release asset may have changed (if extraction fails, rm $ZIP and rerun)"
  fi
  say "extracting (~5.4GB)..."
  unzip -q -o "$ZIP" -d "$ENVS_DIR" \
    || die "extraction failed — check disk space (df -h $VOL); if the zip is corrupt, rm $ZIP and rerun"
  [ -f "$LAUNCHER" ] || die "launcher still missing after extraction — did the zip layout change? (expected: AirSimNH/LinuxNoEditor/AirSimNH.sh)"
  chmod +x "$LAUNCHER" "$NH_DIR/LinuxNoEditor/AirSimNH/Binaries/Linux/"* 2>/dev/null || true
  if [ "${KEEP_ZIP:-0}" = 1 ]; then say "KEEP_ZIP=1 — keeping zip: $ZIP"; else rm -f "$ZIP"; fi
  ok "AirSimNH installed"
fi
# UE writes logs and crash dumps under the game directory (Saved/) — must be owned by the airsim user
if [ "$(stat -c %U "$NH_DIR")" != "airsim" ]; then
  say "fixing ownership (chown -R airsim, once)..."
  # The Network Volume (MooseFS FUSE) rejects chown — use a+rwX on everything instead so airsim can write
  chown -R airsim:airsim "$NH_DIR" 2>/dev/null \
    || { chmod -R a+rwX "$NH_DIR" 2>/dev/null || true; warn "chown not possible (network volume) — using a+rwX instead"; }
fi

# campus (MSBuild2018) and blocks maps — same v1.8.1 public release. Set MAPS="" to use NH only.
for m in ${MAPS-campus blocks}; do
  case "$m" in
    campus) Z=MSBuild2018.zip;   L="$ENVS_DIR/MSBuild2018/LinuxNoEditor/MSBuild2018.sh" ;;
    blocks) Z=Blocks.zip;        L="$ENVS_DIR/LinuxBlocks1.8.1/LinuxNoEditor/Blocks.sh" ;;
    *) warn "unknown map $m — skipped"; continue ;;
  esac
  if [ -f "$L" ]; then ok "$m map already present"; continue; fi
  say "fetching $m map ($Z)"
  curl -fL --retry 3 -o "$VOL/$Z" "https://github.com/microsoft/AirSim/releases/download/v1.8.1/$Z" \
    && unzip -q -o "$VOL/$Z" -d "$ENVS_DIR" && rm -f "$VOL/$Z" || die "fetching $m map failed"
  chmod +x "$L" "$(dirname "$L")"/*/Binaries/Linux/* 2>/dev/null || true
  chmod -R a+rwX "$(dirname "$(dirname "$L")")" 2>/dev/null || true
  ok "$m map installed"
done

say "[5/8] $OPT_LINK symlink (launch_airsim.sh path compatibility)"
if [ -L "$OPT_LINK" ]; then
  ln -sfn "$ENVS_DIR" "$OPT_LINK"; ok "symlink updated: $OPT_LINK → $ENVS_DIR"
elif [ -d "$OPT_LINK" ]; then
  warn "$OPT_LINK already exists as a real directory — skipping symlink (maps installed there directly)"
else
  mkdir -p /opt && ln -s "$ENVS_DIR" "$OPT_LINK"; ok "symlink created: $OPT_LINK → $ENVS_DIR"
fi

say "[6/8] uv + vlmpilot dependencies (uv fetches Python 3.13 by itself)"
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh || die "uv install failed — check access to astral.sh"
  command -v uv >/dev/null 2>&1 || die "uv installed but not on PATH — check $HOME/.local/bin"
fi
[ -d "$VP_DIR" ] || die "$VP_DIR missing — check that the repo was cloned"
# Create the venv on the pod's local disk and make $VP_DIR/.venv a symlink. With the venv on the Network Volume
# (MooseFS), imports were slow and runner startup took about 51 s per episode (2 s on local disk) — the runner's 300 s
# cap starts ticking before startup, so flight time shrinks by that much and results change.
# The symlink path is the same on every pod, so even when several pods share the volume each sees its own local venv.
LOCAL_VENV="${LOCAL_VENV:-/root/vlmpilot-venv}"
if [ -d "$VP_DIR/.venv" ] && [ ! -L "$VP_DIR/.venv" ]; then
  warn "venv found on the volume → moving it to $VP_DIR/.venv.netvol and switching to a local venv (it can be deleted)"
  rm -rf "$VP_DIR/.venv.netvol"; mv "$VP_DIR/.venv" "$VP_DIR/.venv.netvol"
fi
(cd "$VP_DIR" && UV_PROJECT_ENVIRONMENT="$LOCAL_VENV" uv sync) || die "uv sync failed — check pypi access/logs"
ln -sfn "$LOCAL_VENV" "$VP_DIR/.venv"
ok "uv sync done ($LOCAL_VENV, $VP_DIR/.venv → symlink)"

say "[7/8] placing base settings.json ($BASE_SETTINGS)"
mkdir -p "$SETTINGS_DIR"
if [ -f "$BASE_SETTINGS" ]; then
  ok "already present — not overwriting"
else
  # The settings the benchmark runs used: FPV(0) + chase camera, includes the Recording block.
  # The verifier (run_testset) uses this Recording setting via vlmpilot recording.
  # launch_airsim.sh uses this file as the base for per-port settings_<PORT>.json.
  cat > "$BASE_SETTINGS" <<'JSON'
{
    "SettingsVersion": 1.2,
    "SimMode": "Multirotor",
    "ViewMode": "SpringArmChase",
    "Recording": {
        "RecordOnMove": false,
        "RecordInterval": 0.05,
        "Cameras": [
            { "CameraName": "0", "VehicleName": "SimpleFlight", "ImageType": 0, "PixelsAsFloat": false, "Compress": true },
            { "CameraName": "chase", "VehicleName": "SimpleFlight", "ImageType": 0, "PixelsAsFloat": false, "Compress": true }
        ]
    },
    "Vehicles": {
        "SimpleFlight": {
            "VehicleType": "SimpleFlight",
            "AutoCreate": true,
            "Cameras": {
                "0": {
                    "CaptureSettings": [
                        { "ImageType": 0, "Width": 1280, "Height": 720 }
                    ],
                    "X": 0.50, "Y": 0.0, "Z": 0.10,
                    "Pitch": 0.0, "Roll": 0.0, "Yaw": 0.0
                },
                "chase": {
                    "CaptureSettings": [
                        { "ImageType": 0, "Width": 1280, "Height": 720 }
                    ],
                    "X": -7.0, "Y": 0.0, "Z": -2.5,
                    "Pitch": -12.0, "Roll": 0.0, "Yaw": 0.0,
                    "Gimbal": {
                        "Stabilization": 1.0,
                        "Pitch": -12.0, "Roll": 0.0
                    }
                }
            }
        }
    }
}
JSON
  ok "settings.json generated"
fi
chown -R airsim:airsim /home/airsim/Documents

say "[8/8] API keys + smoke test"
# Do not keep keys on the volume — /workspace is a Network Volume and persists after the pod stops.
# Pass them as pod env vars via RunPod Secrets ({{ RUNPOD_SECRET_<name> }}), or export them in your shell (gone when the pod stops).
if [ -f "$VP_DIR/.env" ]; then
  case "$(df -P "$VP_DIR" | awk 'NR==2{print $1}')" in
    *mfs*|*fuse*) warn "$VP_DIR/.env is on the Network Volume — keys survive even after the pod is stopped. Delete it and use env vars: rm $VP_DIR/.env" ;;
    *) ok ".env present" ;;
  esac
fi
have=""; for k in OPENROUTER_API_KEY GEMINI_API_KEY OPENAI_DIRECT_API_KEY; do
  if [ -n "${!k:-}" ] || tr '\0' '\n' < /proc/1/environ 2>/dev/null | grep -q "^$k=."; then have="$have $k"; fi
done
if [ -n "$have" ]; then ok "env var keys:$have"; else
  warn "no API key env vars. Pass RunPod Secrets as pod env vars or export them in the shell:"
  echo "   OPENROUTER_API_KEY   # models via OpenRouter (gpt6-luna-or, gemini38, claude, qwen)"
  echo "   GEMINI_API_KEY       # Google direct (gemini*, the ER of commander+ER)"
  echo "   OPENAI_DIRECT_API_KEY  # OpenAI direct (gpt, gpt6-luna)"
fi

if [ "${SKIP_SMOKE:-0}" = 1 ]; then
  warn "SKIP_SMOKE=1 — skipping smoke test"
else
  say "smoke test: launch_airsim.sh nh $SMOKE_PORT (exits after confirming RPC LISTEN)"
  if bash "$REPO/scripts/airsim/launch_airsim.sh" nh "$SMOKE_PORT" 0; then
    kill_sim "settings_${SMOKE_PORT}.json"
    ok "smoke test passed — sim shut down"
  else
    kill_sim "settings_${SMOKE_PORT}.json"
    die "smoke test failed — check /tmp/airsim_nh_p${SMOKE_PORT}.log. If it is a Vulkan problem, re-verify with --check"
  fi
fi

echo
ok "setup complete. next steps (from the repo root $REPO):"
echo "   1) API keys: RunPod Secrets, or read -s OPENROUTER_API_KEY && export OPENROUTER_API_KEY (never a file on the volume)"
echo "   2) MODEL=gpt6-luna-or PARTS=single bash scripts/runpod/airsim/run_all.sh     # see scripts/runpod/airsim/README.md"
