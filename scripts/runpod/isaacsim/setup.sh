#!/bin/bash
# DroneCATS
# Copyright (c) 2026-present NAVER Cloud Corp.
# Apache-2.0

# Install everything the Isaac Sim benchmark needs on one RunPod GPU pod (~10 min on an RTX 4090):
#   Isaac Sim 5.1.0 · PX4 v1.14.3 SITL · Pegasus Simulator v5.1.0 · vlmpilot (pegasus extra) · scene assets.
#
#   ACCEPT_EULA=Y bash scripts/runpod/isaacsim/setup.sh        # from the repo root
#
# - Isaac Sim requires accepting the NVIDIA Isaac Sim / Omniverse EULA. Read it first, then set
#   ACCEPT_EULA=Y (and OMNI_KIT_ACCEPT_EULA=YES) yourself; this script refuses to run otherwise.
# - Heavy installs go to the container disk (/root). Only results go to /workspace.
# - Per-step logs: /root/setup_logs/<step>.log; finished steps are skipped on a re-run.
# - NVIDIA assets are downloaded from NVIDIA's public CDN/S3; nothing is redistributed by this repo.
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
L=/root/setup_logs; mkdir -p $L
ASSETS=/root/omniverse-assets
# ISAAC_VERSION=6.0.1 selects the Isaac Sim 6 line (experimental): its standalone zip lives on a different host,
# and Pegasus has no 6.x release yet, so a port of it is used (PEGASUS_REPO / PEGASUS_REF).
ISAAC_VERSION=${ISAAC_VERSION:-5.1.0}
case $ISAAC_VERSION in
  5.*) ISAAC_URL=https://download.isaacsim.omniverse.nvidia.com/isaac-sim-standalone-$ISAAC_VERSION-linux-x86_64.zip
       PEGASUS_REPO=${PEGASUS_REPO:-https://github.com/PegasusSimulator/PegasusSimulator.git}; PEGASUS_REF=${PEGASUS_REF:-v5.1.0} ;;
  *)   ISAAC_URL=https://downloads.isaacsim.nvidia.com/isaac-sim-standalone-$ISAAC_VERSION-linux-x86_64.zip
       PEGASUS_REPO=${PEGASUS_REPO:-https://github.com/iftahnaf/PegasusSimulator.git}; PEGASUS_REF=${PEGASUS_REF:-fcb99c0} ;;
esac
VENV=/root/venv-pegasus
# An SSH shell does not inherit the pod's environment variables; read them from PID 1 if set there.
if [ -r /proc/1/environ ]; then
  for k in ACCEPT_EULA OMNI_KIT_ACCEPT_EULA; do
    [ -n "${!k:-}" ] || export "$k=$(tr '\0' '\n' </proc/1/environ 2>/dev/null | sed -n "s/^$k=//p")"
  done
fi
if [ "${ACCEPT_EULA:-}" != "Y" ] && [ "${OMNI_KIT_ACCEPT_EULA:-}" != "YES" ]; then
  echo "Isaac Sim needs the NVIDIA EULA accepted. Read https://docs.isaacsim.omniverse.nvidia.com/latest/common/NVIDIA_Omniverse_License_Agreement.html"
  echo "then run again with ACCEPT_EULA=Y (or set it in the pod's environment variables)."; exit 2
fi
export OMNI_KIT_ACCEPT_EULA=YES
step() { local n=$1; shift; [ -f $L/$n.done ] && { echo "skip $n"; return 0; }
  echo "[$(date +%T)] start $n"
  if ( set -e; "$@" ) > $L/$n.log 2>&1; then touch $L/$n.done; echo "[$(date +%T)] done $n"
  else echo "[$(date +%T)] FAIL $n — see $L/$n.log"; return 1; fi; }

apt_deps() {
  export DEBIAN_FRONTEND=noninteractive
  apt-get update
  apt-get install -y --no-install-recommends unzip ninja-build build-essential ccache \
    gcc-12 g++-12 libvulkan1 vulkan-tools libglu1-mesa libxt6 libxrandr2 libxinerama1 \
    libxcursor1 libxi6 libsm6 libice6 libegl1 libgl1 python3-dev python3-venv rsync ffmpeg curl
}

isaac() {
  cd /root
  curl -fL -C - --retry 5 -o isaac.zip \
    "$ISAAC_URL"
  mkdir -p /root/isaacsim && unzip -q -o isaac.zip -d /root/isaacsim && rm -f isaac.zip
  cd /root/isaacsim && { ./post_install.sh || true; }
  ./python.sh -c "import sys; print(sys.version)"
}

px4() {
  cd /root
  [ -d PX4-Autopilot ] || git clone --recursive -b v1.14.3 --depth 1 --shallow-submodules \
    https://github.com/PX4/PX4-Autopilot.git
  python3 -m venv /root/px4-venv
  /root/px4-venv/bin/pip install -q empy==3.3.4 kconfiglib jinja2 pyros-genmsg jsonschema \
    future packaging toml numpy pyyaml pyserial
  cd PX4-Autopilot
  # A shallow clone has no tags, and the version-header step fails with IndexError without them.
  git tag -f v1.14.3 HEAD
  (cd platforms/nuttx/NuttX/nuttx && git tag -f nuttx-11.0.0 HEAD)
  # gcc-13 (Ubuntu 24.04 default) cannot build v1.14.3 (missing <cstdint> in pxh.cpp); use gcc-12.
  # `make px4_sitl_default none` would also start px4 after building, so build the target only.
  CC=gcc-12 CXX=g++-12 PATH=/root/px4-venv/bin:$PATH make px4_sitl_default
  ls -la build/px4_sitl_default/bin/px4
}

pegasus6_patches() {
  # Isaac Sim 6 port only: apply the body drag, rolling moment and 4 rotor thrusts in one tensor call
  # per physics step instead of 6 (same forces, same frames). Off by default; PEGASUS_BATCHED_FORCES=1
  # applies it. The faster sim raises RTF, and moving-target outcomes change with RTF (see README).
  local P="$REPO/scripts/runpod/isaacsim/patches/pegasus6_multirotor_batched_forces.patch"
  if git -C PegasusSimulator apply --reverse --check "$P" 2>/dev/null; then
    if [ "${PEGASUS_BATCHED_FORCES:-0}" = 0 ]; then git -C PegasusSimulator apply --reverse "$P"; fi
  elif [ "${PEGASUS_BATCHED_FORCES:-0}" != 0 ]; then
    git -C PegasusSimulator apply "$P"
  fi
}

pegasus() {
  cd /root
  [ -d PegasusSimulator ] || git clone "$PEGASUS_REPO" PegasusSimulator
  git -C PegasusSimulator checkout -q "$PEGASUS_REF"
  case $ISAAC_VERSION in 5.*) ;; *) pegasus6_patches ;; esac
  cd PegasusSimulator/extensions
  # Pegasus' build step stops with RuntimeError when ISAACSIM_PATH is unset.
  ISAACSIM_PATH=/root/isaacsim /root/isaacsim/python.sh -m pip install --editable pegasus.simulator
  grep -n "px4_dir" pegasus.simulator/config/configs.yaml     # ~/PX4-Autopilot = /root/PX4-Autopilot
}

vlmpilot() {
  # venv on the container disk: imports from a network volume are slow.
  cd "$REPO/vlmpilot"
  UV_PROJECT_ENVIRONMENT=$VENV uv sync --extra pegasus
  $VENV/bin/python -c "import mavsdk, vlmpilot"
}

assets() {
  # Brownstone demo pack + vMaterials (+ CityDemo, Skies) from NVIDIA's CDN.
  cd "$REPO" && OV_ASSETS=$ASSETS ISAACSIM_DIR=/root/isaacsim bash scripts/isaacsim/setup_datagen.sh
}

people() {
  # Pedestrian characters and animations from Isaac Sim 5.1's public asset bucket.
  local B=https://omniverse-content-production.s3-us-west-2.amazonaws.com
  local P=Assets/Isaac/5.1/Isaac
  for grp in "People/Animations:IsaacPeople/Animations" "People/Characters:IsaacPeople/Characters"; do
    local src="${grp%%:*}" dst="$ASSETS/${grp##*:}" tok="" got=0 xml k out
    while :; do
      local url="$B/?list-type=2&max-keys=1000&prefix=$P/$src/"
      [ -n "$tok" ] && url="$url&continuation-token=$tok"
      xml="$(curl -sS "$url")"
      for k in $(printf '%s' "$xml" | grep -o '<Key>[^<]*' | sed 's|<Key>||'); do
        case "$k" in */|*.thumbs*|*DH_*) continue ;; esac     # DH_* is 9 GB and unused
        out="$dst/${k#*$src/}"; [ -s "$out" ] && continue
        mkdir -p "$(dirname "$out")"; curl -sS -o "$out" "$B/$k"; got=$((got+1))
      done
      tok="$(printf '%s' "$xml" | grep -o '<NextContinuationToken>[^<]*' | sed 's|<NextContinuationToken>||' || true)"
      [ -z "$tok" ] && break
      tok="$(python3 -c 'import urllib.parse,sys;print(urllib.parse.quote(sys.argv[1],safe=""))' "$tok")"
    done
    echo "$src: $got files"
  done
  local n; n=$(find "$ASSETS/IsaacPeople" -type f | wc -l)
  echo "IsaacPeople: $n files (expected 118)"
  [ "$n" -ge 118 ]
}

vehicles() {
  # CarConcept (CC-BY-4.0, Eric Chadwick / Darmstadt Graphics Group, Khronos glTF-Sample-Assets) → USD.
  local D=$ASSETS/Vehicles/CarConcept
  local K=https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/main/Models/CarConcept
  mkdir -p $D
  [ -s $D/CarConcept.glb ] || curl -fL --retry 3 -o $D/CarConcept.glb $K/glTF-Binary/CarConcept.glb
  [ -s $D/metadata.json ] || curl -fL --retry 3 -o $D/metadata.json $K/metadata.json
  [ -s $D/CarConcept.usd ] || /root/isaacsim/python.sh "$REPO/scripts/runpod/isaacsim/convert_vehicle.py" $D/CarConcept.glb $D
  ls -la $D/CarConcept.usd
}

hello() {
  # Driver check: on driver 595.x hosts Isaac Sim 5.1 segfaults right here (pick CUDA 13.0/13.1 hosts).
  local t; t=$(mktemp --suffix=.py)
  cat > "$t" <<'PY'
from isaacsim import SimulationApp
app = SimulationApp({"headless": True})
for _ in range(30):
    app.update()
print("HELLO_OK", flush=True)
app.close()
PY
  VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json /root/isaacsim/python.sh "$t" | grep HELLO_OK
}

nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH=$HOME/.local/bin:$PATH; }
step apt_deps apt_deps || exit 1
step vlmpilot vlmpilot &
step px4 px4 &
step isaac isaac || exit 1
wait
step hello hello || exit 1
step pegasus pegasus || exit 1
step assets assets || exit 1
step people people || exit 1
step vehicles vehicles || echo "WARNING: no car asset — car-target episodes will fail to load"
for s in apt_deps vlmpilot px4 isaac hello pegasus assets people; do
  [ -f $L/$s.done ] || { echo "SETUP_FAILED ($s)"; exit 1; }
done
echo "SETUP_OK"
