#!/usr/bin/env bash
# DroneCATS
# Copyright (c) 2026-present NAVER Cloud Corp.
# Apache-2.0

# Sets up the datagen reproduction environment: **downloads and places assets, then verifies them**.
#
# Why a script: when the procedure only lives in docs, it ends in "downloaded it, why doesn't
# it work". The failures that cost this pipeline the most time were assets:
#   (1) vMaterials not merged, so Brownstone's concrete textures did not resolve
#   (2) only 1 of 12 pedestrian clips downloaded, so characters stood in T-pose **without errors**
# Both fail silently, so this script's job includes **verification** after download.
#
# ⚠️ The assets are NVIDIA distributions and are not redistributed here. This script
#    **fetches them from the official sources.** Each pack's license terms apply.
#
#   bash scripts/isaacsim/setup_datagen.sh            # required only (Brownstone + vMaterials)
#   bash scripts/isaacsim/setup_datagen.sh --all      # + pedestrians + indoor (Warehouse) + skies
#
# About indoor environments: Pegasus registry scenes such as `--env Hospital` load
# **remotely** (`nucleus.get_assets_root_path()` -> S3), so they need not be downloaded
# here to be used. What is downloaded here is a local **mirror for reproducibility**.
# Default is Simple_Warehouse only; pass names to fetch more:
#   OV_ENVS="Simple_Warehouse Hospital Office" bash scripts/isaacsim/setup_datagen.sh --all
# (Hospital about 1.0 GB, Office similar)
#   bash scripts/isaacsim/setup_datagen.sh --check    # no downloads, only show the current state
set -euo pipefail

ASSETS="${OV_ASSETS:-$HOME/omniverse-assets}"
ISAACSIM_DIR="${ISAACSIM_DIR:-$HOME/isaacsim}"
CDN="https://d4i3qtqj3r0z5.cloudfront.net"
S3="https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac"

WANT_ALL=0; CHECK_ONLY=0
for a in "$@"; do
  case "$a" in
    --all) WANT_ALL=1 ;;
    --check) CHECK_ONLY=1 ;;
    -h|--help) sed -n '2,22p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown argument: $a"; exit 2 ;;
  esac
done

say()  { printf '\n\033[1m== %s\033[0m\n' "$*"; }
ok()   { printf '  \033[32m✓\033[0m %s\n' "$*"; }
warn() { printf '  \033[33m!\033[0m %s\n' "$*"; }
die()  { printf '  \033[31m✗\033[0m %s\n' "$*"; exit 1; }

# -- 0. Prerequisites ------------------------------------------------------
say "Checking prerequisites"
[ -x "$ISAACSIM_DIR/python.sh" ] || die "Isaac Sim not found: $ISAACSIM_DIR/python.sh (set ISAACSIM_DIR)"
ok "Isaac Sim $(cat "$ISAACSIM_DIR/VERSION" 2>/dev/null | cut -d' ' -f1 || echo '(no VERSION)')"
command -v curl >/dev/null || die "curl is required"
command -v unzip >/dev/null || die "unzip is required"
python3 -c 'import sys; sys.exit(0)' || die "python3 is required"
df -BG --output=avail "$HOME" | tail -1 | tr -d ' G' | awk '{if ($1 < 40) exit 1}' \
  || warn "less than 40 GB free - the required packs alone need ~8 GB to unpack"

# -- 1. Download -----------------------------------------------------------
fetch_zip() {  # name file expected_size (bytes, 0 skips the check)
  local name="$1" file="$2" want="$3"
  local dst="$ASSETS/_zips/$file"
  mkdir -p "$ASSETS/_zips"
  if [ -s "$dst" ]; then ok "$name - already downloaded ($(du -h "$dst" | cut -f1))"; return; fi
  [ "$CHECK_ONLY" = 1 ] && { warn "$name - missing (run without --check to download)"; return; }
  echo "  ↓ $name"
  # `-C -` resumes; at 2~8 GB, interrupted downloads are common
  curl -fL -C - --retry 3 --retry-delay 5 -o "$dst" "$CDN/$file"
  [ "$want" != 0 ] && [ "$(stat -c%s "$dst")" -lt "$want" ] \
    && die "$name is smaller than expected - download it again"
  ok "$name done"
}

if [ "$CHECK_ONLY" = 0 ]; then
  say "Downloading NVIDIA packs (official CDN)"
  fetch_zip "Brownstone (AEC Demo)" "AECDemo_NVD%4010012.zip" 2000000000
  fetch_zip "vMaterials 2.4"        "vMaterials_2_4_0_NVD%40020240.zip" 2000000000
  if [ "$WANT_ALL" = 1 ]; then
    fetch_zip "Skies (Environments)" "Environments_NVD%4010012.zip" 8000000000
    fetch_zip "City Demo"            "AECO_CityDemoPack_NVD%4010011.zip" 100000000
  fi

  say "Unpacking"
  cd "$ASSETS"
  for z in _zips/*.zip; do
    [ -e "$z" ] || continue
    d="$(basename "$z" .zip)"
    d="${d%%_NVD*}"; d="${d/AECDemo/Brownstone}"; d="${d/Environments/Skies}"
    d="${d/AECO_CityDemoPack/CityDemo}"; d="${d/vMaterials_2_4_0/vMaterials}"
    if [ -d "$d" ] && [ -n "$(ls -A "$d" 2>/dev/null)" ]; then ok "$d - already unpacked"; continue; fi
    echo "  ⇲ $z → $d"
    unzip -q -n "$z" -d "$d"
  done

  # **vMaterials is a dependency of Brownstone, not optional.** The pack contains only a subset
  # (139 textures, 39 MDL), so opened as-is the Concrete materials do not resolve.
  say "Merging vMaterials into Brownstone"
  PACK="$ASSETS/Brownstone/Demos/AEC/BrownstoneDemo"
  if [ -d "$PACK" ]; then
    before=$(find "$PACK/Materials" -type f 2>/dev/null | wc -l)
    unzip -n -q "$ASSETS/_zips/vMaterials_2_4_0_NVD%40020240.zip" 'Materials/*' -d "$PACK" || true
    after=$(find "$PACK/Materials" -type f 2>/dev/null | wc -l)
    ok "Materials $before → $after files"
    [ "$after" -lt 5000 ] && warn "fewer than expected (≈6,273) - the merge may be incomplete"
  else
    warn "Brownstone pack not found: $PACK"
  fi

  if [ "$WANT_ALL" = 1 ]; then
    say "Pedestrians · indoor (Isaac asset server)"
    # ⚠️ Python's urllib fails here with CERTIFICATE_VERIFY_FAILED (observed), so curl is used.
    GRPS="People/Animations:IsaacPeople/Animations
People/Characters:IsaacPeople/Characters"
    for e in ${OV_ENVS:-Simple_Warehouse}; do
      GRPS="$GRPS
Environments/$e:IsaacEnvs/$e"
    done
    for grp in $GRPS; do
      src="${grp%%:*}"; dst="$ASSETS/${grp##*:}"
      mkdir -p "$dst"
      echo "  ↓ $src"
      # S3 listings come in pages of 1,000; follow the continuation token
      tok=""; got=0
      while :; do
        url="$S3/$src/?list-type=2&max-keys=1000"
        [ -n "$tok" ] && url="$url&continuation-token=$tok"
        xml="$(curl -sS "$url")"
        keys="$(printf '%s' "$xml" | grep -o '<Key>[^<]*' | sed 's|<Key>||')"
        for k in $keys; do
          case "$k" in */|*.thumbs*|*DH_*) continue ;; esac   # DH_* is 9.1 GB, skipped
          rel="${k#*$src/}"; out="$dst/$rel"
          [ -s "$out" ] && continue
          mkdir -p "$(dirname "$out")"
          curl -sS -o "$out" "https://omniverse-content-production.s3-us-west-2.amazonaws.com/$k"
          got=$((got+1))
        done
        tok="$(printf '%s' "$xml" | grep -o '<NextContinuationToken>[^<]*' | sed 's|<NextContinuationToken>||')"
        [ -z "$tok" ] && break
        tok="$(python3 -c 'import urllib.parse,sys;print(urllib.parse.quote(sys.argv[1],safe=""))' "$tok")"
      done
      ok "$src - $got new file(s) downloaded"
    done
  fi
fi

# -- 2. Verify -------------------------------------------------------------
say "Verifying"
need=(
  "$ASSETS/Brownstone/Demos/AEC/BrownstoneDemo/World_BrownstoneDemopack_Morning(20Gb).usd"
)
opt=(
  "$ASSETS/IsaacPeople/Characters/Biped_Setup.usd"
  "$ASSETS/IsaacPeople/Characters/female_adult_police_01_new/female_adult_police_01_new.usd"
)
miss=0
for f in "${need[@]}"; do
  [ -f "$f" ] && ok "$(basename "$f")" || { warn "missing: $f"; miss=1; }
done
for f in "${opt[@]}"; do
  [ -f "$f" ] && ok "$(basename "$f") (optional)" || warn "missing (optional): $(basename "$f")"
done

# **Always check the pedestrian clip count.** If even one of the 12 is missing, motion matching
# has no pose to produce and characters stand in T-pose **without errors**; the longest-hunted failure here.
CLIPS="$ASSETS/IsaacPeople/Animations"
if [ -d "$CLIPS" ]; then
  n=$(find "$CLIPS" -name '*.usd' | wc -l)
  [ "$n" -ge 12 ] && ok "$n walk clip(s)" \
    || warn "only $n walk clip(s) - with fewer than 12 characters stand in T-pose (download with --all)"
fi

if [ "$miss" = 1 ]; then
  printf '\n\033[33mRequired assets are missing.\033[0m Re-run with --all or see scripts/runpod/isaacsim/README.md.\n'
  exit 1
fi
printf '\n\033[32mReady.\033[0m Smoke test:\n'
cat <<'SMOKE'
  # A (Isaac): load the scene and place the target
  $ISAACSIM_DIR/python.sh scripts/isaacsim/sim/launch_brownstone.py --variant Morning \
      --spawn-xy=-22,-36 --targets 'box:lime' --target-front 8 \
      --gt --gt-dir /tmp/smoke_gt --gt-targets 'DG_target' --vehicles 1 --render-hz 10
  # Expect: a [DG] target line for the lime cargo box @ (-14.0, -36.0, ...) and the [SIM] ready line
SMOKE
