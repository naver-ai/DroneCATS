#!/usr/bin/env bash
# DroneCATS
# Copyright (c) 2026-present NAVER Cloud Corp.
# Apache-2.0

# Download the official image archives into DEST (default: current directory).
# Resumable: re-running continues partial files and skips complete ones (wget -c).
# Sizes are checked here; SHA-256 is checked by `ddr fetch-check` (step 3 of the README).
# The URLs, sizes and SHA-256 are the same as PACKAGE/ledger/archives.json.
set -euo pipefail
DEST=${1:-.}
command -v wget >/dev/null || { echo "wget is required" >&2; exit 1; }
mkdir -p "$DEST"
cd "$DEST"

TA=https://huggingface.co/datasets/theairlabcmu/tartanair2/resolve/main
FILES=(
  "nyu_depth_v2_labeled.mat 2972037809 https://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat"
  "SUNRGBD.zip 6885481608 https://rgbd.cs.princeton.edu/data/SUNRGBD.zip"
  "tartanair2/Downtown/Data_easy/image_lcam_front.zip 15146118453 $TA/Downtown/Data_easy/image_lcam_front.zip"
  "tartanair2/Downtown/Data_hard/image_lcam_front.zip 8056487474 $TA/Downtown/Data_hard/image_lcam_front.zip"
  "tartanair2/ModularNeighborhoodIntExt/Data_easy/image_lcam_front.zip 9454945875 $TA/ModularNeighborhoodIntExt/Data_easy/image_lcam_front.zip"
  "tartanair2/ModularNeighborhoodIntExt/Data_hard/image_lcam_front.zip 5686948292 $TA/ModularNeighborhoodIntExt/Data_hard/image_lcam_front.zip"
  "tartanair2/NordicHarbor/Data_easy/image_lcam_front.zip 15119876960 $TA/NordicHarbor/Data_easy/image_lcam_front.zip"
  "tartanair2/NordicHarbor/Data_hard/image_lcam_front.zip 8977495994 $TA/NordicHarbor/Data_hard/image_lcam_front.zip"
)

failed=0
for entry in "${FILES[@]}"; do
  read -r path bytes url <<<"$entry"
  if [ -f "$path" ] && [ "$(stat -L -c %s "$path")" = "$bytes" ]; then
    echo "ok    $path (already complete)"
    continue
  fi
  mkdir -p "$(dirname "$path")"
  echo "get   $path"
  wget -c -q --show-progress -O "$path" "$url" || true
  got=$(stat -L -c %s "$path" 2>/dev/null || echo 0)
  if [ "$got" = "$bytes" ]; then
    echo "ok    $path"
  else
    echo "FAIL  $path: $got of $bytes bytes (re-run to resume)" >&2
    failed=1
  fi
done

if [ "$failed" = 0 ]; then
  echo "All archives downloaded. Next: ddr fetch-check (README step 3)."
fi
exit "$failed"
