#!/usr/bin/env bash
# DroneCATS
# Copyright (c) 2026-present NAVER Cloud Corp.
# Apache-2.0

# One command from nothing to the verified dataset:
#   bash datagen/scripts/run_all.sh --out /data/ddr     # from the repository root
#   bash scripts/run_all.sh --out /data/ddr             # from datagen/ (paths are resolved from this file)
# Steps (the same as README steps 1-5): create the pinned environment, download the official
# image archives, download the annotation package from Hugging Face, then fetch-check, reconstruct, verify.
#
# Layout under --out (each folder can be moved with its own option):
#   downloads/     official archives, about 72 GB          (--downloads DIR, reused if already complete)
#   PACKAGE/       annotation package from Hugging Face
#   dataset/       the result: 6 row files + images/*.zip, about 9 GB
#   dataset.work/  temporary files, about 8 GB, removed after a PASS (--keep-work keeps it)
#   .venv/         pinned Python environment               (--venv DIR)
#
# Resumable: re-running skips complete downloads and continues partial ones. Needs Python 3.10
# (python3.10 on PATH, or PYTHON=/path/to/python3.10) and wget. Byte-identical output is guaranteed
# only with the versions pinned in requirements-lock.txt, which is why this script builds its own venv.
set -euo pipefail

HF_ID=naver-ai/DroneCATS-Pilot-54K
SUMS_SHA=2fe1286e6a626b1fe582466f36281ab56c217a508ab1ab6147ee2ef21147e144   # same as README step 2
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(dirname "$HERE")                                                    # the datagen/ folder

OUT=""; DOWNLOADS=""; VENV=""; JOBS=""; KEEP_WORK=0
usage() { sed -n '2,13p' "$0" | sed 's/^# \{0,1\}//'; echo "options: --out DIR (required) [--downloads DIR] [--venv DIR] [--jobs N] [--keep-work]"; }
while [ $# -gt 0 ]; do
  case "$1" in
    --out) OUT=$2; shift 2 ;;
    --downloads) DOWNLOADS=$2; shift 2 ;;
    --venv) VENV=$2; shift 2 ;;
    --jobs) JOBS=$2; shift 2 ;;
    --keep-work) KEEP_WORK=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done
[ -n "$OUT" ] || { echo "--out is required" >&2; usage >&2; exit 2; }
# Absolute paths, without creating anything yet: a failed Python check leaves no folders behind.
abs() { case "$1" in /*) echo "$1" ;; *) echo "$PWD/$1" ;; esac; }
OUT=$(abs "$OUT"); DOWNLOADS=$(abs "${DOWNLOADS:-$OUT/downloads}"); VENV=$(abs "${VENV:-$OUT/.venv}")

echo "== 0. environment ($VENV)"
if [ ! -x "$VENV/bin/ddr" ]; then
  PY=${PYTHON:-$(command -v python3.10 || true)}
  [ -n "$PY" ] || { echo "Python 3.10 not found: install it or set PYTHON=/path/to/python3.10" >&2; exit 1; }
  "$PY" -c 'import sys; sys.exit(sys.version_info[:2] != (3, 10))' \
    || { echo "$PY is not Python 3.10 (the pinned versions were verified on 3.10)" >&2; exit 1; }
  "$PY" -m venv "$VENV"
  "$VENV/bin/python" -m pip install -q -r "$ROOT/requirements-lock.txt"
  "$VENV/bin/python" -m pip install -q "$ROOT" --no-deps
  "$VENV/bin/python" -m pip install -q huggingface_hub
else
  echo "reusing $VENV"
fi
export PATH="$VENV/bin:$PATH"

echo "== 1. image archives -> $DOWNLOADS"
bash "$HERE/download.sh" "$DOWNLOADS"

echo "== 2. annotation package -> $OUT/PACKAGE"
python - "$HF_ID" "$OUT/PACKAGE" <<'PY'
import sys
from huggingface_hub import snapshot_download
snapshot_download(sys.argv[1], repo_type="dataset", local_dir=sys.argv[2])
PY
got=$(sha256sum "$OUT/PACKAGE/reference/SHA256SUMS" | awk '{print $1}')
[ "$got" = "$SUMS_SHA" ] || { echo "SHA256SUMS digest $got != $SUMS_SHA" >&2; exit 1; }
echo "ok    reference/SHA256SUMS $got"

ARCHIVES=(--nyu "$DOWNLOADS/nyu_depth_v2_labeled.mat" --sunrgbd "$DOWNLOADS/SUNRGBD.zip" --tartanair-root "$DOWNLOADS/tartanair2")

echo "== 3. fetch-check"
ddr fetch-check --package "$OUT/PACKAGE" "${ARCHIVES[@]}"

echo "== 4. reconstruct -> $OUT/dataset"
ddr reconstruct --package "$OUT/PACKAGE" "${ARCHIVES[@]}" --out "$OUT/dataset" --work "$OUT/dataset.work" \
  ${JOBS:+--jobs "$JOBS"}

echo "== 5. verify"
ddr verify --package "$OUT/PACKAGE" --dataset "$OUT/dataset" --report "$OUT/verify_report.json"

[ "$KEEP_WORK" = 1 ] || rm -rf "$OUT/dataset.work"
echo "PASS. Dataset: $OUT/dataset"
