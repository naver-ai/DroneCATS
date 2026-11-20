#!/usr/bin/env bash
# DroneCATS
# Copyright (c) 2026-present NAVER Cloud Corp.
# Apache-2.0

# Install the vLLM server environment for DroneCATS-Cosmos3-Edge-2B and download the model.
# Run once (about 15 minutes; 8 on an RTX 4090). Needs Python 3.12 (tested with drivers 535 and 595).
#   VENV      virtualenv location (default ~/dronevllm). Keep it on local disk, not a network volume.
#   HF_HOME   Hugging Face cache (default ~/hf)
#   MODEL_ID  model repository (default naver-ai/DroneCATS-Cosmos3-Edge-2B)
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
VENV=${VENV:-$HOME/dronevllm}
export HF_HOME=${HF_HOME:-$HOME/hf}
MODEL_ID=${MODEL_ID:-naver-ai/DroneCATS-Cosmos3-Edge-2B}

[ -x "$VENV/bin/python" ] || ${PYTHON:-python3.12} -m venv "$VENV"
"$VENV/bin/pip" install -q -U pip
"$VENV/bin/pip" install -q -r "$HERE/vllm_plugin/requirements_server.txt" \
  --extra-index-url https://wheels.vllm.ai/0.25.1/cu129/ \
  --extra-index-url https://download.pytorch.org/whl/cu129
# --no-deps: the pinned requirements above already hold vllm and transformers.
"$VENV/bin/pip" install -q --no-deps "$HERE/vllm_plugin"
"$VENV/bin/python" -m drone_vlm_vllm_plugin.selfcheck | tail -2
"$VENV/bin/python" -c "from huggingface_hub import snapshot_download as s; print(s('$MODEL_ID'))"
