#!/usr/bin/env bash
# DroneCATS
# Copyright (c) 2026-present NAVER Cloud Corp.
# Apache-2.0

# Serve DroneCATS-Cosmos3-Edge-2B with vLLM (OpenAI-compatible API on http://127.0.0.1:18000/v1).
#   GPU_UTIL   --gpu-memory-utilization (default 0.45, which leaves room for a simulator on the same GPU).
#              Too low a value fails to start ("estimated maximum model length is ..."), because the
#              KV cache must hold the 16,384-token context.
#   PORT, HOST bind address (default 18000, 127.0.0.1)
#   MODEL      local model directory (default: the cached snapshot of MODEL_ID)
#   MODEL_ID   model repository (default naver-ai/DroneCATS-Cosmos3-Edge-2B)
#   VENV, HF_HOME as in setup.sh
set -euo pipefail
VENV=${VENV:-$HOME/dronevllm}
export HF_HOME=${HF_HOME:-$HOME/hf}
MODEL_ID=${MODEL_ID:-naver-ai/DroneCATS-Cosmos3-Edge-2B}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export COSMOS3_PATCH_ORDER_FIX=1 VLLM_VIDEO_LOADER_BACKEND=opencv VLLM_USE_FLASHINFER_SAMPLER=0
MODEL=${MODEL:-$("$VENV/bin/python" -c "from huggingface_hub import snapshot_download as s; print(s('$MODEL_ID', local_files_only=True))")}
exec "$VENV/bin/python" -m vllm.entrypoints.cli.main serve "$MODEL" --served-model-name drone-vlm-2b-cosmos3edge \
  --tokenizer-mode hf --chat-template-content-format openai --dtype bfloat16 \
  --tensor-parallel-size 1 --data-parallel-size 1 --gpu-memory-utilization "${GPU_UTIL:-0.45}" \
  --max-model-len 16384 --max-num-batched-tokens 16384 --enforce-eager \
  --limit-mm-per-prompt '{"image":8,"video":1}' --host "${HOST:-127.0.0.1}" --port "${PORT:-18000}"
