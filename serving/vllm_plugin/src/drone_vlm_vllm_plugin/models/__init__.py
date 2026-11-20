# DroneCATS
# Copyright (c) 2026-present NAVER Cloud Corp.
# Apache-2.0

"""Vendored model modules (imported lazily by vLLM via ``ModelRegistry`` strings).

cosmos3_edge.py, qwen3_vl.py, nemotron_h.py: vLLM 0.25.1 code with the Cosmos3-Edge changes;
  only relative imports of stock modules were made absolute, and the plugin-local ones
  (qwen3_vl, nemotron_h) stay relative so cosmos3_edge never touches the stock copies.
"""
