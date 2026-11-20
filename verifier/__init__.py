# DroneCATS
# Copyright (c) 2026-present NAVER Cloud Corp.
# Apache-2.0

"""Episode runners and judges, one subpackage per simulator.

- `verifier.airsim`: `run_testset` (single drone), `run_bench_tasks` (multi-drone), judges and guardrails
- `verifier/isaacsim/`: `batch_run.py` (episode runner), `judge_run.py`, `join_gt.py`, `run_scripted_episode.py`
"""
