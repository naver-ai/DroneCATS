# DroneCATS
# Copyright (c) 2026-present NAVER Cloud Corp.
# Apache-2.0

"""DroneCATS verifier — scores success on top of real executor flights.

Success = at the moment of the arrival declaration (FINISHED), 3D distance ≤ δ  AND  the target is visible on screen.
Distance and visibility are all measured from sim GT (model outputs are never used for scoring).

Layout (see README.md for how to run):
  run_testset  real-flight batch runner (entry point)
  tasks            judging for the 4 tasks (approaching/searching/tracking/search_tracking)
  trace            2Hz trajectory + post-hoc δ computation + success judgment
  judge            judgment at the QF moment (distance GT + engine-detection visibility)
  mover            moving target (tracking)
"""

from .judge import Judgment, judge_qf, measure_qf
from .trace import DELTA_M, is_success, sr_at
from .tasks import judge_task, TASKS

__all__ = [
    "Judgment", "judge_qf", "measure_qf",
    "DELTA_M", "is_success", "sr_at",
    "judge_task", "TASKS",
]
