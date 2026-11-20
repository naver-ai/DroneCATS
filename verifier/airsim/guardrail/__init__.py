# DroneCATS
# Copyright (c) 2026-present NAVER Cloud Corp.
# Apache-2.0

"""Measurement guardrail — the runner asks itself "can this episode go into the results?" and acts on the answer.

Split into pieces. The runner (run_testset) hooks only a few calls into this package's session object.

  integrity   verdict logic. step_events + a few measurements → {valid, reasons, metrics}. Pure function.
  api_failures  per-arm (head/body) call·failure counts. one episode's stdout → 5 numbers. Pure function.
  timing      wiring that attaches the executor's [TIMING] lines to step_events.
  session     **actions** based on the verdict — quarantine·reflight of invalid episodes, early stop on dead server·model stop,
              rollback of invalid episodes on RESUME, slow_vlm baseline. The scope of actions depends on the mode.

Modes (environment variable GUARDRAIL, default off):
  off      no verdict. The verdict file has no integrity and all episodes are aggregated. — release default.
  audit    judges and records in the verdict·report but takes no action (no reflight·exclusion·early stop).
           Used to diagnose someone else's run, or when redrawing thresholds.
  enforce  verdict + action. Contaminated episodes are moved to _invalid/ and the same task is reflown (3 times total),
           a dead server cuts the episode, and 2 in a row stop that model. — use this for team measurements.

Override thresholds with VLMPILOT_INTEG_<NAME>, the reflight budget with VLMPILOT_INTEG_MAX_FLIGHTS,
and the consecutive-failure count for early stop with VLMPILOT_INTEG_API_DEAD_STREAK (see integrity.DEFAULTS).
"""

from .api_failures import count_api_failures, count_api_failures_file
from .integrity import DEFAULTS, assess, thresholds_from_env
from .session import (MODES, GuardrailSession, NoGuardrail, is_excluded, make_session,
                      mode_from_env, summarize_rows)
from .timing import TIMING_RE, attach_timing

__all__ = [
    "DEFAULTS", "assess", "thresholds_from_env",
    "count_api_failures", "count_api_failures_file",
    "TIMING_RE", "attach_timing",
    "MODES", "GuardrailSession", "NoGuardrail", "make_session", "mode_from_env",
    "is_excluded", "summarize_rows",
]
