# DroneCATS
# Copyright (c) 2026-present NAVER Cloud Corp.
# Apache-2.0

"""Guardrail session — gathers the **actions** taken on the verdict (integrity) in one place.

Only hooks remain in the runner's episode loop:

  session = make_session(model, mdir)                # one per model (decided by GUARDRAIL mode)
  ...
  if session.resume_reflight(tag, vpath, v): ...     # RESUME: if the episode is invalid, roll it back and fly again
  guard = session.begin_episode(tag)                 # instrument for one episode
  for line in proc.stdout:
      if guard.feed(line, live): _interrupt_pilot()  # dead server → cut the episode
      ...; guard.vlm_ok()                            # the response is alive
  integ = session.assess(guard, step_evs, ...)       # verdict (None if off)
  decision = session.after_episode(tag, vpath, integ, success)   # keep | reflight | final_invalid
  if session.model_should_stop(): ...                # dead-server circuit breaker

Aggregation filters episodes with is_excluded(row) — only episodes invalidated under enforce are dropped; suspect
episodes under audit and episodes under off are all included. Each row records guardrail_mode, so even if models run
in different modes are mixed in one OUT_DIR, each row is filtered by its own rule.

There are exactly two impure parts: the file move of invalid verdicts to _invalid/ and log output.
Both happen only under enforce.
"""

from __future__ import annotations

import os
import shutil
import statistics
from typing import Callable

from .integrity import assess as assess_integrity

MODES = ("off", "audit", "enforce")
_ALIASES = {"0": "off", "false": "off", "no": "off", "none": "off",
            "1": "enforce", "true": "enforce", "yes": "enforce", "on": "enforce"}


def mode_from_env(default: str = "off") -> str:
    """GUARDRAIL environment variable → mode. If empty, default (release default off)."""
    raw = (os.environ.get("GUARDRAIL") or "").strip().lower()
    if not raw:
        return default
    mode = _ALIASES.get(raw, raw)
    if mode not in MODES:
        raise SystemExit(f"GUARDRAIL={raw!r} — valid values: {', '.join(MODES)} (or 0/1)")
    return mode


def is_excluded(row: dict) -> bool:
    """Should this row be excluded from aggregation? Only episodes invalidated under enforce are excluded.

    Rows without guardrail_mode (data predating this package) followed the enforce rule, so they are excluded as before.
    """
    if row.get("integrity_valid") is not False:
        return False
    return (row.get("guardrail_mode") or "enforce") == "enforce"


def summarize_rows(rows: list[dict]) -> tuple[dict, dict]:
    """(episodes excluded from aggregation, episodes only flagged suspect under audit) — both {model: [{target, reasons}]}."""
    excluded: dict[str, list] = {}
    flagged: dict[str, list] = {}
    for r in rows:
        if r.get("integrity_valid") is not False:
            continue
        item = {"target": r.get("target"), "reasons": r.get("integrity_reasons")}
        (excluded if is_excluded(r) else flagged).setdefault(r["model"], []).append(item)
    return excluded, flagged


class EpisodeGuard:
    """Instrument for one episode — counts VLM failures·sim errors and flags a dead server."""

    def __init__(self, tag: str, *, api_dead_streak: int, may_abort: bool):
        self.tag = tag
        self.vlm_fail_total = 0
        self.vlm_fail_streak = 0
        self.sim_err_total = 0
        self.api_dead = False
        self._streak_limit = api_dead_streak
        self._may_abort = may_abort

    def feed(self, line: str, live: bool = True) -> bool:
        """One line of executor stdout. Returning True means the episode should be cut (suspected dead server, only once)."""
        if "[CTRL] step failed" in line:
            self.sim_err_total += 1
        if "[VLM] request failed" in line or "[VLM] empty response" in line:
            self.vlm_fail_total += 1
            self.vlm_fail_streak += 1
            if (self._may_abort and live and not self.api_dead
                    and self.vlm_fail_streak >= self._streak_limit):
                self.api_dead = True
                return True
        return False

    def vlm_ok(self) -> None:
        """The response is alive — reset consecutive failures."""
        self.vlm_fail_streak = 0

    def sim_error(self) -> None:
        """The runner's own RPC failures (pose at capture time etc.) also count as sim errors."""
        self.sim_err_total += 1


class GuardrailSession:
    """Guardrail state for one model — audit / enforce.

    enforce acts on the verdict. audit records the same verdict in the verdict file but takes no action, so
    feed does not cut episodes (hence api_dead never appears — it is caught as vlm_failures) and after_episode
    is always keep.
    """

    def __init__(self, model: str, mdir: str, *, mode: str = "enforce",
                 max_flights: int | None = None, api_dead_streak: int | None = None,
                 log: Callable[[str], None] = print):
        if mode not in ("audit", "enforce"):
            raise ValueError(f"mode={mode!r}: audit or enforce (off is NoGuardrail)")
        self.model = model
        self.mdir = mdir
        self.mode = mode
        self.enforcing = mode == "enforce"
        # Up to max_flights flights total (default 3: original + 2 reflights). If the load still has not cleared by then,
        # it is confirmed final invalid and left for RESUME re-measurement.
        self.max_flights = (int(os.environ.get("VLMPILOT_INTEG_MAX_FLIGHTS", "3"))
                            if max_flights is None else int(max_flights))
        self.api_dead_streak = (int(os.environ.get("VLMPILOT_INTEG_API_DEAD_STREAK", "10"))
                                if api_dead_streak is None else int(api_dead_streak))
        self._log = log
        self.attempts: dict[str, int] = {}     # tag → number of reflights so far
        # slow_vlm baseline — list of vlm medians from this model's normal episodes. Takes effect from 2 episodes.
        self.vlm_meds: list[float] = []
        self.api_dead_flights = 0              # consecutive api_dead flights — model stops at 2

    # -- RESUME ---------------------------------------------------------------

    def resume_reflight(self, tag: str, vpath: str, verdict: dict) -> bool:
        """An existing verdict encountered by RESUME. True means an invalid episode, so roll it back (fly again).

        False means carry it over — if that episode was valid, its slow_vlm baseline is restored too (the baseline
        lives only in process memory, so without re-adding it the first few episodes after RESUME fly with the verdict deferred).
        """
        integ = verdict.get("integrity") or {}
        if self.enforcing and integ.get("valid") is False:
            self._quarantine(vpath, f"verdict_{tag}.json")
            self._log(f"[{self.model:9}] {tag.split('_', 1)[-1]:9} "
                      f"(invalid episode {integ.get('reasons')} — re-flying)")
            return True
        vm = (integ.get("metrics") or {}).get("vlm_median_s")
        if integ.get("valid") and isinstance(vm, (int, float)):
            self.vlm_meds.append(vm)
        return False

    # -- episode ------------------------------------------------------------------

    def begin_episode(self, tag: str) -> EpisodeGuard:
        return EpisodeGuard(tag, api_dead_streak=self.api_dead_streak,
                            may_abort=self.enforcing)

    def assess(self, guard: EpisodeGuard, step_evs: list, *, sim_restarts: int = 0,
               host_sims: int | None = None, moving_target: bool = False,
               sim_dead: bool = False) -> dict:
        """Integrity verdict for one episode. The vlm median of valid episodes is added to the baseline."""
        baseline = (round(statistics.median(self.vlm_meds), 2)
                    if len(self.vlm_meds) >= 2 else None)
        integ = assess_integrity(step_evs, vlm_failed=guard.vlm_fail_total,
                                 aborted_api=guard.api_dead,
                                 sim_restarts=sim_restarts, host_sims=host_sims,
                                 sim_errors=guard.sim_err_total,
                                 moving_target=moving_target, sim_dead=sim_dead,
                                 vlm_baseline_s=baseline)
        if integ["valid"] and integ["metrics"].get("vlm_median_s"):
            self.vlm_meds.append(integ["metrics"]["vlm_median_s"])
        return integ

    def row_fields(self, integ: dict | None) -> dict:
        """Guardrail fields attached to the result row."""
        return dict(guardrail_mode=self.mode,
                    integrity_valid=None if integ is None else integ["valid"],
                    integrity_reasons=None if integ is None else integ["reasons"])

    def after_episode(self, tag: str, vpath: str, integ: dict, success) -> str:
        """Action after the verdict. Returns: "keep" | "reflight" | "final_invalid".

        On reflight, the verdict has been moved to _invalid/verdict_<tag>.try<n>.json, and
        the caller drops the result row and puts the same task back at the front of the queue. A contaminated episode is
        invalid regardless of success/failure — not only SR but step·time·path metrics are distorted too.
        """
        # Dead-server circuit breaker count — accumulates only over consecutive failed api_dead episodes.
        self.api_dead_flights = (self.api_dead_flights + 1
                                 if "api_dead" in integ["reasons"] and not success else 0)
        if integ["valid"] or not self.enforcing:
            return "keep"
        tries = self.attempts.get(tag, 0)
        if tries < self.max_flights - 1:
            self.attempts[tag] = tries + 1
            self._quarantine(vpath, f"verdict_{tag}.try{tries + 1}.json")
            self._log(f"  !! {tag}: integrity failed {integ['reasons']} — re-flying "
                      f"{tries + 1}/{self.max_flights - 1}")
            return "reflight"
        self._log(f"  !! {tag}: integrity failed on all {self.max_flights} flights {integ['reasons']} — "
                  f"recorded as final invalid (integrity.valid=false, RESUME=1 re-measure target)")
        return "final_invalid"

    def model_should_stop(self) -> bool:
        """Do not burn the remaining episodes on a dead server.
        True if there are 2 consecutive api_dead episodes under enforce."""
        return self.enforcing and self.api_dead_flights >= 2

    # -- internal -----------------------------------------------------------------

    def _quarantine(self, vpath: str, name: str) -> None:
        """Move an invalid verdict to _invalid/ — it must not be picked up again by the verdict_*.json glob (summary rebuild·report·RESUME).
        Kept per attempt to allow forensics."""
        try:
            inv_dir = os.path.join(self.mdir, "_invalid")
            os.makedirs(inv_dir, exist_ok=True)
            shutil.move(vpath, os.path.join(inv_dir, name))
        except Exception:
            pass


class NoGuardrail:
    """GUARDRAIL=off — same hooks, no action. No verdict is made either, so the verdict file has no integrity."""

    mode = "off"
    enforcing = False

    def __init__(self, model: str, mdir: str, **_):
        self.model = model
        self.mdir = mdir
        self.max_flights = 1
        self.attempts: dict[str, int] = {}
        self.vlm_meds: list[float] = []
        self.api_dead_flights = 0

    def resume_reflight(self, tag, vpath, verdict) -> bool:
        return False

    def begin_episode(self, tag: str) -> EpisodeGuard:
        # Instrumentation still counts (for logs·forensics) — it just does not cut.
        return EpisodeGuard(tag, api_dead_streak=1 << 30, may_abort=False)

    def assess(self, guard, step_evs, **_) -> None:
        return None

    def row_fields(self, integ=None) -> dict:
        return dict(guardrail_mode="off", integrity_valid=None, integrity_reasons=None)

    def after_episode(self, tag, vpath, integ, success) -> str:
        return "keep"

    def model_should_stop(self) -> bool:
        return False


def make_session(model: str, mdir: str, *, mode: str | None = None,
                 log: Callable[[str], None] = print):
    """Session for the mode. If mode is not given, the GUARDRAIL environment variable (default off)."""
    mode = mode_from_env() if mode is None else mode
    if mode == "off":
        return NoGuardrail(model, mdir)
    return GuardrailSession(model, mdir, mode=mode, log=log)
