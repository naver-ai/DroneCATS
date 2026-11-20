# DroneCATS
# Copyright (c) 2026-present NAVER Cloud Corp.
# Apache-2.0

"""**API call·failure counts** for one episode — a pure parser that counts executor stdout.

Why count separately: the backend logs failures only from the provider's view (`[VLM] request failed`), so
it does not tell "which arm starved how many times". In a head-body formation this distinction decides
the verdict: if a body starves, only that vehicle stands still; if the head starves, the whole formation keeps
flying on the previous instruction. Either way the episode stays valid while only SR quietly drops.

Lines counted (executor stdout contract):

  `[HEAD] call=<n> trigger=…`     head's **cumulative** call counter. Includes failed attempts
  `[HEAD] no reply (attempt n)`   that call got no body — head failure
  `[HEAD] parse_failed attempt=n` a body arrived but the instruction could not be read — not a failure, but a call was used
  `[RAW] …`                       the body received by one body call (multiple lines, one chunk via PRINT_LOCK)
  `[BODY] call_failed drone=… err=…`  the body call got no body — body failure
  `[VLM] rate_limited …`          429/RESOURCE_EXHAUSTED family (logged by vlm_backends)

`head_calls` is the last cumulative value seen + the number of attempts after it — so even in episodes without
a successful turn, the call count does not stay 0. `body_calls` is the number of `[RAW]` chunks + failures.

Limitation (approximate denominator): if two vehicles' `[RAW]` chunks come out back to back with no line in between, they
are counted as one chunk and body_calls is rarely 1 short. Failure counts (numerator) are exact, one line each.
Centralized arms (not head-body) have no `[HEAD]`·`[BODY]`·`[RAW]` lines, so everything is 0 — that arm's
failures are counted by the guardrail's `vlm_failed` (= `[VLM]` lines).
"""

from __future__ import annotations

import re

_HEAD_CALL_RE = re.compile(r"^\[HEAD\] call=(\d+)\b")
_HEAD_NO_REPLY = "[HEAD] no reply"
_HEAD_PARSE_FAILED = "[HEAD] parse_failed"
_BODY_CALL_FAILED = "[BODY] call_failed"
_RAW = "[RAW]"
_RATE_LIMITED = "[VLM] rate_limited"

KEYS = ("head_calls", "head_failed", "body_calls", "body_failed", "rate_limited")


def count_api_failures(lines) -> dict[str, int]:
    """stdout lines → {head_calls, head_failed, body_calls, body_failed, rate_limited}.

    Pure function — touches no files·sim·clock. The runner feeds pilot.log as-is after the episode ends.
    """
    head_seen = head_pending = head_failed = 0
    body_ok = body_failed = rate_limited = 0
    prev_raw = False
    for raw_line in lines:
        line = raw_line.rstrip("\n")
        is_raw = line.startswith(_RAW)
        if is_raw:
            if not prev_raw:
                body_ok += 1          # first line of a chunk = one body call received a body
            prev_raw = True
            continue
        prev_raw = False
        m = _HEAD_CALL_RE.match(line)
        if m:
            # the cumulative counter already includes all attempts so far — pending ones are absorbed.
            head_seen, head_pending = max(head_seen, int(m.group(1))), 0
        elif line.startswith(_HEAD_NO_REPLY):
            head_failed += 1
            head_pending += 1
        elif line.startswith(_HEAD_PARSE_FAILED):
            head_pending += 1
        elif line.startswith(_BODY_CALL_FAILED):
            body_failed += 1
        elif line.startswith(_RATE_LIMITED):
            rate_limited += 1
    return {"head_calls": head_seen + head_pending, "head_failed": head_failed,
            "body_calls": body_ok + body_failed, "body_failed": body_failed,
            "rate_limited": rate_limited}


def count_api_failures_file(path: str) -> dict[str, int]:
    """Count one episode's worth of pilot.log. Without a log everything is 0 — does not block the verdict."""
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return count_api_failures(f)
    except OSError:
        return dict.fromkeys(KEYS, 0)
