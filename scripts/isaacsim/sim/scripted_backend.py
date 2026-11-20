# DroneCATS
# Copyright (c) 2026-present NAVER Cloud Corp.
# Apache-2.0

"""Scripted backend (B-minimal): replaces only the VLM call and runs the rest of the path for real.

Why not bypass the policy
  `PegasusFlightController.run_step` calls `self.policy.decide(frame, instruction)`, and
  `VlmPolicy.decide` parses the backend's **raw JSON**. Plugging a stub in place of the
  policy would skip prompt assembly, JSON parsing, action validation, `point`
  back-projection, the think flag and `DecisionRecord` creation. Swapping only the backend
  runs all of that as real code and replaces **a single model call**.

  It also needs no VLM quota and does not touch anyone's vLLM server while still running
  one full closed loop.

How it plugs in (no changes to team code)
  `vlm_backends.build_backend` looks up `_BACKEND_CLASSES[provider]`. `register()` adds
  `"scripted"` there, so `api_provider: scripted` in the config selects this backend
  without code changes.

Script
  Read from config `scripted_actions` (a list) or `scripted_path` (a JSON/JSONL file).
  Each entry is exactly an action-space v3 response:

      - {"action": "rotate", "angle_deg": 30}
      - {"action": "go", "point": [500, 520], "depth_m": 3.0}
      - {"action": "think"}
      - {"action": "finished"}

  `point` follows the config's `point_order` (0~1000 scale). On think steps the policy
  requests the plan schema, so those fields are filled too and the plan path also runs.
"""
import json
import os
import time


DEFAULT_SCRIPT = [
    {"action": "rotate", "angle_deg": 30},
    {"action": "go", "point": [500, 520], "depth_m": 3.0},
    {"action": "think"},
    {"action": "go", "point": [500, 520], "depth_m": 3.0},
    {"action": "finished"},
]

_PLAN_FILLER = {
    "target_in_view": "unsure",
    "target_match": "unsure",
    "closing_in": "unknown",
    "current_frame": "Scripted backend - does not look at the image.",
    "action_planning": "Emitting the next scripted entry as-is.",
}


def _load_script(config):
    path = config.get("scripted_path")
    if path:
        path = os.path.expanduser(str(path))
        with open(path, encoding="utf-8") as fh:
            text = fh.read().strip()
        if not text:
            raise ValueError(f"script is empty: {path}")
        if text.lstrip().startswith("["):
            rows = json.loads(text)
        else:                                    # JSONL
            rows = [json.loads(l) for l in text.splitlines() if l.strip()]
    else:
        rows = config.get("scripted_actions") or DEFAULT_SCRIPT
    rows = [dict(r) for r in rows]
    if not rows:
        raise ValueError("script has no entries")
    for r in rows:
        if "action" not in r:
            raise ValueError(f"script entry has no action: {r}")
    return rows


class ScriptedBackend:
    """Scripted responder satisfying the `VlmBackend` protocol. Does not look at images."""

    provider = "scripted"

    def __init__(self, config, *, response_schema=None, schema_name="vlmpilot_action"):
        cfg = config or {}
        self.model = str(cfg.get("model_name") or "scripted")
        self._schema_name = schema_name
        self._response_schema = dict(response_schema) if response_schema else None
        self.script = _load_script(cfg)
        self.loop = bool(cfg.get("scripted_loop", False))
        self.latency_ms = float(cfg.get("scripted_latency_ms", 0.0) or 0.0)
        self._i = 0
        print(f"[SCRIPTED] script {len(self.script)} steps · loop={self.loop} · "
              f"latency {self.latency_ms:.0f} ms", flush=True)
        for k, row in enumerate(self.script):
            print(f"[SCRIPTED]   {k + 1}. {json.dumps(row, ensure_ascii=False)}", flush=True)

    # -- Script progress -------------------------------------------------
    def _next(self, response_schema):
        if self._i >= len(self.script):
            if not self.loop:
                # If the policy asks again after the script ends, answer "finished" to close the episode.
                # Returning `None` would make the policy drop the step and loop until the cap.
                print("[SCRIPTED] script exhausted - closing with finished", flush=True)
                return {"action": "finished"}
            self._i = 0
        row = dict(self.script[self._i])
        self._i += 1

        # On think steps the policy requests the plan schema; fill those fields too
        props = (response_schema or {}).get("properties") or {}
        if "action_planning" in props:
            for k, v in _PLAN_FILLER.items():
                row.setdefault(k, v)
        if self.latency_ms:
            time.sleep(self.latency_ms / 1000.0)
        print(f"[SCRIPTED] {self._i}/{len(self.script)} → "
              f"{json.dumps(row, ensure_ascii=False)}", flush=True)
        return row

    # -- VlmBackend protocol ---------------------------------------------
    def generate(self, prompt, image_bgr, think=False, response_schema=None):
        return json.dumps(self._next(response_schema or self._response_schema),
                          ensure_ascii=False)

    def generate_detailed(self, prompt, image_bgr, think=False, response_schema=None):
        from vlmpilot.policy.vlm_backends import GenerateResult
        text = self.generate(prompt, image_bgr, think=think,
                             response_schema=response_schema)
        return GenerateResult(text=text, thoughts=None)

    def generate_multi(self, prompt, images_bgr, labels, think=False, system=None,
                       response_schema=None):
        return json.dumps(self._next(response_schema or self._response_schema),
                          ensure_ascii=False)

    def generate_multi_detailed(self, prompt, images_bgr, labels, think=False,
                                system=None, response_schema=None):
        from vlmpilot.policy.vlm_backends import MultiGenerateResult
        text = self.generate_multi(prompt, images_bgr, labels, think=think,
                                   system=system, response_schema=response_schema)
        # `usage` defaults to None, which correctly means no provider usage
        return MultiGenerateResult(text=text, thoughts=None)


def register(verbose=True):
    """Add `"scripted"` to the table `build_backend` looks up; no changes to team code."""
    from vlmpilot.policy import vlm_backends

    vlm_backends._BACKEND_CLASSES["scripted"] = ScriptedBackend
    if "scripted" not in vlm_backends.PROVIDERS:      # list used for error messages
        vlm_backends.PROVIDERS = tuple(vlm_backends.PROVIDERS) + ("scripted",)
    if verbose:
        print(f"[SCRIPTED] api_provider registered - available: {vlm_backends.PROVIDERS}", flush=True)
    return ScriptedBackend
