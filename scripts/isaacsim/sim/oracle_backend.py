# DroneCATS
# Copyright (c) 2026-present NAVER Cloud Corp.
# Apache-2.0

"""Action oracle (B-full): computes the correct action from GT.

**Different from** the two other "oracles" already in the repo:

    depth_source=oracle   replaces only `depth_m` with GT instead of the model estimate
    oracle_arrival        verifier judges arrival by GT distance instead of the model's claim
    **action oracle (this)**  computes the **whole action** from GT to produce training data

It plugs in where the scripted backend does: as the **backend** of `VlmPolicy`. That way
prompt assembly, JSON parsing, action validation, `point` back-projection, the think flag,
`DecisionRecord` creation and the PX4 closed loop all run **real code**; only the single
model call is replaced.

GT is read from the sidecar that A replaces atomically (`--gt-live`). A writes the camera
frame and GT on the same tick (`--gt-tick frame`), so the GT matches the render B saw.

## Design principle: labels must come from what the model can observe

Using GT **as the basis for decisions** is fine. But **if the label itself is unobservable,
it teaches hallucination**: the model tries to match the answer and, with no evidence in the
observation, learns to guess. This principle decides the `rotate` rule below.

## Action rules

    finished   3D distance <= arrival_radius **AND** mask area >= threshold
               The AND is the point. Without the visibility condition the model learns to
               "declare arrival while looking elsewhere"; the baseline really has that
               failure (police got within 2.2 m, but all 55 declarations had the target
               out of view, ending in timeout).

    go         `point` = centroid of the **visible part** of the mask (0~1000), not the
               centroid of the full projection, so occluders are not pointed at.
               `depth_m` = GT depth **at that pixel**, not the 3D center distance.
               The controller back-projects along the pixel ray, so this keeps it self-consistent.

    rotate     **fixed-increment sweep** (default 60°). The exact bearing is unobservable to
               the model, so labelling it would teach guessing. With the measured 91.7° FOV,
               60° gives 1.5x overlap.

    think      on observable "ambiguity": matching candidates >= 2 (including the target),
               occlusion ratio > 0.5, or mask < threshold.
               ⚠️ The think rule is still being refined. Here the **signals are recorded
               and think is emitted only conservatively** (per-episode budget `think_budget`).

## Limitations

`point` is the mask **centroid**, so under heavy occlusion it may fall outside the object
(when the center is hidden). Left as-is for now, with the basis recorded in `oracle_gt.point_from`.
"""
import json
import math
import os
import time


def _category(kind):
    """Unit for treating decoys as the "same category". People are not recolored, so the axis is category, not color."""
    if not kind:
        return None
    from sim import targets as _TG
    if kind in getattr(_TG, "PEOPLE", ()):
        return "person"
    return kind


class OracleBackend:
    """GT-based action generator satisfying the `VlmBackend` protocol. Does not look at images."""

    provider = "oracle"

    def __init__(self, config, *, response_schema=None, schema_name="vlmpilot_action"):
        cfg = config or {}
        self.model = str(cfg.get("model_name") or "action-oracle-v1")
        self._response_schema = dict(response_schema) if response_schema else None
        self._schema_name = schema_name

        self.gt_path = str(cfg.get("gt_live_path")
                           or "/tmp/vlmpilot_pegasus/latest_gt.json")
        self.target_prim = cfg.get("oracle_target")          # first target if unset
        self.arrival_m = float(cfg.get("oracle_arrival_m", 5.0))
        self.visible_px = int(cfg.get("oracle_visible_px", 50))
        self.rotate_deg = float(cfg.get("oracle_rotate_deg", 60.0))
        self.think_budget = int(cfg.get("oracle_think_budget", 2))
        self.think_occl = float(cfg.get("oracle_think_occlusion", 0.5))
        # -- Lost-target recovery (off by default) --
        #
        # The default rule is "mask == 0 -> rotate in place". But if the target gets hidden
        # by a structure during approach, it **spins in place forever**. Measured: pair S03
        # lost a mailbox visible at 17 m after advancing 3.75 m and rotated for 58 steps,
        # hitting the 60-step cap; both sides failed.
        #
        # Mid-episode occlusion is **legitimate difficulty**: a person would head toward
        # where they last saw it and search again. Failing at that is an oracle limitation,
        # and it makes SR **underestimate** episode validity. Hence recovery.
        #
        # ⚠️ **Only observable quantities are used.** Reading the target position from GT
        # and flying straight makes the upper bound meaningless (no agent can do that). Only
        # two things are used: (1) **own heading** when last seen (self state) and (2) the
        # **depth** at that time (observation).
        # Off by default because changing the policy mixes data. Enable after measuring before/after.
        self.recover = bool(cfg.get("oracle_recover", False))
        self.recover_steps = int(cfg.get("oracle_recover_steps", 3))
        self._last_yaw = None      # heading when last seen (deg)
        self._last_depth = None    # depth at that time (m)
        self._recover_left = 0     # remaining "blind advance" count
        self.think_px = int(cfg.get("oracle_think_min_px", 400))
        self.max_gt_age_s = float(cfg.get("oracle_max_gt_age_s", 5.0))
        # Without a VLM, B is much faster than A and takes **several steps on the same frame**
        # (measured: 24 of 31 steps saw the same GT). After deciding, wait for the next frame
        # to match A's frame period; this stands in for what VLM latency used to do.
        self.pace = bool(cfg.get("oracle_pace", True))
        self.pace_timeout_s = float(cfg.get("oracle_pace_timeout_s", 10.0))
        self.pace_poll_s = float(cfg.get("oracle_pace_poll_s", 0.02))
        # The oracle must decide on the GT of "the frame B saw". A writes the frame and then
        # GT right after, so `GT.t - jpg.mtime` is that skew. If it grows, labels become newer
        # than the observation and disagree with the judge (measured: oracle declared at
        # 4.82 m vs judge 5.21 m -> too_far). Without measuring it, it fails silently.
        self.frame_path = str(cfg.get("frame_path") or "")
        self.max_skew_s = float(cfg.get("oracle_max_skew_s", 1.0))
        self.point_scale = float(cfg.get("point_scale", 1000.0))

        self._think_left = self.think_budget
        self._last_t = None
        self._steps = 0
        self._waits = []
        self._skew = None
        self._skew_warned = False
        self._skews = []
        print(f"[ORACLE] GT {self.gt_path} · arrival {self.arrival_m} m · "
              f"visible {self.visible_px} px · rotate {self.rotate_deg}° · "
              f"think budget {self.think_budget}", flush=True)

    # -- GT read ---------------------------------------------------------
    def _read_gt(self):
        """`(row, reason)`. If unreadable, `(None, reason)`."""
        if not os.path.exists(self.gt_path):
            return None, "sidecar file does not exist"
        try:
            with open(self.gt_path, encoding="utf-8") as fh:
                row = json.load(fh)
        except Exception as exc:  # noqa: BLE001 - os.replace prevents half-written files, but still
            return None, f"could not read: {exc!r}"
        t = row.get("t")
        if isinstance(t, (int, float)):
            age = time.time() - t
            if age > self.max_gt_age_s:
                return None, f"GT is {age:.1f}s old (is A stalled?)"
            if self._last_t is not None and t == self._last_t and not self.pace:
                # Seeing the same GT twice = A did not write a new frame. Not fatal, but
                # the same observation enters the data twice.
                print(f"[ORACLE] ⚠️ GT not updated (t={t}) - "
                      "check A's frame period (enabling oracle_pace aligns it)",
                      flush=True)
            self._last_t = t
            self._skew = None
            if self.frame_path and os.path.exists(self.frame_path):
                try:
                    self._skew = float(t) - os.path.getmtime(self.frame_path)
                    self._skews.append(self._skew)
                    if self._skew > self.max_skew_s and not self._skew_warned:
                        print(f"[ORACLE] ⚠️ GT is ahead of the frame B saw by "
                              f"{self._skew:.2f}s - labels are newer than the observation "
                              "and will disagree with the judge (check oracle_pace)", flush=True)
                        self._skew_warned = True
                except OSError:
                    self._skew = None
        return row, ""

    def _pick_target(self, row):
        """Target to judge and its GT. If unspecified, **a moving target first, else the first one**."""
        targets = row.get("targets") or {}
        if not targets:
            return None, None, None
        meta = row.get("meta") or {}
        if self.target_prim and self.target_prim in targets:
            p = self.target_prim
        else:
            moving = [k for k in sorted(targets) if (meta.get(k) or {}).get("move")]
            p = moving[0] if moving else sorted(targets)[0]
        return p, targets[p], (meta.get(p) or {})

    @staticmethod
    def _candidates(row, target_prim, tmeta, target_visible):
        """Number of **visible instances** matching the prompt attributes (kind x color), and how many are decoys.

        "Decoy ambiguity" means **2 or more** matching visible instances, with the target
        itself counted as a candidate. Counting only decoys and requiring >= 2 would be off by
        one and miss the genuinely ambiguous case of "two of the same kind and color in view".
        """
        meta = row.get("meta") or {}
        decoys = 0
        grades = {"same_kind": 0, "same_color": 0, "same_category": 0, "other": 0}
        tk, tc = tmeta.get("kind"), tmeta.get("color")
        tcat = _category(tk)
        for p, t in (row.get("targets") or {}).items():
            if p == target_prim or not t.get("visible"):
                continue
            m = meta.get(p) or {}
            k, c = m.get("kind"), m.get("color")
            if (k, c) == (tk, tc):
                decoys += 1
            # **Also counted per grade.** `decoys_visible` counts only same kind + same color,
            # but the generator **deliberately never creates** that combination (the episode
            # would become unsolvable even for people). Keeping only that count would report
            # every new decoy as 0 and erase the fact that distractors were present. Grades are
            # kept so the `think` rule can use them.
            if k == tk and c != tc:
                grades["same_kind"] += 1
            elif c is not None and c == tc and k != tk:
                grades["same_color"] += 1
            elif _category(k) == tcat and k != tk:
                grades["same_category"] += 1
            else:
                grades["other"] += 1
        return decoys + (1 if target_visible else 0), decoys, grades

    # -- Rules -----------------------------------------------------------
    def decide_from_gt(self, row):
        """One GT row -> action dict. Rules exactly as in the module docstring (Action rules)."""
        prim, t, tmeta = self._pick_target(row)
        if t is None:
            return {"action": "rotate", "angle_deg": self.rotate_deg,
                    "reason": "no target in GT"}, None

        mask = int(t.get("mask_px") or 0)
        dist = t.get("dist_m")
        yaw_now = row.get("yaw_deg")
        if yaw_now is None:
            yaw_now = ((row.get("camera") or {}).get("yaw_deg"))
        occl = t.get("occlusion_ratio")
        point = t.get("point")
        dpx = t.get("depth_at_point_m")
        visible = mask >= self.visible_px
        candidates, decoys, grades = self._candidates(row, prim, tmeta, visible)

        gt = {"target": prim, "phrase": tmeta.get("phrase"),
              "gt_frame_skew_s": (None if self._skew is None else round(self._skew, 3)),
              "mask_px": mask,
              "dist_m": dist, "occlusion_ratio": occl,
              "decoys_visible": decoys, "candidates_visible": candidates,
              "distractors_visible": grades,
              "visible": visible, "point": point, "depth_at_point_m": dpx}

        # (1) finished: distance AND visibility
        near = isinstance(dist, (int, float)) and dist <= self.arrival_m
        if near and visible:
            return {"action": "finished",
                    "reason": f"distance {dist:.2f}m <= {self.arrival_m}m and "
                              f"mask {mask}px >= {self.visible_px}px"}, gt

        # If visible, remember the "last seen" state; recovery uses only this
        if visible:
            if yaw_now is not None:
                self._last_yaw = float(yaw_now)
            if isinstance(dpx, (int, float)) and math.isfinite(float(dpx)):
                self._last_depth = float(dpx)
            elif isinstance(dist, (int, float)):
                self._last_depth = float(dist)
            self._recover_left = self.recover_steps

        # (2) If not visible, fixed-increment sweep; bearing is unobservable so GT is not used
        if mask == 0:
            why = "target not in view"
            if near:
                why += f" (close at {dist:.2f}m but not visible, so no arrival declared)"

            # (2-a) **Recovery**: turn toward where it was last seen; if already facing it, advance.
            # Occlusion often clears as distance shrinks (passing the structure reveals it again).
            if (self.recover and self._last_yaw is not None
                    and yaw_now is not None and self._recover_left > 0):
                d = ((self._last_yaw - float(yaw_now) + 180.0) % 360.0) - 180.0
                if abs(d) > self.rotate_deg / 2.0:
                    gt["recover"] = {"mode": "turn_back", "to_yaw": self._last_yaw,
                                     "delta_deg": round(d, 1)}
                    return {"action": "rotate", "angle_deg": round(d, 1),
                            "reason": f"{why} - turning back to last-seen heading "
                                      f"{self._last_yaw:.0f}°"}, gt
                self._recover_left -= 1
                step = self._last_depth if self._last_depth else self.arrival_m
                gt["recover"] = {"mode": "advance", "left": self._recover_left,
                                 "depth_m": round(float(step), 2)}
                # `go` to the image center = "advance along the current heading". Depth is the last observed value.
                return {"action": "go", "point": [500, 500],
                        "depth_m": round(float(step), 2),
                        "reason": f"{why} - advancing toward last-seen direction "
                                  f"({self._recover_left} left)"}, gt

            return {"action": "rotate", "angle_deg": self.rotate_deg,
                    "reason": why}, gt

        # (3) If the observation is ambiguous, think. Signals are always recorded; emission stays within budget
        amb = []
        if candidates >= 2:
            amb.append(f"{candidates} same kind/color candidates ({decoys} decoys)")
        if isinstance(occl, (int, float)) and occl > self.think_occl:
            amb.append(f"occlusion {occl:.2f}")
        if 0 < mask < self.think_px:
            amb.append(f"mask {mask}px < {self.think_px}")
        gt["think_signals"] = amb
        if amb and self._think_left > 0 and not near:
            self._think_left -= 1
            return {"action": "think",
                    "reason": "ambiguous observation - " + " · ".join(amb)}, gt

        # (4) go: to the visible centroid, with GT depth at that pixel
        if not point:
            return {"action": "rotate", "angle_deg": self.rotate_deg,
                    "reason": "mask present but centroid unavailable"}, gt
        depth = dpx
        src = "pixel GT depth"
        if depth is None or not math.isfinite(float(depth)):
            depth = dist
            src = "center distance (no pixel depth)"
        act = {"action": "go",
               "point": [int(round(point[0])), int(round(point[1]))],
               "depth_m": round(float(depth), 2),
               "reason": f"visible centroid {point} · {src} {depth:.2f}m"}
        gt["point_from"] = "visible mask centroid"
        gt["depth_from"] = src
        return act, gt

    # -- VlmBackend protocol ---------------------------------------------
    def _generate(self, response_schema):
        row, why = self._read_gt()
        gt = {}
        if row is None:
            # Without GT, **do not invent an action**; only sweep.
            print(f"[ORACLE] no GT - {why}. Falling back to rotate", flush=True)
            act = {"action": "rotate", "angle_deg": self.rotate_deg,
                   "reason": f"no GT: {why}"}
        else:
            act, gt = self.decide_from_gt(row)
            gt = gt or {}
            self._steps += 1
            tag = act["action"].upper()
            print(f"[ORACLE] {self._steps:>3} {tag:<8} {act.get('reason','')}", flush=True)

        props = (response_schema or {}).get("properties") or {}
        if "action_planning" in props:
            # think step: the policy requires the plan schema. Fill it from GT.
            g = gt
            act.setdefault("target_in_view", "yes" if g.get("visible") else "no")
            act.setdefault("target_match", "match")
            act.setdefault("closing_in", "unknown")
            act.setdefault("current_frame",
                           f"{g.get('phrase')} · mask {g.get('mask_px')}px · "
                           f"distance {g.get('dist_m')}m · occlusion {g.get('occlusion_ratio')}")
            act.setdefault("action_planning", act.get("reason", ""))
        return json.dumps(act, ensure_ascii=False)

    def summary(self):
        """One-line pacing/skew summary after the steps. An audit record of data quality."""
        out = {"steps": self._steps}
        if self._waits:
            out["wait_ms_median"] = round(sorted(self._waits)[len(self._waits) // 2] * 1000, 1)
        if self._skews:
            sk = sorted(self._skews)
            out["skew_s_median"] = round(sk[len(sk) // 2], 3)
            out["skew_s_max"] = round(sk[-1], 3)
        return out

    def _wait_next_frame(self):
        """Wait until the next GT (= next frame) appears.

        Waiting **after the decision** matters. Waiting before it would make GT newer than
        the frame B grabbed, breaking "labels come from what the model saw". Waiting after
        means the next iteration grabs a fresh frame; action emission is one frame late,
        which is exactly what real VLM latency does.
        """
        if not self.pace or self._last_t is None:
            return 0.0
        t0 = time.time()
        base = self._last_t
        while time.time() - t0 < self.pace_timeout_s:
            try:
                with open(self.gt_path, encoding="utf-8") as fh:
                    t = json.load(fh).get("t")
            except Exception:  # noqa: BLE001 - may be mid-replace
                t = None
            if isinstance(t, (int, float)) and t != base:
                break
            time.sleep(self.pace_poll_s)
        w = time.time() - t0
        self._waits.append(w)
        if len(self._waits) == 10:
            avg = sum(self._waits) / len(self._waits)
            print(f"[ORACLE] mean frame wait {avg * 1000:.0f} ms "
                  f"(steps are paced to A's frame period)", flush=True)
        return w

    def generate(self, prompt, image_bgr, think=False, response_schema=None):
        out = self._generate(response_schema or self._response_schema)
        self._wait_next_frame()
        return out

    def generate_detailed(self, prompt, image_bgr, think=False, response_schema=None):
        from vlmpilot.policy.vlm_backends import GenerateResult
        return GenerateResult(text=self.generate(prompt, image_bgr, think=think,
                                                 response_schema=response_schema),
                              thoughts=None)

    def generate_multi(self, prompt, images_bgr, labels, think=False, system=None,
                       response_schema=None):
        out = self._generate(response_schema or self._response_schema)
        self._wait_next_frame()
        return out

    def generate_multi_detailed(self, prompt, images_bgr, labels, think=False,
                                system=None, response_schema=None):
        from vlmpilot.policy.vlm_backends import MultiGenerateResult
        return MultiGenerateResult(
            text=self.generate_multi(prompt, images_bgr, labels, think=think,
                                     system=system, response_schema=response_schema),
            thoughts=None)


def register(verbose=True):
    """Add `"oracle"` to the table `build_backend` looks up; no changes to team code."""
    from vlmpilot.policy import vlm_backends

    vlm_backends._BACKEND_CLASSES["oracle"] = OracleBackend
    if "oracle" not in vlm_backends.PROVIDERS:
        vlm_backends.PROVIDERS = tuple(vlm_backends.PROVIDERS) + ("oracle",)
    if verbose:
        print(f"[ORACLE] api_provider registered - available: {vlm_backends.PROVIDERS}", flush=True)
    return OracleBackend
