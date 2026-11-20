# DroneCATS
# Copyright (c) 2026-present NAVER Cloud Corp.
# Apache-2.0

"""GT sidecar (process A): records rendered GT as a continuous stream.

Why "stream + post-hoc join"
  A (`pegasus_sim.py`) and B (runner/controller) are **separate processes**. Replicator
  annotators live only in A, while the `step` number in `steps.jsonl` is counted by B
  (`_record_step` increments it). Without changing B, the only way to connect them is for
  A to write a timestamped stream and join afterwards, the same discipline used for
  passing GT depth as a sidecar **file** (`pegasus_controller.capture_depth`).

Join key
  Wall clock (`time.time()`). Both processes run on the same machine so it is shared,
  and `steps.jsonl` already records it as `t`. B also records `frame_age_s`, so the
  joiner targets `t - frame_age_s` (the moment the frame was actually captured).
  Sim time (`sim_time_s`) is written on both sides but is **not used as a key**: B's
  value is the PX4 odometry clock and A's is the Isaac world clock, so their origins may
  differ. The joiner measures and reports that offset.

Occlusion ratio
  Validated method (c): hide the occluders and re-render the unoccluded silhouette. In
  real scenes occluders are arbitrary, so we **hide the environment root** (all static
  occluders live under it). This adds one render per step, so it is off by default.
"""
import json
import math
import os
import re
import time

import numpy as np

# Annotators are **derived from need.** Each one attached adds a render pass, so attaching
# "just in case" costs every tick. Measured: `bounding_box_2d_loose` was attached but never
# read, so it was removed.
_ANNOTATORS_CORE = (
    "rgb",
    "distance_to_image_plane",       # depth_at_point: the oracle's depth_m
    "instance_id_segmentation",      # mask_px, key for occlusion ratio
    # `occlusion` returns `(instanceId, semanticId, occlusionRatio)` structs.
    # **Keyed by `instanceId`**, so it can be attached per target. Getting the same value
    # from `bounding_box_2d_tight`'s `occlusionRatio` was keyed only by `semanticId`, so
    # target rows could not be selected (5/5 failures). Its id space is **identical** to
    # `instance_id_segmentation` (measured: id 1 -> /World/T on both).
    "occlusion",
)
# `semantic_segmentation` is used in only three places: saved frames (`_sem.png`),
# render-based occlusion (`--gt-occlusion`, unusable at Brownstone scale), `--gt-debug-luma`.
# **`think_decoy` counts with `n_visible`, not semantic**, so drop it if none of the three apply.
_ANNOTATOR_SEMANTIC = "semantic_segmentation"

# GT_OCCLUSION_VECTORIZED=1 picks the numpy path in `occlusion_from_annotator`. Off by default:
# same result, but a faster sim (higher RTF) can change moving-target outcomes (see the RunPod README).
_OCCLUSION_VECTORIZED = os.environ.get("GT_OCCLUSION_VECTORIZED", "0") == "1"


# BT.601 luma weights. Same definition as `cv2.cvtColor(..., COLOR_RGB2GRAY)`, so luma values
# measured by other tools are directly comparable.
_LUMA_W = np.array([0.299, 0.587, 0.114], np.float32)


def label_map(ids, imap, paths):
    """Fold the instance id map into a **target label map** in one pass.

    Returns `(lab, order)`: `lab[y,x]` is 0 (none) or a 1-based index into `order`.

    Why: previously each target **rescanned the whole frame** with `np.isin(ids, i)`. At
    640x480 with 61 targets that is 307,200 px x 61 per tick (measured 89.5 ms). Building the
    label map once and aggregating with `bincount` is independent of target count
    (8.7 ms, **10.3x**).

    **Values are identical.** A pixel cannot belong to two targets (one instance id each),
    so folding into labels does not change any target's pixel set. If overlapping ids come
    in, the later target wins, whereas the old `np.isin` counted the pixel for **both**, so
    results would differ then. `_instance_map` prevents such input via path boundaries
    (the t1/t10 incident).
    """
    try:
        top = int(ids.max())
    except (ValueError, AttributeError):
        return None, []
    lut = np.zeros(top + 2, np.int32)
    order = []
    for path in paths:
        iv = imap.get(path)
        if not iv:
            continue
        order.append(path)
        k = len(order)
        for i in iv:
            if 0 <= int(i) <= top:
                lut[int(i)] = k
    if not order:
        return None, []
    return lut[ids], order


def target_pixel_stats(ids, rgb, dep, imap, paths):
    """Aggregate per-target `(px, cx, cy, depth, lum, mean_rgb)` in **one frame pass**.

    Uses linearity of the mean: mean luma **equals exactly** the weighted sum of channel
    means (`mean(w·rgb) == w·mean(rgb)`), so pixels need not be gathered again.
    """
    out = {p: {"px": 0, "cx": None, "cy": None, "depth": None,
               "lum": None, "mean_rgb": None} for p in paths}
    lab, order = label_map(ids, imap, paths)
    if lab is None:
        return out
    flat = lab.ravel()
    n = len(order) + 1
    cnt = np.bincount(flat, minlength=n)[:n]
    H, W = lab.shape
    yy, xx = np.indices((H, W))
    sx = np.bincount(flat, weights=xx.ravel().astype(np.float64), minlength=n)[:n]
    sy = np.bincount(flat, weights=yy.ravel().astype(np.float64), minlength=n)[:n]
    ch = rgb[:, :, :3].astype(np.float64)
    sc = [np.bincount(flat, weights=ch[:, :, c].ravel(), minlength=n)[:n]
          for c in range(3)]
    for k, path in enumerate(order, start=1):
        px = int(cnt[k])
        if px <= 0:
            continue
        cx, cy = sx[k] / px, sy[k] / px
        # **Cast back to float32 before rounding.** The old implementation produced
        # `px_rgb.mean()` in float32 and 280 episodes were recorded with it. float64 is more
        # accurate but splits 120.4 vs 120.5 at `round(·,1)` boundaries; speed is no reason
        # to change recorded values.
        mean_rgb = [float(np.float32(sc[c][k] / px)) for c in range(3)]
        v = float(dep[int(cy), int(cx)])
        out[path] = {
            "px": px, "cx": float(cx), "cy": float(cy),
            "depth": (v if math.isfinite(v) else None),
            "lum": float(np.dot(_LUMA_W.astype(np.float64), mean_rgb)),
            "mean_rgb": [round(float(v_), 1) for v_ in mean_rgb],
        }
    return out


def _annotators_for(save_frames, occlusion=False, debug_luma=False):
    """Annotators actually needed for this configuration."""
    need = list(_ANNOTATORS_CORE)
    if ("semantic" in (save_frames or ())) or occlusion or debug_luma:
        need.insert(3, _ANNOTATOR_SEMANTIC)
    return tuple(need)


def _copy_occlusion(oc):
    """Copy the `occlusion` output; like other annotators it is a buffer view."""
    arr = oc.get("data") if isinstance(oc, dict) else oc
    if isinstance(arr, np.ndarray):
        return {"data": np.array(arr, copy=True)}
    return None


def resolve_camera(stage, index=0):
    """Find the `<body>/fpv_cam` attached by `pegasus_sim.setup_camera`.

    The path is defined elsewhere, so we look it up on the stage instead of creating it.
    With multiple vehicles, sorted prim path order is vehicle order (`per_vehicle_path`).
    """
    from pxr import Usd, UsdGeom

    found = sorted(
        str(p.GetPath())
        for p in Usd.PrimRange.Stage(stage, Usd.PrimAllPrimsPredicate)
        if p.IsA(UsdGeom.Camera) and p.GetName() == "fpv_cam"
    )
    if not found:
        return None
    return found[min(index, len(found) - 1)]


def discover_env_root(stage, root="/World"):
    """Pick the static environment root: the direct child of `/World` with the most meshes."""
    from pxr import Usd, UsdGeom

    best, best_n = None, 0
    parent = stage.GetPrimAtPath(root)
    if not parent:
        return None
    for child in parent.GetChildren():
        n = sum(
            1
            for p in Usd.PrimRange(
                child, Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate))
            if p.IsA(UsdGeom.Mesh)
        )
        if n > best_n:
            best, best_n = str(child.GetPath()), n
    return best


class GTStream:
    """Call `tick()` periodically from A's loop to accumulate GT."""

    def __init__(self, out_dir, camera_prim=None, width=640, height=480, *,
                 camera_index=0, target_pattern=None, occl_target=None,
                 targets=(), env_root=None, occlusion=False, hz=5.0,
                 save_frames=("rgb", "depth", "instance", "semantic"),
                 visible_min_px=50, think_min_px=400, orchestrator_step=False,
                 settle_renders=2, debug_luma=False, ray_skip_m=0.6,
                 sample_inset=0.2, live_path=None):
        self.out_dir = out_dir
        self.frames_dir = os.path.join(out_dir, "gt_frames")
        self.camera_prim = camera_prim
        self.camera_index = int(camera_index)
        self.target_pattern = target_pattern
        self.occl_target = occl_target
        self.W, self.H = int(width), int(height)
        self.targets = list(targets)
        self.env_root = env_root
        # The env root must be resolvable lazily: the stage is empty before the scene loads
        self.want_occlusion = bool(occlusion)
        self.occlusion = bool(occlusion) and env_root is not None
        self.min_dt = (1.0 / float(hz)) if hz and hz > 0 else 0.0
        self.save_frames = set(save_frames or ())
        self.visible_min_px = int(visible_min_px)
        self.think_min_px = int(think_min_px)
        self.orchestrator_step = bool(orchestrator_step)
        self.settle_renders = max(1, int(settle_renders))
        self.debug_luma = bool(debug_luma)

        self._ann = None
        self._rp = None
        self._jl = None
        self._n = 0
        self._last = 0.0
        self._busy = False          # guards `tick` re-entry (it wraps app.update)
        self._miss_streak = 0
        self._warned_miss = False
        self._warned_free = False
        # Occlusion state machine. Annotator data **lags rendering by one frame** (measured):
        # reading right after a visibility change returns the pre-change frame, so we let
        # `settle_renders` frames pass after each change.
        # 0=normal read, 1=hide settling, 2=silhouette read + restore settling
        self._phase = 0
        self._settle = 0
        self._pending = None
        self._sem_classes = {}      # global class name -> local uint8
        self._mpu = 1.0
        self._env_img = None
        self._occl_skip = set()
        self._occl_target = None     # the single target whose occlusion is measured
        self._hidden = []            # Imageables hidden during the silhouette pass
        # Pose read path. With Fabric enabled **the vehicle pose in USD freezes**; reading
        # it naively silently records every row at the spawn position (see `_camera_matrix`).
        self._pose_view = None       # XFormPrim view; False = unusable
        self._pose_source = "usd"
        self._pose_warned = False
        self._pose_cmp_left = 20     # first N ticks are compared against USD and logged
        self._pose_cmp_max = 0.0
        # ── Measure mode ─────────────────────────────────────────────────
        # Measured: render cost depends on **whether any annotator is
        # attached** (5 vs 4 annotators: no change; 2 vs 1 products: no change; **only 0
        # annotators: -64%**). The judge only looks at `kind == "finished"` steps,
        # measured at **one declaration per episode**.
        #
        # So annotators stay off normally and are enabled **only near a target**. Since the
        # judge is `ok = (dist ≤ δ) AND visible`, **being always on within δ preserves SR
        # exactly.** A declaration from far away has `near=False` and fails anyway.
        #
        # Measured that detaching **returns** to the 0-annotator level
        # (44.1 -> 1,974.8 -> 37.5 ms; round trip attach ~200-380 / detach ~70-190 ms).
        self.mode = "analysis"
        # ── **Where** to apply the gate ──────────────────────────────────
        # "product"    disable the GT render product (`hydra_texture.set_updates_enabled`)
        # "annotator"  detach annotators; ⚠️ **breaks the policy camera**
        #
        # `rep.AnnotatorRegistry.get_annotator(name)` returns a **singleton per name**. The
        # policy camera (`isaacsim.sensors.camera.Camera`) uses the same function, so `rgb`
        # and `distance_to_image_plane` are **the same objects**. A `detach()` on the GT side
        # also detaches them from the camera, and its `_data_acquisition_callback` raises
        # `AnnotatorRegistryError: Annotator rgb is not attached` every frame. Then
        # `latest.jpg` stops updating and **the VLM sees a stale image.**
        #
        # Measured: **3,432** such exceptions in one measure-mode run, and B
        # reported a frame 203.1 s old. The render products were separate, but we missed
        # that **annotator objects are shared**.
        #
        # Disabling the product does not touch annotator attachment, so the camera is fine.
        # It is also cheaper (same run):
        #
        #     (1) ann5, rp ON   461.5 ms   (0 errors)
        #     (3) ann5, rp OFF  141.3 ms   (0 errors, **-69%**)
        #
        # Switch cost is also rp 0-11 ms vs ann 110-229 ms.
        self.gate_mode = "product"
        self.near_on_m = 10.0      # enable within this distance (δ 5 m + margin)
        self.near_off_m = 15.0     # disable beyond this distance (hysteresis)
        self._ann_on = None        # None=unknown yet, True/False
        self._ann_switches = 0
        self._gate_warned = False
        # ── Takeoff warm-up ──────────────────────────────────────────────
        # Gating by distance alone **silently breaks `seen_at_start`.** Episodes usually
        # start 15 m from the target (plan `start_dist_m`), so at the step-1 capture the
        # annotators are off, `mask_px` is `None`, and B **records it as "not visible"**.
        # The judge uses `seen_at_start is True` in searching tasks as a "effectively an
        # approaching task" warning (`judge_run.py`), and that check would never fire.
        #
        # So we **detect the episode start geometrically.** A does not know B's step
        # number, but it can see that right after reset the vehicle is landed
        # (z ≈ 0.07 m). Near the ground we latch on, and after takeoff above the altitude
        # we keep it on for `warm_hold_s` more **sim seconds**; arming, takeoff and the
        # first capture finish within that.
        #
        # ⚠️ This latch is **mechanism, not insurance**. If it misfires, values are silently
        # wrong; a measure output whose step-1 `mask_px` is empty should not be trusted.
        # ── Warm-up method ───────────────────────────────────────────────
        # "signal"   B signals "GT pixels needed" via a file: **observable**
        # "altitude" A infers episode start from altitude: works with older B
        # "off"      distance gate only
        #
        # Why signal is cheaper: the altitude latch must cover the whole takeoff
        # (measured 129 sim seconds), while only a couple of ticks around step 1 are needed.
        self.warm_mode = "altitude"
        self.hold_path = None       # file touched by B
        self.hold_s = 15.0          # wall-clock seconds a request stays valid
        self.warm_z_m = 1.5        # below this altitude = episode start (landed/taking off)
        self.warm_clear_z_m = 2.0  # release once stable above this altitude
        self.warm_hold_s = 5.0     # sim seconds that altitude must be held **continuously**
        self.warm_max_s = 240.0    # safety cap: warn and release beyond this
        self._warm_on = False      # whether this tick was enabled by warm-up (logged in row)
        self._warm_since = None    # sim time the latch was set
        self._warm_above = None    # time it started staying above the clear altitude
        self._warm_capped = False  # released by the cap before (warn once)
        # ── The tick right after attaching is **not measured** ───────────
        # Annotator data is filled by renders **after** attaching, so reading on the attach
        # tick returns all zeros. Measured: all 61 targets had `mask_px: 0` and **14 px on
        # the next tick**.
        #
        # Writing that row as `ann: True` mixes "not visible" with "not measured". If an
        # arrival declaration lands on that tick the verdict flips to `not_visible`. Rare,
        # but **silently** wrong, so we cut it: the tick right after attaching records
        # geometry only, same as detached (`ann: False`).
        self._ann_settle = 0
        # **Build the BBox cache once and reuse it.** Previously it was rebuilt on every
        # `_world_pos` call: 61 calls per tick for 61 targets (122 in measure mode), making
        # the cache useless. Targets move, so it is **cleared every tick** (`Clear`).
        self._bbox = None
        self._wpos_tick = {}        # world positions already computed this tick
        self._free_method = None
        self._ray = None             # physics raycast (set by the launcher after play)
        self._occl_dbg = None
        self._occl_ann_warned = False
        self._inst_overlap_warned = False
        self.ray_skip_m = float(ray_skip_m)
        self.sample_inset = float(sample_inset)
        # "GT for the current frame" sidecar read by the executor (B). Same discipline as
        # passing GT depth: write to tmp and swap with os.replace.
        self.live_path = live_path
        # prim -> {color, kind, noun, class, phrase, move, speed_mps, mover}
        # `speed_mps` is the speed **actually applied**: pedestrians use the clip's native
        # speed rather than the requested one, so the data must state each episode's speed.
        self.target_meta = {}
        self._live_warned = False
        self._camera_meta = {}

    # ── Setup ───────────────────────────────────────────────────────────
    def _setup(self):
        import omni.replicator.core as rep
        import omni.usd
        from pxr import UsdGeom

        os.makedirs(self.frames_dir, exist_ok=True)
        stage = omni.usd.get_context().get_stage()
        self._mpu = UsdGeom.GetStageMetersPerUnit(stage)

        self._rp = rep.create.render_product(self.camera_prim, (self.W, self.H))
        self._annotators = _annotators_for(self.save_frames, self.want_occlusion,
                                           self.debug_luma)
        self._ann = {n: rep.AnnotatorRegistry.get_annotator(n)
                     for n in self._annotators}
        # **Annotators are always attached regardless of mode.** Measure mode disables the
        # render product, not the annotators: detaching annotators also detaches the policy
        # camera's `rgb` and freezes `latest.jpg` (see the `gate_mode` comment above).
        for a in self._ann.values():
            a.attach(self._rp)
        if self.mode != "measure":
            self._ann_on = True
        else:
            # **Start with the product disabled**: lowering the flag alone disables nothing
            how = self.set_product_updates(False) \
                if self.gate_mode == "product" else "annotator"
            if how is None:
                self._ann_on = True
                self._gate_warned = True
                print("[GT] ⚠️ render product toggle has no effect; measure mode is "
                      "**off, running like analysis** (slower but exact). "
                      "Detaching annotators would also detach the policy camera's `rgb` "
                      "and freeze `latest.jpg`, so that path is not used (measured).",
                      flush=True)
            else:
                self._ann_on = False
                print(f"[GT] measure mode: starting with GT render product off "
                      f"(enabled within {self.near_on_m:g} m of a target, toggle {how})",
                      flush=True)
        print(f"[GT] {len(self._annotators)} annotators {list(self._annotators)}",
              flush=True)

        cam = UsdGeom.Camera(stage.GetPrimAtPath(self.camera_prim))
        if cam:
            focal = cam.GetFocalLengthAttr().Get() or 0.0
            ap_h = cam.GetHorizontalApertureAttr().Get() or 0.0
            ap_v = cam.GetVerticalApertureAttr().Get() or 0.0
            self._camera_meta = {
                "prim": self.camera_prim,
                "width": self.W, "height": self.H,
                # Actual FOV; may differ from the requested one, so both are kept (setup_camera)
                "hfov_deg": (round(math.degrees(2 * math.atan(ap_h / (2 * focal))), 3)
                             if focal else None),
                "vfov_deg": (round(math.degrees(2 * math.atan(ap_v / (2 * focal))), 3)
                             if focal else None),
            }
        if self.occlusion:
            self._env_img = UsdGeom.Imageable(stage.GetPrimAtPath(self.env_root))
            # If a target is **under** the env root, the silhouette pass hides it too, so
            # free_px becomes 0 and occlusion is silently meaningless (measured: all 832
            # pairs 0). Such targets are excluded from occlusion and left as `None`.
            pre = self.env_root.rstrip("/") + "/"
            self._occl_skip = {p for p in self.targets if p.startswith(pre)}
            if self._occl_skip:
                print(f"[GT] ⚠️ {len(self._occl_skip)}/{len(self.targets)} targets are "
                      f"under the env root ({self.env_root}); occlusion not measured for them. "
                      "Spawn targets outside the environment if occlusion is needed", flush=True)
            # Occlusion is measured for **one target** only. The oracle needs occlusion of
            # "the" target, and per-target silhouettes would multiply renders by target count.
            outside = [p for p in self.targets if p not in self._occl_skip]
            self._occl_target = self.occl_target or (outside[0] if outside else None)
            if self._occl_target is None:
                self.occlusion = False
                print("[GT] no target outside the environment; occlusion disabled", flush=True)
            else:
                print(f"[GT] occlusion target {self._occl_target}", flush=True)

        self._jl = open(os.path.join(self.out_dir, "gt_stream.jsonl"), "a",
                        encoding="utf-8")
        json.dump(
            {"camera": self._camera_meta,
             "targets": self.targets,
             "env_root": self.env_root,
             "occlusion_method": ("(c) hide env root + other targets and re-render the "
                                  "silhouette; area counted as non-background semantic pixels "
                                  "(instance idToLabels comes back INVALID after hiding)"
                                  if self.occlusion else None),
             "occlusion_target": self._occl_target,
             "join": "nearest (t - frame_age_s) in steps.jsonl; verifier/isaacsim/join_gt.py",
             "targets_meta": self.target_meta,
             "visible_min_px": self.visible_min_px,
             "think_min_px": self.think_min_px,
             "meters_per_unit": self._mpu,
             "annotators": list(getattr(self, "_annotators", _ANNOTATORS_CORE))},
            open(os.path.join(self.out_dir, "gt_meta.json"), "w", encoding="utf-8"),
            ensure_ascii=False, indent=1)
        if self.live_path:
            os.makedirs(os.path.dirname(self.live_path) or ".", exist_ok=True)
            print(f"[GT] live sidecar {self.live_path}; B receives GT via this file",
                  flush=True)
        print(f"[GT] sidecar ready: cam {self.camera_prim} {self.W}x{self.H}, "
              f"targets {len(self.targets)}, occlusion {'on' if self.occlusion else 'off'}, "
              f"{self.out_dir}", flush=True)

    # ── Acquisition ─────────────────────────────────────────────────────
    def _validate(self, d):
        """Whether every annotator is filled with the expected shape. Checking only rgb lets segs arrive empty."""
        r = d.get("rgb")
        if not (isinstance(r, np.ndarray) and r.ndim == 3 and r.shape[:2] == (self.H, self.W)):
            return "rgb"
        dp = d.get("distance_to_image_plane")
        if not (isinstance(dp, np.ndarray) and dp.shape[:2] == (self.H, self.W)):
            return "depth"
        # **Check only attached ones.** Hardcoding would wait forever on a removed annotator.
        # Measured: after removing semantic, the readiness check failed 30 ticks in a row and
        # GT never ran a single tick (1 MB on disk, 0 ticks).
        seg = [n for n in getattr(self, "_annotators", ())
               if n.endswith("_segmentation")]
        for n in seg:
            v = d.get(n)
            if not isinstance(v, dict):
                return n
            a = v.get("data")
            if not (isinstance(a, np.ndarray) and a.size > 0
                    and a.shape[:2] == (self.H, self.W)):
                return n
        return None

    def _read(self):
        """Read the current render output. Do not call `app.update()` here: we wrap A's
        loop, so it would re-enter and desync physics time."""
        if self.orchestrator_step:
            import omni.replicator.core as rep
            rep.orchestrator.step()
        d = {n: a.get_data() for n, a in self._ann.items()}
        return d, self._validate(d)

    # ── Target GT ───────────────────────────────────────────────────────
    def _inst_map(self, info):
        """Target path -> **list of instance ids.**

        ⚠️ There is not one id per target. Pedestrian characters have sub-meshes split by
        material (`…/opaque__fabric__policeshirt`, `…/opaque__leather__policeshoes` …), so
        one person arrives as **8 instance ids**. Previously only the last one was kept,
        under-measuring a person's mask by 5-25x: a person at 10 m showed 17-28 px
        (expected ~700 px), skewing visibility and `seen_at_start`. Props have one mesh, so
        it went unnoticed. Now **all prefix-matching ids are unioned**.
        """
        # ⚠️ **Path boundaries matter.** With a plain `path in sv`,
        # `/World/DG_targets/t1` is a substring of `/World/DG_targets/t10`, so
        # **t1 absorbs the masks of t10-t15 entirely** (measured: the fire hydrant t1 counted
        # six pedestrians' pixels, and the oracle followed the pedestrian in front instead).
        # So we check **"is the label the target or a descendant of it"**.
        #
        # Previously each label was compared against all 61 targets. Brownstone frames have
        # tens of thousands of instance labels, so millions of string comparisons per tick;
        # measured, `_inst_map` alone took **55.3 s of 71.9 s GT tick time (77%), 19% of
        # wall clock**. Now it is reduced in two steps:
        #   (1) **reject in one shot** by common prefix: env labels (`/World/layout/…`)
        #       diverge from the target prefix (`/World/DG_targets/…`) on the first compare
        #   (2) only the rest **walk up ancestors** and look them up in a set (O(depth))
        # The result is identical: "target or descendant of a target" is the definition.
        paths = list(self.targets)
        exact = set(paths)
        pre = os.path.commonprefix(paths) if paths else ""
        out = {}
        for k, v in (info.get("idToLabels") or {}).items():
            sv = (v if isinstance(v, str) else str(v)).strip()
            if pre and not sv.startswith(pre):
                continue
            if sv in exact:
                out.setdefault(sv, []).append(int(k))
            i = len(sv)
            while True:
                i = sv.rfind("/", 0, i)
                if i <= 0:
                    break
                anc = sv[:i]
                if anc in exact:
                    out.setdefault(anc, []).append(int(k))
        # An overlap means the check above leaks again: log it (never pass silently).
        seen = {}
        for path, ids in out.items():
            for i in ids:
                if i in seen:
                    if not self._inst_overlap_warned:
                        print(f"[GT] ⚠️ instance id {i} matched both {seen[i]} and {path} "
                              f"; masks will mix", flush=True)
                        self._inst_overlap_warned = True
                else:
                    seen[i] = path
        return out

    _BG_LABELS = {"", "background", "unlabelled", "unlabeled", "invalid"}

    def occlusion_from_annotator(self, d, imap, path, vectorized=None):
        """Pick the target's rows from the `occlusion` annotator for a pixel-exact occlusion ratio.

        No re-render, no raycast. Measured to match the mask ratio exactly:
        at cam_x 0.4, annotator 0.857 vs 1 - 112/783 = 0.857.

        Returns `(ratio, method)`, or `(None, reason)` if there is no row.
        `vectorized` (default: GT_OCCLUSION_VECTORIZED) picks the numpy path over the row loop.
        """
        oc = d.get("occlusion")
        arr = oc.get("data") if isinstance(oc, dict) else oc
        if not isinstance(arr, np.ndarray) or arr.dtype.names is None:
            return None, "occlusion not ready"
        if "instanceId" not in arr.dtype.names or "occlusionRatio" not in arr.dtype.names:
            return None, f"missing fields {arr.dtype.names}"
        iids = imap.get(path)
        if not iids:
            return None, "no instance id"
        # With multiple sub-meshes, use the occlusion of the **most visible piece**: values
        # differ per piece, and averaging mixes "shirt visible, shoes occluded".
        if (_OCCLUSION_VECTORIZED if vectorized is None else vectorized):
            # GT_OCCLUSION_VECTORIZED=1: numpy instead of a python loop over every instance
            # row. Same result (test_gt_occlusion.py); only the sim speed (RTF) changes.
            ids = np.fromiter((int(x) for x in iids), dtype=np.int64)
            m = np.isin(arr["instanceId"].astype(np.int64), ids)
            vals = arr["occlusionRatio"][m].astype(np.float64)
            if vals.size == 0:
                return None, "no target row"
            ok = vals[vals >= 0.0]
            if ok.size:
                return min(max(float(ok.min()), 0.0), 1.0), "annotator"
            return None, "off screen (-1)"
        best = None
        for row in arr:
            if int(row["instanceId"]) not in {int(x) for x in iids}:
                continue
            v = float(row["occlusionRatio"])
            if v < 0.0:
                continue
            best = v if best is None else min(best, v)
        if best is not None:
            return min(max(best, 0.0), 1.0), "annotator"
        for row in arr:
            if int(row["instanceId"]) not in {int(x) for x in iids}:
                continue
            v = float(row["occlusionRatio"])
            # ⚠️ The annotator reports "not on screen" as **-1** (measured). That is not a
            # ratio: recording it puts "occlusion -1" into the data and silently breaks rules
            # like `occl > 0.5`. Occlusion is defined only when on screen.
            if v < 0.0:
                return None, "off screen (-1)"
            return min(max(v, 0.0), 1.0), "annotator"
        return None, "no target row"

    def _read_camera_pose(self, stage):
        """Camera (world position, heading in degrees), or (None, None).

        ⚠️ **When it is called matters.** The value must be from the moment the frame was
        captured, not when the row is written: with the two-phase occlusion pass another
        render runs in between, and at low RTF several wall-clock seconds pass. While
        hovering that makes no difference, but in flight a later pose would be attached to
        a frame captured while the vehicle was turning fast (heading off by up to 163°).

        The heading formula was checked against known attitudes: 0.0° error over 16 poses
        (including pitch and roll).
        """
        try:
            m = self._camera_matrix(stage)
            if m is None:
                return None, None
            t = m.ExtractTranslation()
            pos = [float(t[0]), float(t[1]), float(t[2])]
            # The camera looks along its **-Z** (USD convention); world forward is the negated
            # third row. ENU yaw 0° = +X, counter-clockwise: same convention as the step log.
            f = m.GetRow3(2)
            yaw = math.degrees(math.atan2(-float(f[1]), -float(f[0]))) % 360.0
            return pos, yaw
        except Exception:  # noqa: BLE001
            return None, None

    # ── Pose reading ────────────────────────────────────────────────────
    def _sync_annotators(self, min_dist_m):
        """Enable/disable annotators by distance to targets. Returns whether they are on now.

        **One tick late.** Annotator data is filled by renders **after** enabling, so the
        enable tick is still empty. Hence the threshold is **10 m**, not δ (5 m): even at
        1.15 m/s there are 4+ seconds to reach 5 m, so several ticks run.

        Hysteresis (enable 10 m, disable 15 m) prevents toggling at the boundary.
        Measured round trip: attach ~200-380 ms, detach ~70-190 ms.
        """
        if self.mode != "measure" or not self._ann:
            return True
        if min_dist_m is None and not self._warm_on:
            return bool(self._ann_on)
        want = self._ann_on
        if self._warm_on:
            # Episode start: enable regardless of distance. `seen_at_start` comes from here.
            want = True
        elif min_dist_m is None:
            return bool(self._ann_on)
        elif not self._ann_on and min_dist_m <= self.near_on_m:
            want = True
        elif self._ann_on and min_dist_m > self.near_off_m:
            want = False
        if want == self._ann_on:
            return bool(self._ann_on)
        try:
            if self.gate_mode == "product":
                how = self.set_product_updates(want)
                if how is None:
                    # Toggle has no effect. **Do not fall back to detaching
                    # annotators**: that silently freezes the policy camera. Give up
                    # the gate and stay always on (slower but exact).
                    if not self._gate_warned:
                        self._gate_warned = True
                        print("[GT] ⚠️ render product toggle has no effect; **gate "
                              "disabled** (always on). Not falling back to detaching "
                              "annotators: that also detaches the policy camera's `rgb` "
                              "and freezes `latest.jpg` (measured).",
                              flush=True)
                    self._ann_on = True
                    return True
            else:
                # ⚠️ Detaches shared annotators: diagnostics only
                for a in self._ann.values():
                    a.attach(self._rp) if want else a.detach()
            self._ann_on = bool(want)
            if want:
                self._ann_settle = 1   # the enable tick is still empty
            self._ann_switches += 1
            d = "—" if min_dist_m is None else f"{min_dist_m:.1f} m"
            what = "product" if self.gate_mode == "product" else "annotators"
            print(f"[GT] {what} {'on' if want else 'off'} "
                  f"(target {d}, switch #{self._ann_switches})", flush=True)
        except Exception as exc:  # noqa: BLE001 — a failed switch must not block flight
            print(f"[GT] ⚠️ gate switch failed (continuing): {exc!r}", flush=True)
        return bool(self._ann_on)

    def _update_warm(self, now, cam_pos):
        """Whether we are in the episode-start window: latch on by camera altitude, release **by altitude**.

        Why not a timer: the first version was "latch below ground level, release after N
        seconds", but **takeoff measured 92 sim seconds** (PX4 lingered 88 s at 1.8 m and
        reached 2.35 m at 120 s). A fixed 20 s window released long before reaching
        altitude, so annotators were off at B's step-1 capture and `seen_at_start` silently
        became all False, leaving the very bug it was meant to fix.

        So release on an **observable condition**: staying above `warm_clear_z_m`
        **continuously** for `warm_hold_s` means takeoff is done, and B's step 1 follows
        right after. It tracks however long takeoff takes.

        `warm_max_s` is a safety cap: it stops the latch staying on for a whole episode
        that never reaches the clear altitude (e.g. low-altitude episodes). When the cap
        releases it, **warn once** (passing silently would leave `seen_at_start` wrong).

        Landing re-triggers the latch, which is harmless: that is near the target anyway.
        """
        if self.mode != "measure" or self.warm_mode == "off":
            self._warm_on = False
            return False
        if self.warm_mode == "signal":
            self._warm_on = self._hold_requested()
            return self._warm_on
        if cam_pos is None or now is None:
            return self._warm_on
        try:
            z_m = float(cam_pos[2]) * float(self._mpu)
            now = float(now)
        except (TypeError, IndexError, ValueError):
            return self._warm_on

        if z_m < self.warm_z_m:                 # near ground = episode start
            if not self._warm_on:
                self._warm_since = now
            self._warm_on = True
            self._warm_above = None
            return True
        if not self._warm_on:
            return False

        # Latched: check release condition
        if z_m >= self.warm_clear_z_m:
            if self._warm_above is None:
                self._warm_above = now
            elif now - self._warm_above >= self.warm_hold_s:
                self._warm_on = False           # takeoff done; step 1 has passed
                self._warm_above = None
                return False
        else:
            self._warm_above = None             # dropped again; continuity broken

        if (self._warm_since is not None
                and now - self._warm_since >= self.warm_max_s):
            self._warm_on = False
            self._warm_above = None
            if not self._warm_capped:
                self._warm_capped = True
                print(f"[GT] ⚠️ takeoff warm-up released by cap ({self.warm_max_s:g} sim s); "
                      f"vehicle never reached {self.warm_clear_z_m:g} m "
                      f"(now {z_m:.2f} m). `seen_at_start` may be wrong for this "
                      f"episode", flush=True)
            return False
        return True

    def set_product_updates(self, enabled):
        """Enable/disable the GT **render product itself**. Returns the API name that worked, or None.

        We used to attach/detach annotators. The NVIDIA SDG docs say to disable the
        **product**: once created it **renders every tick** even without `step()`. The two
        act on different layers: we remove the consumer, the docs stop the producer. So with
        all annotators detached the product may still be rendering.

        **The API shape varies by version.** So candidates are tried in order and **the one
        that worked is returned**: silently doing nothing while reporting "disabled" is a
        recurring bug class. If nothing works it returns `None`, and the caller must give
        up on measure mode.
        """
        if self._rp is None:
            return None
        want = bool(enabled)
        ht = getattr(self._rp, "hydra_texture", None)
        cands = [
            ("hydra_texture.set_updates_enabled",
             (lambda: ht.set_updates_enabled(want)) if ht is not None else None),
            ("rp.hydra_texture.set_enabled",
             (lambda: ht.set_enabled(want)) if ht is not None else None),
            ("rp.set_updates_enabled",
             (lambda: self._rp.set_updates_enabled(want))
             if hasattr(self._rp, "set_updates_enabled") else None),
        ]
        for name, fn in cands:
            if fn is None:
                continue
            try:
                fn()
                return name
            except Exception:  # noqa: BLE001 — try the next candidate
                continue
        return None

    def _hold_requested(self):
        """Whether B requested "GT pixels needed now": is the file mtime fresh.

        **Measured in wall clock.** B (a separate process) writes this file and does not know
        the sim clock. `hold_s` is the slack between B writing the request and its first capture.

        A missing/unreadable file means **no request**: running with an older B must not
        crash here. `seen_at_start` may then be wrong (step-1 `mask_px` is empty).
        """
        if not self.hold_path:
            return False
        try:
            age = time.time() - os.path.getmtime(self.hold_path)
        except OSError:
            return False
        return bool(0.0 <= age <= float(self.hold_s))

    def _geom_min_dist(self, stage, cam_pos):
        """Distance (m) to the nearest target from **geometry only**, no render. None if none."""
        if not cam_pos:
            return None
        best = None
        for path in self.targets:
            wp = self._world_pos(stage, path)
            if not wp:
                continue
            d = float(np.linalg.norm(np.array(wp) - np.array(cam_pos)) * self._mpu)
            best = d if best is None else min(best, d)
        return best

    def _fabric_on(self):
        """Whether physics output goes to Fabric.

        **Checks two things directly.** `SimulationManager.is_fabric_enabled(enable)`
        requires an argument it never uses (a leftover of the Isaac 5.1 signature); calling it
        without one raises `TypeError`. Measured: that exception was swallowed by `except`,
        **Fabric was on but judged off**, and all 91 GT rows came out `pose_source: usd`.
        So we check the extension and the setting directly instead.

        ⚠️ **A detection failure is never silently treated as "off".** Otherwise a Fabric run
        would read USD (= spawn position) without leaving any trace.
        """
        ext = sett = None
        try:
            import omni.kit.app

            ext = bool(omni.kit.app.get_app().get_extension_manager()
                       .is_extension_enabled("omni.physx.fabric"))
        except Exception:  # noqa: BLE001
            ext = None
        try:
            import carb.settings

            # `enable_fabric` sets `/physics/updateToUsd = not enable`
            v = carb.settings.get_settings().get("/physics/updateToUsd")
            sett = (v is False)
        except Exception:  # noqa: BLE001
            sett = None
        if ext is None and sett is None:
            if not self._pose_warned:
                self._pose_warned = True
                print("[GT] ⚠️⚠️ **cannot determine** whether Fabric is on; reading USD, but "
                      "if Fabric was on, this run's GT is unusable", flush=True)
            return False
        return bool(ext) or bool(sett)

    def _fabric_matrix(self):
        """Read the camera world transform from Fabric as a `Gf.Matrix4d`. None on failure.

        `XFormPrim.get_world_poses(usd=False)` is the supported path for reading Fabric.
        **Why convert back to a matrix**: the heading formula relies on `m.GetRow3(2)` (world
        direction of local +Z), validated at 0.0° error over 16 known attitudes.
        Deriving angles anew from the quaternion would lose that validation, so the same
        formula is reused.
        """
        from pxr import Gf

        if self._pose_view is False:
            return None
        try:
            if self._pose_view is None:
                from isaacsim.core.prims import XFormPrim

                self._pose_view = XFormPrim(self.camera_prim)
            pos, quat = self._pose_view.get_world_poses(usd=False)
            p = [float(v) for v in pos[0]]
            q = [float(v) for v in quat[0]]        # (w, x, y, z)
            m = Gf.Matrix4d()
            m.SetRotate(Gf.Quatd(q[0], Gf.Vec3d(q[1], q[2], q[3])))
            m.SetTranslateOnly(Gf.Vec3d(p[0], p[1], p[2]))
            return m
        except Exception as exc:  # noqa: BLE001
            self._pose_view = False
            print(f"[GT] ⚠️ Fabric pose read failed; falling back to USD: {exc!r}", flush=True)
            return None

    def _usd_matrix(self, stage):
        from pxr import Usd, UsdGeom

        xf = UsdGeom.Xformable(stage.GetPrimAtPath(self.camera_prim))
        return xf.ComputeLocalToWorldTransform(Usd.TimeCode.Default())

    def _camera_matrix(self, stage):
        """Camera world transform. **Reads from Fabric when Fabric is on.**

        ⚠️ Without this branch, enabling Fabric makes **every GT row the spawn position**.
        The view looks fine (rendering reads Fabric) and the VLM works, so verdicts all flip
        to failure **without a single error**: the worst kind of bug.

        For the first few ticks **both paths are read and the difference is logged.** A
        growing difference shows "USD is frozen"; zero means Fabric is not actually on.
        """
        if not self._fabric_on():
            self._pose_source = "usd"
            return self._usd_matrix(stage)
        m = self._fabric_matrix()
        if m is None:
            # **Do not fail open.** If Fabric is on but unreadable, USD is frozen, so
            # announce it loudly instead of using it silently.
            if not self._pose_warned:
                self._pose_warned = True
                print("[GT] ⚠️⚠️ Fabric is on but the pose is unreadable; USD is frozen "
                      "at the spawn position. **This run's GT is unusable**", flush=True)
            self._pose_source = "usd-stale"
            return self._usd_matrix(stage)
        self._pose_source = "fabric"
        if self._pose_cmp_left > 0:
            self._pose_cmp_left -= 1
            try:
                u = self._usd_matrix(stage).ExtractTranslation()
                f = m.ExtractTranslation()
                d = max(abs(float(u[i]) - float(f[i])) for i in range(3))
                self._pose_cmp_max = max(self._pose_cmp_max, d)
                if self._pose_cmp_left == 0:
                    # **Use a threshold.** An 8 mm difference was once read as "USD is
                    # frozen", which was wrong: the vehicle was simply still on the ground.
                    # Overstated diagnostics lead the next reader to wrong conclusions.
                    d_m = self._pose_cmp_max * self._mpu
                    if d_m >= 0.5:
                        note = "(USD lags; the Fabric branch is doing its job)"
                    elif d_m > 0:
                        note = "(small difference; vehicle has barely moved yet)"
                    else:
                        note = "(zero difference; Fabric may not actually be on)"
                    print(f"[GT] using Fabric pose; max difference vs USD over first 20 ticks "
                          f"{d_m:.3f} m {note}", flush=True)
            except Exception:  # noqa: BLE001 — a failed comparison must not block flight
                pass
        return m

    def occlusion_by_raycast(self, stage, path, cam_pos):
        """Measure occlusion from raycast samples, **without re-rendering**, per target.

        Why the render method was dropped: validated method (c) (hide occluders and
        re-render the unoccluded silhouette) does not hold at Brownstone scale.
        `MakeInvisible` on `/World/layout` (38,942 meshes) yields **a uniform sky with the
        whole scene gone** (mean luma 217), and even unhidden targets do not render. Also,
        that frame's instance `idToLabels` all come back `INVALID` (both measured).

        Instead, sample points from the target bbox and cast rays from the camera. Targets
        have `CollisionAPI`, so we can judge by **hit prim path**:

            hit the target       -> visible
            hit something else   -> occluded
            hit nothing          -> dropped from samples (grazed an edge)

        Returns `(ratio, {visible, occluded, missed})`, or `(None, ...)` if no sample is
        valid. This is a **sample-based approximation**, not pixel area; resolution is 1/N.
        """
        import math as _m

        if self._ray is None or cam_pos is None:
            return None, None
        pts = self._sample_points(stage, path)
        if not pts:
            return None, None
        vis = occ = miss = 0
        hits = {}
        for p in pts:
            d = [p[i] - cam_pos[i] for i in range(3)]
            L = _m.sqrt(sum(c * c for c in d))
            if L <= 1e-6:
                continue
            dn = [c / L for c in d]
            # ⚠️ Rays must start **offset** from the camera. The camera prim is a child of
            # the vehicle body (offset 0), so the origin is **inside** the vehicle's own
            # collider, and raycast_closest would hit its own body at distance 0, making
            # everything "occluded".
            o = [cam_pos[i] + dn[i] * self.ray_skip_m for i in range(3)]
            reach = L - self.ray_skip_m
            if reach <= 0:
                continue
            r = self._ray(o, dn, reach * 1.02)
            if not r:
                miss += 1
                continue
            hit = str(r.get("collision") or r.get("rigidBody") or "")
            key = "/".join(hit.split("/")[:4]) or "?"
            hits[key] = hits.get(key, 0) + 1
            if hit.startswith(path):
                vis += 1
            else:
                occ += 1
        total = vis + occ
        counts = {"visible": vis, "occluded": occ, "missed": miss}
        if self.debug_luma and path == self._occl_target:
            self._occl_dbg = hits
        if total == 0:
            return None, counts
        return occ / total, counts

    def _sample_points(self, stage, path):
        """Sample points of the target bbox: 8 corners + 6 face centers + center = 15.

        Samples are pulled toward the center by `sample_inset`. bbox corners lie outside the
        object, so rays just graze (measured 6 of 15 samples "grazed"), and **bottom-face
        samples touch the ground and hit the pavement**, counting as occluded. Pulling them
        inward puts samples inside the object, so rays hit its surface first.
        """
        from pxr import Usd, UsdGeom

        pr = stage.GetPrimAtPath(path)
        if not pr or not pr.IsValid():
            return []
        rng = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
        ).ComputeWorldBound(pr).ComputeAlignedRange()
        if rng.IsEmpty():
            return []
        lo, hi = rng.GetMin(), rng.GetMax()
        cx = [(lo[i] + hi[i]) / 2.0 for i in range(3)]
        k = 1.0 - float(self.sample_inset)          # shrink factor from center toward corners
        alo = [cx[i] + (lo[i] - cx[i]) * k for i in range(3)]
        ahi = [cx[i] + (hi[i] - cx[i]) * k for i in range(3)]
        pts = [tuple(cx)]
        for ix in (alo[0], ahi[0]):
            for iy in (alo[1], ahi[1]):
                for iz in (alo[2], ahi[2]):
                    pts.append((ix, iy, iz))
        for i in range(3):
            for v in (alo[i], ahi[i]):
                q = list(cx)
                q[i] = v
                pts.append(tuple(q))
        return pts

    def _silhouette_px(self, d2):
        """Pixel count of "remaining labelled geometry" in the silhouette frame.

        **Cannot count with instance.** Hiding the env root makes the instance annotator's
        `idToLabels` all `INVALID` (measured): instance ids are assigned per render and
        visibility changes invalidate the table. Semantic labels are authored in USD and
        survive in that frame.

        The silhouette frame keeps **only the measured target** (other targets and the
        environment are hidden), so the non-background pixel count is its unoccluded area.
        Returns `(px, method)`, or `(None, reason)` if it cannot be counted.
        """
        sem = d2.get("semantic_segmentation") or {}
        sids = sem.get("data")
        info = sem.get("info") or {}
        m = info.get("idToLabels") or {}
        if not isinstance(sids, np.ndarray) or not m:
            return None, "no semantic"
        keep = []
        for k, v in m.items():
            cls = (v.get("class") if isinstance(v, dict) else str(v)) or ""
            if str(cls).strip().lower() in self._BG_LABELS:
                continue
            try:
                keep.append(int(k))
            except (TypeError, ValueError):
                continue
        if not keep:
            return 0, "semantic(all background)"
        return int(np.isin(sids, keep).sum()), "semantic"

    def _sem_local(self, ids, info):
        m = info.get("idToLabels") or {}
        out = np.zeros(ids.shape, np.uint8)
        for k, v in m.items():
            cls = (v.get("class") if isinstance(v, dict) else str(v)) or "UNLABELLED"
            if cls not in self._sem_classes:
                self._sem_classes[cls] = len(self._sem_classes) + 1
            out[ids == int(k)] = self._sem_classes[cls]
        return out

    def _bbox_new_tick(self):
        """Tick start: clear caches. **Targets move, so stale values otherwise.**"""
        self._wpos_tick = {}
        if self._bbox is not None:
            try:
                self._bbox.Clear()
            except Exception:  # noqa: BLE001 — if it cannot be cleared, rebuild it
                self._bbox = None

    def _world_pos(self, stage, path):
        """Target world center (stage units). **Computed once** per tick.

        ⚠️ Previously a new `UsdGeom.BBoxCache` was **built on every call**. It is meant to
        be built once and reused; rebuilding makes its internal cache useless. With 61
        targets that was 61 per tick (122 in measure mode, called by both
        `_geom_min_dist` and `_emit_geom`).
        """
        if path in self._wpos_tick:
            return self._wpos_tick[path]
        from pxr import Usd, UsdGeom
        pr = stage.GetPrimAtPath(path)
        if not pr:
            self._wpos_tick[path] = None
            return None
        try:
            if self._bbox is None:
                self._bbox = UsdGeom.BBoxCache(
                    Usd.TimeCode.Default(),
                    [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
                )
            box = self._bbox.ComputeWorldBound(pr).ComputeAlignedRange()
            c = box.GetMidpoint()
            out = [float(c[0]), float(c[1]), float(c[2])]
        except Exception:  # noqa: BLE001 — logging only, give up quietly
            out = None
        self._wpos_tick[path] = out
        return out

    # ── One step ────────────────────────────────────────────────────────
    def tick(self, force=False):
        """Called on every render from A's loop. Has its own rate limiter and re-entry guard."""
        if self._busy:
            return False
        now = time.time()
        # Silhouette phases bypass the rate limiter: they must read on the next render,
        # and the environment stays hidden until then.
        if self._phase == 0 and not force and self.min_dt and (now - self._last) < self.min_dt:
            return False
        self._busy = True
        try:
            return self._tick(now)
        except Exception as exc:  # noqa: BLE001 — GT logging must not block flight
            self._restore_env()
            if not self._warned_miss:
                print(f"[GT] tick failed, continuing: {exc!r}", flush=True)
                self._warned_miss = True
            return False
        finally:
            self._busy = False

    def _hide_for_silhouette(self, stage):
        """Hide the env root and **targets other than the measured one**.

        Only with the measured target left does the silhouette frame's non-background pixel
        count equal its unoccluded area; other targets would be counted too.
        """
        from pxr import UsdGeom

        self._hidden = []
        for path in [self.env_root] + [p for p in self.targets
                                       if p != self._occl_target]:
            pr = stage.GetPrimAtPath(path) if path else None
            if not pr or not pr.IsValid():
                continue
            img = UsdGeom.Imageable(pr)
            try:
                img.MakeInvisible()
                self._hidden.append(img)
            except Exception:  # noqa: BLE001
                pass

    def _restore_env(self):
        """Restore hidden prims; leaving them hidden would fly with the scene gone."""
        self._phase = 0
        self._settle = 0
        for img in self._hidden:
            try:
                img.MakeVisible()
            except Exception:  # noqa: BLE001
                pass
        self._hidden = []
        if self._env_img is not None:
            try:
                self._env_img.MakeVisible()
            except Exception:  # noqa: BLE001
                pass

    def _tick(self, now):
        import omni.usd
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            return False
        self._bbox_new_tick()
        if self._ann is None:
            # The camera prim is created by `setup_camera` just before play; look it up lazily
            if self.camera_prim is None:
                self.camera_prim = resolve_camera(stage, self.camera_index)
                if self.camera_prim is None:
                    self._miss_streak += 1
                    if self._miss_streak == 300:
                        print("[GT] fpv_cam not found; check whether it was launched with "
                              "`--no-camera`. GT keeps waiting", flush=True)
                    return False
            if self.target_pattern and not self.targets:
                self.targets = targets_from_pattern(stage, self.target_pattern)
                print(f"[GT] {len(self.targets)} targets (pattern {self.target_pattern!r})",
                      flush=True)
                for t in self.targets[:8]:
                    print(f"     {t}", flush=True)
                if not self.targets:
                    print("[GT] ⚠️ no prim matches the pattern; target GT stays empty and "
                          "only frames are recorded", flush=True)
            if self.want_occlusion and self.env_root is None:
                self.env_root = discover_env_root(stage)
                self.occlusion = self.env_root is not None
                print(f"[GT] occlusion env root {self.env_root}", flush=True)
            self._miss_streak = 0
            self._setup()

        # ── Hide settling: let renders pass (annotators lag one frame)
        if self._phase == 1:
            self._settle -= 1
            if self._settle > 0:
                return False
            self._phase = 2
            return False

        # ── Silhouette read -> restore -> settle
        if self._phase == 2:
            try:
                d2, miss2 = self._read()
            finally:
                self._restore_env()
            free = {}
            if miss2 is None:
                px, how = self._silhouette_px(d2)
                self._free_method = how
                if px is not None and self._occl_target:
                    free[self._occl_target] = px
                imap2 = self._inst_map(d2["instance_id_segmentation"].get("info", {}))
                if self.debug_luma:
                    sample = list((d2.get(_ANNOTATOR_SEMANTIC, {}).get("info", {})
                                   .get("idToLabels") or {}).items())[:5]
                    print(f"[GT] luma normal {float(np.mean(self._pending['rgb'])):.1f}, "
                          f"silhouette {float(np.mean(d2['rgb'][:, :, :3])):.1f}, "
                          f"silhouette free_px {px} ({how}), "
                          f"instance match {len(imap2)} targets/"
                          f"{sum(len(v) for v in imap2.values())} pieces, semantic {sample}", flush=True)
            elif not self._warned_free:
                print(f"[GT] silhouette frame not ready ({miss2}); occlusion for this tick "
                      "left empty", flush=True)
                self._warned_free = True
            self._emit(stage, self._pending, free)
            self._phase = 0
            self._settle = self.settle_renders
            return True

        if self._settle > 0:            # right after restore: let renders pass
            self._settle -= 1
            return False

        # ── Measure mode: measure distance geometrically, decide whether to attach ─
        # Uses only render-free quantities (camera pose, target world positions, distance),
        # so it runs while detached. Since the judge is `ok = (dist ≤ δ) AND visible`,
        # **being on within δ preserves SR.**
        if self.mode == "measure":
            cam_pos_g, cam_yaw_g = self._read_camera_pose(stage)
            # Run the latch **before distance**: in the episode-start window, attach regardless
            self._update_warm(now, cam_pos_g)
            on = self._sync_annotators(self._geom_min_dist(stage, cam_pos_g))
            if on and self._ann_settle > 0:
                # Just attached: data not filled yet. Reading gives all zeros, which would be
                # recorded as "not visible". Record this tick as **not measured**.
                self._ann_settle -= 1
                on = False
            if not on:
                self._emit_geom(stage, now, cam_pos_g, cam_yaw_g)
                self._last = now
                return True

        d, miss = self._read()
        if miss is not None:
            self._miss_streak += 1
            if self._miss_streak in (30, 300) and not self.orchestrator_step:
                print(f"[GT] annotator not ready for {self._miss_streak} ticks in a row ({miss}); "
                      "if live rendering does not fill it in this environment, retry with "
                      "--gt-orchestrator-step (render timing will jitter)", flush=True)
            return False
        self._miss_streak = 0
        self._last = now

        # ⚠️ **Must copy.** `annotator.get_data()` returns a view of an internal buffer that
        # the next render overwrites. In the two-phase occlusion pass emit happens one
        # render later, so without copying the saved RGB is **the env-hidden frame**
        # (measured: uniform sky at mean luma 217, depth all inf).
        snap = {
            "t": now,
            "rgb": np.array(d["rgb"][:, :, :3], copy=True),
            # `distance_to_image_plane` is **already meters**; multiplying by metersPerUnit makes it 100x too small
            "dep": np.array(d["distance_to_image_plane"], dtype=np.float32, copy=True),
            "ids": np.array(d["instance_id_segmentation"]["data"], copy=True),
            "imap": self._inst_map(d["instance_id_segmentation"].get("info", {})),
            # semantic is read **only when attached**: it is dropped unless saving, render
            # occlusion or debug needs it (`_annotators_for`). None if absent; consumers skip it.
            "sids": (np.array(d[_ANNOTATOR_SEMANTIC]["data"], copy=True)
                     if _ANNOTATOR_SEMANTIC in d else None),
            "sinfo": (d[_ANNOTATOR_SEMANTIC].get("info", {})
                      if _ANNOTATOR_SEMANTIC in d else {}),
            "occlusion": _copy_occlusion(d.get("occlusion")),
        }
        # **Store the camera pose at capture time** (see `_read_camera_pose`).
        snap["cam_pos"], snap["cam_yaw"] = self._read_camera_pose(stage)

        if not self.occlusion:
            self._emit(stage, snap, {})
            return True

        # Occlusion (c). Values change only **after a new render** following a visibility
        # change; re-reading right after toggling returns the same frame.
        self._hide_for_silhouette(stage)
        if self.orchestrator_step:
            # `orchestrator.step()` forces a render, so this finishes within one tick
            try:
                d2, miss2 = self._read()
            finally:
                self._restore_env()
            free = {}
            if miss2 is None:
                px, how = self._silhouette_px(d2)
                self._free_method = how
                if px is not None and self._occl_target:
                    free[self._occl_target] = px
            self._emit(stage, snap, free)
            return True
        # Live rendering: wait for hiding to reach the render, then read the silhouette.
        # One tick uses `2*settle_renders + 1` renders, which is why it is off by default.
        self._pending = snap
        self._phase = 1
        self._settle = self.settle_renders
        return False

    def _emit_geom(self, stage, now, cam_pos, cam_yaw):
        """Row written **without rendering**: measure mode while targets are far.

        Contains: camera pose, target world positions, distances. **Nothing pixel-derived**
        (`mask_px`, `visible`, occlusion, luma). Marked `ann: False` so it can be filtered
        later; without the marker "mask 0, not visible" and "not measured" are
        indistinguishable.
        """
        tg = {}
        for path in self.targets:
            wp = self._world_pos(stage, path)
            dist = (float(np.linalg.norm(np.array(wp) - np.array(cam_pos)) * self._mpu)
                    if (wp and cam_pos) else None)
            tg[path] = {
                "xyz_m": ([round(v * self._mpu, 3) for v in wp] if wp else None),
                "dist_m": (round(dist, 3) if dist is not None else None),
                "mask_px": None, "visible": None, "in_fov": None,
                "occlusion_ratio": None, "occlusion_method": None,
                "point": None, "depth_at_point_m": None,
                "luma": None, "mean_rgb": None,
            }
        row = {
            "i": self._n,
            "t": round(now, 6),
            "sim_time_s": self._sim_time(),
            "camera_xyz_m": ([round(v * self._mpu, 3) for v in cam_pos]
                             if cam_pos else None),
            "yaw_deg": (None if cam_yaw is None else round(cam_yaw, 2)),
            "pose_source": self._pose_source,
            "gt_mode": self.mode,
            "ann_warm": bool(self._warm_on),
            "ann": False,
            "frames": {},
            "targets": tg,
            "n_visible": None,
            "meta": self.target_meta,
        }
        self._n += 1
        if self._jl is not None:
            self._jl.write(json.dumps(row, ensure_ascii=False) + "\n")
            self._jl.flush()
        self._write_live(row)
        return row

    def _emit(self, stage, snap, free):
        """Turn one snapshot into a row and write it."""
        if snap is None:
            return
        now = snap["t"]
        rgb, dep = snap["rgb"], snap["dep"]
        ids, imap = snap["ids"], snap["imap"]
        sids, sinfo = snap["sids"], snap["sinfo"]

        # **Use the values stored in the snapshot**: the pose at the moment of capture.
        # If absent (old path), read it now.
        cam_pos = snap.get("cam_pos")
        cam_yaw = snap.get("cam_yaw")
        if cam_pos is None and cam_yaw is None:
            cam_pos, cam_yaw = self._read_camera_pose(stage)

        tag = f"{self._n:06d}"
        rel = {}
        if self.save_frames:
            import cv2
        if "rgb" in self.save_frames:
            cv2.imwrite(f"{self.frames_dir}/{tag}_rgb.png", rgb[:, :, ::-1])
            rel["rgb"] = f"gt_frames/{tag}_rgb.png"
        if "depth" in self.save_frames:
            # float16 saves 77%. Sky comes as `inf` and is preserved
            np.save(f"{self.frames_dir}/{tag}_depth.npy", dep.astype(np.float16))
            rel["depth"] = f"gt_frames/{tag}_depth.npy"
        if "instance" in self.save_frames:
            # Build the label map once; per-target `np.isin` would scan the frame N times
            paths_ = [p_ for p_, _ in sorted(imap.items())]
            lab_, order_ = label_map(ids, imap, paths_)
            loc = (lab_.astype(np.uint8) if lab_ is not None
                   else np.zeros(ids.shape, np.uint8))
            lmap = {str(j): p_ for j, p_ in enumerate(order_, start=1)}
            cv2.imwrite(f"{self.frames_dir}/{tag}_inst.png", loc)
            rel["instance_id"] = f"gt_frames/{tag}_inst.png"
            rel["instance_map"] = lmap
        if "semantic" in self.save_frames and sids is not None:
            cv2.imwrite(f"{self.frames_dir}/{tag}_sem.png", self._sem_local(sids, sinfo))
            rel["semantic"] = f"gt_frames/{tag}_sem.png"

        tg = {}
        n_visible = 0
        # **Scan the frame once.** Per-target `np.isin` re-read 61 x 307,200 px every tick
        # (89.5 ms -> 8.7 ms). Values are identical; see `target_pixel_stats` above.
        stats = target_pixel_stats(ids, rgb, dep, imap, list(self.targets))
        for path in self.targets:
            st_ = stats[path]
            px = st_["px"]
            cx, cy, dat = st_["cx"], st_["cy"], st_["depth"]
            lum, mean_rgb = st_["lum"], st_["mean_rgb"]
            if px > 0:
                # **Mean luma of the target pixels.** "Visible" does not mean
                # "recognizable": in the indoor warehouse an episode had a fine mask but frame
                # mean luma 11 (the nurse stood in an unlit area). If the label ("nurse in
                # green scrubs") cannot be recovered from pixels, the item violates design
                # principle (1). So we measure **the target's pixels**, not the whole frame,
                # since that is where the label comes from.
                # **Mean RGB is kept too.** Checking afterwards whether the color in the
                # phrase is readable from pixels needs more than luma. Measured:
                # paint washed out in rendering, `red cargo box` was (248,199,201), whitish pink.
                pass            # both already computed by `target_pixel_stats`
            wp = self._world_pos(stage, path)
            dist = (float(np.linalg.norm(np.array(wp) - np.array(cam_pos)) * self._mpu)
                    if (wp and cam_pos) else None)
            fp = free.get(path)
            occl = (1.0 - px / fp) if (fp and fp > 0) else None
            occl_counts = None
            how = "render(c)" if fp else None
            if px == 0:
                # Occlusion is undefined off screen. `in_fov` and `visible` already carry
                # that, so no number is invented here.
                occl, how = None, None
            elif occl is None:
                # (1) annotator: pixel-exact, no extra render
                occl, why = self.occlusion_from_annotator(snap, imap, path)
                if occl is not None:
                    how = why
                elif self._ray is not None:
                    # (2) raycast samples: fallback when live rendering does not fill the annotator
                    occl, occl_counts = self.occlusion_by_raycast(stage, path, cam_pos)
                    if occl is not None:
                        how = "raycast"
                    if not self._occl_ann_warned:
                        print(f"[GT] occlusion annotator unavailable ({why}); "
                              "using raycast samples instead", flush=True)
                        self._occl_ann_warned = True
            vis = px >= self.visible_min_px
            n_visible += int(vis)
            tg[path] = {
                "xyz_m": ([round(v * self._mpu, 3) for v in wp] if wp else None),
                "dist_m": (round(dist, 3) if dist is not None else None),
                "mask_px": px,
                "mask_px_unoccluded": fp,
                "free_px_method": (self._free_method if fp is not None else None),
                "occlusion_ratio": (round(occl, 4) if occl is not None else None),
                "occlusion_method": how,
                "occlusion_samples": occl_counts,
                "visible": bool(vis),
                "in_fov": bool(px > 0),
                # `point` uses the 0-1000 scale of action space v3
                "point": ([round(cx / self.W * 1000, 1), round(cy / self.H * 1000, 1)]
                          if px > 0 else None),
                "depth_at_point_m": (round(dat, 3) if dat is not None else None),
                # 0-255. `None` means off screen, nothing to measure
                "luma": (round(lum, 1) if lum is not None else None),
                "mean_rgb": mean_rgb,
                "think_occlusion": bool(occl is not None and occl > 0.5),
                "think_scale": bool(0 < px < self.think_min_px),
            }

        row = {
            "i": self._n,
            "t": round(now, 6),                       # wall clock: join key
            "sim_time_s": self._sim_time(),           # Isaac world clock (for auditing)
            "camera_xyz_m": ([round(v * self._mpu, 3) for v in cam_pos]
                             if cam_pos else None),
            # **Heading is recorded too.** Two places missed it: the trajectory GIF read it
            # separately from `steps.jsonl` to draw the FOV, and the oracle needs its own
            # heading to remember the "last seen direction". Heading is a camera observation,
            # so it belongs in GT. Same convention as the step log: ENU, yaw 0° = +X, CCW.
            "yaw_deg": (None if cam_yaw is None else round(cam_yaw, 2)),
            # **Where the pose was read from.** With Fabric on, the USD vehicle pose freezes;
            # reading it naively puts every row at the spawn position and flips verdicts
            # **without a single error**. Kept per row so it can be filtered afterwards:
            # a run with even one `usd-stale` row is discarded.
            "pose_source": self._pose_source,
            # **Whether the row was captured with annotators on.** Measure mode detaches them
            # when far from targets, so `mask_px` is empty; using such rows for analysis gives
            # "processed 0, passed". The verifier checks this.
            "gt_mode": self.mode,
            "ann_warm": bool(self._warm_on),
            "ann": True,
            "frames": rel,
            # Whole-frame mean luma. Target luma alone cannot separate "target is in shade"
            # from "whole scene is dark"; both are needed.
            "luma_frame": round(float(_LUMA_W.dot(
                rgb[:, :, :3].reshape(-1, 3).astype(np.float32).T).mean()), 1),
            "targets": tg,
            "n_visible": n_visible,
            "think_decoy": bool(n_visible >= 2),      # signal only; the rule lives elsewhere
        }
        self._write_live(row)
        self._jl.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._jl.flush()
        os.fsync(self._jl.fileno())                   # don't lose everything on a crash
        self._n += 1
        self._pending = None
        if self._n == 1 or self._n % 20 == 0:
            extra = ""
            if self._occl_dbg:
                top = sorted(self._occl_dbg.items(), key=lambda kv: -kv[1])[:3]
                extra = ", ray hits " + ", ".join(f"{k}×{v}" for k, v in top)
            print(f"[GT] {self._n} ticks, visible targets {n_visible}/{len(self.targets)}"
                  + extra, flush=True)

    @property
    def ticks(self):
        return self._n

    def _write_live(self, row):
        """Atomically replace the "GT for the current frame" file read by B.

        A writes the camera frame and GT on **the same tick** (`--gt-tick frame`), so if B
        reads this file right after `latest.jpg` it gets GT from the same render. For the
        same reason frame saving must not kill the sim, failures are reported once and ignored.
        """
        if not self.live_path:
            return
        try:
            live = dict(row)
            live.pop("frames", None)        # B does not need file paths
            live["camera"] = self._camera_meta
            live["meta"] = self.target_meta
            tmp = self.live_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(live, fh, ensure_ascii=False)
            os.replace(tmp, self.live_path)
        except Exception as exc:  # noqa: BLE001
            if not self._live_warned:
                print(f"[GT] live sidecar write failed, ignoring: {exc!r}", flush=True)
                self._live_warned = True

    def _sim_time(self):
        try:
            import omni.timeline
            tl = omni.timeline.get_timeline_interface()
            return round(float(tl.get_current_time()), 6)
        except Exception:  # noqa: BLE001
            return None

    def set_ray(self, ray_fn):
        """Set the physics raycast. It can only be created after play, so the launcher passes it later."""
        self._ray = ray_fn
        print("[GT] raycast occlusion enabled", flush=True)
        return self

    def close(self):
        self._restore_env()     # never exit with the environment hidden
        if self._jl is not None:
            try:
                meta_path = os.path.join(self.out_dir, "gt_meta.json")
                meta = json.load(open(meta_path, encoding="utf-8"))
                meta["semantic_classes"] = self._sem_classes
                meta["targets_meta"] = self.target_meta
                meta["n_ticks"] = self._n
                json.dump(meta, open(meta_path, "w", encoding="utf-8"),
                          ensure_ascii=False, indent=1)
            except Exception:  # noqa: BLE001
                pass
            self._jl.close()
            self._jl = None
        print(f"[GT] sidecar closed: {self._n} ticks", flush=True)


def install_on_frame(stream, rig_cls=None, index=0, verbose=True):
    """Capture GT **at the moment A replaces the camera frame file**.

    Why this beats the `World.step` hook: B sees the frame A writes to `latest.jpg`. If GT
    is captured independently of that, the image and GT come from different renders and
    join error becomes **half the GT tick interval**: measured **median 450 ms** at RTF 0.11
    and 10 Hz rendering (32 cm of drone motion at 0.72 m/s).

    Wrapping `VehicleRig.write_frame` makes each GT tick come from the same render as that
    frame, 1:1. `write_frame` is called only on rendered steps, every `steps_per_frame`,
    i.e. once per frame B can actually read.

    `index` selects which vehicle's camera to follow (must match the sidecar's camera).
    """
    if rig_cls is None:
        import pegasus_sim
        rig_cls = pegasus_sim.VehicleRig
    orig = rig_cls.write_frame

    def write_frame(self, *a, **k):
        out = orig(self, *a, **k)
        if getattr(self, "index", 0) == index:
            stream.tick(force=True)          # same render as the frame; ignore rate limiter
        return out

    rig_cls.write_frame = write_frame
    if verbose:
        print(f"[GT] VehicleRig.write_frame hook installed (vehicle {index}); "
              "GT ticks aligned to the frames B reads", flush=True)
    return stream


def install(stream, world_cls=None, only_when_playing=True, warn_after=600):
    """Insert `tick()` after the main loop's `world.step(render=...)` without modifying it.

    ⚠️ Wrapping `simulation_app.update()` **never runs a single tick** (measured). The
    `pegasus_sim` loop (`PegasusSimApp.run`) only checks `simulation_app.is_running()` and
    its body only calls `self.world.step(render=do_render)`. So `World.step` is what to wrap.

    The World instance is created inside `main()`, so we wrap at **class level**, which
    also applies to instances created later.
    """
    if world_cls is None:
        from isaacsim.core.api.world import World as world_cls
    orig = world_cls.step
    state = {"renders": 0, "warned": False}

    def step(self, render=True, *a, **k):
        out = orig(self, render, *a, **k)
        if not render:
            return out
        state["renders"] += 1
        if only_when_playing:
            try:
                import omni.timeline
                if not omni.timeline.get_timeline_interface().is_playing():
                    return out
            except Exception:  # noqa: BLE001
                return out
        stream.tick()
        if (not state["warned"] and stream.ticks == 0
                and state["renders"] >= warn_after):
            state["warned"] = True
            print(f"[GT] {state['renders']} renders but 0 GT ticks; "
                  "check the camera prim or annotator readiness", flush=True)
        return out

    world_cls.step = step
    print("[GT] World.step hook installed", flush=True)
    return stream


def targets_from_pattern(stage, pattern, limit=32):
    """Pick target prims by name regex; used for smoke tests and per-scene target selection."""
    from pxr import Usd, UsdGeom
    rx = re.compile(pattern, re.I)
    out = []
    for p in Usd.PrimRange.Stage(stage, Usd.PrimAllPrimsPredicate):
        if len(out) >= limit:
            break
        if rx.search(str(p.GetPath())) and (p.IsA(UsdGeom.Xformable)):
            out.append(str(p.GetPath()))
    return out
