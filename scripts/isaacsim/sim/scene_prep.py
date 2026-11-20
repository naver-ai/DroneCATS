# DroneCATS
# Copyright (c) 2026-present NAVER Cloud Corp.
# Apache-2.0

"""Scene preparation: group labelling, collision, ground probes, path validity.

Collects rules confirmed by measurement in reusable form.
"""
import math
import re

_RULES = [
    ("vegetation",  r"vegetation|tree|shrub|acacia|beech|fir|oak|grass|planter"),
    ("streetlight", r"streetlight|streetlamp|\blamp\b"),
    ("road",        r"road|hardscape|pathway|pavement|sidewalk|asphalt|\bground\b|street"),
    ("building",    r"brownstone|massing|bldg|building|facade|revit|tower"),
    ("prop",        r"bench|trash|hydrant|mailbox|fence|shelter|kalmar"),
]


def label_groups(stage, root="/World", max_groups=400, coarse_depth=3, fine_depth=6):
    """Attach semantic labels **to upper-level groups only**, in two passes.

    Labelling leaf meshes (8,704) makes the seg annotator return empty arrays (measured).
    Labelling the `/World` root is also wrong: the whole scene becomes one semantic instance.

    **The key is USD's "nearest ancestor wins" semantic resolution rule**, so class
    convergence is not required. Buildings and vegetation are mixed throughout the scene
    (`Reference_Brownstone5Row_Yard` contains trees); requiring convergence rarely holds at
    upper levels and labels balloon into the hundreds (measured: 531 candidates, over the
    400 cap).

    (1) **Coarse**: attach the dominant class to large shallow containers (mostly `building`).
    (2) **Fine**: attach minority classes (`road`, `vegetation`, `streetlight`, `prop`) to their own nodes.

    (2) is deeper than (1), so it wins by the nearest-ancestor rule. Overlaps stay correct.
    """
    from isaacsim.core.utils.semantics import add_labels

    r = stage.GetPrimAtPath(root)
    if not (r and r.IsValid()):
        return {}
    MINORITY = {"road", "vegetation", "streetlight", "prop"}

    def match(prim):
        s = str(prim.GetPath()).lower()
        for name, pat in _RULES:
            if re.search(pat, s):
                return name
        return None

    coarse, fine = [], []

    def walk(prim, d):
        for c in prim.GetChildren():
            cls = match(c)
            if cls in MINORITY and d <= fine_depth:
                fine.append((c, cls))            # minority class on its own node
                continue                          # do not descend further
            if cls and d <= coarse_depth:
                coarse.append((c, cls))          # coarse container
            if d < fine_depth:
                walk(c, d + 1)

    walk(r, 0)

    # Drop coarse descendants already covered by an ancestor (avoid same-level duplicates)
    coarse.sort(key=lambda x: str(x[0].GetPath()))
    kept_coarse = []
    for prim, cls in coarse:
        p = str(prim.GetPath())
        if any(p.startswith(str(k.GetPath()) + "/") for k, _ in kept_coarse):
            continue
        kept_coarse.append((prim, cls))

    picks = kept_coarse + fine
    if len(picks) > max_groups:
        # Keep minority classes first: coarse ones are covered by ancestor inheritance,
        # but a dropped minority class gets its pixels recorded as the dominant class.
        fine_first = fine[:max_groups]
        room = max_groups - len(fine_first)
        picks = fine_first + kept_coarse[:max(room, 0)]
        print(f"[PREP] {len(kept_coarse) + len(fine)} label candidates > cap {max_groups}; "
              f"truncated with minority classes first (fine {len(fine_first)}, coarse {max(room, 0)})")

    counts = {}
    for prim, cls in picks:
        try:
            add_labels(prim, labels=[cls], instance_name="class")
            counts[cls] = counts.get(cls, 0) + 1
        except Exception:
            pass
    return counts


# "Source object" libraries piled at the origin: originals storage, not copies placed in the
# layout. Adding collision there creates an **invisible solid lump** at the origin.
# Measured: in Morning, a downward raycast at (0,0) hit +14.77 m instead of site ground
# (0.00 m), so a spawned drone got trapped inside it and fell to -16.9 m.
SOURCE_LIBS = r"MassingSourceObjects|SourceObjects|/Prototypes?/|/_class_|/PaintTool/.*/asset/"


def apply_collision(stage, skip_invisible=True, skip_patterns=SOURCE_LIBS,
                    verbose=True):
    """Apply static triangle-mesh collision.

    The pack has 0 `CollisionAPI` (pure visualization pack). Approximation is triangle mesh
    (`none`), not `convexHull`: convex hulls fill courtyards and eaves and break the meaning
    of forward clearance.

    **What is excluded matters.** Applying to everything makes off-screen things solid:
      - prims without `points`: collision creation fails with an error.
      - **invisible** prims: present in physics but not in rendering = "invisible walls".
      - **source object libraries** (`SOURCE_LIBS`): originals storage piled at the origin.
    """
    import re
    from pxr import Usd, UsdGeom, UsdPhysics

    rx = re.compile(skip_patterns) if skip_patterns else None
    n = 0
    skipped = {"points": 0, "invisible": 0, "source_lib": 0}
    for p in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate)):
        if not (p.IsA(UsdGeom.Mesh) and not p.IsInstanceProxy()):
            continue
        path = str(p.GetPath())
        if rx is not None and rx.search(path):
            skipped["source_lib"] += 1
            continue
        if not UsdGeom.Mesh(p).GetPointsAttr().Get():
            skipped["points"] += 1
            continue
        if skip_invisible:
            try:
                if UsdGeom.Imageable(p).ComputeVisibility() == UsdGeom.Tokens.invisible:
                    skipped["invisible"] += 1
                    continue
            except Exception:  # noqa: BLE001 — on check failure, lean toward applying
                pass
        UsdPhysics.CollisionAPI.Apply(p)
        UsdPhysics.MeshCollisionAPI.Apply(p).CreateApproximationAttr().Set(UsdPhysics.Tokens.none)
        n += 1
    if verbose:
        print(f"[PREP] collision applied to {n:,}; skipped "
              f"source libs {skipped['source_lib']:,}, "
              f"invisible {skipped['invisible']:,}, no points {skipped['points']:,}",
              flush=True)
    return n


ANIM_TYPES = ("AnimationData", "ShotClip", "Track", "Sequence")

#: **Never disabled.** This is where our targets live: pedestrians (Isaac People) bring
#: clips in via payload and may have time samples. Today `freeze_env_animation` runs
#: **before** spawning so it happens not to hit them, but relying on that order alone means
#: pedestrians **silently** freeze the moment someone moves the call. Moving episodes would
#: become static ones and nobody would notice.
ANIM_NEVER = ("/World/DG_targets",)


def time_sampled_prims(stage, limit=None, active_only=False):
    """Find prims with time-sampled attributes. Returns `[(path, n_attrs, type)]`.

    **Why count these.** `simulate(dt, current_time)` takes a time. As time advances, time
    samples are re-evaluated and that cost **lands on the physics step.** Measured
    on Brownstone Morning:

        time advancing   simulate 14.211 ms
        time fixed       simulate  0.137 ms      <- **104x**

    Brownstone is a marketing demo pack with a camera flythrough animation
    (`AnimationData 27, ShotClip 25, Track 5, Sequence 1`, 2,214 time-sampled attributes).
    **We never use that camera**; only our own camera attached to the vehicle.
    """
    from pxr import Usd

    out = []
    # **Why `active_only` is needed.** Inactive prims are still visited by a
    # `PrimAllPrimsPredicate` traversal, so recounting with the same predicate after
    # disabling **does not drop even on success**: the metric meant to catch silent failure
    # catches nothing (measured: "2,214 -> 2,214" while `simulate` went 13.74 -> 0.48 ms).
    pred = (Usd.TraverseInstanceProxies(Usd.PrimDefaultPredicate) if active_only
            else Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate))
    for pr in Usd.PrimRange.Stage(stage, pred):
        n = 0
        try:
            for at in pr.GetAuthoredAttributes():
                if at.GetNumTimeSamples() > 0:
                    n += 1
        except Exception:  # noqa: BLE001 — skip attribute access failures
            continue
        if n:
            out.append((str(pr.GetPath()), n, str(pr.GetTypeName() or "")))
            if limit and len(out) >= limit:
                break
    return out


def freeze_env_animation(stage, verbose=True, never=ANIM_NEVER):
    """Remove animation **shipped with the environment asset pack** to stop per-step time re-evaluation.

    ⚠️ **Unrelated to the targets we spawn.** Pedestrians/vehicles live under
    `/World/DG_targets`, protected by path via `ANIM_NEVER`. What is disabled here is the
    **pack's USD time samples**; for Brownstone, a marketing demo camera flythrough.

    **Varies per scene.** Packs without time samples disable 0 and finish (warehouse, flat
    ground). So it searches **generically**, not tied to a scene: a new pack with the same
    trap is caught automatically.

    Returns `(prims disabled, remaining time-sampled attributes)`.

    **Geometry and materials are untouched.** Only prims holding animation and their
    ancestor cameras are disabled; appearance and measurements are unchanged. The remaining
    count is returned because "done" can mean "processed 0, succeeded" (a recurring bug class).
    """
    found = time_sampled_prims(stage)
    skipped = [p for p, _n, _t in found
               if any(p == k or p.startswith(k + "/") for k in never)]
    found = [(p, n, t) for p, n, t in found if p not in set(skipped)]
    killed, targets = 0, set()
    for path, _n, typ in found:
        # Animation-only type: that prim; otherwise the nearest animation ancestor
        if typ in ANIM_TYPES:
            targets.add(path)
            continue
        parts = path.split("/")
        for k in range(len(parts), 2, -1):
            cand = "/".join(parts[:k])
            pr = stage.GetPrimAtPath(cand)
            if pr and pr.IsValid() and str(pr.GetTypeName() or "") in ANIM_TYPES:
                targets.add(cand)
                break
        else:
            targets.add(path)       # no such ancestor: the prim itself
    # Drop descendants of targeted ancestors; disabling twice is pointless
    pruned = {t for t in targets
              if not any(t != o and t.startswith(o + "/") for o in targets)}
    for t in sorted(pruned):
        pr = stage.GetPrimAtPath(t)
        if pr and pr.IsValid():
            try:
                pr.SetActive(False)
                killed += 1
            except Exception as exc:  # noqa: BLE001
                print(f"[PREP] ⚠️ failed to disable animation {t}: {exc!r}", flush=True)
    left = sum(n for p, n, _t in time_sampled_prims(stage, active_only=True)
               if not any(p == k or p.startswith(k + "/") for k in never))
    if verbose:
        print(f"[PREP] animation frozen: {killed:,} prims disabled, "
              f"time-sampled attributes {sum(n for _p, n, _t in found):,} -> {left:,}"
              + (f", {len(skipped):,} in protected area skipped" if skipped else ""),
              flush=True)
    return killed, left


def make_probes(stage, top_margin_m=200.0, max_dist_m=500.0):
    """Return `(ray, ground)`. Requires physics to be playing."""
    from pxr import Usd, UsdGeom
    from omni.physx import get_physx_scene_query_interface

    mpu = UsdGeom.GetStageMetersPerUnit(stage)
    U = 1.0 / mpu
    q = get_physx_scene_query_interface()
    rng = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render]
    ).ComputeWorldBound(stage.GetPseudoRoot()).ComputeAlignedRange()
    top = rng.GetMax()[2] + top_margin_m * U

    def ray(o, d, dist):
        r = q.raycast_closest([float(x) for x in o], [float(x) for x in d], float(dist))
        return r if r and r.get("hit") else None

    def ground(x, y):
        r = ray([x, y, top], [0, 0, -1], max_dist_m * U)
        return (top - r["distance"]) if r else None

    return ray, ground, mpu, U, rng


def sweep_path_validity(ray, ground, mpu, U, p0, p1,
                        half_width_m=0.6, heights_m=(0.6, 1.0), step_m=1.0):
    """Check clearance along a path with a **horizontal sweep**.

    Mover targets only have `CollisionAPI` and are driven by direct transforms, so a path
    through buildings or streetlights is not blocked: they **pass through**. Hand-drawn
    paths can be checked in the editor, but procedurally generated ones must be guaranteed
    by code.

    A single centerline is not enough: targets have width and their edges catch. Three
    parallel rays (left, center, right) approximate the swept volume, at both target and
    pedestrian heights.

    Returns a dict (total segments, valid ratio per height, blocked list, longest span
    valid at both heights).
    """
    ax, ay = p0
    bx, by = p1
    L = math.dist((ax, ay), (bx, by))
    if L <= 0:
        return {"segments": 0, "valid": {}, "blocked": [], "best_run_m": 0.0, "best_span": None}
    ux, uy = (bx - ax) / L, (by - ay) / L
    nx, ny = -uy, ux                      # lateral offset direction
    ns = int((L * mpu) // step_m) + 1
    ok_by_h, blocked = {}, []
    for label, hz in ((f"h{h}", h) for h in heights_m):
        ok = []
        for k in range(ns - 1):
            sx, sy = ax + ux * k * step_m * U, ay + uy * k * step_m * U
            gz = ground(sx, sy)
            if gz is None:
                ok.append(False)
                continue
            z = gz + hz * U
            hit = None
            for lat in (-half_width_m, 0.0, half_width_m):
                ox, oy = sx + nx * lat * U, sy + ny * lat * U
                r = ray([ox, oy, z], [ux, uy, 0.0], step_m * U * 1.05)
                if r:
                    hit = (k, lat, r["distance"] * mpu)
                    break
            ok.append(hit is None)
            if hit:
                blocked.append((label,) + hit)
        ok_by_h[label] = ok
    both = [all(v[i] for v in ok_by_h.values()) for i in range(ns - 1)]
    best_len = cur = start = best_start = 0
    for i, v in enumerate(both):
        if v:
            if cur == 0:
                start = i
            cur += 1
            if cur > best_len:
                best_len, best_start = cur, start
        else:
            cur = 0
    span = None
    if best_len:
        s = (ax + ux * best_start * step_m * U, ay + uy * best_start * step_m * U)
        e = (ax + ux * (best_start + best_len) * step_m * U, ay + uy * (best_start + best_len) * step_m * U)
        span = (s, e)
    return {
        "segments": ns - 1,
        "valid": {k: sum(v) / max(len(v), 1) for k, v in ok_by_h.items()},
        "blocked": blocked,
        "best_run_m": best_len * step_m,
        "best_span": span,
    }


def spawn_report(stage, vehicle_pattern="quadrotor", clearance_warn_m=0.05,
                 verbose=True):
    """After play, measure and report whether vehicles sit properly on the ground.

    Without this check, a drone 0.59 m **underground** was only noticed via the symptom
    "camera fully black". Underground the render sees nothing and depth is all `inf`, so
    all collected data is silently meaningless. Hence it is printed once at each run start.
    """
    from pxr import Usd, UsdGeom

    try:
        ray, ground, mpu, U, _ = make_probes(stage)
    except Exception as exc:  # noqa: BLE001
        print(f"[PREP] spawn check: could not create physics probes: {exc!r}", flush=True)
        return []

    rows = []
    for prim in stage.GetPrimAtPath("/World").GetChildren():
        name = prim.GetName()
        if vehicle_pattern not in name:
            continue
        try:
            m = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            t = m.ExtractTranslation()
        except Exception:  # noqa: BLE001
            continue
        gz = ground(t[0], t[1])
        clr = None if gz is None else (t[2] - gz) * mpu
        rows.append({"vehicle": name, "prim": str(prim.GetPath()),
                     "xyz_m": [round(float(v) * mpu, 3) for v in t],
                     "ground_m": (None if gz is None else round(gz * mpu, 3)),
                     "clearance_m": (None if clr is None else round(clr, 3))})

    if verbose:
        spawn_report_print(rows, vehicle_pattern, clearance_warn_m)
    return rows


def spawn_report_print(rows, vehicle_pattern="quadrotor", clearance_warn_m=0.05):
    """Print `spawn_report` results in human-readable form."""
    if not rows:
        print(f"[PREP] spawn check: no '{vehicle_pattern}' prim found", flush=True)
        return
    for r in rows:
        if r["clearance_m"] is None:
            verdict = "⚠️ no ground below (no collision, outside site, unit mismatch?)"
        elif r["clearance_m"] < -clearance_warn_m:
            verdict = f"⚠️ {-r['clearance_m']:.2f} m underground; render will be black"
        elif r["clearance_m"] > 5.0:
            verdict = f"⚠️ {r['clearance_m']:.2f} m above ground (rooftop?)"
        else:
            verdict = "ok"
        gz = "     —" if r["ground_m"] is None else f"{r['ground_m']:+7.3f}"
        cl = "    —" if r["clearance_m"] is None else f"{r['clearance_m']:+6.3f}"
        print(f"[PREP] spawn {r.get('prim', r['vehicle']):<26} z {r['xyz_m'][2]:+7.3f} m, "
              f"ground {gz} m, clearance {cl} m  {verdict}", flush=True)


def fix_env_units(stage, layout_path, env_usd, verbose=True):
    """Match the **units** of the referenced environment subtree to the stage units.

    Pegasus creates the stage with `stage_units_in_meters=1.0` and attaches the environment
    under `/World/layout` via `AddReference`. But USD **references do not bring the target
    layer's `metersPerUnit`**: the root layer's stage metadata wins. Brownstone is authored
    in centimeters (1 m = 100 u), so attaching it as-is makes it **100x** too large: a 380 m
    site becomes 38 km, nothing is near the drone, the render is black and depth is all
    `inf` (measured: mean luma 0.08, only window lights visible as dots).

    So `/World/layout` gets an `env_mpu / stage_mpu` scale. This must run **before** spawn z
    correction or labelling, since ground height changes by 100x.
    """
    from pxr import Gf, Usd, UsdGeom

    stage_mpu = UsdGeom.GetStageMetersPerUnit(stage)
    env_mpu = None
    try:
        # `Sdf.Layer.GetField` cannot read it (measured: None); it may be in a sublayer.
        # Opening with `LoadNone` skips payloads, so even a 20 GB pack opens instantly.
        probe = Usd.Stage.Open(env_usd, load=Usd.Stage.LoadNone)
        env_mpu = UsdGeom.GetStageMetersPerUnit(probe)
    except Exception:  # noqa: BLE001
        env_mpu = None
    if env_mpu is None:
        env_mpu = 0.01 if stage_mpu != 0.01 else stage_mpu
        if verbose:
            print(f"[PREP] could not read the env layer's metersPerUnit; assuming {env_mpu}",
                  flush=True)
    if abs(float(env_mpu) - float(stage_mpu)) < 1e-9:
        if verbose:
            print(f"[PREP] units match (stage {stage_mpu} = env {env_mpu}); no correction",
                  flush=True)
        return 1.0

    s = float(env_mpu) / float(stage_mpu)
    prim = stage.GetPrimAtPath(layout_path)
    if not prim:
        print(f"[PREP] ⚠️ {layout_path} does not exist; unit correction not applied", flush=True)
        return None
    if not prim.GetTypeName():
        prim.SetTypeName("Xform")       # DefinePrim creates it without a type
    xf = UsdGeom.Xformable(prim)
    for op in xf.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeScale:
            op.Set(Gf.Vec3d(s, s, s))
            break
    else:
        xf.AddScaleOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(s, s, s))
    if verbose:
        print(f"[PREP] unit correction: stage {stage_mpu} m/u vs env {env_mpu} m/u -> "
              f"{layout_path} scaled by {s}", flush=True)
    return s


def import_env_lights(stage, env_usd, src="/Environment", dst="/Environment",
                      verbose=True):
    """Import the environment USD's **sun and sky** into the stage separately.

    Why: Pegasus attaches the environment under `/World/layout` with `AddReference(usd)`,
    and USD references bring **only the default prim subtree**. Brownstone's default prim
    is `/World`, but the sun and sky live in its sibling `/Environment`:

        direct open_stage : LightAPI 1,974, incl. DistantLight 1 + DomeLight 1
        AddReference      : LightAPI 1,972, **those two missing**

    So only interior lights (SphereLight 1357, RectLight 531 …) remain and exteriors render
    nearly black (measured mean luma 0.08-0.14). Using the pack's authored lights matters:
    the Morning/Night difference (DR axis B) lives in exactly these prims.

    This was also the cause of the long-unresolved **DR axis A (lighting)** issue: disabling
    `/World/Lights` changed nothing because the sun and sky were not there.
    """
    from pxr import Usd, UsdLux

    prim = stage.GetPrimAtPath(dst)
    if prim and prim.GetChildren():
        if verbose:
            print(f"[PREP] {dst} already exists; skipping light import", flush=True)
        return 0
    prim = stage.DefinePrim(dst)
    if not prim.GetReferences().AddReference(env_usd, src):
        print(f"[PREP] ⚠️ failed to reference {env_usd}<{src}>; exteriors will render black",
              flush=True)
        return 0

    kinds = {}
    for p in Usd.PrimRange(prim, Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate)):
        if p.HasAPI(UsdLux.LightAPI):
            k = str(p.GetTypeName() or "-")
            kinds[k] = kinds.get(k, 0) + 1
    n = sum(kinds.values())
    if verbose:
        if n:
            print(f"[PREP] imported {n} environment lights {kinds} <- {src}", flush=True)
        else:
            print(f"[PREP] ⚠️ no lights in {src}; in this variant the sun/sky may be at a "
                  "different path", flush=True)
    return n


def import_env_looks(stage, env_usd, src="/Looks", dst="/Looks", verbose=True):
    """Import the environment USD's root `/Looks` into the stage, for the same reason as `/Environment`.

    Each building asset (`Brownstone01.usd` …) authors **geometry under
    `/World/Brownstone01` and materials under the root `/Looks`**. When Pegasus references
    everything under `/World/layout`, `/Looks` falls outside the scope and USD **silently
    drops** those bindings:

        "The relationship target </Looks/Paint_Eggshell_White> … refers to a path
          outside the scope of the reference from </World/layout>. Ignoring."

    Measured: 2,748 prims lost bindings to 80 materials (17,172 warnings), and those 80 are
    within the 115-material union of `/Looks` across the four asset layers. So importing it
    lets them be rebound.
    """
    from pxr import Usd, UsdShade

    prim = stage.GetPrimAtPath(dst)
    if prim and prim.GetChildren():
        if verbose:
            print(f"[PREP] {dst} already exists; skipping material import", flush=True)
        return 0
    prim = stage.DefinePrim(dst)
    if not prim.GetReferences().AddReference(env_usd, src):
        print(f"[PREP] ⚠️ failed to reference {env_usd}<{src}>; cannot restore materials", flush=True)
        return 0
    n = sum(1 for p in Usd.PrimRange(prim, Usd.TraverseInstanceProxies(
        Usd.PrimAllPrimsPredicate)) if p.IsA(UsdShade.Material))
    if verbose:
        print(f"[PREP] imported {n} root materials <- {src}", flush=True)
    return n


def rebind_dropped_materials(stage, layout="/World/layout", looks="/Looks",
                             legacy=False, verbose=True):
    """Re-apply `material:binding`s dropped by composition, **as originally intended**.

    Composition dropped the targets, but **the PrimSpecs still hold the original paths**. So
    we read them from the stack and remap `/Looks/X` -> `<looks>/X`. No second stage needed.

    **Fixing only Gprims is not enough: material bindings are inherited.** This pack authors
    many bindings not on meshes but on their **ancestors** (`Scope`/`Xform`, e.g.
    `…/PavementLights/lightBox`). The first version only scanned Gprims, and against the
    reference **877 prims resolved to `Default` instead of `Chromium_Brushed`** (89.06 %
    match): the ancestor binding was still dropped, so inheritance picked up the wrong one.
    So **every prim that authors a binding** is fixed, regardless of type.

    **Binding strength (`bindMaterialAs`) must be restored too.** With ancestors fixed, 877
    prims were still wrong; the cause was USD binding strength. This pack binds ancestors
    with `strongerThanDescendants` and binds a generic material directly on descendant meshes:

        lightBox                -> /Looks/Chromium_Brushed   strongerThanDescendants
        lightBox/Mesh_0129…     -> …/Looks/Default            weakerThanDescendants

    Rebinding with default strength lets the descendant win, giving `Default` instead of
    `Chromium_Brushed`. So the authored spec's `bindMaterialAs` is read and passed through.

    The reference is a stage opened directly on the environment USD (where `/Looks` is in
    scope).

    Instance proxies are skipped: bindings cannot be authored on them.
    `legacy` is **for diagnostics**: it reproduces the v1 behavior with both defects above
    (Gprims only, strength not preserved), only for before/after render comparisons.
    """
    from pxr import Sdf, Usd, UsdGeom, UsdShade

    root = stage.GetPrimAtPath(layout)
    if not root:
        print(f"[PREP] {layout} does not exist; skipping material restore", flush=True)
        return 0
    pre = "/Looks/"
    fixed = missing = already = 0
    fixed_anc = kept_strength = 0
    for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies(
            Usd.PrimAllPrimsPredicate)):
        if prim.IsInstanceProxy():
            continue
        if legacy and not prim.IsA(UsdGeom.Gprim):
            continue
        rel = prim.GetRelationship("material:binding")
        if rel and rel.GetTargets():
            already += 1
            continue
        want = strength = None
        for spec in prim.GetPrimStack():
            r = spec.relationships.get("material:binding")
            if r is None:
                continue
            items = list(r.targetPathList.explicitItems) or list(
                r.targetPathList.prependedItems)
            if items:
                want = str(items[0])
                # **Strength must be carried over too**; see the docstring
                try:
                    if r.HasInfo("bindMaterialAs"):
                        strength = r.GetInfo("bindMaterialAs")
                except Exception:  # noqa: BLE001
                    strength = None
                break
        if not want or not want.startswith(pre):
            continue
        tgt = stage.GetPrimAtPath(looks.rstrip("/") + "/" + want[len(pre):])
        if not tgt or not tgt.IsValid():
            missing += 1
            continue
        try:
            api = UsdShade.MaterialBindingAPI.Apply(prim)
            if strength and not legacy:
                api.Bind(UsdShade.Material(tgt), bindingStrength=strength)
                kept_strength += 1
            else:
                api.Bind(UsdShade.Material(tgt))
            fixed += 1
            if not prim.IsA(UsdGeom.Gprim):
                fixed_anc += 1
        except Exception:  # noqa: BLE001 — one prim failing must not block the scene
            missing += 1
    if verbose:
        print(f"[PREP] rebound {fixed:,} materials ({fixed_anc:,} on ancestor prims, "
              f"{kept_strength:,} with strength preserved), already bound {already:,}, "
              f"target not found {missing:,}", flush=True)
    return fixed

GROUND_PAT = r"(Sidewalk|Road|Street|Pavement|Ground|Terrain|Site[^/]*$|Curb|Asphalt)"


# Prims to ignore at the ray origin: **the vehicle itself**. The airframe has a collider, so
# a ray cast from its own position hits itself at distance 0 (measured: a wall
# check in 8/8 directions and a camera ground probe on 5/5 pixels all hit `/World/quadrotor/body/body`).
SELF_PRIMS = ("/World/quadrotor", "/World/vehicle", "/World/drone")


def cast_past_self(ray, o, u, dist, U, ignore=SELF_PRIMS, tries=4):
    """`(distance, prim_path)`: first hit on **something else**, past the vehicle."""
    off = 0.0
    for _ in range(tries):
        r = ray([o[i] + u[i] * off for i in range(3)], list(u), dist - off)
        if not r:
            return None, None
        path = str(r.get("collision") or r.get("rigidBody") or "")
        if not path.startswith(tuple(ignore)):
            return off + r["distance"], path
        off += r["distance"] + 0.02 * U
        if off >= dist:
            return None, None
    return None, None


def ground_materials(stage, ray, rng, mpu, U, grid=14, verbose=True,
                     region=None):
    """**Measure ground materials with rays**, without relying on name patterns.

    Why: name matching fails twice. Matching shader paths finds 0 (material names are
    `Terracotta`, not `Sidewalk`); matching mesh names finds 1212, with no guarantee the
    visible slabs are among them (measured: fixing 21 changed the image by only 1/255).
    Whatever a top-down ray hits is **ground by definition**.

    Cast a grid within the site bounds and collect the materials bound to the hit prims.
    The vehicle is skipped.

    ⚠️ **Always pass `region`.** The full stage bbox is 5 km x 5 km (measured:
    `x -227~5001, y -139~5002 m`); a dome light or a distant prim stretches the bounds. A
    14x14 grid over that lands **1 of 196 rays** on the actual site (~360 m). This was
    misread twice as "PhysX is still cooking". Pass the coordinate range where episodes
    actually happen.

    `region` = `(xmin, xmax, ymin, ymax)` (world units). If absent, `rng` is used, with a
    warning when a side exceeds 2 km.

    Returns `{material_path: (material, hit_count, example_prim), "__hits__": total_hits}`
    """
    from pxr import UsdShade

    lo, hi = rng.GetMin(), rng.GetMax()
    top = hi[2] + 50.0 * U
    if region is not None:
        x0, x1, y0, y1 = (float(v) for v in region)
    else:
        x0, x1, y0, y1 = lo[0], hi[0], lo[1], hi[1]
        if verbose and max(x1 - x0, y1 - y0) * mpu > 2000.0:
            print(f"[PREP] ⚠️ probe range is "
                  f"{(x1 - x0) * mpu:.0f}×{(y1 - y0) * mpu:.0f} m; "
                  f"using the stage bbox as-is puts almost every ray off-site", flush=True)
    found = {}
    n_hit = 0
    for a in range(grid):
        for b in range(grid):
            x = x0 + (x1 - x0) * (a + 0.5) / grid
            y = y0 + (y1 - y0) * (b + 0.5) / grid
            d, path = cast_past_self(ray, [x, y, top], [0.0, 0.0, -1.0],
                                     500.0 * U, U)
            if d is None:
                continue
            n_hit += 1
            prim = stage.GetPrimAtPath(path)
            if not (prim and prim.IsValid()):
                continue
            mat, _r = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
            if not mat:
                continue
            k = str(mat.GetPath())
            if k in found:
                found[k] = (found[k][0], found[k][1] + 1, found[k][2])
            else:
                found[k] = (mat, 1, path)
    if verbose:
        print(f"[PREP] ground material probe: {grid * grid} rays, {n_hit} hits, "
              f"{len(found)} materials, range "
              f"x {x0 * mpu:.0f}~{x1 * mpu:.0f}, "
              f"y {y0 * mpu:.0f}~{y1 * mpu:.0f} m", flush=True)
        for k, (_m, n, ex) in sorted(found.items(), key=lambda kv: -kv[1][1])[:8]:
            print(f"[PREP]   {n:3d} hits, {k}", flush=True)
            print(f"[PREP]        e.g. {ex}", flush=True)
    found["__hits__"] = n_hit          # the caller decides whether it is "not ready yet"
    return found


def fix_ground_materials(found, factor=None, target_scale=None, desaturate=0.0,
                         below=0.01, world_only=True, verbose=True):
    """Fix texture tiling of the ground materials measured by `ground_materials`.

    **Why the scene values look broken: the unit fix breaks them.**

    Brownstone is authored in centimeters (`metersPerUnit = 0.01`). Our stage is in meters,
    so `/World/layout` gets **scale 0.01** (`fix_units`). But these materials use
    `project_uvw = true` + `world_or_object = true`, i.e. **world-space triplanar
    projection**: UVs come from the shading point's world coordinates, not from vertices.

        UV = world_coords × texture_scale

    In the centimeter original a 5 m sidewalk is 500 units, and times `0.002` gives exactly
    one tile. After our 0.01 scale the same sidewalk is 5 units, 0.01 tile: **the texture
    is stretched 100x.** So looking down from 3 m there was no grain, only smudges.

    So the correct fix is **authored value × (1/layout scale) = ×100**, not one absolute
    value for all: roads (0.005 -> 0.5 = one tile per 2 m) and sidewalks (0.002 -> 0.2 =
    one tile per 5 m) were intentionally different and that intent should be kept.

    `world_only` touches **world-projected materials only**. Object-projected or UV-mapped
    materials are not affected by the parent scale, so there is nothing to fix; touching
    them breaks them.

    Always keep `below`. Running without it also smeared tree bark (1.0, 0.7) and the pond
    (0.05, 0.1) (measured). Broken values are <= 0.005, so 0.01 is enough.

    Pass only one of `factor` and `target_scale`. Returns `(n_fixed, n_seen, notes)`.
    """
    from pxr import Gf, Usd, UsdShade

    if factor is None and target_scale is None:
        factor = 1.0
    n_fix = n_seen = n_skip_obj = 0
    notes = []
    items = [(k, v) for k, v in found.items() if k != "__hits__"]
    for k, (mat, hits, _ex) in sorted(items, key=lambda kv: -kv[1][1]):
        for sh in Usd.PrimRange(mat.GetPrim()):
            if not sh.IsA(UsdShade.Shader):
                continue
            shd = UsdShade.Shader(sh)
            inp = shd.GetInput("texture_scale")
            if not inp or inp.Get() is None:
                continue
            n_seen += 1
            if world_only:
                pu = shd.GetInput("project_uvw")
                wo = shd.GetInput("world_or_object")
                if not (pu and pu.Get() and wo and wo.Get()):
                    n_skip_obj += 1
                    continue
            v = inp.Get()
            try:
                vals = [float(x) for x in v]
            except TypeError:
                vals = [float(v)]
            if max(vals) >= below:
                continue
            new = ([float(target_scale)] * len(vals) if target_scale is not None
                   else [x * float(factor) for x in vals])
            try:
                inp.Set(Gf.Vec2f(*new) if len(new) == 2 else float(new[0]))
            except Exception as exc:  # noqa: BLE001
                notes.append(f"{k}: write failed {exc!r}")
                continue
            if desaturate is not None:
                di = shd.GetInput("albedo_desaturation")
                if di:
                    try:
                        di.Set(float(desaturate))
                    except Exception:  # noqa: BLE001
                        pass
            n_fix += 1
            if len(notes) < 5:
                tile = (1.0 / new[0]) if new[0] else float("inf")
                notes.append(
                    f"{k.split('/')[-1]} ({hits} hits): "
                    f"{tuple(round(x, 4) for x in vals)} → "
                    f"{tuple(round(x, 4) for x in new)} = one tile per {tile:.1f} m")
    if verbose:
        how = (f"×{factor:g}" if target_scale is None else f"→ {target_scale:g}")
        print(f"[PREP] ground materials {n_fix}/{n_seen} {how}, desaturation {desaturate}"
              + (f", {n_skip_obj} non-world-projected skipped" if n_skip_obj else ""),
              flush=True)
        for t in notes:
            print(f"[PREP]   {t}", flush=True)
        if n_fix == 0:
            print("[PREP]   ⚠️ **nothing fixed**: the measured ground materials are not "
                  "world-projected, or texture_scale is already fine", flush=True)
    return n_fix, n_seen, notes


def bound_shader(prim):
    """Return the **shader** of the material bound to a prim, or `None`."""
    from pxr import UsdShade

    mat, _rel = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
    if not mat:
        return None
    src = mat.ComputeSurfaceSource("mdl")
    shd = src[0] if isinstance(src, tuple) else src
    if shd:
        return shd
    src = mat.ComputeSurfaceSource()
    shd = src[0] if isinstance(src, tuple) else src
    return shd or None


def boost_ground_texture(stage, factor=None, below=0.01, pattern=GROUND_PAT,
                         verbose=True, target_scale=None, desaturate=None):
    """Restore ground material **texture tiling**. Fixes stretching rather than repainting.

    Measured: the material bound to sidewalk/site prims is

        /World/layout/BrownstoneSite/Looks/Terracotta, texture_scale=(0.002, 0.002)
        project_uvw = True, world_or_object = True, albedo_desaturation = 1.0

    and `Terracotta.mdl` correctly contains `Terracotta_BaseColor/ORM/N.png`. Material and
    texture are fine, but the **world projection scale is 0.002**, so one tile covers
    500 m, a value meant for aerial views. From a drone looking down from 3 m there is no
    grain at all, and `albedo_desaturation = 1.0` removes color, leaving a pale slab.

    ⚠️ Matching is done on **mesh names**, and the **materials bound to them** are fixed.
    Material names follow the material type (`Terracotta`), not the prim name
    (`SiteSidewalk_003`), so matching `Sidewalk|Road|…` against shader paths finds 0.

    Only scales below `below` are touched, to avoid breaking normal materials.
    `target_scale` is an **absolute value** (texture tiles per world meter) and, if given,
    overrides regardless of `below`. Scaling by `factor` 100x still gave 0.2 = one tile per
    5 m, which was not enough, hence the absolute option.
    """
    import re

    from pxr import Gf, Usd, UsdGeom, UsdShade

    rx = re.compile(pattern, re.I)
    # (1) find ground **meshes** by name, (2) collect shaders of materials bound to them.
    #
    # Descends into instance proxies: if site geometry is instanced, a plain `Traverse()`
    # sees no meshes at all. What gets edited is not the proxy but the bound **material**
    # (`/World/layout/BrownstoneSite/Looks/…`), which is authorable.
    shaders, mesh_n, mesh_ex = {}, 0, []
    for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        if not rx.search(str(prim.GetPath())):
            continue
        mesh_n += 1
        if len(mesh_ex) < 3:
            mesh_ex.append(str(prim.GetPath()))
        shd = bound_shader(prim)
        if shd:
            shaders.setdefault(str(shd.GetPath()), shd)

    seen, changed, samples = 0, 0, []
    for path, shd in sorted(shaders.items()):
        shd = UsdShade.Shader(shd)
        inp = shd.GetInput("texture_scale")
        if not inp:
            continue
        v = inp.Get()
        if v is None:
            continue
        seen += 1
        try:
            vals = [float(x) for x in v]
        except TypeError:
            vals = [float(v)]
        if target_scale is None and max(vals) >= below:
            continue
        if target_scale is not None:
            new = [float(target_scale)] * len(vals)
        else:
            new = [x * float(factor or 1.0) for x in vals]
        try:
            if len(new) == 2:
                inp.Set(Gf.Vec2f(*new))
            else:
                inp.Set(new[0])
        except Exception as exc:  # noqa: BLE001
            if len(samples) < 3:
                samples.append(f"{path}: write failed {exc!r}")
            continue
        if desaturate is not None:
            di = shd.GetInput("albedo_desaturation")
            if di:
                try:
                    di.Set(float(desaturate))
                except Exception:  # noqa: BLE001
                    pass
        changed += 1
        if len(samples) < 3:
            samples.append(f"{path}: {tuple(round(x, 4) for x in vals)} → "
                           f"{tuple(round(x, 3) for x in new)}")
    if verbose:
        how = (f"→ {target_scale:g}" if target_scale is not None
               else f"×{float(factor or 1.0):g}")
        print(f"[PREP] ground materials {changed}/{seen} {how} "
              f"({mesh_n} ground meshes -> {len(shaders)} materials)"
              + (f", desaturation {desaturate:g}" if desaturate is not None else ""),
              flush=True)
        for t in samples:
            print(f"[PREP]   {t}", flush=True)
        for m in mesh_ex:
            print(f"[PREP]   ground mesh e.g. {m}", flush=True)
        if changed == 0:
            print("[PREP]   ⚠️ **nothing fixed**: the pattern found no ground meshes "
                  "or the materials have no texture_scale", flush=True)
    return changed
