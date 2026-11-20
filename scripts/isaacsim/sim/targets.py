# DroneCATS
# Copyright (c) 2026-present NAVER Cloud Corp.
# Apache-2.0

"""Benchmark targets: spawn painted real pack props and move them along waypoints.

Three conventions matter on the launcher path.

  (1) **Spawn location**. Targets live **outside** `/World/layout` (`/World/DG_targets`).
      Occlusion method (c) hides the environment root to measure silhouettes; a target under
      it would be hidden too and `occlusion_ratio` silently becomes meaningless (measured: all 832 pairs).
  (2) **Units**. The Pegasus stage is 1 m/unit and pack props are authored in centimeters.
      Props get an `env_mpu / stage_mpu` scale, for the same reason as `scene_prep.fix_env_units`.
  (3) **Ground reference**. Flat stretches are found relative to the **spawn-point ground**, not
      the site **minimum**. Using the minimum picks some unrelated low area (same mistake as the spawn-z bug).

Targets have only `CollisionAPI`, no `RigidBodyAPI`; they are driven directly by transforms.
The AirSim benchmark does the same (paths drawn by hand), so a path through geometry
**passes through instead of being blocked.** Procedural paths must be guaranteed in code via
`scene_prep.sweep_path_validity`.
"""
import math
import os

# Palette names from the AirSim benchmark's PAINT_BGR (BGR order).
# ⚠️ **These are pure colors chosen by how they render.** Using the original palette as is,
# the phrase and the visible color disagree; measured: 10 of 20 episodes had wrong color names:
#
#     yellow hydrant   (240,220,20)  → (118,145,48)  a person would say **lime**
#     orange trash can (240,140,30)  → (152,147,91)  saturation 0.40 → **grey**
#     lime cargo box   ( 90,220,40)  → ( 71,140,71)  → **green**
#     magenta barrel   (200, 20,180) → (147, 51,206) → **purple**
#     cyan barrel      (  0,255,255) → (  0,192,217) saturation 1.00 → **cyan** ✓
#
# **That only cyan survived is the clue.** cyan's R is 0 and stays 0 in the render, whereas
# purple's G=30 is **lifted** to 85 and red's G=40 to 89: tone mapping lifts dark channels
# and collapses saturation. Zero channels stay zero, so **pure colors with secondary channels
# at 0** keep their names.
#
# Values differ from the original palette but **the names are the same**: the contract is the
# color name, not the RGB, and the AirSim side paints with textures so it is unaffected.
PAINT_BGR = {
    "magenta": (255, 0, 255), "yellow": (0, 230, 255), "lime": (0, 255, 128),
    "blue": (255, 0, 0), "red": (0, 0, 255), "purple": (255, 0, 128),
    "orange": (0, 90, 255), "green": (0, 180, 0), "white": (240, 240, 240),
    "grey": (128, 128, 128),
    # The AirSim test set uses "black SUV" (`nh_tracking.json`); needed for vehicle colors.
    # Pure black is indistinguishable from shadow, so use **a very dark grey** (measured luma ~18).
    "black": (18, 18, 18),
    "cyan": (255, 255, 0),      # same as the original `spawn_unique.PAINT_BGR`; the only one that matched
}

PACK = os.environ.get(
    "DATAGEN_BROWNSTONE",
    os.path.expanduser("~/omniverse-assets/Brownstone/Demos/AEC/BrownstoneDemo"),
)

# Pack prop -> (relative path, benchmark-vocabulary "kind", semantic class)
#
# ⚠️ Everything here is **street furniture fixed in place**. Never use it as a moving target:
# a hydrant sliding along the street makes sense neither as data nor as a demo.
# Hence the separate `MOVABLE`, which the launcher checks.
PROPS = {
    "trashcan":  ("Props/TrashCan/TrashCan.usd",              "trash can",   "prop"),
    "bench":     ("Props/ParkBench01/ParkBench01.usd",        "bench",       "prop"),
    "hydrant":   ("Props/Hydrant/Hydrant.usd",                "hydrant",     "prop"),
    "mailbox":   ("Props/Mailbox/MailboxType01.usd",          "mailbox",     "prop"),
    "outdoortrash": ("Props/OutdoorTrash01/OutdoorTrash01.usd", "trash bin", "prop"),
    "shelter":   ("Props/MobilaShelter/MobilaShelter.usd",     "bus shelter", "prop"),
    "table":     ("Props/Kalmar_TblChr/Kalmar_TblChr.usd",     "table",      "prop"),
    "streetlamp": ("Props/StreetLamp/StreetLamp.usd",          "street lamp", "streetlight"),
}

# Primitive target -> (shape, vocabulary "kind", semantic class, size m)
#
# The AirSim episode JSON drags `Cube`, `Sphere` and `Mesh_garbageA` around exactly like this,
# and `lime cargo box`, `blue ball`, `yellow barrel` in the vocabulary are these. **Moving
# targets are chosen from here** because moving them makes sense.
PRIMS = {
    "box":    ("Cube",     "cargo box", "prop", 0.8),
    "ball":   ("Sphere",   "ball",      "prop", 0.5),
    "barrel": ("Cylinder", "barrel",    "prop", 0.9),
}

# Pedestrian -> (character USD relative path, vocabulary "kind", semantic class)
#
# The vocabulary is **role/appearance**, not color. Pack props get more variety by painting
# (orange trash can), but people's clothes already say their role, so painting them would be
# nonsensical; same reason AirSim's `police` (police car) carries no color name.
# ⚠️ **Vocabulary follows what is visible, not the asset name.** Our design principle (1) (labels
# come only from what the model can see) applies: even if the asset is `male_adult_police_04`,
# if what renders is **a red polo shirt with sunglasses**, "policeman" contradicts observation,
# and since the phrase becomes the episode ("go to the policeman") it teaches the model a falsehood.
# The descriptions below were verified by eye from renders of all 7 in good light.
#
# ⚠️ **7 kinds means no spares.** Measured: the intended moving-group mix is
# 7 people + 2 cars + 1 drone, and the first 10 slots use all 7 people, so if one person cannot
# be used there is no same-kind spare and the plan has to fill with a car (the mix becomes 6/3/1).
#
# To fix, register more characters from the Isaac People pack here (it has more than 7).
# Follow the principle above when registering: **write the description from the render**.
# The label becomes the episode, so a guessed name teaches the model a falsehood.
PEOPLE = {
    # ⚠️ Don't trust asset names; **describe from the render.** Measured:
    # not a uniform but a blue shirt with beige pants. Calling it "policewoman" would make
    # **the episode wrong**, not the model.
    "police_f":   ("Characters/female_adult_police_01_new/female_adult_police_01_new.usd",
                   "woman in a blue shirt and beige pants", "person"),
    "police_m":   ("Characters/male_adult_police_04/male_adult_police_04.usd",
                   "man in a red polo shirt", "person"),      # not a uniform, despite the asset name
    "worker_m":   ("Characters/male_adult_construction_01_new/male_adult_construction_01_new.usd",
                   "construction worker in a hi-vis vest", "person"),
    "worker_m2":  ("Characters/male_adult_construction_03/male_adult_construction_03.usd",
                   "man in a black work vest", "person"),     # a black vest, not a safety vest
    "medic_f":    ("Characters/F_Medical_01/F_Medical_01.usd",
                   "nurse in green scrubs", "person"),
    # renders as a long pinkish coat with yellow pants; nothing marks a doctor (measured)
    "medic_m":    ("Characters/M_Medical_01/M_Medical_01.usd",
                   "bald man in a long pink coat", "person"),
    "business_f": ("Characters/F_Business_02/F_Business_02.usd",
                   "woman in a purple top", "person"),        # not a suit
}

PEOPLE_ROOT = os.environ.get(
    "DATAGEN_ISAACPEOPLE", os.path.expanduser("~/omniverse-assets/IsaacPeople"))
BIPED_SETUP = "Characters/Biped_Setup.usd"

# Pedestrian effective speed is **not ours to choose**; it is intrinsic to the clip, because
# motion matching owns the root motion. Only the path is ours.
#   * nominal (straight one-way, empty scene, measured): 1.023 m/s; playback rate doesn't change it
#   * real scene (Brownstone tracking episode): 0.883 m/s; start acceleration and stop deceleration lower the mean
# So `--target-speed` is ignored for pedestrians. The dataset records the nominal value; the
# per-episode actual speed can be recomputed from the GT trajectory (position per frame).
PEOPLE_SPEED_MPS = 1.023
PEOPLE_SPEED_OBSERVED_MPS = 0.883

# ── Vehicles and drones: kind -> (path, noun, semantic class, extra scale) ──
#
# In the AirSim test set **drones are the most common target**: 11 of 40 episodes use
# `Quadrotor1`, always recolored (campus_tracking 4/10, campus_approaching 3/10, nh_tracking 3/10).
# Vehicles are also a main moving target (`009_SUV_ISM_NewMat`, `Saloon_ISM_NewMat`, `Car_01`).
# So we include both.
#
# **Why a scale factor.** Pegasus's `iris` is a small 0.52 m airframe, invisible from the
# standard start distance: at 30 m its mask is 24 px, below the visibility threshold of 50
# (measured). The scale was chosen by measurement (at 30 m):
#
#     scale 1    0.52 m    24 px   ✗ below threshold
#     scale 2    1.04 m    89 px   ✅ 1.8x margin
#     scale 2.5  1.30 m   147 px   ✅ 2.9x margin   <- chosen
#     scale 3    1.56 m   210 px   ✅ 4.2x margin
#
# 1.30 m is a real size for large industrial UAVs (DJI M300 is about 1 m; survey/logistics ones 1.3-1.5 m).
# The upscale is physically justified, and the AirSim `Quadrotor1` is also larger than our iris.
#
# ⚠️ Vehicles are **CC-BY**; attribution is required. We don't redistribute, only document how to get them.
# `metadata.json` holds author, year and license URL, so it can be recorded as is.
VEHICLE_ROOT = os.environ.get(
    "DATAGEN_VEHICLES", os.path.expanduser("~/omniverse-assets/Vehicles"))
PEGASUS_IRIS = os.environ.get(
    "DATAGEN_IRIS",
    os.path.expanduser("~/PegasusSimulator/extensions/pegasus.simulator/pegasus/"
                       "simulator/assets/Robots/Iris/iris.usd"))

VEHICLES = {
    # CarConcept — CC-BY-4.0 · Eric Chadwick / Darmstadt Graphics Group, 2024 ·
    # Khronos glTF-Sample-Assets. Converted from glTF (Y-up, cm) to USD, so
    # **both unit and axis fixes are needed**; `spawn_prop` reads the layer metadata to apply them.
    "car": ("CarConcept/CarConcept.usd", "car", "vehicle", 1.0),
    # iris: the same asset as our own vehicle (Pegasus). Available locally, nothing to fetch.
    "drone": (PEGASUS_IRIS, "drone", "vehicle", 2.5),
}

# Kinds that make sense to move. Static street furniture is not here.
MOVABLE = set(PRIMS) | set(PEOPLE) | set(VEHICLES)

# **Kinds whose base is not placed on the ground.** Value is the hover height above ground (m).
#
# Why. `sit_on_ground` puts every target's base on the ground, which is the wrong convention
# for a drone: measured, a drone target sat at z ≈ -0.03 m, and moving it 90 m at 0.3 m/s
# gave "a quadrotor sliding along the ground". The task is "follow a flying vehicle", and the
# episode's subject becomes a physically nonsensical object.
#
# 1.5 m is used. It is below the vehicle's flight altitude, giving a look-down view, and the
# vertical gap stays well inside the arrival radius δ 5 m.
#
# Rotors don't spin since `freeze_physics` disabled the joints. Rotor spin on a 0.52 m
# airframe isn't visible at 15-30 m, so it doesn't matter for a detection target.
HOVER_M = {"drone": 1.5}


# Temporary pre-placement position; must be **outside** the site bbox (x -227..153, y -139..161 m).
# At the origin, the target's collider gets hit by ground probes and breaks spawn checks and path search (measured).
PARKING = (5000.0, 5000.0, 0.0)


# Token meaning "don't paint". **This is the single source of truth**: there are two parsers
# (generator and A) and letting each decide drifts. It did drift, and the first street-furniture target failed to spawn.
NO_COLOR_TOKENS = frozenset({"", "none", "null", "-", "nil"})


def as_color(token):
    """Color field of a spec string -> `None` or a color name. Case-insensitive."""
    if token is None:
        return None
    t = str(token).strip()
    return None if t.lower() in NO_COLOR_TOKENS else t


# Paint blending: 0 = old behavior (cover with a flat color), 1 = fully remove the original
# hue and apply ours (texture is kept either way). The launcher overrides it from the CLI.
PAINT_BLEND = 0.0        # default **off**. Turning it on shifts colors
PAINT_BRIGHTNESS = 1.6


def set_paint_blend(blend, brightness=None):
    """Called by the launcher. Module global, so `paint()` uses it without arguments."""
    global PAINT_BLEND, PAINT_BRIGHTNESS
    PAINT_BLEND = float(blend)
    if brightness is not None:
        PAINT_BRIGHTNESS = float(brightness)
    print(f"[PAINT] blend {PAINT_BLEND:g} · brightness gain {PAINT_BRIGHTNESS:g} "
          f"({'keep original texture' if PAINT_BLEND > 0 else 'flat color overwrite (old behavior)'})",
          flush=True)


def rgb01(color):
    b, g, r = PAINT_BGR[color]
    from pxr import Gf
    return Gf.Vec3f(r / 255.0, g / 255.0, b / 255.0)


def make_paint_material(stage, name, color, root="/World/DG_paint"):
    """Build a flat-color OmniPBR material; recipe verified 5/5."""
    from pxr import Sdf, UsdShade

    path = f"{root}/{name}"
    existing = stage.GetPrimAtPath(path)
    if existing and existing.IsValid():
        return UsdShade.Material(existing)
    mat = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, path + "/Shader")
    sh.SetSourceAsset(Sdf.AssetPath("OmniPBR.mdl"), "mdl")
    sh.SetSourceAssetSubIdentifier("OmniPBR", "mdl")
    sh.CreateIdAttr("OmniPBR")
    sh.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set(rgb01(color))
    # ⚠️ **Must be matte.** The old value 0.5 left gloss that added bright sky as white and
    # collapsed saturation; measured, the phrase and visible color disagreed:
    #
    #     magenta barrel  intended (200,20,180) → measured (79,18,148)  purple
    #     red bench       intended (220,40,40)  → measured (207,139,153) pink
    #     white shelter   intended (240,240,240)→ measured (240,238,237) exact
    #
    # That only white matched is the clue: albedo applies but reflections add white. Specular
    # is also off, since that is how sky reflections get in. Values chosen with `--paint-probe`.
    sh.CreateInput("reflection_roughness_constant", Sdf.ValueTypeNames.Float).Set(0.95)
    sh.CreateInput("specular_level", Sdf.ValueTypeNames.Float).Set(0.0)
    sh.CreateInput("metallic_constant", Sdf.ValueTypeNames.Float).Set(0.0)
    mat.CreateSurfaceOutput("mdl").ConnectToSource(sh.ConnectableAPI(), "out")
    return mat


# Inputs inherited unchanged from the original; all of the surface "texture" lives here:
# albedo texture (grime, seams, wear), normal map (relief), ORM (roughness, AO), UV layout.
_KEEP_ASSET = (
    "diffuse_texture", "normalmap_texture", "ORM_texture",
    "reflectionroughness_texture", "metallic_texture", "ao_texture",
    "detail_normalmap_texture",
)
_KEEP_SCALAR = (
    "texture_scale", "texture_translate", "texture_rotate",
    "detail_texture_scale", "detail_texture_translate", "detail_texture_rotate",
    "project_uvw", "world_or_object", "uv_space_index",
    "bump_factor", "detail_bump_factor", "flip_tangent_u", "flip_tangent_v",
    "enable_ORM_texture", "reflection_roughness_texture_influence",
    "ao_to_diffuse",
)


def _mdl_shader_of(prim):
    """MDL shader of the material bound to the prim, or `None`."""
    from pxr import UsdShade

    mat, _r = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
    if not mat:
        return None
    src = mat.ComputeSurfaceSource("mdl")
    shd = src[0] if isinstance(src, tuple) else src
    return (UsdShade.Shader(shd) if shd else None, mat)


def make_blend_material(stage, name, color, src_shader, blend=1.0,
                        brightness=1.6, root="/World/DG_paint"):
    """Build a material that **keeps the original texture and swaps only the color**.

    Why. Covering everything with a flat material drops all textures, normal and roughness
    maps, and a mailbox becomes "a glaring plastic lump". Measured: rendered RGB was
    `(253, 0, 0)`, `(0, 209, 0)`, `(246, 241, 1)`, with secondary channels all 0-6; shape
    remains but no surface.

    The recipe combines two knobs the OmniPBR-family MDL already exposes:

        albedo_desaturation = 1.0   remove **only the hue** from the original albedo (shading detail stays)
        diffuse_tint        = C     multiply the desired color onto that greyscale

    Multiplication darkens, so `albedo_brightness` compensates. Normal map, ORM and UV
    layout are untouched, so relief and roughness variation survive.

    ⚠️ **Don't replace the original MDL.** The first version made a new `OmniPBR.mdl`
    shader and copied the USD-authored `diffuse_texture`, but these props carry their
    textures **inside the MDL module, not in USD** (measured: inside `Ash.mdl`,
    `diffuse_texture: texture_2d("./Ash/Ash_BaseColor.png")`). There was nothing to copy,
    and 18 of 20 targets went flat with "no original texture found". Now the **original
    shader's MDL asset and sub-identifier are cloned**, authored inputs are carried over,
    and only the color knobs are overridden, so the MDL-internal textures come along.

    ⚠️ **The original material is copied, not edited in place.** Prop materials are sometimes
    shared by several objects; editing in place would tint other mailboxes in the scene too.
    """
    from pxr import Sdf, UsdShade

    path = f"{root}/{name}"
    existing = stage.GetPrimAtPath(path)
    if existing and existing.IsValid():
        return UsdShade.Material(existing)
    mat = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, path + "/Shader")

    src_asset = src_shader.GetSourceAsset("mdl")
    sub = src_shader.GetSourceAssetSubIdentifier("mdl")
    if src_asset:
        sh.SetSourceAsset(src_asset, "mdl")
        if sub:
            sh.SetSourceAssetSubIdentifier(sub, "mdl")
            sh.CreateIdAttr(str(sub))
    else:                       # not MDL: old path, close to flat OmniPBR
        sh.SetSourceAsset(Sdf.AssetPath("OmniPBR.mdl"), "mdl")
        sh.SetSourceAssetSubIdentifier("OmniPBR", "mdl")
        sh.CreateIdAttr("OmniPBR")

    # Carry over every input the original **authored** (UV layout, bump strength, ORM use, ...).
    # Unauthored ones keep their MDL defaults, so leave them alone.
    n_kept = 0
    for inp in src_shader.GetInputs():
        v = inp.Get()
        if v is None:
            continue
        nm = inp.GetBaseName()
        if nm in ("diffuse_tint", "albedo_desaturation", "albedo_brightness",
                  "diffuse_color_constant"):
            continue                      # the ones we override
        try:
            sh.CreateInput(nm, inp.GetTypeName()).Set(v)
            n_kept += 1
        except Exception:  # noqa: BLE001
            pass

    c = rgb01(color)
    sh.CreateInput("albedo_desaturation",
                   Sdf.ValueTypeNames.Float).Set(float(blend))
    sh.CreateInput("diffuse_tint", Sdf.ValueTypeNames.Color3f).Set(c)
    sh.CreateInput("albedo_brightness",
                   Sdf.ValueTypeNames.Float).Set(float(brightness))
    # For an MDL without textures this value provides the color.
    sh.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set(c)
    # Matte: blocks sky reflections adding white and collapsing saturation (measured).
    # The roughness **texture** influence is inherited from the original inputs, so variation remains.
    sh.CreateInput("reflection_roughness_constant",
                   Sdf.ValueTypeNames.Float).Set(0.9)
    sh.CreateInput("specular_level", Sdf.ValueTypeNames.Float).Set(0.1)
    sh.CreateInput("metallic_constant", Sdf.ValueTypeNames.Float).Set(0.0)
    mat.CreateSurfaceOutput("mdl").ConnectToSource(sh.ConnectableAPI(), "out")
    mat.GetPrim().SetCustomDataByKey("dg:src_mdl", str(src_asset or "OmniPBR"))
    mat.GetPrim().SetCustomDataByKey("dg:inputs_kept", n_kept)
    return mat


def paint(stage, prim, color):
    """Bind a flat material to every mesh in the subtree.

    Must bind with `strongerThanDescendants` to beat the original materials the prop sets on
    its own children. Returns the number of painted meshes; 0 means painting didn't take, so don't trust it.

    **If `color` is `None`, nothing is painted and 0 is returned.** Street furniture (bench,
    hydrant, street lamp) should keep the pack's original material: painting a park bench lime
    makes the label "lime bench", which contradicts the scene. "No color" is a normal path, not an error.
    """
    from pxr import Usd, UsdGeom, UsdShade

    if color is None:
        return 0
    if color not in PAINT_BGR:
        raise KeyError(f"unknown color {color!r}. Available colors {sorted(PAINT_BGR)}; "
                       f"use None to skip painting (`none` in a spec string)")
    flat = None
    cache = {}
    n = n_blend = 0
    for m in Usd.PrimRange(prim, Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate)):
        # Looking only at Mesh would leave primitives (Cube, Sphere, Cylinder) unpainted;
        # target all Gprims.
        if not m.IsA(UsdGeom.Gprim):
            continue
        use = None
        if PAINT_BLEND > 0.0:
            got = _mdl_shader_of(m)
            if got and got[0]:
                src, omat = got
                # The criterion is **not** "is diffuse_texture authored in USD": these
                # props keep their textures inside the MDL module, not in USD (measured).
                # If an MDL asset is bound, assume the texture is in there and clone it.
                asset = src.GetSourceAsset("mdl")
                if asset and str(asset).strip("@"):
                    key = (str(omat.GetPath()), color)
                    if key not in cache:
                        safe = str(omat.GetPath()).strip("/").replace("/", "_")
                        cache[key] = make_blend_material(
                            stage, f"blend_{color}_{safe}", color, src,
                            blend=PAINT_BLEND, brightness=PAINT_BRIGHTNESS)
                    use = cache[key]
                    n_blend += 1
        if use is None:
            if flat is None:
                flat = make_paint_material(stage, f"{color}", color)
            use = flat
        UsdShade.MaterialBindingAPI.Apply(m).Bind(
            use, UsdShade.Tokens.strongerThanDescendants)
        n += 1
    if n and PAINT_BLEND > 0.0:
        srcs = sorted({str(m.GetPrim().GetCustomDataByKey("dg:src_mdl") or "?")
                       .split("/")[-1].strip("@") for m in cache.values()})
        print(f"[PAINT] {prim.GetName()} {color}: of {n}, "
              f"**{n_blend} kept original material** ({', '.join(srcs) if srcs else '—'}), "
              f"{n - n_blend} flat"
              + ("  ⚠️ no original MDL found at all" if n_blend == 0 else ""),
              flush=True)
    return n


def _unit_scale(stage, usd_path):
    """Ratio of `metersPerUnit` between the prop layer and the stage. 1.0 means no fix needed."""
    from pxr import Usd, UsdGeom

    stage_mpu = UsdGeom.GetStageMetersPerUnit(stage)
    try:
        probe = Usd.Stage.Open(usd_path, load=Usd.Stage.LoadNone)
        src_mpu = UsdGeom.GetStageMetersPerUnit(probe)
    except Exception:  # noqa: BLE001
        src_mpu = stage_mpu
    return float(src_mpu) / float(stage_mpu)


def _up_axis(usd_path):
    """The layer's up-axis. If unreadable, assume same as stage (= no fix)."""
    from pxr import Usd, UsdGeom

    try:
        probe = Usd.Stage.Open(usd_path, load=Usd.Stage.LoadNone)
        return str(UsdGeom.GetStageUpAxis(probe))
    except Exception:  # noqa: BLE001
        return "Z"


# Dynamics schemas to strip from targets. **Collision is kept**: ground probes and LOS use it.
#
# Measured: an iris drone target was placed at `z = +677.20 m`. It was first written off as
# a misread bbox, but **that was wrong.** Measuring the USD directly, the asset is fine
# (0.52 x 0.48 x 0.11 m, Z-up, meters). The real cause is that iris.usd is **a dynamic body**:
# `PhysicsArticulationRootAPI` with a `PhysicsRigidBodyAPI` and `PhysicsRevoluteJoint` per rotor
# (people and CarConcept have 0 physics prims). So it **free-fell** as soon as it spawned at
# PARKING(5000, 5000, 0), where there is no ground. √(2 x 677 / 9.81) = 11.75 s matches the time
# between spawn and base correction. The bbox was read correctly; the drone really was 677 m down.
#
# We move targets by teleporting. With dynamics attached they fall or get pushed from where we
# put them, so **no target may have them**; strip them regardless of kind.
_DYNAMIC_SCHEMAS = (
    "PhysicsRigidBodyAPI", "PhysicsMassAPI", "PhysicsArticulationRootAPI",
    "PhysxRigidBodyAPI", "PhysxArticulationAPI",
)


def freeze_physics(stage, prim):
    """Strip dynamics from a target. Returns: (schemas removed, joints disabled)."""
    from pxr import Usd, UsdPhysics

    nschema = njoint = 0
    for p in Usd.PrimRange(prim, Usd.TraverseInstanceProxies(
            Usd.PrimAllPrimsPredicate)):
        if p.IsInstanceProxy():
            continue
        if "Joint" in str(p.GetTypeName()):
            p.SetActive(False)
            njoint += 1
            continue
        applied = set(p.GetAppliedSchemas())
        for name in _DYNAMIC_SCHEMAS:
            if name not in applied:
                continue
            # The schema is authored in a referenced layer. `RemoveAppliedSchema` writes a
            # delete entry in the edit-target layer that overrides the reference.
            p.RemoveAppliedSchema(name)
            nschema += 1
        # Also disable the flag in case schema removal doesn't take.
        if UsdPhysics.RigidBodyAPI(p):
            attr = UsdPhysics.RigidBodyAPI(p).GetRigidBodyEnabledAttr()
            if attr:
                attr.Set(False)
    return nschema, njoint


def spawn_prop(stage, path, kind, color, pos_m, yaw_deg=0.0, cls=None,
               pack=PACK, verbose=True):
    """Spawn one target with paint, label and collision. Returns: info dict.

    `kind` is a key of `PROPS` (pack prop) or `PRIMS` (primitive).
    `pos_m` is world coordinates in **stage units (= meters)**. z is the height where the
    **base** goes: after spawning, the bbox is measured and the base aligned to it (origins vary per object).

    The reference goes on a **child prim**. If the referenced prim has its own xformOps,
    `AddXformOp` fails when the parent adds the same op (measured: Isaac People characters).
    """
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    if kind in PRIMS:
        return _spawn_primitive(stage, path, kind, color, pos_m, yaw_deg, cls, verbose)
    extra = 1.0
    if kind in VEHICLES:
        rel, noun, default_cls, extra = VEHICLES[kind]
        usd = rel if os.path.isabs(rel) else os.path.join(VEHICLE_ROOT, rel)
    elif kind in PROPS:
        rel, noun, default_cls = PROPS[kind]
        usd = os.path.join(pack, rel)
    else:
        raise KeyError(f"unknown kind {kind!r}. props {sorted(PROPS)} · "
                       f"primitives {sorted(PRIMS)} · vehicles {sorted(VEHICLES)}")
    if not os.path.exists(usd):
        raise FileNotFoundError(usd)

    xf = UsdGeom.Xform.Define(stage, path)
    prim = xf.GetPrim()
    s = _unit_scale(stage, usd) * float(extra)
    # **The up-axis matters too.** External assets (from glTF) are Y-up while USD is Z-up, and
    # neither the file-format plugin nor the asset converter rotates the axis. Measured: the car
    # stood on its nose (CarConcept read as 2.73x1.31x4.36 m, with 4.36 reported as height).
    up = _up_axis(usd)
    op_t = xf.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)
    op_r = xf.AddRotateZOp(UsdGeom.XformOp.PrecisionDouble)
    op_x = xf.AddRotateXOp(UsdGeom.XformOp.PrecisionDouble)
    op_s = xf.AddScaleOp(UsdGeom.XformOp.PrecisionDouble)
    op_x.Set(90.0 if up.upper().startswith("Y") else 0.0)
    op_s.Set(Gf.Vec3d(s, s, s))
    op_r.Set(float(yaw_deg))
    op_t.Set(Gf.Vec3d(float(pos_m[0]), float(pos_m[1]), float(pos_m[2])))
    inner = stage.DefinePrim(path + "/asset")
    if not inner.GetReferences().AddReference(usd):
        raise RuntimeError(f"reference failed: {usd}")

    # Must be frozen **before the first `app.update()`**. Even one physics frame starts it
    # falling at PARKING (no ground), and the base correction then yields a wrong value.
    nfrozen, njoint = freeze_physics(stage, prim)

    nmesh = paint(stage, prim, color)
    ncol = 0
    for m in Usd.PrimRange(prim, Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate)):
        if not m.IsA(UsdGeom.Mesh) or not UsdGeom.Mesh(m).GetPointsAttr().Get():
            continue
        UsdPhysics.CollisionAPI.Apply(m)
        UsdPhysics.MeshCollisionAPI.Apply(m).CreateApproximationAttr().Set(
            UsdPhysics.Tokens.none)
        ncol += 1

    label = cls or default_cls
    try:
        from isaacsim.core.utils.semantics import add_labels
        add_labels(prim, labels=[label], instance_name="class")
    except Exception as exc:  # noqa: BLE001 — a label failure must not block the spawn
        print(f"[TGT] label failed ({path}): {exc!r}", flush=True)

    info = {"prim": path, "kind": kind, "noun": noun, "color": color,
            "class": label, "scale": s, "mesh": nmesh, "collider": ncol,
            "frozen_schemas": nfrozen, "frozen_joints": njoint,
            "hover_m": HOVER_M.get(kind, 0.0),
            # Without a color use the noun only; a label like "None hydrant" must never become
            # an episode sentence (design principle (1): labels come only from what is observable).
            "phrase": (f"{color} {noun}" if color else noun), "op_translate": op_t}
    if verbose:
        print(f"[TGT] {path} — {info['phrase']} · scale {s:g} · mesh {nmesh} · "
              f"collision {ncol} · label {label}", flush=True)
    return info


def _spawn_primitive(stage, path, kind, color, pos_m, yaw_deg, cls, verbose):
    """Primitive target, with paint, label and collision. Moving targets are chosen from these."""
    from pxr import Gf, UsdGeom, UsdPhysics

    shape, noun, default_cls, size = PRIMS[kind]
    xf = UsdGeom.Xform.Define(stage, path)
    op_t = xf.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)
    op_r = xf.AddRotateZOp(UsdGeom.XformOp.PrecisionDouble)
    op_r.Set(float(yaw_deg))
    op_t.Set(Gf.Vec3d(float(pos_m[0]), float(pos_m[1]), float(pos_m[2])))

    gp = path + "/geom"
    if shape == "Cube":
        g = UsdGeom.Cube.Define(stage, gp); g.CreateSizeAttr(float(size))
    elif shape == "Sphere":
        g = UsdGeom.Sphere.Define(stage, gp); g.CreateRadiusAttr(float(size) / 2.0)
    else:
        g = UsdGeom.Cylinder.Define(stage, gp)
        g.CreateRadiusAttr(float(size) / 2.5); g.CreateHeightAttr(float(size))
        g.CreateAxisAttr(UsdGeom.Tokens.z)
    prim = xf.GetPrim()

    nmesh = paint(stage, prim, color)
    gprim = g.GetPrim()
    UsdPhysics.CollisionAPI.Apply(gprim)
    label = cls or default_cls
    try:
        from isaacsim.core.utils.semantics import add_labels
        add_labels(prim, labels=[label], instance_name="class")
    except Exception as exc:  # noqa: BLE001
        print(f"[TGT] label failed ({path}): {exc!r}", flush=True)

    info = {"prim": path, "kind": kind, "noun": noun, "color": color,
            "class": label, "scale": 1.0, "mesh": nmesh, "collider": 1,
            "phrase": (f"{color} {noun}" if color else noun), "op_translate": op_t}
    if verbose:
        print(f"[TGT] {path} — {info['phrase']} · {shape} {size} m · label {label}",
              flush=True)
    return info


SIT_MAX_DZ_M = 5.0     # a base correction larger than this means the bbox was misread


def sit_on_ground(stage, info, ground_z_m, app=None, updates=8,
                  max_dz_m=SIT_MAX_DZ_M):
    """Align the prop's **base** to `ground_z_m`. Needed because each prop has a different origin.

    ⚠️ **An absurd correction is not applied.** Measured: an iris drone target was placed at
    `z = +677.20 m`.

    ⚠️ The cause, corrected. It was first recorded as "the bbox read was wrong", but **it wasn't.**
    Measuring the USD directly, the asset is fine (0.52 x 0.48 x 0.11 m); iris is a dynamic
    body and free-fell for 11.75 s at PARKING, which has no ground. So **the bbox was right and
    the object really was there.** `freeze_physics` now prevents the fall at spawn time.

    The gate stays anyway. "The base is somewhere absurd" means the data must be discarded
    whatever the cause, and this gate is what found the cause. Note how quiet the symptom was:
    the mover pulled it down to the path z so distances looked plausible (15.9 m for a 20 m plan),
    and it surfaced only when the episode start-distance gate caught the 4.1 m gap and dropped
    it as `bad_start`. Without the gate, "episodes differing from the plan" would mix in silently.

    Instead of applying it, **the numbers are printed.** The position stays as the caller gave
    it, so rather than sitting on the ground it stays "where the plan said".
    """
    from pxr import Gf, Usd, UsdGeom

    if app is not None:
        for _ in range(updates):
            app.update()
    prim = stage.GetPrimAtPath(info["prim"])
    rng = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render]
    ).ComputeWorldBound(prim).ComputeAlignedRange()
    if rng.IsEmpty():
        return None
    bottom = rng.GetMin()[2]
    cur = info["op_translate"].Get()
    # Hovering kinds put the base at "ground + altitude" (see `HOVER_M`). The mover inherits
    # `z_offset_m` as is, so that height is kept along the whole path.
    hover = float(info.get("hover_m") or 0.0)
    dz = float(ground_z_m) + hover - float(bottom)
    info["height_m"] = float(rng.GetMax()[2] - rng.GetMin()[2])
    if abs(dz) > float(max_dz_m):
        print(f"[TGT] ⚠️ {info['prim']} ({info.get('phrase')}): base correction "
              f"{dz:+.1f} m is too large (limit {max_dz_m:g}); bbox misread "
              f"(base {float(bottom):+.1f} · height {info['height_m']:.2f}). "
              f"**Not correcting**; the target stays at the plan's z", flush=True)
        info["sit_skipped"] = True
        info["sit_dz_rejected_m"] = round(dz, 2)
        info["z_offset_m"] = float(cur[2])
        return 0.0
    info["op_translate"].Set(Gf.Vec3d(cur[0], cur[1], cur[2] + dz))
    info["z_offset_m"] = float(cur[2] + dz)
    return dz


def ensure_biped_graph(stage, path="/World/DG_biped", verbose=True):
    """Bring the animation graph into the stage once and return its prim.

    `Biped_Setup.usd` contains `AnimationGraph` + `StateMachine` + 12 walk clips.
    **Its demo mesh comes along, so it is set invisible.**

    ⚠️ If even one clip is missing the character **stands in a T-pose, with no error.**
    Measured: two days were lost with only `stand_walk_1` present. For fetching assets see
    scripts/runpod/isaacsim/README.md.
    """
    from pxr import Usd, UsdGeom

    prim = stage.GetPrimAtPath(path)
    if prim and prim.GetChildren():
        return _find_anim_graph(prim)
    usd = os.path.join(PEOPLE_ROOT, BIPED_SETUP)
    if not os.path.exists(usd):
        print(f"[TGT] ⚠️ Biped_Setup missing: {usd}; pedestrians unavailable", flush=True)
        return None
    prim = stage.DefinePrim(path, "Xform")
    if not prim.GetReferences().AddReference(usd):
        print(f"[TGT] ⚠️ Biped_Setup reference failed: {usd}", flush=True)
        return None
    UsdGeom.Imageable(prim).CreateVisibilityAttr("invisible")
    graph = _find_anim_graph(prim)
    if verbose:
        n = sum(1 for q in Usd.PrimRange(prim) if q.GetTypeName() == "SkelAnimation")
        print(f"[TGT] animation graph {graph.GetPath() if graph else None} · "
              f"{n} clips", flush=True)
        if n < 12:
            print(f"[TGT] ⚠️ only {n} clips; fewer than 12 gives a T-pose. "
                  f"Download the whole Animations/ folder (see scripts/runpod/isaacsim/README.md)", flush=True)
    return graph


def _find_anim_graph(root):
    from pxr import Usd

    for q in Usd.PrimRange(root):
        if q.GetTypeName() == "AnimationGraph":
            return q
    return None


def spawn_person(stage, path, kind, pos_m, yaw_deg=0.0, graph=None, verbose=True):
    """Spawn one pedestrian and attach the animation graph.

    The wiring order is the one NVIDIA uses in `isaacsim.replicator.agent.core/stage_util.py`:
    the reference goes on a **child prim** (referenced prims carry their own xformOps), and the
    graph is attached to the character's **SkelRoot** with `ApplyAnimationGraphAPICommand`.
    Setting only `skel:animationSource` gives a T-pose.
    """
    import omni.kit.commands
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdSkel

    if kind not in PEOPLE:
        raise ValueError(f"unknown pedestrian kind {kind!r}; {sorted(PEOPLE)}")
    rel, noun, cls = PEOPLE[kind]
    usd = os.path.join(PEOPLE_ROOT, rel)
    if not os.path.exists(usd):
        print(f"[TGT] ⚠️ character missing: {usd}", flush=True)
        return None
    if graph is None:
        graph = ensure_biped_graph(stage, verbose=verbose)
    if graph is None:
        return None

    holder = stage.DefinePrim(path, "Xform")
    asset = stage.DefinePrim(path + "/asset", "Xform")
    if not asset.GetReferences().AddReference(usd):
        print(f"[TGT] ⚠️ character reference failed: {usd}", flush=True)
        return None
    xf = UsdGeom.Xformable(holder)
    op = None
    for o in xf.GetOrderedXformOps():
        if o.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            op = o
            break
    if op is None:
        op = xf.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)
    op.Set(Gf.Vec3d(*[float(v) for v in pos_m]))
    if yaw_deg:
        xf.AddRotateZOp(UsdGeom.XformOp.PrecisionDouble).Set(float(yaw_deg))

    skelroots = [q for q in Usd.PrimRange(asset) if q.IsA(UsdSkel.Root)]
    if not skelroots:
        print(f"[TGT] ⚠️ {path} has no SkelRoot; nothing to attach the graph to",
              flush=True)
        return None
    paths = [Sdf.Path(q.GetPath()) for q in skelroots]
    omni.kit.commands.execute("RemoveAnimationGraphAPICommand", paths=paths)
    omni.kit.commands.execute("ApplyAnimationGraphAPICommand", paths=paths,
                              animation_graph_path=Sdf.Path(graph.GetPath()))
    info = {"path": path, "prim": path, "kind": kind, "noun": noun, "class": cls,
            "skelroot": str(skelroots[0].GetPath()), "op": op,
            "phrase": noun, "color": None, "person": True}
    if verbose:
        print(f"[TGT] pedestrian {path} ← {kind} ({noun}) · SkelRoot {info['skelroot']}",
              flush=True)
    return info


class PedestrianMover:
    """Walk a pedestrian along a path: **the path is ours, the speed is the clip's.**

    Unlike `WaypointMover`, which moves the transform directly, the animation graph **owns**
    a pedestrian's transform at runtime; moving the USD xform is ignored (measured 0.0098 m/s).
    So the `PathPoints` variable is fed `[current position, next goal]` **every frame** and
    motion matching produces the steps. Feeding it once moves only 0.19 m in 20 s.

    That is why speed isn't a parameter: it is fixed at a measured 1.023 m/s and reproducible
    (two 30 s runs identical to the 4th decimal). Raising it requires retiming the clip.
    """

    #: **Starve it and it dies.** If `PathPoints` stops, the animation graph evaluates a
    #: degenerate Bezier and A segfaults (measured, see `set_walking`). So this mover is
    #: **not** subject to `set_mover_rate` throttling; it is guaranteed an update every step.
    THROTTLEABLE = False

    def __init__(self, skelroot_path, waypoints_m, oneway=True,
                 arrive_m=0.6, verbose=True, start_m=None, prim=None):
        self.skelroot = str(skelroot_path)
        #: Prim path of the target this mover moves. `set_active_movers` selects by it.
        self.prim = None if prim is None else str(prim)
        self.wps = [tuple(float(v) for v in w) for w in waypoints_m]
        if len(self.wps) < 2:
            raise ValueError("fewer than 2 waypoints")
        self.oneway = bool(oneway)
        self.arrive = float(arrive_m)
        self.verbose = bool(verbose)
        self._char = None
        self._i = 1              # next goal index
        self._dir = 1            # direction for round trips
        self._last = None
        self._warned = False
        self._started = False
        # **After play, placement via USD xform doesn't work**: the graph owns the transform
        # and ignores it. The first update sets it with set_world_transform.
        self.start = None if start_m is None else tuple(float(v) for v in start_m)

    def _ensure(self):
        if self._char is not None:
            return True
        try:
            import omni.anim.graph.core as ag
            self._char = ag.get_character(self.skelroot)
        except Exception:  # noqa: BLE001
            self._char = None
        if self._char is None:
            if not self._warned:
                print(f"[TGT] ⚠️ get_character failed {self.skelroot}; "
                      f"the timeline must be playing", flush=True)
                self._warned = True
            return False
        self._char.set_variable("Action", "Walk")
        self._char.set_variable("Walk", 1.0)
        if self.start is not None:
            import carb
            r = carb.Float4(0.0, 0.0, 0.0, 1.0)
            self._char.get_world_transform(carb.Float3(0.0, 0.0, 0.0), r)
            self._char.set_world_transform(carb.Float3(*self.start), r)
        if self.verbose:
            print(f"[TGT] walking {self.skelroot} · {len(self.wps)} waypoints · "
                  f"{'one-way' if self.oneway else 'round trip'} · clip speed "
                  f"{PEOPLE_SPEED_MPS} m/s (nominal; not settable)", flush=True)
        self._started = True
        return True

    def position(self):
        """Current world position held by the graph. This, not the USD xform, is the truth."""
        if self._char is None:
            return None
        import carb
        p = carb.Float3(0.0, 0.0, 0.0)
        r = carb.Float4(0.0, 0.0, 0.0, 1.0)
        self._char.get_world_transform(p, r)
        return (float(p[0]), float(p[1]), float(p[2]))

    def update(self, sim_time_s):
        """Called every step by `install()`. Time is unused: speed belongs to the clip,
        so only path progress is managed."""
        import carb

        if not self._ensure():
            return None
        pos = self.position()
        if pos is None:
            return None
        tgt = self.wps[self._i]
        if math.dist(pos[:2], tgt[:2]) <= self.arrive:
            nxt = self._i + self._dir
            if 0 <= nxt < len(self.wps):
                self._i = nxt
            elif self.oneway:
                self._char.set_variable("Action", "None")   # stop at the end
                self._last = pos
                return pos
            else:
                self._dir *= -1
                self._i = max(0, min(len(self.wps) - 1, self._i + self._dir))
            tgt = self.wps[self._i]
        self._char.set_variable("PathPoints", [
            carb.Float3(pos[0], pos[1], pos[2]),
            carb.Float3(float(tgt[0]), float(tgt[1]), float(tgt[2]))])
        self._last = pos
        return pos

    def rewind(self, active=True):
        """**Actually move** it back to the start. With `active=False` it stands there.

        ⚠️ **Previously it didn't move when `active=True`.** It only cleared `_started` and
        relied on "the next update will re-place it", but `update()` has no such branch;
        `_started` was written in four places and read by none. So rewinding a moving episode
        had no effect on position.

        Why the old batches didn't show it: generating 4 kinds per spot interleaved static
        episodes, and their `rewind(active=False)` teleported the pedestrian, **accidentally**
        acting as a rewind. After switching to the 2+2 layout, 14 moving episodes ran back to
        back and the drift accumulated (measured: planned 8-11 m, but the step-1 distance grew
        monotonically 8 -> 15 -> 28 -> 49 -> **80 m**; on a one-way path the pedestrian was
        standing at the path's end).

        Yet the episodes were judged "success", because the oracle flew 80 m and arrived.
        Episodes that differ from the plan being silently counted as successes is the nature of this bug.
        """
        self._i = 1
        self._dir = 1
        self._last = None
        if self._char is None:
            # **Grab the handle here.** A pedestrian is only placed on its first `update`;
            # if the mover is turned off without ever running, the character doesn't stand at
            # its authored spot (measured: mask 0 from every start pose).
            self._ensure()
        if self._char is None:
            self._started = False
            return
        try:
            import carb
            # **Moving comes first.** The graph owns the transform, but
            # `set_world_transform` works (static episodes already used it).
            # Setting `Action` after the move makes the next `update` re-apply `PathPoints`
            # from the new position and walk the path from the start.
            if self.start is not None:
                r = carb.Float4(0.0, 0.0, 0.0, 1.0)
                self._char.get_world_transform(carb.Float3(0.0, 0.0, 0.0), r)
                self._char.set_world_transform(carb.Float3(*self.start), r)
            self._char.set_variable("Action", "Walk" if active else "None")
            self._started = True         # placed
        except Exception as exc:  # noqa: BLE001 — a rewind must not block the flight
            self._started = False
            print(f"[TGT] ⚠️ pedestrian rewind failed: {exc!r}", flush=True)

    def revalidate(self, stage=None):
        """Drop the character handle after a deactivation; `_ensure` re-acquires it."""
        self._char = None
        return True

    def set_walking(self, flag):
        """Turn walking on/off. **When `update` is paused, this must be off too.**

        ⚠️ Measured: the takeoff gate blocked only `update` and left `Action` at "Walk".
        The character started walking but no new `PathPoints` arrived, so the animation graph
        evaluated a **degenerate Bezier curve** and A segfaulted
        (`BezierSpline::NearestPointOnCurve` / `NodeDataMotionMatchingInstance`).
        The PX4 moving batch collapsed after 2 episodes; the 20 static ones without the gate were fine.
        """
        if self._char is None:
            return
        try:
            self._char.set_variable("Action", "Walk" if flag else "None")
        except Exception as exc:  # noqa: BLE001
            print(f"[TGT] ⚠️ pedestrian walk toggle failed: {exc!r}", flush=True)


class WaypointMover:
    """Transform mover that goes back and forth by arc length. 0.25% effective-speed error (measured).

    `set_walking` has nothing to do here: the transform is written directly, so there is no
    animation graph. The name is kept so callers needn't care about the kind.

    `oneway=True` stops at the end. AirSim benchmark episodes use `oneway: false`, so round trip
    is the default, and **distance traveled in an episode is twice the path length**; whether the
    90 m path spec means one way or round trip is an episode-design decision.
    """

    #: An update is only a **USD attribute write** (`op.Set`), so skipping one doesn't corrupt
    #: state; the next update recomputes position from sim time. Subject to `set_mover_rate`.
    THROTTLEABLE = True

    def __init__(self, op_translate, waypoints_m, speed_mps=0.15, oneway=False,
                 z_offset_m=None, prim=None):
        self.op = op_translate
        #: Prim path of the target this mover moves. `set_active_movers` selects by it.
        self.prim = None if prim is None else str(prim)
        self.wps = [tuple(float(v) for v in w) for w in waypoints_m]
        if len(self.wps) < 2:
            raise ValueError("fewer than 2 waypoints")
        self.speed = float(speed_mps)
        self.oneway = bool(oneway)
        self.z = z_offset_m
        self.seg = [math.dist(self.wps[k][:2], self.wps[k + 1][:2])
                    for k in range(len(self.wps) - 1)]
        self.total = sum(self.seg)
        self.t0 = None
        self._last = None

    def set_walking(self, flag):
        """Nothing to do; there is no animation graph (see class docstring)."""

    def revalidate(self, stage=None):
        """Revive the `op` handle after `SetActive(False)->True`.

        Deactivating a prim removes its subtree from composition, and a held `UsdAttribute`
        **becomes invalid**. Reactivating brings the prim back but the cached handle stays dead:
        `op.Set()` silently does nothing and the target freezes at its start. Silent is the
        dangerous kind, so re-acquire it by path.
        """
        try:
            if self.op and self.op.IsValid():
                return True
        except Exception:  # noqa: BLE001 — a dead handle may throw on access
            pass
        if stage is None or not self.prim:
            return False
        try:
            pr = stage.GetPrimAtPath(self.prim)
            if not pr or not pr.IsValid():
                return False
            att = pr.GetAttribute("xformOp:translate")
            if not att or not att.IsValid():
                return False
            self.op = att
            return True
        except Exception as exc:  # noqa: BLE001
            print(f"[TGT] ⚠️ op handle recovery failed {self.prim}: {exc!r}", flush=True)
            return False

    def at(self, s):
        """Position at arc length `s` (m). For round trips the path folds back."""
        L = self.total
        if L <= 0:
            return self.wps[0]
        if self.oneway:
            s = min(max(s, 0.0), L)
        else:
            s = s % (2 * L)
            if s > L:
                s = 2 * L - s
        acc = 0.0
        for k, d in enumerate(self.seg):
            if acc + d >= s or k == len(self.seg) - 1:
                t = 0.0 if d <= 0 else (s - acc) / d
                a, b = self.wps[k], self.wps[k + 1]
                return (a[0] + (b[0] - a[0]) * t,
                        a[1] + (b[1] - a[1]) * t,
                        a[2] + (b[2] - a[2]) * t)
            acc += d
        return self.wps[-1]

    def update(self, sim_time_s):
        """Update position from sim time. Using wall time would make speed depend on RTF;
        that mistake halved turns and forward moves."""
        from pxr import Gf

        if sim_time_s is None:
            return None
        if self.t0 is None:
            self.t0 = float(sim_time_s)
        s = self.speed * max(0.0, float(sim_time_s) - self.t0)
        p = self.at(s)
        z = p[2] if self.z is None else self.z
        self.op.Set(Gf.Vec3d(p[0], p[1], z))
        self._last = (p[0], p[1], z)
        return self._last

    def rewind(self, active=True):
        """Return to the start of the path, **including writing the transform now.**

        **Call this every episode.** If only B restarts while A stays up, the target keeps
        walking, so episode 2 starts where episode 1 ended and the episodes are no longer
        the same task.

        Clearing `t0` alone isn't enough. In episodes with movement **off** (approaching,
        searching) `update` is never called, so the target stays where the previous episode
        left it, making a static target's position depend on the previous result. Hence it is set directly here.
        """
        self.t0 = None
        p = self.at(0.0)
        z = p[2] if self.z is None else self.z
        try:
            from pxr import Gf
            self.op.Set(Gf.Vec3d(p[0], p[1], z))
            self._last = (p[0], p[1], z)
        except Exception as exc:  # noqa: BLE001 — a rewind must not block the flight
            self._last = None
            print(f"[TGT] ⚠️ rewind transform write failed: {exc!r}", flush=True)

    @property
    def position(self):
        return self._last


def find_street_run(ground, x_range_m, y_range_m, ref_z_m, tol_m=0.5,
                    nx=41, ny=21, min_cells=8, verbose=True):
    """Find a long straight stretch at the same height as `ref_z_m`.

    Samples the ground on a grid and picks the column (fixed y) with the longest run of cells
    within `ref_z_m ± tol_m`. **The reference is the given `ref_z_m`, not the site's minimum
    ground**; using the minimum picks some unrelated low area (same mistake as the spawn-z bug).

    Returns `(p0, p1, length_m, stats)`, or `(None, None, 0.0, stats)` if none is found.
    """
    x0, x1 = float(x_range_m[0]), float(x_range_m[1])
    y0, y1 = float(y_range_m[0]), float(y_range_m[1])
    xs = [x0 + (x1 - x0) * i / (nx - 1) for i in range(nx)]
    ys = [y0 + (y1 - y0) * j / (ny - 1) for j in range(ny)]

    grid = [[ground(x, y) for y in ys] for x in xs]
    hit = sum(1 for row in grid for v in row if v is not None)
    flat = sum(1 for row in grid for v in row
               if v is not None and abs(v - ref_z_m) <= tol_m)

    best = None            # (length in cells, j, i_start, i_end)
    for j in range(ny):
        run_start = None
        for i in range(nx + 1):
            ok = (i < nx and grid[i][j] is not None
                  and abs(grid[i][j] - ref_z_m) <= tol_m)
            if ok and run_start is None:
                run_start = i
            elif not ok and run_start is not None:
                length = i - run_start
                if length >= min_cells and (best is None or length > best[0]):
                    best = (length, j, run_start, i - 1)
                run_start = None

    stats = {"grid": f"{nx}x{ny}", "hit": hit, "flat": flat,
             "ref_z_m": round(float(ref_z_m), 3), "tol_m": tol_m}
    if best is None:
        if verbose:
            print(f"[TGT] no flat stretch found; grid {nx}x{ny} · ground hits {hit} · "
                  f"within {ref_z_m:+.2f} m ±{tol_m} m: {flat}", flush=True)
        return None, None, 0.0, stats

    _, j, ia, ib = best
    p0 = (xs[ia], ys[j], float(grid[ia][j]))
    p1 = (xs[ib], ys[j], float(grid[ib][j]))
    length = math.dist(p0[:2], p1[:2])
    stats.update({"y_m": round(ys[j], 2), "cells": ib - ia + 1})
    if verbose:
        print(f"[TGT] flat stretch y={ys[j]:+.1f} m · x {xs[ia]:+.1f}→{xs[ib]:+.1f} m "
              f"(length {length:.1f} m · {ib - ia + 1} cells) · ref {ref_z_m:+.2f} m ±{tol_m}",
              flush=True)
        print(f"[TGT] grid {nx}x{ny} · ground hits {hit} · at ref height {flat}", flush=True)
    return p0, p1, length, stats


# Installed movers and their on/off. **The 4 DroneCATS kinds are (static, moving) x (visible,
# not visible)**, so one scene must yield both static and moving episodes. Restarting A per scene
# makes loading the throughput bottleneck (Brownstone is 20 GB), so this switch is toggled at reset.
_CONTROL = {"movers": [], "moving": True,
            # Takeoff gate; used only in real flight (PX4). See `set_takeoff_gate` below.
            "gate_probe": None, "gate_alt_m": 0.0, "airborne": False,
            # Mover update rate; 0 = every step (legacy behavior). See `set_mover_rate`.
            "mover_hz": 0.0, "mover_last_t": None,
            # Set of target prims used in this episode. None = all (legacy behavior).
            "active_prims": None,
            # Whether to hide inactive targets. See `set_target_scope`. Default: not hidden.
            "hide_inactive": False, "all_prims": [], "hidden_prims": [],
            "scope_mode": "hide", "scope_ms": 0.0}


def set_takeoff_gate(probe, alt_m):
    """Keep targets **standing still** until the vehicle is airborne. `probe()` returns vehicle altitude (m).

    ⚠️ Why. Measured, first PX4 episode: the planned start distance was 15 m, but at step 1 it
    was **24.5 m**. The rewind at reset worked, but the pedestrian kept walking during the
    following 60 s settle + EKF convergence + takeoff. At RTF 0.11, 60 s of wall time is 6.6 s
    of sim time, but with convergence and takeoff tens of sim seconds pass; 0.3 m/s times that is the error.

    Kinematic episodes don't have this problem: no takeoff, and the pose is set on the first step.
    So the gate is **opt-in**; without it everything runs as before.

    Why altitude signals "episode start": A doesn't see MAVLink, so it doesn't know arming
    state. But the vehicle reaching takeoff altitude can be read directly from the stage, and
    is effectively the same event as "flight has started".
    """
    _CONTROL["gate_probe"] = probe
    _CONTROL["gate_alt_m"] = float(alt_m)
    _CONTROL["airborne"] = False


def set_moving(flag, rewind=True, verbose=True):
    """Turn movers on/off. With `rewind`, rewind to the start position.

    While off, `update` isn't called and targets stand at their last position.
    So **static episodes get "a static target placed at the path start"** (thanks to the rewind).
    """
    flag = bool(flag)
    changed = flag != _CONTROL["moving"]
    _CONTROL["moving"] = flag
    if rewind:
        # `active=flag`: in an off episode it must be "a static target standing at the start"
        rewind_movers(verbose=False, active=flag)
    if verbose and _CONTROL["movers"]:
        print(f"[TGT] moving {'ON' if flag else 'OFF'}"
              f"{'' if changed else ' (unchanged)'}"
              f"{' · rewound' if rewind else ''}"
              f" · {len(_CONTROL['movers'])} movers", flush=True)


def is_moving():
    return bool(_CONTROL["moving"])


def set_mover_rate(hz, verbose=True):
    """Limit the update rate of `THROTTLEABLE` movers. **0 = every step** (legacy behavior).

    Why. The `install` hook updates movers on every `World.step`, i.e. **at the 250 Hz physics
    rate**. A `WaypointMover` update is `op.Set()`, and a USD attribute write is not a value
    store but **a change notification**: PhysX re-reads the shape transform (moving a static
    collider means broadphase re-insertion, the most expensive path), Fabric syncs, and the
    instance in the render acceleration structure is marked dirty. At a target speed of 0.3 m/s,
    **one write moves it 1.2 mm**.

    The effect on observations is update interval x speed. At 20 Hz and 0.3 m/s that is
    **1.5 cm**: 0.3% of a 5 m arrival radius and **under 1 pixel** at 20 m with 640x480, 90° FOV.

    **Pedestrians are exempt** (`PedestrianMover.THROTTLEABLE = False`): starve them and they die.
    """
    hz = float(hz or 0.0)
    if hz < 0:
        raise ValueError(f"mover_hz cannot be negative: {hz}")
    _CONTROL["mover_hz"] = hz
    _CONTROL["mover_last_t"] = None
    if verbose:
        n = sum(1 for m in _CONTROL["movers"] if getattr(m, "THROTTLEABLE", False))
        print(f"[TGT] mover updates {'every step' if hz <= 0 else f'{hz:g} Hz'}"
              f" · {n} throttled / {len(_CONTROL['movers'])} total", flush=True)


def set_active_movers(prims, verbose=True):
    """Move **only the targets used by this episode**. `None` = all (legacy behavior).

    Why. A batch loads all 28 episodes' targets into the scene **at once** (32 pairs + 38
    decoys = 70). An episode actually uses 1 target + 0-3 decoys, yet during moving episodes
    **all 70 movers** run every step: the cost of moving objects nobody looks at.

    ⚠️ **Observations change.** Other episodes' targets stand at their start instead of walking,
    so the image and the GT `targets` entries differ. Don't call this in batches meant to be
    compared with earlier measurements.

    Going through `set_walking(False)` when turning off matters: cutting updates while a
    pedestrian is left at `Action="Walk"` makes the animation graph evaluate a degenerate
    Bezier and A dies.
    """
    _CONTROL["active_prims"] = None if prims is None else {str(p) for p in prims}
    on_n = 0
    for m in _CONTROL["movers"]:
        on = _mover_active(m) and _CONTROL["moving"]
        on_n += int(bool(on))
        fn = getattr(m, "set_walking", None)
        if fn is None:
            continue
        try:
            fn(bool(on))
        except Exception as exc:  # noqa: BLE001 — a toggle failure must not block the flight
            print(f"[TGT] ⚠️ activation toggle failed {type(m).__name__}: {exc!r}", flush=True)
    if verbose and _CONTROL["movers"]:
        which = "all" if prims is None else f"{len(_CONTROL['active_prims'])} specified"
        print(f"[TGT] active movers {on_n}/{len(_CONTROL['movers'])} ({which})", flush=True)
    return on_n


def set_target_scope(stage, all_prims, hide_inactive, verbose=True, mode="hide"):
    """**Remove inactive targets from view.** One step stronger than stopping movers (`set_active_movers`).

    Why. A batch loads 61 targets for 28 episodes at once, but an episode uses **2**
    (measured 3.3%). The rest stay in the render acceleration structure and cost every pass;
    measured, going from 32 to 70 targets dropped RTF 0.11 -> 0.080 (-27% **with no extra rendering**).

    The means is **visibility**. `SetActive(False)` is stronger since it removes physics too,
    but deactivating a prim invalidates the `UsdAttribute` handles held by movers. Toggling
    visibility is a proven path that `_hide_for_silhouette` already uses per target.

    ⚠️ **Observations change substantially.** On average 1.1 (max 5) other targets lie within
    30 m of an episode start, so some actually vanish from view; fewer decoys make the task
    easier. Don't use in batches compared with earlier measurements. On the other hand it is
    **closer to the plan's intent**: the plan designs 0-3 decoys per episode, while 59 extra
    stand in the scene now.
    """
    import time as _time

    from pxr import UsdGeom

    _CONTROL["hide_inactive"] = bool(hide_inactive)
    _CONTROL["scope_mode"] = str(mode)
    if all_prims is not None:
        _CONTROL["all_prims"] = [str(p) for p in all_prims]
    act = _CONTROL["active_prims"]
    hid = []
    if stage is None:
        return 0
    t0 = _time.perf_counter()
    for path in _CONTROL["all_prims"]:
        want = (not _CONTROL["hide_inactive"]) or act is None or path in act
        pr = stage.GetPrimAtPath(path)
        if not pr or not pr.IsValid():
            continue
        try:
            if mode == "deactivate":
                # **Removes physics too.** Inactive prims leave composition, so they vanish
                # from both the render acceleration structure and PhysX. The cost is invalidated handles, revived below.
                pr.SetActive(bool(want))
            else:
                img = UsdGeom.Imageable(pr)
                img.MakeVisible() if want else img.MakeInvisible()
            if not want:
                hid.append(path)
        except Exception as exc:  # noqa: BLE001 — a toggle failure must not block the flight
            print(f"[TGT] ⚠️ target scope toggle failed {path}: {exc!r}", flush=True)
    # Re-acquire handles of revived prims. Without this the targets silently freeze.
    if mode == "deactivate":
        for m in _CONTROL["movers"]:
            fn = getattr(m, "revalidate", None)
            if fn is not None and _mover_active(m):
                try:
                    fn(stage)
                except Exception as exc:  # noqa: BLE001
                    print(f"[TGT] ⚠️ handle recovery failed {type(m).__name__}: {exc!r}",
                          flush=True)
    dt_ms = (_time.perf_counter() - t0) * 1000.0
    _CONTROL["hidden_prims"] = hid
    _CONTROL["scope_ms"] = round(dt_ms, 1)
    if verbose and _CONTROL["all_prims"]:
        print(f"[TGT] target scope ({mode}): removed {len(hid)}/"
              f"{len(_CONTROL['all_prims'])} · {dt_ms:.0f} ms"
              + ("" if hid else " (all kept)"), flush=True)
    return len(hid)


def hidden_targets():
    """Target paths currently hidden; the GT sidecar drops them from its rows."""
    return list(_CONTROL["hidden_prims"])


def _mover_active(m):
    """Whether this mover should move in this episode.

    **Movers with unknown paths are not turned off**: when unsure, keeping legacy behavior is safer.
    """
    act = _CONTROL["active_prims"]
    if act is None:
        return True
    prim = getattr(m, "prim", None)
    return prim is None or str(prim) in act


def _mover_due(sim_time_s):
    """Whether throttled movers are due for an update this step. Time is **sim time**.

    Measuring wall time makes target speed depend on RTF, a trap this file has hit twice already.
    """
    hz = _CONTROL["mover_hz"]
    if hz <= 0:
        return True
    if sim_time_s is None:
        return True
    last = _CONTROL["mover_last_t"]
    # `sim_time_s < last` means the timeline was rewound; update immediately.
    if last is None or sim_time_s < last or (sim_time_s - last) >= (1.0 / hz):
        _CONTROL["mover_last_t"] = float(sim_time_s)
        return True
    return False


def rewind_movers(verbose=True, active=None):
    """Rewind all movers to their start. **Call every episode**; otherwise episode 2 starts
    where episode 1 ended and it is no longer "the same task".

    Without `active`, the current moving state is used."""
    n = 0
    if active is None:
        active = _CONTROL["moving"]
    # If the takeoff gate is set, **walking is not turned on.** Otherwise the character starts
    # walking while the gate blocks `update`, no new `PathPoints` arrive, and the animation graph
    # dies evaluating a degenerate curve (see `PedestrianMover.set_walking`).
    if _CONTROL["gate_probe"] is not None:
        active = False
    for m in _CONTROL["movers"]:
        fn = getattr(m, "rewind", None)
        if fn is None:
            continue
        try:
            # Inactive movers **stand at the start**: rewound, with walking off.
            fn(active=bool(active) and _mover_active(m))
            n += 1
        except Exception as exc:  # noqa: BLE001
            print(f"[TGT] ⚠️ rewind failed {type(m).__name__}: {exc!r}", flush=True)
    # **The next episode waits for takeoff again.** Without this the gate applies only to the first.
    _CONTROL["airborne"] = False
    # Reset the throttle phase per episode too; otherwise one period is skipped right after a rewind.
    _CONTROL["mover_last_t"] = None
    if verbose and n:
        print(f"[TGT] rewound {n} movers"
              + (" · back to waiting for takeoff" if _CONTROL["gate_probe"] else ""), flush=True)
    return n


def install(movers, world_cls=None, verbose=True):
    """Advance movers after each `World.step`, without modifying Pegasus code.

    Time is the `omni.timeline` sim time (not wall clock).
    """
    if world_cls is None:
        from isaacsim.core.api.world import World as world_cls
    orig = world_cls.step
    state = {"n": 0}
    _CONTROL["movers"] = movers

    def step(self, render=True, *a, **k):
        out = orig(self, render, *a, **k)
        try:
            if not _CONTROL["moving"]:
                return out
            if _CONTROL["gate_probe"] is not None and not _CONTROL["airborne"]:
                try:
                    alt = _CONTROL["gate_probe"]()
                except Exception:  # noqa: BLE001 — a probe failure must not block the flight
                    alt = None
                if alt is None or float(alt) < _CONTROL["gate_alt_m"]:
                    return out          # still on the ground; targets stand at their start
                _CONTROL["airborne"] = True
                # **Only now is walking turned on.** It stayed still while the gate was closed.
                for m in movers:
                    fn = getattr(m, "set_walking", None)
                    if fn is not None:
                        fn(True)
                print(f"[TGT] takeoff detected: altitude {float(alt):.2f} m ≥ "
                      f"{_CONTROL['gate_alt_m']:.2f} m · targets start moving", flush=True)
            import omni.timeline
            tl = omni.timeline.get_timeline_interface()
            if not tl.is_playing():
                return out
            t = tl.get_current_time()
            # Throttle **only eligible movers**. Pedestrians must update every step.
            due = _mover_due(t)
            for m in movers:
                if not _mover_active(m):
                    continue        # target not used in this episode; stands at its start
                if due or not getattr(m, "THROTTLEABLE", False):
                    m.update(t)
            state["n"] += 1
        except Exception:  # noqa: BLE001 — movers must not block the flight
            pass
        return out

    world_cls.step = step
    if verbose:
        print(f"[TGT] {len(movers)} movers · World.step hook installed", flush=True)
    return movers
