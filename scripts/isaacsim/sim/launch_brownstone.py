# DroneCATS
# Copyright (c) 2026-present NAVER Cloud Corp.
# Apache-2.0

"""Launch `pegasus_sim.py` with the Brownstone scene, without modifying it.

How it works
  1. **Import** `pegasus_sim`. `main()` is guarded by `if __name__ == "__main__"`,
     so the import only boots SimulationApp and does not fly.
  2. Inject the Brownstone variants into `SIMULATION_ENVIRONMENTS`. `pegasus_sim` refers
     to the same dict object, so the `--env` gate sees them (checked by id).
  3. Wrap `PegasusInterface.load_environment` to run **scene prep** right after loading:
     group labelling (must happen before play) and collision setup.
  4. Wrap `pegasus_layout.spawn_for` to add the **ground height at the spawn point**
     (`spawn_ground_m`) to spawn z. Pegasus spawns at world z = altitude, so a non-zero
     ground needs correcting. After play the real ground is re-measured and a mismatch
     warns (disable with `--no-spawn-check`).
  5. With `--gt`, attach the GT sidecar (process A). By default it ticks with camera frames.
  6. Call `pegasus_sim.main()`.

Usage
  $ISAACSIM/python.sh scripts/isaacsim/sim/launch_brownstone.py --variant Morning [pegasus_sim args...]
"""
import argparse
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))


def _guard_slot(mine):
    """Refuse to start if this slot's MAVLink port is **already in use**.

    Why: A stays alive as a sim server. Launching another on the same slot without
    stopping the first leads to:

        · the new A's PX4 backend dies with `OSError: [Errno 98] Address already in use`
        · **but A keeps going**, so it can look normal
        · both write the same `--gt-dir` and `latest_gt.json`. GT frames mix and the
          oracle acts on another run's GT: **plausible data with no error**
        · two Brownstone instances on one GPU use 12 GB + 8 GB

    This contamination is invisible and practically undetectable afterwards, so block it **before start**.
    Ports are checked because slot files survive a dead process, while ports are kernel-managed.
    """
    import socket

    # The convention is set by `scripts/isaacsim/pegasus_sim.py`: `4560 + vehicle_id`, and the first
    # vehicle's id is the slot number. Verified: slot 4 -> 4564, slot 6 -> 4566.
    # (Multi-vehicle uses slot+i ports, but a taken first port already means a clash.)
    port = 4560 + int(mine.slot)
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        srv.bind(("127.0.0.1", port))
    except OSError:
        print(f"[DG] ✗ slot {mine.slot} port {port} is already in use: "
              f"an A on the same slot is alive. Starting anyway would share the GT dir, "
              f"mixing frames (plausible data **with no error**), "
              f"and load two scenes on the GPU. Stop that A first or change --slot.",
              flush=True)
        print(f"[DG]   check: ss -ltnp | grep {port}", flush=True)
        srv.close()
        raise SystemExit(4)
    finally:
        try:
            srv.close()
        except OSError:
            pass


def parse_targets(spec):
    """Comma list of 'kind:color[:move]' -> [{kind, color, move}].

    ⚠️ **`none` and the empty string mean "do not paint"**, not a colour name.
    Passing them through breaks two ways: (1) `PAINT_BGR['none']` raises KeyError and the
    target is not spawned, (2) surviving targets get the label "none hydrant". Both were
    observed; (1) was worse because only the **first** `none` target failed (the paint
    material is cached by path, so later ones passed without an exception).

    Street furniture is unpainted by default: a lime park bench would be labelled
    "lime bench" and disagree with the scene. That is also why the plans use
    `none` for colourless kinds.
    """
    out = []
    for chunk in (spec or "").split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        bits = [b.strip() for b in chunk.split(":")]
        if len(bits) < 2:
            raise ValueError(f"target spec must be 'kind:color[:move]': {chunk!r}")
        # `targets.py` owns the convention; re-deciding here would let two parsers drift
        from sim import targets as _TG
        color = _TG.as_color(bits[1])
        out.append({"kind": bits[0], "color": color,
                    "move": len(bits) > 2 and bits[2].lower() in ("move", "moving", "1", "true")})
    return out


def prep_note(shared, key, value):
    """Record a prep count (labels, collisions, ground fixes, targets) in the shared state."""
    shared.setdefault("prep", {})[key] = value


def _plan_region(mine, U, margin_m=30.0):
    """XY extent covered by the `--target-at` coordinates; the ground is probed here.

    The whole-stage bbox is 5 km (dome lights and far prims stretch it). Probing where
    episodes actually happen is right, and that is where the ground is visible anyway.
    Without coordinates, return `None` and let the caller fall back to the bbox.
    """
    xs, ys = [], []
    if getattr(mine, "target_at", None):
        for chunk in str(mine.target_at).split(";"):
            head = chunk.split(":")[0].strip()
            if not head:
                continue
            try:
                v = [float(t) for t in head.split(",")]
                xs.append(v[0]); ys.append(v[1])
            except (ValueError, IndexError):
                pass
    if len(xs) < 2:
        return None
    m = margin_m * U
    return (min(xs) * U - m, max(xs) * U + m,
            min(ys) * U - m, max(ys) * U + m)


def _count_placed(stage, shared, parking_far_m=1000.0):
    """Number of targets that **actually left** PARKING. The only evidence of placement."""
    import math as _m

    from pxr import Usd, UsdGeom

    from sim import targets as TG

    mpu = UsdGeom.GetStageMetersPerUnit(stage)
    px, py, _pz = TG.PARKING
    n = 0
    for info in (shared.get("targets") or []):
        path = info.get("prim") if isinstance(info, dict) else None
        if not path:
            continue
        pr = stage.GetPrimAtPath(path)
        if not (pr and pr.IsValid()):
            continue
        try:
            t = UsdGeom.Xformable(pr).ComputeLocalToWorldTransform(
                Usd.TimeCode.Default()).ExtractTranslation()
        except Exception:  # noqa: BLE001
            continue
        if _m.hypot(float(t[0]) * mpu - px, float(t[1]) * mpu - py) > parking_far_m:
            n += 1
    return n


def _fix_ground(stage, SP, shared, mine):
    """Restore texture tiling of ground materials. Must be called **after physics plays**.

    ⚠️ Why after play: the ground is measured as "what a downward ray hits", and that
    ray is a PhysX scene query. Called during prep (before play), **all 196 rays miss**
    and "0 materials" results; measured, and 16 episodes were built while silently
    fixing nothing. It is now called from the same hook as the spawn check.

    ⚠️ **Right after play is still too early.** PhysX cooks ~8k triangle meshes in sequence.
    One spawn spot being ready does not mean the whole site is; in a second measurement
    only **1** of 196 rays hit (foliage). So the hit count is returned and the caller
    calls again until enough is cooked.

    Returns: `(materials fixed, rays hit)`

    Brownstone is authored in cm, so `/World/layout` gets scale 0.01. Sidewalk/road
    materials use world-space triplanar projection, so UVs come from world coordinates and
    100x smaller coordinates stretch the texture 100x. Hence restore **authored x (1/layout scale)**.
    """
    if mine.ground_texture_scale == 0:
        return 0, 999                   # "disabled, so done"
    try:
        ls = shared.get("layout_scale") or 1.0
        fac = (1.0 / ls) if ls and ls > 0 else 1.0
        tgt = None
        if mine.ground_texture_scale > 0:
            tgt, fac = mine.ground_texture_scale, None   # manual override
        ray, _g, mpu, U, rng = SP.make_probes(stage)
        found = SP.ground_materials(stage, ray, rng, mpu, U,
                                    region=_plan_region(mine, U))
        hits = found.get("__hits__", 0)
        n_fix, _seen, _n = SP.fix_ground_materials(
            found, factor=fac, target_scale=tgt,
            desaturate=(None if mine.ground_desaturate < 0
                        else mine.ground_desaturate),
            below=mine.ground_texture_below)
        prep_note(shared, "ground_hits", int(hits))
        prep_note(shared, "ground_fixed", int(n_fix))
        return n_fix, hits
    except Exception as exc:  # noqa: BLE001
        print(f"[PREP] ground material fix failed ({exc!r}); keeping scene values",
              flush=True)
        return 0, 999


# Anti-aliasing / upscaling: **directly affects data quality.**
#
# Measured: launched with `--camera-width 640`, the log said:
#
#     [rtx.postprocessing.plugin] DLSS increasing input dimensions:
#       Render resolution of (320, 240) is below minimal input resolution of 300.
#
# i.e. **the actual render was 320x240 and DLSS upscaled it to 640x480**. At 3 m altitude
# gravel ground smears and building textures vanish. Every frame the model sees was
# upscaled, which is bad benchmark input. Textures were fine (4096², 2048²,
# all 3.48 GB present); the problem was render resolution.
#
#   dlss   default. Render at half resolution and upscale with AI. Fast but blurry
#   dlaa   **native resolution** with AI anti-aliasing. Quality first
#   fxaa   native + cheap post-process AA. Fallback where dlaa is unavailable
#   off    native, no AA. Jaggies visible
_AA_OPS = {"off": 0, "taa": 1, "fxaa": 2, "dlss": 3, "dlaa": 4}


def _set_render_aa(mode):
    """Select `/rtx/post/aa/op`. Must be called **after** the app is up."""
    if not mode or mode == "keep":
        return
    op = _AA_OPS.get(mode)
    if op is None:
        print(f"[RENDER] unknown --render-aa {mode!r}; valid: "
              f"{sorted(_AA_OPS)} · using default", flush=True)
        return
    try:
        import carb.settings
        st = carb.settings.get_settings()
        st.set("/rtx/post/aa/op", op)
        if mode in ("dlaa", "off", "fxaa", "taa"):
            # Even with DLSS off, a remaining upscale path cuts the resolution again
            st.set("/rtx/post/dlss/enabled", False)
            st.set("/rtx/post/scaling/enabled", False)
        got = st.get("/rtx/post/aa/op")
        print(f"[RENDER] anti-aliasing {mode} (op={got}): "
              + ("rendering at native resolution" if mode != "dlss"
                 else "half resolution + AI upscale"), flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[RENDER] AA setup failed, continuing with default: {exc!r}", flush=True)


#: `omni.anim.graph.core` drives the pedestrians. Isaac Sim 5.1 enables it as a dependency of
#: `omni.anim.people`; 6.0.1 ships it but leaves it off, and every pedestrian then logs
#: `get_character failed` and stands still for the whole run (measured: 0.00 m after takeoff).
#: It goes on the Kit command line: enabling it after the app is up is **not** enough on 6.0.1
#: (measured: `getCharacter ... is not a SkelRoot`, still no walking). A no-op on 5.1.
_KIT_ENABLE = ["--enable", "omni.anim.graph.core"]


def _check_anim_graph():
    """Log whether `omni.anim.graph.core` is on. Must be called **after** the app is up."""
    try:
        import omni.kit.app
        on = omni.kit.app.get_app().get_extension_manager().is_extension_enabled(
            "omni.anim.graph.core")
    except Exception as exc:  # noqa: BLE001
        on = f"unknown ({exc!r})"
    print(f"[TGT] omni.anim.graph.core enabled: {on}"
          + ("" if on is True else " ⚠️ pedestrians will not walk"), flush=True)


def _place_targets(stage, SP, spawn_rows, mine, shared):  # noqa: C901
    """After play, settle targets on the ground and fix their paths.

    Before play there is no physics raycast, so the ground is unknown. Spawning is done
    earlier (labels must be attached before play); only **positions** are decided here.
    """
    from pxr import Gf

    from sim import targets as TG

    try:
        ray, ground, mpu, U, _rng = SP.make_probes(stage)
    except Exception as exc:  # noqa: BLE001
        print(f"[DG] target placement: physics probe failed: {exc!r}", flush=True)
        return

    # Reference height is **the ground at the drone spawn point**. The site's lowest ground
    # picks some unrelated low area; the same mistake as the spawn z bug.
    ref = next((r["ground_m"] for r in spawn_rows if r["ground_m"] is not None), None)
    if ref is None:
        print("[DG] target placement: no reference ground, skipping", flush=True)
        return
    # spawn coordinates: reference point for frontal/relative paths
    cx0, cy0 = 0.0, 0.0
    for r in spawn_rows:
        if r.get("xyz_m"):
            cx0, cy0 = r["xyz_m"][0], r["xyz_m"][1]
            break

    if mine.target_path:
        try:
            a_, b_ = mine.target_path.split(":")
            ax, ay = (float(v) for v in a_.split(","))
            bx, by = (float(v) for v in b_.split(","))
        except ValueError:
            print(f"[DG] --target-path must be 'x0,y0:x1,y1': "
                  f"{mine.target_path!r}", flush=True)
            return
        ga, gb = ground(cx0 + ax, cy0 + ay), ground(cx0 + bx, cy0 + by)
        p0 = (cx0 + ax, cy0 + ay, ref if ga is None else ga)
        p1 = (cx0 + bx, cy0 + by, ref if gb is None else gb)
        length = math.dist(p0[:2], p1[:2])
        print(f"[TGT] given path ({p0[0]:+.1f}, {p0[1]:+.1f}) -> "
              f"({p1[0]:+.1f}, {p1[1]:+.1f}) · length {length:.1f} m · "
              f"ground {p0[2]:+.2f}/{p1[2]:+.2f} m", flush=True)
    else:
        a = float(mine.target_area)
        p0, p1, length, _stats = TG.find_street_run(
            ground, (cx0 - a, cx0 + a), (cy0 - a, cy0 + a), ref, tol_m=0.5)
    if p0 is None:
        print("[DG] target placement: no flat run, placing static near the spawn point", flush=True)
        p0 = p1 = (4.0, 0.0, ref)     # 4 m ahead of the spawn point
        length = 0.0

    sweep = None
    if length > 0:
        try:
            sweep = SP.sweep_path_validity(ray, ground, 1.0, 1.0,
                                           p0[:2], p1[:2])
        except Exception as exc:  # noqa: BLE001
            print(f"[DG] path validity check failed: {exc!r}", flush=True)

    # Per-target positions (and paths), relative to spawn: same convention as `--target-path`.
    at = []
    if mine.target_at:
        for chunk in mine.target_at.split(";"):
            chunk = chunk.strip()
            if not chunk:
                continue
            try:
                head, _, tail = chunk.partition(":")
                pos = tuple(float(v) for v in head.split(","))
                vec = tuple(float(v) for v in tail.split(",")) if tail else None
                if len(pos) != 2 or (vec is not None and len(vec) != 2):
                    raise ValueError
                at.append((pos, vec))
            except ValueError:
                print(f"[DG] could not parse --target-at item: {chunk!r} "
                      f"(format 'X,Y' or 'X,Y:DX,DY')", flush=True)
    if at:
        print(f"[DG] {len(at)} per-target positions given; they override path/front placement",
              flush=True)

    front = []
    if mine.target_front:
        try:
            front = [float(v) for v in mine.target_front.split(",") if v.strip()]
        except ValueError:
            print(f"[DG] --target-front must be comma-separated metres: "
                  f"{mine.target_front!r}", flush=True)

    if shared["stream"] is not None:
        shared["stream"].set_ray(ray)

    movers = []
    static_i = 0
    for k, info in enumerate(shared["targets"]):
        # Select by spec index (see the `spec_index` comment above); without it (old path), in order.
        si = info.get("spec_index", k)
        own = at[si] if si < len(at) else None
        # Static targets are spread along the path; moving targets start at the path start
        if own is not None:
            sx, sy = cx0 + own[0][0], cy0 + own[0][1]
            g = ground(sx, sy)
            sz = ref if g is None else g
        elif front and not info["move"]:
            # Front placement: yaw 0 so +X is the view centre. Relative to the (moved) spawn.
            d = front[min(static_i, len(front) - 1)]
            static_i += 1
            sx, sy = cx0 + d, cy0
            g = ground(sx, sy)
            sz = ref if g is None else g
        elif info["move"] or length <= 0:
            sx, sy, sz = p0
        else:
            t = (k + 1) / (len(shared["targets"]) + 1)
            sx = p0[0] + (p1[0] - p0[0]) * t
            sy = p0[1] + (p1[1] - p0[1]) * t
            sz = p0[2] + (p1[2] - p0[2]) * t
        op = info.get("op_translate") or info.get("op")
        op.Set(Gf.Vec3d(sx, sy, sz))
        # `app=None`: this function is called **inside** `World.step`; calling `app.update()`
        # there would re-enter. The bbox is read straight from USD transforms, so no app tick is needed.
        if not info.get("person"):
            TG.sit_on_ground(stage, info, sz)
        # This target's route: its **own vector** if it got its own position, else the shared path
        if own is not None:
            if own[1] is None:
                leg = None                       # position only = static
            else:
                leg = ((sx, sy), (sx + own[1][0], sy + own[1][1]))
        else:
            leg = ((p0[0], p0[1]), (p1[0], p1[1])) if length > 0 else None
        if info.get("person") and info["move"] and leg:
            # For pedestrians **the anim graph owns the transform**; moving the xform with
            # WaypointMover is ignored (measured 0.0098 m/s). Feed PathPoints every frame.
            movers.append(TG.PedestrianMover(
                info["skelroot"],
                [(leg[0][0], leg[0][1], sz), (leg[1][0], leg[1][1], sz)],
                oneway=mine.target_oneway, start_m=(sx, sy, sz),
                prim=info.get("prim")))
        elif info["move"] and leg:
            z = info.get("z_offset_m", sz)
            movers.append(TG.WaypointMover(
                info["op_translate"],
                [(leg[0][0], leg[0][1], z), (leg[1][0], leg[1][1], z)],
                speed_mps=mine.target_speed, oneway=mine.target_oneway,
                z_offset_m=z, prim=info.get("prim")))
        # **Effective speed depends on the target kind.** For pedestrians the anim graph owns
        # root motion, so `--target-speed` has no effect; it is fixed by the clip (measured).
        # Logging the requested value would be a lie, so log per kind.
        moving = bool(info["move"] and (leg if own is not None else length > 0))
        info["mover"] = (("pedestrian" if info.get("person") else "waypoint")
                         if moving else None)
        info["speed_mps"] = ((TG.PEOPLE_SPEED_MPS if info.get("person")
                              else mine.target_speed) if moving else 0.0)
        info["speed_settable"] = (not info.get("person")) if moving else None
        print(f"[DG] target {info.get('prim') or info['path']}: {info['phrase']} @ "
              f"({sx:+.1f}, {sy:+.1f}, {info.get('z_offset_m', sz):+.2f}) m · "
              f"height {info.get('height_m', 0):.2f} m · "
              + ("moving" if moving else "static")
              + ((f" · speed {info['speed_mps']:.3f} m/s"
                  + (" (fixed by clip, not settable)" if info.get("person") else ""))
                 if moving else ""), flush=True)
        # GT meta was already written at spawn; speed is only known here, so overwrite it
        if shared["stream"] is not None:
            m = getattr(shared["stream"], "target_meta", {}).get(info.get("prim"))
            if m is not None:
                m["speed_mps"] = info["speed_mps"]
                m["mover"] = info["mover"]
                m["speed_settable"] = info["speed_settable"]

    if movers:
        TG.install(movers)
        shared["movers"] = movers
        if mine.mover_hz > 0:
            TG.set_mover_rate(mine.mover_hz)
        if mine.mover_takeoff_gate_m > 0.0:
            # Altitude probe: read world z of `<body>/fpv_cam` found by the GT sidecar.
            # A does not see MAVLink so it does not know arming state, but "climbed to takeoff
            # altitude" is readable from the stage and is the same event as "flight started".
            def _ego_alt():
                import omni.usd                      # imported at call time
                from pxr import Usd, UsdGeom
                stream = shared.get("stream")
                prim_path = getattr(stream, "camera_prim", None) if stream else None
                if not prim_path:
                    return None            # camera not attached yet: treat as on the ground
                st = omni.usd.get_context().get_stage()
                prim = st.GetPrimAtPath(prim_path)
                if not prim or not prim.IsValid():
                    return None
                xf = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
                    Usd.TimeCode.Default())
                return float(xf.ExtractTranslation()[2])

            TG.set_takeoff_gate(_ego_alt, mine.mover_takeoff_gate_m)
            print(f"[DG] moving-target takeoff gate {mine.mover_takeoff_gate_m:g} m: "
                  f"targets stay at their start until the vehicle exceeds that altitude", flush=True)
        span = length * (1.0 if mine.target_oneway else 2.0)
        ped = [t for t in shared["targets"] if t.get("mover") == "pedestrian"]
        spd = TG.PEOPLE_SPEED_MPS if ped else mine.target_speed
        if ped:
            print(f"[DG] ⚠️ {len(ped)} pedestrian(s) ignore --target-speed({mine.target_speed}): "
                  f"the anim graph owns root motion, so speed is fixed at the clip's "
                  f"{TG.PEOPLE_SPEED_MPS} m/s (measured; retiming does not change it, "
                  f"see scripts/runpod/isaacsim/README.md)", flush=True)
        print(f"[DG] path {length:.1f} m · {'one-way' if mine.target_oneway else 'round trip'} "
              f"-> {span:.1f} m per cycle · speed {spd} m/s "
              f"({span / max(spd, 1e-9):.0f} s sim per cycle)", flush=True)
    if sweep:
        # `valid` values are already ratios (float); treating them as lists died with TypeError
        vr = {k: f"{v * 100:.0f}%" for k, v in sweep["valid"].items()}
        print(f"[DG] path validity: segments {sweep['segments']} · per height {vr} · "
              f"blocked {len(sweep['blocked'])} · longest valid at both heights "
              f"{sweep['best_run_m']:.0f} m", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--env", default=None, metavar="NAME",
                    help="Use a **scene from the Pegasus registry** instead of Brownstone "
                         "(e.g. 'Office' · 'Full Warehouse' · 'Hospital'). Indoor registry scenes "
                         "run faster than Brownstone. Brownstone-specific prep (units, lighting, "
                         "material restore) is skipped. "
                         "PX4 runs in lockstep with the simulator (Pegasus `enable_lockstep`), so "
                         "low RTF only slows wall-clock time; physics and sensors are identical")
    ap.add_argument("--variant", default="Morning",
                    help="Morning (default) · Night (DR day/night) · Lite (low-res LOD) · Close (one house, close up)")
    ap.add_argument("--no-prep", action="store_true", help="skip scene prep (debug)")
    ap.add_argument("--keep-env-animation", action="store_true",
                    help="Keep the animation **shipped with the environment asset pack** "
                         "(default: frozen). ⚠️ **Unrelated** to the targets we spawn (pedestrians, "
                         "vehicles): those live under `/World/DG_targets` and are protected by "
                         "`ANIM_NEVER`. This freezes the pack's USD time samples. "
                         "Measured: with time samples, `simulate(dt, t)` "
                         "re-evaluates every time step, **14.2 ms**; with time fixed it is "
                         "**0.137 ms** (104x). Brownstone is a marketing demo with a camera "
                         "fly-through we do not use. "
                         "Use this flag only to reproduce that comparison")
    ap.add_argument("--spawn-yaw", type=float, default=None,
                    help="Initial heading (deg) at launch. Upstream code has no launch arg for this; "
                         "per-episode headings are set through the reset request")
    ap.add_argument("--spawn-xy", default=None,
                    help="Move the drone spawn x,y to these coordinates (m, e.g. '0,-36'). The default origin "
                         "is under the eaves with a blocked view; move it to shoot from the open street")
    ap.add_argument("--targets", default=None,
                    help="Bench target spawns: 'kind:color[:move]', comma-separated. "
                         "e.g. 'box:lime:move,bench:white'. kind is a "
                         "PROPS (pack prop) or PRIMS key from scripts/isaacsim/sim/targets.py. "
                         "⚠️ Motion (:move) only works for PRIMS (box·ball·barrel); pack props "
                         "are street furniture fixed in place")
    ap.add_argument("--target-speed", type=float, default=0.15,
                    help="moving target speed m/s (default 0.15, the benchmark item value). "
                         "⚠️ Does not apply to pedestrians: the anim graph owns root "
                         "motion, so speed is fixed by the clip")
    ap.add_argument("--target-oneway", action="store_true",
                    help="one-way (stop at the end). Default is round trip, so travel is 2x the path")
    ap.add_argument("--target-scope", choices=("all", "hide", "deactivate"),
                    default="all",
                    help="What to do with targets this episode does not use. all (default)=leave them "
                         "· hide=visibility off (not rendered) · "
                         "deactivate=deactivate the prim (removed from **both render and physics**). "
                         "A scene holds many targets and one episode uses a few. Applies only when "
                         "the reset request carries an `active_targets` list. "
                         "⚠️ Anything but all changes observations a lot: within 30 m of an episode start "
                         "there are on average 1.1 (max 5) other targets, which vanish from view. "
                         "Use all for batches compared against earlier measurements")
    ap.add_argument("--mover-hz", type=float, default=0.0, metavar="HZ",
                    help="Transform update rate for moving targets. **Default 0 = every physics step "
                         "(250 Hz)**. Positive values cap the rate: "
                         "USD writes dirty PhysX, Fabric and render acceleration structures each time, "
                         "so this is costly. At 20, a 0.3 m/s target moves 1.5 cm per update, "
                         "0.3% of the 5 m arrival radius and under 1 pixel at 20 m. "
                         "⚠️ Observations are not exactly identical; use 0 for batches compared "
                         "against earlier measurements. Pedestrians are excluded (starving them kills A)")
    ap.add_argument("--target-at", default=None, metavar="X,Y[:DX,DY];…",
                    help="Place targets **at different positions each**: spawn-relative coordinates "
                         "joined by semicolons. Appending `:DX,DY` gives that target **its own "
                         "path** (it walks or rolls that way). "
                         "e.g. '10,0:30,0;6,12;-8,5:0,20'. "
                         "⚠️ Without this, all moving targets stack at the start of `--target-path` "
                         "and static ones spread along it, so (start, "
                         "target) pairs have **no positional diversity**")
    ap.add_argument("--target-front", default=None,
                    help="Place static targets at these distances straight ahead (+X) of the drone spawn (m, comma). "
                         "e.g. '6,12,20'. For deterministic smoke tests and occlusion checks; "
                         "does not use flat-run search")
    ap.add_argument("--target-path", default=None,
                    help="Give the moving-target path directly in **spawn-relative coordinates** "
                         "(m, 'x0,y0:x1,y1'). Auto search only finds x-axis runs, so use this "
                         "when a path crossing the view is needed")
    ap.add_argument("--target-area", type=float, default=60.0,
                    help="path search radius around the spawn point, m (default 60)")
    ap.add_argument("--mover-takeoff-gate-m", type=float, default=0.0, metavar="M",
                    help="**Hold** moving targets at their start until vehicle altitude exceeds "
                         "this (0 = off). Without it, pedestrians keep walking during settling, "
                         "EKF convergence and takeoff, and the start distance at step 1 no longer "
                         "matches the plan")
    ap.add_argument("--keep-alive", action="store_true",
                    help="Accepted for compatibility: the launcher always keeps the simulator "
                         "running after setup, so B can attach for several episodes. Stop it "
                         "before starting another A on the same slot")
    ap.add_argument("--render-aa", default="dlaa",
                    choices=("dlaa", "dlss", "fxaa", "taa", "off", "keep"),
                    help="Anti-aliasing/upscaling. Default dlaa = **native resolution**. "
                         "dlss renders at half resolution and upscales with AI, smearing ground and "
                         "texture (measured: 640x480 requested, actual render 320x240). "
                         "keep leaves the app default untouched")
    ap.add_argument("--ground-texture-scale", type=float, default=-1.0,
                    metavar="TILES_PER_M",
                    help="Texture tiling of ground materials. **-1 (default) = auto-correct**: "
                         "multiply the authored value by the inverse of the unit scale (0.01) we "
                         "put on `/World/layout`. 0 = leave alone. A positive value is "
                         "set as an absolute value (manual override). "
                         "Auto is the default because these materials use world-space "
                         "triplanar projection, so the parent scale stretches the texture: "
                         "the scene value 0.002 was one tile per 5 m in the original cm units, "
                         "and our unit fix stretched it to 500 m"
                         "")
    ap.add_argument("--paint-blend", type=float, default=0.0, metavar="0~1",
                    help="Blend target paint with the original texture. **Default 0 = off** "
                         "(solid colour). At 1, the original's "
                         "hue is removed and our colour applied, keeping seams, grime, normal maps and roughness. "
                         "0 = solid overpaint (texture lost). "
                         "⚠️ Enabling it now **shifts colours**: vMaterials-family MDLs do not "
                         "expose an albedo knob, so 6 of 7 rendered with the wrong colour name "
                         "(measured)."
                         "")
    ap.add_argument("--paint-brightness", type=float, default=1.6, metavar="X",
                    help="Blending **multiplies** colour onto a greyscale albedo, so it darkens. "
                         "This is the compensation (OmniPBR `albedo_brightness`)")
    ap.add_argument("--ground-texture-below", type=float, default=0.01,
                    metavar="MAX",
                    help="Only fix materials whose texture tiling is **smaller** than this. Opening it "
                         "to 1e9 overwrites healthy materials too: measured, it smeared "
                         "tree bark (1.0, 0.7) and a pond (0.05, 0.1) to 0.002. "
                         "The scene's broken values are road 0.005 and sidewalk 0.002, so 0.01 suffices")
    ap.add_argument("--ground-desaturate", type=float, default=0.0,
                    metavar="0~1",
                    help="Ground material `albedo_desaturation`. The scene default 1.0 "
                         "removes texture colour entirely, making asphalt and gravel flat "
                         "grey. 0 = keep texture colour. "
                         "Negative = leave alone")
    ap.add_argument("--no-material-fix", action="store_true",
                    help="skip restoring `/Looks` materials that fell outside the reference scope (debug)")
    ap.add_argument("--no-spawn-check", action="store_true",
                    help="skip re-measuring spawn ground after play")
    ap.add_argument("--gt", action="store_true",
                    help="Enable the GT sidecar: writes rgb/depth/instance/semantic + target GT to "
                         "`gt_stream.jsonl`. Joined afterwards with the runner's steps.jsonl via "
                         "`verifier/isaacsim/join_gt.py`")
    ap.add_argument("--gt-dir", default=None,
                    help="GT output directory (default: ./datagen_runs/<timestamp>)")
    ap.add_argument("--gt-tick", choices=("frame", "hz"), default="frame",
                    help="When GT ticks. frame (default)=on the render where A swaps the camera frame: "
                         "1:1 with what B sees, so the join is exact (measured 5/5 vs "
                         "3/5). hz=wall-clock rate (`--gt-hz`)")
    ap.add_argument("--gt-hz", type=float, default=5.0,
                    help="tick cap for `--gt-tick hz` (default 5, 0=every render)")
    ap.add_argument("--gt-mode", choices=("analysis", "measure"), default="analysis",
                    help="**analysis (default)**: annotators always attached, everything recorded "
                         "every tick; for dataset generation, oracle rollouts and review. "
                         "**measure**: detached normally, attached only within `--gt-near-m` of the target "
                         "and in the takeoff warm-up window; **VLM benchmark measurement only**. "
                         "Judging is `ok = (dist ≤ δ) AND visible` and annotators are always "
                         "on within δ, so **SR on the VLM path is preserved**. "
                         "⚠️ Not for oracle rollouts that use `mask_px`/`point` as navigation "
                         "input: away from the target those are empty. "
                         "⚠️ In measure outputs `mask_px` is mostly empty; "
                         "do not use them for analysis/review (each row records `gt_mode`, `ann`, "
                         "`ann_warm`)")
    ap.add_argument("--gt-near-m", type=float, default=10.0, metavar="M",
                    help="Distance at which measure mode attaches annotators (default 10 m). "
                         "Margin over δ 5 m: data fills from the render **after** attaching, "
                         "so one tick of latency must be absorbed. Detach distance is "
                         "1.5x that (hysteresis)")
    ap.add_argument("--gt-warm", choices=("signal", "altitude", "off"),
                    default="altitude",
                    help="In measure mode, **how GT at episode step 1 is guaranteed**. "
                         "Episodes start beyond 15 m from the target, so the distance gate alone "
                         "gives no pixels at step 1 and `seen_at_start` silently becomes all "
                         "False. "
                         "**signal**: B notifies via `gt_hold.request`; "
                         "observable and **cheap** (attaches only for the few ticks needed). "
                         "Requires a B version that sends the signal. "
                         "**altitude (default)**: A infers episode start from altitude; "
                         "works with old B but covers the whole takeoff (measured 129 sim s), "
                         "so it is costly. **off**: distance gate only (not recommended)")
    ap.add_argument("--gt-hold-s", type=float, default=15.0, metavar="SEC",
                    help="With `--gt-warm signal`, wall-clock seconds a B request stays valid "
                         "(default 15). Slack between B writing the request and the first capture")
    ap.add_argument("--gt-warm-max-s", type=float, default=240.0, metavar="SEC",
                    help="**Safety cap on the takeoff warm-up** in measure mode (sim s, default 240). "
                         "Warm-up engages while the vehicle is near the ground and releases once stable at "
                         "flight altitude: released by **altitude**, not time. Measured: takeoff "
                         "took 92 sim s (PX4 lingered 88 s at 1.8 m), and a fixed 20 s "
                         "window released before the vehicle reached altitude, silently breaking "
                         "`seen_at_start` (all False). This cap is only hit when altitude is "
                         "never reached (e.g. low-altitude episodes), and then it warns")
    ap.add_argument("--gt-res", default="640x480", help="GT render resolution (default 640x480)")
    ap.add_argument("--gt-targets", default=None,
                    help="name regex selecting target prims (e.g. 'DG_target')")
    ap.add_argument("--gt-occlusion", action="store_true",
                    help="⚠️ Render-based occlusion ratio (c). **Unusable at Brownstone scale**: "
                         "frames with the environment hidden come out as sky with the whole scene missing, "
                         "and instance idToLabels arrive INVALID. Occlusion is "
                         "measured with raycasts without this option")
    ap.add_argument("--gt-live", default="/tmp/vlmpilot_pegasus/latest_gt.json",
                    help="Path of the \"current frame GT\" sidecar read by B (runner). "
                         "Empty string disables it. The oracle backend reads this")
    ap.add_argument("--gt-ray-skip", type=float, default=0.6,
                    help="Offset of the occlusion ray start from the camera (m, default 0.6). "
                         "The camera is inside the vehicle collider, so 0 hits its own body")
    ap.add_argument("--gt-occl-target", default=None,
                    help="target prim for occlusion (default: first target outside the environment)")
    ap.add_argument("--gt-settle", type=int, default=2,
                    help="renders to flush after a visibility change (default 2). Needed because "
                         "annotators lag the render by one frame")
    ap.add_argument("--gt-debug-luma", action="store_true",
                    help="print mean luminance of both occlusion passes (contamination diagnosis)")
    ap.add_argument("--gt-frames", default="rgb,depth,instance,semantic",
                    help="frame kinds to save (comma; empty string = metadata only)")
    ap.add_argument("--gt-orchestrator-step", action="store_true",
                    help="Fallback when annotators are not filled by live rendering. "
                         "Calls `rep.orchestrator.step()`, so render timing jitters")
    ap.add_argument("--slot", type=int, default=None, metavar="N",
                    help="Slot number. Passed through to `pegasus_sim.py --slot`, and "
                         "GT output (`--gt-live`) is switched to that slot's path. With several "
                         "simulators on one pod, give each its own slot, otherwise they "
                         "**silently overwrite each other's frames**. Omitted = the default paths")
    ap.add_argument("-h", "--help", action="store_true")
    mine, passthru = ap.parse_known_args()
    if mine.help:
        print(__doc__, flush=True)
        ap.print_help()
        print("\nArguments not listed above are passed through to pegasus_sim.py.", flush=True)
        return 0

    sys.path.insert(0, os.path.join(REPO, "scripts", "isaacsim"))
    sys.path.insert(0, os.path.join(REPO, "vlmpilot", "src"))

    from sim import brownstone_env as BE

    pack_env = mine.env is None       # Brownstone pack? (subject to unit/lighting/material restore)
    if pack_env:
        env_name = f"Brownstone {mine.variant}"
        if env_name not in BE.VARIANTS:
            print(f"[DG] unknown variant '{mine.variant}'. Valid: "
                  f"{[k.split(' ',1)[1] for k in BE.VARIANTS]}", flush=True)
            return 2
        usd = BE.usd_path(env_name)
        if not os.path.exists(usd):
            print(f"[DG] USD missing: {usd}\n     check that the pack was downloaded "
                  f"(scripts/runpod/isaacsim/README.md)", flush=True)
            return 2
    else:
        # A scene from the upstream registry: pegasus_sim knows the USD path, and its own gate
        # checks existence (sys.exit(2) if `--env` is not a key).
        env_name, usd = mine.env, None

    # pegasus_sim parses argv at module level, so set it before import
    # -- slot -------------------------------------------------------------------------
    # `pegasus_sim.py` derives its four paths and vehicle_id itself, so passing `--slot`
    # through is enough. What we handle here are the **GT paths**: they belong to this
    # launcher and pegasus_sim does not know them. Without splitting GT, two slots share one
    # `latest_gt.json` and the oracle acts on another run's GT (plausible data with no error).
    if mine.slot is not None:
        sys.path.insert(0, os.path.join(REPO, "vlmpilot", "src"))
        from vlmpilot.controller.isaacsim import pegasus_slots as PS   # noqa: N814

        res = PS.resolve(mine.slot)
        if mine.gt_live == ap.get_default("gt_live"):
            mine.gt_live = res["paths"]["gt_live"]
        if mine.gt_dir is None:
            mine.gt_dir = os.path.join(res["paths"]["root"], "gt")
        if not any(a == "--slot" or a.startswith("--slot=") for a in passthru):
            passthru = ["--slot", str(int(mine.slot))] + passthru
        print(f"[DG] slot {mine.slot} · GT live {mine.gt_live} · GT output {mine.gt_dir}"
              + (f" · owner {res['owner']!r}" if res.get("owner") else ""), flush=True)
        _guard_slot(mine)

    sys.argv = ["pegasus_sim.py", "--env", env_name] + passthru + _KIT_ENABLE
    print(f"[DG] scene {env_name}"
          + (f"\n[DG] USD  {usd}" if usd else "")
          + f"\n[DG] pegasus_sim args {passthru}", flush=True)

    import pegasus_sim  # noqa: E402  (boots SimulationApp)

    _set_render_aa(mine.render_aa)
    _check_anim_graph()

    if pack_env:
        added = BE.register(pegasus_sim.SIMULATION_ENVIRONMENTS)
        print(f"[DG] registered {len(added)} environments: {added}", flush=True)
    else:
        keys = list(pegasus_sim.SIMULATION_ENVIRONMENTS)
        if env_name not in keys:
            print(f"[DG] '{env_name}' is not in the Pegasus registry. Valid: {keys}",
                  flush=True)
            return 2
        print(f"[DG] upstream registry scene {env_name!r}: skipping Brownstone-specific prep",
              flush=True)

    # State shared by prep (before play), the spawn check (after play) and the GT stream.
    # Prep runs inside `main()` and the GT stream is created before it, so this holder links them.
    shared = {"stream": None, "specs": [], "targets": [], "movers": []}
    if mine.targets:
        try:
            shared["specs"] = parse_targets(mine.targets)
        except ValueError as exc:
            print(f"[DG] {exc}", flush=True)
            return 2

    if not mine.no_prep:
        from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface
        from sim import scene_prep as SP

        _orig_load = PegasusInterface.load_environment
        # Ground correction is a measured pack value. Registry scenes are authored in metres with ground at 0.
        street_z = BE.VARIANTS[env_name]["spawn_ground_m"] if pack_env else 0.0

        def _load_and_prep(self, usd_path, force_clear=False):
            out = _orig_load(self, usd_path, force_clear)
            # `load_environment` **only schedules** via asyncio.ensure_future and returns immediately.
            # So the stage is empty right after it returns (measured: 0 labels, 0 collisions).
            # Tick the app until the stage is populated, then run prep.
            try:
                import omni.usd
                from pxr import Usd, UsdGeom
                app = pegasus_sim.simulation_app
                ctxq = omni.usd.get_context()

                def mesh_count():
                    st = ctxq.get_stage()
                    if st is None:
                        return 0
                    return sum(
                        1 for pr in Usd.PrimRange.Stage(
                            st, Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate))
                        if pr.IsA(UsdGeom.Mesh))

                prev, stable, waited = -1, 0, 0
                for waited in range(1, 1201):          # at most 1200 updates
                    app.update()
                    if waited % 20:
                        continue
                    n = mesh_count()
                    if n > 100 and n == prev:
                        stable += 1
                        if stable >= 2:                # two equal counts in a row = loaded
                            break
                    else:
                        stable = 0
                    prev = n
                print(f"[DG] waited {waited} updates for stage load · mesh {mesh_count():,}", flush=True)

                stage = ctxq.get_stage()
                if pack_env:
                    # Fix units **before** labels, collision and spawn: a 100x error also
                    # skews ground height 100x and renders come out black.
                    # **Remember** this scale: restoring the texture tiling of world-projected
                    # materials uses exactly its inverse (ground fix below).
                    shared["layout_scale"] = SP.fix_env_units(
                        stage, "/World/layout", usd)
                    # The reference only brings in the default prim (`/World`), so the sun and sky
                    # in sibling `/Environment` are missing; import them separately
                    SP.import_env_lights(stage, usd)
                    # Materials fall outside the scope for the same reason; import and rebind them
                    if not mine.no_material_fix:
                        SP.import_env_looks(stage, usd)
                        SP.rebind_dropped_materials(stage)
                if not mine.keep_env_animation:
                    # **Must be removed before play.** Time-sample re-evaluation tied to the
                    # physics step was the main cause of RTF 0.2 (104x).
                    SP.freeze_env_animation(stage)
                labels = SP.label_groups(stage)          # must happen before play
                ncol = SP.apply_collision(stage)
                prep_note(shared, "labels", int(sum(labels.values())))
                prep_note(shared, "collisions", int(ncol))
                print(f"[DG] scene prep: {sum(labels.values())} group labels {labels} · "
                      f"{ncol:,} collisions", flush=True)

                if shared["specs"]:
                    # Targets live **outside** `/World/layout`: the occlusion silhouette pass
                    # hides the environment root, which would hide targets under it too.
                    from sim import targets as TG
                    TG.set_paint_blend(mine.paint_blend, mine.paint_brightness)
                    bad = [sp for sp in shared["specs"]
                           if sp["move"] and sp["kind"] not in TG.MOVABLE]
                    if bad:
                        # A fire hydrant sliding along the street works neither as data nor as
                        # a demo. Rather than passing silently, revert to static.
                        for sp in bad:
                            sp["move"] = False
                        print(f"[DG] ⚠️ {[sp['kind'] for sp in bad]} are fixed street "
                              f"furniture and cannot be moving targets; reverting to static. "
                              f"Pick moving targets from {sorted(TG.MOVABLE)}", flush=True)
                    for k, spec in enumerate(shared["specs"]):
                        # Prim names follow **spec order**: the plan's `target_prim` is
                        # computed in this order, so it cannot change.
                        path = f"/World/DG_targets/t{k}"
                        try:
                            # ⚠️ The temporary position must be **outside the site**. At the origin
                            # the target's collision is caught by the ground probe at the drone spawn,
                            # reading ground as +1.42 m (= top of the bin); the spawn check then
                            # misjudges "underground" and flat-run search uses a wrong reference (measured).
                            if spec["kind"] in TG.PEOPLE:
                                # Pedestrians are not painted: clothing already conveys the role.
                                # The graph is imported once and shared by all.
                                info = TG.spawn_person(stage, path, spec["kind"],
                                                       TG.PARKING)
                            else:
                                info = TG.spawn_prop(stage, path, spec["kind"],
                                                     spec["color"], TG.PARKING)
                            if info is None:
                                print(f"[DG] target spawn skipped {spec}", flush=True)
                                continue
                        except Exception as exc:  # noqa: BLE001
                            # The plan's `target_prim` for this spec now points to a missing
                            # prim, so its episodes never see the target. Check this line.
                            print(f"[DG] target spawn failed {spec}: {exc!r}", flush=True)
                            continue
                        info["move"] = spec["move"]
                        # ⚠️ **Stamp the spec index on the target.** `_place_targets` uses it
                        # to pick `--target-at`. Using the order of the surviving list means one
                        # failed spawn **shifts every coordinate by one**; this happened (bench
                        # failed, the hydrant stood in the bench's spot, and the plan's t0
                        # did not exist at all).
                        info["spec_index"] = k
                        shared["targets"].append(info)
                    paths = [t["prim"] for t in shared["targets"]]
                    if shared["stream"] is not None and paths:
                        # Give the sidecar the paths directly instead of letting it match a pattern
                        shared["stream"].targets.extend(paths)
                        shared["stream"].target_pattern = None
                        # For the oracle to judge "is the decoy the same kind and colour",
                        # colour and kind must sit next to GT
                        for t in shared["targets"]:
                            shared["stream"].target_meta[t["prim"]] = {
                                "color": t["color"], "kind": t["kind"],
                                "noun": t["noun"], "class": t["class"],
                                "phrase": t["phrase"], "move": t["move"],
                                # Effective speed is filled in at placement (`_place_targets`),
                                # since it depends on whether it is a pedestrian
                                "speed_mps": None, "mover": None,
                            }
                    print(f"[DG] spawned {len(paths)} targets (positions fixed after play): "
                          + ", ".join(t["phrase"] for t in shared["targets"]), flush=True)
            except Exception as exc:  # noqa: BLE001 — prep failure must not block flight
                print(f"[DG] scene prep failed, continuing: {exc!r}", flush=True)
            return out

        PegasusInterface.load_environment = _load_and_prep

        if mine.spawn_yaw is not None:
            # `PegasusSimApp.__init__` fixes vehicle attitude to `[1,0,0,0]` (heading 0).
            # Overwrite right after construction so the rigs copy it.
            from scipy.spatial.transform import Rotation as _Rot
            _orig_app_init = pegasus_sim.PegasusSimApp.__init__

            def _init_with_yaw(self, *a, **k):
                _orig_app_init(self, *a, **k)
                q = _Rot.from_euler("XYZ", [0.0, 0.0, float(mine.spawn_yaw)],
                                    degrees=True).as_quat()      # x,y,z,w
                self.spawn_quat_wxyz = [q[3], q[0], q[1], q[2]]  # Isaac wants w first
                for rig in getattr(self, "rigs", []):
                    rig.spawn_quat_wxyz = list(self.spawn_quat_wxyz)
                print(f"[DG] launch heading {mine.spawn_yaw:.1f}°: "
                      f"quat {['%.4f' % v for v in self.spawn_quat_wxyz]}", flush=True)

            pegasus_sim.PegasusSimApp.__init__ = _init_with_yaw

        # -- toggle motion on reset: one A produces all 4 types ------------------
        # The 4 DroneCATS types are (static, moving) x (in view, out of view). View is decided
        # by B's yaw, but **static/moving is a scene-side decision**. Relaunching A per scene
        # makes loading (20 GB) the throughput, so `target_move` rides on the upstream reset request and is switched here.
        #
        # The original handler reads and **deletes** the request file, so we read it first.
        # Same thread, so no race; the upstream handler simply ignores unknown keys.
        #
        # **Always rewind**, even without `target_move`. Episode independence depends on it:
        # otherwise episode 2 starts where episode 1 ended and is no longer the same item.
        _orig_reset_check = pegasus_sim.PegasusSimApp._check_reset_request

        def _reset_with_target_move(self):
            req = None
            try:
                if self.reset_file and os.path.exists(self.reset_file):
                    with open(self.reset_file, encoding="utf-8") as fh:
                        req = json.load(fh)
            except (OSError, ValueError):
                req = None
            out = _orig_reset_check(self)
            if req is None:
                return out
            from sim import targets as TG
            # **Apply the active set first.** For the rewind to "hold inactive targets",
            # the set must already be in place at that moment.
            if "active_targets" in req:
                act = req.get("active_targets")
                TG.set_active_movers(None if act in (None, "") else
                                     [t for t in str(act).split(",") if t])
                if mine.target_scope != "all":
                    # Keep only the 2 of the scene's 61 targets this episode uses; the point is
                    # removing them from render acceleration structures (and PhysX with deactivate).
                    import omni.usd
                    TG.set_target_scope(
                        omni.usd.get_context().get_stage(),
                        [t.get("prim") for t in shared["targets"] if t.get("prim")],
                        True, mode=mine.target_scope)
            mv = req.get("target_move")
            if mv is None:
                TG.rewind_movers()
            else:
                TG.set_moving(mv)
            # GT meta must say **whether it actually moved in this episode**. That varies per
            # episode, so leaving the spawn-time value (`move`) would be false.
            stream = shared.get("stream")
            if stream is not None and mv is not None:
                for t in shared["targets"]:
                    meta = getattr(stream, "target_meta", {}).get(t.get("prim"))
                    if meta is None or not t.get("move"):
                        continue
                    meta["move"] = bool(mv)
                    meta["speed_mps"] = t.get("speed_mps") if mv else 0.0
            return out

        pegasus_sim.PegasusSimApp._check_reset_request = _reset_with_target_move

        spawn_xy = None
        if mine.spawn_xy:
            try:
                spawn_xy = tuple(float(v) for v in mine.spawn_xy.split(","))
                if len(spawn_xy) != 2:
                    raise ValueError
            except ValueError:
                print(f"[DG] --spawn-xy must be 'x,y': {mine.spawn_xy!r}", flush=True)
                return 2

        if street_z or spawn_xy:
            # pegasus_sim uses `from vlmpilot.controller.isaacsim import pegasus_layout` (not a top-level module).
            # Patching the module object it references avoids guessing the import path.
            pegasus_layout = pegasus_sim.pegasus_layout
            _orig_spawn = pegasus_layout.spawn_for
            # **Move only the launch reference point.** `--spawn-xy` means "default spawn goes here",
            # but reset (`--pos`) goes through the same function, so moving unconditionally gives
            # `pos_final = base + (spawn_xy - base) = spawn_xy` and **every per-episode spawn
            # collapses to the launch spawn** (measured: 8.22 m off the planned spawn, bad_start).
            # So remember the first call's base and move only when equal to it; a new base
            # from reset is authoritative. The z (ground) correction is always applied.
            _launch = {"base": None}

            def _spawn_move(base, i):
                pos = list(_orig_spawn(base, i))
                b = [float(v) for v in base]
                if _launch["base"] is None:
                    _launch["base"] = b
                same = all(abs(x - y) < 1e-6 for x, y in zip(b, _launch["base"]))
                if spawn_xy is not None and same:
                    # keep the multi-vehicle spread (layout), move only the reference point
                    pos[0] += float(spawn_xy[0]) - b[0]
                    pos[1] += float(spawn_xy[1]) - b[1]
                if street_z:
                    pos[2] = float(pos[2]) + float(street_z)
                if i == 0:
                    bits = []
                    if spawn_xy is not None and same:
                        bits.append(f"xy → ({pos[0]:+.1f}, {pos[1]:+.1f})")
                    elif spawn_xy is not None:
                        bits.append(f"keeping given base point ({pos[0]:+.1f}, {pos[1]:+.1f}) "
                                    f"(not overwritten by --spawn-xy)")
                    if street_z:
                        bits.append(f"z {street_z:+.2f} m → {pos[2]:.3f}")
                    print("[DG] spawn: " + " · ".join(bits), flush=True)
                return pos

            pegasus_layout.spawn_for = _spawn_move
        elif street_z == 0.0:
            print(f"[DG] {env_name} spawn ground 0.000 m; no correction", flush=True)
        else:
            print(f"[DG] {env_name} has no measured spawn_ground_m; skipping spawn correction",
                  flush=True)

        if not mine.no_spawn_check:
            # After play, measure the real ground to see whether the spawn went underground. Before
            # this check, a drone 0.59 m underground showed only as "the camera is black".
            from isaacsim.core.api.world import World
            _orig_step = World.step
            # PhysX cooks ~8k triangle meshes right after play; measuring on the first render
            # gives "no ground below" (measured). Retry until the ground is found.
            _chk = {"renders": 0, "tries": 0, "done": False}

            def _step_and_check(self, render=True, *a, **k):
                out = _orig_step(self, render, *a, **k)
                if _chk["done"] or not render:
                    return out
                _chk["renders"] += 1
                if _chk["renders"] % 60:
                    return out
                try:
                    import omni.timeline
                    if not omni.timeline.get_timeline_interface().is_playing():
                        return out
                    import omni.usd
                    _chk["tries"] += 1
                    rows = SP.spawn_report(omni.usd.get_context().get_stage(),
                                           verbose=False)
                    got = [r for r in rows if r["ground_m"] is not None]
                    if not (got or _chk["tries"] >= 8):
                        return out
                    # The ground fix happens **here**: rays are physics queries and all
                    # miss before play. Right after play is still early: one spawn spot may
                    # be ready while the site is still cooking (once only 1 of 196 rays hit,
                    # measured). **Wait here** until it is cooked enough;
                    # fixing after episodes start would leave early frames with the old ground.
                    if not _chk.get("gfixed"):
                        # Counted **separately** from spawn retries (8). Cooking all 8,613
                        # colliders takes longer; moving on after checking only the spawn spot
                        # gave 1 hit of 196 (measured).
                        # ⚠️ **Keep the cap short so cosmetics never block essentials.** It was
                        # 25 at first; with no region the hits stayed at 1, so it early-returned
                        # all 25 times and **blocked target placement**: 61 targets stayed
                        # in the parking lot (6,948 m) while B flew, and the batch was meaningless.
                        # The ground fix is nice to have, not a requirement.
                        _chk["gtries"] = _chk.get("gtries", 0) + 1
                        _n_fix, _n_hit = _fix_ground(
                            omni.usd.get_context().get_stage(), SP, shared, mine)
                        if _n_hit >= 50 or _chk["gtries"] >= 5:
                            _chk["gfixed"] = True
                            if _n_hit < 50:
                                print(f"[PREP] ⚠️ ground hits stalled at {_n_hit}; "
                                      f"continuing without ground fix", flush=True)
                        else:
                            print(f"[PREP] ground not cooked yet "
                                  f"(hits {_n_hit}); measuring again "
                                  f"[{_chk['gtries']}/5]", flush=True)
                            return out
                    SP.spawn_report_print(rows)
                    _chk["done"] = True
                    prep_note(shared, "targets_spawned",
                              len(shared.get("targets") or []))
                    if shared["targets"]:
                        _place_targets(omni.usd.get_context().get_stage(),
                                       SP, rows, mine, shared)
                    # **Spawned != placed.** A target that never left the parking lot still
                    # counts as spawned, so count placement separately.
                    prep_note(shared, "targets_placed",
                              _count_placed(
                                  omni.usd.get_context().get_stage(),
                                  shared))
                except Exception as exc:  # noqa: BLE001
                    _chk["done"] = True
                    print(f"[DG] spawn check failed: {exc!r}", flush=True)
                return out

            World.step = _step_and_check

    if mine.gt:
        import datetime
        from sim import gt_sidecar as GT

        gt_dir = mine.gt_dir or os.path.join(
            os.getcwd(), "datagen_runs",
            datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
        os.makedirs(gt_dir, exist_ok=True)
        try:
            w, h = (int(v) for v in mine.gt_res.lower().split("x"))
        except ValueError:
            print(f"[DG] --gt-res must be WxH: {mine.gt_res!r}", flush=True)
            return 2
        stream = GT.GTStream(
            gt_dir, width=w, height=h,
            target_pattern=mine.gt_targets, occl_target=mine.gt_occl_target,
            occlusion=mine.gt_occlusion, hz=mine.gt_hz,
            save_frames=[s_ for s_ in mine.gt_frames.split(",") if s_],
            orchestrator_step=mine.gt_orchestrator_step,
            settle_renders=mine.gt_settle, debug_luma=mine.gt_debug_luma,
            ray_skip_m=mine.gt_ray_skip, live_path=(mine.gt_live or None),
        )
        # Measure mode: annotators detached normally, attached only near the target
        stream.mode = mine.gt_mode
        stream.near_on_m = float(mine.gt_near_m)
        stream.near_off_m = float(mine.gt_near_m) * 1.5
        stream.warm_max_s = float(mine.gt_warm_max_s)
        stream.warm_mode = mine.gt_warm
        stream.hold_s = float(mine.gt_hold_s)
        try:
            from vlmpilot.controller.isaacsim.pegasus_layout import slot_paths
            stream.hold_path = slot_paths(int(mine.slot))["gt_hold"]
        except Exception as exc:  # noqa: BLE001
            stream.hold_path = None
            if mine.gt_warm == "signal":
                print(f"[DG] ⚠️ could not resolve slot paths, so `--gt-warm signal` is "
                      f"unavailable ({exc!r}); falling back to the altitude latch", flush=True)
                stream.warm_mode = "altitude"
        if mine.gt_mode == "measure":
            print(f"[DG] GT **measure mode**: annotators attached only within {stream.near_on_m:g} m "
                  f"of a target (detach {stream.near_off_m:g} m · "
                  f"warm-up {stream.warm_mode}"
                  + ({"signal": f" ← {stream.hold_path}",
                      "altitude": (f" (until stable at {stream.warm_clear_z_m:g} m · "
                                   f"cap {stream.warm_max_s:g} sim s)"),
                      "off": " (**step-1 GT not guaranteed**)"}
                     .get(stream.warm_mode, "")) + "). "
                  "⚠️ In these outputs `mask_px` is mostly empty; not for analysis/review",
                  flush=True)
        # The upstream loop calls `world.step(render=...)`; wrapping `simulation_app.update`
        # ran GT for zero ticks (measured). Camera and targets are resolved late, after play.
        if mine.gt_occlusion:
            print("[DG] ⚠️ --gt-occlusion (render-based) does not work at Brownstone scale: "
                  "frames with the environment hidden come out as sky only. Occlusion is "
                  "measured with raycasts without this option", flush=True)
        shared["stream"] = stream       # prep puts target paths here
        if mine.gt_tick == "frame":
            GT.install_on_frame(stream)
        else:
            GT.install(stream)
        print(f"[DG] GT sidecar {gt_dir} · {w}x{h} · "
              + ("frame-aligned" if mine.gt_tick == "frame" else f"{mine.gt_hz} Hz")
              + (" · occlusion on" if mine.gt_occlusion else ""), flush=True)
        try:
            return pegasus_sim.main() or 0
        finally:
            stream.close()

    return pegasus_sim.main() or 0


if __name__ == "__main__":
    sys.exit(main())
