# DroneCATS
# Copyright (c) 2026-present NAVER Cloud Corp.
# Apache-2.0

"""**Scene setup** for bench episodes (editor-v2) — spawning targets, objects, and walls; painting; plate/sign textures; mission sentence.

Rewriting the spawn, paint, plate, or start-pose rules would produce **different episodes**, so these rules are fixed.
Episodes give `z_ned` directly (or flat ground is assumed), and executor invocation is done by `run_bench_tasks.py`.

⚠ Sim caveat: respawning right after destroy kills UE (`WorldSimApi::spawnObject` SIGABRT).
   Building for the first time in an empty sim is safe — when repeating the same episode, **reuse** it (only `FORCE_SETUP=1` rebuilds).
"""
import math
import os
import time

# Paint palette (BGR). simSetObjectMaterialFromTexture returns True for any asset, so do not trust its return value.
PAINT_BGR = {
    "magenta": (180, 20, 200), "yellow": (20, 220, 240), "lime": (40, 220, 90),
    "blue": (200, 60, 20), "red": (40, 40, 220), "purple": (160, 30, 110),
    "orange": (30, 140, 240), "green": (60, 160, 60), "white": (240, 240, 240),
    "grey": (128, 128, 128),
}


def target_name_for(task, port):
    """Separate names per port — if parallel runners use the same `BM_*` names, they delete each other's targets."""
    return f'{task["target"]["spawn"]}_p{port}'


def prop_name_for(task, prop_name, port):
    """Object name in the episode → actor name in the scene. Uses the one rule that `spawn_extras` builds with."""
    return f'BM_{task.get("id", "T")}_{prop_name}_p{port}'


LAST_GROUND_Z = 0.0     # ground z (NED) measured by the last setup_target — for the runner's start z computation


def mission_text(task):
    """Mission sentence sent to the model (one stdin line). If there is a hint, it is appended."""
    m = (task.get("prompt") or "").strip()
    h = (task.get("hint") or "").strip()
    return f"{m} {h}".strip() if h else m


def spawn_z(gz, size_z, pivot):
    """z (NED) when standing on the ground. Center pivots (Blocks primitives) are lifted by half, base pivots by 0.3 m."""
    return gz - (float(size_z) / 2.0 if pivot == "ctr" else 0.3)


def entry_z(entry, size_z, pivot, gz=0.0):
    """Decide spawn z (NED): z_ned (absolute) > ground (gz) − z_off_m > ground default.

    Editor v2 episodes write height as `z_off_m` (above ground, +) — if this is not read, signs
    lie flat on the ground and the post pierces the letters (measured on blocks).
    gz: measured ground z (NED). The blocks map plaza's ground is not 0 but **around +1.0**
    (drop-probe measurements +1.01/+1.02), so building relative to 0 makes everything look 1 m afloat.
    """
    if "z_ned" in entry:
        return float(entry["z_ned"])
    if "z_off_m" in entry:
        return gz - float(entry["z_off_m"])
    return spawn_z(gz, size_z, pivot)


def ground_z_of(task, defaults=None):
    """Ground z (NED) written in the episode. task.ground_z_ned > defaults.ground_z_ned > None (measurement needed).

    Measurement (drop probe) jumped from episode to episode (X10 870 episodes: Blocks +1.02 is canonical, but 1~3 episodes per question gave
    −0.93~+1.16). If the episode records the ground, it is not re-measured — same convention as single-drone episodes."""
    for src in (task or {}, defaults or {}):
        v = src.get("ground_z_ned")
        if v is not None:
            return float(v)
    return None


def measure_ground_z(c, x, y, port=None, timeout_s=3.0):
    """Ground z (NED) at (x,y). Uses the editor Prober (validation residual 0.35 m) first, and falls back to our own drop if it fails.

    Why our own drop alone is not enough (all measured): UE level streaming — tiles far from the observer are
    unloaded and **have no collision**. In far episodes such as E-19 (y=102) and E-20 (x=128) the probe
    fell through and got the ground wrong as 0.0/+0.78, leaving the car 1.7 m afloat or 2.3 m buried.
    The Prober first moves the vehicle above that spot to load the tiles, and also does stop detection (4 in a row) and
    neighbor-median validation (rejecting landings on roofs or trees) — reused as is.
    """
    if port:
        try:
            import sys as _sys
            tools = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                 "demos", "benchmark-editor", "tools")
            if tools not in _sys.path:
                _sys.path.insert(0, tools)
            from probe import Prober
            pr = Prober(int(port))
            try:
                z, warn = pr.surf_checked(float(x), float(y))
            finally:
                pr.close()
            if warn:
                print(f"  · ground probe: {warn}", flush=True)
            if z == z and abs(z) <= 10.0:
                return round(float(z), 3)
        except Exception as e:
            print(f"  ! editor Prober failed ({type(e).__name__}: {str(e)[:60]}) — falling back to our own drop probe", flush=True)
    return _drop_ground_z(c, x, y, timeout_s)


def _drop_ground_z(c, x, y, timeout_s=3.0):
    """Our own drop probe (fallback). 0.0 on failure.

    Assuming z=0 was 1 m off on the blocks plaza (measured +1.01). Two pitfalls (measured):
    ① on nh the probe settles on tree canopy/power lines → the whole scene floats 6.5 m in the air.
    ② simDestroyObject is asynchronous, so it may land on the previous scene's posts.
    → drop at 5 points (center + ±2 m ring), take **the lowest landing point (max z)** as the ground, and
      if it is outside ±2.5 m, treat it as unmeasured and use 0.0 (better than breaking the meaning of z_off)."""
    import airsim
    half = 0.25
    best = None
    for i, (dx, dy) in enumerate([(0, 0), (2, 0), (-2, 0), (0, 2), (0, -2)]):
        nm = f"BM_GPROBE_TMP{i}"
        try:
            try:
                if c.simListSceneObjects(nm):
                    c.simDestroyObject(nm)
            except Exception:
                pass
            c.simSpawnObject(nm, "Cube",
                             airsim.Pose(airsim.Vector3r(float(x) + dx, float(y) + dy, -18.0),
                                         airsim.to_quaternion(0, 0, 0)),
                             airsim.Vector3r(half * 2, half * 2, half * 2), True)
            t0 = time.time()
            z_prev, z, stable = None, None, False
            while time.time() - t0 < max(timeout_s, 6.0):
                time.sleep(0.4)
                z = c.simGetObjectPose(nm).position.z_val
                if z_prev is not None and abs(z - z_prev) < 0.01:
                    stable = True
                    break
                z_prev = z
            c.simDestroyObject(nm)
            # discard if it fails to stabilize — using mid-fall z as ground drops to 0.0 like E-20
            # (clamp) or puts in a bogus value (measured).
            if not stable or z is None or z != z or abs(z) > 25:
                continue
            g = z + half
            if best is None or g > best:                # NED: down is + — the lowest surface
                best = g
        except Exception:
            continue
    # nh terrain undulates by more than ±2 m by region (measured E-19 −1.74 · E-20 −1.51 · plaza +1.5)
    if best is None or abs(best) > 6.0:
        return 0.0
    return round(best, 3)


# Background colors in BGR — uses the same palette as paint_object's color names (for sign background color)
_TXT_BGR = {"white": (250, 250, 250), "grey": (128, 128, 128), "black": (20, 20, 20),
            "red": (40, 40, 230), "orange": (30, 140, 255), "yellow": (40, 220, 240),
            "lime": (60, 230, 60), "green": (60, 160, 40), "blue": (230, 90, 30),
            "magenta": (200, 40, 220), "purple": (200, 60, 140), "cyan": (220, 210, 40)}


def text_texture(text, aspect=2.0, ratio=0.55, bg=None):
    """Path of a texture PNG with drawn text (plates/signs). aspect = width/height of the visible face, ratio = text height/texture height.

    bg: background color name (paint palette). White background if not given. Text color auto-contrasts with background brightness
    (light background → black, dark background → white). Previously, when text was present, paint was ignored
    entirely and every sign was white — color-decoy episodes did not work.
    """
    import re as _re
    import cv2 as _cv
    import numpy as _np
    bgr = _TXT_BGR.get(str(bg or "white").lower(), (250, 250, 250))
    key = _re.sub(r"\W+", "_", str(text))[:24]
    path = f"/tmp/txt_{key}_{aspect:.2f}_{ratio:.2f}_{bg or 'white'}.png"
    if os.path.exists(path):
        return path
    H = 256
    W = max(64, int(H * aspect))
    img = _np.full((H, W, 3), bgr, _np.uint8)
    # contrasting text color: perceived luminance (BGR)
    lum = 0.114 * bgr[0] + 0.587 * bgr[1] + 0.299 * bgr[2]
    fg = (10, 10, 10) if lum > 140 else (245, 245, 245)
    sc = H * ratio / 22.0
    th = max(2, int(sc * 3.0))
    (tw, tht), _ = _cv.getTextSize(str(text), _cv.FONT_HERSHEY_SIMPLEX, sc, th)
    fit = (W * 0.82) / max(tw, 1)          # prevents horizontal overflow (what used to clip "1024" → "024")
    if fit < 1.0:
        sc *= fit
        th = max(2, int(sc * 3.0))
        (tw, tht), _ = _cv.getTextSize(str(text), _cv.FONT_HERSHEY_SIMPLEX, sc, th)
    _cv.putText(img, str(text), ((W - tw) // 2, (H + tht) // 2),
                _cv.FONT_HERSHEY_SIMPLEX, sc, fg, th, _cv.LINE_AA)
    _cv.rectangle(img, (3, 3), (W - 4, H - 4), fg, 3)
    _cv.imwrite(path, img)
    os.chmod(path, 0o644)                   # the sim runs as the airsim account and needs read permission
    return path


def paint_object(c, name, color):
    """Paint by color name. Looks for an exact-match object first (prevents mis-painting where a prefix match grabs an object)."""
    if not color:
        return None
    tex = f"/tmp/uniq_{color}.png"
    if not os.path.exists(tex):
        import cv2 as _cv
        import numpy as _np
        _cv.imwrite(tex, _np.full((64, 64, 3), PAINT_BGR.get(color, (180, 20, 200)), _np.uint8))
        os.chmod(tex, 0o644)
    cands = c.simListSceneObjects(name + ".*") or []
    real = next((o for o in cands if o == name),
                next((o for o in cands if o.startswith(name)
                      and not o[len(name):].startswith("_")), None))
    if real is None:
        real = cands[0] if cands else name
    c.simSetObjectMaterialFromTexture(real, tex)
    time.sleep(0.35)
    return real


def spawn_extras(c, task, port, verbose=True, gz=0.0):
    """Build the episode's **objects and walls**. Names follow the same `BM_*_p{port}` rule as the target (cleaned up in one go, isolated across parallel runs)."""
    import airsim
    made = []
    tid = task.get("id", "T")
    for i, pr in enumerate(task.get("props") or []):
        nm = prop_name_for(task, pr.get("name", f"P{i+1}"), port)
        x, y = pr["xy"]
        sc = [float(v) for v in pr.get("scale", [1, 1, 1])]
        pivot = pr.get("pivot", "ctr")
        z = entry_z(pr, sc[2], pivot, gz)
        c.simSpawnObject(nm, pr["asset"],
                         airsim.Pose(airsim.Vector3r(float(x), float(y), z),
                                     airsim.to_quaternion(0, 0, math.radians(pr.get("yaw_deg", 0)))),
                         airsim.Vector3r(*sc), False)
        time.sleep(0.35)
        if pr.get("text") is not None:                   # paint is the background color
            asp = max(sc[0], sc[1]) / max(sc[2], 1e-6)
            tex = text_texture(pr["text"], aspect=asp, ratio=float(pr.get("text_ratio", 0.55)),
                               bg=pr.get("paint"))
            real = (c.simListSceneObjects(nm + ".*") or [nm])[0]
            c.simSetObjectMaterialFromTexture(real, tex)
            time.sleep(0.35)
        else:
            paint_object(c, nm, pr.get("paint"))
        p = c.simGetObjectPose(nm).position
        if any(v != v for v in (p.x_val, p.y_val)):
            raise RuntimeError(f"object '{nm}' ({pr['asset']}) spawn failed — check the asset name")
        made.append(nm)
        if verbose:
            print(f"  object '{nm}' {pr['asset']} {sc} {pr.get('paint', '-')} → ({p.x_val:.1f}, {p.y_val:.1f}, {z:.2f})",
                  flush=True)
    for i, w in enumerate(task.get("walls") or []):
        nm = f'BM_{tid}_{w.get("name", f"W{i+1}")}_p{port}'
        (x1, y1), (x2, y2) = w["from"], w["to"]
        dx, dy = float(x2) - float(x1), float(y2) - float(y1)
        L = math.hypot(dx, dy)
        if L < 0.5:
            print(f"  [skipped] wall '{nm}' length {L:.2f} m — too short", flush=True)
            continue
        th, hgt = float(w.get("thickness_m", 0.6)), float(w.get("height_m", 3.0))
        mx, my = (float(x1) + float(x2)) / 2.0, (float(y1) + float(y2)) / 2.0
        yaw = math.atan2(dy, dx)
        z = entry_z(w, hgt, "ctr", gz)
        c.simSpawnObject(nm, w.get("asset", "Cube"),
                         airsim.Pose(airsim.Vector3r(mx, my, z), airsim.to_quaternion(0, 0, yaw)),
                         airsim.Vector3r(L, th, hgt), False)
        time.sleep(0.35)
        paint_object(c, nm, w.get("paint", "grey"))
        p = c.simGetObjectPose(nm).position
        if any(v != v for v in (p.x_val, p.y_val)):
            raise RuntimeError(f"wall '{nm}' spawn failed")
        made.append(nm)
        if verbose:
            print(f"  wall '{nm}' length {L:.1f} thickness {th} height {hgt} yaw {math.degrees(yaw):+.0f}° "
                  f"→ ({p.x_val:.1f}, {p.y_val:.1f}, {z:.2f})", flush=True)
    return made


def setup_target(c, task, defaults, port, verbose=True, keep=False):
    """Place the target in the scene (after cleaning up this port's old BM_*) and return the object name. Also builds objects and walls. With keep=True, old ones are not deleted."""
    import airsim
    tgt = task["target"]
    if "existing" in tgt:
        hits = c.simListSceneObjects(tgt["existing"] + ".*")
        if not hits:
            raise RuntimeError(f"'{tgt['existing']}' is not in the scene")
        return hits[0]
    name = target_name_for(task, port)
    x, y = tgt["xy"]
    if not keep:
        try:
            for o in c.simListSceneObjects(f"BM_.*_p{port}"):
                c.simDestroyObject(o)
            time.sleep(1.2)     # simDestroyObject is asynchronous — prevents the ground probe from landing on debris
        except Exception:
            pass
    asset = tgt.get("asset") or defaults.get("spawn_asset")
    if not asset:
        raise RuntimeError("no target asset — set target.asset or defaults.spawn_asset")
    yaw = math.radians(tgt.get("yaw_deg", defaults.get("spawn_yaw_deg", 0)))
    sc = [float(v) for v in tgt.get("scale", [1, 1, 1])]
    pivot = tgt.get("pivot", "base")
    gz = ground_z_of(task, defaults)
    if gz is None:
        gz = measure_ground_z(c, float(x), float(y), port=port)
        if verbose and abs(gz) > 0.05:
            print(f"  ground measured z = {gz:+.2f} (all spawns corrected)", flush=True)
    elif verbose:
        print(f"  ground given z = {gz:+.2f} (ground_z_ned — drop probe skipped)", flush=True)
    global LAST_GROUND_Z
    LAST_GROUND_Z = gz          # used by the runner when building the start pose z (terrain-relative)
    tz = entry_z(tgt, sc[2], pivot, gz)
    c.simSpawnObject(name, asset,
                     airsim.Pose(airsim.Vector3r(float(x), float(y), tz), airsim.to_quaternion(0, 0, yaw)),
                     airsim.Vector3r(*sc), False)
    time.sleep(1.0)
    if tgt.get("text") is not None:                      # sign episode: text on the target itself (paint is the background color)
        asp = max(sc[0], sc[1]) / max(sc[2], 1e-6)
        tex = text_texture(tgt["text"], aspect=asp, ratio=float(tgt.get("text_ratio", 0.55)),
                           bg=tgt.get("paint"))
        real = (c.simListSceneObjects(name + ".*") or [name])[0]
        c.simSetObjectMaterialFromTexture(real, tex)
        time.sleep(0.35)
    else:
        paint = tgt.get("paint") or (task.get("meta") or {}).get("paint_texture")
        if paint:
            real = paint_object(c, name, paint)
            if verbose:
                print(f"  paint: {real} ← {paint}", flush=True)
    p = c.simGetObjectPose(name).position
    if any(v != v for v in (p.x_val, p.y_val)):
        raise RuntimeError(f"'{name}' coordinates are NaN after spawn — check the asset name")
    if abs(p.x_val - x) > 0.5 or abs(p.y_val - y) > 0.5:
        raise RuntimeError(f"spawn coordinate mismatch: requested ({x}, {y}) vs actual ({p.x_val:.2f}, {p.y_val:.2f})")
    if verbose:
        print(f"  target '{name}' placed ({p.x_val:.1f}, {p.y_val:.1f}, {tz:.2f})", flush=True)
    extras = spawn_extras(c, task, port, verbose=verbose, gz=gz)
    if extras and verbose:
        print(f"  {len(extras)} objects/walls placed", flush=True)
    return name


def scene_sane(c, task, target_name):
    """Scene-level validation — is the line of sight open from above the first start pose toward the target?

    If the ground measurement jumps, walls and signs overlap terrain/blocks (observed: robotics E-01 target inside a wall).
    Checks the approach-direction line of sight with simTestLineOfSightBetweenPoints.
    """
    import airsim
    try:
        t = c.simGetObjectPose(target_name).position
        if t.x_val != t.x_val:
            return False
        poses = task.get("start_poses") or {}
        if not poses:
            return True
        sp = next(iter(poses.values()))
        dx, dy = t.x_val - float(sp["x"]), t.y_val - float(sp["y"])
        L = math.hypot(dx, dy) or 1.0
        k = max(0.0, (L - 8.0) / L)        # point 8 m in front of the target, at target height
        p1 = airsim.Vector3r(float(sp["x"]) + dx * k, float(sp["y"]) + dy * k, t.z_val)
        return bool(c.simTestLineOfSightBetweenPoints(p1, t))
    except Exception:
        return True                        # pass when unverifiable (the episode validator filters again later)


def ensure_scene(c, task, defaults, port, force=False, verbose=True):
    """Reuse the target if it already exists (avoids the respawn crash); build anew if missing or force. Returns (name, reused)."""
    name = target_name_for(task, port)
    if "existing" in task["target"]:
        return setup_target(c, task, defaults, port, verbose), True
    hits = c.simListSceneObjects(name + ".*") or []
    exact = [h for h in hits if h == name or (h.startswith(name) and not h[len(name):].startswith("_"))]
    if exact and not force:
        if scene_sane(c, task, exact[0]):
            if verbose:
                print(f"  scene reused: {exact[0]}  (FORCE_SETUP=1 to rebuild it)", flush=True)
            return exact[0], True
        print("  ! reused scene failed the line-of-sight check — rebuilding", flush=True)
    made = setup_target(c, task, defaults, port, verbose)
    if not scene_sane(c, task, made):
        print("  ! new scene also failed the line-of-sight check — rebuilding once more", flush=True)
        made = setup_target(c, task, defaults, port, verbose)
        if not scene_sane(c, task, made):
            print("  ⚠ scene failed the line-of-sight check twice — the target may be occluded (continuing anyway)", flush=True)
    return made, False
