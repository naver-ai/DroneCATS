# DroneCATS
# Copyright (c) 2026-present NAVER Cloud Corp.
# Apache-2.0

"""QF judge — answers only one question: "at this moment the model declared arrival, is it a success?"

Fully separated from the approach algorithm, scanning, and back-projection. Those will keep improving,
and the judging criteria must live in this one file only for cross-experiment comparisons to hold.

All judging inputs are simulator GT (model outputs are never used as inputs):
  - distance : 3D Euclidean from simGetVehiclePose / simGetObjectPose
           **to what** the distance is measured is decided by `judge_anchors` — by default the target; for vehicle episodes,
           the target car's 2 license plates (the object the mission points to is what is judged)
  - visible : simGetDetections — detection results provided directly by the engine

Why visibility is judged with the detection API (measured on the NH map):
  Compared against segmentation toggle-diff (the previous method) at 8 bearings × 12m, the conclusions agreed.
  Bearings where detection fires (0/45/180/225/315°) had toggle 633~1584px; bearings with 0 detections
  (90/135°) had toggle 0~7px as well. So detection is not a mere frustum test but looks at actual render
  visibility — field-of-view and occlusion are handled in a single call.
  Toggle-diff takes 6 frames (~2s), whereas detection takes tens of ms.
  One difference: at 270°, toggle 202px / detection 0 — detection is more conservative about partial occlusion.
  We judged this better for the bench (counting a barely-peeking state as success
  invites threshold-tuning debates).

⚠️ Calling convention: simGetDetections must be called **with no delay** right after simGetImages.
   Putting a sleep in between turns the capture component off again, the "recently rendered" state is lost,
   and 0 hits are silently returned (measured: a single 0.3s sleep → no detections at any bearing → every verdict a failure).

⚠️ APIs not used and why (do not try again):
   simTestLineOfSightToPoint      — hits the vehicle's own collision, so everything beyond 0.5m is False.
                                     Even 50m up in empty sky is False.
   simTestLineOfSightBetweenPoints — the function itself works (True in open space aloft), but giving the target
                                     as the endpoint makes the target itself block the ray, so always False.
"""

import fnmatch
import math
from dataclasses import asdict, dataclass

import airsim


@dataclass
class Judgment:
    """Verdict for a single QF."""
    success: bool
    reason: str            # "ok" | "too_far" | "not_visible" | "too_far+not_visible"
    dist_m: float          # 3D distance (sim GT)
    visible: bool          # detected or not (reflects both field of view and occlusion)
    bbox_px: float         # 2D detection box area (px²). Not used for judging, for logging
    delta_m: float
    detail: str = ""
    # distance from the camera to the target's "nearest surface" (nearest point of box3D). Unlike center distance
    # dist_m, it reflects object size and viewing direction. None if no detection.
    # not used for judging (success) — for the record.
    dist_surface_m: float | None = None
    # minimum 3D distance to the judgment anchors (judge_anchors) and that anchor's name. If the only anchor is the target,
    # it equals dist_m. **δ comparison uses this value** — dist_m is kept for post-hoc comparison.
    dist_anchor_m: float | None = None
    anchor: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


# ──────────────────────────────────────────────────────────────── judgment anchors
# The episode decides "what you must reach to count as arrived". By default it is the single target object, but for vehicle episodes
# (NH license-plate scenario) the anchors are **the target car's 2 license plates** — the mission is "find the car with
# plate 5217", so measuring δ from the car center would penalize a drone reading the plate up close by half the car's
# length (2.3 m front/back for Car_01). What is judged is the object the mission points to.
# The mission sentence is not changed — only the judgment anchors change.
#
# This is the one place that builds the anchor list. Online verdict (capture time), post-hoc replay, and post-hoc re-judging
# all use the minimum distance over this list — if each path used a different distance, the same episode would be scored differently.
TARGET_ANCHOR = "target"        # name used when the anchor is the target itself


def is_target_plate(entry) -> bool:
    """Is this prop a license plate of **the target car**? `plate_of == "TARGET"` or name `TARGET_PL*`.

    Why both rules are checked: in early episodes (E-11) the target plates have no `plate_of`, only the names
    `TARGET_PL1/2`. Decoy car plates (`P4_PL1`, `plate_of: "P4"`) match neither.
    """
    name = str((entry or {}).get("name") or "")
    return str((entry or {}).get("plate_of") or "") == "TARGET" or name.startswith("TARGET_PL")


def judge_anchors(task) -> list[tuple[str, dict]]:
    """Judgment anchors for this episode — `[(name, episode entry)]`.

    The target car's plates if any (usually 2, front and rear), otherwise just the target itself
    (`TARGET_ANCHOR`). Sign episodes (hard·sum) have no plates, so it is the target center as before.
    """
    plates = [p for p in (task.get("props") or []) if is_target_plate(p)]
    if plates:
        return [(str(p.get("name")), p) for p in plates]
    return [(TARGET_ANCHOR, task.get("target") or {})]


def anchor_xyz_from_task(task, ground_z: float = 0.0) -> dict[str, list[float]]:
    """Global anchor positions (NED) from episode coordinates alone. Used for re-judging without a sim.

    The z rule is the same as spawning (`bench_scene.entry_z`): `z_ned` as is if present, otherwise
    ground − `z_off_m`. If the anchor is the target itself, returns an empty dict — in that case the target position
    must come from the recorded `target_xyz` (including moving targets).
    """
    out: dict[str, list[float]] = {}
    for name, e in judge_anchors(task):
        if name == TARGET_ANCHOR:
            continue
        z = float(e["z_ned"]) if "z_ned" in e else float(ground_z) - float(e.get("z_off_m", 0.0))
        out[name] = [float(e["xy"][0]), float(e["xy"][1]), z]
    return out


def anchor_min_dist(drone_xyz, anchor_xyz) -> tuple[float, str]:
    """3D distance from the drone to the nearest anchor and that anchor's name. `(inf, "")` if there are no anchors."""
    best = (math.inf, "")
    for name, xyz in (anchor_xyz or {}).items():
        d = math.dist(drone_xyz, xyz)
        if d < best[0]:
            best = (d, name)
    return best


def judged_dist(ev):
    """The **judged distance** of this measurement — the minimum anchor distance if attached, otherwise the target-center distance.

    Old episodes (verdict.json) and the single-drone path have no `dist_anchor_m` — in that case it is as before.
    """
    d = (ev or {}).get("dist_anchor_m")
    return (ev or {}).get("dist_m") if d is None else d


def detect_target(client, target_name: str, camera: str = "front_0",
                  vehicle_name: str = "", image_type=airsim.ImageType.Scene,
                  radius_m: float = 300.0, mesh_pattern: str = None):
    """Whether the target is visible to the camera right now. Returns (visible, bbox_px²).

    mesh_pattern is a UE wildcard (`*`/`?`, not a regex) matched against the **mesh name**, not the actor
    name. NH's placed actors match by their name ("Car_10") directly, but
    actors spawned with `simSpawnObject` have the asset ("Car_01") as their mesh name, so an actor-name
    pattern never catches them (measured: querying with `*`, the detection result's
    name is the actor name "BM_E-02_p64002"). So when mesh_pattern is given, filter with the asset
    wildcard and pick out the returned detections by **actor name (target_name)** —
    only the target remains even if decoys of the same asset (BM_E-02_P4_…) come along.
    """
    dets = _get_detections(client, target_name, camera=camera,
                           vehicle_name=vehicle_name, image_type=image_type,
                           radius_m=radius_m, mesh_pattern=mesh_pattern)
    if not dets:
        return False, 0.0
    area = max(_box_area(d.box2D) for d in dets)
    return True, area


def _get_detections(client, target_name: str, camera: str = "front_0",
                    vehicle_name: str = "", image_type=airsim.ImageType.Scene,
                    radius_m: float = 300.0, mesh_pattern: str = None, render: bool = True):
    """Returns the raw detections (DetectionInfo list) as is — shared by 2D/3D boxes.

    render=False: does not request a new render, only queries detections on the current frame (≈8ms). Used right after capture
    in flight, while the sim keeps rendering. In a paused (simPause) sim, render=True is needed to turn a frame.
    """
    pattern = mesh_pattern or target_name
    client.simSetDetectionFilterRadius(camera, image_type, radius_m * 100,
                                       vehicle_name=vehicle_name)
    client.simClearDetectionMeshNames(camera, image_type, vehicle_name=vehicle_name)
    client.simAddDetectionFilterMeshName(camera, image_type, pattern,
                                         vehicle_name=vehicle_name)
    if render:
        # trigger a render → query immediately (no sleep in between, see the comment at the top of the file)
        client.simGetImages([airsim.ImageRequest(camera, image_type, False, False)],
                            vehicle_name=vehicle_name)
    dets = client.simGetDetections(camera, image_type, vehicle_name=vehicle_name)
    if mesh_pattern:
        # the asset pattern also returns other actors of the same asset — keep only the target by actor name
        dets = [d for d in dets
                if d.name == target_name or fnmatch.fnmatchcase(d.name, target_name)]
    return dets


def _box_area(b) -> float:
    return abs((b.max.x_val - b.min.x_val) * (b.max.y_val - b.min.y_val))


def _box3d_surface_dist(box) -> float:
    """Nearest distance from the camera origin to the surface of box3D (3D bounding box).

    box3D is **in the camera frame** — measured (NH/Car_01): fixing the target
    and backing the drone off to 6/10/14m pushes min/max out by that much, and
    relative_pose.x matches separation-0.5m (camera front-mount offset) exactly.
    So the origin (0,0,0) = camera, and the origin clamped into the box is the nearest surface point.

    min/max are not guaranteed to be sorted per component (measured: on the NED z axis
    min.z=-0.13 > max.z=-1.84) — sort per axis, then clamp.
    Unlike a sphere approximation, as a box it reflects surface differences by viewing direction (front/side/diagonal, etc.).
    """
    d2 = 0.0
    for lo, hi in (sorted((box.min.x_val, box.max.x_val)),
                   sorted((box.min.y_val, box.max.y_val)),
                   sorted((box.min.z_val, box.max.z_val))):
        if lo > 0.0:
            d2 += lo * lo        # origin outside the box (below) → nearest point is lo
        elif hi < 0.0:
            d2 += hi * hi        # origin outside the box (above) → nearest point is hi
        # if lo <= 0 <= hi, inside the box on this axis → contribution 0
    return math.sqrt(d2)


def _grade(dist: float, visible: bool, bbox: float, delta_m: float,
           require_visible: bool, dist_surface_m: float = None,
           dist_anchor_m: float = None, anchor: str = "") -> Judgment:
    """Measurements (distance, visibility) → success verdict. Wherever the inputs were measured, this is the single criterion.

    `dist` is to the target center, `dist_anchor_m` is the minimum distance to the judgment anchors (`judge_anchors`).
    If anchors are given, **δ is compared against the anchor distance** — the center distance is only recorded (for post-hoc comparison).
    """
    judged = dist if dist_anchor_m is None else dist_anchor_m
    near = judged <= delta_m
    ok = near and (visible or not require_visible)

    if ok:
        reason = "ok"
    elif not near and require_visible and not visible:
        reason = "too_far+not_visible"
    elif not near:
        reason = "too_far"
    else:
        reason = "not_visible"

    parts = []
    if not near:
        where = f" ({anchor})" if anchor and anchor != TARGET_ANCHOR else ""
        parts.append(f"distance {judged:.1f}m{where} > δ{delta_m}m")
    if require_visible and not visible:
        parts.append("target not detected (out of view or occluded)")

    return Judgment(success=bool(ok), reason=reason, dist_m=float(dist),
                    visible=bool(visible), bbox_px=float(bbox),
                    delta_m=float(delta_m), detail="; ".join(parts),
                    dist_surface_m=(None if dist_surface_m is None
                                    else float(dist_surface_m)),
                    dist_anchor_m=(None if dist_anchor_m is None else float(dist_anchor_m)),
                    anchor=str(anchor))


def vehicle_global_position(client, vehicle_name: str = ""):
    """The vehicle's **global** position.

    simGetVehiclePose gives coordinates **relative to that vehicle's spawn origin**, so for Drone2~N in multi-drone
    the distance is off by the spawn offset (8~11 m on an 8 m grid) — the δ=5 m verdict silently breaks down.
    The target uses simGetObjectPose (global), so the vehicle is measured in the same frame. Single (offset 0) gives the same value.
    If simGetObjectPose is NaN (vehicle object not found), falls back to the relative pose.
    """
    try:
        name = vehicle_name or (client.listVehicles() or [""])[0]
        p = client.simGetObjectPose(name).position
        if not any(v != v for v in (p.x_val, p.y_val, p.z_val)):
            return p
    except Exception:
        pass
    return client.simGetVehiclePose(vehicle_name=vehicle_name).position


def measure_qf(client, target_name: str, delta_m: float,
               require_visible: bool = True, camera: str = "0",
               vehicle_name: str = "", mesh_pattern: str = None,
               anchor_xyz: dict = None) -> Judgment:
    """QF judgment with a raw AirSim client — called directly by the runner without the io adapter.

    Entry point for the verifier to obtain verdicts without planting a judging hook inside the executor (for compatibility).
    The current runner instead measures distance (_pose_now) and visibility (capture_visibility) separately at the
    capture moment and attaches them to the frame — so that distance and visibility come from the same pose.
    """
    d = vehicle_global_position(client, vehicle_name)
    t = client.simGetObjectPose(target_name).position
    dist = math.dist((d.x_val, d.y_val, d.z_val), (t.x_val, t.y_val, t.z_val))
    dets = _get_detections(client, target_name, camera=camera,
                           vehicle_name=vehicle_name, mesh_pattern=mesh_pattern)
    visible = bool(dets)
    bbox = max(_box_area(d2.box2D) for d2 in dets) if dets else 0.0
    # nearest-surface distance: nearest surface among the matched detections (None if no detection)
    dist_surface = (min(_box3d_surface_dist(d2.box3D) for d2 in dets)
                    if dets else None)
    da, anchor = anchor_min_dist((d.x_val, d.y_val, d.z_val), anchor_xyz)
    return _grade(dist, visible, bbox, delta_m, require_visible,
                  dist_surface_m=dist_surface,
                  dist_anchor_m=(None if anchor_xyz is None else da), anchor=anchor)


def judge_qf(io, target_name: str, delta_m: float,
             require_visible: bool = True, camera: str = None,
             mesh_pattern: str = None, anchor_xyz: dict = None) -> Judgment:
    """Success verdict at the moment of the QF declaration (io adapter version — an object holding client/vehicle/camera).

    The runner measures distance (_pose_now) and visibility (capture_visibility) separately at the capture moment. This function is kept because it is the
    compatibility contract imported by the judging hook (_verify_here) of the old control-improvement branch.

    Success = 3D distance ≤ delta_m  AND  (if require_visible) the target is detected.
    If `anchor_xyz` (name → global xyz) is given, δ is measured as the minimum distance to those anchors.

    Calling contract: the runner calls it right after the model outputs qf=True, **before moving the drone
    any further**. Calling it after moving measures a different state.
    """
    cam = camera or getattr(io, "camera", "front_0")
    veh = getattr(io, "vehicle", "")
    drone = io.drone_xyz()
    dist = float(io.dist3d(drone, io.object_xyz(target_name)))
    visible, bbox = detect_target(io.client, target_name, camera=cam,
                                  vehicle_name=veh, mesh_pattern=mesh_pattern)
    da, anchor = anchor_min_dist(drone, anchor_xyz)
    return _grade(dist, visible, bbox, delta_m, require_visible,
                  dist_anchor_m=(None if anchor_xyz is None else da), anchor=anchor)
