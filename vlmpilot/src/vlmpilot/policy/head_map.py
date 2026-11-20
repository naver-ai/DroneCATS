# DroneCATS
# Copyright (c) 2026-present NAVER Cloud Corp.
# Apache-2.0

"""Overview map for the commander (``head.map: true``) — a top-down "radar" picture with NO text.

The commander otherwise sees only the bodies' forward cameras. A control-tower view fixes the
failure class we measured (two drones sent to one marker, drones wandering off to distant
scenery, the head believing a drone is in a lane it never entered) without touching the bodies.

What it deliberately does NOT carry: any text or number. Markers are drawn as coloured shapes
only, so the map can never substitute for a drone reading a marker up close — the information
regime "numbers are read by bodies" is preserved by construction, not by hiding them.

Inputs are plain data (the runner writes ``scene`` into the episode config; the head reads live
global positions), so the renderer is deterministic and unit-testable without a simulator.

Frame: NED x = north (drawn up), y = east (drawn right). Scale bar and compass are drawn as
shapes only (no text) — the commander does not need metres, it needs relations.
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

MAP_LABEL = "OVERVIEW MAP (top-down radar, north up; no text is readable on it):"
MAP_FRAME_NAME = "MAP"          # head_frames/call_NNN_MAP.jpg

_PAINT_BGR: dict[str, tuple[int, int, int]] = {
    "red": (40, 40, 220), "blue": (200, 60, 20), "white": (245, 245, 245), "grey": (128, 128, 128),
    "green": (60, 160, 60), "lime": (40, 220, 90), "magenta": (180, 20, 200), "yellow": (20, 220, 240),
    "orange": (30, 140, 240), "purple": (160, 30, 110), "cyan": (220, 210, 40), "black": (20, 20, 20),
}
_DRONE_BGR = [(180, 119, 31), (14, 127, 255), (44, 160, 44), (189, 103, 148), (127, 127, 127), (75, 190, 23)]
_BG = (236, 236, 236)
_STRUCT = (85, 85, 85)

# A car is drawn as its footprint, not as a blob: 4.6 m long, 1.8 m wide (the Car_01 asset the
# benches spawn). A square told the commander nothing about which way a car lies, so two drones
# sent to the same row could not be told apart on the map from two sent down opposite kerbs.
CAR_FOOTPRINT_M = (4.6, 1.8)
_CAR_ASSET_PREFIX = "car"


def _is_car(entry: Mapping[str, Any]) -> bool:
    return str(entry.get("asset") or "").lower().startswith(_CAR_ASSET_PREFIX)


def _is_plate(entry: Mapping[str, Any]) -> bool:
    """A licence plate cube — it belongs to a car and is never its own marker on the map.

    Two rules because the earlier tasks (E-11) name the target's plates ``TARGET_PL1/2`` without
    a ``plate_of`` field, while every later one carries ``plate_of``."""
    return bool(entry.get("plate_of")) or "_PL" in str(entry.get("name") or "")


def _plates_of(task: Mapping[str, Any], owner: str) -> list[Mapping[str, Any]]:
    """The plates of one car, in ``PL1``-first order. ``owner`` is a prop name, or ``TARGET``."""
    mine = [p for p in (task.get("props") or [])
            if _is_plate(p) and (str(p.get("plate_of") or "") == owner
                                 or str(p.get("name") or "").startswith(owner + "_PL"))]
    return sorted(mine, key=lambda p: str(p.get("name") or ""))


def _car_heading_deg(task: Mapping[str, Any], owner: str, yaw_deg: float) -> float:
    """Which way the car points: the vector from its rear plate to its front one.

    Without plates, fall back to ``yaw_deg + 90``: the Car_01 mesh lies along +y (east) at yaw 0.
    Measured on the NH tasks — E-12's car is at yaw 0 with its plates at (0.26, 12.91) and
    (0.26, 7.81), i.e. 5.1 m apart along y alone; every car in the set agrees with yaw + 90 to
    within 0.1°, so the fallback is the same heading, not a guess."""
    pl = _plates_of(task, owner)
    if len(pl) >= 2 and all(p.get("xy") for p in pl[:2]):
        front, rear = pl[0], pl[1]
        dx = float(front["xy"][0]) - float(rear["xy"][0])
        dy = float(front["xy"][1]) - float(rear["xy"][1])
        if math.hypot(dx, dy) > 0.5:                 # two plates on top of each other say nothing
            return math.degrees(math.atan2(dy, dx))
    return float(yaw_deg or 0.0) + 90.0


def car_corners_m(xy: Sequence[float], heading_deg: float,
                  size: Sequence[float] = CAR_FOOTPRINT_M) -> list[tuple[float, float]]:
    """The four footprint corners in world metres (x north, y east), front pair first.

    Pure geometry so the shape can be checked without rendering."""
    h = math.radians(float(heading_deg))
    fx, fy = math.cos(h), math.sin(h)                # along the car
    sx, sy = -math.sin(h), math.cos(h)               # across it
    hl, hw = float(size[0]) / 2.0, float(size[1]) / 2.0
    cx, cy = float(xy[0]), float(xy[1])
    return [(cx + fx * hl + sx * hw, cy + fy * hl + sy * hw),
            (cx + fx * hl - sx * hw, cy + fy * hl - sy * hw),
            (cx - fx * hl - sx * hw, cy - fy * hl - sy * hw),
            (cx - fx * hl + sx * hw, cy - fy * hl + sy * hw)]


def scene_of_task(task: Mapping[str, Any]) -> dict:
    """Geometry the map needs, and nothing that reads: walls, coloured markers (no ``text``),
    start poses. Returns ``{}`` for a task without any of it (single-drone benches).

    A car becomes a footprint (``kind: "car"`` + ``heading_deg``), and its licence plates are
    dropped — the plate is what the car IS on the map, not a second object beside it."""
    walls = [
        {"from": [float(w["from"][0]), float(w["from"][1])], "to": [float(w["to"][0]), float(w["to"][1])],
         "thickness_m": float(w.get("thickness_m", 0.6))}
        for w in (task.get("walls") or []) if w.get("from") and w.get("to")
    ]
    markers = []
    entries = [(str(p.get("name") or ""), p) for p in (task.get("props") or [])]
    if task.get("target"):
        entries.append(("TARGET", task["target"]))
    for owner, p in entries:
        if not p.get("xy") or not p.get("paint") or str(p.get("paint")).lower() == "grey":
            continue   # posts are grey and carry nothing
        if _is_plate(p):
            continue   # a plate is part of its car, and its 0.7 m cube only litters the map
        m = {"xy": [float(p["xy"][0]), float(p["xy"][1])], "paint": str(p["paint"]).lower()}
        if _is_car(p):
            m["kind"] = "car"
            m["heading_deg"] = round(_car_heading_deg(task, owner, p.get("yaw_deg", 0.0)), 1)
            m["size"] = list(CAR_FOOTPRINT_M)
        else:
            sc = p.get("scale") or [1, 1, 1]
            m["size"] = [float(sc[0]), float(sc[1])]
        markers.append(m)
    starts = {str(k): [float(v["x"]), float(v["y"])] for k, v in (task.get("start_poses") or {}).items()
              if isinstance(v, Mapping) and "x" in v and "y" in v}
    if not walls and not markers:
        return {}
    return {"walls": walls, "markers": markers, "start_xy": starts}


def _extent(scene: Mapping[str, Any], drones: Mapping[str, Any], margin: float = 8.0) -> tuple[float, float, float, float]:
    xs: list[float] = []; ys: list[float] = []
    for w in scene.get("walls", []):
        xs += [w["from"][0], w["to"][0]]; ys += [w["from"][1], w["to"][1]]
    for m in scene.get("markers", []):
        xs.append(m["xy"][0]); ys.append(m["xy"][1])
    for xy in (scene.get("start_xy") or {}).values():
        xs.append(xy[0]); ys.append(xy[1])
    for d in drones.values():
        if d and d.get("xy"):
            xs.append(d["xy"][0]); ys.append(d["xy"][1])
    if not xs:
        xs, ys = [0.0], [0.0]
    x0, x1, y0, y1 = min(xs) - margin, max(xs) + margin, min(ys) - margin, max(ys) + margin
    side = max(x1 - x0, y1 - y0)                     # square, so metres are metres in both axes
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    return cx - side / 2, cx + side / 2, cy - side / 2, cy + side / 2


def _draw_car(img: np.ndarray, marker: Mapping[str, Any], px, heading_mark: bool) -> None:
    """One car: its footprint as a filled rotated rectangle, outlined so two parked side by side
    stay two. ``px(x, y)`` maps world metres to pixels."""
    col = _PAINT_BGR.get(marker["paint"], (120, 120, 120))
    corners = car_corners_m(marker["xy"], marker.get("heading_deg", 0.0),
                            marker.get("size") or CAR_FOOTPRINT_M)
    pts = np.array([px(x, y) for x, y in corners], np.int32)
    cv2.fillConvexPoly(img, pts, col, cv2.LINE_AA)
    cv2.polylines(img, [pts], True, (30, 30, 30), 1, cv2.LINE_AA)
    if heading_mark:                       # a notch on the front edge, pointing the way it faces
        (fx1, fy1), (fx2, fy2) = corners[0], corners[1]
        mid = ((fx1 + fx2) / 2, (fy1 + fy2) / 2)
        h = math.radians(float(marker.get("heading_deg", 0.0)))
        tip = (mid[0] + 0.9 * math.cos(h), mid[1] + 0.9 * math.sin(h))
        cv2.fillConvexPoly(img, np.array([px(fx1, fy1), px(fx2, fy2), px(*tip)], np.int32),
                           (30, 30, 30), cv2.LINE_AA)


def render(scene: Mapping[str, Any], drones: Mapping[str, Mapping[str, Any] | None],
           trails: Mapping[str, Sequence[Sequence[float]]] | None = None, size: int = 640,
           heading_mark: bool = False) -> np.ndarray:
    """BGR image. ``drones``: name → {"xy": [x, y], "yaw_deg": h} (global) or None when unknown.
    ``trails``: name → recent global xy list. Drone order in ``drones`` fixes colour and number.

    ``heading_mark`` puts a small notch at the front end of each car footprint. Off by default:
    which end is the front is not information the commander needs to place its drones, and every
    extra glyph is one more thing on a map whose whole point is that it says nothing in words."""
    img = np.full((size, size, 3), _BG, np.uint8)
    x0, x1, y0, y1 = _extent(scene, drones)
    ppm = size / (x1 - x0)                            # pixels per metre

    def px(x: float, y: float) -> tuple[int, int]:    # north up: x → up, y → right
        return int(round((y - y0) * ppm)), int(round((x1 - x) * ppm))

    for w in scene.get("walls", []):
        t = max(3, int(round(w.get("thickness_m", 0.6) * ppm)))
        cv2.line(img, px(*w["from"]), px(*w["to"]), _STRUCT, t, cv2.LINE_AA)
    for m in scene.get("markers", []):
        if m.get("kind") == "car":
            _draw_car(img, m, px, heading_mark)
            continue
        col = _PAINT_BGR.get(m["paint"], (120, 120, 120))
        hw = max(3, int(round(max(m["size"][1], 1.2) / 2 * ppm)))     # along y (east)
        hd = max(3, int(round(max(m["size"][0], 1.2) / 2 * ppm)))     # along x (north)
        cx, cy = px(*m["xy"])
        cv2.rectangle(img, (cx - hw, cy - hd), (cx + hw, cy + hd), col, -1)
        cv2.rectangle(img, (cx - hw, cy - hd), (cx + hw, cy + hd), (30, 30, 30), 1)
    for xy in (scene.get("start_xy") or {}).values():          # start slots: hollow grey circles
        cv2.circle(img, px(*xy), max(3, int(0.8 * ppm)), (150, 150, 150), 1, cv2.LINE_AA)

    names = list(drones.keys())
    for i, name in enumerate(names):
        col = _DRONE_BGR[i % len(_DRONE_BGR)]
        pts = [tuple(p) for p in (trails or {}).get(name, []) if p is not None]
        for a, b in zip(pts, pts[1:]):
            cv2.line(img, px(*a), px(*b), col, 2, cv2.LINE_AA)
        d = drones.get(name)
        if not d or not d.get("xy"):
            continue
        cx, cy = px(*d["xy"])
        r = max(7, int(1.6 * ppm))
        cv2.circle(img, (cx, cy), r, col, -1, cv2.LINE_AA)
        cv2.circle(img, (cx, cy), r, (255, 255, 255), 2, cv2.LINE_AA)
        yaw = math.radians(float(d.get("yaw_deg", 0.0) or 0.0))
        tip = px(d["xy"][0] + 4.0 * math.cos(yaw), d["xy"][1] + 4.0 * math.sin(yaw))
        cv2.arrowedLine(img, (cx, cy), tip, col, 3, cv2.LINE_AA, tipLength=0.35)
        # the drone's roster number is the one glyph on the map — it names the image next to it
        cv2.putText(img, str(i + 1), (cx - 5, cy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

    # compass (north arrow) and a 10 m scale bar — shapes only
    cv2.arrowedLine(img, (size - 30, 60), (size - 30, 18), (30, 30, 30), 2, cv2.LINE_AA, tipLength=0.3)
    bar = int(round(10.0 * ppm))
    cv2.line(img, (16, size - 16), (16 + bar, size - 16), (30, 30, 30), 3)
    cv2.line(img, (16, size - 22), (16, size - 10), (30, 30, 30), 2)
    cv2.line(img, (16 + bar, size - 22), (16 + bar, size - 10), (30, 30, 30), 2)
    return img


def global_positions(clients: Sequence[Any], names: Sequence[str]) -> dict[str, dict | None]:
    """name → {"xy", "yaw_deg"} in the GLOBAL frame via ``simGetObjectPose`` (vehicle poses from
    ``getMultirotorState`` are spawn-relative and differ per vehicle). A failed
    read leaves that drone off the map rather than mis-placing it."""
    out: dict[str, dict | None] = {}
    for client, name in zip(clients, names):
        try:
            pose = client.simGetObjectPose(name)
            p, o = pose.position, pose.orientation
            # yaw from quaternion (NED, z-down): standard formula, no airsim import needed
            siny = 2.0 * (o.w_val * o.z_val + o.x_val * o.y_val)
            cosy = 1.0 - 2.0 * (o.y_val * o.y_val + o.z_val * o.z_val)
            yaw = math.degrees(math.atan2(siny, cosy))
            xy = [float(p.x_val), float(p.y_val)]
            if any(math.isnan(v) for v in xy):
                raise ValueError("nan position")
            out[name] = {"xy": xy, "yaw_deg": yaw}
        except Exception:  # noqa: BLE001 — one unknown position must not blank the map
            out[name] = None
    return out
