# DroneCATS
# Copyright (c) 2026-present NAVER Cloud Corp.
# Apache-2.0

"""Unique-target spawning — the runner places it directly per episode from the testset's (asset, position).

Why spawn:
  It is hard to create a "unique reference" from objects already in the scene. NH has 69 cars and 119 trees
  packed so densely that a full turn from any start position shows several of the same kind.
  If we place it ourselves, we choose position and color so uniqueness is guaranteed by design. Moving targets
  also require spawning, since simSetObjectPose only works on spawned objects (below).

Constraints confirmed by measurement (AirSimNH):
  - `simSetObjectPose` returns False for actors originally in the scene and does not move them (static).
    For spawned actors it returns True and they actually move (confirmed 24m of travel in 6 s at 4 m/s).
    → moving targets are only possible with spawned objects.
  - Whether an asset renders varies by case (measured on the current sim package.
    Spawn and verify new assets by measurement before using them):
      Car_01 / Saloon_ISM_NewMat / 009_SUV_ISM_NewMat /
      Vehicle_Police_ISM / BicycleMan       ✅ renders and detects (8m occupancy 1~9%, brightness 102~139)
      Ambulance_Truck_FixedTransforms       ✗ huge, floating, black slab (brightness 37) — detection says
                                              visible=True, so verifier numbers do not catch it
      Schoolbus_Small_FixedTransforms       △ oversized (occupancy 22%), floating, paint does not apply
      boxtruck_...LOD0                      ✗ no material (white blob, 0 detections)
    ⚠️ Old names without the suffix (Ambulance_Truck / Saloon_ISM / 009_SUV_ISM /
       Schoolbus_Small) are gone from the registry (not among the 283 in simListAssets) —
       spawning a name that does not exist kills UE entirely (observed 4/4).
    However, the default color of the Car family is gray — color references ("lime car") are made true by paint().
  - `simSetObjectMaterialFromTexture` takes a file path. The sim process (airsim
    user) has to read it, so textures are written to /tmp.

Cleanup of spawned objects uses **the target names written in the testset**, not a name prefix
(the runner deletes the previous target at episode start) — no separate convention or CLI needed.
"""

import math
import os
import time

import airsim

PAINT_BGR = {"orange": (0, 120, 255), "cyan": (255, 255, 0), "magenta": (255, 0, 255),
             "lime": (50, 220, 50), "yellow": (0, 210, 255), "purple": (180, 40, 130),
             "green": (50, 160, 50), "white": (245, 245, 245),
             "black": (15, 15, 15),
             "red": (0, 0, 210),
             "blue": (210, 70, 0)}


def spawn_one(c, name, asset, xyz, yaw_deg=0.0, scale=1.0):
    """Spawn one object with the given name, asset, and coordinates (if it exists, delete it and spawn anew).

    Called directly by the runner with the (asset, position) baked into the testset — the builder has
    already chosen the spot; here we only deterministically place 'that asset at that spot'.
    Placing a moving target anew every episode also effectively resets it to its start position.
    scale: the scale factor specified by the episode (default 1.0). Used when the original asset is oversized, like indoor props
    — the value tuned during episode review is carried into the testset as is.
    Returns: the name of the object actually spawned.
    """
    if name in c.simListSceneObjects():
        try:
            c.simDestroyObject(name)
            time.sleep(0.2)
        except Exception:
            pass
    x, y, z = xyz
    pose = airsim.Pose(airsim.Vector3r(float(x), float(y), float(z)),
                       airsim.to_quaternion(0, 0, math.radians(yaw_deg)))
    s = float(scale or 1.0)
    return c.simSpawnObject(name, asset, pose, airsim.Vector3r(s, s, s), False)


class UnknownPaintColor(ValueError):
    """A color not in the palette. The color-referencing instruction would become false, so the runner stops immediately on this exception."""


def paint(c, name, color):
    """Paint a spawned object a solid color — so color references hold even with the gray default render.

    (Observed: the Car_01 family is gray by default, so 'lime/magenta car' instructions were false, and
     boxtruck had no material at all and did not even render. Painting fixes both.)
    """
    import numpy as _np
    import cv2 as _cv2
    if color not in PAINT_BGR:
        # do not proceed without color — instructions reference color like "the blue ball", so
        # running with a gray target silently contaminates that episode. Stopping before measurement is right.
        # (callers catch UnknownPaintColor separately so this exception is not mistaken for a spawn failure)
        raise UnknownPaintColor(f"unknown paint color '{color}' — add it to the palette or "
                                f"choose one of: {sorted(PAINT_BGR)}")
    tex = f"/tmp/uniq_{color}.png"
    if not os.path.exists(tex):
        _cv2.imwrite(tex, _np.full((64, 64, 3), PAINT_BGR[color], _np.uint8))
        os.chmod(tex, 0o644)
    return c.simSetObjectMaterialFromTexture(name, tex)
