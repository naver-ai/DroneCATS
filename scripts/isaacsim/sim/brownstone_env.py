# DroneCATS
# Copyright (c) 2026-present NAVER Cloud Corp.
# Apache-2.0

"""Register the Brownstone variants in the Pegasus environment list.

`pegasus_sim.py` accepts `--env` only as a **key** of `SIMULATION_ENVIRONMENTS`, not a path
(`sys.exit(2)` if it is `not in` the dict). `PegasusInterface.load_environment(usd_path)` itself
takes a plain USD path, so adding keys to that dict lets us load external scenes without
patching Pegasus.

`SIMULATION_ENVIRONMENTS` is a module-level dict and `pegasus_sim` references the same object
(ids verified equal), so injecting after import is visible to the gate.

`spawn_ground_m` is the **measured ground z at the spawn point (0,0)**. Pegasus spawns at
world z = altitude, so this value must be added when the ground is not at 0.

⚠️ The old field `street_z_m` was the **site-wide minimum ground** (-0.66 m) from a 41x21 probe,
and it was being added as if it were the spawn-point ground. Morning's ground at (0,0) measures
**0.000 m**, so that offset put the drone 0.59 m **underground**: the camera was fully black
(mean luminance 0.0) and depth was all `inf`. The site minimum is kept as `site_ground_min_m`
for reference only.

Ground must be measured **after `apply_collision` excludes the source library**. Before the
exclusion, the original asset store stacked at the origin gets hit and (0,0) reads +14.77 m.
"""
import os

PACK = os.environ.get(
    "DATAGEN_BROWNSTONE",
    os.path.expanduser("~/omniverse-assets/Brownstone/Demos/AEC/BrownstoneDemo"),
)

# Measured per variant
VARIANTS = {
    "Brownstone Morning": {   # default. Brick facades and window openings render
        "usd": "World_BrownstoneDemopack_Morning(20Gb).usd",
        "extent_m": (380.0, 300.0, 36.2),
        "spawn_ground_m": 0.0, "site_ground_min_m": -0.66, "mesh": 39008,
    },
    "Brownstone Night": {     # DR axis B (day/night). Same footprint, luminance 113 -> 56
        "usd": "World_BrownstoneDemopack_Night(20Gb).usd",
        "extent_m": (380.0, 300.0, 220.2),
        "spawn_ground_m": 0.0, "site_ground_min_m": -0.66, "mesh": 39008,
    },
    "Brownstone Lite": {      # DR axis D (LOD), low end. White massing; do not use as default
        "usd": "World_BrownstoneDemopack_Lite(11Gb).usd",
        "extent_m": (380.0, 300.0, 37.2),
        "spawn_ground_m": 0.0, "site_ground_min_m": -0.66, "mesh": 7123,
    },
    "Brownstone Close": {     # single-building close-up (69 x 69.5 m). Different frame from the three above
        "usd": "World_BrownstoneDemopack_Brownstone(8Gb).usd",
        "extent_m": (69.0, 69.5, 37.1),
        "spawn_ground_m": None, "site_ground_min_m": None, "mesh": 4760,
    },
}


def usd_path(name: str) -> str:
    return os.path.join(PACK, VARIANTS[name]["usd"])


def register(sim_environments: dict) -> list[str]:
    """Add the variants to `SIMULATION_ENVIRONMENTS` and return the registered keys.
    Variants whose file is missing are skipped (the pack may be partially downloaded)."""
    added = []
    for name in VARIANTS:
        p = usd_path(name)
        if not os.path.exists(p):
            continue
        sim_environments[name] = p
        added.append(name)
    return added
