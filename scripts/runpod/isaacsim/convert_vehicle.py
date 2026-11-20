# DroneCATS
# Copyright (c) 2026-present NAVER Cloud Corp.
# Apache-2.0

"""Convert the CarConcept glTF sample (CC-BY-4.0, Khronos glTF-Sample-Assets) to USD for the car targets.

Run with Isaac Sim's bundled Python:

    ~/isaacsim/python.sh scripts/runpod/isaacsim/convert_vehicle.py <CarConcept.glb> <out dir>

Writes <out dir>/CarConcept.usd (+ textures). The benchmark reads it from
$DATAGEN_VEHICLES/CarConcept/CarConcept.usd (default ~/omniverse-assets/Vehicles).
Units and up-axis are fixed later at spawn time from the layer metadata (glTF is Y-up, cm after conversion).
"""
import asyncio
import os
import sys

from isaacsim import SimulationApp

app = SimulationApp({"headless": True})

from isaacsim.core.utils.extensions import enable_extension  # noqa: E402

enable_extension("omni.kit.asset_converter")
app.update()

import omni.kit.asset_converter as converter  # noqa: E402


async def convert(src, dst):
    ctx = converter.AssetConverterContext()
    ctx.ignore_materials = False
    ctx.ignore_animations = True
    ctx.ignore_camera = True
    ctx.ignore_light = True
    task = converter.get_instance().create_converter_task(src, dst, None, ctx)
    ok = await task.wait_until_finished()
    if not ok:
        print(f"[VEHICLE] conversion failed: {task.get_status()} {task.get_error_message()}", flush=True)
    return ok


def main():
    src, out_dir = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
    os.makedirs(out_dir, exist_ok=True)
    dst = os.path.join(out_dir, "CarConcept.usd")
    fut = asyncio.ensure_future(convert(src, dst))
    while not fut.done():
        app.update()
    ok = fut.result() and os.path.exists(dst)
    print(f"[VEHICLE] {'OK' if ok else 'FAILED'} → {dst}", flush=True)
    app.close()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
