#!/usr/bin/env python
# DroneCATS
# Copyright (c) 2026-present NAVER Cloud Corp.
# Apache-2.0

"""Vehicle reset: returns vehicles to their spawn point (landed) without restarting the sim.

Counterpart of AirSim's `scripts/airsim/reset_drone.sh` (= `client.reset()`).

    python3 scripts/isaacsim/pegasus_reset.py            # restore pose + restart PX4
    python3 scripts/isaacsim/pegasus_reset.py --no-px4   # restore pose only (keep PX4)

**When it is needed.** If a vehicle clips the ground and flips, PX4 ends up in this state:

    WARN [health_and_arming_checks] Preflight Fail: Attitude failure (pitch)
    WARN [health_and_arming_checks] Compass needs calibration - Land now!
    WARN [failsafe] Failsafe activated

After that `is_armable` stays False and process B gets stuck at the connect stage. Without
this, process A had to be restarted every time (~100 s including scene loading).

**Why not via MAVSDK.** Restoring the vehicle pose is a sim (USD stage) job, and PX4's EKF
and failsafe state must be reset too. Only process A can do both. So this script writes a
request file and waits for process A to handle it; files are used for the same reason as
frame delivery, so no new infrastructure is needed.

⚠️ Restarting PX4 means **process B must reconnect.** Do not call this during flight.
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                "vlmpilot", "src"))
from vlmpilot.controller.isaacsim import pegasus_layout  # noqa: E402

DEFAULT_REQUEST = "/tmp/vlmpilot_pegasus/reset.request"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--slot", type=int, default=None, metavar="N",
                    help="slot number. Switches the request file path to that slot's. "
                         "**Without it the slot 0 file is used**: if you work in slot 1 and "
                         "forget this, you reset another slot's vehicle. An explicit "
                         "`--request-file` takes precedence")
    ap.add_argument("--request-file", default=DEFAULT_REQUEST,
                    help="request file watched by process A "
                         "(must match --reset-file of pegasus_sim.py)")
    ap.add_argument("--no-px4", action="store_true",
                    help="restore pose only, without restarting PX4. May not clear a "
                         "failsafe state")
    ap.add_argument("--pos", default=None,
                    help="new spawn position x,y,z (m). Becomes the reference for later "
                         "resets; for a fixed spawn per task")
    ap.add_argument("--yaw", type=float, default=None,
                    help="new spawn heading (deg)")
    ap.add_argument("--vehicle", action="append", default=None, metavar="I:X,Y,Z[@YAW]",
                    help="per-vehicle start pose. Can be repeated: "
                         "`--vehicle 0:0,0,0.07 --vehicle 1:3,0,0.07@90`. "
                         "Used when a benchmark episode gives coordinates per vehicle. AirSim "
                         "used simSetVehiclePose, but pegasus has no such RPC, so this "
                         "request is the only path. Unspecified vehicles are left as-is. "
                         "**Takes precedence over --pos/--yaw**")
    ap.add_argument("--target-move", choices=["on", "off"], default=None,
                    help="turn benchmark target motion on/off (datagen). In DroneCATS, "
                         "(approaching, searching) use static targets and (tracking, "
                         "search_tracking) moving targets, so producing all four with one A "
                         "requires switching per episode. If unset, A keeps the motion state "
                         "and **only rewinds targets to their start positions**")
    ap.add_argument("--active-targets", default=None, metavar="PRIM[,PRIM...]",
                    help="target prim paths used by this episode (datagen). A batch loads "
                         "70 targets from 28 episodes into the scene at once, but one episode "
                         "uses only 1 target + 0~3 decoys; parking the other movers reduces "
                         "per-step USD writes. Empty string reverts to all. "
                         "⚠️ Changes what is observed: other episodes' targets stand still "
                         "instead of walking, so view and GT differ. Do not use for batches compared with earlier measurements")
    ap.add_argument("--timeout", type=float, default=90.0,
                    help="completion wait limit (s). Takes tens of seconds when PX4 restarts")
    args = ap.parse_args()

    # If a slot is given and the request file is not explicit, switch to that slot's file
    if args.slot is not None and args.request_file == DEFAULT_REQUEST:
        from vlmpilot.controller.isaacsim import pegasus_slots  # noqa: PLC0415
        res = pegasus_slots.resolve(args.slot)
        args.request_file = res["paths"]["reset_file"]
        print(f"[RESET] slot {args.slot} → {args.request_file}"
              + (f" (owner {res['owner']!r})" if res.get("owner") else ""))

    req = args.request_file
    done = req + ".done"

    os.makedirs(os.path.dirname(req) or ".", exist_ok=True)
    for stale in (done,):
        try:
            os.remove(stale)
        except OSError:
            pass

    payload = {"px4": not args.no_px4}
    if args.vehicle:
        # Fail here on bad format: silently falling back to default placement leads to
        # "I specified it, why is it there?" and records false measurement conditions.
        try:
            payload["vehicles"] = pegasus_layout.build_pose_request(args.vehicle)
        except ValueError as e:
            print(f"❌ invalid --vehicle argument: {e}")
            return 2
    if args.pos:
        payload["pos"] = [float(v) for v in args.pos.split(",")]
    if args.yaw is not None:
        payload["yaw_deg"] = args.yaw
    if args.target_move is not None:
        payload["target_move"] = (args.target_move == "on")
    if args.active_targets is not None:
        # Empty string means "all" (None); "not specified" and "revert to all" differ.
        payload["active_targets"] = args.active_targets or None
    if args.vehicle and (args.pos or args.yaw is not None):
        print("⚠️ both --vehicle and --pos/--yaw given - A uses the per-vehicle spec")
    with open(req, "w") as f:
        json.dump(payload, f)
    print(f"reset request: {req}"
          f"{' (keep PX4)' if args.no_px4 else ' (includes PX4 restart)'}"
          + (f" per-vehicle {len(payload['vehicles'])}" if "vehicles" in payload else "")
          + (f" pos={payload['pos']}" if "pos" in payload else "")
          + (f" yaw={payload['yaw_deg']}°" if "yaw_deg" in payload else "")
          + (f" target_move={'ON' if payload['target_move'] else 'OFF'}"
             if "target_move" in payload else "")
          + (f" active_targets={payload['active_targets'] or 'all'}"
             if "active_targets" in payload else ""))

    deadline = time.time() + args.timeout
    while time.time() < deadline:
        if os.path.exists(done):
            with open(done) as f:
                result = f.read().strip()
            os.remove(done)
            if result == "ok":
                print("✅ reset done - vehicles are at spawn, landed")
                if not args.no_px4:
                    print("   PX4 was restarted. Start a new process B")
                return 0
            print(f"❌ process A failed the reset ({result}) - check A's log")
            return 1
        time.sleep(0.3)

    print(f"❌ no response within {args.timeout:.0f}s.")
    print("   Check that process A is running and the --reset-file paths match:")
    print(f"     ps -eo args | grep pegasus_sim")
    return 1


if __name__ == "__main__":
    sys.exit(main())
