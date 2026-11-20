#!/usr/bin/env python3
# DroneCATS
# Copyright (c) 2026-present NAVER Cloud Corp.
# Apache-2.0

"""View, claim, release and clean up slots: one simulator per slot (and port set) on a pod.

Every process on a pod runs as the same user, so `ps` cannot answer "which simulator is
using which port and which file path right now". This tool answers that.

    python3 scripts/isaacsim/pegasus_slots.py list                 # slots in use
    python3 scripts/isaacsim/pegasus_slots.py claim --keep         # reserve a free slot
    python3 scripts/isaacsim/pegasus_slots.py claim --slot 2 --vehicles 3
    python3 scripts/isaacsim/pegasus_slots.py show 2               # all ports/paths of that slot
    python3 scripts/isaacsim/pegasus_slots.py release 2            # free it
    python3 scripts/isaacsim/pegasus_slots.py kill 2               # clean up only that slot's processes
    python3 scripts/isaacsim/pegasus_slots.py prune                # remove dead entries

`kill` replaces `pkill -f mavsdk_server`: pattern matching **kills the other slots' processes too.**
Here the process's `VLMPILOT_SLOT` tag is read so only this slot's processes are chosen.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                "vlmpilot", "src"))
from vlmpilot.controller.isaacsim import pegasus_layout as pl   # noqa: E402
from vlmpilot.controller.isaacsim import pegasus_slots as ps    # noqa: E402


def _fmt(entry: dict) -> str:
    ports = entry.get("ports") or {}
    return (f"  slot {entry['slot']:<3} {'alive' if entry.get('live') else 'dead '} "
            f"owner {str(entry.get('owner'))[:22]:<22} PID {str(entry.get('pid')):<8} "
            f"vehicles {entry.get('vehicle_ids')} "
            f"MAVSDK {ports.get('mavsdk_udp')} gRPC {ports.get('grpc_tcp')}\n"
            f"      root {(entry.get('paths') or {}).get('root')}"
            + (f" · {entry['note']}" if entry.get("note") else "")
            + (f" · started {entry.get('started')}" if entry.get("started") else ""))


def cmd_list(a) -> int:
    rows = ps.list_slots(include_dead=a.all)
    if not rows:
        print("No slots in use. (claim one with `claim`)")
        return 0
    print(f"{len(rows)} slot(s) · registry {ps.REGISTRY}")
    for e in rows:
        print(_fmt(e))
    taken = ps.taken_vehicle_ids()
    free = [i for i in range(pl.MAX_VEHICLES) if i not in taken]
    print(f"free vehicle_id {free} (ceiling {pl.MAX_VEHICLES} - "
          f"upstream PX4 px4-rc.mavlink:6)")
    return 0


def cmd_claim(a) -> int:
    try:
        e = ps.claim(vehicles=a.vehicles, slot=a.slot, owner=a.owner,
                     keep=a.keep, note=a.note)
    except ValueError as exc:
        print(f"❌ {exc}", file=sys.stderr)
        return 1
    e["live"] = True
    print(f"claimed slot {e['slot']}" + (" (keep - held without a process)" if a.keep else ""))
    print(_fmt(e))
    if not a.keep:
        # Death detection is PID-based. When claimed from the CLI the PID is this command,
        # so the entry becomes "dead" and disappears as soon as the command exits.
        print("\n⚠️ This entry **disappears when this command exits**: death detection is "
              "PID-based and that PID is this process.\n"
              "   To reserve a slot ahead of time, pass --keep. If you are starting A right "
              "away, just start it with `--slot N`\n"
              "   (A attaches and records its own PID).")
    print("\nWhen starting A:")
    print(f"  bash scripts/isaacsim/pegasus_sim.sh <GPU> -- --slot {e['slot']} ...")
    print("Values for the B (runner) config:")
    p = e["paths"]
    print(f"  mavsdk_address: \"udpin://0.0.0.0:{e['ports']['mavsdk_udp'][0]}\"")
    print(f"  frame_path:     \"{p['frame_path']}\"")
    print(f"  depth_path:     \"{p['depth_path']}\"")
    print(f"  sim_info_path:  \"{p['info_file']}\"")
    print(f"  gt_live_path:   \"{p['gt_live']}\"")
    return 0


def cmd_show(a) -> int:
    r = ps.resolve(a.slot, a.vehicles)
    print(json.dumps(r, ensure_ascii=False, indent=1))
    if not r["registered"]:
        print("\n(not in registry - resolved with default allocation. `claim` records it)",
              file=sys.stderr)
    return 0


def cmd_release(a) -> int:
    try:
        ok = ps.release(a.slot, force=a.force)
    except ValueError as exc:
        print(f"❌ {exc}", file=sys.stderr)
        return 1
    print(f"slot {a.slot} {'freed' if ok else 'has no entry'}")
    return 0


def cmd_prune(a) -> int:
    gone = ps.prune()
    print(f"removed {len(gone)} dead entr(ies) {gone}" if gone else "nothing to remove")
    return 0


def cmd_kill(a) -> int:
    """Clean up only processes carrying that slot's tag.

    Two reasons not to use `pkill -f <pattern>`: it kills other slots too, and the pattern
    string appears in its own shell's cmdline so it kills itself (observed). Here processes
    are chosen by tag and the ancestor chain is excluded.
    """
    pids = pl.slot_pids(a.slot)
    if not pids:
        print(f"slot {a.slot} has no live processes")
        return 0
    for pid in pids:
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as fh:
                cmd = fh.read().replace(b"\0", b" ").decode("utf-8", "replace")[:90]
        except OSError:
            cmd = "(gone)"
        print(f"  {pid}  {cmd}")
    if not a.yes:
        print(f"\nWill terminate {len(pids)} process(es). Pass --yes to proceed.")
        return 1
    for sig in (signal.SIGTERM, signal.SIGKILL):
        alive = [p for p in pids if pl.slot_of_pid(p) == int(a.slot)]
        if not alive:
            break
        for pid in alive:
            try:
                os.kill(pid, sig)
            except OSError:
                pass
        time.sleep(3.0 if sig == signal.SIGTERM else 1.0)
    left = pl.slot_pids(a.slot)
    print(f"done · remaining {left or 'none'}")
    return 0 if not left else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("list", help="slots in use")
    p.add_argument("--all", action="store_true", help="include dead entries")
    p.set_defaults(fn=cmd_list)

    p = sub.add_parser("claim", help="claim a slot")
    p.add_argument("--slot", type=int, default=None, help="slot number (default: lowest free number)")
    p.add_argument("--vehicles", type=int, default=1)
    p.add_argument("--owner", default=None, help="human-readable name (default $VLMPILOT_OWNER)")
    p.add_argument("--keep", action="store_true",
                   help="hold the slot without a process (when A is not started yet)")
    p.add_argument("--note", default="", help="note (what the work is)")
    p.set_defaults(fn=cmd_claim)

    p = sub.add_parser("show", help="all ports/paths of that slot (JSON)")
    p.add_argument("slot", type=int)
    p.add_argument("--vehicles", type=int, default=1)
    p.set_defaults(fn=cmd_show)

    p = sub.add_parser("release", help="free a slot")
    p.add_argument("slot", type=int)
    p.add_argument("--force", action="store_true", help="free it even if alive")
    p.set_defaults(fn=cmd_release)

    p = sub.add_parser("prune", help="remove dead entries")
    p.set_defaults(fn=cmd_prune)

    p = sub.add_parser("kill", help="clean up only that slot's processes (pkill replacement)")
    p.add_argument("slot", type=int)
    p.add_argument("--yes", action="store_true", help="actually terminate")
    p.set_defaults(fn=cmd_kill)

    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
