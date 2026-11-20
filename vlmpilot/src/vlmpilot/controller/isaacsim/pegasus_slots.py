#!/usr/bin/env python
# DroneCATS
# Copyright (c) 2026-present NAVER Cloud Corp.
# Apache-2.0

"""Slot registry: records who uses which resources on a machine and refuses overlaps.

**Why a file.** Everyone may log in with the same account, so the OS cannot tell you who is
using what: every `ps` owner is the same. The user therefore **records it themselves.**
Without a record two people grab the same port and the same file path; the port collision
is loud, but **files are silently overwritten.**

**Allocation happens here, not by arithmetic.** Computing `slot × vehicles` lets the
`vehicles` value move slot boundaries so they overlap (see the comment on
`pegasus_layout.default_vehicle_base`). So for multi-vehicle use, the registry picks the
`vehicle_id` range by **finding free space**.

Standard library only: both A (Isaac's bundled Python) and B import it.

    from vlmpilot.controller.isaacsim import pegasus_slots as ps
    claim = ps.claim(vehicles=1, owner="alice")   # -> grabs a free slot
    ...
    ps.release(claim["slot"])

Dead entries are cleaned up automatically: when the process dies (PID gone or slot tag
missing) the entry is treated as unused, so a forgotten cleanup does not block the slot.
"""

from __future__ import annotations

import json
import os
import socket
import time

from vlmpilot.controller.isaacsim import pegasus_layout as pl

REGISTRY = pl.SLOT_REGISTRY
MAX_SLOT = 63              # upper bound on slot numbers; separate from the port ceiling (headroom for people)


def _path(slot: int) -> str:
    return os.path.join(REGISTRY, f"{int(slot)}.json")


def _read(slot: int) -> dict | None:
    try:
        with open(_path(slot), encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None


def _pid_alive(pid) -> bool:
    try:
        os.kill(int(pid), 0)
    except (OSError, TypeError, ValueError):
        return False
    return True


def is_live(entry: dict | None) -> bool:
    """Is this entry still in use?

    The criterion is the **PID**: if the process that created the entry is alive, it is in
    use. Entries with `keep=True` are kept regardless of the process; this is an escape
    hatch for reserving a slot ahead of time (A not started yet, or restarting).
    """
    if not entry:
        return False
    if entry.get("keep"):
        return True
    return _pid_alive(entry.get("pid"))


def owner_tag(owner: str | None = None) -> str:
    """Owner label. Since the account is shared, a **human-readable name** is needed.

    Priority: argument -> `$VLMPILOT_OWNER` -> `$STY`/`$TMUX_PANE` (session) -> `host:PID`.
    """
    if owner:
        return str(owner)
    for key in ("VLMPILOT_OWNER", "STY", "TMUX_PANE"):
        v = os.environ.get(key)
        if v:
            return v
    return f"{socket.gethostname()}:{os.getpid()}"


def list_slots(include_dead: bool = False) -> list:
    """List registered slots. Live ones only by default."""
    out = []
    try:
        names = sorted(os.listdir(REGISTRY))
    except OSError:
        return out
    for name in names:
        if not name.endswith(".json"):
            continue
        try:
            slot = int(name[:-5])
        except ValueError:
            continue
        entry = _read(slot)
        if entry is None:
            continue
        entry["slot"] = slot
        entry["live"] = is_live(entry)
        if entry["live"] or include_dead:
            out.append(entry)
    return out


def taken_vehicle_ids() -> dict:
    """`vehicle_id` -> slot number for ids held by live entries."""
    taken = {}
    for e in list_slots():
        for vid in e.get("vehicle_ids") or []:
            taken[int(vid)] = e["slot"]
    return taken


def claim(vehicles: int = 1, slot: int | None = None, owner: str | None = None,
          pid: int | None = None, keep: bool = False,
          max_vehicles: int = pl.MAX_VEHICLES, note: str = "") -> dict:
    """Claim a slot. Uses `slot` if given, else **the lowest free number**.

    The `vehicle_id` range is found **contiguously in free space**, not computed.
    Overlaps are refused with `ValueError`. A silent overlap is exactly the accident this
    tool exists to prevent, so failing is the right behaviour here.
    """
    vehicles = int(vehicles)
    if vehicles < 1:
        raise ValueError(f"vehicles {vehicles} - must be at least 1")
    os.makedirs(REGISTRY, exist_ok=True)

    taken = taken_vehicle_ids()
    live = {e["slot"] for e in list_slots()}

    if slot is None:
        slot = next((s for s in range(MAX_SLOT + 1) if s not in live), None)
        if slot is None:
            raise ValueError(f"no free slot (0~{MAX_SLOT} all in use)")
    else:
        slot = int(slot)
        if slot < 0 or slot > MAX_SLOT:
            raise ValueError(f"slot {slot} - must be in range 0~{MAX_SLOT}")
        if slot in live:
            cur = _read(slot) or {}
            raise ValueError(
                f"slot {slot} is already in use - owner {cur.get('owner')!r} · "
                f"PID {cur.get('pid')} · started {cur.get('started')}. "
                f"Check with `pegasus_slots.py list` and use another number, or, "
                f"if that process is really dead, free it with `release {slot}`")

    # vehicle_id range: **try the slot number first**, and search for free space if taken.
    #
    # Why the slot number first: the default for an unregistered slot is
    # `default_vehicle_base(slot)` = slot number. If allocation always took the lowest free
    # space, ports would differ before and after registration and A and B would see different
    # values (measured: slot 4 got 14540, mismatching its default 14544).
    # The common case (one person, one slot, one vehicle) must be predictable.
    base = None
    prefer = pl.default_vehicle_base(slot)
    candidates = [prefer] + [c for c in range(0, max_vehicles - vehicles + 1)
                             if c != prefer]
    for cand in candidates:
        if cand + vehicles > max_vehicles:
            continue
        if all(i not in taken for i in range(cand, cand + vehicles)):
            base = cand
            break
    if base is None:
        busy = ", ".join(f"{v}→slot{s}" for v, s in sorted(taken.items()))
        raise ValueError(
            f"no {vehicles} contiguous free vehicle_id(s) (ceiling {max_vehicles} · "
            f"in use {busy or 'none'}). To raise the ceiling see the comment on "
            f"pegasus_layout.check_vehicle_range")
    ids = pl.check_vehicle_range(base, vehicles, max_vehicles)

    entry = {
        "slot": slot,
        "owner": owner_tag(owner),
        "pid": int(os.getpid() if pid is None else pid),
        "vehicles": vehicles,
        "vehicle_ids": ids,
        "paths": pl.slot_paths(slot),
        "ports": pl.slot_ports(slot, vehicles, base=base, max_vehicles=max_vehicles),
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "keep": bool(keep),
        "note": str(note),
    }
    tmp = _path(slot) + f".tmp{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(entry, fh, ensure_ascii=False, indent=1)
    os.replace(tmp, _path(slot))     # atomic replace: never leave a half-written entry
    return entry


def release(slot: int, force: bool = False) -> bool:
    """Free a slot. Another owner's live entry is refused without `force`."""
    entry = _read(slot)
    if entry is None:
        return False
    if is_live(entry) and not force:
        mine = entry.get("pid") == os.getpid() or pl.slot_of_pid(os.getpid()) == int(slot)
        if not mine:
            raise ValueError(
                f"slot {slot} is still alive (owner {entry.get('owner')!r} · "
                f"PID {entry.get('pid')}). Pass force to really free it")
    try:
        os.remove(_path(slot))
    except OSError:
        return False
    return True


def prune() -> list:
    """Remove dead entries and return the freed slot numbers."""
    gone = []
    for e in list_slots(include_dead=True):
        if not e.get("live"):
            try:
                os.remove(_path(e["slot"]))
                gone.append(e["slot"])
            except OSError:
                pass
    return gone


def resolve(slot: int, vehicles: int = 1) -> dict:
    """All resources of a slot: the registry allocation if present, else defaults.

    This is the single source that makes A and B **see the same values**. It must work
    with only `--slot` and no registration (a solo user may skip the registry).
    """
    entry = _read(slot)
    if entry and is_live(entry):
        return {"slot": int(slot), "registered": True,
                "vehicle_ids": entry["vehicle_ids"], "paths": entry["paths"],
                "ports": entry["ports"], "owner": entry.get("owner")}
    base = pl.default_vehicle_base(slot)
    return {"slot": int(slot), "registered": False,
            "vehicle_ids": pl.check_vehicle_range(base, vehicles),
            "paths": pl.slot_paths(slot),
            "ports": pl.slot_ports(slot, vehicles, base=base), "owner": None}

def attach(slot: int, vehicles: int = 1, owner: str | None = None,
           note: str = "") -> dict:
    """Process A **takes over** a slot: claims it if absent, refreshes the PID if it is ours.

    Why separate from `claim`: a person can reserve a slot with `pegasus_slots.py claim --keep`,
    and the PID recorded then is that short-lived CLI process. Once A starts and becomes the
    real owner, **the PID must be replaced with its own** for death detection to work.

    If someone else is alive and using it, refuse; that is the point of this tool.

    ⚠️ **Environment-variable tags cannot be planted here.** Changing `os.environ` does not
    change the process's `/proc/self/environ`, which keeps its exec-time value. So A's own
    ownership is determined by **the registry PID**, and A's children (PX4 etc.) get the tag
    by inheriting `os.environ`. If the wrapper (`scripts/isaacsim/pegasus_sim.sh`) exports it before
    exec, A itself has the tag too.
    """
    slot = int(slot)
    entry = _read(slot)
    if entry and is_live(entry):
        mine = entry.get("pid") == os.getpid()
        by_tag = pl.slot_of_pid(entry.get("pid") or -1) == slot
        if not (mine or entry.get("keep") or by_tag):
            raise ValueError(
                f"slot {slot} is already used by a live process - "
                f"owner {entry.get('owner')!r} · PID {entry.get('pid')} · "
                f"started {entry.get('started')}. Use another slot, or, if that process is "
                f"really dead, free it with `pegasus_slots.py release {slot} --force`")
        entry["pid"] = os.getpid()
        entry["keep"] = False          # a real owner exists now, so drop the reservation
        entry["owner"] = owner_tag(owner) if owner else entry.get("owner")
        if note:
            entry["note"] = str(note)
        os.makedirs(REGISTRY, exist_ok=True)
        tmp = _path(slot) + f".tmp{os.getpid()}"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(entry, fh, ensure_ascii=False, indent=1)
        os.replace(tmp, _path(slot))
        return entry
    return claim(vehicles=vehicles, slot=slot, owner=owner, note=note)
