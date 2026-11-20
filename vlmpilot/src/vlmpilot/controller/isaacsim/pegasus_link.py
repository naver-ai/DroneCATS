# DroneCATS
# Copyright (c) 2026-present NAVER Cloud Corp.
# Apache-2.0

"""Bridge between MAVSDK (async) and the controller (threaded, blocking) for PX4 SITL.

`flight_controller.py` runs commands on a plain queue-consuming thread and blocks
until each finishes (the AirSim `*Async().join()` convention). The MAVSDK Python API
is entirely `async def`. This file absorbs that gap in one place: it runs an asyncio
loop on a dedicated thread, and public methods block via
`run_coroutine_threadsafe(...).result()`.

So the pegasus controller can execute the backend-neutral commands from
`step_planner.py` (`turn_in_place` / `body_velocity` / `hold`) as-is; the only
difference is offboard setpoints instead of AirSim RPCs.

Differences from AirSim handled here:

1. **PX4 offboard needs a continuous setpoint stream.** Without a new setpoint within
   500 ms it drops out of offboard. `_hold()` resends at 20 Hz. AirSim kept a command
   alive for the given duration on its own.
2. **Fire-and-return in continuous mode (`continuous_control`).** In AirSim, skipping
   `join()` still let the sim finish the command. PX4 holds the last setpoint forever,
   so something must reset it to zero after the duration. Non-blocking calls schedule
   the hold coroutine on the loop and return immediately.

3. **Duration is measured in sim time.** AirSim handled duration in its own clock.
   Measuring with the wall clock (`asyncio.sleep`) makes commands proportionally short
   when the sim runs slower than real time: at RTF 0.5 both turns and forward moves
   came out half as long, and even at an average RTF of 1.0 momentary dips caused
   occasional errors. PX4's `odometry.time_usec` is sim time, so we use that.

The frame matches AirSim's body frame: forward/right/**down+**. The NED vertical
velocity `step_planner.py` emits works without changes.
"""

import asyncio
import json
import math
import os
import socket
import threading
import time

from mavsdk import System
from mavsdk.offboard import OffboardError, VelocityBodyYawspeed

# Offboard setpoint resend rate. Must be well under PX4's offboard timeout (500 ms).
_SETPOINT_HZ = 20.0
_ZERO = (0.0, 0.0, 0.0, 0.0)
# Wall-clock cap multiplier for sim-time holds, so a stalled sim does not hang us.
#
# Raised 10 -> 20. The multiplier sets the **lowest RTF we survive** (cap = duration x
# multiplier). Measured: Brownstone + PX4 runs at RTF 0.093-0.134, so a 2.5 s hold needs
# 25-27 s of wall time. A multiplier of 10 allows only 30 s: one dip below RTF 0.083
# truncates the hold and the step falls short. Since that only logs a warning, it
# becomes a **silently shortened step**. 20 survives down to RTF 0.045.
_WALL_GUARD = 20.0

# Fingerprints of a vanished mavsdk_server. gRPC only says "stream removed", so we
# match on strings; this must not catch other errors (e.g. permission denied).
_SERVER_DEAD = ("Stream removed", "Socket closed", "Connection reset by peer",
                "failed to connect to all addresses", "Connection refused")


def _server_died(exc):
    """Does this exception mean the server died?"""
    t = str(exc)
    return any(p in t for p in _SERVER_DEAD)

# Escape hatch to proceed even if ports are taken. Same idea as `FORCE=1` in
# `pegasus_sim.sh`: refuse by default, let a human decide to override.
ENV_FORCE_PORTS = "PEGASUS_FORCE_PORTS"


def _udp_port_of(address):
    """`udpin://0.0.0.0:14540` -> 14540. None if unparsable (check is skipped)."""
    try:
        return int(str(address).rsplit(":", 1)[1])
    except (IndexError, ValueError):
        return None


def _port_taken(port, kind):
    """Is `port` already taken? False if undecidable, so the check never blocks a flight.

    TCP is tested with `SO_REUSEADDR` on; otherwise a TIME_WAIT left by the previous run
    counts as "taken" and a normal run is refused. UDP cannot bind twice without
    `SO_REUSEPORT`, so a plain bind test is accurate.
    """
    family = socket.SOCK_STREAM if kind == "tcp" else socket.SOCK_DGRAM
    try:
        with socket.socket(socket.AF_INET, family) as s:
            if kind == "tcp":
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(("0.0.0.0", int(port)))
        return False
    except OSError:
        return True
    except Exception:            # noqa: BLE001 — undecidable: let it pass
        return False


def _mavsdk_servers_holding(port):
    """PIDs of `mavsdk_server` processes whose command line contains `port`.

    Diagnostic only: shows a human who holds the port. Empty if /proc is missing or
    unreadable; the refusal message is still emitted.
    """
    found = []
    try:
        for name in os.listdir("/proc"):
            if not name.isdigit():
                continue
            try:
                with open(f"/proc/{name}/cmdline", "rb") as fh:
                    cmd = fh.read().replace(b"\0", b" ").decode("utf-8", "replace")
            except OSError:
                continue
            if "mavsdk_server" in cmd and str(port) in cmd:
                found.append((int(name), cmd.strip()))
    except OSError:
        pass
    return found


def _reap(pids):
    for pid in pids:
        try:
            os.kill(pid, 15)
        except OSError as exc:
            print(f"[LINK] failed to kill PID {pid}: {exc}")
    if pids:
        time.sleep(3.0)


def check_ports_free(grpc_port, address):
    """Check that the gRPC and UDP ports are free **before** connecting; refuse if not.

    Without this check we silently **attach to someone else's mavsdk_server**. The new
    server spawned by `System(port=...)` fails to bind, prints
    `bind error: Address in use` to stderr and exits, but that is not an exception, so
    Python never learns of it. The client then attaches to the pre-existing server,
    reports a successful connection, and collapses mid-flight with
    `Connection reset by peer` when that server dies.

    One whole run was ruined this way in real flight: a run proceeded despite the
    `bind error`, and the run before it had every takeoff command fail, flew 5 steps on a
    frozen frame, and was recorded as `outcome=arrived`. So **refusal is the default**.

    With `PEGASUS_FORCE_PORTS=1`, the `mavsdk_server` holding the port is cleaned up and
    we proceed. It is not the default so that we never kill other people's processes;
    a teammate's run may be using the same machine (same reasoning as the duplicate-A
    guard in `pegasus_sim.sh`).
    """
    busy = []
    if _port_taken(grpc_port, "tcp"):
        busy.append(("gRPC/TCP", int(grpc_port)))
    udp = _udp_port_of(address)
    if udp is not None and _port_taken(udp, "udp"):
        busy.append(("MAVLink/UDP", udp))
    if not busy:
        return

    lines = []
    holders = []
    for kind, port in busy:
        procs = _mavsdk_servers_holding(port)
        holders += [pid for pid, _ in procs]
        who = ", ".join(f"PID {pid}" for pid, _ in procs) or "unknown (not mavsdk_server)"
        lines.append(f"  {kind} {port} — {who}")
    detail = "\n".join(lines)

    if os.environ.get(ENV_FORCE_PORTS) == "1":
        # **Only kill processes of my own slot.** The account is shared, so without this
        # we would kill a teammate's mavsdk_server and crash their flight mid-air.
        # Untagged processes count as slot 0 (a solo user not using slots).
        from vlmpilot.controller.isaacsim import pegasus_layout as _pl  # noqa: PLC0415 — avoid circular import
        my_slot = int(os.environ.get(_pl.SLOT_ENV) or 0)
        mine, others = [], []
        for pid in sorted(set(holders)):
            (mine if (_pl.slot_of_pid(pid) or 0) == my_slot else others).append(pid)
        if others:
            raise RuntimeError(
                f"MAVSDK ports are held by processes of **another slot**; "
                f"not killing them even with {ENV_FORCE_PORTS}=1.\n{detail}\n"
                f"  PIDs from other slots: {others} (my slot {my_slot})\n"
                f"  · To work in another slot, set the config ports to that slot's values\n"
                f"    (`vlmpilot.controller.isaacsim.pegasus_slots.resolve(<slot>)` gives them)\n"
                f"  · If someone else's process really must go, decide that manually")
        print(f"[LINK] {ENV_FORCE_PORTS}=1 — cleaning up only slot {my_slot} servers "
              f"and proceeding:\n{detail}")
        _reap(mine)
        return

    raise RuntimeError(
        "MAVSDK ports are already taken. Proceeding would attach to someone else's\n"
        "server: the connection looks fine, then collapses mid-flight.\n"
        f"{detail}\n"
        "  · Leftovers from a previous run:  python3 scripts/isaacsim/pegasus_slots.py kill <slot>\n"
        "    (`pkill -f mavsdk_server` **also kills other slots**)\n"
        f"  · Checked, and want to clean up and proceed:  {ENV_FORCE_PORTS}=1 <command>"
    )


class PegasusLink:
    """Blocking facade over MAVSDK for PX4 SITL.

    Every public method blocks the calling thread and returns the result (except
    movement commands passed `nonblocking=True`).
    """

    def __init__(self, address="udpin://0.0.0.0:14540", timeout=30.0,
                 takeoff_altitude=2.5, sim_clock=True, grpc_port=50051,
                 server_address=None):
        self.address = address
        # MAVSDK gRPC port. `System()` starts `mavsdk_server` on 50051 by default;
        # **with two links, the second attaches to the first server and sees the same
        # vehicle**. In multi-vehicle runs two links reported the same position for this
        # reason. Each vehicle needs its own port. Single-vehicle default is unchanged.
        self.grpc_port = int(grpc_port)
        # **Diagnostics**: attach to an already-running mavsdk_server (with an address,
        # MAVSDK does not spawn one). Default None: MAVSDK spawns it as before.
        #
        # Why: the server silently vanishes during `arm` (zero log lines). When MAVSDK
        # spawns it we cannot touch the process: no stderr, core, or debugger. When we
        # spawn it ourselves we can run it under gdb and **get a stack trace**.
        self.server_address = str(server_address) if server_address else None
        self.timeout = float(timeout)
        self.takeoff_altitude = float(takeoff_altitude)
        # Measure duration in sim time (True) or wall time (False).
        # On real hardware the two are equal, so either is harmless.
        self._sim_clock = bool(sim_clock)

        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._run_loop, name="mavsdk-loop", daemon=True
        )
        self._thread.start()

        self._drone = None
        self._offboard_on = False
        # In-flight non-blocking hold. Cancelled first when a new command arrives.
        self._pending = None
        self._pending_lock = threading.Lock()
        self._closed = False

    # ------------------------------------------------------------------ internals

    def _run_loop(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _call(self, coro, timeout=None):
        """Submit a coroutine to the loop and block until it finishes."""
        fut = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return fut.result(timeout if timeout is not None else self.timeout)

    def _cancel_pending(self):
        """Cancel the in-flight non-blocking hold (the next command overwrites the setpoint)."""
        with self._pending_lock:
            fut, self._pending = self._pending, None
        if fut is not None and not fut.done():
            fut.cancel()

    async def _hold(self, vx, vy, vz, yawspeed, duration):
        """Hold the setpoint for `duration` (**sim time**), then reset it to zero.

        When cancelled it does not reset to zero: the next command overwrites it right
        away, and writing zero here would insert needless braking into continuous flight.
        """
        sp = VelocityBodyYawspeed(vx, vy, vz, yawspeed)
        try:
            if self._sim_clock:
                await self._hold_sim_time(sp, duration)
            else:
                await self._hold_wall_time(sp, duration)
        except asyncio.CancelledError:
            raise
        await self._drone.offboard.set_velocity_body(VelocityBodyYawspeed(*_ZERO))

    async def _hold_sim_time(self, sp, duration):
        """Measure `duration` with the sim clock from odometry.

        Odometry arrives at ~30 Hz and `time_usec` is sim time. Setpoint resends are also
        timed in sim time while iterating the stream; PX4's offboard timeout uses PX4's own
        clock, so sending at a steady sim-time rate is actually correct.

        A wall-clock cap is kept as well, because a stalled sim would never end this
        loop. In that case we log a warning and return (hanging silently is the worst).
        """
        period = 1.0 / _SETPOINT_HZ
        wall_deadline = time.monotonic() + duration * _WALL_GUARD + 5.0
        await self._drone.offboard.set_velocity_body(sp)

        t0 = last_send = None
        async for od in self._drone.telemetry.odometry():
            t = od.time_usec / 1e6
            if t0 is None:
                t0 = last_send = t
            if t - t0 >= duration:
                return
            if t - last_send >= period:
                await self._drone.offboard.set_velocity_body(sp)
                last_send = t
            if time.monotonic() > wall_deadline:
                got = (t - t0) if t0 is not None else 0.0
                print(f"[LINK] warning: waited for {duration:.2f}s of sim time but only "
                      f"{got:.2f}s passed. Sim stalled? Stopping at wall-clock cap")
                return

    async def _hold_wall_time(self, sp, duration):
        """Wall-clock fallback. Runs proportionally short if the sim is not real time."""
        period = 1.0 / _SETPOINT_HZ
        n = max(1, int(round(duration * _SETPOINT_HZ)))
        for _ in range(n):
            await self._drone.offboard.set_velocity_body(sp)
            await asyncio.sleep(period)

    # ------------------------------------------------------------- connect / close

    def connect(self):
        """MAVSDK connect + wait for position estimate + **state cleanup**.

        Why cleanup: if a previous session died, PX4 stays in OFFBOARD mode. In that
        state `action.arm()` is rejected with COMMAND_DENIED even when landed and
        disarmed. Process B gets restarted often during development, so we drop
        offboard once on connect and start from a known state.
        """
        try:
            self._connect_steps()
        except BaseException:
            # Connect failures are common (vehicle in failsafe, port clash, ...). If this
            # object is just dropped, mavsdk_server is orphaned holding 14540 and **the
            # next run cannot even connect**. Always clean up on the failure path.
            # BaseException because Ctrl+C while waiting leaks the same way.
            self.close()
            raise

    def _connect_steps(self):
        # Check ports first: once `System()` exists there is no way to tell
        # "attached, but to someone else's server".
        if self.server_address:
            # Someone else started the server, so the port must be taken, not free.
            # Skip the free-port check.
            print(f"[LINK] attaching to running mavsdk_server "
                  f"({self.server_address}:{self.grpc_port})")
            self._drone = System(mavsdk_server_address=self.server_address,
                                 port=self.grpc_port)
        else:
            check_ports_free(self.grpc_port, self.address)
            self._drone = System(port=self.grpc_port)

        # (1) Bring up the link only
        self._call(self._link_up(), timeout=max(60.0, self.timeout))

        # (1b) **Catch a connection window and speed up the heartbeat inside it.**
        #
        # Measured: the abort is not specific to `arm`. **Any RPC** arriving in the
        # "no vehicle" window kills the server (one episode died on
        # `set_mavlink_timeout`, the call after `offboard.stop()`). The window is only
        # 3 s and the connect sequence is longer, so **gating each call is inherently
        # insufficient**.
        #
        # Instead, send `SET_MESSAGE_INTERVAL` (511) once inside the window to raise
        # PX4's HEARTBEAT rate. PX4 runs lockstep, so 1 Hz is 9 s of wall time; at 5 Hz
        # it is 1.8 s even at RTF 0.11, inside MAVSDK's 3 s timeout. After that the
        # window never closes, so every later RPC is safe.
        self._wait_connected()
        self._speed_up_heartbeat()

        # (2) Clean up the previous session's state **before** waiting for health.
        # Order matters: if PX4 is still in OFFBOARD, `is_armable` is False, and waiting
        # for health first never finishes. It actually hung that way.
        #
        # Note: **the timeout is wall time; give more when the sim is slow.**
        # Measured: Brownstone + PX4 runs at RTF ~0.11, so 5 s of wall time is 0.55 s of
        # sim time. PX4 (lockstep) cannot answer in time and MAVSDK **forgets** the
        # command after its timeout; PX4's late ACK then shows up as
        # `Received ack for not-existing command`.
        #
        # Correction: this 5 s timeout was once blamed for link failures (35%); **that was
        # wrong**. Failures kept the same rate after raising it to 60 s. The stray ACK
        # already appears **before** this call, while MAVSDK brings up the link.
        #
        # Nor is that ACK a fingerprint of the crash: counting only the pre-`arm` part per
        # attempt gave 38% for completed vs 43% for crashed attempts, **no correlation**.
        # At low RTF the line is simply common; the cause of link failures is still
        # unknown.
        #
        # The change stays anyway: 5 s of wall time at RTF 0.11 is too short in itself,
        # and `except: pass` used to swallow the timeout so that "was not in offboard" and
        # "PX4 could not keep up" were indistinguishable.
        #
        # Now we still swallow it but **log it**. If not in offboard, PX4 rejects
        # immediately, so this does not use up the timeout.
        try:
            self._call(self._drone.offboard.stop(),
                       timeout=max(60.0, self.timeout))
            print("[LINK] cleared offboard left over from previous session")
        except Exception as exc:  # noqa: BLE001
            # Not being in offboard is normal. **A timeout is different**: the late ACK
            # then kills the next command, so it must be visible.
            print(f"[LINK] offboard cleanup skipped ({type(exc).__name__})")
        self._offboard_on = False

        # (2b) **Raise the MAVLink send timeout.** The default 0.5 s is wall time, i.e.
        # 0.055 s of sim time at RTF 0.11; PX4 cannot answer that fast. This is the source
        # of the common `Received ack for not-existing command` (a late ACK arriving after
        # MAVSDK gave up). That line was unrelated to crashes (38% vs 43%), but the short
        # timeout was wrong in itself.
        try:
            self._call(self._drone.core.set_mavlink_timeout(10.0), timeout=10.0)
        except Exception as exc:  # noqa: BLE001
            print(f"[LINK] MAVLink timeout setting skipped ({type(exc).__name__})")

        # (3) Then wait for position estimate and armability. Wait generously: EKF
        # convergence takes seconds of sim time, which stretches in wall time at low RTF.
        try:
            self._call(self._wait_healthy(), timeout=max(180.0, self.timeout))
        except TimeoutError:
            # A bare TimeoutError says nothing about why. In practice the vehicle once
            # tipped over, PX4 entered failsafe and is_armable stuck at False
            # (PX4 log: Attitude failure (pitch) / Compass needs calibration).
            # Print that diagnosis right here.
            self._report_health()
            raise
        print(f"[LINK] MAVSDK connected ({self.address})")

    def _report_health(self):
        """On connect-wait failure, print which flag is blocking."""
        try:
            h = self._call(self._first(self._drone.telemetry.health()), timeout=10.0)
            print("[LINK] connect wait failed — PX4 health:")
            print(f"       local_position_ok={h.is_local_position_ok} "
                  f"armable={h.is_armable} home_ok={h.is_home_position_ok}")
            if not h.is_armable:
                print("[LINK] armable=False. The vehicle may have tipped over or be in "
                      "failsafe; check process A's PX4 log for 'Preflight Fail' / 'Failsafe'.")
                print("[LINK] reset vehicle: python3 scripts/isaacsim/pegasus_reset.py "
                      "(restarts PX4; relaunch this process afterwards)")
        except Exception as e:
            print(f"[LINK] health query also failed: {e}")

    async def _link_up(self):
        """Bring up the MAVLink link only (health is not checked yet)."""
        await self._drone.connect(system_address=self.address)
        async for state in self._drone.core.connection_state():
            if state.is_connected:
                return

    async def _wait_healthy(self):
        """Wait until the position estimate converges and the vehicle is armable.

        Do not include `is_home_position_ok`. Home is set **at the moment of arming**, so
        it stays False forever on a fresh PX4, yet it is True on a PX4 that was armed
        before; that shows up as "worked yesterday, hangs on connect today". Offboard
        velocity control needs local position, and is_armable answers armability.
        """
        async for health in self._drone.telemetry.health():
            if health.is_local_position_ok and health.is_armable:
                return

    def close(self):
        """Shut down the loop and **mavsdk_server**. Vehicle state is untouched (see stop()).

        mavsdk_server must be killed. This separate process spawned by `System()` holds
        `udpin://...:14540`; if it survives, the next run fails to connect with
        `bind error: Address in use`. `System.__del__` at Python exit fails because the
        interpreter is shutting down (`ImportError: sys.meta_path is None`), so we clean
        up explicitly here. Process B is restarted often during development, so this
        matters every time.
        """
        if self._closed:
            return
        self._closed = True
        self._cancel_pending()
        try:
            self._loop.call_soon_threadsafe(self._loop.stop)
        except RuntimeError:
            pass
        self._thread.join(timeout=5.0)
        drone, self._drone = self._drone, None
        if drone is not None:
            proc = getattr(drone, "_server_process", None)
            try:
                drone._stop_mavsdk_server()
            except Exception as e:
                print(f"[LINK] mavsdk_server cleanup failed (ignored): {e}")
            # kill() only sends a signal. **Wait until the socket is actually released**:
            # otherwise the next process fails to bind UDP 14540, or worse, the dying old
            # server answers the connect probe, the connection looks successful, and the
            # next call blows up with 50051 Connection refused.
            # (Reproduced when running a task set back to back.)
            if proc is not None and hasattr(proc, "wait"):
                try:
                    proc.wait(timeout=5.0)
                except Exception:
                    print("[LINK] mavsdk_server did not exit within 5 s")

    # ------------------------------------------------------------ arm / takeoff / land

    def is_armed(self):
        return bool(self._call(self._first(self._drone.telemetry.armed())))

    def _speed_up_heartbeat(self, hz=5.0):
        """Raise PX4's HEARTBEAT rate. **This is the root fix for the abort.**

        We cannot change MAVSDK's heartbeat timeout (3 s): it is not in the CLI or env
        vars, and C++ `set_heartbeat_timeout_s` is not exposed over gRPC. What we can
        change is **PX4's sending side**: shortening the HEARTBEAT (message id 0) interval
        via `MAV_CMD_SET_MESSAGE_INTERVAL` (511) keeps the wall-clock interval inside the
        timeout even with a lockstep-slowed clock.

            RTF 0.11 · 1 Hz  -> 9.0 s wall  -> exceeds 3 s timeout, "gone" 2/3 of the time
            RTF 0.11 · 5 Hz  -> 1.8 s wall  -> safe

        Failure does not block flight; we fall back to aiming for the connection window.
        """
        try:
            from mavsdk.mavlink_direct import MavlinkMessage

            interval_us = int(1e6 / float(hz))
            fields = {
                "target_system": 1, "target_component": 1,
                "command": 511, "confirmation": 0,
                "param1": 0.0,                    # HEARTBEAT message id
                "param2": float(interval_us),     # interval (µs)
                "param3": 0.0, "param4": 0.0, "param5": 0.0,
                "param6": 0.0, "param7": 0.0,
            }
            msg = MavlinkMessage("COMMAND_LONG", 0, 0, 1, 1, json.dumps(fields))
            self._call(self._drone.mavlink_direct.send_message(msg), timeout=15.0)
            print(f"[LINK] raised HEARTBEAT rate to {hz:g} Hz "
                  f"(lockstep clock fix, see `_speed_up_heartbeat`)")
            return True
        except Exception as exc:  # noqa: BLE001 — flight continues even on failure
            print(f"[LINK] failed to raise HEARTBEAT rate ({type(exc).__name__}); "
                  f"falling back to aiming for the connection window")
            return False

    def _wait_connected(self, timeout_s=30.0):
        """Wait for a window in which the vehicle appears connected.

        **Without this, `arm` kills the server.** Measured gdb stack:

            abort ()
            mavsdk::mavsdk_server::LazyPlugin<mavsdk::Action>::maybe_plugin()
            ActionServiceImpl<Action, LazyPlugin<Action>>::Arm(...)

        If the `arm` RPC arrives while there is "no vehicle", the server aborts instead of
        returning an error (both MAVSDK v3.17.2 and v3.17.4). Python sees
        `AioRpcError: Stream removed` and the episode comes out empty; 59% of attempts
        died that way.

        Why "no vehicle": PX4 is **lockstep** and sends heartbeats at 1 Hz on its own
        clock. At RTF 0.11 that is a **9 s** wall-clock interval, while MAVSDK's heartbeat
        timeout is **3 s**. So an episode cycles "lost -> rediscovered" 36 times and spends
        about 2/3 of the time with no vehicle (the 59% failure rate matches 6/9 ~ 67%).

        We cannot change that timeout: not in the CLI or env vars, and C++
        `set_heartbeat_timeout_s` is not exposed over gRPC (`SetMavlinkTimeout` calls
        `set_timeout_s`, which is something else). So **the caller picks the window**.

        `connection_state()` emits on state changes. During a gap we first get `False`,
        then `True` the moment the next heartbeat arrives; right after that is the safest
        point (the window stays open for about 3 s).
        """
        # **"Connected" alone is not enough.** Measured: the gate logged a connection and
        # the server still died, because "connected" then meant "a heartbeat within the
        # last 3 s" and the window closed before `arm` reached the server. So we wait for
        # a **False -> True transition**: a heartbeat just arrived, so almost the full
        # window (~3 s) remains. If it starts as True, we wait for one drop and reconnect;
        # that costs up to ~12 s more, which is cheap for ~400 s episodes.
        async def _w():
            saw_down = False
            async for st in self._drone.core.connection_state():
                up = bool(getattr(st, "is_connected", False))
                if not up:
                    saw_down = True
                elif saw_down:
                    return True          # fresh heartbeat
            return False

        try:
            return bool(self._call(_w(), timeout=timeout_s))
        except Exception as exc:  # noqa: BLE001 — failing to wait does not block arm
            print(f"[LINK] connection window wait failed ({type(exc).__name__}); arming anyway")
            return False

    def _rebuild(self):
        """Drop the dead server and attach again.

        When `arm` aborts the server, that `System` object is unusable: the process is
        gone. Picking the window does not fully remove the race (it is an upstream bug),
        so on a crash we **recover instead of discarding the episode**. The dead server
        has released the port.
        """
        old, self._drone = self._drone, None
        if old is not None:
            try:
                proc = getattr(old, "_server_process", None)
                if proc is not None:
                    proc.kill()
            except Exception:  # noqa: BLE001
                pass
        if self.server_address:
            self._drone = System(mavsdk_server_address=self.server_address,
                                 port=self.grpc_port)
        else:
            self._drone = System(port=self.grpc_port)
        self._call(self._link_up(), timeout=max(60.0, self.timeout))
        try:
            self._call(self._drone.core.set_mavlink_timeout(10.0), timeout=10.0)
        except Exception:  # noqa: BLE001
            pass
        self._offboard_on = False

    def arm(self, tries=3):
        """Arm inside a connection window; if the server dies, reconnect and retry."""
        for k in range(1, int(tries) + 1):
            if self._wait_connected():
                print(f"[LINK] connected — arm (attempt {k}/{tries})")
            try:
                self._call(self._drone.action.arm())
                return
            except Exception as exc:  # noqa: BLE001
                dead = _server_died(exc)
                if not dead or k >= int(tries):
                    raise
                print(f"[LINK] server died during arm ({type(exc).__name__}); "
                      f"reconnecting and retrying ({k}/{tries})")
                self._rebuild()

    def disarm(self):
        self._call(self._drone.action.disarm())

    def takeoff(self):
        """Enter offboard after takeoff has **finished**. Counterpart of AirSim's takeoffAsync().join().

        Waiting for `in_air` is not enough. `in_air` turns True a few cm off the ground;
        switching to offboard then makes the zero-velocity setpoint from
        `start_offboard()` override the climb, and PX4 ends like this:

            INFO [commander] Takeoff detected
            INFO [commander] Landing detected      <- immediately
            INFO [commander] Disarmed by landing

        AirSim's `takeoffAsync().join()` waited until takeoff finished, so to match that
        we wait **until the climb stops**.

        We do not use an altitude threshold (e.g. 80% of target): PX4 settles below the
        commanded altitude. In practice 2.5 m was commanded and it went to HOLD at
        1.74 m. The cause was not confirmed (likely EKF origin vs home altitude offset);
        "climb stopped" does not depend on that number. (Hit twice in practice.)
        """
        # **Check arm state here.** PX4 disarms by itself if it does not take off within
        # `COM_DISARM_PRFLT` (default 10 s); the log says "Disarmed by auto preflight
        # disarming". If the controller arms in its constructor and a human takes more
        # than 10 s to type a command, it disarms in between and takeoff silently fails.
        # (AirSim had no such timeout, so arming in the constructor was enough.)
        if not self.is_armed():
            print("[LINK] vehicle is disarmed (PX4 auto-disarm); arming again")
            self.arm()
        self._call(self._drone.action.set_takeoff_altitude(self.takeoff_altitude))
        self._call(self._drone.action.takeoff())
        # Takeoff also takes seconds of sim time; give a generous wall cap for low RTF
        alt = self._call(self._wait_climb_settled(), timeout=max(300.0, self.timeout))
        print(f"[LINK] takeoff done — altitude {alt:.2f} m "
              f"(commanded {self.takeoff_altitude:.2f} m)")
        self.start_offboard()

    async def _wait_climb_settled(self, min_alt_m=0.5, still_eps=0.15, still_n=5):
        """Wait until airborne and vertical speed has settled. Returns reached altitude.

        min_alt_m: below this we are still on the ground (needed since vz~0 on the ground)
        still_eps: |vz| below this counts as "stopped"
        still_n:   consecutive stopped samples required (guards against telemetry noise)
        """
        still = 0
        async for pv in self._drone.telemetry.position_velocity_ned():
            alt = -pv.position.down_m
            if alt >= min_alt_m and abs(pv.velocity.down_m_s) < still_eps:
                still += 1
                if still >= still_n:
                    return alt
            else:
                still = 0

    def is_airborne(self):
        """Is the vehicle airborne? Used for pre-cleanup when running checks repeatedly."""
        return bool(self._call(self._first(self._drone.telemetry.in_air())))

    def land(self, wait=False, wait_timeout=None):
        """Land. With `wait=True`, wait until `in_air` goes False.

        `wait_timeout` caps that wait (seconds). The default 300 s gets fully used when
        the vehicle is stuck on a ceiling or shelf and cannot descend; in practice it did
        not finish for over 4 minutes at 8.9 m. The caller can set the cap.
        """
        self.stop_offboard()
        self._call(self._drone.action.land())
        if wait:
            bound = wait_timeout if wait_timeout is not None else max(300.0, self.timeout)
            self._call(self._wait_landed(), timeout=max(float(bound), self.timeout))

    async def _wait_landed(self):
        async for in_air in self._drone.telemetry.in_air():
            if not in_air:
                return

    # -------------------------------------------------------------- offboard

    def start_offboard(self):
        """Start offboard. PX4 accepts it only if a setpoint was sent beforehand."""
        if self._offboard_on:
            return
        self._call(self._start_offboard())
        self._offboard_on = True

    async def _start_offboard(self):
        await self._drone.offboard.set_velocity_body(VelocityBodyYawspeed(*_ZERO))
        try:
            await self._drone.offboard.start()
        except OffboardError as e:
            raise RuntimeError(f"failed to start offboard: {e._result.result}") from e

    def stop_offboard(self):
        if not self._offboard_on:
            return
        self._cancel_pending()
        try:
            self._call(self._drone.offboard.stop())
        except Exception as e:
            print(f"[LINK] offboard stop failed (ignored): {e}")
        self._offboard_on = False

    # ------------------------------------------------------------------ control

    def hover(self):
        """Hold position. Counterpart of AirSim's hoverAsync().join()."""
        self._cancel_pending()
        self._call(
            self._drone.offboard.set_velocity_body(VelocityBodyYawspeed(*_ZERO))
        )

    def cancel(self):
        """Cancel the in-flight command. Counterpart of AirSim's cancelLastTask()."""
        self._cancel_pending()

    def rotate_yaw_rate(self, rate_deg_s, duration_s):
        """Yaw in place: zero linear velocity + yawspeed. Counterpart of AirSim's rotateByYawRateAsync."""
        self.move_velocity_body(0.0, 0.0, 0.0, duration_s, yawspeed=rate_deg_s)

    def move_velocity_body(self, vx, vy, vz, duration_s, yawspeed=0.0,
                           nonblocking=False):
        """Hold a body-frame velocity for `duration`. vz is **positive down**.

        With nonblocking=True, schedule the hold on the loop and return immediately
        (`continuous_control`). The next command cancels that hold.
        """
        self._cancel_pending()
        coro = self._hold(vx, vy, vz, yawspeed, duration_s)
        if nonblocking:
            fut = asyncio.run_coroutine_threadsafe(coro, self._loop)
            with self._pending_lock:
                self._pending = fut
            return
        # In sim time this can take longer in wall time (see _WALL_GUARD)
        self._call(coro, timeout=duration_s * _WALL_GUARD + self.timeout)

    # ------------------------------------------------------------------ state

    def attitude_euler(self):
        """(roll, pitch, yaw) in degrees."""
        return self._call(self._first(self._drone.telemetry.attitude_euler()))

    def yaw_rad(self):
        """Heading (rad). Used by `_compensate_yaw` in place of AirSim's to_eularian_angles."""
        return math.radians(self.attitude_euler().yaw_deg)

    def position_ned(self):
        """PositionVelocityNed. position.north_m / east_m / down_m."""
        return self._call(
            self._first(self._drone.telemetry.position_velocity_ned())
        )

    def sim_time_s(self):
        """Sim time (s). `odometry.time_usec` is sim time, the same clock `_hold_sim_time`
        uses for durations. Recorded so run logs can be cross-checked against A's RTF."""
        od = self._call(self._first(self._drone.telemetry.odometry()))
        return None if od is None else od.time_usec / 1e6

    @staticmethod
    async def _first(stream):
        """Take a single item from an async stream (one-shot telemetry query)."""
        async for item in stream:
            return item
        return None
