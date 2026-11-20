# DroneCATS
# Copyright (c) 2026-present NAVER Cloud Corp.
# Apache-2.0

"""Target mover for the tracking tasks.

In approach/search the target is fixed, but in track/search_track it has to move.
Vehicles have physics disabled (physics_enabled=False), so they are moved by simSetObjectPose teleport.

The path is 'constant-speed motion in one direction from the start position'. It is reproducible without
road-network information, and fixing direction and speed as parameters makes cross-model comparison valid. Ground height
by default keeps the start z (flat-ground assumption); if waypoints are 3-element [x,y,z], z is linearly
interpolated per segment to follow the terrain (measured 0.5 m step between driveway↔road — with constant z it drives floating).

**One-way (oneway=True) is the default for tracking.** Because judging was unified as "any one of the arrival declarations
within δ + visible" — with a round trip (ping-pong) the target passes by a drone that just sits still,
so a single declaration at that moment passes even without ever following.
With one-way, declaring within δ requires actually catching up.

The cost of one-way: the path must be as long as `speed × episode cap` (e.g. 0.5 m/s × 300s
= 150m). If the waypoint polyline is shorter than that, it keeps going straight in the last segment's direction.
So the target may cross sidewalks and flower beds — it is a teleport so there are no collisions, and bench judging
(distance/visibility) is unaffected.

Defaults:
  speed 1.2 m/s (same as drone cruise — too fast makes tracking itself impossible)
  direction = the direction the vehicle faces (specify with heading_deg if needed)
  span_m is used only in round-trip mode (oneway=False)
"""

import math
import threading
import time


class TargetMover:
    def __init__(self, port, target_name, speed=1.2, span_m=20.0,
                 heading_deg=None, hz=20.0, vehicle_hint=None,
                 waypoints=None, yaw_offset_deg=0.0, oneway=True):
        self.port = port
        self.target = target_name
        self.speed = speed
        self.span = span_m
        self.heading_deg = heading_deg
        # waypoints: [[x,y], ...] — starts at the spawn position and follows the polyline at
        # constant speed. For paths that turn corners. If given, the straight-line params (span/heading) are ignored.
        self.waypoints = waypoints
        # one-way (default) / round trip. One-way never comes back — since judging is declaration-based,
        # a round trip is broken by "the target passes by even if you stay still" (see top of file).
        self.oneway = oneway
        # correction when the asset mesh is misaligned with the pose x axis (measured: Car_01 is -90 deg —
        # with yaw=hd it looks like it crab-walks sideways)
        self.yaw_offset = math.radians(yaw_offset_deg)
        self.period = 1.0 / hz
        self._stop = threading.Event()
        self._thread = None
        self.path = []          # the (t, x, y, z) actually traveled — for reproduction/verification
        self._errors = 0
        self._warned = False

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()
        return False

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self, timeout=5.0):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=timeout)

    def _run(self):
        import airsim
        try:
            # the default with no timeout (3600s) is the root of hangs — if the sim wedges, the mover
            # stalls for that long and the target stops.
            c = airsim.MultirotorClient(port=self.port, timeout_value=30)
            c.confirmConnection()
            p0 = c.simGetObjectPose(self.target)
            x0, y0, z0 = p0.position.x_val, p0.position.y_val, p0.position.z_val
            if any(math.isnan(v) for v in (x0, y0, z0)):
                raise ValueError("target NaN")
            _, _, tyaw = airsim.to_eularian_angles(p0.orientation)
        except Exception:
            self._errors += 1
            return

        hd = (math.radians(self.heading_deg) if self.heading_deg is not None
              else tyaw)          # if not specified, round trip along the direction the vehicle faces
        dx, dy = math.cos(hd), math.sin(hd)

        # waypoint-mode setup: cumulative-length table of the polyline spawn point→wp1→wp2….
        # if vertices are 3-element [x,y,z], z is also linearly interpolated per segment — so it does not
        # float or sink when passing segments with different road heights (measured 0.5 m step driveway↔road).
        # with 2-element [x,y], spawn z stays fixed as before (flat-ground backward compatibility).
        # cumulative length and speed are measured in xy only — z just follows the terrain and does not mix into speed.
        wps, cum, wz = None, None, None
        if self.waypoints:
            wps = [(x0, y0)] + [(float(p[0]), float(p[1])) for p in self.waypoints]
            wz = [z0] + [(float(p[2]) if len(p) > 2 else z0) for p in self.waypoints]
            cum = [0.0]
            for a, b in zip(wps, wps[1:]):
                cum.append(cum[-1] + math.dist(a, b))
            total = cum[-1] or 1.0

        t0 = time.time()
        while not self._stop.is_set():
            t = time.time() - t0
            travel = self.speed * t
            z = z0                     # straight-line and round-trip modes keep spawn z fixed
            if wps:
                if self.oneway:
                    # after reaching the end of the polyline, keep going straight in the last segment's direction.
                    # (so the target does not stop even if the path is shorter than the cap — if it stopped
                    #  it would become a static target, indistinguishable from approaching.)
                    s = min(travel, total)
                else:
                    # polyline ping-pong: fold 0→total→0→…
                    ph = travel % (2 * total)
                    s = ph if ph <= total else 2 * total - ph
                i = 1
                while i < len(cum) - 1 and cum[i] < s:
                    i += 1
                a, b = wps[i - 1], wps[i]
                seg = (cum[i] - cum[i - 1]) or 1.0
                k = (s - cum[i - 1]) / seg
                x, y = a[0] + (b[0] - a[0]) * k, a[1] + (b[1] - a[1]) * k
                z = wz[i - 1] + (wz[i] - wz[i - 1]) * k
                hd = math.atan2(b[1] - a[1], b[0] - a[0])
                if self.oneway:
                    over = travel - total
                    if over > 0:                  # end of path — extend in the last direction
                        x += math.cos(hd) * over
                        y += math.sin(hd) * over
                        z = wz[-1]                # the extension keeps the last vertex's height
                elif travel % (2 * total) > total:
                    hd += math.pi                 # on the return leg the body turns around too
                hd_now = hd
            elif self.oneway:
                # one-way straight line: keep going from the start point at bearing hd.
                x, y = x0 + dx * travel, y0 + dy * travel
                hd_now = hd
            else:
                # constant-speed round trip: triangle wave. Phase period span/speed.
                # folded position: 0→span→0→-span→0 ...  (range [-span, span])
                phase = (travel + self.span) % (4 * self.span)
                if phase < 2 * self.span:
                    off = phase - self.span
                    fwd = True                    # moving in the +bearing direction
                else:
                    off = 3 * self.span - phase
                    fwd = False                   # return leg
                x, y = x0 + dx * off, y0 + dy * off
                # the car always faces its direction of travel — rotates 180 deg on the return leg.
                # (with a fixed yaw it looks like it reverses/crab-walks the whole way back — observed)
                hd_now = hd if fwd else hd + math.pi
            try:
                c.simSetObjectPose(self.target, airsim.Pose(
                    airsim.Vector3r(x, y, z),
                    airsim.to_quaternion(0, 0, hd_now + self.yaw_offset)), True)
                # whether it moved is judged by **actual position**, not by the return value.
                # the current NH package returns False even for spawned actors
                # while applying the teleport correctly — looking only at the return value gives false alarms.
                # truly static ones (native map vehicles) stay in place, so they are caught here.
                if not self._warned and t > 3.0 and self.speed * t > 2.0:
                    got = c.simGetObjectPose(self.target).position
                    if math.dist((got.x_val, got.y_val), (x0, y0)) < 0.5:
                        print(f"  ⚠ TargetMover: {self.target} does not actually move"
                              f" (static?) — a moving target must be a spawned object", flush=True)
                    self._warned = True
                self.path.append((round(t, 2), round(x, 2), round(y, 2), round(z, 2)))
            except Exception:
                self._errors += 1
            self._stop.wait(self.period)
