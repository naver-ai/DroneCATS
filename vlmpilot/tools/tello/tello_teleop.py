#!/usr/bin/env python3
# DroneCATS
# Copyright (c) 2026-present NAVER Cloud Corp.
# Apache-2.0

# A teammate's manual control script, kept as-is. Used as a reference point for video diagnosis —
# if this window shows the full camera view in real time, the vehicle and environment are fine; if only the top part shows,
# it is not a vlmpilot code problem. This code was confirmed working at a teammate's home. Requires pygame: uv pip install pygame
"""Tello manual control (keyboard + video window + HUD). Runs on a MacBook.

  python mac_teleop.py               # real vehicle (Mac WiFi must be connected to TELLO-xxxxxx)
  python mac_teleop.py --dry-run     # check UI/keys only, no vehicle
  python mac_teleop.py --host 127.0.0.1   # via relay (stage 2)

Keys:  T takeoff · SPACE land · BACKSPACE emergency stop (motors stop immediately, vehicle falls) · Q/ESC quit (lands first if flying)
     ↑↓←→ forward/back/left/right · W/S up/down · A/D rotate left/right · 1~5 speed (10~100 steps)
Takeoff is sent only when a person presses T. There is no path by which the program takes off on its own.
"""
import argparse
import datetime as dt
import logging
import os
import sys
import time

import numpy as np

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")


# --------------------------------------------------------------------------- fake
class _FakeFrameRead:
    def __init__(self):
        self.t0 = time.time()

    @property
    def frame(self):
        h, w = 720, 960
        x = np.linspace(0, 255, w, dtype=np.uint8)
        img = np.zeros((h, w, 3), np.uint8)
        img[..., 0] = x
        img[..., 1] = int((time.time() - self.t0) * 40) % 255
        img[..., 2] = 80
        return img

    def stop(self):
        pass


class FakeTello:
    """Same method signatures as djitellopy.Tello. No network."""

    def __init__(self, host="fake"):
        self.is_flying = False
        self.h = 0
        self.bat = 87
        self.t_takeoff = None
        self.rc = (0, 0, 0, 0)
        self.log = logging.getLogger("faketello")

    def connect(self, wait_for_state=True):
        self.log.info("fake connect ok")

    def get_battery(self):
        return self.bat

    def streamon(self):
        pass

    def streamoff(self):
        pass

    def get_frame_read(self):
        return _FakeFrameRead()

    def takeoff(self):
        self.log.info("fake takeoff")
        self.is_flying = True
        self.h = 80
        self.t_takeoff = time.time()

    def land(self):
        self.log.info("fake land")
        self.is_flying = False
        self.h = 0

    def emergency(self):
        self.log.info("fake EMERGENCY")
        self.is_flying = False
        self.h = 0

    def send_rc_control(self, lr, fb, ud, yaw):
        self.rc = (lr, fb, ud, yaw)
        if self.is_flying:
            self.h = max(20, self.h + ud // 10)

    def get_current_state(self):
        ft = int(time.time() - self.t_takeoff) if self.t_takeoff and self.is_flying else 0
        return {"bat": self.bat, "h": self.h, "tof": self.h + 5, "templ": 60, "temph": 65,
                "time": ft, "baro": 0.0, "pitch": 0, "roll": 0, "yaw": 0, "vgx": 0, "vgy": 0, "vgz": 0}

    def end(self):
        pass


# --------------------------------------------------------------------------- main
def setup_logging(tag):
    os.makedirs(LOG_DIR, exist_ok=True)
    path = os.path.join(LOG_DIR, f"{tag}_{dt.datetime.now():%Y%m%d_%H%M%S}.log")
    fmt = logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")
    fh = logging.FileHandler(path)
    fh.setFormatter(fmt)
    for name in ("djitellopy", "faketello", "teleop"):
        lg = logging.getLogger(name)
        lg.setLevel(logging.INFO)
        lg.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logging.getLogger("teleop").addHandler(sh)
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="run without a drone")
    ap.add_argument("--host", default="192.168.10.1", help="Tello (or relay) address")
    ap.add_argument("--speed", type=int, default=40, help="rc magnitude 10~100 (default 40, recommended indoors)")
    ap.add_argument("--no-video", action="store_true", help="skip video reception (commands/state only)")
    ap.add_argument("--max-seconds", type=float, default=0, help="exit automatically after N seconds (for smoke tests)")
    args = ap.parse_args()

    log_path = setup_logging("dry" if args.dry_run else "mac_teleop")
    log = logging.getLogger("teleop")
    log.info("log file: %s", log_path)

    import pygame  # noqa: E402  (prints a banner on import)

    if args.dry_run:
        tello = FakeTello()
    else:
        from djitellopy import Tello
        tello = Tello(host=args.host)

    tello.connect()
    bat = tello.get_battery()
    log.info("connected. battery=%s%%", bat)
    if not args.dry_run and bat < 20:
        log.warning("battery %s%% — takeoff not recommended (risk of low-battery auto-landing)", bat)

    frame_read = None
    if not args.no_video:
        tello.streamon()
        frame_read = tello.get_frame_read()

    pygame.init()
    W, H = 960, 720
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Tello teleop")
    font = pygame.font.SysFont("menlo,monaco,monospace", 18)
    clock = pygame.time.Clock()

    speed = max(10, min(100, args.speed))
    lr = fb = ud = yaw = 0
    t_start = time.time()
    last_state_log = 0
    running = True

    def hud_lines(st):
        return [
            f"bat {st.get('bat','?')}%   h {st.get('h','?')}cm   tof {st.get('tof','?')}cm   "
            f"temp {st.get('templ','?')}-{st.get('temph','?')}C   flight {st.get('time','?')}s",
            f"rc lr={lr:+d} fb={fb:+d} ud={ud:+d} yaw={yaw:+d}   speed={speed}   "
            f"{'FLYING' if tello.is_flying else 'landed'}   {'DRY-RUN' if args.dry_run else args.host}",
            "T takeoff  SPACE land  BACKSPACE EMERGENCY  Q quit | arrows move  W/S up/down  A/D yaw  1-5 speed",
        ]

    try:
        while running:
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    running = False
                elif ev.type == pygame.KEYDOWN:
                    k = ev.key
                    if k in (pygame.K_q, pygame.K_ESCAPE):
                        running = False
                    elif k == pygame.K_t:
                        if not tello.is_flying:
                            log.info("KEY T -> takeoff")
                            tello.takeoff()
                    elif k == pygame.K_SPACE:
                        log.info("KEY SPACE -> land")
                        tello.land()
                    elif k == pygame.K_BACKSPACE:
                        log.info("KEY BACKSPACE -> EMERGENCY")
                        tello.emergency()
                    elif pygame.K_1 <= k <= pygame.K_5:
                        speed = (k - pygame.K_0) * 20
                        log.info("speed -> %d", speed)

            keys = pygame.key.get_pressed()
            lr = speed * (keys[pygame.K_RIGHT] - keys[pygame.K_LEFT])
            fb = speed * (keys[pygame.K_UP] - keys[pygame.K_DOWN])
            ud = speed * (keys[pygame.K_w] - keys[pygame.K_s])
            yaw = speed * (keys[pygame.K_d] - keys[pygame.K_a])
            # prevent the 15 s no-command auto-landing + control: keep sending regardless of flight state
            tello.send_rc_control(lr, fb, ud, yaw)

            st = tello.get_current_state()
            now = time.time()
            if now - last_state_log >= 1.0:
                log.info("state %s", {k: st.get(k) for k in ("bat", "h", "tof", "templ", "temph", "time")})
                last_state_log = now

            if frame_read is not None:
                fr = frame_read.frame  # RGB (djitellopy 2.5: PIL → ndarray)
                surf = pygame.surfarray.make_surface(fr.swapaxes(0, 1))  # (h,w,3) -> (w,h,3)
                if surf.get_size() != (W, H):
                    surf = pygame.transform.smoothscale(surf, (W, H))
                screen.blit(surf, (0, 0))
            else:
                screen.fill((20, 20, 20))
            for i, line in enumerate(hud_lines(st)):
                screen.blit(font.render(line, True, (255, 255, 0)), (8, 8 + 22 * i))
            try:
                hot = float(st.get("temph", 0)) >= 85
            except (TypeError, ValueError):
                hot = False
            if hot:  # the base model shuts down from overheating (~90C) while idling on the ground — take off immediately or power off
                screen.blit(font.render("!! OVERHEAT %sC: fly now or power off !!" % st.get("temph"),
                                        True, (255, 60, 60)), (8, 8 + 22 * 3))
            pygame.display.flip()
            clock.tick(20)  # 20Hz rc

            if args.max_seconds and now - t_start > args.max_seconds:
                log.info("max-seconds reached, quitting")
                running = False
    finally:
        if tello.is_flying:
            log.info("quit while flying -> land")
            try:
                tello.land()
            except Exception as e:  # noqa: BLE001
                log.error("land failed: %s", e)
        if frame_read is not None:
            try:
                frame_read.stop()
                tello.streamoff()
            except Exception:  # noqa: BLE001
                pass
        tello.end()
        pygame.quit()
        log.info("bye. log: %s", log_path)


if __name__ == "__main__":
    main()
