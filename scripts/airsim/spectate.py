#!/usr/bin/env python3
# DroneCATS
# Copyright (c) 2026-present NAVER Cloud Corp.
# Apache-2.0

"""AirSim spectator viewer — streams simGetImages frames to a browser as MJPEG.

On the server:
    cd vlmpilot && uv run python ../scripts/airsim/spectate.py --rpc 41451 --port 8901
On your local machine:
    ssh -N -L 8901:localhost:8901 <server>   # tunnel
    open http://localhost:8901 in a browser

- A single capture thread keeps only the latest frame → sim load is the same however many spectators there are.
- JPEG encoding happens in the viewer process (client side) → sim threads are untouched.
- Binds to 127.0.0.1 (SSH tunnel only, no external exposure)
- --rpc defaults to the optional environment variable MY_PORT, else 41451.
"""
import argparse
import os
import threading
import time

import airsim
import cv2
import numpy as np
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

latest = {"jpg": None, "t": 0.0, "fps": 0.0}


def capture_loop(rpc_port, camera, quality):
    req = [airsim.ImageRequest(camera, airsim.ImageType.Scene, False, False)]
    while True:  # if disconnected (e.g. sim restart), create a new client and reconnect
        try:
            # timeout is required: if the sim dies, msgpackrpc calls block forever without raising,
            # so the reconnect logic never runs. A 10s timeout breaks out into the reconnect loop.
            c = airsim.MultirotorClient(port=rpc_port, timeout_value=10)
            c.confirmConnection()
            print(f"[spectate] AirSim RPC {rpc_port} connected, capturing camera '{camera}'")
            n, t0 = 0, time.time()
            while True:
                r = c.simGetImages(req)[0]
                if r.width == 0:
                    time.sleep(0.2); continue
                img = np.frombuffer(r.image_data_uint8, dtype=np.uint8).reshape(r.height, r.width, 3)
                ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
                if ok:
                    latest["jpg"] = buf.tobytes()
                    latest["t"] = time.time()
                    n += 1
                    if n % 30 == 0:
                        latest["fps"] = 30 / (time.time() - t0); t0 = time.time()
        except Exception as e:
            print(f"[spectate] connection lost ({e.__class__.__name__}); reconnecting in 3s")
            time.sleep(3)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # disable request logging
        pass

    def do_GET(self):
        if self.path.startswith("/stream"):
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            last_t = 0.0
            try:
                while True:
                    if latest["jpg"] is None or latest["t"] == last_t:
                        time.sleep(0.01); continue
                    last_t = latest["t"]
                    jpg = latest["jpg"]
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(jpg)}\r\n\r\n".encode())
                    self.wfile.write(jpg + b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                return
        else:
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            # Stream watchdog: if the frame time from /stat has not changed for 3+ seconds (tunnel drop etc.),
            # re-point img at /stream → recovers automatically without a page reload.
            self.wfile.write(f"""<!doctype html><title>AirSim spectate</title>
<body style="margin:0;background:#111;color:#eee;font-family:sans-serif">
<div style="padding:6px">AirSim spectate — RPC {ARGS.rpc} / cam {ARGS.camera}
 <span id=f></span><span id=s style="color:#f66"></span></div>
<img id=v src=/stream style="width:100%;max-width:1280px;display:block">
<script>
let lastT = 0, stall = 0;
setInterval(async () => {{
  try {{
    const st = await (await fetch('/stat', {{cache:'no-store'}})).json();
    f.textContent = ' | ' + st.fps.toFixed(1) + ' fps';
    if (st.t === lastT) {{ stall++; }} else {{ stall = 0; s.textContent=''; }}
    lastT = st.t;
    if (stall >= 3) {{ s.textContent = ' reconnecting...'; v.src = '/stream?ts=' + Date.now(); stall = 0; }}
  }} catch (e) {{ s.textContent = ' lost server connection, retrying...'; }}
}}, 1000);
v.onerror = () => setTimeout(() => {{ v.src = '/stream?ts=' + Date.now(); }}, 1000);
</script>
</body>""".encode())

    def do_HEAD(self):
        self.send_response(200); self.end_headers()


class FpsHandler(Handler):
    pass


def main():
    global ARGS
    p = argparse.ArgumentParser()
    p.add_argument("--rpc", type=int, default=int(os.environ.get("MY_PORT", 41451)),
                   help="AirSim RPC port (default: $MY_PORT or 41451)")
    p.add_argument("--port", type=int, default=None,
                   help="viewer HTTP port (default: sim port + 1000 — one number to remember)")
    p.add_argument("--camera", default="chase")
    p.add_argument("--quality", type=int, default=80)
    ARGS = p.parse_args()
    if ARGS.port is None:
        ARGS.port = ARGS.rpc + 1000

    threading.Thread(target=capture_loop, args=(ARGS.rpc, ARGS.camera, ARGS.quality), daemon=True).start()

    # add /fps and /stat endpoints
    import json as _json
    orig_get = Handler.do_GET
    def do_GET(self):
        if self.path == "/fps":
            self.send_response(200); self.send_header("Content-Type", "text/plain"); self.end_headers()
            self.wfile.write(f"{latest['fps']:.1f}".encode()); return
        if self.path == "/stat":
            self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers()
            self.wfile.write(_json.dumps({"t": latest["t"], "fps": latest["fps"]}).encode()); return
        orig_get(self)
    Handler.do_GET = do_GET

    srv = ThreadingHTTPServer(("127.0.0.1", ARGS.port), Handler)
    print(f"[spectate] http://localhost:{ARGS.port}  (from a laptop: ssh -N -L {ARGS.port}:localhost:{ARGS.port} <server>)")
    srv.serve_forever()


if __name__ == "__main__":
    main()
