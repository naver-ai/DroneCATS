# DroneCATS
# Copyright (c) 2026-present NAVER Cloud Corp.
# Apache-2.0

"""artifacts.py — on-disk outputs of one run.

Pure disk logic only: run-folder creation, per-step decision files, the recognition
grid, and AirSim recording → mp4 assembly. No AirSim RPC calls happen here
(recording start/stop belongs to ``mission``).

Contract strings emitted from this module:

* ``[GRID] saved: {out} ({k} steps)``
* ``[RECORD] saved: {out}  (cam {cam}, {n} frames, {fps:.1f}fps, {codec})``

Multi-drone additions: ``save_team_decision`` writes one
``decision_<ts>.jpg`` mosaic of per-drone tiles + ``decision_<ts>.json`` with the same naming
rule as ``save_decision``, so ``build_recognition_grid`` works untouched.
"""

from __future__ import annotations

import glob
import json
import math
import os
import subprocess
import time
from typing import Any, Iterable, Sequence

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"
RUN_DIR_SUFFIX_RETRIES = 99  # "", "-<pid>", "-<pid>-1" … "-<pid>-99"
DECISION_PREFIX = "decision_"
ARRIVED_SUFFIX = "_arrived"
WAIT_PREFIX = "wait_"          # (old) hover-wait view: kept for reader compatibility
LIVE_PREFIX = "live_"          # 'current view' uploaded every 2s throughout the flight (head/body)
VIEW_FRAMES_KEEP = 3           # only the 'current' view matters: keep the latest few, delete the rest
WAIT_FRAMES_KEEP = VIEW_FRAMES_KEEP    # old name
GRID_FILENAME = "recognition_grid.jpg"
RAW_SUFFIX = "_raw"          # decision_<ts>_raw.jpg = the frame as-is, without the baked overlay

# Saved decision frames use the preprocessing every model receives (vlm_backends: width <= 640, JPEG quality 85), not the
# 1280x720 capture: ~60 KB instead of ~440 KB per step, and identical in size to what the model saw. Turn off with
# config save_frames_at_sent_size: false or env VLMPILOT_SAVE_SENT_SIZE=0 (full-size frames, the previous behaviour).
# decision_<ts>.json pixel_u/pixel_v stay in capture pixels: multiply by (saved width / capture width) to draw on the jpg.
SAVE_SENT_SIZE = os.environ.get("VLMPILOT_SAVE_SENT_SIZE", "1").strip() != "0"


def set_save_sent_size(on: bool | None) -> None:
    """Config override of SAVE_SENT_SIZE (None keeps the env/default)."""
    global SAVE_SENT_SIZE
    if on is not None:
        SAVE_SENT_SIZE = bool(on)


def _imwrite_frame(path: str, image_bgr: Any) -> bool:
    if not SAVE_SENT_SIZE:
        return bool(cv2.imwrite(path, image_bgr))
    try:  # lazy: no import cycle; the same rule as the frame sent to the model
        from vlmpilot.policy.vlm_backends import _JPEG_QUALITY as q, _RESIZE_INTERPOLATION as interp, sent_frame_size
    except Exception:  # noqa: BLE001 — backend deps missing (offline tools): same constants as vlm_backends
        q, interp = 85, cv2.INTER_AREA

        def sent_frame_size(w, h):  # noqa: E306
            mw = int(os.environ.get("VLMPILOT_VLM_MAX_WIDTH", "640") or 640)
            return (mw, int(h * mw / w)) if mw > 0 and w > mw else (w, h)
    h, w = image_bgr.shape[:2]
    sw, sh = sent_frame_size(w, h)
    img = cv2.resize(image_bgr, (sw, sh), interpolation=interp) if (sw, sh) != (w, h) else image_bgr
    return bool(cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, q]))

# Recognition grid geometry (R-61g)
GRID_CELL_W = 384
GRID_CELL_H = 216
GRID_PAD = 6
GRID_BANNER_H = 30
GRID_HEADER_H = 34
GRID_COLS = 5
GRID_BG_GRAY = 30
GRID_JPEG_QUALITY = 92
_HEADER_BG_BGR = (52, 44, 36)
_HEADER_MODEL_BGR = (120, 220, 255)
_WHITE = (255, 255, 255)
_BLACK = (0, 0, 0)
_STEP_NO_BGR = (0, 255, 255)

# Banner label colours per action kind (R-64g)
_BANNER_GO_BGR = (0, 170, 0)
_BANNER_ROTATE_BGR = (0, 140, 255)
_BANNER_THINK_BGR = (0, 215, 215)
_BANNER_FINISHED_BGR = (0, 0, 220)
_BANNER_ARRIVED_BGR = (150, 75, 0)
_BANNER_OTHER_BGR = (90, 90, 90)

# Video assembly (R-93/R-94)
FPS_MIN = 1.0
FPS_MAX = 30.0
FPS_FALLBACK = 10.0
VIDEO_CODECS = ("h264_nvenc", "libx264")
MIN_FRAMES_PER_CAMERA = 2
RECORDING_IMAGES_SUBDIR = "images"


def _say(line: str = "") -> None:
    print(line, flush=True)


# ---------------------------------------------------------------------------
# §3.11 run folder
# ---------------------------------------------------------------------------


def create_run_dir(output_root: str) -> str:
    """Atomically claim ``<output_root>/<YYYYmmdd_HHMMSS>[-<pid>[-<i>]]`` (R-50).

    ``os.makedirs`` without ``exist_ok`` is the lock: two processes started in the
    same second can never share a folder (R-51). Absolute roots work as-is (R-52).
    """
    stamp = time.strftime(TIMESTAMP_FORMAT)
    base = os.path.join(output_root, stamp)
    pid = os.getpid()
    candidates = ["", f"-{pid}"] + [f"-{pid}-{i}" for i in range(1, RUN_DIR_SUFFIX_RETRIES + 1)]
    for suffix in candidates:
        path = base + suffix
        try:
            os.makedirs(path)  # exist_ok deliberately NOT used (R-51)
        except FileExistsError:
            continue
        return path
    raise RuntimeError(f"could not create the output folder: {base}")


# ---------------------------------------------------------------------------
# §3.12 per-step decision record
# ---------------------------------------------------------------------------


def _unique_stem(run_dir: str, stem: str) -> str:
    """Return ``stem`` or ``stem_1``, ``stem_2`` … so neither .jpg nor .json exists (R-59b).

    # SPEC-GAP: the original rule (R-59b) names the suffix ``-1`` and claims it sorts after
    # ``decision_<ts>.jpg``; in ASCII ``-`` (0x2D) < ``.`` (0x2E), so ``decision_<ts>-1.jpg``
    # would sort *before* the first step and the runner (which sorts decision_*.json by
    # name) would see the two steps swapped. ``_`` (0x5F) sorts after ``.`` and ``_1`` <
    # ``_arrived``, which is the order the rule actually wants: <ts> < <ts>_1 < <ts>_arrived.
    """
    candidate = stem
    i = 0
    while os.path.exists(os.path.join(run_dir, candidate + ".jpg")) or os.path.exists(
        os.path.join(run_dir, candidate + ".json")
    ):
        i += 1
        candidate = f"{stem}_{i}"
    return candidate


def _jsonable(obj: Any) -> Any:
    """``json.dumps(default=…)`` fallback: keep the value, never drop it.

    numpy scalars/arrays → Python; then ``model_dump()`` → ``to_dict()`` → ``dict()``;
    then the public part of ``__dict__``; last resort ``str(obj)``.
    """
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (set, frozenset, tuple)):
        return list(obj)
    for name in ("model_dump", "to_dict", "dict"):
        fn = getattr(obj, name, None)
        if callable(fn):
            try:
                return fn()
            except Exception:  # noqa: BLE001 — try the next representation
                pass
    attrs = getattr(obj, "__dict__", None)
    if isinstance(attrs, dict):
        public = {k: v for k, v in attrs.items() if not str(k).startswith("_")}
        if public:
            return public
    return str(obj)


def _dump_json_text(payload: Any) -> str:
    return json.dumps(payload, indent=2, ensure_ascii=False, default=_jsonable)


def _write_json_text(json_path: str, text: str) -> None:
    # Serialise FIRST (the caller did), then stream the finished string —
    # ``json.dump(obj, fh)`` would leave a truncated file behind if serialisation failed midway.
    with open(json_path, "w", encoding="utf-8") as fh:
        fh.write(text)


def _write_pair(run_dir: str, stem: str, image_bgr: Any, payload: Any) -> tuple[str, str]:
    jpg_path = os.path.join(run_dir, stem + ".jpg")
    json_path = os.path.join(run_dir, stem + ".json")
    text = _dump_json_text(payload)  # M21: nothing on disk until the whole JSON exists
    if not _imwrite_frame(jpg_path, image_bgr):
        raise OSError(f"jpg write failed: {jpg_path}")
    _write_json_text(json_path, text)
    return jpg_path, json_path


def save_decision(run_dir: str, record: Any) -> tuple[str, str]:
    """Write ``decision_<ts>.jpg`` + ``.json`` from a policy ``DecisionRecord`` (R-55).

    Reads only ``record.timestamp``, ``record.overlay_image_bgr`` and ``record.payload``;
    the content is not interpreted. Never raises: on failure a line is printed and
    ``("", "")`` is returned (the controller also isolates ``on_decision`` failures).
    """
    try:
        stem = _unique_stem(run_dir, f"{DECISION_PREFIX}{record.timestamp}")
        out = _write_pair(run_dir, stem, record.overlay_image_bgr, record.payload)
        # Unmarked original, for paper figures. The decision record (above) is already saved, so swallow failures.
        raw = getattr(record, "raw_image_bgr", None)
        if raw is not None:
            try:
                _imwrite_frame(os.path.join(run_dir, stem + RAW_SUFFIX + ".jpg"), raw)
            except Exception as exc:  # noqa: BLE001
                _say(f"[DECISION] raw frame write failed (ignored): {exc}")
        return out
    except Exception as exc:  # noqa: BLE001 — file logging must never stop the flight
        _say(f"[DECISION] write failed (ignored): {exc}")
        return "", ""


def save_view_frame(run_dir: str, image_bgr: Any, prefix: str = LIVE_PREFIX) -> str:
    """Write ``<prefix><ts>.jpg`` — what this drone sees *now*, for a commander watching remotely.

    Image only, no ``.json`` and no ``decision_`` prefix: this is **not** a decision. Everything
    that reads the decision stream (offline evaluation, an external viewer's step list, the
    arrival↔frame pairing) globs ``decision_*`` and must not see a fabricated action here. An
    external viewer can upload these as live frames, so a commander polling faster than one flight step still sees a fresh picture.
    Written through a ``.part`` temp so a reader never globs a half-file, and only the newest
    ``VIEW_FRAMES_KEEP`` survive — a 300 s flight at 2 s per frame would leave 150 files nobody
    will look at.
    Never raises — a missing frame costs the commander one turn of sight, not the flight.
    """
    try:
        stem = _unique_stem(run_dir, f"{prefix}{time.strftime(TIMESTAMP_FORMAT)}")
        path = os.path.join(run_dir, stem + ".jpg")
        ok, buf = cv2.imencode(".jpg", image_bgr)   # imwrite picks the encoder by extension, so it can't write .part
        if not ok:
            return ""
        with open(path + ".part", "wb") as fh:
            fh.write(buf.tobytes())
        os.replace(path + ".part", path)
        prune_view_frames(run_dir, prefix)
        return path
    except Exception as exc:  # noqa: BLE001 — file logging must never stop the flight
        _say(f"[VIEW] frame write failed (ignored): {exc}")
        return ""


def save_wait_frame(run_dir: str, image_bgr: Any) -> str:
    """(old) hover-wait view. :func:`save_view_frame` now does the same throughout the flight."""
    return save_view_frame(run_dir, image_bgr, WAIT_PREFIX)


def prune_view_frames(run_dir: str, prefix: str = LIVE_PREFIX) -> None:
    """Keep only the newest ``VIEW_FRAMES_KEEP`` ``<prefix>*.jpg``.

    The name carries the timestamp and ``_unique_stem`` appends ``_1``, ``_2`` … which sort after
    ``.jpg`` (``.`` < ``_``), so sorting by name is sorting by time — the same property R-59b
    relies on. A file already gone (another writer pruned it) is not an error.
    """
    for path in sorted(glob.glob(os.path.join(run_dir, f"{prefix}*.jpg")))[:-VIEW_FRAMES_KEEP]:
        try:
            os.remove(path)
        except OSError:  # noqa: PERF203 — delete failures don't affect the flight
            pass


def save_arrival_frame(run_dir: str, image_bgr: Any, drone: str = "") -> tuple[str, str]:
    """Write the final ``decision_<ts>_arrived.jpg`` + ``.json``.

    The JSON holds only ``{"actions": [{"kind": "arrived"}]}`` — the minimum
    the grid needs for its last cell. Multi-drone passes the
    declaring vehicle as ``drone`` and the entry gains ``"drone": <name>``; the single path
    passes nothing and the JSON is unchanged. Raises on failure; ``mission`` wraps the call.
    """
    stem = f"{DECISION_PREFIX}{time.strftime(TIMESTAMP_FORMAT)}{ARRIVED_SUFFIX}"
    entry: dict[str, Any] = {"kind": "arrived"}
    if drone:
        entry["drone"] = str(drone)
    return _write_pair(run_dir, stem, image_bgr, {"actions": [entry]})


# ---------------------------------------------------------------------------
# Team decision: mosaic + per-drone json
# ---------------------------------------------------------------------------

MOSAIC_COLS = 2
MOSAIC_TILE_WIDTH = 640
MOSAIC_PAD = 6
MOSAIC_BG_BGR = (30, 30, 30)


def build_mosaic(
    tiles: Sequence["np.ndarray"],
    cols: int = MOSAIC_COLS,
    tile_width: int = MOSAIC_TILE_WIDTH,
    pad: int = MOSAIC_PAD,
) -> np.ndarray:
    """Tile per-drone images left→right, top→bottom on a dark canvas.

    Every tile is resized to ``tile_width`` (aspect preserved); rows are as tall as the tallest
    tile; ``pad`` px of BGR (30,30,30) between and around tiles. ``None`` tiles are skipped
    (that slot simply disappears). ``cols`` is capped at the tile count so a lone tile does not
    leave an empty half. Raises ValueError when no tile is left.
    """
    scaled: list[np.ndarray] = []
    for tile in tiles:
        if tile is None:
            continue
        img = np.asarray(tile)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        h, w = int(img.shape[0]), int(img.shape[1])
        if w != tile_width:
            new_h = max(1, int(round(h * tile_width / w)))
            interp = cv2.INTER_AREA if w > tile_width else cv2.INTER_LINEAR
            img = cv2.resize(img, (int(tile_width), new_h), interpolation=interp)
        scaled.append(img)
    if not scaled:
        raise ValueError("build_mosaic: there are no tiles at all")
    cols = max(1, min(int(cols), len(scaled)))
    rows = math.ceil(len(scaled) / cols)
    pad = max(0, int(pad))
    cell_h = max(t.shape[0] for t in scaled)
    canvas = np.empty((pad + rows * (cell_h + pad), pad + cols * (int(tile_width) + pad), 3), np.uint8)
    canvas[:] = MOSAIC_BG_BGR
    for i, tile in enumerate(scaled):
        r, c = divmod(i, cols)
        y = pad + r * (cell_h + pad)
        x = pad + c * (int(tile_width) + pad)
        canvas[y : y + tile.shape[0], x : x + tile.shape[1]] = tile
    return canvas


def save_team_decision(run_dir: str, record: Any) -> tuple[str, str]:
    """Write ``decision_<ts>.jpg`` (mosaic) + ``decision_<ts>.json`` from a ``TeamDecisionRecord``.

    Same stem rule as :func:`save_decision`. Reads ``record.timestamp``, ``record.tiles_bgr``,
    ``record.payload`` and, if present, calls ``record.sync_actions()`` first so controller-filled
    fields (``gt_depth_m``) land in the JSON. No tiles → JSON only, jpg path ``""``.
    Never raises: on failure one line is printed and ``("", "")`` returned.
    """
    try:
        sync = getattr(record, "sync_actions", None)
        if callable(sync):
            sync()
        text = _dump_json_text(record.payload)  # M21: serialise before touching the disk
        stem = _unique_stem(run_dir, f"{DECISION_PREFIX}{record.timestamp}")
        jpg_path = os.path.join(run_dir, stem + ".jpg")
        json_path = os.path.join(run_dir, stem + ".json")
        tiles = [t for t in (record.tiles_bgr or []) if t is not None]
        jpg_out = ""
        if tiles:
            mosaic = build_mosaic(tiles, cols=MOSAIC_COLS)
            if not cv2.imwrite(jpg_path, mosaic):
                raise OSError(f"jpg write failed: {jpg_path}")
            jpg_out = jpg_path
        _write_json_text(json_path, text)
        return jpg_out, json_path
    except Exception as exc:  # noqa: BLE001 — file logging must never stop the flight
        _say(f"[DECISION] team write failed (ignored): {exc}")
        return "", ""


# ---------------------------------------------------------------------------
# §3.13 recognition grid
# ---------------------------------------------------------------------------


def _banner_for(json_path: str) -> tuple[str | None, tuple[int, int, int] | None]:
    """Label + BGR colour for one cell from its sibling ``.json`` (R-64g / R-65g)."""
    try:
        with open(json_path, encoding="utf-8") as fh:
            data = json.load(fh)
        first = (data.get("actions") or [None])[0] or {}
        kind = first.get("kind", first.get("action_type"))  # older runs wrote "action_type"
    except Exception:  # noqa: BLE001 — missing/corrupt json → no banner
        return None, None
    if not kind:
        return None, None

    if kind == "go":
        sx = first.get("pixel_u", first.get("screen_x"))
        sy = first.get("pixel_v", first.get("screen_y"))
        depth = first.get("depth_m")
        if sx is not None and sy is not None and depth is not None:
            try:
                return f"GO  x={int(sx)} y={int(sy)} depth={float(depth):g}m", _BANNER_GO_BGR
            except (TypeError, ValueError):
                pass
        return str(kind), _BANNER_OTHER_BGR  # R-65g: missing fields → grey fallback
    if kind == "rotate":
        angle = first.get("angle_deg")
        if angle is not None:
            try:
                return f"ROTATE  angle={float(angle):g} deg", _BANNER_ROTATE_BGR
            except (TypeError, ValueError):
                pass
        return str(kind), _BANNER_OTHER_BGR
    if kind == "think":
        return "THINK (reason next)", _BANNER_THINK_BGR
    if kind == "finished":
        return "FINISHED", _BANNER_FINISHED_BGR
    if kind == "arrived":
        return "ARRIVED (final view)", _BANNER_ARRIVED_BGR
    return str(kind), _BANNER_OTHER_BGR


def _draw_header(canvas: np.ndarray, instruction: str | None, model: str | None) -> None:
    width = canvas.shape[1]
    canvas[0:GRID_HEADER_H, :] = _HEADER_BG_BGR
    if instruction:
        txt = f'"{instruction}"'
        maxc = max(20, int((width - 260) / 11))
        if len(txt) > maxc:
            txt = txt[: maxc - 1] + '…"'
        cv2.putText(canvas, txt, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.62, _WHITE, 1, cv2.LINE_AA)
    if model:
        label = f"model: {model}"
        (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.58, 1)
        cv2.putText(
            canvas, label, (width - tw - 12, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.58,
            _HEADER_MODEL_BGR, 1, cv2.LINE_AA,
        )


def _render_cell(jpg_path: str, index: int) -> np.ndarray | None:
    img = cv2.imread(jpg_path)
    if img is None:
        return None
    img = cv2.resize(img, (GRID_CELL_W, GRID_CELL_H))
    cv2.rectangle(img, (0, 0), (46, 26), _BLACK, -1)
    cv2.putText(img, str(index + 1), (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, _STEP_NO_BGR, 2)
    label, colour = _banner_for(os.path.splitext(jpg_path)[0] + ".json")
    if label and colour:
        cv2.rectangle(img, (0, GRID_CELL_H - GRID_BANNER_H), (GRID_CELL_W, GRID_CELL_H), colour, -1)
        cv2.putText(
            img, label, (7, GRID_CELL_H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, _WHITE, 1, cv2.LINE_AA
        )
    return img


def build_recognition_grid(
    run_dir: str,
    cols: int = GRID_COLS,
    instruction: str | None = None,
    model: str | None = None,
) -> str | None:
    """Tile every ``decision_*.jpg`` of ``run_dir`` (sorted by name) into one JPEG (R-60g…R-66g).

    Returns the output path, or ``None`` when there is nothing to tile.
    """
    frames = sorted(glob.glob(os.path.join(run_dir, f"{DECISION_PREFIX}*.jpg")))
    if not frames:
        return None
    cols = max(1, int(cols))
    rows = math.ceil(len(frames) / cols)
    header_h = GRID_HEADER_H if (instruction or model) else 0
    canvas = np.full(
        (header_h + rows * (GRID_CELL_H + GRID_PAD) + GRID_PAD, cols * (GRID_CELL_W + GRID_PAD) + GRID_PAD, 3),
        GRID_BG_GRAY,
        np.uint8,
    )
    if header_h:
        _draw_header(canvas, instruction, model)

    for i, path in enumerate(frames):
        try:
            cell = _render_cell(path, i)
        except Exception:  # noqa: BLE001 — R-65g: one bad cell must not kill the grid
            cell = None
        if cell is None:
            continue
        r, c = divmod(i, cols)
        y = header_h + GRID_PAD + r * (GRID_CELL_H + GRID_PAD)
        x = GRID_PAD + c * (GRID_CELL_W + GRID_PAD)
        canvas[y : y + GRID_CELL_H, x : x + GRID_CELL_W] = cell

    out = os.path.join(run_dir, GRID_FILENAME)
    cv2.imwrite(out, canvas, [cv2.IMWRITE_JPEG_QUALITY, GRID_JPEG_QUALITY])
    _say(f"[GRID] saved: {out} ({len(frames)} steps)")
    return out


# ---------------------------------------------------------------------------
# §3.14 AirSim recording → mp4
# ---------------------------------------------------------------------------


def snapshot_recording_dirs(recording_root: str) -> set[str]:
    """Names currently under the AirSim recording root (taken before ``startRecording``).

    Raises if the root is missing/unreadable — ``mission`` turns that into
    ``[RECORD] recording start failed (continuing)`` (E-9).
    """
    return set(os.listdir(recording_root))


def _frame_timestamp_ns(path: str) -> int:
    stem = os.path.splitext(os.path.basename(path))[0]
    return int(stem.rsplit("_", 1)[-1])


def _camera_token(path: str) -> str:
    """``img_<vehicle>_<camera>_<imagetype>_<ts>.png`` → ``<camera>`` (index 2, R-92)."""
    return os.path.basename(path).split("_")[2]


def _measure_fps(frames: Iterable[str]) -> float:
    ordered = list(frames)
    n = len(ordered)
    try:
        dur = (_frame_timestamp_ns(ordered[-1]) - _frame_timestamp_ns(ordered[0])) / 1e9
    except (ValueError, IndexError):
        return FPS_FALLBACK
    if dur <= 0:
        return FPS_FALLBACK
    return max(FPS_MIN, min(FPS_MAX, (n - 1) / dur))


def _ffmpeg_command(pattern: str, fps: float, codec: str, out: str) -> list[str]:
    return [
        "ffmpeg", "-y",
        "-framerate", f"{fps:.2f}",
        "-pattern_type", "glob",
        "-i", pattern,
        "-c:v", codec,
        "-pix_fmt", "yuv420p",
        out,
    ]


FFMPEG_TIMEOUT_S = 300.0  # a hung ffmpeg must not block the finally chain (grid, landing)


def _run_ffmpeg(cmd: list[str], codec: str) -> int:
    """Run ffmpeg quietly; absorb one Ctrl+C and retry the same command (R-95).

    Bounded by FFMPEG_TIMEOUT_S (W5 review, MAJOR): on timeout the child is killed and a
    non-zero code is returned so the caller falls through like any encode failure.
    """
    def _once() -> int:
        try:
            return subprocess.run(cmd, capture_output=True, timeout=FFMPEG_TIMEOUT_S).returncode
        except subprocess.TimeoutExpired:
            _say(f"[RECORD] encoding exceeded {FFMPEG_TIMEOUT_S:g}s — aborted ({codec})")
            return -1

    try:
        return _once()
    except KeyboardInterrupt:
        _say(f"[RECORD] interrupted during encoding — retrying ({codec})")
        return _once()


def _encode_camera(images_dir: str, cam: str, out_dir: str) -> None:
    # imagetype token fixed to 0 (Scene) so chase frames never leak into cam0 (R-93)
    pattern = os.path.join(images_dir, f"img_*_{cam}_0_*.png")
    frames = sorted(glob.glob(pattern))
    n = len(frames)
    if n < MIN_FRAMES_PER_CAMERA:
        return
    fps = _measure_fps(frames)
    out = os.path.join(out_dir, f"flight_cam{cam}.mp4")
    for codec in VIDEO_CODECS:
        rc = _run_ffmpeg(_ffmpeg_command(pattern, fps, codec, out), codec)
        if rc == 0:
            _say(f"[RECORD] saved: {out}  (cam {cam}, {n} frames, {fps:.1f}fps, {codec})")
            return
    _say(f"[RECORD] encoding failed: cam {cam}")


def encode_flight_videos(recording_root: str, known_dirs: set[str], out_dir: str) -> None:
    """Assemble the newest recording folder into per-camera mp4s (R-90…R-98).

    Always returns normally — the finally chain (grid, landing) must continue.
    """
    try:
        fresh = sorted(set(os.listdir(recording_root)) - set(known_dirs))
        if not fresh:
            _say("[RECORD] new recording folder not found — skipping assembly")
            return
        rec_dir = os.path.join(recording_root, fresh[-1])
        images_dir = os.path.join(rec_dir, RECORDING_IMAGES_SUBDIR)
        all_frames = glob.glob(os.path.join(images_dir, "img_*.png"))
        if not all_frames:
            _say(f"[RECORD] no frames: {rec_dir}")
            return
        cameras = sorted({_camera_token(p) for p in all_frames})
        os.makedirs(out_dir, exist_ok=True)
        for cam in cameras:
            _encode_camera(images_dir, cam, out_dir)
    except KeyboardInterrupt:
        _say("[RECORD] assembly cancelled")
    except Exception as exc:  # noqa: BLE001 — R-97
        _say(f"[RECORD] assembly error: {exc}")
