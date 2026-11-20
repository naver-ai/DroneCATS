# DroneCATS
# Copyright (c) 2026-present NAVER Cloud Corp.
# Apache-2.0

"""vlm_backends: one BGR frame + text prompt → the model's raw string.

This module knows nothing of prompt content, JSON parsing,
or action interpretation. It only handles request assembly for the 2 providers (gemini / openai),
pre-send image downscaling / JPEG / base64, timeouts, and a pure helper (`extract_action_json`) that keeps only
the JSON body of the model's raw output.

Observable stdout is the 4 `[VLM] …` lines of §5 plus the rate-limit line (`[VLM] rate_limited`, only when a 429
is hit); nothing is printed on the normal path.
"""

from __future__ import annotations

import copy
import base64
import json
import os
import re
import time
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Mapping, Protocol, Sequence

import cv2
import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types as genai_types
from openai import OpenAI

from vlmpilot.policy.camera import canonical_depth_scale

__all__ = [
    "VlmConfigError",
    "VlmBackend",
    "MultiGenerateResult",
    "build_backend",
    "supports_multi_image",
    "extract_action_json",
    "encode_frame_jpeg_b64",
    "camera_exif_bytes",
    "GeminiBackend",
    "OpenAICompatBackend",
]

# --------------------------------------------------------------------------- #
# Constants (§4.4: not exposed via config)
# --------------------------------------------------------------------------- #

PROVIDERS = ("gemini", "openai")

_DEFAULT_MAX_WIDTH = 640
_JPEG_QUALITY = 85
_RESIZE_INTERPOLATION = cv2.INTER_AREA
_DEFAULT_TIMEOUT_S = 8.0

# Rate-limit (429/RESOURCE_EXHAUSTED/503) retries; overridden by config key `api_max_retries`.
# Why 2 by default: a large multi-drone run throws hundreds of calls per minute with 15 lanes and 60 body streams. The 429s
# hit then are mostly burst limits that clear within seconds, so one or two waits recover them; without retries
# that call just starves and the run stays valid while only SR drops. Total wait is bounded by _RATE_LIMIT_BUDGET_S:
# the body's order TTL is 6 steps, so if one call is tied up for tens of seconds that drone stands
# in place on an expired order. 2 retries by default is 1 s + 2 s = 3 s, done within one step.
_API_MAX_RETRIES = 2
_RATE_LIMIT_BASE_WAIT_S = 1.0
_RATE_LIMIT_BUDGET_S = 10.0
_OPENAI_OUTPUT_TOKENS = 8192
_OPENAI_TEMPERATURE = 0.4
_OPENAI_DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
_OPENAI_DEFAULT_KEY_ENV = "OPENAI_API_KEY"

_GEMINI_TEMPERATURE = 0.4
_GEMINI_TOP_P = 0.95
_GEMINI_TOP_K = 40
_GEMINI_MAX_OUTPUT_TOKENS = 8192
_GEMINI_KEY_ENV = "GEMINI_API_KEY"
# Default thinking budget and the config value meaning "let the model decide" (GeminiBackend docstring).
_GEMINI_DEFAULT_THINKING_BUDGET = 0
_GEMINI_DYNAMIC_THINKING = "dynamic"

# Cap on exception messages put in logs/results. Provider 400 bodies run to hundreds of chars, which wouldn't fit one line.
_EXC_MESSAGE_CHARS = 200

_DEFAULT_MODEL = {
    "openai": "google/gemini-2.5-flash",
    "gemini": "gemini-2.5-flash",
}

_ENV_MAX_WIDTH = "VLMPILOT_VLM_MAX_WIDTH"
_ENV_TIMEOUT = "VLMPILOT_VLM_TIMEOUT"

_JPEG_DATA_URL_PREFIX = "data:image/jpeg;base64,"

# --------------------------------------------------------------------------- #
# once-per-process flags (§3.5, §5)
# --------------------------------------------------------------------------- #

_dotenv_loaded = False
_warned_schema_missing = False
_warned_gemini_timeout = False


def _reset_process_flags() -> None:
    """Test-only hook: resets the .env load flag and the once-per-process warning flags."""
    global _dotenv_loaded, _warned_schema_missing, _warned_gemini_timeout
    _dotenv_loaded = False
    _warned_schema_missing = False
    _warned_gemini_timeout = False


def _ensure_dotenv() -> None:
    global _dotenv_loaded
    if _dotenv_loaded:
        return
    _dotenv_loaded = True
    # Don't override the existing process environment (dotenv default). A missing .env is not an error.
    load_dotenv()


# --------------------------------------------------------------------------- #
# stdout (§5)
# --------------------------------------------------------------------------- #


def _fold(text: Any) -> str:
    """Fold an exception message into one line (newline → space)."""
    return " ".join(str(text).split())


def _trim(text: str) -> str:
    """Length cap for one-line logs (so the §5 lines don't get buried in provider 400 bodies)."""
    return text if len(text) <= _EXC_MESSAGE_CHARS else text[:_EXC_MESSAGE_CHARS] + "…"


def _describe_exc(exc: BaseException) -> str:
    """Exception as a one-line summary: ``Class: message``.

    This string goes not only to stdout but also into the result object (``MultiGenerateResult.error``):
    the caller must be able to state the reason when it prints "no response".
    """
    return f"{type(exc).__name__}: {_trim(_fold(exc))}"


def _emit(line: str) -> None:
    print(line, flush=True)


@dataclass(frozen=True)
class GenerateResult:
    """``generate``'s result with the response's reasoning text attached.

    ``thoughts is None`` means "no reasoning text arrived", not "reasoning was off".
    Recording both as the same value would make them indistinguishable later.
    """

    text: str | None
    thoughts: str | None = None


_THINK_CLOSE = "</think>"


def _openai_thoughts(message: Any) -> str | None:
    """The response's reasoning text. Never raises.

    There are two paths. The ``reasoning_content`` / ``reasoning`` fields exist only when the server
    runs with ``--reasoning-parser``. Our local vLLM runs without that option, so
    those fields are absent entirely; instead the chat template pre-fills ``<think>``,
    so the reasoning arrives inside ``content`` as-is, with no opening tag and only the closing
    tag. Looking only at the first path, even a response with 12000 chars of reasoning is recorded as "no reasoning".
    """
    try:
        for attr in ("reasoning_content", "reasoning"):
            value = getattr(message, attr, None)
            if isinstance(value, str) and value.strip():
                return value
        extra = getattr(message, "model_extra", None)
        if isinstance(extra, Mapping):
            for attr in ("reasoning_content", "reasoning"):
                value = extra.get(attr)
                if isinstance(value, str) and value.strip():
                    return value
        content = getattr(message, "content", None)
        if isinstance(content, str):
            # Cut at the LAST closing tag, which is where the action parser cuts
            # (policy.py `_extract_action_json` uses rfind). Splitting on the
            # first one instead would put the boundary in a different place on a
            # reply carrying two, so the reasoning we record and the action we
            # execute would disagree about where thinking ended.
            cut = content.rfind(_THINK_CLOSE)
            if cut >= 0:
                head = content[:cut].replace("<think>", "", 1).strip()
                if head:
                    return head
    except Exception:
        pass
    return None


def _deep_copy_mapping(src: Mapping[str, Any] | None) -> dict[str, Any]:
    """Copy a config mapping one level deep so a request never mutates the config."""
    if not src:
        return {}
    return {k: (dict(v) if isinstance(v, Mapping) else v) for k, v in src.items()}


def _turn_reasoning_on(extra_body: dict[str, Any]) -> None:
    """Flip a config-provided reasoning switch on, for a ``think`` step only.

    Routers take ``reasoning: {enabled: bool}``; the roster sets it false so the
    loop does not pay for reasoning tokens on ordinary steps.

    NOTE: with the shipped ``plan_schema`` arm this never runs. A plan step calls
    the backend with ``think=False`` (policy.py: ``want_think = False if plan_step
    else think``) because the plan step takes its deliberation as schema fields,
    not as native reasoning — so no call ever arrives here with ``think=True``.
    The support is kept for ``plan_schema: false``, which is how the base/A arms
    are reproduced; it is not dead code, just unused by the current configs.
    """
    reasoning = extra_body.get("reasoning")
    if isinstance(reasoning, Mapping):
        r = dict(reasoning)
        if r.get("enabled") is False:
            r["enabled"] = True
        extra_body["reasoning"] = r


_RATE_LIMIT_MARKERS = ("429", "resource_exhausted", "rate limit", "rate_limit",
                       "too many requests", "503", "unavailable", "overloaded")


def _is_rate_limited(exc: BaseException) -> bool:
    """Is this a provider failure that looks like rate limiting / transient overload (429 / RESOURCE_EXHAUSTED / 503)?

    Exception types differ per provider (openai.RateLimitError, google.genai.errors.ClientError,
    plain HTTP wrappers …), so type alone can't tell. Use the status code if present, else the type name +
    markers in the message. A false positive only costs holding an already-failed call a few seconds longer.
    """
    code = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    if code in (429, 503, "429", "503", "RESOURCE_EXHAUSTED", "UNAVAILABLE"):
        return True
    text = f"{type(exc).__name__} {_fold(exc)}".lower()
    return any(m in text for m in _RATE_LIMIT_MARKERS)


def _backoff_waits(retries: int) -> tuple[float, ...]:
    """Exponential waits of 1, 2, 4 … s, cut off the moment the total exceeds _RATE_LIMIT_BUDGET_S."""
    waits: list[float] = []
    total = 0.0
    for i in range(max(0, int(retries))):
        w = _RATE_LIMIT_BASE_WAIT_S * (2 ** i)
        if total + w > _RATE_LIMIT_BUDGET_S:
            break
        waits.append(w)
        total += w
    return tuple(waits)


def _resolve_api_max_retries(config: Mapping[str, Any]) -> int:
    """config `api_max_retries` (default _API_MAX_RETRIES). Default on invalid values."""
    raw = config.get("api_max_retries")
    if raw is None:
        return _API_MAX_RETRIES
    try:
        return max(0, int(raw))
    except (TypeError, ValueError):
        return _API_MAX_RETRIES


def _call_with_backoff(fn, waits: Sequence[float]):
    """Run ``fn()``, retrying **only rate-limit-type failures** after an exponential wait.

    Other exceptions are raised as-is: retrying 400s / auth / schema errors gives the same answer and only eats
    the run's time. When retries run out, the last exception is also raised as-is to the caller's §6.1 path
    (= None return + `[VLM] rate_limited` + `[VLM] request failed`). Same as the previous behavior.
    """
    for wait in waits:
        try:
            return fn()
        except Exception as exc:
            if not _is_rate_limited(exc):
                raise
            _emit_rate_limited(exc, wait)
            time.sleep(wait)
    return fn()


def _emit_rate_limited(exc: BaseException, wait_s: float | None = None) -> None:
    tail = f" — retrying in {wait_s:g}s" if wait_s else " — retries exhausted"
    _emit(f"[VLM] rate_limited ({type(exc).__name__}): {_trim(_fold(exc))}{tail}")


def _emit_request_failed(exc: BaseException) -> None:
    # Rate limiting is also reported on its own line: the marker the runner uses to count "starved runs" separately
    # from model faults (rate_limited in verifier/airsim/guardrail/api_failures.py).
    if _is_rate_limited(exc):
        _emit_rate_limited(exc)
    _emit(f"[VLM] request failed ({type(exc).__name__}): {_trim(_fold(exc))}")


def _emit_empty_response(model: str) -> None:
    _emit(f"[VLM] empty response from {model}")


def _warn_schema_missing_once() -> None:
    global _warned_schema_missing
    if _warned_schema_missing:
        return
    _warned_schema_missing = True
    _emit("[VLM] structured_output=true but there is no schema, ignoring it")


def _warn_gemini_timeout_once() -> None:
    global _warned_gemini_timeout
    if _warned_gemini_timeout:
        return
    _warned_gemini_timeout = True
    _emit("[VLM] gemini SDK without timeout support — continuing without a timeout")


# --------------------------------------------------------------------------- #
# Public types
# --------------------------------------------------------------------------- #


class VlmConfigError(RuntimeError):
    """Misconfiguration prevents building the backend (missing key, unknown provider, etc.).
    The process must die with this exception: a config failure is a non-zero exit."""


class VlmBackend(Protocol):
    provider: str
    model: str

    def generate(
        self,
        prompt: str,
        image_bgr: np.ndarray,
        think: bool = False,
        response_schema: Mapping[str, Any] | None = None,
    ) -> str | None: ...

    def generate_multi(
        self,
        prompt: str,
        images_bgr: Sequence[np.ndarray],
        labels: Sequence[str],
        think: bool = False,
        system: str | None = None,
        response_schema: Mapping[str, Any] | None = None,
        max_widths: Sequence[int | None] | None = None,
    ) -> str | None: ...


@dataclass(frozen=True)
class MultiGenerateResult:
    """``generate_multi`` plus the request metadata the team decision record wants.

    ``text`` follows ``generate`` semantics exactly: ``None`` = request failed, ``""`` = no body.
    ``usage`` is the provider's raw usage object (may not be JSON-native — artifacts folds it),
    ``thoughts`` the reasoning text when the provider exposes one. Both are best-effort / ``None``.
    ``error`` is the reason when ``text is None`` (``Class: message``). The exception is swallowed here, so
    this field is the only clue; it is what the caller appends to its line when it prints "no response".
    """

    text: str | None
    usage: Any | None = None
    thoughts: str | None = None
    error: str | None = None


def supports_multi_image(config: Mapping[str, Any]) -> bool:
    """Can this provider carry N labelled images in one request?

    Unknown providers → False (``build_backend`` rejects them anyway).
    """
    provider = _normalize_provider((config or {}).get("api_provider"))
    return provider in ("gemini", "openai")


class _FrameEncodeError(RuntimeError):
    """Image preprocessing/encoding failure. Handled via the §6.2 path (request failed → None)."""


# --------------------------------------------------------------------------- #
# §3.1 Image preprocessing
# --------------------------------------------------------------------------- #


def _resolve_max_width(max_width: int | None) -> int:
    """Argument → env VLMPILOT_VLM_MAX_WIDTH → 640. env is consulted only when the argument is None."""
    if max_width is not None:
        return int(max_width)
    raw = os.environ.get(_ENV_MAX_WIDTH)
    if raw is None:
        return _DEFAULT_MAX_WIDTH
    try:
        return int(str(raw).strip())
    except (TypeError, ValueError):
        # §8 ㉣: parse failures silently fall back to 640 (no log).
        return _DEFAULT_MAX_WIDTH


def sent_frame_size(width: int, height: int, max_width: int | None = None) -> tuple[int, int]:
    """The frame size the model actually receives. Must match _encode_frame_jpeg's downscale rule."""
    w, h = int(width), int(height)
    max_w = _resolve_max_width(max_width)
    if max_w > 0 and w > max_w:
        return max_w, int(h * max_w / w)
    return w, h
def _config_max_width(config: Mapping[str, Any]) -> int:
    """The backend's default width cap. config vlm_max_width > env VLMPILOT_VLM_MAX_WIDTH > 640.
    If per-frame caps (``max_widths``) are given, they win for those frames."""
    raw = (config or {}).get("vlm_max_width")
    if raw is None:
        return _resolve_max_width(None)
    try:
        return int(str(raw).strip())
    except (TypeError, ValueError):
        return _resolve_max_width(None)


# --------------------------------------------------------------------------- #
# §3.1e Camera EXIF: the very tags the training pipeline reads
# --------------------------------------------------------------------------- #
#
# Training extracts `{width, height, fx, fy}` from the image EXIF and feeds it to the model via `<|image_aux_start|>`.
# If serving doesn't carry EXIF, the model never sees the aux it always saw in training.
# Exact path of the training parser (camera_from_exif):
#     fx = FocalLength(mm) * width / (w_orig / (FocalPlaneXResolution / mm_per_unit))
# and `w_orig` falls back to the decoded width when ExifImageWidth is absent, so if we
# don't write ExifImageWidth it simply becomes `fx = FocalLength * FPRes / mm_per_unit`.
# Matching `fx_sent = W_sent * k` to this gives
#     FocalPlaneX/YResolution = W_sent / 36 * 10   (Unit=3 → mm_per_unit=10)
#     FocalLength             = 36 * k             (independent of W)
# hFOV 90° (k=0.5) · W 640 → FocalLength 18.0, FPRes 177.7778, fx 320.0,
# matching the values measured on the stored data. Vertically the same formula cancels via height/h_orig, giving fy == fx
# (square pixels). ExifImageWidth/Length are deliberately not written: the parser uses the decoded pixel size,
# so they are unnecessary, and a mismatch would only skew fx.
_EXIF_FULL_FRAME_MM = 36.0
_EXIF_RES_UNIT_CM = 3  # FocalPlaneResolutionUnit: 3 = centimetre(=10 mm)
_EXIF_IFD_POINTER = 0x8769
_EXIF_TAG_FOCAL_LENGTH = 0x920A
_EXIF_TAG_FOCAL_PLANE_X_RES = 0xA20E
_EXIF_TAG_FOCAL_PLANE_Y_RES = 0xA20F
_EXIF_TAG_FOCAL_PLANE_RES_UNIT = 0xA210
_EXIF_RATIONAL_DENOM_LIMIT = 10**6
_APP1_MARKER = b"\xff\xe1"
_SOI_MARKER = b"\xff\xd8"


def camera_exif_bytes(width_sent: int, scale: float) -> bytes:
    """Send width ``width_sent`` · canonical scale ``k`` → Exif TIFF bytes to put in APP1.

    Includes the ``Exif`` + 2 NUL byte prefix (``Image.Exif.tobytes()`` convention). Focal tags go into
    the Exif sub-IFD (0x8769), not the base IFD; the training parser merges that one in.
    Rationals are written exactly, without approximation, so ``fx`` matches ``width_sent * scale`` within 1e-6.
    """
    from PIL import Image  # noqa: PLC0415 — don't hoist an optional dependency to import time
    from PIL.TiffImagePlugin import IFDRational

    if not width_sent > 0:
        raise ValueError(f"width_sent must be > 0, got {width_sent!r}")
    k = Fraction(float(scale)).limit_denominator(_EXIF_RATIONAL_DENOM_LIMIT)
    exif = Image.Exif()
    ifd = exif.get_ifd(_EXIF_IFD_POINTER)
    # FocalLength = 36 * k  (keeping the denominator makes it exactly 18/1 at k=0.5)
    ifd[_EXIF_TAG_FOCAL_LENGTH] = IFDRational(int(_EXIF_FULL_FRAME_MM) * k.numerator,
                                              k.denominator)
    # FocalPlaneResolution = W / 36 * 10 px per cm
    res = IFDRational(10 * int(width_sent), int(_EXIF_FULL_FRAME_MM))
    ifd[_EXIF_TAG_FOCAL_PLANE_X_RES] = res
    ifd[_EXIF_TAG_FOCAL_PLANE_Y_RES] = res
    ifd[_EXIF_TAG_FOCAL_PLANE_RES_UNIT] = _EXIF_RES_UNIT_CM
    return exif.tobytes()


def _insert_app1(jpeg: bytes, exif_bytes: bytes) -> bytes:
    """Insert an APP1 (Exif) segment right after SOI. Pixels are untouched.

    Why no re-encode: the flag ON/OFF must not change quality, size, or speed.
    """
    if not jpeg.startswith(_SOI_MARKER):
        raise _FrameEncodeError("JPEG does not start with SOI (FFD8), so EXIF cannot be inserted")
    length = len(exif_bytes) + 2
    if length > 0xFFFF:
        raise _FrameEncodeError(f"EXIF segment is too large ({length} B)")
    return jpeg[:2] + _APP1_MARKER + length.to_bytes(2, "big") + exif_bytes + jpeg[2:]


def _exif_camera_enabled(config: Mapping[str, Any]) -> bool:
    """Whether to carry camera EXIF.

    Prompt, conversion, and EXIF are one set: when ``canonical_depth`` is on, EXIF is on too.
    An explicit experimental ``exif_camera`` wins (same pattern as ``canonical_depth_convert``).
    If neither, off → the sent bytes are exactly as before.
    """
    override = _optional_bool((config or {}).get("exif_camera"))
    if override is not None:
        return override
    return _optional_bool((config or {}).get("canonical_depth")) is True


def _config_exif_scale(config: Mapping[str, Any]) -> float | None:
    """``k = fx / W`` used when carrying EXIF. ``None`` if off or unobtainable.

    ``k`` comes from the same helper as the controller (canonical→metric conversion); two implementations
    would diverge. The backend doesn't receive the camera object, so it only looks at config ``fov_horizontal``.
    If the key is missing, no default is filled in: it warns and omits EXIF. The default FOV differs per path
    (airsim 90° / tello 82.6°), so guessing here could silently carry a wrong fx.
    """
    if not _exif_camera_enabled(config):
        return None
    fov = (config or {}).get("fov_horizontal")
    scale = None
    if fov is not None:
        try:
            scale = canonical_depth_scale(None, float(fov))
        except (TypeError, ValueError):
            scale = None
    if scale is None:
        _emit(f"[VLM] camera EXIF is on but fov_horizontal cannot be parsed, skipping it"
              f" (fov_horizontal={fov!r})")
        return None
    try:
        camera_exif_bytes(640, scale)  # surface Pillow absence/version problems once, here
    except Exception as exc:
        _emit(f"[VLM] could not build the camera EXIF, skipping it ({_describe_exc(exc)})")
        return None
    return float(scale)


def _encode_frame_jpeg(image_bgr: np.ndarray, max_width: int | None = None,
                       exif_scale: float | None = None) -> bytes:
    """§3.1 steps 1–6. Returns JPEG bytes. _FrameEncodeError on failure.

    If ``exif_scale`` (= k of ``camera.canonical_depth_scale``) is given, step 7 appends
    camera EXIF (APP1) (§3.1e). With ``None`` this function produces byte-for-byte the same result
    as before; pixels are identical either way (no re-encode).
    """
    if image_bgr is None or getattr(image_bgr, "size", 0) == 0:
        raise _FrameEncodeError("image_bgr is None or empty")

    img = np.array(image_bgr, copy=True)  # don't mutate the caller's array
    if img.ndim < 2:
        raise _FrameEncodeError(f"unexpected image shape {img.shape}")
    h, w = int(img.shape[0]), int(img.shape[1])
    if h <= 0 or w <= 0:
        raise _FrameEncodeError(f"unexpected image shape {img.shape}")

    max_w = _resolve_max_width(max_width)
    if max_w > 0 and w > max_w:
        new_size = (max_w, int(h * max_w / w))  # width-based, height rounded down
        img = cv2.resize(img, new_size, interpolation=_RESIZE_INTERPOLATION)

    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, _JPEG_QUALITY])
    if not ok or buf is None:
        raise _FrameEncodeError("cv2.imencode('.jpg') failed")
    jpeg = bytes(np.asarray(buf).tobytes())
    if exif_scale is None:
        return jpeg  # existing path: bytes unchanged
    # Based on the send size (the size after downscaling, not the original camera size).
    return _insert_app1(jpeg, camera_exif_bytes(int(img.shape[1]), float(exif_scale)))


def encode_frame_jpeg_b64(image_bgr: np.ndarray, max_width: int | None = None,
                          exif_scale: float | None = None) -> str:
    """Performs §3.1 preprocessing and returns a base64 (ASCII) string. Public for tests."""
    return _b64(_encode_frame_jpeg(image_bgr, max_width, exif_scale))


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("utf-8")


def _encode_labeled_frames(
    images_bgr: Sequence[np.ndarray],
    labels: Sequence[str],
    max_widths: Sequence[int | None] | None = None,
    default_width: int | None = None,
    exif_scale: float | None = None,
) -> list[tuple[str, bytes]]:
    """§3.1 preprocessing for every frame of a multi-image request, paired with its label.

    An empty label means "no label part" (the single-drone path sends one). Label/image
    count mismatch is a caller bug and is surfaced as ValueError (→ ``[VLM] request failed``).
    ``max_widths`` are per-frame width caps; positions with None use ``default_width``.
    """
    images = list(images_bgr or [])
    labels = list(labels or [])
    if not images:
        raise _FrameEncodeError("images_bgr is empty")
    if len(labels) != len(images):
        raise ValueError(f"labels({len(labels)}) and images_bgr({len(images)}) differ in count")
    widths = [None] * len(images) if max_widths is None else list(max_widths)
    if len(widths) != len(images):
        raise ValueError(f"max_widths({len(widths)}) and images_bgr({len(images)}) differ in count")
    return [("" if label is None else str(label),
             _encode_frame_jpeg(img, default_width if mw is None else mw, exif_scale))
            for label, img, mw in zip(labels, images, widths)]


# --------------------------------------------------------------------------- #
# §3.6 Response text cleanup
# --------------------------------------------------------------------------- #

_THINK_CLOSE = "</think>"
_FENCE_JSON = "```json"
_FENCE = "```"
_RE_JSON_ARRAY = re.compile(r"\[\s*\{[\s\S]*\}\s*\]")
_RE_JSON_OBJECT = re.compile(r"\{[\s\S]*\}")


def _strip_code_fence(text: str) -> str:
    """Keep only the first fenced block. If there is no closing fence, take everything to the end."""
    for token in (_FENCE_JSON, _FENCE):
        start = text.find(token)
        if start < 0:
            continue
        begin = start + len(token)
        end = text.find(_FENCE, begin)
        return text[begin:] if end < 0 else text[begin:end]
    return text


def extract_action_json(raw: str | None) -> str:
    """Keep only the JSON body of the model's raw output (§3.6). Pure function, no I/O, never raises."""
    if raw is None:
        return ""
    try:
        text = raw if isinstance(raw, str) else str(raw)

        # 2. Strip the think trace: keep only what follows the last </think>.
        cut = text.rfind(_THINK_CLOSE)
        if cut >= 0:
            text = text[cut + len(_THINK_CLOSE):]

        # 3. Strip code fences (first block only).
        text = _strip_code_fence(text)

        # 4. Strip leading/trailing whitespace.
        text = text.strip()

        # 5. Extract the JSON body: arrays first, a single object is promoted to an array. Both greedy.
        m = _RE_JSON_ARRAY.search(text)
        if m:
            return m.group(0).strip()
        m = _RE_JSON_OBJECT.search(text)
        if m:
            return f"[{m.group(0)}]"
        return text
    except Exception:  # 7. Returns a str for any input.
        return "" if raw is None else str(raw)


# --------------------------------------------------------------------------- #
# Config resolution helpers (§3.3, §3.4, §4)
# --------------------------------------------------------------------------- #


def _normalize_provider(value: Any) -> str:
    if value is None:
        return "gemini"
    text = str(value).strip().lower()
    return text or "gemini"


def _resolve_model(config: Mapping[str, Any], provider: str) -> str:
    raw = config.get("model_name")
    name = "" if raw is None else str(raw).strip()
    return name or _DEFAULT_MODEL[provider]


def _positive_float(value: Any) -> float | None:
    """The value if it parses as a positive float, else None (moves on to the next stage)."""
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        # SPEC-GAP: config/env value is not numeric; the spec only states the '8' fallback for env.
        # A non-numeric config value moves on to the next stage instead of raising (so a typo doesn't kill the closed loop).
        return None
    if f <= 0:
        return None
    return f


def _resolve_timeout(config: Mapping[str, Any]) -> float:
    """config vlm_timeout → env VLMPILOT_VLM_TIMEOUT → 8."""
    t = _positive_float(config.get("vlm_timeout"))
    if t is not None:
        return t
    t = _positive_float(os.environ.get(_ENV_TIMEOUT))
    if t is not None:
        return t
    return _DEFAULT_TIMEOUT_S


def _optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    return bool(value)


def _resolve_thinking_budget(config: Mapping[str, Any]) -> int | None:
    """config ``gemini_thinking_budget`` → ThinkingConfig budget. ``None`` = no budget given.

    Missing key → 0 (existing behavior) / integer >= 0 → that value / -1 (any negative) or "dynamic" → None.
    Non-numeric typos go to the default instead of raising (same convention as ``_positive_float``: the closed loop
    doesn't die from one typo).
    """
    raw = config.get("gemini_thinking_budget")
    if raw is None:
        return _GEMINI_DEFAULT_THINKING_BUDGET
    if isinstance(raw, str):
        text = raw.strip().lower()
        if not text:
            return _GEMINI_DEFAULT_THINKING_BUDGET
        if text == _GEMINI_DYNAMIC_THINKING:
            return None
        raw = text
    try:
        budget = int(raw)
    except (TypeError, ValueError):
        return _GEMINI_DEFAULT_THINKING_BUDGET
    return None if budget < 0 else budget


# --------------------------------------------------------------------------- #
# Common backend skeleton
# --------------------------------------------------------------------------- #


class _BackendBase:
    provider: str = ""
    # Whether it can actually send multi-image requests. The presence of ``generate_multi`` alone
    # can't tell, since every backend inherits that name.
    supports_multi: bool = True

    def __init__(self, model: str, response_schema: Mapping[str, Any] | None, schema_name: str,
                 max_width: int | None = None, exif_scale: float | None = None) -> None:
        self.model = model
        self._response_schema = dict(response_schema) if response_schema is not None else None
        self._schema_name = schema_name
        self.max_width = _resolve_max_width(max_width)
        # None = don't carry camera EXIF (§3.1e). If set, fx = W_sent * exif_scale.
        self.exif_scale = exif_scale

    # Per-request mutable state is not kept on the instance (§6.15).
    def generate(
        self,
        prompt: str,
        image_bgr: np.ndarray,
        think: bool = False,
        response_schema: Mapping[str, Any] | None = None,
    ) -> str | None:
        return self.generate_detailed(
            prompt, image_bgr, think=think, response_schema=response_schema
        ).text

    def generate_detailed(
        self,
        prompt: str,
        image_bgr: np.ndarray,
        think: bool = False,
        response_schema: Mapping[str, Any] | None = None,
    ) -> GenerateResult:
        """``generate`` plus the reply's reasoning text, when the provider returns one.

        ``generate`` hands back a bare string, so a run could not afterwards say
        whether an endpoint reasoned. That matters now that ``think`` turns
        reasoning on for one call: the reply is the only evidence it did.
        """
        schema = response_schema if response_schema is not None else self._response_schema
        try:
            jpeg = _encode_frame_jpeg(image_bgr, self.max_width, self.exif_scale)
            text, thoughts = self._request_detailed(prompt, jpeg, bool(think), schema)
        except Exception as exc:  # §6.1/6.2 — provider exceptions are not propagated to the caller
            _emit_request_failed(exc)
            return GenerateResult(None)
        if text is None:
            _emit_empty_response(self.model)
            return GenerateResult("", thoughts)
        if not isinstance(text, str):
            text = str(text)
        if text == "":
            _emit_empty_response(self.model)
        return GenerateResult(text, thoughts)

    def _request_detailed(
        self,
        prompt: str,
        jpeg: bytes,
        think: bool,
        schema: Mapping[str, Any] | None,
    ) -> tuple[str | None, str | None]:
        """(text, reasoning text). Backends that can't expose reasoning return None."""
        return self._request(prompt, jpeg, think, schema), None

    def _request(
        self,
        prompt: str,
        jpeg: bytes,
        think: bool,
        schema: Mapping[str, Any] | None,
    ) -> str | None:
        """One request. Returns the body string; None if a response came but had no body."""
        raise NotImplementedError

    # -- multi-image ------------------------------------------------------------

    def generate_multi(
        self,
        prompt: str,
        images_bgr: Sequence[np.ndarray],
        labels: Sequence[str],
        think: bool = False,
        system: str | None = None,
        response_schema: Mapping[str, Any] | None = None,
        max_widths: Sequence[int | None] | None = None,
    ) -> str | None:
        return self.generate_multi_detailed(
            prompt, images_bgr, labels, think=think, system=system,
            response_schema=response_schema, max_widths=max_widths,
        ).text

    def generate_multi_detailed(
        self,
        prompt: str,
        images_bgr: Sequence[np.ndarray],
        labels: Sequence[str],
        think: bool = False,
        system: str | None = None,
        response_schema: Mapping[str, Any] | None = None,
        max_widths: Sequence[int | None] | None = None,
    ) -> MultiGenerateResult:
        """Same as ``generate_multi`` but also returns usage / thoughts.

        Request body order: ``[prompt, label₁, img₁, …, labelₙ, imgₙ]`` — the prompt text
        is FIRST, then each frame preceded by its label.
        ``system`` (when non-empty) goes to the provider's system slot.
        Failure semantics are those of ``generate`` (§6.1/6.2): never raises, ``None`` on failure.
        """
        schema = response_schema if response_schema is not None else self._response_schema
        system_text = system if system else None
        try:
            items = _encode_labeled_frames(images_bgr, labels, max_widths,
                                           default_width=self.max_width,
                                           exif_scale=self.exif_scale)
            text, usage, thoughts = self._request_multi(prompt, items, bool(think), system_text, schema)
        except Exception as exc:  # don't propagate provider exceptions to the caller
            _emit_request_failed(exc)
            return MultiGenerateResult(None, error=_describe_exc(exc))
        if text is None:
            _emit_empty_response(self.model)
            # An empty body is a failure too. Leaving error empty makes the caller misdiagnose a parse failure (that line is the only clue).
            return MultiGenerateResult("", usage, thoughts, error="empty response body")
        if not isinstance(text, str):
            text = str(text)
        if text == "":
            _emit_empty_response(self.model)
        return MultiGenerateResult(text, usage, thoughts)

    def _request_multi(
        self,
        prompt: str,
        items: Sequence[tuple[str, bytes]],
        think: bool,
        system: str | None,
        schema: Mapping[str, Any] | None,
    ) -> tuple[str | None, Any | None, str | None]:
        """One multi-image request → (text, usage, thoughts). text is None if there is no body."""
        raise NotImplementedError(f"the {self.provider} backend does not support multi-image requests")


# --------------------------------------------------------------------------- #
# (a) gemini
# --------------------------------------------------------------------------- #


class GeminiBackend(_BackendBase):
    """google-genai backend.

    Config key ``gemini_thinking_budget`` (optional) sets the thinking budget. It applies only to normal calls
    with ``think=False``; ``think=True`` has never attached ``thinking_config`` at all.
      - key missing ............ budget 0. Existing behavior; 2.5-flash / 3.7-flash accept this value
      - integer >= 0 ........... that value is given as the budget
      - -1 or ``"dynamic"`` .... no ``thinking_config`` attached (the model decides)

    ⚠ gemini-2.5-pro can't turn thinking off. With budget 0 every call
    dies with ``400 INVALID_ARGUMENT: Budget 0 is invalid. This model only works in thinking mode.``.
    Measured: 0 → 400 / 128 → 1.9s OK / budget omitted (dynamic) → 6.3s OK.
    That is why arms using 2.5-pro as commander set ``head.gemini_thinking_budget: 128``
    (128 is the minimum allowed, closest in conditions to the other heads with thinking off).
    """

    provider = "gemini"

    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        response_schema: Mapping[str, Any] | None = None,
        schema_name: str = "vlmpilot_action",
    ) -> None:
        super().__init__(_resolve_model(config, "gemini"), response_schema, schema_name,
                         _config_max_width(config), _config_exif_scale(config))
        api_key = os.environ.get(_GEMINI_KEY_ENV, "")
        if not api_key or not api_key.strip():
            raise VlmConfigError(f"env {_GEMINI_KEY_ENV} is empty, so the gemini backend cannot be built")

        self.timeout_s = _resolve_timeout(config)
        client_kwargs: dict[str, Any] = {"api_key": api_key}
        # Parity fix (found in real flight): the Gemini API rejects deadlines under 10 s
        # ("Manually set deadline 8s is too short") — with the generic 8 s default every request
        # failed with 400 and the drone never moved. The previous runtime never set a Gemini
        # HTTP timeout; only an explicit config `vlm_timeout` opts in.
        explicit_timeout = _positive_float(config.get("vlm_timeout"))
        if explicit_timeout is not None:
            try:
                client_kwargs["http_options"] = genai_types.HttpOptions(timeout=int(explicit_timeout * 1000))
            except Exception:
                _warn_gemini_timeout_once()
        self._client = genai.Client(**client_kwargs)

        # ``None`` = don't attach thinking_config at all. Must be decided before _make_config.
        self._thinking_budget = _resolve_thinking_budget(config)
        # google-genai doesn't resend 429s by itself (unlike the openai SDK); handled here.
        self._retry_waits = _backoff_waits(_resolve_api_max_retries(config))

        # The generation config is built once and reused (§3.2(a)).
        base = dict(
            temperature=_GEMINI_TEMPERATURE,
            top_p=_GEMINI_TOP_P,
            top_k=_GEMINI_TOP_K,
            max_output_tokens=_GEMINI_MAX_OUTPUT_TOKENS,
        )
        self._gen_base = base
        # This path applies the schema too. The SDK takes it as response_mime_type + response_schema, and
        # in measurements JSON came out even with a schema conflicting with a "don't write JSON" instruction,
        # i.e. it is enforced. Previously the schema argument was accepted and dropped, so only the 2 gemini models decoded freely.
        # Structured output has only two values: ON unless set (same contract as the openai path).
        _explicit = _optional_bool(config.get("structured_output"))
        self._structured_output = True if _explicit is None else _explicit
        self._config_think = self._make_config(think=True, system=None)
        self._config_nothink = self._make_config(think=False, system=None)
        # System-instruction variants are keyed by (think, system) so a fixed team
        # system prompt is also built once and reused (bounded; a new system text evicts all).
        self._system_configs: dict[tuple[bool, str], Any] = {}

    def _make_config(self, *, think: bool, system: str | None, schema: Mapping[str, Any] | None = None) -> Any:
        kwargs: dict[str, Any] = dict(self._gen_base)
        if system:
            kwargs["system_instruction"] = system
        if schema is not None:
            kwargs["response_mime_type"] = "application/json"
            kwargs["response_schema"] = dict(schema)
        if think or self._thinking_budget is None:
            return genai_types.GenerateContentConfig(**kwargs)
        # no-think step: send thinking_budget=0. But this lever can't turn reasoning off
        # across the whole roster. Live API measurements (same prompt, response
        # usage_metadata.thoughts_token_count):
        #                                 not set     budget=0   level=low
        #   gemini-robotics-er-2-preview       348          0         195
        #   gemini-3.7-flash                   575        185         203
        # For robotics-ER 2, budget=0 is truly 0 (thoughts_token_count itself is None).
        # 3.7 Flash doesn't drop the parameter; it accepts it and cuts 575 -> 185,
        # but never down to 0 ("dynamic thinking by default" in the official docs).
        # So on this model reasoning can only be reduced, not turned off, and unifying the roster's
        # no-think steps at the same reasoning amount is impossible in principle: in the comparison, the two gemini
        # models' no-think differ as "0 tokens" vs "185 tokens".
        # thinking_level="low" is not an alternative: both models are higher than budget=0 (195/203).
        try:
            tc = genai_types.ThinkingConfig(thinking_budget=self._thinking_budget)
            return genai_types.GenerateContentConfig(**kwargs, thinking_config=tc)
        except Exception:
            # Guard for SDK versions that don't provide ThinkingConfig. Only local 'construction'
            # failures are caught here; both are plain dataclasses with no server round trip.
            # A server rejecting the budget value is not caught by this except (2.5-pro's
            # budget=0 rejection is the example, which is why that arm sets 128 via config).
            return genai_types.GenerateContentConfig(**kwargs)

    def _config_for(self, think: bool, system: str | None) -> Any:
        if not system:
            return self._config_think if think else self._config_nothink
        key = (bool(think), system)
        cfg = self._system_configs.get(key)
        if cfg is None:
            if len(self._system_configs) >= 8:
                self._system_configs.clear()
            cfg = self._make_config(think=think, system=system)
            self._system_configs[key] = cfg
        return cfg

    def _use_schema(self, think: bool) -> bool:
        """Same contract as the openai path: ON by default, off only via config structured_output: false.

        ``think`` is no longer consulted (the argument is for call-site compatibility). Previously, with the key missing, only think steps
        turned the schema off, but since plan_schema, think steps also go out with think=False, which killed that distinction.
        """
        return self._structured_output

    def _request(self, prompt, jpeg, think, schema):
        # token_param / sampling / reasoning_effort / enable_thinking are ignored on this
        # path (§6.7). response_schema is supported by the SDK, so it is applied.
        part = genai_types.Part.from_bytes(data=jpeg, mime_type="image/jpeg")
        if schema is not None and self._use_schema(think):
            cfg = self._make_config(think=think, system=None, schema=schema)
        else:
            cfg = self._config_think if think else self._config_nothink
        response = _call_with_backoff(
            lambda: self._client.models.generate_content(
                model=self.model,
                contents=[prompt, part],
                config=cfg,
            ), self._retry_waits)
        # If the accessor raises (safety-filter block etc.), it goes straight to the §6.2 path.
        return response.text

    def _request_multi(self, prompt, items, think, system, schema):
        contents: list[Any] = [prompt]
        for label, jpeg in items:
            if label:
                contents.append(label)
            contents.append(genai_types.Part.from_bytes(data=jpeg, mime_type="image/jpeg"))
        cfg = (self._make_config(think=think, system=system, schema=schema)
               if schema is not None and self._use_schema(think)
               else self._config_for(think, system))
        response = _call_with_backoff(
            lambda: self._client.models.generate_content(
                model=self.model,
                contents=contents,
                config=cfg,
            ), self._retry_waits)
        text = response.text
        return text, getattr(response, "usage_metadata", None), _gemini_thoughts(response)


def _gemini_thoughts(response: Any) -> str | None:
    """Thought parts of the first candidate, if the SDK exposes them. Best-effort, never raises."""
    try:
        candidates = getattr(response, "candidates", None) or []
        parts = getattr(getattr(candidates[0], "content", None), "parts", None) or []
        texts = [str(p.text) for p in parts if getattr(p, "thought", False) and getattr(p, "text", None)]
        return "\n".join(texts) or None
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# (b) openai-compatible
# --------------------------------------------------------------------------- #


def _resolve_openai_base_url(config: Mapping[str, Any]) -> str:
    raw = config.get("openai_base_url")
    if raw is not None and str(raw).strip():
        return str(raw).strip()
    env = os.environ.get("OPENAI_BASE_URL")
    if env is not None and env.strip():
        return env.strip()
    return _OPENAI_DEFAULT_BASE_URL


def _resolve_openai_api_key(config: Mapping[str, Any]) -> str:
    raw = config.get("api_key_env")
    key_env = str(raw).strip() if raw is not None and str(raw).strip() else _OPENAI_DEFAULT_KEY_ENV
    value = os.environ.get(key_env, "")
    if not value or not value.strip():
        if _resolve_openai_base_url(config).startswith(("http://127.0.0.1", "http://localhost")):
            return "EMPTY"  # local OpenAI-compatible server (vLLM): no key needed
        raise VlmConfigError(f"env {key_env} is empty, so the openai backend cannot be built")
    return value  # dummy values allowed; the format is not validated


def _openai_strict_schema(schema: Mapping[str, Any]) -> dict:
    """Copy conforming to OpenAI strict json_schema rules: every key required, optional keys nullable, additionalProperties false.

    The gpt-6 family (including via OpenRouter) rejects with 400 invalid_json_schema if any key is missing from required
    (measured). The original schema is left untouched since other backends use it as-is.
    """
    out = copy.deepcopy(dict(schema))

    def walk(n: Any) -> None:
        if isinstance(n, dict):
            if n.get("type") == "object" and "properties" in n:
                req = set(n.get("required", []))
                for k, v in n["properties"].items():
                    walk(v)
                    if k not in req:
                        t = v.get("type")
                        if isinstance(t, str):
                            v["type"] = [t, "null"]
                        elif isinstance(t, list) and "null" not in t:
                            v["type"] = t + ["null"]
                        if "enum" in v and None not in v["enum"]:
                            v["enum"] = list(v["enum"]) + [None]
                n["required"] = list(n["properties"].keys())
                n["additionalProperties"] = False
            for k in ("items", "anyOf", "oneOf"):
                if k in n:
                    walk(n[k])
        elif isinstance(n, list):
            for x in n:
                walk(x)

    walk(out)
    return out


def _drop_json_nulls(text: str | None) -> str | None:
    """Clean up a strict response: keep only the first JSON value (drop trailing whitespace / duplicate JSON) and remove null keys.

    In strict mode gpt-6-luna often appends thousands of lines of whitespace, or the same JSON once more, after the JSON
    (measured; the parser dropped the step with JSONDecodeError: Extra data). The parser treats only 'key missing'
    as an optional field (null would hit float(None) etc.), so null keys are removed too.
    """
    if not text:
        return text
    body = text.strip()
    starts = [k for k in (body.find("{"), body.find("[")) if k >= 0]
    if not starts:
        return text
    try:
        data, end = json.JSONDecoder().raw_decode(body, min(starts))
    except ValueError:
        return text

    def clean(n: Any) -> Any:
        if isinstance(n, dict):
            return {k: clean(v) for k, v in n.items() if v is not None}
        if isinstance(n, list):
            return [clean(x) for x in n]
        return n

    cleaned = clean(data)
    if cleaned == data and min(starts) == 0 and not body[end:].strip():
        return body              # nothing to clean: return the raw text as-is (whitespace and order preserved)
    return json.dumps(cleaned, ensure_ascii=False, separators=(",", ":"))


class OpenAICompatBackend(_BackendBase):
    provider = "openai"

    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        response_schema: Mapping[str, Any] | None = None,
        schema_name: str = "vlmpilot_action",
    ) -> None:
        super().__init__(_resolve_model(config, "openai"), response_schema, schema_name,
                         _config_max_width(config), _config_exif_scale(config))
        self.base_url = _resolve_openai_base_url(config)
        self.timeout_s = _resolve_timeout(config)

        # Structured output has only two values: ON unless set. Previously None was a third state, 'OFF only on think
        # steps', but since plan_schema, think steps also go out with think=False, which killed that distinction; it remained a trap
        # where only configs missing the key silently behaved differently (tello_claude did).
        # Warn only when the user explicitly set true, so as not to be noisy for normal use of the backend
        # without a schema, like probe_vlm, and for the head-body commander backend built without a schema.
        _explicit = _optional_bool(config.get("structured_output"))
        self._structured_output = True if _explicit is None else _explicit
        self._structured_output_explicit = _explicit is True
        # OpenAI strict schema (_openai_strict_schema). Default off; only configs that turn it on change.
        self._strict_schema = _optional_bool(config.get("strict_schema")) is True
        self._enable_thinking = _optional_bool(config.get("enable_thinking"))
        # Provider-specific argument passthrough. E.g. OpenRouter only turns reasoning off via
        # extra_body {"reasoning": {"enabled": false}} (chat_template_kwargs has no effect).
        # Measured: left on, Qwen3.5-27B spends 4,800 reasoning tokens and ~95 s per step.
        raw_extra = config.get("extra_body")
        self._extra_body = dict(raw_extra) if isinstance(raw_extra, Mapping) else None
        token_param = config.get("token_param")
        self._token_param = (
            str(token_param).strip() if token_param is not None and str(token_param).strip() else "max_tokens"
        )
        sampling = config.get("sampling")
        self._sampling = True if sampling is None else bool(sampling)
        effort = config.get("reasoning_effort")
        self._reasoning_effort = effort if effort else None
        # Some providers express reasoning as a level rather than a switch, so
        # there is nothing for ``think`` to flip. This is that model's think
        # setting: unset means the level does not change, which is now what the
        # whole shipped roster does — under ``plan_schema`` the backend never sees
        # ``think=True``, so a think-only level could not fire even if set. Kept
        # for ``plan_schema: false`` (base/A arms); no shipped config sets it.
        effort_think = config.get("reasoning_effort_think")
        self._reasoning_effort_think = effort_think if effort_think else None
        api_key = _resolve_openai_api_key(config)
        # The SDK resends 429/5xx itself, honoring Retry-After; here only the count is given.
        self._client = OpenAI(
            api_key=api_key,
            base_url=self.base_url,
            timeout=self.timeout_s,
            max_retries=_resolve_api_max_retries(config),
        )

    def _build_kwargs(self, think: bool, schema: Mapping[str, Any] | None) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}

        # 1. Structured output: ON by default, off only via config. Attach the schema if present.
        if self._structured_output:
            if schema is not None:
                js: dict[str, Any] = {"name": self._schema_name, "schema": schema}
                if self._strict_schema:
                    js = {"name": self._schema_name, "schema": _openai_strict_schema(schema), "strict": True}
                kwargs["response_format"] = {"type": "json_schema", "json_schema": js}
            elif self._structured_output_explicit:
                # Warn only if config explicitly set true. The default (no key) + no schema is the normal form for probe_vlm
                # and the head-body commander, so pass quietly.
                _warn_schema_missing_once()

        # 2. Reasoning switch. config sets the default, and ``think`` flips it for that one call
        #    only.
        #     ⚠️ In the current roster this flip never fires: the plan_schema path calls plan steps
        #     with think=False (want_think in policy.py), so no call arrives here
        #     with think=True. The user's decision is the same: don't change the reasoning level for think steps,
        #     use only the usual level. Code support is kept because it is needed to reproduce the base/A arms
        #     with plan_schema:false.
        #     Order is the contract: config lays down the default first (provider-specific argument passthrough,
        #     e.g. OpenRouter's reasoning {enabled: false}), then ``think`` flips it for that one
        #     call only. Merging afterwards would let the raw config overwrite the flip, so
        #     think steps would also go out with reasoning off (this actually happened during a rebase).
        extra_body: dict[str, Any] = _deep_copy_mapping(self._extra_body)
        if think:
            _turn_reasoning_on(extra_body)
        if self._enable_thinking is not None:
            on = bool(self._enable_thinking) or think
            extra_body["chat_template_kwargs"] = {
                "enable_thinking": on,
                "skip_reasoning": not on,
            }
        if extra_body:
            kwargs["extra_body"] = extra_body

        # 3. Output cap: only the key name varies, the value is fixed at 8192.
        kwargs[self._token_param] = _OPENAI_OUTPUT_TOKENS

        # 4. Sampling: temperature only, only when sampling is true.
        if self._sampling:
            kwargs["temperature"] = _OPENAI_TEMPERATURE

        # 5. Reasoning depth. The current roster uses only the usual level: no config sets
        #    reasoning_effort_think, and even if one did, think=True never reaches here on the plan_schema
        #    path, so the left branch of the ternary below is never taken. For plan_schema:false reproduction.
        effort = (self._reasoning_effort_think if think and self._reasoning_effort_think
                  else self._reasoning_effort)
        if effort:
            kwargs["reasoning_effort"] = effort

        return kwargs

    def _request_detailed(self, prompt, jpeg, think, schema):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": _JPEG_DATA_URL_PREFIX + _b64(jpeg)}},
                ],
            }
        ]
        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            **self._build_kwargs(think, schema),
        )
        choices = getattr(response, "choices", None)
        if not choices:
            return None, None  # request succeeded but no body → ""
        message = choices[0].message
        content = _drop_json_nulls(message.content) if self._strict_schema else message.content
        return content, _openai_thoughts(message)

    def _request_multi(self, prompt, items, think, system, schema):
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for label, jpeg in items:
            if label:
                content.append({"type": "text", "text": label})
            content.append({"type": "image_url", "image_url": {"url": _JPEG_DATA_URL_PREFIX + _b64(jpeg)}})
        messages: list[dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": content})
        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            **self._build_kwargs(think, schema),
        )
        usage = getattr(response, "usage", None)
        choices = getattr(response, "choices", None)
        if not choices:
            return None, usage, None
        message = choices[0].message
        content = _drop_json_nulls(message.content) if self._strict_schema else message.content
        return content, usage, _openai_thoughts(message)


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #

_BACKEND_CLASSES = {
    "gemini": GeminiBackend,
    "openai": OpenAICompatBackend,
}


def build_backend(
    config: Mapping[str, Any],
    *,
    response_schema: Mapping[str, Any] | None = None,
    schema_name: str = "vlmpilot_action",
) -> VlmBackend:
    """Build a backend from config (= YAML dict). VlmConfigError on failure."""
    _ensure_dotenv()
    config = config or {}
    provider = _normalize_provider(config.get("api_provider"))
    cls = _BACKEND_CLASSES.get(provider)
    if cls is None:
        raise VlmConfigError(
            f"unknown api_provider={provider!r}. allowed values: {', '.join(PROVIDERS)}"
        )
    return cls(config, response_schema=response_schema, schema_name=schema_name)
