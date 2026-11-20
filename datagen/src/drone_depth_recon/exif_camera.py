# DroneCATS
# Copyright (c) 2026-present NAVER Cloud Corp.
# Apache-2.0

"""Camera intrinsics (pixel fx, fy) in standard EXIF tags."""
from __future__ import annotations

import struct
from fractions import Fraction
from io import BytesIO

from PIL import Image
from PIL.TiffImagePlugin import IFDRational

# FocalLength = fx * 36 / W (mm); FocalPlane{X,Y}Resolution = 10 * f{x,y} / FocalLength (px/cm).
EXIF_IFD = 0x8769
TAG_FOCAL_LENGTH = 0x920A
TAG_FP_X_RES = 0xA20E
TAG_FP_Y_RES = 0xA20F
TAG_FP_UNIT = 0xA210
TAG_EXIF_WIDTH = 0xA002
TAG_EXIF_HEIGHT = 0xA003
TAG_FOCAL_35MM = 0xA405
TAG_ORIENTATION = 0x0112

CAMERA_TAGS = (TAG_FOCAL_LENGTH, TAG_FP_X_RES, TAG_FP_Y_RES, TAG_FP_UNIT,
               TAG_EXIF_WIDTH, TAG_EXIF_HEIGHT, TAG_FOCAL_35MM)

FULL_FRAME_WIDTH_MM = Fraction(36)
UNIT_MM = {2: Fraction(254, 10), 3: Fraction(10), 4: Fraction(1)}
WRITE_UNIT = 3
RATIONAL_MAX = 0xFFFFFFFF
# Orientations 5-8 rotate by 90 degrees, so displayed width and height swap.
SWAPPED_ORIENTATIONS = frozenset({5, 6, 7, 8})


def _frac(value: float | int | str | Fraction) -> Fraction:
    """Decimal spelling, not the binary expansion: the release was written from str(float)."""
    if isinstance(value, Fraction):
        return value
    if isinstance(value, int):
        return Fraction(value)
    return Fraction(str(float(value)))


def _rational(value: Fraction) -> IFDRational:
    fr = value.limit_denominator(RATIONAL_MAX)
    while fr.numerator > RATIONAL_MAX or fr.denominator > RATIONAL_MAX:
        fr = fr.limit_denominator(fr.denominator // 2)
    return IFDRational(fr.numerator, fr.denominator)


def build_camera_exif(fx: float | str, fy: float | str, width: int, height: int) -> Image.Exif:
    fx_r, fy_r = _frac(fx), _frac(fy)
    if not (fx_r > 0 and fy_r > 0 and width > 0 and height > 0):
        raise ValueError(f"fx, fy, width, height must be positive: {fx=} {fy=} {width=} {height=}")
    focal_mm = fx_r * FULL_FRAME_WIDTH_MM / width
    unit_mm = UNIT_MM[WRITE_UNIT]
    exif = Image.Exif()
    sub = exif.get_ifd(EXIF_IFD)
    sub[TAG_FOCAL_LENGTH] = _rational(focal_mm)
    sub[TAG_FP_X_RES] = _rational(fx_r * unit_mm / focal_mm)
    sub[TAG_FP_Y_RES] = _rational(fy_r * unit_mm / focal_mm)
    sub[TAG_FP_UNIT] = WRITE_UNIT
    sub[TAG_EXIF_WIDTH] = int(width)
    sub[TAG_EXIF_HEIGHT] = int(height)
    sub[TAG_FOCAL_35MM] = int(round(float(focal_mm)))
    return exif


def exif_payload(exif: Image.Exif) -> bytes:
    payload = exif.tobytes()
    return payload if payload.startswith(b"Exif\x00\x00") else b"Exif\x00\x00" + payload


def jpeg_insert_app1(data: bytes, payload: bytes) -> bytes:
    """Insert APP1 right after SOI, no re-encode (piexif places it elsewhere: bytes differ)."""
    if data[:2] != b"\xff\xd8":
        raise ValueError("not a JPEG: missing SOI")
    if len(payload) + 2 > 0xFFFF:
        raise ValueError(f"APP1 payload exceeds 64 KiB: {len(payload)}")
    segments: list[bytes] = []
    i = 2
    while i < len(data):
        if data[i] != 0xFF:
            raise ValueError(f"expected a marker at offset {i}, got {data[i]:#04x}")
        marker = data[i + 1]
        if marker == 0xD8 or marker == 0x01 or 0xD0 <= marker <= 0xD7:
            segments.append(data[i:i + 2])
            i += 2
            continue
        if marker == 0xDA:  # SOS: everything after it is kept verbatim
            break
        length = struct.unpack(">H", data[i + 2:i + 4])[0]
        seg = data[i:i + 2 + length]
        if not (marker == 0xE1 and seg[4:10] == b"Exif\x00\x00"):
            segments.append(seg)
        i += 2 + length
    app1 = b"\xff\xe1" + struct.pack(">H", len(payload) + 2) + payload
    return b"\xff\xd8" + app1 + b"".join(segments) + data[i:]


def add_camera_exif(data: bytes, fx: float | str, fy: float | str,
                    width: int, height: int) -> bytes:
    """PNG is re-saved by Pillow, JPEG is not re-encoded; size must match (fx is read from W)."""
    payload = exif_payload(build_camera_exif(fx, fy, width, height))
    with Image.open(BytesIO(data)) as img:
        fmt = (img.format or "").upper()
        if img.size != (int(width), int(height)):
            raise ValueError(f"image size {img.size} != ({int(width)}, {int(height)})")
        if fmt == "PNG":
            img.load()
            out = BytesIO()
            img.save(out, format="PNG", exif=payload)
            return out.getvalue()
        if fmt not in ("JPEG", "MPO"):
            raise ValueError(f"unsupported image format: {fmt}")
    return jpeg_insert_app1(data, payload)


def camera_exif_tags(data: bytes) -> dict[int, tuple]:
    with Image.open(BytesIO(data)) as img:
        sub = dict(img.getexif().get_ifd(EXIF_IFD))
    out: dict[int, tuple] = {}
    for tag in CAMERA_TAGS:
        if tag in sub:
            v = sub[tag]
            num, den = getattr(v, "numerator", None), getattr(v, "denominator", None)
            out[tag] = (int(num), int(den)) if den is not None else (int(v), 1)
    return out


def expected_camera_exif_tags(fx: float | str, fy: float | str,
                              width: int, height: int) -> dict[int, tuple]:
    sub = build_camera_exif(fx, fy, width, height).get_ifd(EXIF_IFD)
    out: dict[int, tuple] = {}
    for tag in CAMERA_TAGS:
        v = sub[tag]
        num, den = getattr(v, "numerator", None), getattr(v, "denominator", None)
        out[tag] = (int(num), int(den)) if den is not None else (int(v), 1)
    return out


def _as_fraction(value) -> Fraction:
    num, den = getattr(value, "numerator", None), getattr(value, "denominator", None)
    if num is not None and den:
        return Fraction(int(num), int(den))
    return _frac(value)


def read_camera_exif(data: bytes) -> dict:
    """Falls back to FocalLengthIn35mmFilm (approximate) when the exact tags are absent."""
    with Image.open(BytesIO(data)) as img:
        width, height = img.size
        exif = img.getexif()
        orientation = exif.get(TAG_ORIENTATION)
        sub = dict(exif.get_ifd(EXIF_IFD))
    if orientation in SWAPPED_ORIENTATIONS:
        width, height = height, width
    out: dict = {"width": int(width), "height": int(height)}
    focal = sub.get(TAG_FOCAL_LENGTH)
    x_res, y_res = sub.get(TAG_FP_X_RES), sub.get(TAG_FP_Y_RES)
    unit_mm = UNIT_MM.get(int(sub.get(TAG_FP_UNIT) or 0))
    if focal and x_res and unit_mm:
        focal_f = _as_fraction(focal)
        w_orig = int(sub.get(TAG_EXIF_WIDTH) or width)
        h_orig = int(sub.get(TAG_EXIF_HEIGHT) or height)
        sensor_w = w_orig / (_as_fraction(x_res) / unit_mm)
        out["fx"] = float(focal_f * width / sensor_w)
        res_y = _as_fraction(y_res) if y_res else _as_fraction(x_res)
        sensor_h = h_orig / (res_y / unit_mm)
        out["fy"] = float(focal_f * height / sensor_h)
        return out
    f35 = sub.get(TAG_FOCAL_35MM)
    if f35:
        fx = float(_frac(f35) * width / FULL_FRAME_WIDTH_MM)
        out["fx"] = fx
        out["fy"] = fx
    return out
