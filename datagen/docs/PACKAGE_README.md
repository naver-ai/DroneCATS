# Depth-estimation instruction dataset: annotation package

This package, together with the official NYU Depth v2, SUN RGB-D and TartanAir v2
downloads, rebuilds the released dataset byte for byte. It contains annotations, the
derivation ledger and reference hashes only; **no image of any source is included**. Use
the `drone_depth_recon` tools (`ddr fetch-check`, `ddr reconstruct`, `ddr verify`) from the
`datagen/` folder of the DroneCATS code repository (https://github.com/naver-ai/DroneCATS).

## Contents

| Path | What |
|---|---|
| `rows/` | The 6 row files `{train,val}_{depth,point_depth,action_point_depth}.jsonl`, one JSON row per line. Rows reference images as `images/<zip>#<member>`. |
| `ledger/derivations.jsonl` | One line per distinct derived image file: `path`, `source`, `frame`, `ops`, `sha256`, `bytes`, `pixel_sha256`. |
| `ledger/zip_layout.jsonl` | One line per released zip: `zip`, `sha256`, `bytes`, `members` (in archive order) with `name`, `path`, `date_time`. |
| `ledger/frames_nyuv2.jsonl` | `frame` -> `mat_index` (0-based index into `images` of `nyu_depth_v2_labeled.mat`), `scene`, size. |
| `ledger/frames_sunrgbd.jsonl` | `frame` -> `archive_member` (path inside `SUNRGBD.zip`), size. |
| `ledger/frames_tartanair.jsonl` | `frame` -> environment, difficulty, trajectory, frame index, and the member path in the official TartanAir v2 archive. |
| `ledger/archives.json` | Official URL, size and SHA-256 of every archive the reconstruction reads: `nyu_depth_v2_labeled.mat`, `SUNRGBD.zip`, and the TartanAir v2 archives keyed by their path in the official layout (`<env>/<Data_diff>/image_lcam_front.zip`). |
| `reference/SHA256SUMS` | SHA-256 of the 6 row files, of the 89 zips (`images/<zip>`) and of every zip member (`images/<zip>#<member>`). |
| `LICENSES/` | Annotation license (`ANNOTATIONS.md`, `CC-BY-NC-4.0.txt`) and source dataset notices (`THIRD_PARTY_DATA.md`). |

No image is included (only frame identifiers and digests).

## Rows

The rows are the training rows with the top-level `meta` reduced to the 8 fields below and
the files renamed; `messages` and the images are the same as in training. Each row is
`{"meta": {...}, "tools": [], "messages": [...]}`.

| `meta` field | Meaning |
|---|---|
| `source` | Source dataset of the frame: `nyuv2`, `sunrgbd` or `tartanair`. |
| `oid` | Target id `<frame>-<x>x<y>`: the source frame id and the target pixel in the original frame. |
| `desc` | Referring expression of the target object. It appears in the question of `point_depth` and `action_point_depth` rows; `depth` rows ask by coordinates only. |
| `point` | Target point `[x, y]` normalized to `[0, 1]` by the width and height of the image shown in the row (for crops, relative to the crop). `depth` questions and `point_depth` answers use this scale; `action_point_depth` prompts ask for integer coordinates in `0-1000`, and a `go` answer's point is `round(1000 * point)`. |
| `gt_depth_m` | Metric depth of the target in meters: median depth of the target region in the source depth map. |
| `target_kind` | Which depth the answer reports: `canonical_fov` (canonical depth, `gt_depth_target`) or `metric_paired` (a second row for the same `oid` that asks for metric depth; its answer is `gt_depth_m`). |
| `gt_depth_target` | Depth the answer is built from: `gt_depth_m / focal_ratio` (within 0.2 % for resized and cropped images; reference camera with horizontal FOV 2·atan(1/2) ≈ 53.13°, `fx = width`) for `canonical_fov`, `gt_depth_m` for `metric_paired`. |
| `focal_ratio` | `fx / width` of the image shown in the row (horizontal focal length over image width, both in pixels). |

The `meta` objects inside messages and content parts (for example `{"trainable": false}` or
`{}`) are part of the training format and are kept as is; `trainable` marks which turns are
training targets.

## Derivation paths and operations

A derivation path encodes how a file was made:

```
images/<source>/<kind>/[<W>x<H>/<base>/]<frame>[__<cell>__<x0>_<x1>]__(fx<fx>_fy<fy>_<W>x<H> | nofx).<ext>
```

`ops` in `derivations.jsonl` spells the same thing out; operations run in order:

| op | Meaning |
|---|---|
| `source` | Original frame bytes: SUN RGB-D JPEG as stored in the zip; TartanAir PNG as stored in the archive; NYU Depth v2 RGB array saved by Pillow as PNG with `compress_level=1`. |
| `crop_x` | Decode with OpenCV (`IMREAD_COLOR`) and keep columns `x0:x1`. |
| `fit_width_jpeg` | If wider than `max_width`, resize to that width with `cv2.INTER_AREA` (height `round(h * s)`), then encode JPEG with `cv2.imencode`, quality `quality`. |
| `resize_lanczos` | Pillow `resize((width, height), LANCZOS)`; PNG saved with `optimize=True`, JPEG saved with the source quantization tables, subsampling and progressive flag. |
| `camera_exif` | Add EXIF camera tags encoding `fx`, `fy` (pixels) for a `width` x `height` image. PNG is re-saved by Pillow with an `eXIf` chunk; JPEG gets an APP1 segment right after SOI without re-encoding. |

A file without `camera_exif` (`nofx`) carries no added EXIF block.

Several derivation paths can have identical bytes; the ledger lists the one used for each
zip member.

## Licenses

The content of this package (rows, ledger, reference hashes) is licensed under
CC BY-NC 4.0, for non-commercial research purposes only (`LICENSES/ANNOTATIONS.md`). The
referring expressions and object tags were generated with OpenAI GPT models; users must
also comply with OpenAI's Terms of Use (https://openai.com/policies/terms-of-use). Images
are not redistributed: the images you rebuild remain under the terms of NYU Depth v2,
SUN RGB-D and TartanAir v2 (download links and required citations in
`LICENSES/THIRD_PARTY_DATA.md`).
