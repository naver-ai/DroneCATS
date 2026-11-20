# datagen: rebuild the DroneCATS depth instruction dataset

## Introduction

This tool rebuilds a depth-estimation instruction dataset for drone vision-language
models. The dataset is single-turn and in English. It covers point depth, referring point +
depth, and drone-action questions over frames from NYU Depth v2, SUN RGB-D and TartanAir v2.

No image is redistributed. You download the images from their official sources and our
small **annotation package** (questions, answers and a description of how each image is
derived). `ddr reconstruct` combines them into the dataset, and `ddr verify` checks that the
result is byte-identical to the released dataset. The questions, answers and images of the
released dataset are the same as in our training data; only the top-level `meta` of each row
is reduced to 8 documented fields.

```
  official image archives                    annotation package (10 MB, no images)
  nyu_depth_v2_labeled.mat                   rows/       questions and answers
  SUNRGBD.zip                                ledger/     how each image is derived
  TartanAir v2 image_lcam_front.zip x 6      reference/  SHA-256 of every dataset file
                 \                              /
                  +----> ddr reconstruct <-----+
                               |
                               v
                 dataset/  rows + images/*.zip
                               |
                               v
                          ddr verify  -->  PASS
```

## Quick start

```bash
bash scripts/run_all.sh --out /data/ddr      # from datagen/ in a clone of this repository
# or, from the repository root: bash datagen/scripts/run_all.sh --out /data/ddr
```

One command does everything below: it creates the pinned environment, downloads the
official image archives and the annotation package, then runs `ddr fetch-check`, `ddr reconstruct`
and `ddr verify`. The dataset ends up in `/data/ddr/dataset/`. It needs `python3.10` and `wget`,
and it is resumable. Use `--downloads DIR` to keep the 72 GB of archives somewhere else and
`--venv DIR` to place the environment; `bash scripts/run_all.sh --help` lists all options.
The sections below are the same steps one by one.

## Requirements

- Python 3.10 on Linux x86_64. Byte-identical output is guaranteed only with the pinned
  versions in `requirements-lock.txt`, because JPEG and PNG encoders differ between library
  builds.
- About 90 GB of free disk: 72 GB for the downloads, 9 GB for the dataset and 8 GB of
  temporary files.
- No GPU and no API key.

## Installation

```bash
cd datagen                       # in a clone of this repository
python3.10 -m venv .venv && . .venv/bin/activate
pip install -r requirements-lock.txt
pip install -e . --no-deps
```

This installs the `ddr` command from this repository (`python -m drone_depth_recon` is the
same).

## Try it without downloads

```bash
python examples/tiny_reconstruct.py /tmp/ddr_tiny
```

This writes a two-frame stand-in for `SUNRGBD.zip` and a matching annotation package, then
runs `ddr reconstruct` and `ddr verify` on them (PASS). It shows the package layout and the
two commands; the images are synthetic.

## Usage

### 1. Download the image archives

| Dataset | Files | Official source |
|---|---|---|
| NYU Depth v2 (labeled) | `nyu_depth_v2_labeled.mat` (2.8 GiB) | https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html |
| SUN RGB-D | `SUNRGBD.zip` (6.4 GiB) | https://rgbd.cs.princeton.edu/ |
| TartanAir v2 | `image_lcam_front.zip` in `Data_easy` and `Data_hard` of `Downtown`, `ModularNeighborhoodIntExt` and `NordicHarbor` (6 files, about 62 GB) | Hugging Face [`theairlabcmu/tartanair2`](https://huggingface.co/datasets/theairlabcmu/tartanair2) |

Read each source's terms of use first. Then download these files into one working
directory on a disk with enough space (the commands below use it as the current directory):

```bash
bash /path/to/datagen/scripts/download.sh .
```

The script fetches the files from the official URLs above, keeps the TartanAir files in their
published folder layout under `tartanair2/`, and checks each file's size. It is resumable:
if a download stops, run it again and it continues where it left off. SHA-256 is checked in
step 3.

### 2. Download the annotation package

Download the package from the Hugging Face dataset
[naver-ai/DroneCATS-Pilot-54K](https://huggingface.co/datasets/naver-ai/DroneCATS-Pilot-54K)
into a local folder `PACKAGE` (the command comes with `pip install -U huggingface_hub`):

```bash
hf download naver-ai/DroneCATS-Pilot-54K --repo-type dataset --local-dir PACKAGE
sha256sum PACKAGE/reference/SHA256SUMS
# expected: 2fe1286e6a626b1fe582466f36281ab56c217a508ab1ab6147ee2ef21147e144
```

`reference/SHA256SUMS` lists the SHA-256 of the row files and of every dataset file, so this
one digest anchors everything `ddr verify` checks. It does not change when the package
documentation changes.

`PACKAGE` is the package folder in the commands below. Run them from the working directory
of step 1.

### 3. Check the downloads

```bash
ddr fetch-check --package PACKAGE \
    --nyu nyu_depth_v2_labeled.mat --sunrgbd SUNRGBD.zip --tartanair-root tartanair2
```

This compares the size and SHA-256 of all of these archives with `PACKAGE/ledger/archives.json`.
If an archive does not match, the source has changed and the dataset cannot be rebuilt
from it.

### 4. Reconstruct

```bash
ddr reconstruct --package PACKAGE \
    --nyu nyu_depth_v2_labeled.mat --sunrgbd SUNRGBD.zip --tartanair-root tartanair2 \
    --out dataset/ --jobs 16
```

`dataset/` then holds 6 row files (`.jsonl`) and 89 image zips under
`images/` (about 9 GB). Rows reference images as `images/<zip>#<member>`. Temporary files go
to `dataset.work/`; remove it when you are done.

### 5. Verify

```bash
ddr verify --package PACKAGE --dataset dataset/
```

PASS (exit code 0) means every file matches `PACKAGE/reference/SHA256SUMS` byte for byte.
Use `--report report.json` for the full report, and `--mode pixel` to compare decoded
pixels instead of bytes when you run with other library versions.

With the pinned environment we rebuilt the full dataset from the official archives and got
PASS: 25,727 / 25,727 zip members and 6 / 6 row files byte-identical.

## How to Cite

If you use the dataset or this tool, cite the DroneCATS paper; see *Citation* in the root
[`README.md`](../README.md#citation).

Also cite the source datasets: Silberman et al., ECCV 2012
(NYU Depth v2); Song et al., CVPR 2015, Janoch et al., ICCV-W 2011, Xiao et al., ICCV 2013
(SUN RGB-D and its constituents); Wang et al., IROS 2020 (TartanAir). Full references are
in [`LICENSES/THIRD_PARTY_DATA.md`](LICENSES/THIRD_PARTY_DATA.md).

## License

| Part | License |
|---|---|
| Code in `datagen/` | The DroneCATS repository license (see *License* in the root [`README.md`](../README.md#license)); third-party package notices in [`NOTICE`](NOTICE) |
| Annotations (rows, ledger, reference hashes) | CC BY-NC 4.0, non-commercial research purposes only ([`LICENSES/ANNOTATIONS.md`](LICENSES/ANNOTATIONS.md)) |
| Images (NYU Depth v2, SUN RGB-D, TartanAir v2) | Not redistributed; each source's own terms apply ([`LICENSES/THIRD_PARTY_DATA.md`](LICENSES/THIRD_PARTY_DATA.md)) |

The referring expressions and object tags in the annotations were generated with OpenAI GPT
models; users of the annotations must also comply with OpenAI's Terms of Use
(https://openai.com/policies/terms-of-use). Depth labels come from the source datasets'
ground-truth depth.
