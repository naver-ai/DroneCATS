# Third-party datasets (images not redistributed)

The rows and ledger reference frames of the three datasets below, but **no image from any
of them is included in this repository's data package**. Users download the originals from
the official sources themselves, under the terms stated there, and rebuild the derived
images locally with `ddr reconstruct`. The derived images you reconstruct remain subject to
the original dataset's terms. Where a dataset states no redistribution license, the absence
of a license is not a grant of rights.

## NYU Depth v2 (labeled set)

- Download: `nyu_depth_v2_labeled.mat` (2.8 GiB) from
  https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html
  (direct file: https://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat)
- Used: RGB frames of the labeled set (a subset of the 1,449 frames).
- Redistributed by us: no.
- Required citation:
  N. Silberman, D. Hoiem, P. Kohli, R. Fergus. "Indoor Segmentation and Support Inference
  from RGBD Images." ECCV 2012.

## SUN RGB-D

- Download: `SUNRGBD.zip` (6.4 GiB) from https://rgbd.cs.princeton.edu/
  (direct file: https://rgbd.cs.princeton.edu/data/SUNRGBD.zip)
- Subsets used:
  - train split: `kv1/NYUdata` only;
  - validation split: `kv1/b3dodata`, `kv2/kinect2data`, `kv2/align_kv2`,
    `xtion/sun3ddata`, `kv1/NYUdata`.
- Redistributed by us: no.
- Required citations (SUN RGB-D and its constituent datasets):
  - S. Song, S. P. Lichtenberg, J. Xiao. "SUN RGB-D: A RGB-D Scene Understanding Benchmark
    Suite." CVPR 2015.
  - N. Silberman, D. Hoiem, P. Kohli, R. Fergus. "Indoor Segmentation and Support Inference
    from RGBD Images." ECCV 2012.
  - A. Janoch, S. Karayev, Y. Jia, J. T. Barron, M. Fritz, K. Saenko, T. Darrell.
    "A Category-Level 3-D Object Dataset: Putting the Kinect to Work."
    ICCV Workshop on Consumer Depth Cameras for Computer Vision, 2011.
  - J. Xiao, A. Owens, A. Torralba. "SUN3D: A Database of Big Spaces Reconstructed Using
    SfM and Object Labels." ICCV 2013.

## TartanAir v2

- Download: Hugging Face dataset `theairlabcmu/tartanair2`
  (https://huggingface.co/datasets/theairlabcmu/tartanair2), front camera archives
  `<env>/<Data_easy|Data_hard>/image_lcam_front.zip` for the environments `Downtown`,
  `ModularNeighborhoodIntExt` and `NordicHarbor` (6 archives, about 62 GB in total). The
  official toolkit `tartanairpy` (https://tartanair.org) can download the same files.
- Used: front camera `lcam_front` RGB frames (640 x 640).
- Redistributed by us: no.
- License as stated by the source: obtained from Hugging Face (`theairlabcmu/tartanair2`),
  whose dataset card states BSD-3-Clause. If you obtain TartanAir v2 from tartanair.org
  instead, that site states CC BY 4.0.
- Required citation:
  W. Wang, D. Zhu, X. Wang, Y. Hu, Y. Qiu, C. Wang, Y. Hu, A. Kapoor, S. Scherer.
  "TartanAir: A Dataset to Push the Limits of Visual SLAM." IEEE/RSJ International
  Conference on Intelligent Robots and Systems (IROS), 2020.
  No citation specific to v2 was found; cite the original TartanAir paper (Wang et al.,
  IROS 2020).

## What the data package contains for these sources

- The annotation rows (questions, answers, points, depths, frame identifiers).
- The derivation ledger: per derived file, the source frame identifier, the list of
  operations, and SHA-256 digests of the derived file and of its decoded pixels.
- Frame tables mapping our frame identifiers to positions in the official archives, and the
  official URL, size and SHA-256 of every archive the reconstruction reads.

Digests allow verification but do not allow recovering any image.
