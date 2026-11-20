# Annotation license

Copyright (c) 2026-present NAVER Cloud Corp.

The annotations are licensed under the **Creative Commons Attribution-NonCommercial 4.0
International License (CC BY-NC 4.0)** and are provided for **non-commercial research
purposes only**. The full legal code is in [`CC-BY-NC-4.0.txt`](CC-BY-NC-4.0.txt) and at
https://creativecommons.org/licenses/by-nc/4.0/legalcode.

## Scope

"Annotations" means the content of the data package:

- the row files (`rows/*.jsonl`);
- the ledger (`ledger/*.jsonl`, `ledger/archives.json`) and the reference hashes
  (`reference/SHA256SUMS`);

for all sources (NYU Depth v2, SUN RGB-D, TartanAir v2).

The annotation license does **not** cover:

- **Images.** No image of any source is included in the data package. The images you
  rebuild with `ddr reconstruct` from the official downloads remain under the terms of their
  source dataset ([`THIRD_PARTY_DATA.md`](THIRD_PARTY_DATA.md)).
- **Code.** The code (the `datagen/` folder of the DroneCATS repository, https://github.com/naver-ai/DroneCATS) is licensed under that
  repository's license; see "License" in the root `README.md` and `datagen/NOTICE`.

## Model-generated text

Parts of the annotations were generated with OpenAI GPT models: the object referring
expressions and object tags in the rows (for example `meta.desc` and the target
descriptions inside questions). Recipients of the annotations must also comply with
OpenAI's Terms of Use (https://openai.com/policies/terms-of-use) and the policies referenced
there.

## Attribution

When you use the annotations, cite the DroneCATS paper (see "Citation" in the root `README.md`
of https://github.com/naver-ai/DroneCATS)
and the source datasets listed in [`THIRD_PARTY_DATA.md`](THIRD_PARTY_DATA.md).
