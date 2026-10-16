# DroneCATS

This is the official repository for the paper **"Evaluating Multimodal LLMs as Generalist Vision-Language-Action Agents for Drone Control: Commanding, Approaching, Tracking and Searching"**.

[![arXiv](https://img.shields.io/badge/arXiv-2609.01404-b31b1b.svg)](https://arxiv.org/abs/2609.01404)
[![Project Page](https://img.shields.io/badge/Project-Page-03C75A.svg)](https://naver-ai.github.io/DroneCATS/)
![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Model-coming%20soon-7488A6.svg)

Jaewoo Park\*, Minyoung Lee\*, Sukmin Seo\*, Moonbin Yim, Hyunwook Yoon, Dohoon Ryu, Daehee Kim,
Myungseo Song, Jihyuk Byun, Seunggyu Chang, Taeho Kil, Jiseob Kim, Bado Lee, Geewook Kim<sup>†</sup>

Drone AI Team, NAVER Cloud &nbsp;·&nbsp; \*Equal contribution &nbsp;·&nbsp; <sup>†</sup>Corresponding author

## News

- **2026.10.16** The paper is on [arXiv](https://arxiv.org/abs/2609.01404) and the [project page](https://naver-ai.github.io/DroneCATS/) is live.

## Overview

- **Agent.** DroneCATS Agent makes the MLLM the sole decision maker of a drone through 4 prompt-defined actions. It covers out-of-view and moving targets, and scales to multi-drone and commander + sub-agents structures.
- **Benchmark.** DroneCATS scores the agent under one declaration-based criterion, and includes diverse scenes across multiple simulators.
- **Evaluation.** 10 MLLMs, from frontier API models to small open models, together with our fine-tuned 2B model, are evaluated in every suite.
- **Real-drone validation.** The single-drone and commander + sub-agents suites are flown on real drones, and the results broadly follow simulation.

Results, replays and real-flight videos are on the [project page](https://naver-ai.github.io/DroneCATS/).

## Release Plan

The benchmark, the code and the model will be released soon. Stay tuned for updates!

## Citation

```bibtex
@article{park2026dronecats,
  title   = {Evaluating Multimodal LLMs as Generalist Vision-Language-Action Agents
             for Drone Control: Commanding, Approaching, Tracking and Searching},
  author  = {Park, Jaewoo and Lee, Minyoung and Seo, Sukmin and Yim, Moonbin and
             Yoon, Hyunwook and Ryu, Dohoon and Kim, Daehee and Song, Myungseo and
             Byun, Jihyuk and Chang, Seunggyu and Kil, Taeho and Kim, Jiseob and
             Lee, Bado and Kim, Geewook},
  journal = {arXiv preprint arXiv:2609.01404},
  year    = {2026}
}
```
