# Isaac Sim benchmark on one RunPod GPU

The outdoor DroneCATS benchmark in Isaac Sim: 28 closed-loop episodes in the NVIDIA Brownstone demo scene. They cover static targets and moving targets (cars, pedestrians), and approaching, searching, tracking and search & track tasks. The drone is a PX4 SITL quadrotor in [Pegasus Simulator](https://github.com/PegasusSimulator/PegasusSimulator), and the VLM pilot is `vlmpilot` in `pegasus` mode.

> ⚠️ **Results depend on the hardware.** The simulation keeps running while the model thinks. How much sim time passes per decision therefore depends on how fast the machine simulates (RTF = sim seconds per wall second). Moving targets keep walking or driving during that time, so moving-target episodes change with RTF. With the same model we saw pedestrians 8/8 at RTF 0.10 and 0/7 at RTF 0.48. **Report the RTF with every result.** `compare.py` does this for you.

## What you need

| | |
|---|---|
| GPU | **RTX 4090** (24 GB) or another RTX card. Isaac Sim needs RT cores, so A100/H100 will not work |
| Host driver | **CUDA 13.0 or 13.1 hosts (driver 580/590).** Isaac Sim 5.1 crashes on start on driver 595 (CUDA 13.2) hosts |
| CPU / RAM | 12+ vCPU, 40+ GB RAM. Physics runs on the CPU; with 6 vCPU the first scene load took over 13 min |
| Disk | Container disk 120 GB (installs) + volume disk 20 GB at `/workspace` (results, kept when the pod stops) |
| Image | `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` |
| Ports | `22/tcp` (SSH) |
| Env vars | `ACCEPT_EULA=Y`, `OMNI_KIT_ACCEPT_EULA=YES` (only after you read and accept the [Isaac Sim / Omniverse license](https://docs.isaacsim.omniverse.nvidia.com/latest/common/NVIDIA_Omniverse_License_Agreement.html)), `NVIDIA_DRIVER_CAPABILITIES=all`, and your model key from a RunPod Secret, e.g. `OPENROUTER_API_KEY={{ RUNPOD_SECRET_openrouter_api_key }}` |
| Cost | RTX 4090 Secure Cloud was $0.89/h (Oct 2026) |

As of October 2026 such hosts are often not available on RunPod Secure Cloud (most RTX 4090 hosts run driver 595); if none can be found, use another provider or your own machine with driver 580/590.

In the RunPod console: **Deploy** → RTX 4090 → *Additional filters* → CUDA versions **13.0, 13.1** → template image above → set the disks, ports and env vars above. With `runpodctl` or the API, pass `allowedCudaVersions: ["13.0","13.1"]`.

## Isaac Sim 6 (experimental)

`ISAAC_VERSION=6.0.1 bash scripts/runpod/isaacsim/setup.sh` installs Isaac Sim 6.0.1 and a community Pegasus port (`iftahnaf/PegasusSimulator@fcb99c0`); 5.1.0 stays the default.

- Checked on an RTX 4090 pod with 40+ GB RAM on a **driver 595** host, where 5.1 does not start. Check `nvidia-smi` before running.
- Setup takes 30-50 min, depending on the download server (measured: 51 min, with the 13 GB Isaac Sim zip at 4 MB/s).
- The `hello` step may log `Fatal Python error: Aborted` in `/root/setup_logs/hello.log` after `HELLO_OK`, when Isaac Sim 6 shuts down. This is known and harmless; the step passes on `HELLO_OK` alone.
- Two speed-ups are off by default: `PEGASUS_BATCHED_FORCES=1` for `setup.sh` applies `patches/pegasus6_multirotor_batched_forces.patch` to the port (one force call per physics step instead of 6; on a finished setup, delete `/root/setup_logs/pegasus.done` and re-run), and `GT_OCCLUSION_VECTORIZED=1` for `run.sh` uses numpy in the GT occlusion check.
- They compute the same values, but a faster simulator (higher RTF) can change moving-target episodes such as pedestrian tracking (see the warning at the top). Oracle on the p28 batch: RTF 0.10 → 4/4, 0.15 → 3/4, 0.23 (both on) → 2/4; the failures were pedestrian tracking. They stay off for reproducibility; if you turn them on, report the RTF.

## Run

```bash
git clone <this repo> /workspace/DroneCATS && cd /workspace/DroneCATS
bash scripts/runpod/isaacsim/setup.sh                       # ~10 min (6.0.1: 30-50 min); ends with SETUP_OK
# key: either a RunPod Secret in the pod env (recommended), or type it without echo:
#   read -s OPENROUTER_API_KEY && export OPENROUTER_API_KEY
setsid nohup bash scripts/runpod/isaacsim/run.sh gemini37 vlmpilot/configs/isaacsim/single-drone/eval_px4_gemini37-or.yaml \
    </dev/null >/dev/null 2>&1 &
tail -f /workspace/results/isaac/gemini37/run.log          # ▶ batch start, ■ batch done, ALL DONE at the end
```

- **Automatic stop:** `run.sh` starts `watchdog.sh`, which stops the pod when the run ends, after 1 h without progress, or 12 h after start (`HARD_CAP_UTC`). Results stay on the volume disk at `/workspace/results/isaac/<name>/` plus `<name>.tgz`. Start the pod again to copy them off, then terminate it. Set `AUTOSTOP=0` to keep the pod running.
  - ⚠️ **Only with a volume disk.** A pod stop wipes the container disk. Without a volume at `/workspace` (or `OUT_ROOT` on one), the automatic stop deletes the results; `run.sh` then writes a warning to `run.log`. Run such a pod with `AUTOSTOP=0`, copy the results off and stop it yourself.
- **Smoke test:** `LIMIT=2 bash scripts/runpod/isaacsim/run.sh smoke <config> p28` runs only the first 2 episodes of each batch given.
- **Faster with two pods:** run `... keepA` on one pod and `... keepB topup p28` on the other. Each half has 14 episodes. Pass both result folders to `compare.py`.
- **Dry run:** `DRY_RUN=1 bash scripts/runpod/isaacsim/run.sh x vlmpilot/configs/isaacsim/single-drone/eval_px4_gemini37-or.yaml` prints the steps without starting or writing anything.

## Compare

```bash
python3 scripts/runpod/isaacsim/compare.py --results /workspace/results/isaac/gemini37 --md compare.md
```

This prints success per group (static / vehicle / pedestrian) and per episode next to a reference run, with both RTFs and the caveat above. Two reference runs of Gemini 3.7 Flash ship in `episodes/isaacsim/single-drone/`:

| Reference | Hardware | RTF | Model API | SR |
|---|---|---|---|---|
| `reference_gemini37_rtf010.jsonl` (default) | L40S server, 2–4 simulators per server | 0.10 | Google Gemini API | 21/28 |
| `reference_gemini37_rtf048.jsonl` | RunPod RTX 4090, one simulator per pod | 0.48 | OpenRouter | 12/24 (4 not measured) |

Pass `--baseline episodes/isaacsim/single-drone/reference_gemini37_rtf048.jsonl --baseline-rtf 0.48` to compare against the second one.

## Models

Configs live in `vlmpilot/configs/isaacsim/single-drone/`. Every model uses the same control settings as `pegasus_oracle.yaml`, including `continuous_control: false`.

| Model | Config | Key |
|---|---|---|
| Gemini 3.7 Flash (OpenRouter) | `eval_px4_gemini37-or.yaml` | `OPENROUTER_API_KEY` |
| Gemini 3.7 Flash (Google API) | `eval_px4_gemini37.yaml` | `GEMINI_API_KEY` |
| Gemini 3.8 Flash (OpenRouter) | `eval_px4_gemini38-or.yaml` | `OPENROUTER_API_KEY` |
| GPT-6 Luna (OpenRouter) | `eval_px4_gpt6luna-or.yaml` | `OPENROUTER_API_KEY` |
| Claude Opus 5 (OpenRouter) | `eval_px4_claude-opus5.yaml` | `OPENROUTER_API_KEY` |
| Qwen3.5 9B / 27B (OpenRouter) | `eval_px4_qwen35-9b.yaml` / `-27b.yaml` | `OPENROUTER_API_KEY` |

Keys are read from environment variables only. Never write them into the repo, the volume or a config file.

## Time and cost (measured, RTX 4090, Gemini 3.7 Flash via OpenRouter)

| | |
|---|---|
| Setup | ~10 min per pod |
| Scene load per batch | ~250 s the first time (12 vCPU), faster afterwards |
| Wall time per episode | median ~9 min; a few run 25+ min |
| 28 episodes, two pods in parallel | ~5.5 h, pod cost ~$8.6 |
| 28 episodes, one pod | ~9–10 h (estimate), ~$8–9 |

These times were measured with `SETTLE_S=150`; the default 20 s saves about 2 min per episode (~1 h per 28 episodes). Model API cost is billed by the provider.

## How it works

- **Two processes.** The simulator (Isaac Sim + Pegasus + PX4, physics 250 Hz on the CPU, lockstep with PX4) writes camera frames to a file. The pilot (`vlmpilot pegasus`) reads a frame, asks the model, and sends a velocity command over MAVLink. With `continuous_control: false` each command runs 2.5 s of *sim* time before the next frame, so the drone moves the same distance whatever the RTF. The simulation is not paused while the model thinks.
- **Per episode** (`verifier/isaacsim/batch_run.py`): reset the drone and restart PX4 → wait for PX4's port and heartbeat, plus a margin (`SETTLE_S`, 20 s; the reference runs used 150 s, which an A/B showed makes no difference to arming) → fly → join ground truth → judge. **Success** means the model declared arrival (`finished`) while at most 5 m from the target, with the target visible (≥ 50 px of its mask in the drone camera). The flight ends at that declaration. There is no step limit; see Known issues.
- **Targets:**
  - Static props stay put.
  - Cars move at 0.3 m/s.
  - Pedestrians walk at their animation's own speed, about 1.0 m/s. `--target-speed` does not apply to them, because the animation graph owns their root motion.
  - Movers wait until the drone has taken off (`--mover-takeoff-gate-m`).
- **Before each batch** `batch_run.py` waits until the simulator writes fresh ground truth (`--wait-a-ready-s`, give-up limit 900 s) and until the targets have left their parking spot (`--wait-placed-s`, 300 s). The simulator starts writing ground truth before it places the targets; an episode started in that gap would chase a target 7 km away.
- **`--gt-mode measure`** renders ground truth only within 10 m of the target. Judging needs it only within 5 m, so outcomes are unchanged, and the simulation runs about 2× faster.

## Known issues

- **Arm failure after a PX4 restart** (`armable=False`, PX4 log `Preflight Fail: ekf2 missing data`): 3 of 28 episodes in one run. `RETRIES=2` re-tries them. Hosts with few vCPUs make it more likely. A longer `SETTLE_S` is not the lever: in a 2026-10-21 A/B of 45 resets (20 · 30 · 150 s, 15 each) every reset armed and mean connect-to-arm was 15.6-16.3 s whatever the value (individual runs 13.3-19.2 s) — the failure did not reproduce at any setting. The margin was 150 s until 2026-10-21 and is 20 s now, which removes about an hour of idle per pod.
- **No step limit:** neither the pilot nor the judge caps the number of steps. A flight ends when the model declares arrival, or when it has searched for 24 steps in a row without finding the target (`no_finish`, a failure). Otherwise only the 3,000 s wall limit per episode (`--b-timeout`) stops it. A `timeout` is retried like the other transient failures (`RETRIES`); if every attempt times out, the episode is reported as not measured and left out of SR (`compare.py` lists such episodes).
- **Assets are not redistributed:**
  - The Brownstone scene, vMaterials and the pedestrians are downloaded from NVIDIA's public servers.
  - The car is the CC-BY-4.0 *CarConcept* sample (Eric Chadwick / Darmstadt Graphics Group, Khronos glTF-Sample-Assets). `setup.sh` converts it to USD.
