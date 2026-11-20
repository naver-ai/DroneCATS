# Running the benchmark on RunPod

One RunPod GPU pod is enough for the whole benchmark. The first setup takes about 6 minutes on an empty volume; later pods are ready in 1–2 minutes.

## 0. What you need

| Item | Where | Note |
|---|---|---|
| RunPod account with at least $5 credit | runpod.io → Billing | Network volumes cannot be created below $5 |
| SSH public key | RunPod → Settings → SSH Public Keys | e.g. `~/.ssh/id_ed25519.pub` |
| Model API keys | OpenRouter, Google AI Studio | Only for the models you run. Commander + Robotics-ER sub-agents (the paper condition) needs both: the commander goes through OpenRouter, the four sub-agents are Google's Gemini Robotics-ER (`gemini-robotics-er-2-preview`), which is not on OpenRouter. The OpenRouter-only variant below needs only the OpenRouter key. Save them as RunPod Secrets: console left menu → **Secrets** (on newer consoles: **Credentials** → Secrets tab) |

## 1. Network volume (once)

RunPod → Storage → New Network Volume
- Data center: one with RTX 4090 stock (we used **EU-RO-1**)
- Size: **20 GB** for one model. Code and the three maps take about 9 GB; each full model run (80 single + 60 multi episodes) adds 3–4 GB of outputs. Add about 4 GB for every further run you keep on the volume. A full volume stops runs silently.

## 2. Pod

Deploy the pod **from the volume** (Storage → your volume → Deploy) so it is created in the volume's data center. A volume only attaches to pods in its own data center; a pod deployed elsewhere starts without it, everything then installs on the container disk and is lost when the pod stops. `setup.sh` warns when `/workspace` is not a network volume.

- GPU: **RTX 4090**, Secure Cloud
- Network volume: the one above, mounted at `/workspace`
- Image: `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`, container disk 30 GB
- Environment variables:
  - `NVIDIA_DRIVER_CAPABILITIES` = `all` (**required** — without it the simulator cannot use the GPU)
  - `OPENROUTER_API_KEY` = `{{ RUNPOD_SECRET_openrouter_api_key }}` and/or `GEMINI_API_KEY` = `{{ RUNPOD_SECRET_gemini_api_key }}` (names of your Secrets)
- Expose TCP port `22`

## 3. Connect

Use **SSH over exposed TCP** from the pod's Connect panel: `ssh root@<IP> -p <PORT> -i ~/.ssh/id_ed25519`.
The default `ssh.runpod.io` proxy cannot run remote commands or scp.

## 4. Code and setup

```bash
cd /workspace && git clone <this repo> DroneCATS && cd DroneCATS
bash scripts/runpod/airsim/setup.sh
```

- If the pod cannot reach the repository (a private mirror, for example), clone it on your own machine and copy it over instead of `git clone`:
  `rsync -rlptz --no-owner --no-group -e "ssh -p <PORT> -i ~/.ssh/id_ed25519" --exclude .git DroneCATS root@<IP>:/workspace/`
  (the network volume does not allow changing file owners, so do not preserve them; for a tarball use `tar xzf <file> --no-same-owner`)
- First run downloads the three maps (NH, campus, blocks; about 3 GB) — about 5–6 minutes in total. Later pods reuse the volume: 1–2 minutes.
- The Python environment is created on the pod's local disk. On the network volume, imports take about 50 s per episode and eat into the 300 s episode limit.
- **Do not put API keys on the volume** (`/workspace` survives pod stops). Use Secrets as in step 2, or `export OPENROUTER_API_KEY=...` in your shell (gone when the pod stops). The setup script warns if it finds a `.env` on the volume.

## 5. Run

### Whole benchmark for one model (parallel)

```bash
export OPENROUTER_API_KEY=...        # skip if set through Secrets; commander + Gemini Robotics-ER also needs GEMINI_API_KEY (Google AI Studio)
MODEL=gpt6-luna-or AGENTS_CFG=configs/airsim/multi-drone-with-subagents/airsim_agents_hluna_robotics_low_h40.yaml bash scripts/runpod/airsim/run_all.sh
python3 scripts/runpod/airsim/summarize.py       # success rate per set and level for the newest run (also printed when run_all.sh ends)
```

- Runs the single-drone part (8 sets × 10 episodes, 10 simulators), then multi-drone with the model driving all four drones, then commander + Robotics-ER sub-agents (30 episodes each, 6 simulators). About 2.2 hours on one RTX 4090. See Cost below: with Robotics-ER sub-agents the commander part costs $15–43 in Google API calls.
- Long runs: start them detached and follow the log, so a dropped SSH connection does not stop the run:
  `(MODEL=gpt6-luna-or PARTS=single setsid nohup bash scripts/runpod/airsim/run_all.sh > /workspace/run.log 2>&1 < /dev/null &)` then `tail -f /workspace/run.log`.
- `PARTS=single` runs only the single-drone part (about 55 min); `SETS="nh_approaching nh_searching"` limits the single-drone sets; `PARTS="single multi_self"` skips the commander part (no Gemini key needed).
- Episodes skipped because a simulator crashed are rerun automatically, one fresh simulator each.
- Before starting, `run_all.sh` checks that every API key the chosen parts need is set and stops if one is missing (e.g. `GEMINI_API_KEY` for commander + Robotics-ER). It also sends one free request per key and stops if the provider rejects the key or an OpenRouter key has no credit left, since every model call would fail otherwise (`SKIP_KEY_CHECK=1` skips this).

### Local model (DroneCATS-2B)

[DroneCATS-Cosmos3-Edge-2B](https://huggingface.co/naver-ai/DroneCATS-Cosmos3-Edge-2B) runs as `MODEL=dronecats-2b` (config `vlmpilot/configs/airsim/single-drone/airsim_dronecats2b.yaml`). The runner expects an OpenAI-compatible server on the pod at `http://127.0.0.1:18000/v1` with `--served-model-name drone-vlm-2b-cosmos3edge`. No API key is needed for a local endpoint.

For the model itself (installation, input format, camera information) see the [model card](https://huggingface.co/naver-ai/DroneCATS-Cosmos3-Edge-2B); this section covers only running it in the benchmark.

Serve it with vLLM on the same pod as described in [`serving/README.md`](../../../serving/README.md) (`serving/setup.sh` once, about 8 min on an RTX 4090; then `serving/serve.sh`). Once the server is up:

```bash
GPU_UTIL=0.45 setsid nohup bash serving/serve.sh > vllm.log 2>&1 < /dev/null &
until curl -sf 127.0.0.1:18000/health; do sleep 5; done     # about 90 s after the server starts
MODEL=dronecats-2b PARTS=single SIMS=8 bash scripts/runpod/airsim/run_all.sh
```

- Single-drone part only: the model was trained on one image per step, so it has no multi-drone config.
- Model and simulators on one RTX 4090 (24 GB): 10 simulators take about 13.7 GB. Start vLLM with `--gpu-memory-utilization 0.45` and use `SIMS=8` (22.3 GB in total). At 0.30 vLLM does not start: the KV cache is too small for the 16384-token context.
- On one GPU the simulators and the model compete, so step latency depends on the simulator load. **For final numbers use two RTX 4090s**: one for 8 simulators, one only for vLLM. The 80 single-drone episodes then cost under $3 in pod time.

### OpenRouter only (one key)

The paper's commander runs use Gemini Robotics-ER for the four sub-agents. Robotics-ER is served only by Google, so that part needs a Google AI Studio key and costs $15–43 per 30 episodes, depending on the commander. With a single OpenRouter key, run the commander part with the same model for the commander and the sub-agents:

```bash
MODEL=gpt6-luna-or AGENTS_CFG=configs/airsim/multi-drone-with-subagents/airsim_agents_hluna_bluna_h40.yaml bash scripts/runpod/airsim/run_all.sh
MODEL=gemini38 AGENTS_CFG=configs/airsim/multi-drone-with-subagents/airsim_agents_hg38f_bg38f_h40.yaml bash scripts/runpod/airsim/run_all.sh
```

- Only `OPENROUTER_API_KEY` is needed. Each model is then measured end to end on its own: single drone, multi-drone driving all four, and commander + four sub-agents.
- The commander thinks; the sub-agents run at the lowest reasoning, as Robotics-ER does in the paper runs (thinking budget 0).
- Otherwise the sub-agents send the same request as the model's multi-drone config (`airsim_multi_gpt6luna_or.yaml`, `airsim_multi_gemini38_or.yaml`). Those configs also run at the lowest reasoning (Luna off, Gemini 3.8 Flash minimal, since OpenRouter cannot turn it off), so the two multi-drone parts differ only in whether there is a commander.
- **This is not the paper condition.** Report it separately from commander + Robotics-ER numbers.

### Where results go

| Run | Output directory |
|---|---|
| `run_all.sh` | `/workspace/results/<MODEL>-<time>/` (`single/`, `multi_self/`, `multi_agents/`) |
| `verifier.airsim.run_testset` by hand | `vlmpilot/outputs/verifier/<time>/<model>/` |
| `verifier.airsim.run_bench_tasks` by hand | `vlmpilot/outputs/bench/<time>/<task>/run1/` |

`run_all.sh` also writes `/workspace/results/<MODEL>-<time>.tar.gz`: the same directory without camera frames (about 2 MB for 80 episodes). The network volume opens only on pods in its own data center, so copy this one file off the pod (`scp -P <PORT> root@<IP>:/workspace/results/<run>.tar.gz .`) before you stop it.

Each single-drone set has `summary.json` (one row per episode: `success`, `min_dist_m`, `steps`, `wall_s`) plus per-episode `verdict_*.json` and `grid_*.jpg` (camera frames). Each multi-drone episode has `verdict.json` and one `trace_<drone>.jsonl` per drone.
`summarize.py` prints, per single-drone set, the success rate, median steps per episode, and median step latency (`vlm_ms` of `decision_<ts>.json`, the model call alone), for comparing models.

Saved camera frames (`decision_<ts>.jpg`, one per step):
- They are saved at the size every model receives (width 640, JPEG quality 85; about 60 KB instead of about 440 KB for the 1280×720 capture). `VLMPILOT_SAVE_SENT_SIZE=0` or config `save_frames_at_sent_size: false` keeps the full-size capture.
- `pixel_u` / `pixel_v` in `decision_<ts>.json` are in capture pixels (1280 wide). To draw them on the saved jpg, multiply by 640/1280.
- By default the chosen point is drawn into the frame (`burn_overlay`, json `frame_clean: false`). To re-evaluate other models offline on the same frames, run with `burn_overlay: false` in the config (clean frame, no markers).

### One set by hand

```bash
bash scripts/airsim/launch_airsim.sh nh 46001                  # campus / blocks work the same way
PORT=46001 MODELS=gpt6-luna-or vlmpilot/.venv/bin/python -m verifier.airsim.run_testset episodes/airsim/single-drone/nh_approaching.json
```

Run several simulators on different ports with one runner per port. **8–10 simulators per RTX 4090** (`run_all.sh` uses 10); from 12 the GPU saturates and camera capture exceeds 1 s.

Multi-drone (start the simulator with the four-vehicle settings):

```bash
SETTINGS_BASE=vlmpilot/settings.multidrone.json.example bash scripts/airsim/launch_airsim.sh blocks 46101
PORT=46101 vlmpilot/.venv/bin/python -m verifier.airsim.run_bench_tasks episodes/airsim/multi-drone/blocks_multi-drone-hard.json \
    --config configs/airsim/multi-drone-with-subagents/airsim_agents_hluna_robotics_low_h40.yaml --runs 1
```

Known issues
- GPT-6 Luna needs OpenAI strict JSON schemas; its configs set `strict_schema: true`.
- Blocks + commander + Robotics-ER sometimes crashes the simulator after a few episodes (`pure virtual method called`). The runner logs `ECONNREFUSED` and skips the rest; `run_all.sh` reruns those episodes with a fresh simulator each.
- Gemini 3.8 Flash through OpenRouter cannot turn reasoning off (400 "Reasoning is mandatory"); its configs use `effort: minimal`. With the default thinking it took about 11 s per step and most episodes hit the 300 s limit.

## 6. When done

Pod → **Stop**. GPU billing stops; only the volume is billed (about $0.07/GB per month, about $1.4 for 20 GB).
Stopped pods restart only if the same GPU is free, so creating a new pod is usually faster.

## Cost

Measured on one RTX 4090 at $0.89/h. All figures are approximate. Pod time is about the same for every model:

| Part | Episodes | Simulators | Time | Pod |
|---|---|---|---|---|
| Single drone | 80 | 10 | 55 min | $0.8 |
| Multi-drone, the model drives all four | 30 | 6 | 35–40 min | $0.6 |
| Multi-drone, commander + four sub-agents | 30 | 6 | 35–40 min | $0.6 |
| Whole `run_all.sh` | 140 | | 2.2 h | $2.0 |

A multi-drone simulator renders four cameras, so it loads the GPU about as much as two single-drone simulators; 6 of them is close to the 10–12 single-drone simulators an RTX 4090 holds.

API cost is billed separately by each provider and depends on the model; check the provider's usage page after a run. The one large item is the Robotics-ER sub-agents (Google): $15–43 per 30 episodes in our runs, depending on the commander. `PARTS="single multi_self"` skips the commander part; the same-model variant needs only the OpenRouter key (see "OpenRouter only" above).
