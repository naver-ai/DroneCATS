# vlmpilot

VLM closed-loop drone navigation runner (AirSim, Isaac Sim + PX4, DJI Tello). See the repository root README for usage.

```bash
uv sync
bash ../scripts/airsim/launch_airsim.sh nh 46001      # simulator on RPC port 46001 (as root)
export OPENROUTER_API_KEY=...                         # your own key
uv run python -m vlmpilot airsim --config configs/airsim/single-drone/airsim_gpt6luna_or.yaml --port 46001
```
