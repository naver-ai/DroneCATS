# Serving DroneCATS-Cosmos3-Edge-2B with vLLM

This folder serves [DroneCATS-Cosmos3-Edge-2B](https://huggingface.co/naver-ai/DroneCATS-Cosmos3-Edge-2B)
behind an OpenAI-compatible API, so the benchmark in this repository (`MODEL=dronecats-2b`) and your own
code can call it like any hosted model. The model card shows the same model through `transformers`.

| Path | What it is |
|---|---|
| `vllm_plugin/` | vLLM 0.25.1 out-of-tree plugin that adds the `Cosmos3EdgeForConditionalGeneration` architecture and fills the camera information into the prompt. Installed with `pip`, nothing in vLLM is modified. |
| `setup.sh` | Creates the Python environment, installs the pinned requirements and the plugin, runs the plugin self-check and downloads the model. |
| `serve.sh` | Starts the server on `http://127.0.0.1:18000/v1`. |
| `example_request.py` | Sends one frame and a command, prints the action JSON. |

## Requirements

- Linux with an NVIDIA GPU, Python 3.12. Tested on an A100 80 GB (driver 535) and an RTX 4090 (driver 595).
- About 22 GB of local disk for the environment and the model. Put the environment on local disk;
  imports are slow from a network volume.
- Versions are pinned: `vllm==0.25.1`, `transformers==5.18.0` (see `vllm_plugin/requirements_server.txt`).
  The plugin refuses to load on any other vLLM release.

## Install

```bash
bash serving/setup.sh
```

It prints `SELF-CHECK PASSED` when the plugin is registered and behaves as expected. Set `VENV`,
`HF_HOME` or `MODEL_ID` to change the environment location, the model cache or the model repository.

## Serve

```bash
GPU_UTIL=0.45 setsid nohup bash serving/serve.sh > vllm.log 2>&1 < /dev/null &
until curl -sf 127.0.0.1:18000/health; do sleep 5; done      # about 60 s
```

`GPU_UTIL` is vLLM's `--gpu-memory-utilization`. The default 0.45 leaves room for a simulator on the
same GPU. A value that is too low fails to start, because the KV cache must hold the 16,384-token
context (vLLM then reports "estimated maximum model length is ..."; this depends on the GPU memory). For a final measurement, use one GPU for the simulator and
another for the server.

Flags that matter, all set in `serve.sh`: `--tokenizer-mode hf` and `--chat-template-content-format openai`
(the model's own tokenizer and chat template), `--enforce-eager`, `--max-model-len 16384`, and the
environment variables `COSMOS3_PATCH_ORDER_FIX=1`, `VLLM_VIDEO_LOADER_BACKEND=opencv`,
`VLLM_USE_FLASHINFER_SAMPLER=0`.

## Send a request

```bash
"$VENV/bin/python" serving/example_request.py --image frame.jpg --instruction "go to the egg" --hfov 56.4
# {"action": "go", "point": [402, 656], "depth_m": 3.9}
```

The prompt and the camera input follow the model card (decoding differs: the model card example is greedy):

- Messages: a system message and a user message with the command and one image. Use the files in the
  model repository's `prompts/` folder as they are; the model was trained on exactly this wording.
- Image: JPEG, at most 640 px wide, quality 85.
- Camera information: the focal length goes into the JPEG EXIF block. With `k = fx / width`, FocalLength
  is `36 * k` mm and FocalPlaneX/YResolution is `10 * width / 36` pixels per cm (unit: cm).
  `example_request.py` shows the code. The plugin turns this into the `<|camera_aux|>` JSON that the
  `transformers` example passes as `camera_aux`, so both paths give the model the same input. A frame
  without this EXIF block reaches the model without camera information.
- `$VENV` is the environment from `setup.sh` (default `~/dronevllm`); the script needs `Pillow` and `huggingface_hub`, which it has.
- Example frame: `examples/tello_egg_mid.jpg` in the model repository (the example above uses it). `setup.sh` caches the
  repository under `~/hf` (`HF_HOME`), so the file is at
  `~/hf/hub/models--naver-ai--DroneCATS-Cosmos3-Edge-2B/snapshots/<revision>/examples/tello_egg_mid.jpg`. To get the path,
  set the same `HF_HOME`; without it `huggingface_hub` looks in `~/.cache/huggingface` and finds nothing:
  `IMG=$(HF_HOME=~/hf "$VENV/bin/python" -c "from huggingface_hub import hf_hub_download as d; print(d('naver-ai/DroneCATS-Cosmos3-Edge-2B', 'examples/tello_egg_mid.jpg'))")`
- Response: `response_format` JSON schema `vlmpilot_action`, `temperature` 0.4,
  `chat_template_kwargs {"enable_thinking": false, "skip_reasoning": true}`.

At startup vLLM may log `2 camera_aux placeholders vs 1 images` once. This is harmless.

The server receives a JPEG, while the `transformers` example in the model card feeds raw pixels, so the
two can differ slightly. On the model card's example frame the server answers `depth_m` 3.9, as
`transformers` does when it is fed the same JPEG, and 3.2 for the raw pixels.

## License

The plugin's `models/qwen3_vl.py`, `models/nemotron_h.py` and `models/cosmos3_edge.py` are adapted from
vLLM and the Qwen team's code and keep their Apache-2.0 notices. Everything else follows the license of
this repository (see the root `README.md`).
