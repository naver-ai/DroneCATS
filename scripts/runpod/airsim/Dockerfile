# Draft image for RunPod pods — base for the AirSimNH (UE4) benchmark stack.
#
# Not required: running scripts/runpod/airsim/setup.sh on any CUDA ubuntu image reaches the same
# state. This image pre-bakes the apt/user/uv steps to speed up pod boot.
#
# Build/use:
#   docker build -t drone-bench:latest scripts/runpod/airsim/
#   → push to a registry, then set it as the RunPod custom image.
#   When creating the pod, always set: env NVIDIA_DRIVER_CAPABILITIES=all (also baked in via ENV, but
#   depending on RunPod's injection path the pod env may be needed) + Network Volume → /workspace.
#
# Base: runpod/pytorch images also work, but torch is unnecessary, so keep it minimal with nvidia/cuda runtime.
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    NVIDIA_DRIVER_CAPABILITIES=all

# UE4.27 offscreen Vulkan runtime + tools used by the setup/launcher scripts
# (no X server needed, but UE4 links X client libraries, so libsm6/libx* are required)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libvulkan1 vulkan-tools \
        libsm6 libxext6 libxrandr2 libxcursor1 libxi6 \
        curl ca-certificates unzip git python3 iproute2 procps tmux ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# UE4 refuses to run as root → runuser as the airsim user (same as scripts/airsim/launch_airsim.sh)
RUN useradd -m -s /bin/bash airsim

# uv — vlmpilot dependency manager (Python 3.13 is installed automatically on uv sync)
RUN curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh

# The AirSimNH binary (~5.4GB extracted) is not baked into the image — it lives on the Network Volume
# (/workspace), and scripts/runpod/airsim/setup.sh downloads it and creates the /opt/airsim_envs symlink.
# (keeps the image small + reuses the volume when the pod is recreated)

# No ENTRYPOINT — interactive pod. Only a CMD so RunPod keeps the container alive
# (attach via web terminal/SSH and run scripts/runpod/airsim/setup.sh).
CMD ["sleep", "infinity"]
