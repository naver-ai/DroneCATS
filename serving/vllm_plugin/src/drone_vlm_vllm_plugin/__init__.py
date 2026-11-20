# DroneCATS
# Copyright (c) 2026-present NAVER Cloud Corp.
# Apache-2.0

"""vLLM out-of-tree plugin for the DroneCATS Cosmos3-Edge 2B model, on stock ``vllm==0.25.1``.

Loaded by vLLM through the ``vllm.general_plugins`` entry point (``register`` below) in
every vLLM process. Nothing in site-packages is modified: the models live in this package's
own vendored modules and are added to vLLM's registry under the architecture names the
checkpoints' ``config.json`` use.

=====================================  ==========================================================
architecture (config.json)             plugin class
=====================================  ==========================================================
Cosmos3EdgeForConditionalGeneration    models.cosmos3_edge (imports the plugin's qwen3_vl /
                                       nemotron_h copies, never the stock ones)
=====================================  ==========================================================

The name is absent from stock 0.25.1, so the registration is purely additive: stock
``Qwen3VLForConditionalGeneration``, ``NemotronHForCausalLM`` ... keep resolving to
``vllm.model_executor.models.*``.

HF configs: ``cosmos3_edge`` is registered with ``transformers.AutoConfig`` (exist_ok) to the
vendored ``Cosmos3EdgeConfig`` -- the same class vLLM resolves for this model type;
it adds ``video_start/end_token_id`` over the transformers-native class.

Env:
  DRONE_VLM_PLUGIN_SELFCHECK  ``full`` (default): registration + behavioural checks (the
                              registration and behavioural checks);
                              ``registry``: registration checks only. Failures raise.
  COSMOS3_PATCH_ORDER_FIX     read by models.cosmos3_edge, default ``1`` (keep it on).
"""

from __future__ import annotations

import logging
import os

__version__ = "0.2.0"

EXPECTED_VLLM_VERSION = "0.25.1"
EXPECTED_TRANSFORMERS_VERSIONS = ("5.18.0", "5.18.0.dev0")  # PyPI, git a5d98c34; both GPU-validated

_PKG = __name__
ARCHITECTURES: dict[str, str] = {
    "Cosmos3EdgeForConditionalGeneration": (
        f"{_PKG}.models.cosmos3_edge:Cosmos3EdgeForConditionalGeneration"
    ),
}

logger = logging.getLogger("vllm." + _PKG)  # under the "vllm" tree so it reaches the server log
_registered = False


class PluginVersionError(RuntimeError):
    """Raised when the installed vLLM is not the release this plugin was ported to."""


def check_vllm_version() -> str:
    """Fail unless the installed vLLM is 0.25.1.

    A local build label (``0.25.1+cu129``) is the same release and is accepted.

    Returns:
        The installed ``vllm.__version__``.

    Raises:
        PluginVersionError: any other vLLM release.
    """
    import vllm

    found = getattr(vllm, "__version__", "unknown")
    if found.split("+", 1)[0] != EXPECTED_VLLM_VERSION:
        raise PluginVersionError(
            f"drone_vlm_vllm_plugin {__version__} supports vllm=={EXPECTED_VLLM_VERSION} only, "
            f"but vllm {found} is installed. The vendored model files (cosmos3_edge, qwen3_vl, "
            f"nemotron_h) are tied to 0.25.1 internals; "
            f"install vllm=={EXPECTED_VLLM_VERSION} or uninstall this plugin."
        )
    return found


def _register_hf_configs() -> None:
    from transformers import AutoConfig

    from .configs.cosmos3_edge import Cosmos3EdgeConfig

    AutoConfig.register("cosmos3_edge", Cosmos3EdgeConfig, exist_ok=True)


def _register_models() -> None:
    from vllm import ModelRegistry

    for arch, target in ARCHITECTURES.items():
        # Lazy "<module>:<class>" form: no CUDA / model import at plugin load.
        ModelRegistry.register_model(arch, target)


def register() -> None:
    """``vllm.general_plugins`` entry point. Idempotent within a process."""
    global _registered
    if _registered:
        return
    found = check_vllm_version()

    import transformers

    if transformers.__version__ not in EXPECTED_TRANSFORMERS_VERSIONS:
        logger.warning(
            "drone_vlm_vllm_plugin was validated with transformers %s; "
            "found %s. Cosmos3-Edge processor behaviour may differ.",
            " / ".join(EXPECTED_TRANSFORMERS_VERSIONS),
            transformers.__version__,
        )
    if os.environ.get("COSMOS3_PATCH_ORDER_FIX", "1") != "1":
        logger.warning(
            "COSMOS3_PATCH_ORDER_FIX=%s disables the Cosmos3-Edge patch-order fix: "
            "the 2B model will serve silently wrong vision features.",
            os.environ["COSMOS3_PATCH_ORDER_FIX"],
        )

    _register_hf_configs()
    _register_models()

    from . import selfcheck

    mode = os.environ.get("DRONE_VLM_PLUGIN_SELFCHECK", "full").strip().lower()
    if mode not in ("full", "registry"):
        raise ValueError(f"DRONE_VLM_PLUGIN_SELFCHECK must be 'full' or 'registry', got {mode!r}")
    selfcheck.run(full=(mode == "full"), raise_on_fail=True, verbose=False)
    _registered = True
    logger.info(
        "drone_vlm_vllm_plugin %s registered %s on vllm %s (selfcheck=%s)",
        __version__,
        ", ".join(ARCHITECTURES),
        found,
        mode,
    )
