# DroneCATS
# Copyright (c) 2026-present NAVER Cloud Corp.
# Apache-2.0

"""Load-time self-check of the plugin.

The checks assert *behaviour*, not that files were copied, because every failure mode they
guard against is a server that starts fine and serves subtly wrong prompts. They run
against the plugin's own modules:

* registration: each architecture resolves to the plugin module, the stock architectures
  still resolve to ``vllm.model_executor.models.*`` (stock left untouched), and
  ``AutoConfig`` maps ``cosmos3_edge`` to the vendored config class;
* isolation: the 2B model imports the plugin's ``qwen3_vl`` / ``nemotron_h`` copies;

``run(full=False)`` only does the registration part (no model-module import).
CLI: ``python -m drone_vlm_vllm_plugin.selfcheck`` prints every line and exits 1 on failure.
"""

from __future__ import annotations

import inspect
import logging
import sys
from types import SimpleNamespace

logger = logging.getLogger(__name__)

_PKG = __name__.rsplit(".", 1)[0]

# Stock 0.25.1 architectures that must keep resolving to stock modules.
STOCK_ARCHS = {
    "Qwen3VLForConditionalGeneration": "vllm.model_executor.models.qwen3_vl",
    "NemotronHForCausalLM": "vllm.model_executor.models.nemotron_h",
    "Cosmos3ForConditionalGeneration": "vllm.model_executor.models.cosmos3",
}


class SelfCheckError(RuntimeError):
    """At least one self-check assertion failed."""


class _Checker:
    def __init__(self, verbose: bool) -> None:
        self.verbose = verbose
        self.failed: list[str] = []
        self.passed = 0

    def chk(self, name: str, cond: bool, detail: str = "") -> None:
        line = f"   [{'PASS' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail else "")
        if self.verbose:
            print(line)
        if cond:
            self.passed += 1
        else:
            self.failed.append(name)
            logger.error("drone_vlm_vllm_plugin self-check %s", line.strip())

    def guard(self, name: str, fn) -> None:
        try:
            fn()
        except Exception as e:  # noqa: BLE001 - every exception is a failed check
            self.chk(name, False, repr(e))


# ------------------------------------------------------------------ registration
def _check_registration(c: _Checker) -> None:
    from vllm import ModelRegistry

    from . import ARCHITECTURES

    archs = ModelRegistry.get_supported_archs()
    for arch, target in ARCHITECTURES.items():
        mod, cls = target.split(":")
        entry = ModelRegistry.models.get(arch)
        got = (getattr(entry, "module_name", None), getattr(entry, "class_name", None))
        c.chk(f"{arch} -> plugin", arch in archs and got == (mod, cls), f"-> {got}")
    for arch, mod in STOCK_ARCHS.items():
        entry = ModelRegistry.models.get(arch)
        got = getattr(entry, "module_name", None)
        c.chk(f"stock {arch} untouched", got == mod, f"-> {got}")

    from transformers import AutoConfig
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING

    from .configs.cosmos3_edge import Cosmos3EdgeConfig

    c.chk(
        "AutoConfig cosmos3_edge -> plugin Cosmos3EdgeConfig",
        CONFIG_MAPPING["cosmos3_edge"] is Cosmos3EdgeConfig
        and type(AutoConfig.for_model("cosmos3_edge")) is Cosmos3EdgeConfig,
    )


# ------------------------------------------------------------------ 2B (cosmos3_edge)
def _check_cosmos3(c: _Checker) -> None:
    import torch

    from .models import cosmos3_edge as CE
    from .models import nemotron_h as NH
    from .models import qwen3_vl as QV

    # isolation: the 2B model is built on the plugin copies, not on stock modules
    c.chk(
        "cosmos3_edge uses plugin qwen3_vl / nemotron_h",
        CE.Qwen3VLForConditionalGeneration.__module__ == QV.__name__
        and CE.NemotronHModel.__module__ == NH.__name__,
        f"{CE.Qwen3VLForConditionalGeneration.__module__}, {CE.NemotronHModel.__module__}",
    )

    # (c) nemotron_h backport present (decoder_layer_types kwarg)
    c.chk("nemotron_h decoder_layer_types backport", "decoder_layer_types" in inspect.getsource(NH))

    # (d) THE PATCH-ORDER FIX -- assert the permutation, not just that the file exists
    def d() -> None:
        f = CE._blockmajor_to_raster
        ok_all = True
        for H, W, MS in [(4, 6, 2), (38, 54, 2)]:
            n = H * W
            x = torch.arange(n, dtype=torch.float32).unsqueeze(1).repeat(1, 3)
            got = f(x, torch.tensor([[H, W]]), MS)[:, 0].to(torch.int64).tolist()
            fwd = [
                (by * MS + dy) * W + (bx * MS + dx)
                for by in range(H // MS)
                for bx in range(W // MS)
                for dy in range(MS)
                for dx in range(MS)
            ]
            inv = [0] * n
            for k, r in enumerate(fwd):
                inv[r] = k
            ok_all &= got == inv
        c.chk("patch-order reorder == inverse(blockmajor->raster)", ok_all)
        n = 38 * 54
        x = torch.arange(n, dtype=torch.float32).unsqueeze(1).repeat(1, 3)
        got = f(x, torch.tensor([[38, 54]]), 2)[:, 0].to(torch.int64).tolist()
        moved = sum(1 for i, v in enumerate(got) if v != i)
        c.chk("reorder is non-trivial at 38x54", moved > 0.9 * n, f"{moved}/{n} patches move")

    c.guard("patch-order fix active", d)

    # (e) the env gate exists and defaults ON
    c.chk(
        "COSMOS3_PATCH_ORDER_FIX gate present (default on)",
        'COSMOS3_PATCH_ORDER_FIX", "1"' in inspect.getsource(CE).replace("'", '"'),
    )

    # (f) _video_wrapper_ids: reads the new fields, falls back for stock checkpoints
    def f_() -> None:
        VWI = QV._video_wrapper_ids
        stock = SimpleNamespace(vision_start_token_id=20, vision_end_token_id=21)
        ours = SimpleNamespace(
            vision_start_token_id=20, vision_end_token_id=21,
            video_start_token_id=22, video_end_token_id=23,
        )
        nulls = SimpleNamespace(
            vision_start_token_id=20, vision_end_token_id=21,
            video_start_token_id=None, video_end_token_id=None,
        )
        ok = VWI(stock) == (20, 21) and VWI(ours) == (22, 23) and VWI(nulls) == (20, 21)
        c.chk("video wrapper ids: read new fields, fall back otherwise", ok,
              f"stock={VWI(stock)} ours={VWI(ours)} nulls={VWI(nulls)}")

    c.guard("video wrapper ids helper", f_)

    # (g) the config class carries the additive fields
    def g() -> None:
        from .configs.cosmos3_edge import Cosmos3EdgeConfig as C3C

        d0 = C3C()
        dv = C3C(video_start_token_id=22, video_end_token_id=23)
        c.chk(
            "Cosmos3EdgeConfig video_start/end_token_id additive",
            d0.video_start_token_id is None
            and dv.video_start_token_id == 22
            and dv.video_end_token_id == 23,
        )

    c.guard("Cosmos3EdgeConfig video wrapper fields", g)

    # (h) no name-based media lookups left in the two hot paths
    def h() -> None:
        def code_only(fn) -> str:
            return "\n".join(ln.split("#", 1)[0] for ln in inspect.getsource(fn).splitlines())

        dt = code_only(QV.Qwen3VLDummyInputsBuilder.get_dummy_text)
        pu = code_only(QV.Qwen3VLMultiModalProcessor._get_prompt_updates)
        stock_names = ("<|image_pad|>", "<|video_pad|>", "<|vision_start|>", "<|vision_end|>")
        c.chk("get_dummy_text is id-derived",
              "_image_placeholder_str" in dt and not any(n in dt for n in stock_names))
        c.chk("_get_prompt_updates image target is a token id",
              "target=[image_token_id]" in pu and not any(n in pu for n in stock_names))

    c.guard("id-driven placeholder lookups", h)

    # (i) _iter_mm_grid_hw's VIDEO branch honours the separate video wrapper pair
    def i_() -> None:
        G = QV.Qwen3VLForConditionalGeneration

        class _E:
            def __init__(self, t):
                self.data = t

        feat = SimpleNamespace(
            modality="video",
            data={"video_grid_thw": _E(torch.tensor([2, 4, 4]))},
            mm_position=SimpleNamespace(offset=5),
        )
        toks = [0] * 5 + [22] + [18] * 4 + [23] + [22] + [18] * 4 + [23] + [0] * 3
        got = list(G._iter_mm_grid_hw(
            toks, [feat], video_token_id=18, vision_start_token_id=20,
            vision_end_token_id=21, spatial_merge_size=2,
            video_start_token_id=22, video_end_token_id=23))
        ok_new = len(got) == 2 and all(x[1] == 2 and x[2] == 2 and x[3] == 4 for x in got)
        try:
            list(G._iter_mm_grid_hw(
                toks, [feat], video_token_id=18, vision_start_token_id=20,
                vision_end_token_id=21, spatial_merge_size=2))
            ok_old = False  # must NOT find 20/21 in this stream
        except ValueError:
            ok_old = True
        c.chk("_iter_mm_grid_hw video branch uses video_start/end_token_id",
              ok_new and ok_old, f"frames={got}")

    c.guard("_iter_mm_grid_hw video branch", i_)

    # (j) the HF processor gets retagged from config ids
    def j() -> None:
        PI = CE.Cosmos3EdgeProcessingInfo
        names = {19: "<|IMAGE_PAD|>", 18: "<|VIDEO_PAD|>", 20: "<|image_start|>",
                 21: "<|image_end|>", 22: "<|video_start|>", 23: "<|video_end|>"}
        fake_tok = SimpleNamespace(convert_ids_to_tokens=lambda i: names[int(i)])
        fake_proc = SimpleNamespace(
            tokenizer=fake_tok, image_token="<|image_pad|>", video_token="<|video_pad|>",
            vision_start_token="<|vision_start|>", vision_end_token="<|vision_end|>")
        cfg = SimpleNamespace(image_token_id=19, video_token_id=18, vision_start_token_id=20,
                              vision_end_token_id=21, video_start_token_id=22,
                              video_end_token_id=23)
        PI._retag_media_tokens(SimpleNamespace(get_hf_config=lambda: cfg), fake_proc)
        c.chk(
            "HF processor retagged from config ids",
            fake_proc.image_token == "<|IMAGE_PAD|>"
            and fake_proc.image_token_id == 19
            and fake_proc.video_token == "<|VIDEO_PAD|>"
            and fake_proc.vision_start_token == "<|video_start|>"
            and fake_proc.vision_start_token_id == 22
            and fake_proc.vision_end_token_id == 23,
            f"image_token={fake_proc.image_token} vision_start_token={fake_proc.vision_start_token}",
        )

    c.guard("HF processor retag", j)

    # processor registered on the plugin class (not on a stock class)
    pf = CE.Cosmos3EdgeForConditionalGeneration.__dict__.get("_processor_factory")
    c.chk("Cosmos3Edge multimodal processor registered on plugin class",
          pf is not None and pf.processor is CE.Cosmos3EdgeMultiModalProcessor)


def run(full: bool = True, raise_on_fail: bool = True, verbose: bool = False) -> list[str]:
    """Run the checks.

    Args:
        full: also run the behavioural checks (imports the plugin model modules).
        raise_on_fail: raise ``SelfCheckError`` listing the failed checks.
        verbose: print one line per check.

    Returns:
        Names of the failed checks (empty on success).
    """
    c = _Checker(verbose)
    c.guard("registration", lambda: _check_registration(c))
    if full:
        c.guard("cosmos3_edge checks", lambda: _check_cosmos3(c))
    if c.failed and raise_on_fail:
        raise SelfCheckError(
            "drone_vlm_vllm_plugin SELF-CHECK FAILED: " + ", ".join(c.failed)
            + "  (run `python -m drone_vlm_vllm_plugin.selfcheck` for details)")
    if verbose:
        print()
        print("SELF-CHECK FAILED: " + ", ".join(c.failed) if c.failed
              else f"SELF-CHECK PASSED ({c.passed} checks, plugin registration and behaviour verified)")
    return c.failed


def main() -> int:
    import vllm  # noqa: F401  (loads general plugins lazily below)

    from . import register

    register()  # idempotent; runs the silent check once
    return 1 if run(full=True, raise_on_fail=False, verbose=True) else 0


if __name__ == "__main__":
    sys.exit(main())
