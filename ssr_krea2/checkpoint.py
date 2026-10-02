"""Writing the merged LoRA and baking it into a (ComfyUI fp8-scaled or plain) Krea 2 checkpoint."""

from __future__ import annotations

from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from torch import Tensor

from .krea2 import dequantize_checkpoint_weight

FP8 = torch.float8_e4m3fn
FP8_MAX = torch.finfo(FP8).max


def save_lora(merged: dict[str, tuple[str, Tensor, Tensor]], path: str | Path, metadata: dict[str, str]) -> None:
    """Save in the same diffusers/PEFT key layout as the inputs (no alpha key, i.e. scale 1)."""
    sd = {}
    for out_name, A, B in merged.values():
        sd[out_name + ".lora_A.weight"] = A.contiguous()
        sd[out_name + ".lora_B.weight"] = B.contiguous()
    save_file(sd, str(path), metadata=metadata)


def round_to_fp8(x: Tensor, generator: torch.Generator | None) -> Tensor:
    """Round float32 values onto the float8_e4m3fn grid.

    With a generator the rounding is stochastic (unbiased), as ComfyUI does when it patches
    an fp8 model with a LoRA. Round-to-nearest would snap most of a small LoRA delta back to
    the base weight, because the base values already sit exactly on the fp8 grid.
    """
    if generator is None:
        return x.clamp(-FP8_MAX, FP8_MAX).to(FP8)
    a = x.abs().clamp_(max=FP8_MAX)
    exponent = (torch.frexp(a).exponent - 1).clamp_(-6, 8)  # below 2^-6 e4m3 is subnormal
    step = torch.exp2((exponent - 3).float())  # 3 mantissa bits
    a = a.div_(step).add_(torch.rand(a.shape, device=a.device, generator=generator)).floor_().mul_(step)
    return (a.clamp_(max=FP8_MAX) * torch.sign(x)).to(FP8)


def bake_checkpoint(
    base_path: str | Path,
    merged: dict[str, tuple[str, Tensor, Tensor]],
    out_path: str | Path,
    strength: float = 1.0,
    stochastic: bool = True,
    seed: int = 0,
    device: str = "cuda",
    progress=None,
) -> None:
    """W' = W + strength * B_merged A_merged for every LoRA layer, saved in the base checkpoint's format."""
    out: dict[str, Tensor] = {}
    generator = torch.Generator(device=device).manual_seed(seed) if stochastic else None
    with safe_open(str(base_path), framework="pt") as f:
        metadata = f.metadata()
        keys = set(f.keys())
        for name, (_, A, B) in merged.items():
            if name + ".weight" not in keys:
                raise KeyError(f"{name}.weight is not in {base_path}")
        names = sorted(keys)
        for i, key in enumerate(names):
            layer = key[: -len(".weight")] if key.endswith(".weight") else None
            if key in out:  # a weight_scale we already rewrote
                pass
            elif layer in merged:
                _, A, B = merged[layer]
                w = dequantize_checkpoint_weight(f, key, keys, device)
                w += strength * (B.to(device, torch.float32) @ A.to(device, torch.float32))
                stored = f.get_slice(key).get_dtype()
                if stored == "F8_E4M3":
                    scale_key = key + "_scale"
                    scale = f.get_tensor(scale_key) if scale_key in keys else torch.ones(())
                    # Keep the stored scale so untouched precision is not re-rounded; a peak that
                    # overshoots by less than one fp8 step is clamped. Only rescale beyond that.
                    needed = w.abs().max().item() / FP8_MAX
                    if needed > scale.item() * (1 + 2**-4):
                        scale = torch.full_like(scale, needed)
                    if scale_key in keys:
                        out[scale_key] = scale
                    out[key] = round_to_fp8(w.div_(scale.item()), generator).cpu()
                else:
                    out[key] = w.to(f.get_tensor(key).dtype).cpu()
                del w
            else:
                out[key] = f.get_tensor(key)
            if progress is not None:
                progress(i + 1, len(names))
    save_file(out, str(out_path), metadata=metadata)
