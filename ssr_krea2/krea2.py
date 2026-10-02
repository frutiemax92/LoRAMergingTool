"""Inference-only Krea 2 DiT with NF4 base weights, plus its text encoder and sampler.

The forward pass is a port of ``mmdit.py`` / ``sampling.py`` / ``encoder.py`` from
https://github.com/krea-ai/krea-2 (Apache-2.0), restructured so that:

* every large Linear (attention + MLP projections, ~12B parameters) is stored on the
  GPU as 4-bit NF4 (bitsandbytes) and dequantized to bf16 only while it is used;
* everything else (norm scales, modulation tables, in/out projections) stays bf16;
* each large Linear exposes a ``tap`` slot, which is how LoRAs are applied and how
  SSR-Merge observes the layer's inputs.

Batch size is always 1 and prompts are not padded, so no attention masks are needed.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import torch
import torch.nn.functional as F
from einops import rearrange
from safetensors import safe_open
from torch import Tensor

BF16 = torch.bfloat16

# Linears that LoRAs target and that hold almost all of the parameters.
BIG_LINEAR = re.compile(r"^(blocks|txtfusion\.(layerwise|refiner)_blocks)\.\d+\.(attn|mlp)\.\w+\.weight$")

TEXT_LAYERS = (2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35)
PROMPT_PREFIX = (
    "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, "
    "text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n"
)
PROMPT_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n"
TEXT_ENCODER_REPOS = ("krea/Krea-2-Turbo", "krea/Krea-2-Raw")


def dequantize_checkpoint_weight(f, key: str, keys: set[str], device) -> Tensor:
    """Read one weight as float32 on ``device``, undoing ComfyUI's ``*_scaled`` fp8 format."""
    w = f.get_tensor(key).to(device).to(torch.float32)
    scale_key = key + "_scale"
    if scale_key in keys:
        w *= f.get_tensor(scale_key).to(device, torch.float32)
    return w


class NF4Linear:
    """A bias-free Linear whose weight lives on the GPU as 4-bit NF4."""

    def __init__(self, weight: Tensor):
        import bitsandbytes.functional as bnbF

        self.out_features, self.in_features = weight.shape
        self.packed, self.state = bnbF.quantize_4bit(weight.to(BF16), blocksize=64, quant_type="nf4")
        self.tap = None  # optional callable (x, y) -> y

    def weight(self) -> Tensor:
        import bitsandbytes.functional as bnbF

        return bnbF.dequantize_4bit(self.packed, self.state, quant_type="nf4").to(BF16)

    def __call__(self, x: Tensor) -> Tensor:
        y = F.linear(x, self.weight())
        return y if self.tap is None else self.tap(x, y)


def _rms_norm(x: Tensor, scale: Tensor) -> Tensor:
    return F.rms_norm(x.float(), (x.shape[-1],), weight=scale.float() + 1.0, eps=1e-5).to(x.dtype)


def _rope(pos: Tensor, dim: int, theta: float) -> Tensor:
    scale = torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim
    out = torch.einsum("...n,d->...nd", pos, 1.0 / (theta**scale))
    out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
    return rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2).float()


def _rope_apply(xq: Tensor, xk: Tensor, freqs: Tensor) -> tuple[Tensor, Tensor]:
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
    freqs = freqs[:, None]
    xq_ = freqs[..., 0] * xq_[..., 0] + freqs[..., 1] * xq_[..., 1]
    xk_ = freqs[..., 0] * xk_[..., 0] + freqs[..., 1] * xk_[..., 1]
    return xq_.reshape(*xq.shape).to(xq.dtype), xk_.reshape(*xk.shape).to(xk.dtype)


class Krea2DiT:
    patch = 2
    channels = 16
    headdim = 128
    rope_theta = 1e3

    def __init__(self, params: dict[str, Tensor], linears: dict[str, NF4Linear]):
        self.p = params
        self.lin = linears
        self.layers = 1 + max(int(k.split(".")[1]) for k in linears if k.startswith("blocks."))
        self.tdim = params["tmlp.0.weight"].shape[1]
        self.device = params["first.weight"].device
        d = self.headdim
        self.rope_axes = [d - 12 * (d // 16), 6 * (d // 16), 6 * (d // 16)]

    @classmethod
    def load(cls, path: str | Path, device: str = "cuda", progress=None) -> "Krea2DiT":
        """Stream a checkpoint to the GPU, quantizing each large Linear to NF4 as it arrives."""
        params, linears = {}, {}
        with safe_open(str(path), framework="pt") as f:
            keys = set(f.keys())
            names = sorted(k for k in keys if not k.endswith(".weight_scale"))
            for i, key in enumerate(names):
                if BIG_LINEAR.match(key):
                    w = dequantize_checkpoint_weight(f, key, keys, device)
                    linears[key[: -len(".weight")]] = NF4Linear(w)
                    del w
                else:
                    params[key] = f.get_tensor(key).to(device, BF16)
                if progress is not None:
                    progress(i + 1, len(names))
        return cls(params, linears)

    # ------------------------------------------------------------------ blocks
    def _attn(self, pre: str, x: Tensor, freqs: Tensor | None) -> Tensor:
        lin, p, d = self.lin, self.p, self.headdim
        q, k, v, gate = lin[pre + ".wq"](x), lin[pre + ".wk"](x), lin[pre + ".wv"](x), lin[pre + ".gate"](x)
        heads, kvheads = q.shape[-1] // d, k.shape[-1] // d
        q = rearrange(q, "B L (H D) -> B H L D", H=heads)
        k = rearrange(k, "B L (H D) -> B H L D", H=kvheads)
        v = rearrange(v, "B L (H D) -> B H L D", H=kvheads)
        q = _rms_norm(q, p[pre + ".qknorm.qnorm.scale"])
        k = _rms_norm(k, p[pre + ".qknorm.knorm.scale"])
        if freqs is not None:
            q, k = _rope_apply(q, k, freqs)
        out = F.scaled_dot_product_attention(q, k, v, enable_gqa=heads != kvheads)
        out = rearrange(out, "B H L D -> B L (H D)")
        return lin[pre + ".wo"](out * torch.sigmoid(gate))

    def _mlp(self, pre: str, x: Tensor) -> Tensor:
        lin = self.lin
        return lin[pre + ".down"](F.silu(lin[pre + ".gate"](x)) * lin[pre + ".up"](x))

    def _text_block(self, pre: str, x: Tensor) -> Tensor:
        x = x + self._attn(pre + ".attn", _rms_norm(x, self.p[pre + ".prenorm.scale"]), None)
        return x + self._mlp(pre + ".mlp", _rms_norm(x, self.p[pre + ".postnorm.scale"]))

    def _block(self, pre: str, x: Tensor, tvec: Tensor, freqs: Tensor) -> Tensor:
        p = self.p
        prescale, preshift, pregate, postscale, postshift, postgate = (tvec + p[pre + ".mod.lin"]).chunk(6, dim=-1)
        h = (1 + prescale) * _rms_norm(x, p[pre + ".prenorm.scale"]) + preshift
        x = x + pregate * self._attn(pre + ".attn", h, freqs)
        h = (1 + postscale) * _rms_norm(x, p[pre + ".postnorm.scale"]) + postshift
        return x + postgate * self._mlp(pre + ".mlp", h)

    # ----------------------------------------------------------------- forward
    @torch.no_grad()
    def __call__(self, img: Tensor, context: Tensor, t: Tensor, pos: Tensor) -> Tensor:
        """img (1, L, 64) · context (1, l, 12, 2560) · t (1,) · pos (1, l + L, 3) -> velocity (1, L, 64)."""
        p = self.p
        img = F.linear(img, p["first.weight"], p["first.bias"])

        half = self.tdim // 2
        freqs = torch.exp(-math.log(1e4) * torch.arange(half, dtype=torch.float32, device=img.device) / half)
        args = (t.float() * 1e3)[:, None, None] * freqs
        tv = torch.cat((torch.cos(args), torch.sin(args)), dim=-1).to(img.dtype)
        tv = F.linear(tv, p["tmlp.0.weight"], p["tmlp.0.bias"])
        tv = F.linear(F.gelu(tv, approximate="tanh"), p["tmlp.2.weight"], p["tmlp.2.bias"])
        tvec = F.linear(F.gelu(tv, approximate="tanh"), p["tproj.1.weight"], p["tproj.1.bias"])

        b, l, n, d = context.shape
        ctx = context.reshape(b * l, n, d)
        for i in range(2):
            ctx = self._text_block(f"txtfusion.layerwise_blocks.{i}", ctx)
        ctx = rearrange(ctx, "(b l) n d -> b l d n", b=b, l=l)
        ctx = F.linear(ctx, p["txtfusion.projector.weight"]).squeeze(-1)
        for i in range(2):
            ctx = self._text_block(f"txtfusion.refiner_blocks.{i}", ctx)
        ctx = _rms_norm(ctx, p["txtmlp.0.scale"])
        ctx = F.linear(ctx, p["txtmlp.1.weight"], p["txtmlp.1.bias"])
        ctx = F.linear(F.gelu(ctx, approximate="tanh"), p["txtmlp.3.weight"], p["txtmlp.3.bias"])

        x = torch.cat((ctx, img), dim=1)
        rope = torch.cat([_rope(pos[..., i], a, self.rope_theta) for i, a in enumerate(self.rope_axes)], dim=-3)
        for i in range(self.layers):
            x = self._block(f"blocks.{i}", x, tvec, rope)

        scale, shift = (tv + p["last.modulation.lin"][None]).chunk(2, dim=1)
        x = (1 + scale) * _rms_norm(x, p["last.norm.scale"]) + shift
        x = F.linear(x, p["last.linear.weight"], p["last.linear.bias"])
        return x[:, l:]

    # ---------------------------------------------------------------- sampling
    @torch.no_grad()
    def sample(self, context: Tensor, width=1024, height=1024, steps=8, seed=0, mu=1.15) -> Tensor:
        """Euler flow-matching sampler without CFG (the Turbo recipe). Returns the latent (1, 16, H/8, W/8).

        With ``steps=1`` this is a single forward pass on pure noise at t=1, which is the
        one-shot calibration pass SSR-Merge uses.
        """
        dev, pt = self.device, self.patch
        h, w = height // 8, width // 8
        noise = torch.randn(1, self.channels, h, w, generator=torch.Generator().manual_seed(seed))
        img = rearrange(noise.to(dev, BF16), "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=pt, pw=pt)

        ids = torch.zeros(h // pt, w // pt, 3, device=dev)
        ids[..., 1] = torch.arange(h // pt, device=dev)[:, None]
        ids[..., 2] = torch.arange(w // pt, device=dev)[None, :]
        context = context.to(dev, BF16)
        pos = torch.cat((torch.zeros(1, context.shape[1], 3, device=dev), ids.reshape(1, -1, 3)), dim=1)

        ts = torch.linspace(1, 0, steps + 1)
        ts = (math.exp(mu) / (math.exp(mu) + (1.0 / ts - 1.0))).tolist()
        for tcurr, tprev in zip(ts[:-1], ts[1:]):
            t = torch.full((1,), tcurr, dtype=img.dtype, device=dev)
            img = img + (tprev - tcurr) * self(img, context, t, pos)
        return rearrange(img, "b (h w) (c ph pw) -> b c (h ph) (w pw)", ph=pt, pw=pt, h=h // pt, w=w // pt)


# ---------------------------------------------------------------------- text
def find_krea2_component(subfolders: tuple[str, ...], override: str | None = None) -> Path:
    """Locate a diffusers-format Krea 2 repo holding ``subfolders``; prefers the local HF cache."""
    if override:
        return Path(override)
    from huggingface_hub import snapshot_download

    patterns = [f"{s}/*" for s in subfolders]
    for repo in TEXT_ENCODER_REPOS:
        try:
            root = Path(snapshot_download(repo, allow_patterns=patterns, local_files_only=True))
        except Exception:
            continue
        if all((root / s).is_dir() and any((root / s).iterdir()) for s in subfolders):
            return root
    return Path(snapshot_download(TEXT_ENCODER_REPOS[0], allow_patterns=patterns))


@torch.no_grad()
def encode_prompts(prompts: list[str], device: str = "cuda", repo: str | None = None) -> list[Tensor]:
    """Encode prompts with Qwen3-VL-4B (bf16). Returns one (1, l, 12, 2560) CPU tensor per prompt.

    The encoder is loaded, used and freed here so it never shares VRAM with the DiT.
    """
    from transformers import AutoModel, AutoTokenizer

    root = find_krea2_component(("text_encoder", "tokenizer"), repo)
    tokenizer = AutoTokenizer.from_pretrained(root / "tokenizer")
    model = AutoModel.from_pretrained(root / "text_encoder", dtype=BF16).to(device).eval()
    skip = len(tokenizer(PROMPT_PREFIX)["input_ids"])

    out = []
    for prompt in prompts:
        inputs = tokenizer(PROMPT_PREFIX + prompt + PROMPT_SUFFIX, return_tensors="pt").to(device)
        states = model(**inputs, output_hidden_states=True).hidden_states
        out.append(torch.stack([states[i] for i in TEXT_LAYERS], dim=2)[:, skip:].to("cpu", BF16))
    del model
    torch.cuda.empty_cache()
    return out
