"""Generate a test image with the NF4 Krea 2 model, optionally with a LoRA applied.

    .venv/bin/python preview.py "a fox walking in the snow"
    .venv/bin/python preview.py "..." --lora output/krea2_turbo_ssr_lora.safetensors
    .venv/bin/python preview.py "..." --checkpoint output/krea2_turbo_ssr_fp8_scaled.safetensors
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image

from ssr_krea2.krea2 import BF16, Krea2DiT, encode_prompts, find_krea2_component
from ssr_krea2.ssr import load_lora

ROOT = Path(__file__).resolve().parent


class LoraTap:
    def __init__(self, A, B, strength, device):
        self.A, self.B, self.strength = A.to(device, BF16), B.to(device, BF16), strength

    def __call__(self, x, y):
        return y + self.strength * ((x @ self.A.T) @ self.B.T)


@torch.no_grad()
def decode(latent: torch.Tensor, device: str, repo: str | None) -> Image.Image:
    from diffusers import AutoencoderKLQwenImage

    vae = AutoencoderKLQwenImage.from_pretrained(find_krea2_component(("vae",), repo) / "vae", torch_dtype=BF16).to(device)
    mean = torch.tensor(vae.config.latents_mean, device=device).view(1, -1, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std, device=device).view(1, -1, 1, 1, 1)
    x = (latent.float().unsqueeze(2) * std + mean).to(BF16)
    img = vae.decode(x).sample[:, :, 0].float().clamp(-1, 1) * 0.5 + 0.5
    return Image.fromarray((img[0].permute(1, 2, 0) * 255).round().byte().cpu().numpy())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("prompt")
    ap.add_argument("--checkpoint", type=Path)
    ap.add_argument("--lora", type=Path, nargs="*", default=[])
    ap.add_argument("--lora-strength", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", type=Path, default=ROOT / "output" / "preview.png")
    ap.add_argument("--krea2-repo", default=None, help="diffusers-format Krea 2 folder (default: HF cache)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    if args.checkpoint is None:
        found = sorted((ROOT / "Checkpoint").glob("*.safetensors"))
        if len(found) != 1:
            sys.exit("Pass --checkpoint.")
        args.checkpoint = found[0]

    context = encode_prompts([args.prompt], args.device, args.krea2_repo)[0]
    dit = Krea2DiT.load(args.checkpoint, args.device)
    for path in args.lora:
        for name, (_, A, B) in load_lora(path).items():
            previous = dit.lin[name].tap
            tap = LoraTap(A, B, args.lora_strength, args.device)
            dit.lin[name].tap = tap if previous is None else (lambda x, y, a=previous, b=tap: b(x, a(x, y)))
    latent = dit.sample(context, args.width, args.height, args.steps, args.seed)
    del dit
    torch.cuda.empty_cache()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    decode(latent, args.device, args.krea2_repo).save(args.output)
    print(f"saved {args.output}  (peak VRAM {torch.cuda.max_memory_allocated(args.device) / 2**30:.1f} GiB)")


if __name__ == "__main__":
    main()
