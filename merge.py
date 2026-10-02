"""Merge every LoRA in Loras/ into the Krea 2 checkpoint in Checkpoint/ with SSR-Merge.

    .venv/bin/python merge.py

Outputs (in output/):
    <name>_lora.safetensors         the merged LoRA (rank = sum of input ranks), usable on its own
    <name>_fp8_scaled.safetensors   the checkpoint with the merged LoRA baked in

SSR-Merge needs one calibration forward pass per LoRA, with that LoRA active and a prompt
that is typical for it. Prompts come from calibration.json; see the README.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

from ssr_krea2.checkpoint import bake_checkpoint, save_lora
from ssr_krea2.krea2 import Krea2DiT, encode_prompts
from ssr_krea2.ssr import SSRMerger, load_lora

ROOT = Path(__file__).resolve().parent
DEFAULT_PROMPT = "A photo of a woman standing in a sunlit room, looking at the camera."


def progress(label: str):
    def update(done: int, total: int) -> None:
        if done == total or done % 25 == 0:
            print(f"\r  {label}: {done}/{total}", end="\n" if done == total else "", flush=True)

    return update


def load_calibration(path: Path, lora_paths: list[Path]) -> list[dict]:
    """Per LoRA: {"prompts": [...], "weight": float}. LoRAs missing from the file get a generic prompt."""
    config = json.loads(path.read_text()) if path.exists() else {}
    tasks = []
    for p in lora_paths:
        entry = config.get(p.stem) or config.get(p.name)
        if entry is None:
            print(f"  ! no entry for '{p.stem}' in {path.name}; using a generic prompt. "
                  "SSR routes by what each LoRA sees, so a prompt typical of this LoRA will merge better.")
            entry = {}
        prompts = entry.get("prompts") or [DEFAULT_PROMPT]
        tasks.append({"prompts": [prompts] if isinstance(prompts, str) else list(prompts),
                      "weight": float(entry.get("weight", 1.0))})
    return tasks


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, help="base checkpoint (default: the only .safetensors in Checkpoint/)")
    ap.add_argument("--loras", type=Path, nargs="+", help="LoRA files (default: every .safetensors in Loras/)")
    ap.add_argument("--calibration", type=Path, default=ROOT / "calibration.json")
    ap.add_argument("--output-dir", type=Path, default=ROOT / "output")
    ap.add_argument("--name", default=None, help="output file prefix (default: <checkpoint name>_ssr)")
    ap.add_argument("--lambda-reg", type=float, default=1e-4, help="ridge term on the correlation matrix")
    ap.add_argument("--calib-steps", type=int, default=1,
                    help="sampler steps per calibration pass; 1 = single pass at t=1, as in the paper")
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--strength", type=float, default=1.0, help="strength of the merged LoRA when baked in")
    ap.add_argument("--rounding", choices=("stochastic", "nearest"), default="stochastic",
                    help="fp8 rounding for the baked checkpoint")
    ap.add_argument("--lora-only", action="store_true", help="skip writing the merged checkpoint")
    ap.add_argument("--text-encoder", default=None,
                    help="diffusers-format Krea 2 folder with text_encoder/ and tokenizer/ (default: HF cache)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    if args.checkpoint is None:
        found = sorted((ROOT / "Checkpoint").glob("*.safetensors"))
        if len(found) != 1:
            sys.exit(f"Expected exactly one checkpoint in Checkpoint/, found {len(found)}; pass --checkpoint.")
        args.checkpoint = found[0]
    lora_paths = args.loras or sorted((ROOT / "Loras").glob("*.safetensors"))
    if len(lora_paths) < 2:
        sys.exit(f"SSR merging needs at least 2 LoRAs, found {len(lora_paths)}.")
    name = args.name or args.checkpoint.stem.replace("_fp8_scaled", "") + "_ssr"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print(f"Checkpoint: {args.checkpoint.name}")
    tasks = load_calibration(args.calibration, lora_paths)
    for p, task in zip(lora_paths, tasks):
        print(f"  LoRA {p.name}  weight={task['weight']}  prompts={len(task['prompts'])}")

    print("Encoding calibration prompts (Qwen3-VL-4B, bf16)...")
    flat = [prompt for task in tasks for prompt in task["prompts"]]
    contexts = iter(encode_prompts(flat, args.device, args.text_encoder))
    for task in tasks:
        task["contexts"] = [next(contexts) for _ in task["prompts"]]

    print("Loading the base model as NF4...")
    dit = Krea2DiT.load(args.checkpoint, args.device, progress("tensors"))
    print(f"  VRAM in use: {torch.cuda.memory_allocated(args.device) / 2**30:.1f} GiB")

    merger = SSRMerger([load_lora(p, t["weight"]) for p, t in zip(lora_paths, tasks)], args.device)
    merger.attach(dit.lin)
    print(f"Calibrating {len(merger.taps)} layers...")
    for k, (p, task) in enumerate(zip(lora_paths, tasks)):
        merger.set_task(k, collect=True)
        for i, context in enumerate(task["contexts"]):
            t1 = time.time()
            dit.sample(context, args.width, args.height, steps=args.calib_steps, seed=args.seed)
            print(f"  [{k + 1}/{len(tasks)}] {p.stem} prompt {i + 1}/{len(task['contexts'])}: {time.time() - t1:.1f}s")
    merger.set_task(None)

    print("Solving routers...")
    merged, report = merger.solve(args.lambda_reg)
    print("  relative reconstruction error of each LoRA's own output (lower is better):")
    for group, errors in [("all layers", report.errors), *report.groups.items()]:
        print(f"    {group:<12} " + "  ".join(f"{r}={e:.4f}" for r, e in sorted(errors.items())))
    if report.undersampled:
        print(f"  ! {len(report.undersampled)} layers saw fewer calibration tokens than the merged rank "
              f"(e.g. {report.undersampled[0]}); add or lengthen prompts in {args.calibration.name}.")
    print(f"  peak VRAM: {torch.cuda.max_memory_allocated(args.device) / 2**30:.1f} GiB")

    lora_out = args.output_dir / f"{name}_lora.safetensors"
    save_lora(merged, lora_out, {
        "modelspec.architecture": "Krea-2/lora",
        "modelspec.title": f"SSR-Merge of {', '.join(p.stem for p in lora_paths)}",
        "ssr_merge.sources": json.dumps([p.name for p in lora_paths]),
        "ssr_merge.weights": json.dumps([t["weight"] for t in tasks]),
        "ssr_merge.lambda_reg": str(args.lambda_reg),
    })
    print(f"Merged LoRA -> {lora_out}")

    if not args.lora_only:
        merger.detach(dit.lin)
        del dit, merger
        torch.cuda.empty_cache()
        print("Baking the merged LoRA into the checkpoint...")
        suffix = args.checkpoint.stem[len(args.checkpoint.stem.replace("_fp8_scaled", "")):]
        ckpt_out = args.output_dir / f"{name}{suffix}.safetensors"
        bake_checkpoint(args.checkpoint, merged, ckpt_out, args.strength, args.rounding == "stochastic",
                        args.seed, args.device, progress("tensors"))
        print(f"Merged checkpoint -> {ckpt_out}")
    print(f"Done in {time.time() - t0:.0f}s.")


if __name__ == "__main__":
    main()
