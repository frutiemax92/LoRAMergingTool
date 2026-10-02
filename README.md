# SSR-Merge for Krea 2

Merges several Krea 2 LoRAs into one, and bakes the result into a Krea 2 checkpoint, using
[SSR-Merge](https://arxiv.org/abs/2606.10617) (Subspace Signal Routing, ICML 2026;
reference code: [nagara214/SSR-Merge](https://github.com/nagara214/SSR-Merge)).
Runs on a single 12 GB GPU.

## Usage

```bash
.venv/bin/pip install -r requirements.txt     # once
.venv/bin/python merge.py
```

This takes the checkpoint in `Checkpoint/` and every LoRA in `Loras/`, and writes to `output/`:

| File | What it is |
|------|------------|
| `krea2_turbo_ssr_lora.safetensors` | The merged LoRA (rank = sum of the input ranks, bf16). Loads like any Krea 2 LoRA, at strength 1. |
| `krea2_turbo_ssr_fp8_scaled.safetensors` | The checkpoint with the merged LoRA baked in, in the same ComfyUI fp8-scaled format as the input. |

Useful options (`merge.py --help` for all): `--lora-only`, `--strength`, `--calib-steps`,
`--lambda-reg`, `--checkpoint`, `--loras`.

## Calibration prompts matter

SSR does not just add the LoRAs. For each LoRA it runs the model once with that LoRA active
and records what the LoRA's layers see; it then solves, per layer, for a router that sends
each kind of input to the LoRA it belongs to. What tells the LoRAs apart is the calibration
prompt, so each LoRA needs prompts that are typical of how *it* is used, including its
trigger words if it has any. If every LoRA gets the same prompt, the result degenerates to a
plain average of the LoRAs.

Prompts live in `calibration.json`, keyed by LoRA file name:

```json
{
  "MyLora": { "weight": 1.0, "prompts": ["first prompt", "second prompt"] }
}
```

`weight` scales that LoRA before merging. Several prompts per LoRA give better statistics,
particularly for the text-fusion layers, which only see the prompt tokens.

`merge.py` prints the reconstruction error of the SSR router next to plain averaging and
plain summing ("task arithmetic") on the calibration data, as a check that routing helps.

## Memory

- The text encoder (Qwen3-VL-4B, bf16) is loaded, used and freed before the DiT is loaded.
- The DiT's large Linear layers (~12B parameters) are held on the GPU as 4-bit NF4
  (bitsandbytes) and dequantized to bf16 one layer at a time. Everything else is bf16.
- LoRA weights are bf16. The GPU holds every LoRA's down-projection (A) but only the
  active LoRA's up-projection (B); the rest wait in system RAM.
- The per-layer SSR statistics (float64) are kept in system RAM.

VRAM grows by roughly 0.1 GiB per rank-32 LoRA, so the number of LoRAs is not limited by
a 12 GB card in practice. Measured peak VRAM at 1024×1024: 8.7 GiB for 4 LoRAs, 10.0 GiB for 16 (plus about 14 GB of
system RAM). NF4 is only used for the
calibration pass; the baked checkpoint is built from the original checkpoint weights.

## The baked checkpoint and fp8

The base weights sit exactly on the fp8 grid, and a LoRA delta is usually smaller than one
fp8 step, so rounding to nearest would discard much of it. The bake therefore uses
stochastic rounding (as ComfyUI does when it applies a LoRA to an fp8 model), which keeps
the delta on average at the cost of some rounding noise. For the exact merge, use the base
checkpoint with the merged LoRA file instead.

## Preview

```bash
.venv/bin/python preview.py "a fox walking in the snow"
.venv/bin/python preview.py "..." --lora output/krea2_turbo_ssr_lora.safetensors
.venv/bin/python preview.py "..." --checkpoint output/krea2_turbo_ssr_fp8_scaled.safetensors
```

Generates an 8-step Turbo image with the same NF4 model, to `output/preview.png`.

## Layout

- `merge.py` — command line entry point.
- `ssr_krea2/ssr.py` — SSR statistics and router solve.
- `ssr_krea2/krea2.py` — NF4 Krea 2 DiT, text encoder, sampler (ported from
  [krea-ai/krea-2](https://github.com/krea-ai/krea-2), Apache-2.0).
- `ssr_krea2/checkpoint.py` — LoRA output and fp8 checkpoint baking.

The text encoder, tokenizer and VAE are taken from a diffusers-format Krea 2 repo in the
Hugging Face cache (`krea/Krea-2-Turbo` or `krea/Krea-2-Raw`), or downloaded if absent.
