"""SSR-Merge (Subspace Signal Routing) statistics and router solve.

Wei et al., "SSR-Merge: Subspace Signal Routing for Training-Free LoRA Merging in
Diffusion Models", ICML 2026 — https://arxiv.org/abs/2606.10617, reference code at
https://github.com/nagara214/SSR-Merge.

Per layer, with K LoRAs (A_k, B_k):

    A_comb = [A_1; ...; A_K]            B_comb = [B_1 ... B_K]
    Z_k    = A_comb X_k                 (X_k: layer inputs while LoRA k is active)
    G      = sum_k Z_k Z_k^T            Q = sum_k E_k (A_k X_k) Z_k^T
    R      = Q (G/N + lambda I)^-1      B_merged = B_comb R

G and Q (two R x R float64 matrices per layer, R = sum of ranks) are all that is kept, and
they live in system RAM. They also give the exact reconstruction error of any router, which
is used for the diagnostics. On the GPU each layer holds only A_comb and the active B_k.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors.torch import load_file
from torch import Tensor

# diffusers / PEFT module names -> native Krea 2 checkpoint names
_RENAMES = (
    (r"^(transformer|diffusion_model)\.", ""),
    (r"^transformer_blocks\.", "blocks."),
    (r"^text_fusion\.", "txtfusion."),
    (r"\.attn\.to_q$", ".attn.wq"),
    (r"\.attn\.to_k$", ".attn.wk"),
    (r"\.attn\.to_v$", ".attn.wv"),
    (r"\.attn\.to_gate$", ".attn.gate"),
    (r"\.attn\.to_out(\.0)?$", ".attn.wo"),
    (r"\.ff\.(gate|up|down)$", r".mlp.\1"),
)


def native_name(lora_module: str) -> str:
    for pattern, repl in _RENAMES:
        lora_module = re.sub(pattern, repl, lora_module)
    return lora_module


def load_lora(path: str | Path, weight: float = 1.0) -> dict[str, tuple[str, Tensor, Tensor]]:
    """Load a LoRA as ``{native layer name: (name in file, A, B)}`` with alpha/weight folded into B."""
    sd = load_file(str(path))
    out = {}
    for key, A in sd.items():
        if not key.endswith(".lora_A.weight"):
            continue
        module = key[: -len(".lora_A.weight")]
        B = sd[module + ".lora_B.weight"]
        scale = weight
        if module + ".alpha" in sd:
            scale *= float(sd[module + ".alpha"]) / A.shape[0]
        out[native_name(module)] = (module, A, B * scale if scale != 1.0 else B)
    if not out:
        raise ValueError(f"{path}: no '<module>.lora_A.weight' keys found")
    return out


class Context:
    """Which LoRA is currently active in the model, and whether to record statistics."""

    task: int | None = None
    collect: bool = False


class LayerTap:
    """Applies the active task's LoRA on one Linear and accumulates its SSR statistics."""

    def __init__(self, ctx: Context, name: str, out_name: str, tasks: list[int], As: list[Tensor], Bs: list[Tensor], device):
        self.ctx, self.name, self.out_name = ctx, name, out_name
        self.local = {k: j for j, k in enumerate(tasks)}
        self.A = torch.cat(As, dim=0).to(device, torch.bfloat16)  # (R, in)
        self.B = torch.cat(Bs, dim=1).to(torch.bfloat16)  # (out, R), in RAM
        self.B_active = None  # the active task's B_k, on the GPU
        self.slices, start = [], 0
        for a in As:
            self.slices.append(slice(start, start + a.shape[0]))
            start += a.shape[0]
        self.G = torch.zeros(start, start, dtype=torch.float64)  # in RAM
        self.Q = torch.zeros(start, start, dtype=torch.float64)
        self.count = 0

    def activate(self, task: int | None) -> None:
        j = self.local.get(task)
        self.B_active = None if j is None else self.B[:, self.slices[j]].to(self.A.device)

    def __call__(self, x: Tensor, y: Tensor) -> Tensor:
        j = self.local.get(self.ctx.task)
        if j is None:
            return y
        sl = self.slices[j]
        z = x.reshape(-1, x.shape[-1]).float() @ self.A.float().T  # (N, R)
        if self.ctx.collect:
            zd = z.double()
            self.G += (zd.T @ zd).cpu()
            self.Q[sl] += (zd[:, sl].T @ zd).cpu()
            self.count += z.shape[0]
        delta = z[:, sl] @ self.B_active.float().T
        return y + delta.reshape(y.shape).to(y.dtype)

    def routers(self, lambda_reg: float) -> dict[str, Tensor]:
        """The SSR router plus the two static baselines it generalizes."""
        dev = self.A.device
        eye = torch.eye(self.G.shape[0], dtype=torch.float64, device=dev)
        out = {"task_arithmetic": eye, "average": eye / len(self.slices)}
        if self.count > 0 and len(self.slices) > 1:
            G = self.G.to(dev) / self.count + lambda_reg * eye
            out["ssr"] = torch.linalg.solve(G, self.Q.to(dev).T / self.count).T  # Q G^-1, G symmetric
        else:
            out["ssr"] = eye
        return out

    def errors(self, routers: dict[str, Tensor]) -> tuple[dict[str, float], float]:
        """sum_k ||B_comb R Z_k - B_k A_k X_k||_F^2 for each router, and sum_k ||B_k A_k X_k||_F^2."""
        # Expanding the square leaves only G, Q and the diagonal blocks of Q.
        dev = self.A.device
        Bd = self.B.to(dev).double()
        BtB, G, Q = Bd.T @ Bd, self.G.to(dev), self.Q.to(dev)
        ref = sum(torch.sum(BtB[sl, sl] * Q[sl, sl]).item() for sl in self.slices)
        err = {
            name: torch.sum(BtB * (R @ G @ R.T)).item() - 2 * torch.sum(BtB * (R @ Q.T)).item() + ref
            for name, R in routers.items()
        }
        return err, ref


@dataclass
class MergeReport:
    errors: dict[str, float]  # relative reconstruction error per router, over all layers
    groups: dict[str, dict[str, float]]  # same, split by layer family
    undersampled: list[str]  # layers that saw fewer tokens than the merged rank


class SSRMerger:
    def __init__(self, loras: list[dict[str, tuple[str, Tensor, Tensor]]], device="cuda"):
        if len(loras) < 2:
            raise ValueError(f"SSR merging needs at least 2 LoRAs (got {len(loras)}).")
        self.ctx = Context()
        self.taps: dict[str, LayerTap] = {}
        for name in sorted({n for lora in loras for n in lora}):
            tasks = [k for k, lora in enumerate(loras) if name in lora]
            entries = [loras[k][name] for k in tasks]
            self.taps[name] = LayerTap(
                self.ctx, name, entries[0][0], tasks, [e[1] for e in entries], [e[2] for e in entries], device
            )

    def attach(self, linears: dict) -> None:
        missing = [n for n in self.taps if n not in linears]
        if missing:
            raise KeyError(f"{len(missing)} LoRA layers have no match in the checkpoint, e.g. {missing[:3]}")
        for name, tap in self.taps.items():
            if tap.A.shape[1] != linears[name].in_features or tap.B.shape[0] != linears[name].out_features:
                raise ValueError(f"LoRA shape mismatch on {name}")
            linears[name].tap = tap

    def set_task(self, task: int | None, collect: bool = False) -> None:
        """Activate one LoRA in the model (or none), moving only its B matrices to the GPU."""
        self.ctx.task, self.ctx.collect = task, collect
        for tap in self.taps.values():
            tap.activate(task)

    def detach(self, linears: dict) -> None:
        for name in self.taps:
            linears[name].tap = None

    def solve(self, lambda_reg: float = 1e-4) -> tuple[dict[str, tuple[str, Tensor, Tensor]], MergeReport]:
        """Returns ``{native name: (output name, A_merged, B_merged)}`` (bf16, CPU) and diagnostics."""
        merged, undersampled = {}, []
        totals: dict[str, dict[str, float]] = {}
        refs: dict[str, float] = {}
        for name, tap in self.taps.items():
            routers = tap.routers(lambda_reg)
            B = (tap.B.to(tap.A.device).double() @ routers["ssr"]).to("cpu", torch.bfloat16)
            merged[name] = (tap.out_name, tap.A.cpu(), B)
            if len(tap.slices) < 2:
                continue
            if tap.count < tap.A.shape[0]:
                undersampled.append(name)
            err, ref = tap.errors(routers)
            for group in ("all", "text fusion" if name.startswith("txtfusion") else "image blocks"):
                acc = totals.setdefault(group, {})
                for router, e in err.items():
                    acc[router] = acc.get(router, 0.0) + e
                refs[group] = refs.get(group, 0.0) + ref
        rel = {g: {r: e / max(refs[g], 1e-30) for r, e in acc.items()} for g, acc in totals.items()}
        return merged, MergeReport(rel.pop("all", {}), rel, undersampled)
