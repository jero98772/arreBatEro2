"""
STEP 3 (hybrid variant) — one Transformer whose layers run on different
computing paradigms, trained on this pipeline's tokenized corpus
(data/tokenized/{train,val}.npy from step2_tokenizer.py).

Every sublayer (embedding, attention, MLP, norm, LM head) is assigned a
substrate — classical, photonic, quantum or thermodynamic — from the per-layer
benchmark table (time / energy / cost / quality per layer × substrate):

    classical      plain torch (MHA, GELU MLP, LayerNorm)
    photonic       Q.ANT NPU linears (step3_photonic_autoregrsive.QantLinear)
    quantum        QFT mixer replacing attention (step3_quantum_autoregrsive)
    thermodynamic  AFT-conv attention / SiLU MLP / DyT norm (PyTorch port of the
                   JAX model in step3_termodinamic_autoregrsive.py)

The map comes from `select_substrates` (cheapest weighted time/energy/cost among
implementable substrates whose benchmarked quality >= --q_min) and can be
overridden per layer with --assign AR_ATT2=quantum,AR_MLP1=thermodynamic.
Layer groups 1/2 (ATT1/ATT2, MLP1/MLP2, NORM1/NORM2) apply to the first/second
half of the blocks.

Standardized to match the other step3 trainers: argparse CLI, unified
logs/hybrid_training_metrics.jsonl schema (model_type="hybrid", readable by
monitor_training.py), ckpt_step_XXXXXXX.pt + ckpt_best.pt checkpoints with
resume (the checkpoint stores the substrate map and architecture so
inference.py can rebuild the model), pynvml energy/cost accounting,
SIGINT/SIGTERM-safe shutdown, and torchrun-based multi-GPU DDP.

Usage: python step3_hybrid_autoregrsive.py [--max_steps N] [--assign L=S,...] ...
       torchrun --nproc_per_node=NGPU step3_hybrid_autoregrsive.py ...
"""

from __future__ import annotations

import argparse
import csv
import datetime
import io
import json
import logging
import math
import os
import signal
import time
from contextlib import nullcontext
from pathlib import Path
from time import perf_counter
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

BASE = Path(__file__).resolve().parents[1]  # repo root (this file lives in <paradigm>/)
CKPT_DIR = BASE / "checkpoints_hybrid"
LOG_DIR = BASE / "logs"
TOK_DIR = BASE / "data" / "tokenized"
CKPT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

PAD_ID = 0  # "[PAD]" in step2_tokenizer.SPECIAL_TOKENS

LOG_PATH = LOG_DIR / "step3_hybrid_training.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_PATH), logging.StreamHandler()],
)
log = logging.getLogger(__name__)
METRICS_PATH = LOG_DIR / "hybrid_training_metrics.jsonl"
MODEL_TYPE = "hybrid"

# The photonic trainer provides QantLinear/PhotonicMHA/PhotonicFFN, the shared
# data/eval/DDP helpers and NVML power sampling; the quantum trainer provides
# the QFT mixer (needs pennylane, so it is only required when assigned).
import sys  # noqa: E402

for _d in ("photonic", "quantum", "hybrid"):  # sibling paradigm folders
    if str(BASE / _d) not in sys.path:
        sys.path.insert(0, str(BASE / _d))

import step3_photonic_autoregrsive as ph  # noqa: E402

try:
    import step3_quantum_autoregrsive as qu  # noqa: E402
except Exception as _exc:  # pennylane not installed
    qu = None
    log.info(f"quantum substrate unavailable ({type(_exc).__name__}: {_exc})")

_sample_power_w = ph._sample_power_w
TokenDataset = ph.TokenDataset
read_tokenized_metadata = ph.read_tokenized_metadata
evaluate = ph.evaluate
setup_distributed = ph.setup_distributed
cleanup_distributed = ph.cleanup_distributed
get_lr = ph.get_lr


def _r(v, nd):
    return None if v is None else round(v, nd)


def log_metrics(record: dict) -> None:
    with open(METRICS_PATH, "a") as fh:
        fh.write(json.dumps(record) + "\n")


# ════════════════════════════════════════════════════════════════════════════
# Per-layer substrate benchmark + assignment
# ════════════════════════════════════════════════════════════════════════════
SUBSTRATES = ("classical", "photonic", "quantum", "thermodynamic")

DEFAULT_BENCH_CSV = """layer,substrate,time_ms,energy_j,cost_usd,quality
AR_ATT1,classical,16.889650,11.804730,0.019601,0.957533
AR_ATT1,photonic,0.965967,1.077628,0.028557,0.847067
AR_ATT1,quantum,2.661250,2.894947,0.073409,0.689467
AR_ATT1,thermodynamic,7.316600,0.464019,0.012834,0.728300
AR_ATT2,classical,20.754517,14.748485,0.018553,0.951883
AR_ATT2,photonic,1.341567,1.177499,0.024158,0.847417
AR_ATT2,quantum,2.343483,3.054398,0.060258,0.702433
AR_ATT2,thermodynamic,7.872133,0.456474,0.011300,0.712150
AR_EMB,classical,15.249700,7.991173,0.013571,0.938233
AR_EMB,photonic,0.896683,0.789386,0.021123,0.838533
AR_HEAD,classical,15.355617,11.757956,0.014990,0.933167
AR_HEAD,photonic,1.238600,0.849456,0.023932,0.880867
AR_HEAD,thermodynamic,6.689183,0.502189,0.008794,0.795367
AR_MLP1,classical,28.204583,16.440124,0.024968,0.930050
AR_MLP1,photonic,1.464433,1.235009,0.034870,0.833100
AR_MLP1,quantum,4.272917,5.007322,0.086926,0.618967
AR_MLP1,thermodynamic,10.985200,0.613372,0.015129,0.761917
AR_MLP2,classical,22.288700,17.646251,0.020566,0.942417
AR_MLP2,photonic,1.625767,1.302645,0.032124,0.847350
AR_MLP2,quantum,2.708417,5.072518,0.113544,0.686650
AR_MLP2,thermodynamic,13.307533,0.514227,0.013731,0.746700
AR_NORM1,classical,6.808917,3.718861,0.004738,0.936183
AR_NORM1,photonic,0.326917,0.311375,0.007535,0.851133
AR_NORM1,quantum,0.800600,0.944516,0.026120,0.671650
AR_NORM2,classical,5.935217,3.765308,0.004451,0.943550
AR_NORM2,photonic,0.370267,0.233690,0.008218,0.877417
AR_NORM2,quantum,0.621767,0.995896,0.027463,0.728767
"""

# Substrates that have a working implementation for each layer kind. Benchmark
# rows outside this set (e.g. a photonic embedding or a quantum MLP) are ignored.
IMPLEMENTABLE = {
    "EMB": {"classical"},
    "ATT": {"classical", "photonic", "quantum", "thermodynamic"},
    "MLP": {"classical", "photonic", "thermodynamic"},
    "NORM": {"classical", "thermodynamic"},
    "HEAD": {"classical", "photonic", "thermodynamic"},
}
LAYER_KEYS = (
    "AR_EMB", "AR_ATT1", "AR_ATT2", "AR_MLP1", "AR_MLP2", "AR_NORM1", "AR_NORM2", "AR_HEAD",
)
METRIC_COLS = ("time_ms", "energy_j", "cost_usd")


def _kind(layer: str) -> str:
    return layer.removeprefix("AR_").rstrip("12")


def load_bench(csv_path: Optional[str] = None) -> dict:
    """{layer: {substrate: {time_ms, energy_j, cost_usd, quality}}}"""
    text = Path(csv_path).read_text() if csv_path else DEFAULT_BENCH_CSV
    bench: dict = {}
    for row in csv.DictReader(io.StringIO(text)):
        bench.setdefault(row["layer"], {})[row["substrate"]] = {
            k: float(row[k]) for k in (*METRIC_COLS, "quality")
        }
    return bench


def allowed_substrates(layers=LAYER_KEYS) -> dict:
    return {layer: IMPLEMENTABLE[_kind(layer)] for layer in layers}


def select_substrates(
    bench: dict,
    q_min: float = 0.80,
    weights=(1.0, 1.0, 1.0),
    layers=LAYER_KEYS,
    allowed: Optional[dict] = None,
) -> dict:
    """Per layer: the implementable substrate with quality >= q_min minimising
    Σ w_k · metric_k / max_candidates(metric_k). Falls back to the highest
    quality implementable substrate when none reaches q_min. `layers` /
    `allowed` ({layer: {substrates}}) default to the autoregressive table and
    are overridden by step3_hybrid_difussion.py."""
    allowed = allowed or allowed_substrates(layers)
    chosen = {}
    for layer in layers:
        rows = {
            s: m
            for s, m in bench.get(layer, {}).items()
            if s in allowed[layer]
        }
        if not rows:
            chosen[layer] = "classical"
            continue
        ok = {s: m for s, m in rows.items() if m["quality"] >= q_min}
        if not ok:
            chosen[layer] = max(rows, key=lambda s: rows[s]["quality"])
            continue
        top = {c: max(m[c] for m in ok.values()) or 1.0 for c in METRIC_COLS}
        chosen[layer] = min(
            ok,
            key=lambda s: sum(
                w * ok[s][c] / top[c] for w, c in zip(weights, METRIC_COLS)
            ),
        )
    return chosen


def parse_assign(spec: str, layers=LAYER_KEYS, allowed: Optional[dict] = None) -> dict:
    allowed = allowed or allowed_substrates(layers)
    out = {}
    for item in filter(None, (p.strip() for p in spec.split(","))):
        layer, _, sub = item.partition("=")
        layer, sub = layer.strip().upper(), sub.strip().lower()
        if layer not in layers:
            raise ValueError(f"--assign: unknown layer {layer!r} (choose from {tuple(layers)})")
        if sub not in allowed[layer]:
            raise ValueError(
                f"--assign: {layer} cannot run on {sub!r} "
                f"(implemented: {sorted(allowed[layer])})"
            )
        out[layer] = sub
    return out


def build_substrate_map(cfg: dict) -> dict:
    bench = load_bench(cfg.get("bench_csv"))
    if cfg["policy"] == "fixed":
        smap = {layer: "classical" for layer in LAYER_KEYS}
    else:
        weights = (cfg["w_time"], cfg["w_energy"], cfg["w_cost"])
        smap = select_substrates(bench, cfg["q_min"], weights)
    smap.update(parse_assign(cfg["assign"]))
    return smap


def log_substrate_map(smap: dict, bench: dict, layers=LAYER_KEYS) -> None:
    tot = {c: 0.0 for c in (*METRIC_COLS, "quality")}
    for layer in layers:
        row = bench.get(layer, {}).get(smap[layer])
        tag = "" if row else "  (not in benchmark)"
        log.info(f"    {layer:<9s} → {smap[layer]:<14s}{tag}")
        if row:
            for c in tot:
                tot[c] += row[c]
    n = len(layers)
    log.info(
        f"    benchmark total: time={tot['time_ms']:.2f} ms  energy={tot['energy_j']:.2f} J  "
        f"cost=${tot['cost_usd']:.4f}  mean quality={tot['quality'] / n:.3f}"
    )


def block_group(i: int, n_layer: int) -> str:
    return "1" if i < (n_layer + 1) // 2 else "2"


# ════════════════════════════════════════════════════════════════════════════
# Profiling (same record schema as the other trainers)
# ════════════════════════════════════════════════════════════════════════════
_profile_active = False
_profile_accum: dict = {}


def timed(name: str, fn, *args, **kwargs):
    if not _profile_active:
        return fn(*args, **kwargs)
    cuda = torch.cuda.is_available()
    if cuda:
        torch.cuda.synchronize()
    t0 = perf_counter()
    out = fn(*args, **kwargs)
    if cuda:
        torch.cuda.synchronize()
    acc = _profile_accum.setdefault(name, {"time": 0.0, "count": 0})
    acc["time"] += perf_counter() - t0
    acc["count"] += 1
    return out


# ════════════════════════════════════════════════════════════════════════════
# Substrate implementations
# ════════════════════════════════════════════════════════════════════════════
class ClassicalMHA(nn.Module):
    """Causal multi-head attention on plain torch."""

    def __init__(self, cfg: dict):
        super().__init__()
        d, h = cfg["n_embd"], cfg["n_head"]
        assert d % h == 0
        self.n_head = h
        self.dropout = cfg["dropout"]
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.n_head, C // self.n_head).permute(2, 0, 3, 1, 4)
        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=True
        )
        return self.proj(out.transpose(1, 2).reshape(B, T, C))


class ClassicalFFN(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        d = cfg["n_embd"]
        self.fc1 = nn.Linear(d, 4 * d)
        self.fc2 = nn.Linear(4 * d, d)
        self.drop = nn.Dropout(cfg["dropout"])

    def forward(self, x):
        return self.drop(self.fc2(self.drop(F.gelu(self.fc1(x)))))


def _causal_dwconv(x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """Depthwise causal conv: x (B, C, T), kernel (C, s) → (B, C, T)."""
    s = kernel.shape[-1]
    return F.conv1d(F.pad(x, (s - 1, 0)), kernel[:, None, :], groups=x.shape[1])


class AFTConv(nn.Module):
    """PyTorch port of the thermodynamic model's causal AFT-conv attention:
        ctx_t = Σ_{τ≤t} w_{t−τ} e^{K_τ} V_τ / Σ_{τ≤t} w_{t−τ} e^{K_τ},  w = 1 + exp(w_conv) − 1."""

    def __init__(self, cfg: dict):
        super().__init__()
        d, h = cfg["n_embd"], cfg["n_head"]
        assert d % h == 0
        self.dim, self.heads = d, h
        self.qkv_proj = nn.Linear(d, 2 * d + h)
        self.out_proj = nn.Linear(d, d)
        self.w_conv = nn.Parameter(torch.zeros(h, cfg["aft_ksize"]))

    def forward(self, x):
        B, T, C = x.shape
        h, hd = self.heads, C // self.heads
        Q, V, K = torch.split(self.qkv_proj(x).float(), [C, C, h], dim=-1)
        eK = torch.exp(torch.tanh(K))  # (B, T, h)
        eKV = eK[..., None] * V.reshape(B, T, h, hd)
        kernel = torch.exp(self.w_conv) - 1.0
        num = _causal_dwconv(
            eKV.reshape(B, T, C).transpose(1, 2), kernel.repeat_interleave(hd, dim=0)
        ).transpose(1, 2).reshape(B, T, h, hd) + torch.cumsum(eKV, dim=1)
        den = _causal_dwconv(eK.transpose(1, 2), kernel).transpose(1, 2) + torch.cumsum(eK, dim=1)
        ctx = num / (den[..., None] + 1e-6)
        return self.out_proj((torch.tanh(Q.reshape(B, T, h, hd)) * ctx).reshape(B, T, C))


class ThermoMLP(nn.Module):
    """SiLU MLP as in the thermodynamic model."""

    def __init__(self, cfg: dict):
        super().__init__()
        d = cfg["n_embd"]
        self.fc1 = nn.Linear(d, 4 * d)
        self.fc2 = nn.Linear(4 * d, d)
        self.drop = nn.Dropout(cfg["dropout"])

    def forward(self, x):
        return self.drop(self.fc2(F.silu(self.fc1(x))))


class DyT(nn.Module):
    """Dynamic Tanh normalisation: weight · tanh(α x) + bias."""

    def __init__(self, dim: int, alpha_init: float = 0.5):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        return self.weight * torch.tanh(self.alpha * x) + self.bias


def make_attention(sub: str, cfg: dict, layer_idx: int) -> nn.Module:
    if sub == "classical":
        return ClassicalMHA(cfg)
    if sub == "photonic":
        return ph.PhotonicMHA(cfg)
    if sub == "thermodynamic":
        return AFTConv(cfg)
    if sub == "quantum":
        if qu is None:
            raise ImportError("quantum substrate needs pennylane: pip install pennylane")
        return qu.QuantumFourierMixer(cfg, layer_idx)
    raise ValueError(f"no attention for substrate {sub!r}")


def make_mlp(sub: str, cfg: dict) -> nn.Module:
    if sub == "classical":
        return ClassicalFFN(cfg)
    if sub == "photonic":
        return ph.PhotonicFFN(cfg)
    if sub == "thermodynamic":
        return ThermoMLP(cfg)
    raise ValueError(f"no MLP for substrate {sub!r}")


def make_norm(sub: str, d: int) -> nn.Module:
    if sub == "classical":
        return nn.LayerNorm(d)
    if sub == "thermodynamic":
        return DyT(d)
    raise ValueError(f"no norm for substrate {sub!r}")


def make_head(sub: str, tok_emb: nn.Embedding, cfg: dict) -> nn.Module:
    """LM head tied to the embedding. photonic → Q.ANT matmul; classical and
    thermodynamic → plain digital linear."""
    V, d = tok_emb.weight.shape
    head = (
        ph.make_qant_linear(d, V, cfg, bias=False)
        if sub == "photonic"
        else nn.Linear(d, V, bias=False)
    )
    head.weight = tok_emb.weight
    return head


class HybridBlock(nn.Module):
    """Pre-norm block; attention, MLP and norms may each sit on another substrate."""

    def __init__(self, cfg: dict, smap: dict, layer_idx: int):
        super().__init__()
        g = block_group(layer_idx, cfg["n_layer"])
        self.s_att, self.s_mlp, self.s_norm = (
            smap[f"AR_ATT{g}"], smap[f"AR_MLP{g}"], smap[f"AR_NORM{g}"],
        )
        d = cfg["n_embd"]
        self.ln1 = make_norm(self.s_norm, d)
        self.attn = make_attention(self.s_att, cfg, layer_idx)
        self.ln2 = make_norm(self.s_norm, d)
        self.ffn = make_mlp(self.s_mlp, cfg)

    def _forward(self, x):
        n1 = timed(f"{self.s_norm}_norm", self.ln1, x)
        x = x + timed(f"{self.s_att}_attention", self.attn, n1)
        n2 = timed(f"{self.s_norm}_norm", self.ln2, x)
        return x + timed(f"{self.s_mlp}_mlp", self.ffn, n2)

    def forward(self, x):
        return timed("transformer_block_total", self._forward, x)


class HybridTransformerLM(nn.Module):
    """Token-level Transformer with a per-layer substrate assignment."""

    def __init__(self, vocab_size: int, cfg: dict, smap: dict):
        super().__init__()
        d, T = cfg["n_embd"], cfg["seq_len"]
        self.smap = dict(smap)
        self.tok_emb = nn.Embedding(vocab_size, d)
        self.pos_emb = nn.Embedding(T, d)
        self.drop = nn.Dropout(cfg["dropout"])
        self.blocks = nn.ModuleList(
            [HybridBlock(cfg, smap, i) for i in range(cfg["n_layer"])]
        )
        self.ln_f = make_norm(smap["AR_NORM2"], d)
        self.lm_head = make_head(smap["AR_HEAD"], self.tok_emb, cfg)

        self.apply(self._init)
        n = sum(p.numel() for p in self.parameters() if p.requires_grad)
        log.info(f"HybridTransformerLM: {n:,} parameters")

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
            if getattr(m, "bias", None) is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, idx, targets=None, ablate_layer: Optional[int] = None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = timed("classical_embedding", lambda: self.tok_emb(idx) + self.pos_emb(pos))
        x = self.drop(x)
        for i, blk in enumerate(self.blocks):
            if i == ablate_layer:
                continue
            x = blk(x)
        logits = timed(
            f"{self.smap['AR_HEAD']}_lm_head", lambda: self.lm_head(self.ln_f(x))
        )
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=PAD_ID
            )
        return logits, loss

    def set_nonidealities(self, phase_std, gamma_std, crosstalk):
        """Propagate the hardware-noise curriculum to every Q.ANT layer."""
        for m in self.modules():
            if isinstance(m, ph.QantLinear):
                m.set_phase_variation(phase_std)
                m.set_gamma_noise(gamma_std)
                m.set_crosstalk_factor(crosstalk)


CFG = {
    # seq_len is set at runtime from data/tokenized/metadata.json (max_seq_len - 1).
    "n_embd": 128,
    "n_head": 4,
    "n_layer": 4,
    "dropout": 0.1,
    "n_qubits": 8,  # quantum attention
    "sim_backend": "torch",  # quantum attention: "torch" | "pennylane"
    "aft_ksize": 4,  # thermodynamic attention
    "qant_device_id": 0,  # photonic layers
    "phase_noise_std": 0.01,
    "gamma_noise_std": 0.002,
    "crosstalk_factor": 0.01,
    # substrate selection
    "policy": "auto",  # "auto": from the benchmark | "fixed": all classical
    "q_min": 0.80,
    "w_time": 1.0,
    "w_energy": 1.0,
    "w_cost": 1.0,
    "assign": "",
    "bench_csv": None,
    "batch_size": 16,
    "max_steps": 5_000,
    "eval_every": 500,
    "max_eval_batches": 20,
    "lr": 3e-4,
    "weight_decay": 1e-1,
    "grad_clip": 1.0,
    "warmup_steps": 200,
    "log_every": 10,
    "save_every": 500,
    "use_amp": True,
    "use_val": True,
    "profile_every": 500,
    "enable_block_ablation": True,
    "ablation_batches": 2,
    "cost_per_wh": 1.031,
    "resume": True,
    "seed": 42,
}
# Architecture keys restored from a checkpoint on resume / used by inference.py
ARCH_KEYS = ("n_embd", "n_head", "n_layer", "n_qubits", "sim_backend", "aft_ksize", "seq_len")


# ════════════════════════════════════════════════════════════════════════════
# Diagnostics
# ════════════════════════════════════════════════════════════════════════════
def _raw(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model


def _global_norm(tensors) -> float:
    norms = [t.detach().float().norm(2) for t in tensors]
    return torch.stack(norms).norm(2).item() if norms else 0.0


def _param_snapshot_per_block(model: nn.Module) -> dict:
    return {
        i: torch.cat([p.detach().clone().flatten() for p in b.parameters()])
        for i, b in enumerate(_raw(model).blocks)
    }


def compute_param_update_ratio(model: nn.Module, pre: dict) -> dict:
    out = {}
    for i, b in enumerate(_raw(model).blocks):
        post = torch.cat([p.detach().flatten() for p in b.parameters()])
        delta = (post - pre[i]).float().norm(2).item()
        norm = post.float().norm(2).item()
        out[f"transformer_block_{i}"] = {
            "param_norm": round(norm, 4),
            "update_norm": round(delta, 6),
            "ratio": round(delta / norm if norm > 0 else 0.0, 8),
        }
    return out


def _per_block_grad_norms(model: nn.Module) -> dict:
    return {
        f"transformer_block_{i}": _global_norm(
            p.grad for p in b.parameters() if p.grad is not None
        )
        for i, b in enumerate(_raw(model).blocks)
    }


def compute_grad_norm_and_clip(model, optimizer, scaler, grad_clip, compute_per_block=False):
    scaler.unscale_(optimizer)
    grad_norm = _global_norm(p.grad for p in model.parameters() if p.grad is not None)
    per_block = _per_block_grad_norms(model) if compute_per_block else None
    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    return grad_norm, per_block


def compute_step_cost_usd(energy_j: Optional[float], cost_per_wh: float) -> Optional[float]:
    return None if energy_j is None else energy_j / 3_600.0 * cost_per_wh


def _collect_batches(loader_obj, n: int):
    it, out = iter(loader_obj), []
    for _ in range(n):
        try:
            out.append(next(it))
        except StopIteration:
            break
    return out


@torch.no_grad()
def compute_block_ablation_ppl(model, batches, device, cfg, world_size):
    """Loss/ppl with each block skipped in turn vs. the full model (ΔPPL)."""
    raw = _raw(model)
    was_training = model.training
    model.eval()
    amp_ctx = (
        torch.autocast("cuda", dtype=torch.float16)
        if cfg["use_amp"] and device.type == "cuda"
        else nullcontext()
    )

    def _avg_loss(ablate_layer):
        total_loss = torch.zeros(1, device=device)
        total_tokens = torch.zeros(1, device=device)
        for x, y in batches:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with amp_ctx:
                logits, _ = raw(x, ablate_layer=ablate_layer)
                loss_sum = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)), y.reshape(-1),
                    ignore_index=PAD_ID, reduction="sum",
                )
            total_loss += loss_sum.detach()
            total_tokens += (y.reshape(-1) != PAD_ID).sum()
        if world_size > 1 and dist.is_available() and dist.is_initialized():
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_tokens, op=dist.ReduceOp.SUM)
        return (total_loss / total_tokens.clamp(min=1)).item()

    base_loss = _avg_loss(None)
    base_ppl = math.exp(min(base_loss, 20))
    per_layer = {}
    for i in range(len(raw.blocks)):
        loss_i = _avg_loss(i)
        ppl_i = math.exp(min(loss_i, 20))
        per_layer[f"transformer_block_{i}"] = {
            "substrates": {
                "attention": raw.blocks[i].s_att,
                "mlp": raw.blocks[i].s_mlp,
                "norm": raw.blocks[i].s_norm,
            },
            "ablated_loss": round(loss_i, 5),
            "ablated_ppl": round(ppl_i, 3),
            "delta_loss": round(loss_i - base_loss, 5),
            "delta_ppl": round(ppl_i - base_ppl, 3),
        }
    if was_training:
        model.train()
    return base_loss, base_ppl, per_layer


def substrate_breakdown(components: dict, total_time: float, energy_j: Optional[float]) -> dict:
    """Time/energy per substrate, summed over its profiled components (block
    totals and the dataset fetch are excluded)."""
    out: dict = {}
    for name, c in components.items():
        sub = name.split("_", 1)[0]
        if sub not in SUBSTRATES:
            continue
        acc = out.setdefault(sub, {"time_s": 0.0, "energy_j": 0.0 if energy_j is not None else None})
        acc["time_s"] += c["time_s"]
        if energy_j is not None and c["energy_j"] is not None:
            acc["energy_j"] += c["energy_j"]
    for acc in out.values():
        acc["pct_of_step"] = _r(acc["time_s"] / total_time * 100, 3) if total_time > 0 else None
        acc["time_s"] = round(acc["time_s"], 6)
        acc["energy_j"] = _r(acc["energy_j"], 6)
    return out


def log_profile_and_diagnostics(
    *, step, epoch, cfg, model, total_time, avg_power, batch_tokens,
    cumulative_energy_j, cumulative_cost_usd, vram_gb,
    per_block_grad_norm, per_block_update_ratio,
):
    """Write profile / layer_energy_cost / gradient_contribution /
    param_update_ratio records for one profiled step (same schema as the other
    trainers; the profile also carries a per-substrate breakdown)."""
    energy_j = avg_power * total_time if avg_power is not None else None
    cost_usd = compute_step_cost_usd(energy_j, cfg["cost_per_wh"])
    energy_per_token = energy_j / max(batch_tokens, 1) if energy_j is not None else None

    def _share(t):
        pct = (t / total_time) * 100 if total_time > 0 else None
        e = (t / total_time) * energy_j if (energy_j is not None and total_time > 0) else None
        return pct, e

    components = {}
    for name, vals in _profile_accum.items():
        t, c = vals["time"], vals["count"]
        pct, e = _share(t)
        components[name] = {
            "time_s": round(t, 6),
            "calls": c,
            "avg_time_ms": round((t / c) * 1000, 4) if c else 0.0,
            "pct_of_step": _r(pct, 3),
            "energy_j": _r(e, 6),
            "cost_usd": _r(compute_step_cost_usd(e, cfg["cost_per_wh"]), 8),
        }
    by_substrate = substrate_breakdown(components, total_time, energy_j)

    raw = _raw(model)
    per_layer_ec = {}
    blk = _profile_accum.get("transformer_block_total")
    if blk:
        t_each = blk["time"] / max(len(raw.blocks), 1)
        pct, e = _share(t_each)
        for i, b in enumerate(raw.blocks):
            per_layer_ec[f"transformer_block_{i}"] = {
                "substrates": {"attention": b.s_att, "mlp": b.s_mlp, "norm": b.s_norm},
                "time_ms": round(t_each * 1000, 4),
                "pct_of_step": _r(pct, 3),
                "energy_j": _r(e, 6),
                "cost_usd": _r(compute_step_cost_usd(e, cfg["cost_per_wh"]), 8),
            }

    base = {"model_type": MODEL_TYPE, "step": step, "epoch": epoch}
    log_metrics(
        {
            "record_type": "profile", **base, "split": "profile",
            "profile_every": cfg["profile_every"],
            "total_time_s": round(total_time, 6),
            "vram_gb": round(vram_gb, 3),
            "avg_power_w": _r(avg_power, 2),
            "energy_j": _r(energy_j, 6),
            "hardware_energy_joules_per_token": _r(energy_per_token, 8),
            "cost_per_wh": cfg["cost_per_wh"],
            "step_cost_usd": _r(cost_usd, 8),
            "cumulative_energy_j": round(cumulative_energy_j, 3),
            "cumulative_cost_usd": round(cumulative_cost_usd, 6),
            "components": components,
            "substrates": by_substrate,
        }
    )
    log_metrics(
        {"record_type": "layer_energy_cost", **base, "split": "layer_energy_cost",
         "per_layer": per_layer_ec}
    )

    log.info(
        f"── Component profile @ step {step} "
        f"(training_pipeline_total = {total_time * 1000:.2f} ms) ──"
    )
    for name, v in sorted(components.items(), key=lambda kv: -kv[1]["time_s"]):
        pct = f"{v['pct_of_step']:>6.2f}%" if v["pct_of_step"] is not None else "   n/a"
        log.info(f"    {name:<32s} {v['time_s'] * 1000:>11.3f} ms  ({pct} of step)  x{v['calls']}")
    for sub, v in sorted(by_substrate.items(), key=lambda kv: -kv[1]["time_s"]):
        e_str = f"  {v['energy_j']:.6f} J" if v["energy_j"] is not None else ""
        log.info(f"    [substrate] {sub:<14s} {v['time_s'] * 1000:>9.3f} ms{e_str}")
    if energy_j is not None:
        log.info(f"    est. energy:          {energy_j:.6f} J  (avg power {avg_power:.1f} W)")
        log.info(f"    est. cost (step):    ${cost_usd:.8f}  (@${cfg['cost_per_wh']:.4f}/Wh)")
    else:
        log.info("    est. energy/cost: n/a (install pynvml for GPU power sampling)")

    if per_block_grad_norm is not None:
        log_metrics(
            {"record_type": "gradient_contribution", **base, "split": "gradient_contribution",
             "per_layer": {k: round(v, 4) for k, v in per_block_grad_norm.items()}}
        )
        log.info(f"── Gradient contribution ‖∂L/∂θ_i‖ @ step {step} ──")
        for name, g in sorted(per_block_grad_norm.items(), key=lambda kv: -kv[1]):
            log.info(f"    {name:<20s} grad_norm={g:>10.4f}")

    if per_block_update_ratio is not None:
        log_metrics(
            {"record_type": "param_update_ratio", **base, "split": "param_update_ratio",
             "per_layer": per_block_update_ratio}
        )
        log.info(f"── Parameter/update quality ‖Δθ_i‖/‖θ_i‖ @ step {step} ──")
        for name, v in sorted(per_block_update_ratio.items(), key=lambda kv: -kv[1]["ratio"]):
            log.info(
                f"    {name:<20s} ratio={v['ratio']:.6f}  "
                f"(update_norm={v['update_norm']:.4f}, param_norm={v['param_norm']:.1f})"
            )
    return energy_j, cost_usd, energy_per_token


# ════════════════════════════════════════════════════════════════════════════
# Checkpoints
# ════════════════════════════════════════════════════════════════════════════
def latest_checkpoint() -> Optional[Path]:
    ckpts = sorted(CKPT_DIR.glob("ckpt_step_*.pt"))
    return ckpts[-1] if ckpts else None


def _atomic_save(obj, path: Path) -> None:
    """Temp file + rename, so a crash mid-save never leaves a truncated checkpoint."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _arch(cfg: dict) -> dict:
    return {k: cfg[k] for k in ARCH_KEYS if k in cfg}


def save_checkpoint(model, optimizer, scaler, step, epoch, loss, cfg, rank=0):
    if rank != 0:
        return
    path = CKPT_DIR / f"ckpt_step_{step:07d}.pt"
    raw = _raw(model)
    _atomic_save(
        {
            "step": step,
            "epoch": epoch,
            "model_state": raw.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scaler_state": scaler.state_dict(),
            "loss": loss,
            "model_type": MODEL_TYPE,
            "substrate_map": raw.smap,
            "arch": _arch(cfg),
            "timestamp": datetime.datetime.utcnow().isoformat(),
        },
        path,
    )
    log.info(f"  💾 Checkpoint saved → {path.name}  (step={step}, loss={loss:.4f})")
    for p in sorted(CKPT_DIR.glob("ckpt_step_*.pt"))[:-5]:
        p.unlink()


def save_best_checkpoint(model, step, epoch, val_loss, vocab_size, cfg, rank=0):
    if rank != 0:
        return
    path = CKPT_DIR / "ckpt_best.pt"
    raw = _raw(model)
    _atomic_save(
        {
            "step": step,
            "epoch": epoch,
            "model_state": raw.state_dict(),
            "val_loss": val_loss,
            "model_type": MODEL_TYPE,
            "substrate_map": raw.smap,
            "arch": _arch(cfg),
            "timestamp": datetime.datetime.utcnow().isoformat(),
        },
        path,
    )
    meta = {
        **cfg,
        "vocab_size": vocab_size,
        "best_val_loss": val_loss,
        "step": step,
        "model_type": MODEL_TYPE,
        "substrate_map": raw.smap,
    }
    with open(CKPT_DIR / "model_config.json", "w") as f:
        json.dump(meta, f, indent=2)
    log.info(f"  🏆 New best checkpoint → {path.name}  (step={step}, val_loss={val_loss:.4f})")


def load_checkpoint(ckpt, model, optimizer, scaler):
    _raw(model).load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    scaler.load_state_dict(ckpt["scaler_state"])
    return ckpt["step"], ckpt["epoch"], ckpt.get("loss", float("nan"))


DEVICE = (
    torch.device("cuda")
    if torch.cuda.is_available()
    else torch.device("mps")
    if torch.backends.mps.is_available()
    else torch.device("cpu")
)
log.info(f"device={DEVICE}")


# ════════════════════════════════════════════════════════════════════════════
# Training
# ════════════════════════════════════════════════════════════════════════════
def _make_loader(ds, cfg, world_size, rank, device, train: bool):
    sampler = (
        DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=train, seed=cfg["seed"])
        if world_size > 1
        else None
    )
    loader = DataLoader(
        ds,
        batch_size=cfg["batch_size"],
        sampler=sampler,
        shuffle=(train and sampler is None),
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    return loader, sampler


def train(cfg):
    global DEVICE, _profile_active

    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    rank, world_size, local_rank, device = setup_distributed()
    is_main = rank == 0
    DEVICE = device

    meta = read_tokenized_metadata()
    cfg["seq_len"] = meta["max_seq_len"] - 1
    vocab = meta["vocab_size"]

    # Resume: the checkpoint's architecture and substrate map win over the CLI,
    # so the weights always load into the model they were trained as.
    ckpt_path = latest_checkpoint() if cfg["resume"] else None
    ckpt = None
    smap = build_substrate_map(cfg)
    if ckpt_path:
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        if ckpt.get("model_type") != MODEL_TYPE:
            raise ValueError(f"{ckpt_path.name} is not a hybrid checkpoint — use --no_resume")
        if ckpt["substrate_map"] != smap and is_main:
            log.warning("Requested substrate map differs from the checkpoint — using the checkpoint's.")
        smap = ckpt["substrate_map"]
        cfg.update(ckpt["arch"])

    used = set(smap.values())
    if is_main:
        log.info("╔══════════════════════════════════════════════════════════════╗")
        log.info("║  STEP 3 (hybrid) — Multi-substrate Transformer Training      ║")
        log.info("╚══════════════════════════════════════════════════════════════╝")
        log.info(f"Device: {device}  |  World size: {world_size}  |  Q.ANT backend: {ph._BACKEND}")
        log.info("Substrate assignment:")
        log_substrate_map(smap, load_bench(cfg.get("bench_csv")))
        log.info(f"Energy cost rate: ${cfg['cost_per_wh']:.4f}/Wh")
        log.info(
            f"Component profiling / gradient contribution / update ratio every "
            f"{cfg['profile_every']} steps (and on the first step)."
        )
    if "photonic" in used:
        ph._init_qant(cfg)

    train_ds = TokenDataset("train")
    loader, sampler = _make_loader(train_ds, cfg, world_size, rank, device, train=True)

    val_loader = None
    if cfg["use_val"]:
        try:
            val_loader, _ = _make_loader(TokenDataset("val"), cfg, world_size, rank, device, train=False)
            if is_main:
                log.info(f"Validation: every {cfg['eval_every']} steps, max_batches={cfg['max_eval_batches']}")
        except FileNotFoundError:
            if is_main:
                log.warning("No val.npy found — skipping validation.")

    model = HybridTransformerLM(vocab, cfg, smap).to(device)
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    decay, no_decay = [], []
    for p in model.parameters():
        if p.requires_grad:
            (decay if p.ndim >= 2 else no_decay).append(p)
    opt = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": cfg["weight_decay"]},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=cfg["lr"], betas=(0.9, 0.95), eps=1e-8,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=cfg["use_amp"] and device.type == "cuda")

    start_step, start_epoch, last_loss = 0, 0, float("nan")
    if ckpt is not None:
        start_step, start_epoch, last_loss = load_checkpoint(ckpt, model, opt, scaler)
        del ckpt
        if is_main:
            log.info(f"Resumed from {ckpt_path.name} (step {start_step})")

    best_val_loss = float("inf")
    best_ckpt_path = CKPT_DIR / "ckpt_best.pt"
    if best_ckpt_path.exists():
        try:
            best_val_loss = torch.load(str(best_ckpt_path), map_location="cpu", weights_only=False).get(
                "val_loss", float("inf")
            )
        except Exception:
            pass

    interrupted = False

    def _signal_handler(sig, frame):
        nonlocal interrupted
        interrupted = True
        if is_main:
            log.warning(f"\n⚠️  Signal {sig} caught — saving checkpoint and exiting …")

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    def noise_at(step):
        r = min(1.0, 2.0 * step / cfg["max_steps"])
        return (
            r * cfg["phase_noise_std"],
            r * cfg["gamma_noise_std"],
            r * cfg["crosstalk_factor"],
        )

    cumulative_energy_j = cumulative_cost_usd = 0.0
    step_energy_j = step_cost_usd = step_energy_per_token = None
    last_val_loss = last_val_ppl = last_val_step = None

    model.train()
    step, epoch = start_step, start_epoch
    running_loss = 0.0 if math.isnan(last_loss) else last_loss
    t_start = t_log = time.time()
    last_log_step = step
    tokens_processed = step * cfg["batch_size"] * cfg["seq_len"] * world_size
    data_iter = iter(loader)
    if is_main:
        log.info(f"Checkpoints → {CKPT_DIR}  (every {cfg['save_every']} steps; starting at step {step})")
        log.info(f"Metrics     → {METRICS_PATH}  (every {cfg['log_every']} steps)")

    try:
        while step < cfg["max_steps"] and not interrupted:
            scheduled = (step + 1) % cfg["profile_every"] == 0
            profile_now = scheduled or step == start_step
            log_now = is_main and ((step + 1) % cfg["log_every"] == 0 or step == start_step)
            _profile_active = profile_now
            if profile_now:
                _profile_accum.clear()

            step_t0 = power_start = None
            if profile_now and is_main:
                if device.type == "cuda":
                    torch.cuda.synchronize()
                power_start = _sample_power_w()
                step_t0 = perf_counter()

            fetch_t0 = perf_counter()
            try:
                x, y = next(data_iter)
            except StopIteration:
                epoch += 1
                if sampler is not None:
                    sampler.set_epoch(epoch)
                data_iter = iter(loader)
                x, y = next(data_iter)
            if profile_now:
                acc = _profile_accum.setdefault("dataset", {"time": 0.0, "count": 0})
                acc["time"] += perf_counter() - fetch_t0
                acc["count"] += 1

            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            lr_now = get_lr(step, cfg)
            for pg in opt.param_groups:
                pg["lr"] = lr_now
            if "photonic" in used:
                _raw(model).set_nonidealities(*noise_at(step))

            opt.zero_grad(set_to_none=True)
            amp_ctx = (
                torch.autocast("cuda", dtype=torch.float16)
                if cfg["use_amp"] and device.type == "cuda"
                else nullcontext()
            )
            with amp_ctx:
                logits, loss = model(x, y)
            del logits
            timed("backpropagation", scaler.scale(loss).backward)

            pre_block_snapshot = _param_snapshot_per_block(model) if (profile_now and is_main) else None
            grad_norm, per_block_grad_norm = compute_grad_norm_and_clip(
                model, opt, scaler, cfg["grad_clip"], compute_per_block=(profile_now and is_main)
            )

            def _opt_step():
                scaler.step(opt)
                scaler.update()

            timed("optimizer", _opt_step)

            step += 1
            running_loss = loss.item()
            batch_tokens = x.size(0) * cfg["seq_len"] * world_size
            tokens_processed += batch_tokens

            per_block_update_ratio = (
                compute_param_update_ratio(model, pre_block_snapshot)
                if pre_block_snapshot is not None
                else None
            )

            if val_loader is not None and step % cfg["eval_every"] == 0:
                val_loss, val_ppl = evaluate(model, val_loader, device, cfg, world_size)
                last_val_loss, last_val_ppl, last_val_step = val_loss, val_ppl, step
                if is_main:
                    log_metrics(
                        {
                            "record_type": "val", "model_type": MODEL_TYPE, "step": step,
                            "epoch": epoch, "split": "val",
                            "val_loss": round(val_loss, 5), "val_ppl": round(val_ppl, 3),
                            "final": False,
                        }
                    )
                    log.info(f"           ↳ val_loss={val_loss:.4f} | val_ppl={val_ppl:.2f}")
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        save_best_checkpoint(model, step, epoch, val_loss, vocab, cfg, rank)

            if log_now:
                n_steps = max(step - last_log_step, 1)
                last_log_step = step
                now = time.time()
                dt = now - t_log
                t_log = now
                elapsed = now - t_start

                power_w = _sample_power_w()
                if power_w is not None:
                    interval_j = power_w * dt
                    step_energy_j = interval_j / n_steps
                    step_cost_usd = step_energy_j / 3_600.0 * cfg["cost_per_wh"]
                    step_energy_per_token = step_energy_j / max(batch_tokens, 1)
                    cumulative_energy_j += interval_j
                    cumulative_cost_usd += interval_j / 3_600.0 * cfg["cost_per_wh"]
                else:
                    step_energy_j = step_cost_usd = step_energy_per_token = None

                ppl = math.exp(min(running_loss, 20))
                vram_gb = torch.cuda.memory_allocated(device) / 1e9 if device.type == "cuda" else 0.0
                param_norm = _global_norm(model.parameters())

                record = {
                    "record_type": "train",
                    "model_type": MODEL_TYPE,
                    "step": step,
                    "epoch": epoch,
                    "split": "train",
                    "tokens_processed": tokens_processed,
                    "loss": round(running_loss, 5),
                    "perplexity": round(ppl, 3),
                    "learning_rate": round(lr_now, 8),
                    "gradient_norm": round(grad_norm, 4),
                    "parameter_norm": round(param_norm, 4),
                    "val_loss": _r(last_val_loss, 5),
                    "val_ppl": _r(last_val_ppl, 3),
                    "val_step": last_val_step,
                    "vram_gb": round(vram_gb, 3),
                    "elapsed_s": round(elapsed, 1),
                    "steps_per_s": round((step - start_step) / elapsed, 2),
                    "hardware_energy_joules_per_token": _r(step_energy_per_token, 8),
                    "step_energy_j": _r(step_energy_j, 6),
                    "step_cost_usd": _r(step_cost_usd, 8),
                    "cumulative_energy_j": round(cumulative_energy_j, 3),
                    "cumulative_cost_usd": round(cumulative_cost_usd, 6),
                    "cost_per_wh": cfg["cost_per_wh"],
                    "substrate_map": smap,
                }
                if "photonic" in used:
                    ph_, gm_, ct_ = noise_at(step)
                    record.update(
                        phase_noise_sigma=round(ph_, 5),
                        gamma_noise_sigma=round(gm_, 5),
                        crosstalk_factor=round(ct_, 5),
                    )
                if "quantum" in used:
                    record["n_qubits"] = cfg["n_qubits"]
                log_metrics(record)

                val_str = (
                    f"val_loss={last_val_loss:.4f} | val_ppl={last_val_ppl:>8.2f}"
                    if last_val_loss is not None
                    else "val_loss=n/a    | val_ppl=n/a"
                )
                log.info(
                    f"step={step:>7,}/{cfg['max_steps']} | loss={running_loss:.4f} | "
                    f"ppl={ppl:>8.2f} | lr={lr_now:.2e} | vram={vram_gb:.2f}GB | "
                    f"grad={grad_norm:.3f} | param={param_norm:.1f} | "
                    f"{dt / n_steps:.2f}s/step | {val_str}"
                )

            if is_main and step % cfg["save_every"] == 0:
                save_checkpoint(model, opt, scaler, step, epoch, running_loss, cfg, rank)

            if profile_now and is_main:
                if device.type == "cuda":
                    torch.cuda.synchronize()
                power_end = _sample_power_w()
                avg_power = (
                    (power_start + power_end) / 2
                    if power_start is not None and power_end is not None
                    else None
                )
                log_profile_and_diagnostics(
                    step=step, epoch=epoch, cfg=cfg, model=model,
                    total_time=perf_counter() - step_t0, avg_power=avg_power,
                    batch_tokens=batch_tokens,
                    cumulative_energy_j=cumulative_energy_j,
                    cumulative_cost_usd=cumulative_cost_usd,
                    vram_gb=torch.cuda.memory_allocated(device) / 1e9 if device.type == "cuda" else 0.0,
                    per_block_grad_norm=per_block_grad_norm,
                    per_block_update_ratio=per_block_update_ratio,
                )

            if profile_now:
                _profile_accum.clear()
                _profile_active = False

            # every rank runs the ablation (its losses are all-reduced)
            if scheduled and cfg["enable_block_ablation"]:
                src_loader = val_loader if val_loader is not None else loader
                ablation_batches = _collect_batches(src_loader, cfg["ablation_batches"])
                if ablation_batches:
                    ab_loss, ab_ppl, ab_per_layer = compute_block_ablation_ppl(
                        model, ablation_batches, device, cfg, world_size
                    )
                    if is_main:
                        log_metrics(
                            {
                                "record_type": "block_ablation", "model_type": MODEL_TYPE,
                                "step": step, "epoch": epoch, "split": "block_ablation",
                                "baseline_loss": round(ab_loss, 5),
                                "baseline_ppl": round(ab_ppl, 3),
                                "n_batches": len(ablation_batches),
                                "per_layer": ab_per_layer,
                            }
                        )
                        log.info(
                            f"── Block ablation ΔPPL @ step {step} "
                            f"(baseline_loss={ab_loss:.4f}, baseline_ppl={ab_ppl:.2f}) ──"
                        )
                        for name, v in sorted(ab_per_layer.items(), key=lambda kv: -kv[1]["delta_ppl"]):
                            s = v["substrates"]
                            log.info(
                                f"    {name:<20s} ΔPPL={v['delta_ppl']:>+9.3f}  "
                                f"ΔLoss={v['delta_loss']:>+9.5f}  "
                                f"[att={s['attention']} mlp={s['mlp']} norm={s['norm']}]"
                            )
                elif is_main:
                    log.info("    block ablation: skipped (no eval batches available)")

    except Exception as exc:
        log.exception(f"Training error at step {step}: {exc}")
        raise
    finally:
        _profile_active = False
        if val_loader is not None:
            val_loss, val_ppl = evaluate(model, val_loader, device, cfg, world_size)
            if is_main:
                log_metrics(
                    {
                        "record_type": "val", "model_type": MODEL_TYPE, "step": step,
                        "epoch": epoch, "split": "val",
                        "val_loss": round(val_loss, 5), "val_ppl": round(val_ppl, 3),
                        "final": True,
                    }
                )
                log.info(f"Final val_loss={val_loss:.4f} | val_ppl={val_ppl:.2f}")
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_best_checkpoint(model, step, epoch, val_loss, vocab, cfg, rank)

        if is_main:
            save_checkpoint(model, opt, scaler, step, epoch, running_loss, cfg, rank)
            log.info(
                f"✅ Training complete.  Final checkpoint saved.  "
                f"Total energy: {cumulative_energy_j / 3600:.4f} Wh  |  "
                f"Total cost: ${cumulative_cost_usd:.6f}"
            )
        if "photonic" in used:
            ph._release_qant(cfg)
        cleanup_distributed()

    return step, running_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_steps", type=int, default=CFG["max_steps"])
    parser.add_argument("--batch_size", type=int, default=CFG["batch_size"])
    parser.add_argument("--lr", type=float, default=CFG["lr"])
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--no_val", action="store_true")
    parser.add_argument("--eval_every", type=int, default=CFG["eval_every"])
    parser.add_argument("--max_eval_batches", type=int, default=CFG["max_eval_batches"])
    parser.add_argument("--save_every", type=int, default=CFG["save_every"])
    parser.add_argument("--cost_per_wh", type=float, default=CFG["cost_per_wh"])
    parser.add_argument("--no_resume", action="store_true")
    parser.add_argument("--n_layer", type=int, default=CFG["n_layer"])
    parser.add_argument("--n_head", type=int, default=CFG["n_head"])
    parser.add_argument("--n_qubits", type=int, default=CFG["n_qubits"])
    parser.add_argument("--sim_backend", choices=["torch", "pennylane"], default=CFG["sim_backend"])
    parser.add_argument("--profile_every", type=int, default=CFG["profile_every"])
    parser.add_argument("--no_block_ablation", action="store_true")
    parser.add_argument("--ablation_batches", type=int, default=CFG["ablation_batches"])
    parser.add_argument(
        "--policy", choices=["auto", "fixed"], default=CFG["policy"],
        help="auto: pick substrates from the benchmark; fixed: all classical (then --assign)",
    )
    parser.add_argument("--q_min", type=float, default=CFG["q_min"],
                        help="minimum benchmarked quality for the auto policy")
    parser.add_argument("--w_time", type=float, default=CFG["w_time"])
    parser.add_argument("--w_energy", type=float, default=CFG["w_energy"])
    parser.add_argument("--w_cost", type=float, default=CFG["w_cost"])
    parser.add_argument("--assign", default=CFG["assign"],
                        help="per-layer override, e.g. AR_ATT2=quantum,AR_MLP1=thermodynamic")
    parser.add_argument("--bench_csv", default=CFG["bench_csv"],
                        help="layer,substrate,time_ms,energy_j,cost_usd,quality CSV (default: built-in)")
    args = parser.parse_args()

    CFG["max_steps"] = args.max_steps
    CFG["batch_size"] = args.batch_size
    CFG["lr"] = args.lr
    if args.no_amp:
        CFG["use_amp"] = False
    if args.no_val:
        CFG["use_val"] = False
    CFG["eval_every"] = args.eval_every
    CFG["max_eval_batches"] = args.max_eval_batches
    CFG["save_every"] = args.save_every
    CFG["cost_per_wh"] = args.cost_per_wh
    if args.no_resume:
        CFG["resume"] = False
    CFG["n_layer"] = args.n_layer
    CFG["n_head"] = args.n_head
    CFG["n_qubits"] = args.n_qubits
    CFG["sim_backend"] = args.sim_backend
    CFG["profile_every"] = args.profile_every
    if args.no_block_ablation:
        CFG["enable_block_ablation"] = False
    CFG["ablation_batches"] = args.ablation_batches
    CFG["policy"] = args.policy
    CFG["q_min"] = args.q_min
    CFG["w_time"], CFG["w_energy"], CFG["w_cost"] = args.w_time, args.w_energy, args.w_cost
    CFG["assign"] = args.assign
    CFG["bench_csv"] = args.bench_csv

    final_step, final_loss = train(CFG)
    log.info(f"\nFinal step={final_step}  loss={final_loss:.4f}")
