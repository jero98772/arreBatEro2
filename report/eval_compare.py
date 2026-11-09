"""Uniform evaluation of models A, B, C.

python eval_compare.py --model A --gpu 1
Writes results/<model>.json
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PIPE = os.path.expanduser("~/Documents/llm_pipeline")
os.environ.setdefault("LLM_WORK", os.path.join(PIPE, "work"))
sys.path.insert(0, PIPE)
from common.model import load_model  # noqa: E402
from common.data import Tok, chat_prompt  # noqa: E402

AUX = os.path.join(PIPE, "work", "1_data_extraction", "aux")
SPECS = {
    "A": dict(ckpt=os.path.join(ROOT, "a.pt"),
              tok=os.path.expanduser("~/Desktop/QPT-LLM-prototype-no-lib/models/bpe_tokenizer.json")),
    "B": dict(ckpt=os.path.join(ROOT, "b.pt"), tok=os.path.join(ROOT, "tokens_b", "tokenizer.json")),
    # NOTE: tokens_c/ holds the smoke-run tokenizer (vocab 4096); the real one is in work_350m.
    "C": dict(ckpt=os.path.join(ROOT, "c.pt"),
              tok=os.path.join(PIPE, "work_350m", "3_tokenization", "tokenizer.json")),
}
DOMAINS = ["web", "code", "math", "wiki"]
TEXT_TOKENS = 150_000          # B-tokens of raw text per domain
BENCH_LIMIT = 2000


# ---------------------------------------------------------------- model A (copied from step3_model_training.py)
class AAttn(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.h, self.dh = h, d // h
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).reshape(B, T, 3, self.h, self.dh).permute(2, 0, 3, 1, 4).unbind(0)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(y.transpose(1, 2).reshape(B, T, C))


class ABlock(nn.Module):
    def __init__(self, d, h, dff):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = AAttn(d, h)
        self.ff = nn.Module()
        self.ff.net = nn.Sequential(nn.Linear(d, dff), nn.GELU(), nn.Dropout(0.0), nn.Linear(dff, d), nn.Dropout(0.0))
        self.ablate = False

    def forward(self, x):
        if self.ablate:
            return x
        x = x + self.attn(self.ln1(x))
        return x + self.ff.net(self.ln2(x))


class ModelA(nn.Module):
    def __init__(self, V=32000, T=512, d=512, h=8, L=12, dff=2048):
        super().__init__()
        self.max_len = T
        self.tok_emb = nn.Embedding(V, d)
        self.pos_emb = nn.Embedding(T, d)
        self.blocks = nn.ModuleList([ABlock(d, h, dff) for _ in range(L)])
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, V, bias=False)
        self.head.weight = self.tok_emb.weight

    def forward(self, idx, targets=None):
        T = idx.shape[1]
        x = self.tok_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device))
        for b in self.blocks:
            x = b(x)
        return self.head(self.ln_f(x)), None


class TokA:
    def __init__(self, path):
        from tokenizers import Tokenizer
        self.tk = Tokenizer.from_file(path)
        self.pad, self.bos, self.eos = 0, 2, 3
        self.vocab_size = self.tk.get_vocab_size()

    def encode(self, s):
        return self.tk.encode(s, add_special_tokens=False).ids

    def encode_batch(self, ss):
        return [e.ids for e in self.tk.encode_batch(ss, add_special_tokens=False)]

    def decode(self, ids, skip_special=True):
        return self.tk.decode(list(ids), skip_special_tokens=skip_special)


HF = {
    "gpt2": "openai-community/gpt2",
    "gpt2-medium": "openai-community/gpt2-medium",
    "pythia-160m": "EleutherAI/pythia-160m",
    "pythia-410m": "EleutherAI/pythia-410m",
    "opt-125m": "facebook/opt-125m",
    "smollm2-135m": "HuggingFaceTB/SmolLM2-135M",
    "smollm2-360m": "HuggingFaceTB/SmolLM2-360M",
}
FP32_MODELS = {"smollm2-360m"}   # NaN under fp16 autocast


class HFWrap(nn.Module):
    """Expose a transformers causal LM as forward(idx) -> (logits, None)."""
    def __init__(self, m, max_len, fp32=False):
        super().__init__()
        self.m, self.max_len, self.fp32 = m, max_len, fp32

    def forward(self, idx, targets=None):
        if self.fp32:  # bf16-trained model whose activations overflow fp16
            with torch.autocast("cuda", enabled=False):
                return self.m(input_ids=idx).logits.float(), None
        return self.m(input_ids=idx).logits, None


class HFTok:
    def __init__(self, t):
        self.t = t
        self.bos = t.bos_token_id if t.bos_token_id is not None else t.eos_token_id
        self.eos = t.eos_token_id
        self.pad = t.pad_token_id if t.pad_token_id is not None else self.eos
        self.vocab_size = len(t)

    def encode(self, s):
        return self.t(s, add_special_tokens=False)["input_ids"]

    def encode_batch(self, ss):
        return self.t(ss, add_special_tokens=False)["input_ids"]

    def decode(self, ids, skip_special=True):
        return self.t.decode(list(ids), skip_special_tokens=skip_special)


def load_hf(name, device):
    import transformers.utils.import_utils as iu  # the local torchaudio install is broken; tell
    import transformers.utils as tu               # transformers it is absent (text-only models)
    iu.is_torchaudio_available = tu.is_torchaudio_available = lambda: False
    from transformers import AutoModelForCausalLM, AutoTokenizer
    repo = HF[name]
    t = AutoTokenizer.from_pretrained(repo, local_files_only=True)
    m = AutoModelForCausalLM.from_pretrained(repo, local_files_only=True, dtype=torch.float32)
    m.to(device).eval()
    cfg = m.config
    max_len = min(getattr(cfg, "n_positions", None) or getattr(cfg, "max_position_embeddings", 2048), 2048)
    fp32 = name in FP32_MODELS
    w = HFWrap(m, max_len, fp32)
    w.blocks = []
    return w, HFTok(t), dict(repo=repo, max_len=max_len, fp32=fp32)


def load(name, device):
    if name in HF:
        return load_hf(name, device)
    s = SPECS[name]
    if name == "A":
        ck = torch.load(s["ckpt"], map_location="cpu", weights_only=False)
        m = ModelA()
        missing, unexpected = m.load_state_dict(ck["model_state"], strict=False)
        assert not [k for k in missing if "head" not in k], missing
        m.to(device).eval()
        info = dict(step=ck["step"], epoch=ck["epoch"], logged_loss=ck["loss"], max_len=512)
        return m, TokA(s["tok"]), info
    m, ck = load_model(s["ckpt"], device=device)
    m.eval()
    m.max_len = m.cfg.max_seq_len
    info = dict(step=ck["step"], cfg=ck["model_cfg"], max_len=m.cfg.max_seq_len)
    return m, Tok(s["tok"]), info


def blocks_of(m):
    return m.blocks


def set_ablate(m, i, on):
    b = m.blocks[i]
    if hasattr(b, "ablate_attn"):
        b.ablate_attn = b.ablate_ffn = on
    else:
        b.ablate = on


# ---------------------------------------------------------------- shared raw text
def load_texts():
    path = os.path.join(HERE, "results", "eval_texts.json")
    if os.path.exists(path):
        return json.load(open(path))
    tb = Tok(SPECS["B"]["tok"])
    texts = {}
    for d in DOMAINS:
        raw = np.fromfile(os.path.join(ROOT, "tokens_b", f"val_{d}.bin"), dtype=np.uint16)[:TEXT_TOKENS]
        # split into documents at <|bos|>/<|eos|> markers
        docs, cur = [], []
        for t in raw.tolist():
            if t in (tb.bos, tb.eos):
                if cur:
                    docs.append(cur)
                cur = []
            else:
                cur.append(t)
        if cur:
            docs.append(cur)
        texts[d] = [tb.decode(x, skip_special=True) for x in docs if len(x) > 8]
    # A's own held-out test split (already lower-cased by A's normalizer)
    ta = TokA(SPECS["A"]["tok"])
    arr = np.load(os.path.join(ROOT, "tokens_a", "test.npy"), mmap_mode="r")
    docs, n = [], 0
    for row in arr:
        ids = [int(t) for t in row if t > 5]
        docs.append(ta.decode(ids))
        n += len(ids)
        if n >= TEXT_TOKENS:
            break
    texts["a_test"] = docs
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(texts, open(path, "w"))
    return texts


@torch.no_grad()
def bpb(model, tok, docs, window, device, bs=8):
    """Bits per UTF-8 byte of the original text; docs concatenated with BOS separators,
    scored in non-overlapping windows of `window` tokens (each window starts with BOS)."""
    # Per-document scoring: every document starts at BOS, no cross-document context
    # (model A was trained on single padded documents, so packed streams would be unfair to it).
    nbytes = sum(len(d.encode("utf-8")) for d in docs)
    W = window
    chunks = []
    for ids in tok.encode_batch(docs):
        chunks += [ids[i:i + W - 1] for i in range(0, len(ids), W - 1)]
    chunks.sort(key=len)
    nll = 0.0
    ntok = 0
    for i in range(0, len(chunks), bs):
        group = chunks[i:i + bs]
        L = max(len(c) for c in group) + 1
        x = torch.full((len(group), L), tok.pad, dtype=torch.long)
        y = torch.full((len(group), L), -100, dtype=torch.long)
        for j, c in enumerate(group):
            seq = [tok.bos] + c
            x[j, :len(seq)] = torch.tensor(seq)
            y[j, :len(c)] = torch.tensor(c)
        x, y = x.to(device), y.to(device)
        with torch.autocast("cuda", dtype=torch.float16):
            logits, _ = model(x)
        ce = F.cross_entropy(logits.float().view(-1, logits.size(-1)), y.view(-1), ignore_index=-100, reduction="sum")
        nll += ce.item()
        ntok += (y != -100).sum().item()
    return dict(bpb=nll / math.log(2) / nbytes, nats_per_token=nll / ntok, tokens=ntok, bytes=nbytes,
                bytes_per_token=nbytes / ntok)


# ---------------------------------------------------------------- benchmarks (from eval/benchmarks.py, max_len aware)
def _items(name):
    rows = [json.loads(l) for l in open(os.path.join(AUX, f"eval_{name}.jsonl"))]
    out = []
    if name == "hellaswag":
        for r in rows:
            ends = r["endings"] if isinstance(r["endings"], list) else eval(r["endings"])
            out.append((r["ctx"], [" " + e for e in ends], int(r["label"])))
    elif name == "arc_easy":
        for r in rows:
            ch = r["choices"] if isinstance(r["choices"], dict) else eval(r["choices"])
            labels = list(ch["label"])
            if r["answerKey"] not in labels:
                continue
            out.append((f"Question: {r['question']}\nAnswer:", [" " + t for t in ch["text"]],
                        labels.index(r["answerKey"])))
    elif name == "piqa":
        for r in rows:
            out.append((f"Question: {r['goal']}\nAnswer:", [" " + r["sol1"], " " + r["sol2"]], int(r["label"])))
    return out


@torch.no_grad()
def score_choices(model, tok, ctx, choices, device, max_len):
    c_ids = tok.encode(ctx)
    seqs, spans = [], []
    for ch in choices:
        e = tok.encode(ch)
        seqs.append(([tok.bos] + c_ids + e)[-max_len:])
        spans.append(len(e))
    L = max(len(s) for s in seqs)
    x = torch.full((len(seqs), L), tok.pad, dtype=torch.long)
    for i, s in enumerate(seqs):
        x[i, :len(s)] = torch.tensor(s)
    x = x.to(device)
    with torch.autocast("cuda", dtype=torch.float16):
        logits, _ = model(x)
    lp = F.log_softmax(logits.float(), -1)
    res = []
    for i, s in enumerate(seqs):
        n, e = spans[i], len(s)
        res.append((lp[i, e - n - 1:e - 1].gather(-1, x[i, e - n:e][:, None]).sum().item(), n))
    return res


def multiple_choice(model, tok, name, device, max_len):
    items = _items(name)[:BENCH_LIMIT]
    c = cn = 0
    for ctx, choices, label in items:
        sc = score_choices(model, tok, ctx, choices, device, max_len)
        c += int(max(range(len(sc)), key=lambda i: sc[i][0]) == label)
        cn += int(max(range(len(sc)), key=lambda i: sc[i][0] / max(sc[i][1], 1)) == label)
    n = len(items)
    return {"acc": c / n, "acc_norm": cn / n, "n": n, "chance": sum(1 / len(ch) for _, ch, _ in items) / n}


@torch.no_grad()
def lambada(model, tok, device, max_len):
    rows = [json.loads(l) for l in open(os.path.join(AUX, "eval_lambada.jsonl"))][:BENCH_LIMIT]
    hit = 0
    nll = ntok = 0.0
    for row in rows:
        ctx, last = row["text"].rsplit(" ", 1)
        t = tok.encode(" " + last)
        c = ([tok.bos] + tok.encode(ctx))[-(max_len - len(t)):]
        x = torch.tensor([c + t], device=device)
        with torch.autocast("cuda", dtype=torch.float16):
            logits, _ = model(x)
        lp = F.log_softmax(logits[0, len(c) - 1:-1].float(), -1)
        tgt = torch.tensor(t, device=device)
        nll -= lp.gather(-1, tgt[:, None]).sum().item()
        ntok += len(t)
        hit += int((lp.argmax(-1) == tgt).all().item())
    return {"acc": hit / len(rows), "ppl": math.exp(nll / ntok), "n": len(rows)}


# ---------------------------------------------------------------- efficiency
def param_breakdown(model):
    seen, out = set(), {"embedding": 0, "attention": 0, "mlp": 0, "norm": 0, "other": 0}
    for n, p in model.named_parameters():
        if p.data_ptr() in seen:
            continue
        seen.add(p.data_ptr())
        if any(k in n for k in ("emb", "head", "wte", "wpe", "embed")):
            k = "embedding"
        elif "attn" in n or "attention" in n:
            k = "attention"
        elif "ff" in n or "mlp" in n or "fc1" in n or "fc2" in n:
            k = "mlp"
        elif "ln" in n or "norm" in n or ".n1" in n or ".n2" in n:
            k = "norm"
        else:
            k = "other"
        out[k] += p.numel()
    out["total"] = sum(out.values())
    return out


@torch.no_grad()
def throughput(model, device, vocab, seq=512):
    out = {}
    for bs in (1, 8):
        x = torch.randint(10, vocab - 10, (bs, seq), device=device)
        for _ in range(2):
            with torch.autocast("cuda", dtype=torch.float16):
                model(x)
        torch.cuda.synchronize()
        t = time.time()
        n = 5
        for _ in range(n):
            with torch.autocast("cuda", dtype=torch.float16):
                model(x)
        torch.cuda.synchronize()
        out[f"prefill_tok_s_bs{bs}"] = bs * seq * n / (time.time() - t)
    return out


@torch.no_grad()
def generate(model, tok, ids, max_new, device, temperature=0.0, top_p=0.9, stop=(), seed=0):
    g = torch.Generator(device=device).manual_seed(seed)
    ids = list(ids)[-(model.max_len - max_new):]
    out = []
    t0 = time.time()
    for _ in range(max_new):
        x = torch.tensor([ids[-model.max_len:]], device=device)
        with torch.autocast("cuda", dtype=torch.float16):
            logits, _ = model(x)
        lg = logits[0, -1].float()
        if temperature == 0:
            nt = int(lg.argmax())
        else:
            p = F.softmax(lg / temperature, -1)
            sp, si = p.sort(descending=True)
            keep = sp.cumsum(0) - sp < top_p
            sp = sp * keep
            nt = int(si[torch.multinomial(sp / sp.sum(), 1, generator=g)])
        if nt in stop:
            break
        ids.append(nt)
        out.append(nt)
    torch.cuda.synchronize()
    dt = time.time() - t0
    return out, dt / max(len(out), 1)


PROMPTS = [
    "The capital of France is",
    "Once upon a time, in a small village by the sea,",
    "def fibonacci(n):",
    "Water boils at a temperature of",
]
CHAT_PROMPTS = ["Explain why the sky is blue in two sentences.", "What is 12 * 7?"]


# ---------------------------------------------------------------- diagnostics
@torch.no_grad()
def weight_rms(model):
    res = []
    for b in model.blocks:
        attn = [p for n, p in b.named_parameters() if "attn" in n and p.dim() == 2]
        mlp = [p for n, p in b.named_parameters() if ("ff" in n) and p.dim() == 2]
        rms = lambda ps: math.sqrt(sum((p.float() ** 2).sum().item() for p in ps) / sum(p.numel() for p in ps))
        res.append({"attn_rms": rms(attn), "mlp_rms": rms(mlp)})
    return res


@torch.no_grad()
def layer_ablation(model, tok, docs, device):
    base = bpb(model, tok, docs, 512, device)["bpb"]
    out = []
    for i in range(len(model.blocks)):
        set_ablate(model, i, True)
        out.append(bpb(model, tok, docs, 512, device)["bpb"] - base)
        set_ablate(model, i, False)
    return {"base_bpb": base, "delta_bpb": out}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    a = ap.parse_args()
    torch.cuda.set_device(a.gpu)
    device = f"cuda:{a.gpu}"
    torch.manual_seed(0)
    texts = load_texts()
    model, tok, info = load(a.model, device)
    R = {"model": a.model, "info": info}
    out_path = os.path.join(HERE, "results", f"{a.model}.json")

    def save():
        json.dump(R, open(out_path, "w"), indent=1, default=str)

    def log(*s):
        print(time.strftime("%H:%M:%S"), a.model, *s, flush=True)

    R["params"] = param_breakdown(model)
    if a.model in SPECS:
        R["ckpt_mb"] = os.path.getsize(SPECS[a.model]["ckpt"]) / 2**20
        R["weights"] = weight_rms(model)
    else:
        R["ckpt_mb"] = R["params"]["total"] * 4 / 2**20
    torch.cuda.reset_peak_memory_stats(device)
    R["speed"] = throughput(model, device, tok.vocab_size)
    R["peak_vram_gb"] = torch.cuda.max_memory_allocated(device) / 2**30
    log("params", R["params"], R["speed"])
    save()

    R["bpb"] = {}
    for d, docs in texts.items():
        R["bpb"][d] = {"512": bpb(model, tok, docs, 512, device)}
        if info["max_len"] >= 2048:
            R["bpb"][d]["2048"] = bpb(model, tok, docs, 2048, device, bs=2)
        log("bpb", d, {k: round(v["bpb"], 4) for k, v in R["bpb"][d].items()})
        save()

    if a.model in SPECS:
        R["ablation"] = layer_ablation(model, tok, texts["web"][:40] + texts["wiki"][:20], device)
        log("ablation", [round(x, 3) for x in R["ablation"]["delta_bpb"]])
        save()

    R["bench"] = {}
    for name in ("hellaswag", "arc_easy", "piqa"):
        R["bench"][name] = multiple_choice(model, tok, name, device, info["max_len"])
        log(name, R["bench"][name])
        save()
    R["bench"]["lambada"] = lambada(model, tok, device, info["max_len"])
    log("lambada", R["bench"]["lambada"])
    save()

    stop = {tok.eos} | ({tok.end} if hasattr(tok, "end") else set())
    R["samples"] = []
    lat = []
    for p in PROMPTS:
        ids = [tok.bos] + tok.encode(p)
        g, ms = generate(model, tok, ids, 64, device, 0.0, stop=stop)
        s, _ = generate(model, tok, ids, 64, device, 0.8, stop=stop)
        lat.append(ms)
        R["samples"].append({"mode": "raw", "prompt": p, "greedy": tok.decode(g), "sample": tok.decode(s)})
    if a.model in ("B", "C"):
        for p in CHAT_PROMPTS:
            ids = chat_prompt(tok, p)
            g, ms = generate(model, tok, ids, 96, device, 0.0, stop=stop)
            R["samples"].append({"mode": "chat", "prompt": p, "greedy": tok.decode(g, skip_special=True)})
    R["speed"]["decode_ms_per_token_nocache"] = 1000 * float(np.mean(lat))
    log("samples done")
    save()


if __name__ == "__main__":
    main()
