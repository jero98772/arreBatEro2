"""
Inference for hybrid.pt (custom model + raw `tokenizers` BPE file).

Modes
  --mode raw : plain text continuation  ->  [BOS] your text ...
  --mode qa  : question/answer format used by distill_rl_server.py  ->  [BOS]User: <q>\nAnswer:
               (use this for the distilled / RL-improved checkpoint)

Examples
  # one prompt
  python infer_hybrid.py --ckpt models/hybrid.pt --tokenizer models/bpe_tokenizer.json \
      --model_class model_file:YourModelClass --model_kwargs '{"vocab_size":32000}' \
      --mode raw --prompt "to be, or not to be"

  # interactive chat with the improved model
  python infer_hybrid.py --ckpt hybrid_improved.pt ... --mode qa

  # compare original vs improved on the same prompt (same random seed)
  python infer_hybrid.py --ckpt models/hybrid.pt hybrid_improved.pt ... --mode qa --prompt "What is gravity?"
"""
import argparse, importlib, json, os, sys, time
import torch
import torch.nn as nn
from tokenizers import Tokenizer


# ----------------------------------------------------------------------------- loading
def _clean_keys(sd):
    out = {}
    for k, v in sd.items():
        for pre in ("module.", "_orig_mod.", "net."):
            if k.startswith(pre):
                k = k[len(pre) :]
        out[k] = v
    return out


def load_model(path, args, device, dtype):
    sys.path.insert(0, os.getcwd())
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, nn.Module):  # whole pickled model
        net = obj
    else:
        assert (
            args.model_class
        ), "this checkpoint is a state_dict: pass --model_class module:Class"
        mod, cls = args.model_class.split(":")
        net = getattr(importlib.import_module(mod), cls)(
            **json.loads(args.model_kwargs)
        )
        sd = obj
        if isinstance(obj, dict):
            for k in ("state_dict", "model", "model_state_dict", "model_state"):
                if k in obj and isinstance(obj[k], dict):
                    sd = obj[k]
                    break
        res = net.load_state_dict(sd, strict=False)
        if res.missing_keys or res.unexpected_keys:  # retry without wrapper prefixes
            res = net.load_state_dict(_clean_keys(sd), strict=False)
        print(
            f"[load] {path}: missing={len(res.missing_keys)} unexpected={len(res.unexpected_keys)}"
        )
    return net.to(device=device, dtype=dtype).eval()


def to_logits(o):
    if hasattr(o, "logits"):
        return o.logits
    return o[0] if isinstance(o, (tuple, list)) else o


# ----------------------------------------------------------------------------- sampling
def sample_next(logits, history, temperature, top_k, top_p, rep_penalty):
    logits = logits.float().clone()
    if rep_penalty != 1.0 and history:
        idx = torch.tensor(sorted(set(history[-256:])), device=logits.device)
        l = logits[idx]
        logits[idx] = torch.where(l > 0, l / rep_penalty, l * rep_penalty)
    if temperature <= 0:
        return int(logits.argmax())
    logits = logits / temperature
    if top_k > 0:
        kth = torch.topk(logits, min(top_k, logits.numel())).values[-1]
        logits[logits < kth] = float("-inf")
    probs = torch.softmax(logits, -1)
    if top_p < 1.0:
        sp, si = probs.sort(descending=True)
        sp[(sp.cumsum(0) - sp) > top_p] = 0
        probs = torch.zeros_like(probs).scatter_(0, si, sp)
    return int(torch.multinomial(probs / probs.sum(), 1))


@torch.no_grad()
def generate(net, tok, prompt_ids, args, device, stream=True):
    eos_id = tok.token_to_id("[EOS]")
    ids = list(prompt_ids)
    new, printed = [], ""
    t0 = time.time()
    for _ in range(args.max_new):
        ctx = torch.tensor([ids[-args.max_ctx :]], device=device)
        logits = to_logits(net(ctx))[0, -1]
        nxt = sample_next(
            logits, ids, args.temperature, args.top_k, args.top_p, args.rep_penalty
        )
        if nxt == eos_id:
            break
        ids.append(nxt)
        new.append(nxt)
        if stream:
            text = tok.decode(new, skip_special_tokens=True)
            if not text.endswith("\ufffd"):  # wait for complete multi-byte characters
                print(text[len(printed) :], end="", flush=True)
                printed = text
    text = tok.decode(new, skip_special_tokens=True)
    if stream:
        print(text[len(printed) :], end="", flush=True)
        print()
    dt = time.time() - t0
    return text, len(new), dt


# ----------------------------------------------------------------------------- prompt building
def build_prompt_ids(tok, text, mode):
    bos = tok.token_to_id("[BOS]")
    if mode == "qa":
        text = f"User: {text}\nAnswer:"
    ids = tok.encode(
        text, add_special_tokens=False
    ).ids  # no automatic [EOS] at the end
    return [bos] + ids[-(args_global.max_ctx - 1 - args_global.max_new) :]


# ----------------------------------------------------------------------------- main
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--ckpt",
        nargs="+",
        default=["hybrid.pt"],
        help="one or more checkpoints (several = compare)",
    )
    p.add_argument("--tokenizer", required=True, help="e.g. models/bpe_tokenizer.json")
    p.add_argument(
        "--model_class",
        default=None,
        help="module:Class, needed if ckpt is a state_dict",
    )
    p.add_argument(
        "--model_kwargs", default="{}", help="JSON kwargs for the model class"
    )
    p.add_argument("--mode", choices=["raw", "qa"], default="raw")
    p.add_argument(
        "--prompt", default=None, help="single prompt; omit for interactive mode"
    )
    p.add_argument("--max_ctx", type=int, default=512)
    p.add_argument("--max_new", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.8, help="0 = greedy")
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--rep_penalty", type=float, default=1.1, help="1.0 = off")
    p.add_argument(
        "--dtype", default="float32", choices=["float32", "bfloat16", "float16"]
    )
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def run_prompt(models, tok, text, args, device):
    ids = build_prompt_ids(tok, text, args.mode)
    for name, net in models:
        torch.manual_seed(args.seed)  # same seed so checkpoints are comparable
        if len(models) > 1:
            print(f"\n--- {name} ---")
        out, n, dt = generate(net, tok, ids, args, device, stream=True)
        print(f"[{n} tokens, {n / max(dt, 1e-9):.1f} tok/s]")


def main():
    global args_global
    args = args_global = get_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[args.dtype]
    if device == "cpu":
        dtype = torch.float32

    tok = Tokenizer.from_file(args.tokenizer)
    models = [
        (os.path.basename(p), load_model(p, args, device, dtype)) for p in args.ckpt
    ]

    if args.prompt is not None:
        run_prompt(models, tok, args.prompt, args, device)
        return

    print(
        f"Interactive mode ({args.mode}). Empty line or Ctrl-C to quit. "
        f"Settings: T={args.temperature} top_k={args.top_k} top_p={args.top_p}"
    )
    try:
        while True:
            text = input("\n> ").strip()
            if not text:
                break
            run_prompt(models, tok, text, args, device)
    except (KeyboardInterrupt, EOFError):
        print()


if __name__ == "__main__":
    main()
