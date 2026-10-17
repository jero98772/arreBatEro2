"""
Improve a student LLM (hybrid.pt) using a teacher served by llama.cpp's `llama-server`
(Gemma 4 E2B). The teacher is only reached over HTTP, so it never has to fit in the
training process, and student/teacher may use different tokenizers.

  Phase 1  Sequence-level distillation : SFT on the teacher's answers.
  Phase 2  Reinforcement learning      : GRPO-style policy gradient. The student samples
           several answers per prompt and each one gets a reward:
              (a) ROUGE-L RECALL against the teacher's reference answer
              (b) TEACHER JUDGE: the teacher rates the answer 0-9 (expected value over
                  the digit log-probs, so the signal is smooth), scaled to 0-1
           Advantages are group-normalised; a KL term keeps the student near its start.

Progress is measured on held-out prompts at step 0 (baseline), every --eval_every steps
and at the end -> run_out/metrics.csv, run_out/metrics.png, best checkpoint saved.

Start the teacher first (see README in chat), then e.g.:
  python distill_rl_server.py --student models/hybrid.pt \
     --student_tokenizer models/bpe_tokenizer.json \
     --student_class model_file:YourModelClass --student_kwargs '{"vocab_size":32000}'

pip install torch transformers tokenizers datasets matplotlib
"""
import argparse, concurrent.futures as cf, copy, csv, importlib, json, math, os, random, re, sys, time
import urllib.request
from types import SimpleNamespace
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizerFast,
)

P_MAX, A_MAX = (
    256,
    200,
)  # prompt / answer tokens (student side); must fit the student's context
CONCISE = "\n\nAnswer concisely, in at most 80 words."
# Gemma 4 otherwise writes its reasoning into `content`, truncating real answers.
NO_THINK = {"enable_thinking": False}
JUDGE_PROMPT = (
    "You are grading an answer.\n\nQuestion:\n{q}\n\nReference answer:\n{r}\n\nCandidate answer:\n{c}\n\n"
    "How well does the candidate match the reference in meaning and helpfulness? "
    "Reply with ONE digit from 0 (unrelated or garbage) to 9 (equivalent). Digit only."
)


# ----------------------------------------------------------------------------- args
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--student", default="hybrid.pt")
    p.add_argument(
        "--student_hf",
        default=None,
        help="alternative: HF id/dir for a HF-style student",
    )
    p.add_argument(
        "--student_config",
        default=None,
        help="HF config id/dir if hybrid.pt is a state_dict of a HF model",
    )
    p.add_argument(
        "--student_class",
        default=None,
        help="custom model as module:Class (e.g. model:QPT)",
    )
    p.add_argument(
        "--student_kwargs", default="{}", help="JSON kwargs for the custom class"
    )
    p.add_argument(
        "--student_tokenizer",
        required=True,
        help="e.g. models/bpe_tokenizer.json or HF dir",
    )
    p.add_argument("--max_ctx", type=int, default=512, help="student context length")
    p.add_argument("--server", default="http://localhost:8004", help="llama-server URL")
    p.add_argument(
        "--teacher",
        default=None,
        help="informational only; the teacher is the llama-server at --server",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=4,
        help="parallel requests (match llama-server -np)",
    )
    p.add_argument("--prompts", default=None, help="jsonl with {'prompt': ...}")
    p.add_argument("--n_train", type=int, default=5000)
    p.add_argument("--n_eval", type=int, default=40)
    p.add_argument("--max_new", type=int, default=128)
    p.add_argument(
        "--dtype", default="float32", choices=["float32", "bfloat16", "float16"]
    )
    p.add_argument("--sft_steps", type=int, default=2000)
    p.add_argument("--sft_bs", type=int, default=16)
    p.add_argument("--sft_lr", type=float, default=3e-4)
    p.add_argument("--sft_warmup", type=int, default=100)
    p.add_argument("--rl_steps", type=int, default=0, help="RL hurt the last run; enable explicitly")
    p.add_argument("--rl_prompts", type=int, default=8)
    p.add_argument("--group", type=int, default=8)
    p.add_argument("--micro", type=int, default=4)
    p.add_argument("--rl_lr", type=float, default=5e-6)
    p.add_argument("--beta", type=float, default=0.02)
    p.add_argument(
        "--w_rouge", "--w_recall", dest="w_rouge", type=float, default=0.2,
        help="weight of ROUGE-L F1 (recall rewarded length)",
    )
    p.add_argument("--w_teacher", type=float, default=0.8)
    p.add_argument("--temp", type=float, default=1.0)
    p.add_argument("--eval_every", type=int, default=100)
    p.add_argument("--out", default="hybrid_improved.pt")
    p.add_argument("--workdir", default="run_out")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


# ----------------------------------------------------------------------------- teacher (llama-server)
class Teacher:
    def __init__(self, url, workers):
        self.url = url.rstrip("/")
        self.pool = cf.ThreadPoolExecutor(max_workers=workers)

    def _req(self, path, payload=None, retries=3):
        err = None
        for k in range(retries):
            try:
                data = None if payload is None else json.dumps(payload).encode()
                req = urllib.request.Request(
                    self.url + path,
                    data=data,
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(req, timeout=180) as r:
                    return json.loads(r.read())
            except Exception as e:  # noqa
                err = e
                print(f"[teacher] {path} attempt {k + 1}/{retries} failed: {e}", flush=True)
                time.sleep(1 + 2 * k)
        raise RuntimeError(f"llama-server request {path} failed: {err}")

    def wait(self):
        try:
            self._req("/health", retries=2)
        except RuntimeError:
            sys.exit(
                f"Cannot reach llama-server at {self.url}. Start it first (see instructions)."
            )
        print(f"[teacher] llama-server OK at {self.url}")

    def answer(self, prompt, max_tokens):
        out = self._req(
            "/v1/chat/completions",
            {
                "messages": [{"role": "user", "content": prompt + CONCISE}],
                "max_tokens": max_tokens,
                "temperature": 0.0,
                "chat_template_kwargs": NO_THINK,
            },
        )
        ch = out["choices"][0]
        text = (ch["message"].get("content") or "").strip()
        # Truncated answers or leaked reasoning make bad references; "" = dropped.
        if ch.get("finish_reason") == "length" or text.startswith("Thinking Process"):
            return ""
        return text

    def judge(self, q, ref, cand):
        """Teacher grade in [0,1]: expected digit under the teacher's own log-probs."""
        prompt = JUDGE_PROMPT.format(
            q=q[:500], r=ref[:1200], c=(cand.strip() or "(empty)")[:1200]
        )
        out = self._req(
            "/v1/chat/completions",
            {
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 1,
                "temperature": 0.0,
                "logprobs": True,
                "top_logprobs": 20,
                "chat_template_kwargs": NO_THINK,
            },
        )
        try:
            tops = out["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
            probs = {}
            for t in tops:
                tk = t["token"].strip()
                if len(tk) == 1 and tk.isdigit():
                    probs[int(tk)] = probs.get(int(tk), 0.0) + math.exp(t["logprob"])
            z = sum(probs.values())
            if z > 0:
                return sum(k * v for k, v in probs.items()) / z / 9.0
        except (KeyError, IndexError, TypeError):
            pass
        m = re.search(
            r"\d", out["choices"][0]["message"].get("content") or ""
        )  # fallback: sampled digit
        return int(m.group()) / 9.0 if m else 0.0

    def answers(self, prompts, max_tokens):
        return list(self.pool.map(lambda p: self.answer(p, max_tokens), prompts))

    def judge_many(self, qs, refs, cands):
        return list(self.pool.map(lambda a: self.judge(*a), zip(qs, refs, cands)))


def teacher_answers(teacher, prompts, cache_path, max_tokens):
    cache = {}
    if os.path.exists(cache_path):
        for l in open(cache_path, encoding="utf-8"):
            d = json.loads(l)
            cache[d["prompt"]] = d["answer"]
    todo = [p for p in prompts if p not in cache]
    print(f"[teacher] generating {len(todo)} reference answers (cached: {len(cache)})", flush=True)
    t0, done, empty = time.time(), 0, 0
    with open(cache_path, "a", encoding="utf-8") as f:
        futs = {teacher.pool.submit(teacher.answer, p, max_tokens): p for p in todo}
        for fut in cf.as_completed(futs):
            p, a = futs[fut], fut.result()
            done += 1
            if a:
                cache[p] = a
                f.write(json.dumps({"prompt": p, "answer": a}, ensure_ascii=False) + "\n")
                f.flush()
            else:
                empty += 1
            if done % 10 == 0 or done == len(todo):
                rate = done / max(time.time() - t0, 1e-9)
                print(
                    f"[teacher] {done}/{len(todo)} refs  {rate:.1f}/s  "
                    f"ETA {(len(todo) - done) / rate:.0f}s  empty={empty}",
                    flush=True,
                )
    return cache


# ----------------------------------------------------------------------------- student model plumbing
def fmt(tok, prompt):
    """(text, add_special_tokens). Manual [BOS], no auto [EOS] after the prompt."""
    bos = tok.bos_token or ""
    return f"{bos}User: {prompt}\nAnswer:", False


class GenWrapper(nn.Module):
    """Gives a plain `net(input_ids) -> logits` model an HF-like forward()/generate()."""

    def __init__(self, net, max_ctx=512, pad_id=0):
        super().__init__()
        self.net, self.max_ctx, self.pad_id = net, max_ctx, pad_id
        self.eos_ids = torch.tensor([3])
        self.was_full = False

    @property
    def device(self):
        return next(self.net.parameters()).device

    @staticmethod
    def _to_logits(o):
        if hasattr(o, "logits"):
            return o.logits
        return o[0] if isinstance(o, (tuple, list)) else o

    def forward(self, input_ids, attention_mask=None, position_ids=None):
        if attention_mask is None:
            return SimpleNamespace(logits=self._to_logits(self.net(input_ids)))
        B, L = input_ids.shape
        logits = None
        for b in range(
            B
        ):  # run each row un-padded so pad tokens never leak into attention
            idx = attention_mask[b].nonzero().squeeze(-1)
            o = self._to_logits(self.net(input_ids[b, idx].unsqueeze(0)))
            if logits is None:
                logits = torch.zeros(B, L, o.size(-1), device=o.device, dtype=o.dtype)
            logits[b, idx] = o[0]
        return SimpleNamespace(logits=logits)

    @torch.no_grad()
    def generate(
        self,
        input_ids,
        attention_mask=None,
        max_new_tokens=128,
        do_sample=False,
        pad_token_id=None,
        temperature=1.0,
        top_p=1.0,
        top_k=0,
        **_,
    ):
        B, P = input_ids.shape
        pad = self.pad_id if pad_token_id is None else pad_token_id
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        out = torch.full(
            (B, max_new_tokens), pad, dtype=torch.long, device=input_ids.device
        )
        eos = self.eos_ids.to(input_ids.device)
        for b in range(B):
            s = input_ids[b][attention_mask[b].bool()].unsqueeze(0)
            for t in range(max_new_tokens):
                lg = self._to_logits(self.net(s[:, -self.max_ctx :]))[0, -1].float()
                if do_sample:
                    probs = torch.softmax(lg / max(temperature, 1e-5), -1)
                    sp, si = probs.sort(descending=True)
                    sp[(sp.cumsum(0) - sp) > top_p] = 0
                    nxt = si[torch.multinomial(sp / sp.sum(), 1)]
                else:
                    nxt = lg.argmax().view(1)
                out[b, t] = nxt
                s = torch.cat([s, nxt.view(1, 1)], 1)
                if torch.isin(nxt, eos).any():
                    break
        return torch.cat([input_ids, out], 1)


def load_tok(path):
    if str(path).endswith(
        ".json"
    ):  # raw `tokenizers` file, e.g. models/bpe_tokenizer.json
        return PreTrainedTokenizerFast(
            tokenizer_file=str(path),
            bos_token="[BOS]",
            eos_token="[EOS]",
            pad_token="[PAD]",
            unk_token="[UNK]",
        )
    return AutoTokenizer.from_pretrained(path, trust_remote_code=True)


def _extract_sd(obj):
    if isinstance(obj, dict):
        for k in ("state_dict", "model", "model_state_dict", "model_state"):
            if k in obj and isinstance(obj[k], dict):
                return obj[k]
    return obj


def load_student(args, dtype):
    if args.student_class:
        sys.path.insert(0, os.getcwd())
        obj = torch.load(args.student, map_location="cpu", weights_only=False)
        if isinstance(obj, nn.Module):
            net = obj
        else:
            mod, cls = args.student_class.split(":")
            net = getattr(importlib.import_module(mod), cls)(
                **json.loads(args.student_kwargs)
            )
            print("[load]", net.load_state_dict(_extract_sd(obj), strict=False))
        w = GenWrapper(net, args.max_ctx)
        w.was_full = isinstance(obj, nn.Module)
        return w.to(dtype), False
    full = False
    if args.student_hf:
        model = AutoModelForCausalLM.from_pretrained(
            args.student_hf, torch_dtype=dtype, trust_remote_code=True
        )
    else:
        obj = torch.load(args.student, map_location="cpu", weights_only=False)
        if isinstance(obj, nn.Module):
            model, full = obj, True
        else:
            assert (
                args.student_config
            ), "hybrid.pt is a state_dict: pass --student_class or --student_config"
            cfg = AutoConfig.from_pretrained(
                args.student_config, trust_remote_code=True
            )
            model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True)
            print("[load]", model.load_state_dict(_extract_sd(obj), strict=False))
    return model.to(dtype), full


def ckpt_obj(model, is_full):
    if isinstance(model, GenWrapper):
        return model.net if model.was_full else model.net.state_dict()
    return model if is_full else model.state_dict()


def fwd(model, ids, att, pos=None):
    kw = dict(input_ids=ids, attention_mask=att)
    if pos is not None:
        kw["position_ids"] = pos
    try:
        out = model(**kw)
    except TypeError:
        out = model(input_ids=ids, attention_mask=att)
    return out.logits if hasattr(out, "logits") else out


def eos_ids_of(model, tok):
    ids = {tok.eos_token_id}
    ge = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    if isinstance(ge, int):
        ids.add(ge)
    elif isinstance(ge, (list, tuple)):
        ids.update(ge)
    return torch.tensor(sorted(i for i in ids if i is not None))


def pad_right(seqs, pad_id, device, fill=None):
    L = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), L), pad_id if fill is None else fill, dtype=torch.long)
    att = torch.zeros((len(seqs), L), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, : len(s)] = torch.tensor(s)
        att[i, : len(s)] = 1
    return ids.to(device), att.to(device)


# ----------------------------------------------------------------------------- metrics
def lcs_len(a, b):
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(prev[j + 1], cur[j]))
        prev = cur
    return prev[-1]


def rouge_l(pred, ref):
    p, r = pred.lower().split()[:150], ref.lower().split()[:150]
    if not p or not r:
        return 0.0, 0.0, 0.0
    l = lcs_len(p, r)
    rec, prec = l / len(r), l / len(p)
    f1 = 0.0 if l == 0 else 2 * rec * prec / (rec + prec)
    return rec, prec, f1


# ----------------------------------------------------------------------------- student generation / scoring
@torch.no_grad()
def generate(model, tok, prompts, max_new, sample, temp=1.0):
    texts, add = zip(*[fmt(tok, p) for p in prompts])
    enc = tok(
        list(texts),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=P_MAX,
        add_special_tokens=add[0],
    ).to(model.device)
    kw = dict(max_new_tokens=max_new, do_sample=sample, pad_token_id=tok.pad_token_id)
    if sample:
        kw.update(temperature=temp, top_p=0.95, top_k=0)
    out = model.generate(**enc, **kw)
    return enc, out[:, enc["input_ids"].shape[1] :]


def completion_mask(comp, eos_ids):
    is_eos = torch.isin(comp, eos_ids.to(comp.device)).long()
    return ((is_eos.cumsum(1) - is_eos) == 0).long()  # up to and including first EOS


@torch.no_grad()
def mean_logprob(model, tok, prompts, answers, bs=4):
    """Mean per-token log p(answer | prompt) under the student."""
    res, dev = [], model.device
    for i in range(0, len(prompts), bs):
        seqs, spans = [], []
        for p, a in zip(prompts[i : i + bs], answers[i : i + bs]):
            t, add = fmt(tok, p)
            pi = tok(t, add_special_tokens=add).input_ids[-P_MAX:]
            ai = tok(a, add_special_tokens=False).input_ids[:A_MAX] or [
                tok.eos_token_id
            ]
            seqs.append(pi + ai)
            spans.append((len(pi), len(pi) + len(ai)))
        ids, att = pad_right(seqs, tok.pad_token_id, dev)
        logits = fwd(model, ids, att)
        for k, (s, e) in enumerate(spans):
            lp = torch.log_softmax(logits[k, s - 1 : e - 1].float(), -1)
            res.append(
                lp.gather(-1, ids[k, s:e].unsqueeze(-1)).squeeze(-1).mean().item()
            )
    return res


def token_logprobs(model, prompt_ids, prompt_mask, comp, comp_mask):
    ids = torch.cat([prompt_ids, comp], 1)
    att = torch.cat([prompt_mask, comp_mask], 1)
    pos = (att.cumsum(1) - 1).clamp(min=0)
    logits = fwd(model, ids, att, pos)
    P = prompt_ids.shape[1]
    lp = torch.log_softmax(logits[:, P - 1 : -1].float(), -1)
    return lp.gather(-1, comp.unsqueeze(-1)).squeeze(-1)


# ----------------------------------------------------------------------------- data / eval
def load_prompts(args):
    if args.prompts:
        ps = [
            json.loads(l)["prompt"]
            for l in open(args.prompts, encoding="utf-8")
            if l.strip()
        ]
    else:
        from datasets import load_dataset

        ds = load_dataset("tatsu-lab/alpaca", split="train")
        ps = [
            (r["instruction"] + ("\n" + r["input"] if r["input"] else ""))
            for r in ds
            if len(r["instruction"]) < 200 and not r["input"]
        ]
    random.shuffle(ps)
    ps = ps[: args.n_train + args.n_eval]
    return ps[args.n_eval :], ps[: args.n_eval]


@torch.no_grad()
def evaluate(model, stok, teacher, eval_prompts, refs, args):
    model.eval()
    outs = []
    for i in range(0, len(eval_prompts), 8):
        _, comp = generate(
            model, stok, eval_prompts[i : i + 8], args.max_new, sample=False
        )
        outs += [o.strip() for o in stok.batch_decode(comp, skip_special_tokens=True)]
    ref_l = [refs[p] for p in eval_prompts]
    rl = [rouge_l(o, r) for o, r in zip(outs, ref_l)]
    judge = teacher.judge_many(eval_prompts, ref_l, outs)
    slp = mean_logprob(model, stok, eval_prompts, ref_l)
    rec = sum(x[0] for x in rl) / len(rl)
    jd = sum(judge) / len(judge)
    f1 = sum(x[2] for x in rl) / len(rl)
    return dict(
        rougeL_recall=rec,
        rougeL_f1=f1,
        teacher_judge=jd,  # 0-1, teacher's rating of student answers
        student_ppl_on_teacher=math.exp(
            -sum(slp) / len(slp)
        ),  # lower = imitates teacher better
        eval_reward=args.w_rouge * f1 + args.w_teacher * jd,
        avg_words=sum(len(o.split()) for o in outs) / len(outs),
    )


# ----------------------------------------------------------------------------- main
def main():
    args = get_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.workdir, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[args.dtype]
    if dev == "cpu":
        dtype = torch.float32

    if args.teacher:
        print(f"[teacher] --teacher {args.teacher} noted; using llama-server at {args.server}")
    teacher = Teacher(args.server, args.workers)
    teacher.wait()

    stok = load_tok(args.student_tokenizer)
    if stok.pad_token is None:
        stok.pad_token = stok.eos_token
    stok.padding_side = "left"
    model, is_full = load_student(args, dtype)
    model.to(dev)
    s_eos = eos_ids_of(model, stok)
    if isinstance(model, GenWrapper):
        model.eos_ids, model.pad_id = s_eos, stok.pad_token_id

    train, evalp = load_prompts(args)
    refs = teacher_answers(
        teacher,
        train + evalp,
        os.path.join(args.workdir, "teacher_refs_nothink.jsonl"),
        args.max_new,
    )
    train = [p for p in train if refs.get(p)]
    evalp = [p for p in evalp if refs.get(p)]

    rows, best = [], -1e9
    csv_path = os.path.join(args.workdir, "metrics.csv")

    def log_eval(step, phase):
        nonlocal best
        m = evaluate(model, stok, teacher, evalp, refs, args)
        m.update(step=step, phase=phase)
        rows.append(m)
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        base = rows[0]
        print(f"\n=== EVAL step {step} [{phase}] ===")
        for k in (
            "rougeL_recall",
            "rougeL_f1",
            "teacher_judge",
            "student_ppl_on_teacher",
            "eval_reward",
            "avg_words",
        ):
            print(f"  {k:24s} {m[k]:8.4f}   (vs baseline {m[k] - base[k]:+.4f})")
        if m["eval_reward"] > best and step > 0:
            best = m["eval_reward"]
            torch.save(ckpt_obj(model, is_full), args.out)
            print(f"  -> new best, saved {args.out}")

    log_eval(0, "baseline")
    gstep = 0

    # ---------------- Phase 1: sequence-level distillation
    opt = torch.optim.AdamW(model.parameters(), lr=args.sft_lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(  # linear warmup, cosine decay to 10%
        opt,
        lambda s: min(1.0, (s + 1) / max(args.sft_warmup, 1))
        * (0.1 + 0.45 * (1 + math.cos(math.pi * s / max(args.sft_steps, 1)))),
    )
    for s in range(args.sft_steps):
        model.train()
        seqs, labs = [], []
        for p in random.sample(train, min(args.sft_bs, len(train))):
            t, add = fmt(stok, p)
            pi = stok(t, add_special_tokens=add).input_ids[-P_MAX:]
            ai = stok(refs[p], add_special_tokens=False).input_ids[:A_MAX] + [
                stok.eos_token_id
            ]
            seqs.append(pi + ai)
            labs.append([-100] * len(pi) + ai)
        ids, att = pad_right(seqs, stok.pad_token_id, dev)
        lab, _ = pad_right(labs, -100, dev, fill=-100)
        logits = fwd(model, ids, att)
        loss = F.cross_entropy(
            logits[:, :-1].float().reshape(-1, logits.size(-1)),
            lab[:, 1:].reshape(-1),
            ignore_index=-100,
        )
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        gstep += 1
        if (s + 1) % 10 == 0:
            print(f"[SFT {s+1}/{args.sft_steps}] loss={loss.item():.4f} lr={sched.get_last_lr()[0]:.2e}")
        if (s + 1) % args.eval_every == 0:
            log_eval(gstep, "sft")
    del opt, sched

    # ---------------- Phase 2: RL (GRPO-style)
    ref_model = None
    if args.beta > 0 and args.rl_steps > 0:
        ref_model = copy.deepcopy(model).eval()
        for p in ref_model.parameters():
            p.requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=args.rl_lr, weight_decay=0.0)
    for s in range(args.rl_steps):
        model.eval()
        bp = random.sample(train, min(args.rl_prompts, len(train)))
        rep = [p for p in bp for _ in range(args.group)]
        enc, comp = generate(
            model, stok, rep, args.max_new, sample=True, temp=args.temp
        )
        cmask = completion_mask(comp, s_eos)
        texts = [t.strip() for t in stok.batch_decode(comp, skip_special_tokens=True)]

        f1 = torch.tensor([rouge_l(t, refs[p])[2] for t, p in zip(texts, rep)])
        jd = torch.tensor(teacher.judge_many(rep, [refs[p] for p in rep], texts))
        reward = args.w_rouge * f1 + args.w_teacher * jd
        R = reward.view(-1, args.group)
        adv = (
            ((R - R.mean(1, keepdim=True)) / (R.std(1, keepdim=True) + 1e-4))
            .view(-1)
            .to(dev)
        )

        ntok = cmask.sum().clamp(min=1)
        tot_kl = 0.0
        for i in range(0, len(rep), args.micro):
            sl = slice(i, i + args.micro)
            lp = token_logprobs(
                model,
                enc["input_ids"][sl],
                enc["attention_mask"][sl],
                comp[sl],
                cmask[sl],
            )
            m = cmask[sl].float()
            loss = -(adv[sl].unsqueeze(1) * lp * m).sum() / ntok
            if ref_model is not None:
                with torch.no_grad():
                    rlp = token_logprobs(
                        ref_model,
                        enc["input_ids"][sl],
                        enc["attention_mask"][sl],
                        comp[sl],
                        cmask[sl],
                    )
                d = rlp - lp
                kl = ((torch.exp(d) - d - 1) * m).sum() / ntok
                loss = loss + args.beta * kl
                tot_kl += kl.item()
            loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        gstep += 1
        print(
            f"[RL {s+1}/{args.rl_steps}] reward={reward.mean():.3f} rougeL_f1={f1.mean():.3f} "
            f"judge={jd.mean():.3f} kl={tot_kl:.4f}"
        )
        if (s + 1) % args.eval_every == 0:
            log_eval(gstep, "rl")

    if rows[-1]["step"] != gstep:
        log_eval(gstep, "final")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        keys = [
            "rougeL_recall",
            "rougeL_f1",
            "teacher_judge",
            "student_ppl_on_teacher",
            "eval_reward",
            "avg_words",
        ]
        fig, ax = plt.subplots(2, 3, figsize=(14, 7))
        for a, k in zip(ax.ravel(), keys):
            a.plot([r["step"] for r in rows], [r[k] for r in rows], marker="o")
            a.set_title(k)
            a.set_xlabel("step")
            a.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(args.workdir, "metrics.png"), dpi=120)
        print(f"[plot] saved {args.workdir}/metrics.png")
    except Exception as e:
        print("[plot] skipped:", e)
    print(f"\nDone. Best model: {args.out} | metrics: {csv_path}")


if __name__ == "__main__":
    main()
