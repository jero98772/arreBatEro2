"""Collect existing training / energy / stage-eval logs for A, B, C into results/logs.json."""
import glob
import json
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PIPE = os.path.expanduser("~/Documents/llm_pipeline")
PROTO = os.path.expanduser("~/Desktop/QPT-LLM-prototype-no-lib")
WORK = {"B": os.path.join(PIPE, "work"), "C": os.path.join(PIPE, "work_350m")}
STAGES = ["base", "mid", "sft", "sft_lora", "dpo", "grpo", "distill"]
POST = ["sft", "sft_lora", "reward_model", "dpo", "grpo", "distill", "best_of_n"]


def jl(path):
    out = []
    for line in open(path):
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return out


def cumulative(vals):
    """Sum a cumulative counter that may reset on resume."""
    tot, prev = 0.0, 0.0
    for v in vals:
        if v < prev:
            tot += prev
        prev = v
    return tot + prev


def model_a():
    L = jl(os.path.join(PROTO, "logs", "training_metrics.jsonl"))
    # keep only the last run (an aborted first run reached step 4300 before a restart)
    steps = [(i, r["step"]) for i, r in enumerate(L) if r.get("record_type") == "train"]
    start = max([b[0] for a, b in zip(steps, steps[1:]) if b[1] < a[1]] or [0])
    L = L[start:]
    tr = [r for r in L if r.get("record_type") == "train"]
    va = [r for r in L if r.get("record_type") == "val" and r.get("val_loss") is not None]
    abl = [r for r in L if r.get("record_type") == "block_ablation"]
    arr = np.load(os.path.join(ROOT, "tokens_a", "train.npy"), mmap_mode="r")
    nonpad = float((arr[:20000] != 0).mean())
    tok_per_step = tr[1]["tokens_processed"] - tr[0]["tokens_processed"]
    tok_per_step /= (tr[1]["step"] - tr[0]["step"])
    sub = tr[::10]
    return {
        "nonpad_frac": nonpad,
        "train": [{"step": r["step"], "tokens": r["tokens_processed"] * nonpad, "loss": r["loss"]} for r in sub],
        "val": [{"step": r["step"], "tokens": r["step"] * tok_per_step * nonpad, "loss": r["val_loss"]} for r in va],
        "pretrain_kwh": tr[-1]["cumulative_energy_j"] / 3.6e6,
        "pretrain_seconds": tr[-1]["elapsed_s"],
        "pretrain_tokens_seen": tr[-1]["tokens_processed"] * nonpad,
        "pretrain_tokens_incl_pad": tr[-1]["tokens_processed"],
        "unique_train_tokens": int((arr != 0).sum()),
        "epochs": 10,
        "block_ablation_logged": abl[-1]["per_layer"] if abl else None,
        "final_val_loss": va[-1]["val_loss"] if va else None,
        "sources": ["wikitext", "ag_news", "imdb", "yelp_review_full", "amazon_polarity", "cc_news",
                    "billsum", "yahoo_answers_topics", "wikipedia_en"],
    }


def model_bc(w):
    out = {}
    s = json.load(open(os.path.join(w, "4_pre_train", "summary.json")))
    M = jl(os.path.join(w, "4_pre_train", "metrics.jsonl"))
    tr = [r for r in M if "loss" in r]
    ev = [r for r in M if "eval" in r]
    out["train"] = [{"step": r["step"], "tokens": r["tokens"], "loss": r["loss"]} for r in tr[::3]]
    out["val"] = [{"step": r["step"], "tokens": r["tokens"], **r["eval"]} for r in ev]
    out["pretrain_kwh"] = cumulative([r.get("kwh", 0) for r in tr])
    out["pretrain_seconds"] = cumulative([r.get("time", 0) for r in tr])
    out["pretrain_tokens_seen"] = s["tokens"]
    out["flops"] = s["flops"]
    out["flops_per_token"] = s["flops_per_token"]
    out["params"] = s["params"]
    out["params_nonemb"] = s["params_nonemb"]
    out["tokens_per_param"] = s["tokens"] / s["params"]
    out["final_eval"] = s["final_eval"]
    tk = json.load(open(os.path.join(w, "3_tokenization", "tokenize.json")))
    out["data_share"] = tk["domain_token_share"]
    out["unique_train_tokens"] = tk["train_tokens"]
    mid = json.load(open(os.path.join(w, "6_mid_training", "summary.json")))
    out["mid_kwh"] = mid["energy"]["total_kwh"]
    out["mid_seconds"] = mid["energy"]["seconds"]
    out["mid_tokens"] = mid.get("tokens")
    post = {}
    for st in POST:
        p = os.path.join(w, "7_post_training", st, "summary.json")
        if os.path.exists(p):
            e = json.load(open(p)).get("energy", {})
            post[st] = {"kwh": e.get("total_kwh"), "seconds": e.get("seconds")}
    out["post"] = post
    # stage evaluations recorded by the pipeline
    stages = {}
    for st in STAGES + ["pre_snap_010", "pre_snap_025", "pre_snap_050", "pre_snap_075"]:
        p = os.path.join(w, "eval", f"eval_{st}.json")
        if not os.path.exists(p):
            continue
        e = json.load(open(p))
        stages[st] = {k: e.get(k) for k in ("val", "benchmarks", "code", "math", "instruction_following",
                                             "safety_refusal_rate", "reward_model_score", "conversation")}
        if stages[st].get("conversation"):
            stages[st]["conversation"].pop("examples", None)
    out["stages"] = stages
    # snapshots fraction -> tokens
    out["snap_tokens"] = {f"pre_snap_{int(f*100):03d}": f * s["tokens"] for f in (0.1, 0.25, 0.5, 0.75)}
    # needle-in-a-haystack
    needle = {}
    for tag in ("needle_before", "needle_yarn", "needle_after"):
        p = os.path.join(w, "logs", f"{tag}.log")
        if os.path.exists(p):
            for line in open(p):
                m = re.search(r"\[(\w+)\] L=(\d+): ([\d. ]+)", line)
                if m:
                    needle.setdefault(m.group(1), {})[m.group(2)] = float(np.mean([float(x) for x in m.group(3).split()]))
    out["needle"] = needle
    j = os.path.join(w, "eval", "judge.json")
    out["judge"] = json.load(open(j)) if os.path.exists(j) else None
    return out


def main():
    R = {"A": model_a(), "B": model_bc(WORK["B"]), "C": model_bc(WORK["C"])}
    # A FLOPs ~ 6 N T (N = non-embedding params + attention term ignored)
    R["A"]["params"] = 54451200
    R["A"]["params_nonemb"] = 54451200 - 32000 * 512 - 512 * 512
    R["A"]["flops"] = 6 * R["A"]["params"] * R["A"]["pretrain_tokens_seen"]
    R["A"]["tokens_per_param"] = R["A"]["pretrain_tokens_seen"] / R["A"]["params"]
    json.dump(R, open(os.path.join(HERE, "results", "logs.json"), "w"), indent=1, default=float)
    for k, v in R.items():
        print(k, {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items()
                  if isinstance(vv, (int, float)) and not isinstance(vv, bool)})
        if "post" in v:
            print("  post", v["post"], "\n  stages", list(v["stages"]), "\n  needle", v["needle"])


if __name__ == "__main__":
    main()
