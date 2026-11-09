"""Figures (figs/*.pdf) and LaTeX macros/tables (results/*.tex) from results/{A,B,C,logs}.json."""
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES, FIG = os.path.join(HERE, "results"), os.path.join(HERE, "figs")
os.makedirs(FIG, exist_ok=True)

MODELS = ["A", "B", "C"]
COL = {"A": "#2a78d6", "B": "#eb6834", "C": "#1baf7a"}          # categorical slots 1-3 (validated all-pairs)
MARK = {"A": "o", "B": "s", "C": "D"}                             # secondary encoding
SEQ = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]               # ordinal blue ramp (>=250)
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
DOMAINS = ["web", "code", "math", "wiki"]
DLAB = {"web": "Web", "code": "Code", "math": "Math", "wiki": "Wiki", "a_test": "A-test\n(lower-cased)"}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False, "lines.linewidth": 1.6,
    "lines.markersize": 5, "savefig.bbox": "tight", "savefig.pad_inches": 0.03, "pdf.fonttype": 42,
})

R = {m: json.load(open(os.path.join(RES, f"{m}.json"))) for m in MODELS}
LOG = json.load(open(os.path.join(RES, "logs.json")))
NAME = {"A": "A (54M, GPT-2 style)", "B": "B (101M, Llama style)", "C": "C (344M, Llama style)"}


def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".pdf"))
    fig.savefig(os.path.join(FIG, name + ".png"), dpi=110)
    plt.close(fig)


def wilson(p, n, z=1.96):
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def mean_bpb(m, ctx="512"):
    return float(np.mean([R[m]["bpb"][d][ctx]["bpb"] for d in DOMAINS]))


def legend_models(ax, **kw):
    hs = [plt.Line2D([], [], color=COL[m], marker=MARK[m], lw=0, markersize=6) for m in MODELS]
    ax.legend(hs, [NAME[m] for m in MODELS], **kw)


# ---------------------------------------------------------------- 1. bpb per domain
def fig_bpb():
    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    doms = DOMAINS + ["a_test"]
    x = np.arange(len(doms))
    w = 0.26
    for i, m in enumerate(MODELS):
        v = [R[m]["bpb"][d]["512"]["bpb"] for d in doms]
        ax.bar(x + (i - 1) * w, v, w, color=COL[m], edgecolor="white", linewidth=1, label=NAME[m] + " @512")
        for xi, vi in zip(x + (i - 1) * w, v):
            ax.text(xi, vi + 0.03, f"{vi:.2f}", ha="center", va="bottom", fontsize=6, color=INK2)
        if "2048" in R[m]["bpb"]["web"]:
            v2 = [R[m]["bpb"][d]["2048"]["bpb"] for d in doms]
            ax.scatter(x + (i - 1) * w, v2, marker="_", s=90, color=INK, linewidths=1.4, zorder=3)
    ax.scatter([], [], marker="_", s=90, color=INK, label="same model @2048 ctx")
    ax.set_xticks(x, [DLAB[d] for d in doms])
    ax.set_ylabel("bits per byte  (lower is better)")
    ax.set_ylim(0, 2.75)
    ax.grid(axis="x", visible=False)
    ax.legend(ncol=4, fontsize=6.5, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    save(fig, "bpb_domains")


# ---------------------------------------------------------------- 2. benchmarks
BENCH = [("hellaswag", "acc_norm", "HellaSwag (acc_norm)"), ("arc_easy", "acc", "ARC-Easy (acc)"),
         ("piqa", "acc_norm", "PIQA (acc_norm)"), ("lambada", "acc", "LAMBADA (last-word acc)")]


def fig_bench():
    fig, axes = plt.subplots(1, 4, figsize=(6.6, 2.1), sharey=False)
    for ax, (b, k, title) in zip(axes, BENCH):
        for i, m in enumerate(MODELS):
            r = R[m]["bench"][b]
            p, n = r[k], r["n"]
            lo, hi = wilson(p, n)
            ax.errorbar(i, p, yerr=[[p - lo], [hi - p]], fmt=MARK[m], color=COL[m], ecolor=COL[m],
                        elinewidth=1.4, capsize=3, markersize=6)
            ax.text(i + 0.12, p, f"{100*p:.1f}", va="center", fontsize=6.5, color=INK2)
        ch = R["A"]["bench"][b].get("chance")
        if ch:
            ax.axhline(ch, color=INK2, lw=1, ls=(0, (3, 2)))
            ax.text(2.45, ch, "chance", fontsize=6, color=INK2, va="bottom", ha="right")
        ax.set_title(title, fontsize=7.5)
        ax.set_xticks(range(3), MODELS)
        ax.set_xlim(-0.5, 2.6)
        ax.grid(axis="x", visible=False)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    fig.tight_layout(w_pad=0.8)
    save(fig, "benchmarks")


# ---------------------------------------------------------------- 3. training curves (bpb vs tokens)
def fig_curves():
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.5))
    ax = axes[0]
    bpt_a = R["A"]["bpb"]["a_test"]["512"]["bytes_per_token"]
    va = LOG["A"]["val"]
    ax.plot([r["tokens"] for r in va], [r["loss"] / math.log(2) / bpt_a for r in va], color=COL["A"],
            label="A (own val set, approx.)")
    for m in ("B", "C"):
        v = LOG[m]["val"]
        ax.plot([r["tokens"] for r in v], [r["bpb"] for r in v], color=COL[m], label=f"{m} (pipeline val, 4 domains)")
    ax.set_xscale("log")
    ax.set_xlabel("pre-training tokens seen (non-pad)")
    ax.set_ylabel("validation bits per byte")
    ax.set_ylim(0.75, 2.2)
    ax.set_title("Learning curves")
    ax.legend(fontsize=6.5)
    # 3b. HellaSwag vs tokens (snapshots)
    ax = axes[1]
    for m in ("B", "C"):
        st = LOG[m]["stages"]
        pts = [(LOG[m]["snap_tokens"][k], st[k]["benchmarks"]["hellaswag"]["acc_norm"])
               for k in sorted(LOG[m]["snap_tokens"]) if k in st]
        pts.append((LOG[m]["pretrain_tokens_seen"], st["base"]["benchmarks"]["hellaswag"]["acc_norm"]))
        ax.plot(*zip(*pts), color=COL[m], marker=MARK[m], label=f"{m} pre-training snapshots")
    ax.scatter([LOG["A"]["pretrain_tokens_seen"]], [R["A"]["bench"]["hellaswag"]["acc_norm"]], color=COL["A"],
               marker=MARK["A"], s=36, zorder=3, label="A final")
    ax.axhline(0.25, color=INK2, lw=1, ls=(0, (3, 2)))
    ax.text(LOG["A"]["pretrain_tokens_seen"], 0.252, "chance", fontsize=6, color=INK2, va="bottom")
    ax.set_xscale("log")
    ax.set_xlabel("pre-training tokens seen")
    ax.set_ylabel("HellaSwag acc_norm")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_title("Commonsense emerges with data")
    ax.legend(fontsize=6.5, loc="upper left")
    fig.tight_layout()
    save(fig, "curves")


# ---------------------------------------------------------------- 4. scaling: compute vs quality, energy by stage
def fig_scaling():
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.5), gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axes[0]
    for m in MODELS:
        x, y = LOG[m]["flops"], mean_bpb(m)
        ax.scatter(x, y, color=COL[m], marker=MARK[m], s=60, zorder=3, edgecolor="white", linewidth=1)
        ax.annotate(f"{m}  {y:.3f}", (x, y), textcoords="offset points", xytext=(7, 3), fontsize=7, color=INK)
    xs = np.array([LOG[m]["flops"] for m in MODELS])
    ys = np.array([mean_bpb(m) for m in MODELS])
    k, b = np.polyfit(np.log10(xs), np.log10(ys), 1)
    xx = np.logspace(16.8, 19.4, 50)
    ax.plot(xx, 10 ** (b + k * np.log10(xx)), color=INK2, lw=1, ls=(0, (3, 2)))
    ax.text(3e17, 10 ** (b + k * np.log10(3e17)) * 0.93, f"power-law fit: bpb ∝ C^{k:.3f}", fontsize=6.5, color=INK2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.1f"))
    ax.yaxis.set_minor_formatter(matplotlib.ticker.FormatStrFormatter("%.1f"))
    ax.set_xlabel("pre-training compute (FLOPs)")
    ax.set_ylabel("mean bpb, 4 domains @512")
    ax.set_title("Compute vs. quality")
    ax = axes[1]
    stages = [("pre-train", lambda m: LOG[m]["pretrain_kwh"]),
              ("mid-train", lambda m: LOG[m].get("mid_kwh", 0)),
              ("post-train", lambda m: sum((v["kwh"] or 0) for v in LOG[m].get("post", {}).values()))]
    left = np.zeros(3)
    for (lab, f), c in zip(stages, SEQ[1:]):
        v = np.array([f(m) for m in MODELS])
        ax.barh(range(3), v, left=left, color=c, edgecolor="white", linewidth=1, label=lab, height=0.55)
        left += v
    for i, t in enumerate(left):
        ax.text(t * 1.02, i, f"{t:.1f} kWh", va="center", fontsize=7, color=INK)
    ax.set_yticks(range(3), MODELS)
    ax.invert_yaxis()
    ax.set_xlim(0, left.max() * 1.35)
    ax.set_xlabel("measured energy (kWh)")
    ax.set_title("Training energy by stage")
    ax.grid(axis="y", visible=False)
    ax.legend(fontsize=6.5, loc="center right")
    fig.tight_layout()
    save(fig, "scaling_energy")


# ---------------------------------------------------------------- 5. efficiency
def fig_efficiency():
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.7))
    ax = axes[0]
    for m in MODELS:
        x, y = R[m]["speed"]["prefill_tok_s_bs8"], mean_bpb(m)
        ax.scatter(x, y, s=40 + R[m]["params"]["total"] / 2e6, color=COL[m], marker=MARK[m], alpha=0.9,
                   edgecolor="white", linewidth=1.5, zorder=3)
        ax.annotate(m, (x, y), textcoords="offset points", xytext=(9, 4), fontsize=8)
    ax.set_xlim(0, 175000)
    ax.set_ylim(0.85, 1.95)
    ax.set_xlabel("prefill tokens/s (batch 8, seq 512, fp16)")
    ax.set_ylabel("mean bpb @512 (lower is better)")
    ax.set_title("Quality vs. speed\n(marker area ∝ parameters)")
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v/1000:.0f}k"))
    ax = axes[1]
    metr = [("decode\nms/token", lambda m: R[m]["speed"]["decode_ms_per_token_nocache"]),
            ("peak VRAM\n(GB)", lambda m: R[m]["peak_vram_gb"]),
            ("checkpoint\n(GB)", lambda m: R[m]["ckpt_mb"] / 1024)]
    x = np.arange(len(metr))
    w = 0.26
    for i, m in enumerate(MODELS):
        vals = [f(m) for _, f in metr]
        ref = [max(f(mm) for mm in MODELS) for _, f in metr]
        ax.bar(x + (i - 1) * w, np.array(vals) / np.array(ref), w, color=COL[m], edgecolor="white", linewidth=1)
        for xi, v, r in zip(x + (i - 1) * w, vals, ref):
            ax.text(xi, v / r + 0.02, f"{v:.1f}", ha="center", fontsize=6, color=INK2)
    ax.set_xticks(x, [l for l, _ in metr])
    ax.set_ylabel("relative to largest")
    ax.set_ylim(0, 1.18)
    ax.set_title("Inference cost\n(lower is better)")
    ax.grid(axis="x", visible=False)
    legend_models(ax, fontsize=6, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    save(fig, "efficiency")


# ---------------------------------------------------------------- 6. ablation + weights
def fig_layers():
    fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.2))
    ax = axes[0]
    for m in MODELS:
        d = np.array(R[m]["ablation"]["delta_bpb"])
        L = len(d)
        ax.plot((np.arange(L) + 0.5) / L, d, color=COL[m], marker=MARK[m], markersize=3.5,
                label=m)
    ax.set_yscale("symlog", linthresh=0.05, linscale=0.6)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xlabel("relative depth (layer / L)")
    ax.set_ylabel("Δ bpb when layer removed")
    ax.set_title("Layer ablation")
    ax.legend(fontsize=6.5, loc="upper center", ncol=3)
    for ax, key, title in ((axes[1], "attn_rms", "Attention weight RMS"), (axes[2], "mlp_rms", "MLP weight RMS")):
        for m in MODELS:
            v = [w[key] for w in R[m]["weights"]]
            L = len(v)
            ax.plot((np.arange(L) + 0.5) / L, v, color=COL[m], marker=MARK[m], markersize=3.5)
        ax.set_xlabel("relative depth")
        ax.set_title(title)
    fig.tight_layout()
    save(fig, "layers")


# ---------------------------------------------------------------- 7. stage progression (post-training)
STAGE_ORDER = ["base", "mid", "sft", "dpo", "grpo", "distill"]
STAGE_METRICS = [
    ("HellaSwag", lambda s: s["benchmarks"]["hellaswag"]["acc_norm"]),
    ("ARC-Easy", lambda s: s["benchmarks"]["arc_easy"]["acc"]),
    ("GSM8K", lambda s: s["math"]["gsm8k_acc"]),
    ("Synthetic math", lambda s: s["math"]["synthetic_acc"]),
    ("HumanEval pass@1", lambda s: s["code"]["humaneval_pass@1_greedy"]),
    ("Instruction following", lambda s: s["instruction_following"]["overall"]
     if isinstance(s.get("instruction_following"), dict) else s.get("instruction_following")),
]


def fig_stages():
    fig, axes = plt.subplots(2, 3, figsize=(6.6, 3.6), sharex=True)
    for ax, (lab, f) in zip(axes.flat, STAGE_METRICS):
        for m in ("B", "C"):
            st = LOG[m]["stages"]
            xs, ys = [], []
            for i, s in enumerate(STAGE_ORDER):
                if s in st:
                    try:
                        y = f(st[s])
                    except (KeyError, TypeError):
                        y = None
                    if y is not None:
                        xs.append(i)
                        ys.append(y)
            ax.plot(xs, ys, color=COL[m], marker=MARK[m], markersize=4, label=m)
        # fresh final-checkpoint measurements from this report where available
        if lab in ("HellaSwag", "ARC-Easy"):
            b = "hellaswag" if lab == "HellaSwag" else "arc_easy"
            k = "acc_norm" if lab == "HellaSwag" else "acc"
            for m in MODELS:
                ax.scatter(len(STAGE_ORDER) - 1 + 0.18, R[m]["bench"][b][k], marker=MARK[m], facecolor="white",
                           edgecolor=COL[m], linewidth=1.4, s=26, zorder=4)
        ax.set_title(lab, fontsize=8)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    for ax in axes[1]:
        ax.set_xticks(range(len(STAGE_ORDER)), STAGE_ORDER, rotation=35, fontsize=7)
    hs = [plt.Line2D([], [], color=COL[m], marker=MARK[m]) for m in ("B", "C")]
    hs.append(plt.Line2D([], [], color=INK2, marker="o", markerfacecolor="white", lw=0))
    fig.legend(hs, ["B (pipeline logs)", "C (pipeline logs)", "final ckpt, this report (A/B/C)"], ncol=3,
               loc="upper center", bbox_to_anchor=(0.5, 1.04), fontsize=7)
    fig.tight_layout()
    save(fig, "stages")


# ---------------------------------------------------------------- 8. needle-in-a-haystack
def fig_needle():
    fig, ax = plt.subplots(figsize=(3.3, 2.2))
    styles = [("C", "before", "C before mid-training", "-", "white"),
              ("C", "before_yarn", "C before, YaRN-scaled RoPE", (0, (3, 2)), "white"),
              ("C", "after", "C after mid-training", "-", COL["C"]),
              ("B", "after", "B after mid-training", "-", COL["B"])]
    for m, k, lab, ls, fc in styles:
        d = LOG[m]["needle"].get(k, {})
        xs = sorted(int(x) for x in d)
        ax.plot(xs, [d[str(x)] for x in xs], color=COL[m], ls=ls, marker=MARK[m], markerfacecolor=fc,
                markersize=4, label=lab)
    ax.set_xscale("log", base=2)
    ax.set_xticks([512, 1024, 2048, 4096], ["512", "1k", "2k", "4k"])
    ax.set_xlabel("context length (tokens)")
    ax.set_ylabel("retrieval accuracy")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.axvline(512, color=COL["A"], lw=1, ls=":")
    ax.text(530, 0.45, "A's hard\nlimit (512)", fontsize=6, color=INK2)
    ax.legend(fontsize=6, loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=2)
    ax.set_title("Needle-in-a-haystack (pipeline logs)")
    save(fig, "needle")


# ---------------------------------------------------------------- 9. scorecard heatmap
def scorecard_rows():
    rows = [
        ("Mean bpb @512 ↓", [mean_bpb(m) for m in MODELS], False, "{:.3f}"),
        ("Web bpb ↓", [R[m]["bpb"]["web"]["512"]["bpb"] for m in MODELS], False, "{:.3f}"),
        ("Code bpb ↓", [R[m]["bpb"]["code"]["512"]["bpb"] for m in MODELS], False, "{:.3f}"),
        ("A-test bpb ↓", [R[m]["bpb"]["a_test"]["512"]["bpb"] for m in MODELS], False, "{:.3f}"),
        ("HellaSwag ↑", [R[m]["bench"]["hellaswag"]["acc_norm"] for m in MODELS], True, "{:.1%}"),
        ("ARC-Easy ↑", [R[m]["bench"]["arc_easy"]["acc"] for m in MODELS], True, "{:.1%}"),
        ("PIQA ↑", [R[m]["bench"]["piqa"]["acc_norm"] for m in MODELS], True, "{:.1%}"),
        ("LAMBADA ↑", [R[m]["bench"]["lambada"]["acc"] for m in MODELS], True, "{:.1%}"),
        ("Throughput ↑", [R[m]["speed"]["prefill_tok_s_bs8"] / 1000 for m in MODELS], True, "{:.0f}k tok/s"),
        ("Peak VRAM ↓", [R[m]["peak_vram_gb"] for m in MODELS], False, "{:.2f} GB"),
        ("Train energy ↓", [LOG[m]["pretrain_kwh"] + LOG[m].get("mid_kwh", 0) +
                            sum((v["kwh"] or 0) for v in LOG[m].get("post", {}).values()) for m in MODELS],
         False, "{:.1f} kWh"),
        ("Context window ↑", [R[m]["info"]["max_len"] for m in MODELS], True, "{:.0f}"),
    ]
    return rows


def fig_scorecard():
    rows = scorecard_rows()
    M = np.zeros((len(rows), 3))
    for i, (_, v, hib, _) in enumerate(rows):
        v = np.array(v, float)
        s = (v - v.min()) / (v.max() - v.min() + 1e-12)
        M[i] = s if hib else 1 - s
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seq", ["#f4f8fd", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab"])
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    ax.imshow(M, cmap=cmap, aspect="auto", vmin=0, vmax=1)
    for i, (_, v, _, fmt) in enumerate(rows):
        for j in range(3):
            ax.text(j, i, fmt.format(v[j]), ha="center", va="center", fontsize=7,
                    color="white" if M[i, j] > 0.62 else INK)
    ax.set_xticks(range(3), [NAME[m].replace(" (", "\n(") for m in MODELS], fontsize=7)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=7)
    ax.tick_params(length=0)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-0.5, 3), minor=True)
    ax.set_yticks(np.arange(-0.5, len(rows)), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.xaxis.tick_top()
    ax.set_title("Scorecard (cell shade = relative standing, darker = better)", fontsize=7.5, pad=26)
    save(fig, "scorecard")


# ---------------------------------------------------------------- 10. data and size
def fig_data():
    fig, ax = plt.subplots(figsize=(3.3, 2.4))
    for m in MODELS:
        n, t = LOG[m]["params"], LOG[m]["pretrain_tokens_seen"]
        ax.scatter(n, t, color=COL[m], marker=MARK[m], s=55, zorder=3, edgecolor="white", linewidth=1)
        ax.annotate(f"{m}: {t/n:.0f} tok/param", (n, t), textcoords="offset points", xytext=(7, -3), fontsize=6.5)
    nn_ = np.logspace(7.5, 9, 10)
    ax.plot(nn_, 20 * nn_, color=INK2, lw=1, ls=(0, (3, 2)))
    ax.text(3.3e7, 20 * 3.3e7 * 3.2, "Chinchilla-optimal\n(20 tokens/param)", fontsize=6, color=INK2)
    ax.set_ylim(2e8, 3e10)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("parameters (unique)")
    ax.set_ylabel("pre-training tokens seen")
    ax.set_title("Model size vs. data")
    save(fig, "data_scale")


# ---------------------------------------------------------------- LaTeX macros + tables
def tex_escape(s):
    rep = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
           "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}", "°": r"\textdegree{}"}
    s = "".join(rep.get(c, c) for c in s)
    return "".join(c for c in s if ord(c) < 0x2000 or c in "–—’“”")


def write_tex():
    mac = []

    def nc(name, val):
        mac.append(f"\\newcommand{{\\{name}}}{{{val}}}")

    for m in MODELS:
        p = R[m]["params"]
        nc(f"params{m}", f"{p['total']/1e6:.1f}")
        nc(f"meanbpb{m}", f"{mean_bpb(m):.3f}")
        for d in DOMAINS + ["a_test"]:
            nc(f"bpb{m}{d.replace('_', '')}", f"{R[m]['bpb'][d]['512']['bpb']:.3f}")
        for b, k, _ in BENCH:
            nc(f"{b.replace('_', '')}{m}", f"{100*R[m]['bench'][b][k]:.1f}")
        nc(f"lambadappl{m}", f"{R[m]['bench']['lambada']['ppl']:.0f}")
        nc(f"tput{m}", f"{R[m]['speed']['prefill_tok_s_bs8']/1000:.0f}")
        nc(f"decode{m}", f"{R[m]['speed']['decode_ms_per_token_nocache']:.1f}")
        nc(f"vram{m}", f"{R[m]['peak_vram_gb']:.2f}")
        nc(f"tokens{m}", f"{LOG[m]['pretrain_tokens_seen']/1e9:.2f}")
        nc(f"flops{m}", f"{LOG[m]['flops']:.2e}".replace("e+", r"\times10^{") + "}")
        nc(f"kwhpre{m}", f"{LOG[m]['pretrain_kwh']:.2f}")
        tot = LOG[m]["pretrain_kwh"] + LOG[m].get("mid_kwh", 0) + sum((v["kwh"] or 0) for v in LOG[m].get("post", {}).values())
        nc(f"kwhtot{m}", f"{tot:.1f}")
        nc(f"hours{m}", f"{(LOG[m]['pretrain_seconds'])/3600:.1f}")
        nc(f"tpp{m}", f"{LOG[m]['tokens_per_param']:.0f}")
        if "2048" in R[m]["bpb"]["web"]:
            nc(f"meanbpblong{m}", f"{mean_bpb(m, '2048'):.3f}")
    nc("nonpadA", f"{100*LOG['A']['nonpad_frac']:.0f}")
    nc("uniqueA", f"{LOG['A']['unique_train_tokens']/1e6:.0f}")
    nc("benchN", str(R["A"]["bench"]["hellaswag"]["n"]))
    open(os.path.join(RES, "macros.tex"), "w").write("\n".join(mac) + "\n")

    # full numbers table
    lines = [r"\begin{tabular}{lrrr}", r"\toprule", r"Metric & A & B & C \\", r"\midrule"]
    for lab, v, hib, fmt in scorecard_rows():
        best = int(np.argmax(v) if hib else np.argmin(v))
        cells = []
        for j in range(3):
            s = tex_escape(fmt.format(v[j]))
            cells.append(r"\textbf{" + s + "}" if j == best else s)
        lab = lab.replace("↓", r"$\downarrow$").replace("↑", r"$\uparrow$")
        lines.append(f"{lab} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(RES, "table_scorecard.tex"), "w").write("\n".join(lines) + "\n")

    # bpb table incl. 2048
    lines = [r"\begin{tabular}{l" + "r" * 5 + "}", r"\toprule",
             r"Model / context & Web & Code & Math & Wiki & A-test \\", r"\midrule"]
    for m in MODELS:
        for ctx in ("512", "2048"):
            if ctx not in R[m]["bpb"]["web"]:
                continue
            vals = [R[m]["bpb"][d][ctx]["bpb"] for d in DOMAINS + ["a_test"]]
            lines.append(f"{m} @ {ctx} & " + " & ".join(f"{v:.3f}" for v in vals) + r" \\")
    lines.append(r"\midrule")
    for m in MODELS:
        vals = [R[m]["bpb"][d]["512"]["bytes_per_token"] for d in DOMAINS + ["a_test"]]
        lines.append(f"{m} bytes/token & " + " & ".join(f"{v:.2f}" for v in vals) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(RES, "table_bpb.tex"), "w").write("\n".join(lines) + "\n")

    # samples
    out = []
    for m in MODELS:
        for s in R[m]["samples"]:
            txt = s["greedy"].replace("\n", " ⏎ ").strip()
            txt = txt[:230] + ("…" if len(txt) > 230 else "")
            out.append(f"{m} & {s['mode']} & \\texttt{{{tex_escape(s['prompt'][:60])}}} & "
                       f"{tex_escape(txt).replace('⏎', '$\\hookleftarrow$').replace('…', '\\ldots')} \\\\")
    head = ("\\begin{longtable}{p{0.05\\linewidth}p{0.05\\linewidth}p{0.22\\linewidth}p{0.58\\linewidth}}\n"
            "\\caption{Greedy samples from each model ($\\hookleftarrow$ marks a newline).}\\label{tab:samples}\\\\\n"
            "\\toprule\nModel & Mode & Prompt & Greedy continuation (64--96 tokens, truncated) \\\\\n\\midrule\n\\endhead\n")
    open(os.path.join(RES, "table_samples.tex"), "w").write(
        head + "\n\\midrule\n".join(out) + "\n\\bottomrule\n\\end{longtable}\n")


if __name__ == "__main__":
    fig_bpb()
    fig_bench()
    fig_curves()
    fig_scaling()
    fig_efficiency()
    fig_layers()
    fig_stages()
    fig_needle()
    fig_scorecard()
    fig_data()
    write_tex()
    print("ok")
