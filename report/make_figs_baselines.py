"""Comparison of A/B/C with GPT-3 (published, Brown et al. 2020) and open small models (measured here).
Writes figs/base_*.pdf, results/table_baselines.tex, results/macros_baselines.tex."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import make_figs as MF  # reuse style, palette and helpers  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES, FIG = MF.RES, MF.FIG
COL, MARK, INK, INK2 = MF.COL, MF.MARK, MF.INK, MF.INK2
GREY = "#8a8985"
DOMAINS = MF.DOMAINS

# GPT-3, zero-shot, Table H.1 (p. 63) and Tables 2.1 / D.1 of arXiv:2005.14165. Not re-measured.
GPT3 = {
    "names": ["Small", "Medium", "Large", "XL", "2.7B", "6.7B", "13B", "175B"],
    "params": [125e6, 356e6, 760e6, 1.32e9, 2.65e9, 6.66e9, 12.85e9, 174.6e9],
    "flops": [2.25e20, 6.41e20, 1.37e21, 2.38e21, 4.77e21, 1.20e22, 2.31e22, 3.14e23],
    "tokens": 300e9,
    "hellaswag": [33.7, 43.6, 51.0, 54.7, 62.8, 67.4, 70.9, 78.9],
    "lambada": [42.7, 54.3, 60.4, 63.6, 67.1, 70.3, 72.5, 76.2],
    "lambada_ppl": [18.6, 9.09, 6.53, 5.44, 4.60, 4.00, 3.56, 3.00],
    "piqa": [64.6, 70.2, 72.9, 75.1, 75.6, 78.0, 78.5, 81.0],
    "arc_easy": [43.6, 46.5, 53.0, 53.8, 58.2, 60.2, 63.8, 68.8],
}

# open baselines measured with our harness; training tokens as reported by the authors
OPEN = {
    "gpt2": dict(label="GPT-2 124M", year=2019, tokens=None, data="WebText"),
    "gpt2-medium": dict(label="GPT-2 355M", year=2019, tokens=None, data="WebText"),
    "opt-125m": dict(label="OPT 125M", year=2022, tokens=180e9, data="OPT mix"),
    "pythia-160m": dict(label="Pythia 160M", year=2023, tokens=300e9, data="The Pile"),
    "pythia-410m": dict(label="Pythia 410M", year=2023, tokens=300e9, data="The Pile"),
    "smollm2-135m": dict(label="SmolLM2 135M", year=2024, tokens=2e12, data="FineWeb-Edu, code, math"),
    "smollm2-360m": dict(label="SmolLM2 360M", year=2024, tokens=4e12, data="FineWeb-Edu, code, math"),
}
OURS = {"A": dict(label="A", year=2026), "B": dict(label="B", year=2026), "C": dict(label="C", year=2026)}

R = {m: json.load(open(os.path.join(RES, f"{m}.json"))) for m in list(OURS) + list(OPEN)
     if os.path.exists(os.path.join(RES, f"{m}.json"))}
LOG = MF.LOG
for m in OURS:
    OURS[m]["tokens"] = LOG[m]["pretrain_tokens_seen"]
BENCH = [("hellaswag", "acc_norm", "HellaSwag"), ("arc_easy", "acc", "ARC-Easy"),
         ("piqa", "acc_norm", "PIQA"), ("lambada", "acc", "LAMBADA")]


def acc(m, b, k):
    return 100 * R[m]["bench"][b][k]


def npar(m):
    return R[m]["params"]["total"]


def mean_bpb(m):
    return float(np.mean([R[m]["bpb"][d]["512"]["bpb"] for d in DOMAINS]))


def tokens(m):
    return (OURS.get(m) or OPEN.get(m))["tokens"]


def label(m):
    return (OURS.get(m) or OPEN.get(m))["label"]


def fig_bench_vs_size():
    fig, axes = plt.subplots(1, 4, figsize=(6.8, 2.35))
    for ax, (b, k, title) in zip(axes, BENCH):
        ax.plot(GPT3["params"], GPT3[b], color=INK, lw=1.4, marker="^", markersize=3.5,
                label="GPT-3 (paper, 300B tok.)")
        for m in OPEN:
            if m in R:
                ax.scatter(npar(m), acc(m, b, k), color=GREY, marker="o", s=16, zorder=3)
        for m in OURS:
            ax.scatter(npar(m), acc(m, b, k), color=COL[m], marker=MARK[m], s=34, zorder=4,
                       edgecolor="white", linewidth=0.8)
            ax.annotate(m, (npar(m), acc(m, b, k)), textcoords="offset points", xytext=(4, -9), fontsize=6.5,
                        color=INK)
        ch = 100 * R["A"]["bench"][b].get("chance", 0) if b != "lambada" else None
        if ch:
            ax.axhline(ch, color=INK2, lw=0.8, ls=(0, (3, 2)))
        ax.set_xscale("log")
        ax.set_xlim(3e7, 3e11)
        ax.set_title(title, fontsize=8)
        ax.set_xlabel("parameters", fontsize=7)
        ax.tick_params(labelsize=6.5)
    axes[0].set_ylabel("zero-shot accuracy (%)")
    hs = [plt.Line2D([], [], color=INK, marker="^", markersize=4, lw=1.4),
          plt.Line2D([], [], color=GREY, marker="o", lw=0, markersize=4)] + \
         [plt.Line2D([], [], color=COL[m], marker=MARK[m], lw=0, markersize=5) for m in OURS]
    fig.legend(hs, ["GPT-3 (published, Brown et al. 2020)", "open models (GPT-2, OPT, Pythia, SmolLM2; measured)",
                    "A", "B", "C"], ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.07), fontsize=6.5)
    fig.tight_layout(w_pad=0.6)
    MF.save(fig, "base_bench_vs_size")


def fig_bpb_rank():
    ms = [m for m in list(OURS) + list(OPEN) if m in R]
    ms.sort(key=mean_bpb)
    fig, ax = plt.subplots(figsize=(3.4, 2.9))
    y = np.arange(len(ms))
    vals = [mean_bpb(m) for m in ms]
    ax.barh(y, vals, color=[COL.get(m, GREY) for m in ms], height=0.62, edgecolor="white", linewidth=1)
    for yi, v in zip(y, vals):
        ax.text(v + 0.02, yi, f"{v:.3f}", va="center", fontsize=6.5, color=INK2)
    ax.set_yticks(y, [label(m) + ("" if m in OURS else "") for m in ms], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlim(0, max(vals) * 1.18)
    ax.set_xlabel("mean bpb, 4 domains (lower is better)")
    ax.grid(axis="y", visible=False)
    ax.set_title("Language-modelling quality, same text")
    MF.save(fig, "base_bpb_rank")


OFF = {"pythia-410m": (-4, 3), "pythia-160m": (-4, -8)}


def fig_data_efficiency():
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.6))
    # HellaSwag vs training tokens
    ax = axes[0]
    for i, n in enumerate(GPT3["names"][:4]):
        ax.scatter(GPT3["tokens"], GPT3["hellaswag"][i], color=INK, marker="^", s=18, zorder=3)
    ax.annotate("GPT-3\nSmall→XL", (GPT3["tokens"], GPT3["hellaswag"][3]), textcoords="offset points",
                xytext=(-38, -4), fontsize=6, color=INK)
    for m in list(OPEN) + list(OURS):
        if m not in R or tokens(m) is None:
            continue
        c = COL.get(m, GREY)
        ax.scatter(tokens(m), acc(m, "hellaswag", "acc_norm"), color=c, marker=MARK.get(m, "o"),
                   s=34 if m in OURS else 16, zorder=4, edgecolor="white", linewidth=0.6)
        ax.annotate(label(m), (tokens(m), acc(m, "hellaswag", "acc_norm")), textcoords="offset points",
                    xytext=OFF.get(m, (4, 2)), ha="right" if m in OFF else "left", fontsize=5.8, color=INK if m in OURS else INK2)
    ax.set_xscale("log")
    ax.set_xlabel("pre-training tokens")
    ax.set_ylabel("HellaSwag acc_norm (%)")
    ax.set_title("Data budget vs. commonsense")
    # HellaSwag vs compute
    ax = axes[1]
    ax.plot(GPT3["flops"][:4], GPT3["hellaswag"][:4], color=INK, marker="^", markersize=3.5, lw=1.4)
    ax.annotate("GPT-3", (GPT3["flops"][0], GPT3["hellaswag"][0]), textcoords="offset points", xytext=(4, -9),
                fontsize=6.5)
    for m in list(OPEN) + list(OURS):
        if m not in R or tokens(m) is None:
            continue
        f = LOG[m]["flops"] if m in OURS else 6 * npar(m) * tokens(m)
        ax.scatter(f, acc(m, "hellaswag", "acc_norm"), color=COL.get(m, GREY), marker=MARK.get(m, "o"),
                   s=34 if m in OURS else 16, zorder=4, edgecolor="white", linewidth=0.6)
        ax.annotate(label(m), (f, acc(m, "hellaswag", "acc_norm")), textcoords="offset points",
                    xytext=OFF.get(m, (4, 2)), ha="right" if m in OFF else "left",
                    fontsize=5.8, color=INK if m in OURS else INK2)
    ax.set_xscale("log")
    ax.set_xlabel("pre-training compute (FLOPs; open models ≈ 6ND)")
    ax.set_title("Compute vs. commonsense")
    fig.tight_layout()
    MF.save(fig, "base_efficiency")


def fmt_tok(t):
    if t is None:
        return "n/r"
    return f"{t/1e12:.0f}T" if t >= 1e12 else (f"{t/1e9:.0f}B" if t >= 1e9 else f"{t/1e9:.2f}B")


def write_tex():
    rows = []
    # GPT-3 (published)
    for i in (0, 1, 2, 3, 7):
        rows.append(("GPT-3 " + GPT3["names"][i] + r"$^\dagger$", "2020", f"{GPT3['params'][i]/1e6:,.0f}M",
                     "300B", "2048", f"{GPT3['hellaswag'][i]:.1f}", f"{GPT3['arc_easy'][i]:.1f}",
                     f"{GPT3['piqa'][i]:.1f}", f"{GPT3['lambada'][i]:.1f}", f"{GPT3['lambada_ppl'][i]:.1f}", "--"))
    for m in list(OPEN) + list(OURS):
        if m not in R:
            continue
        meta = OURS.get(m) or OPEN.get(m)
        lab = meta["label"] if m not in OURS else rf"\textbf{{\{m}{{}}}} (this work)"
        rows.append((lab, str(meta["year"]), f"{npar(m)/1e6:.0f}M", fmt_tok(meta["tokens"]),
                     str(R[m]["info"]["max_len"] if m in OURS else R[m]["info"]["max_len"]),
                     f"{acc(m,'hellaswag','acc_norm'):.1f}", f"{acc(m,'arc_easy','acc'):.1f}",
                     f"{acc(m,'piqa','acc_norm'):.1f}", f"{acc(m,'lambada','acc'):.1f}",
                     f"{R[m]['bench']['lambada']['ppl']:.1f}", f"{mean_bpb(m):.3f}"))
    L = [r"\begin{tabular}{llrrrrrrrrr}", r"\toprule",
         r"Model & Year & Params & Train tok. & Ctx & HellaSw. & ARC-E & PIQA & LAMB. acc & LAMB. ppl & bpb \\",
         r"\midrule"]
    for i, r in enumerate(rows):
        if i == 5 or (i == len(rows) - 3):
            L.append(r"\midrule")
        L.append(" & ".join(r) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(RES, "table_baselines.tex"), "w").write("\n".join(L) + "\n")

    mac = []

    def nc(n, v):
        mac.append(f"\\newcommand{{\\{n}}}{{{v}}}")
    for m, key in (("gpt2", "gptTwo"), ("gpt2-medium", "gptTwoMed"), ("pythia-160m", "pythiaS"),
                   ("pythia-410m", "pythiaM"), ("opt-125m", "optS"), ("smollm2-135m", "smolS"),
                   ("smollm2-360m", "smolM")):
        if m in R:
            nc(f"hs{key}", f"{acc(m,'hellaswag','acc_norm'):.1f}")
            nc(f"lamb{key}", f"{acc(m,'lambada','acc'):.1f}")
            nc(f"arc{key}", f"{acc(m,'arc_easy','acc'):.1f}")
            nc(f"piqa{key}", f"{acc(m,'piqa','acc_norm'):.1f}")
            nc(f"bpb{key}", f"{mean_bpb(m):.3f}")
    nc("flopsratioCgptS", f"{GPT3['flops'][0]/LOG['C']['flops']:.0f}")
    nc("tokratioCgptS", f"{GPT3['tokens']/LOG['C']['pretrain_tokens_seen']:.0f}")
    open(os.path.join(RES, "macros_baselines.tex"), "w").write("\n".join(mac) + "\n")


ALL = ["A", "B", "C", "gpt2", "gpt2-medium", "opt-125m", "pythia-160m", "pythia-410m", "smollm2-135m", "smollm2-360m"]


def fig_domain_heatmap():
    from matplotlib.colors import LinearSegmentedColormap
    ms = sorted([m for m in ALL if m in R], key=mean_bpb)
    doms = DOMAINS + ["a_test"]
    M = np.array([[R[m]["bpb"][d]["512"]["bpb"] for d in doms] for m in ms])
    cmap = LinearSegmentedColormap.from_list("seq", ["#1c5cab", "#3987e5", "#86b6ef", "#cde2fb", "#f4f8fd"])
    fig, ax = plt.subplots(figsize=(4.4, 3.3))
    ax.imshow(M, cmap=cmap, aspect="auto", vmin=M.min(), vmax=min(M.max(), 2.0))
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            dark = (M[i, j] - M.min()) / (min(M.max(), 2.0) - M.min()) < 0.35
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=6.8, color="white" if dark else INK)
    ax.set_xticks(range(len(doms)), ["Web", "Code", "Math", "Wiki", "A-test"], fontsize=7)
    ax.set_yticks(range(len(ms)), [label(m) for m in ms], fontsize=7)
    for t, m in zip(ax.get_yticklabels(), ms):
        if m in OURS:
            t.set_fontweight("bold")
    ax.tick_params(length=0)
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(doms)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(ms)), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.xaxis.tick_top()
    ax.set_title("Bits per byte by domain (darker = better)", fontsize=8, pad=18)
    MF.save(fig, "base_domain_heatmap")


def fig_speed_quality():
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    for m in ALL:
        if m not in R:
            continue
        x, y = R[m]["speed"]["prefill_tok_s_bs8"], mean_bpb(m)
        fp32 = R[m]["info"].get("fp32")
        ax.scatter(x, y, color=COL.get(m, GREY), marker=MARK.get(m, "o"), s=30 + npar(m) / 6e6,
                   facecolor="white" if fp32 else COL.get(m, GREY), edgecolor=COL.get(m, GREY), linewidth=1.2,
                   zorder=3)
        ax.annotate(label(m) + (" (fp32)" if fp32 else ""), (x, y), textcoords="offset points",
                    xytext={"pythia-410m": (6, -10), "smollm2-135m": (-6, 4)}.get(m, (5, 3)),
                    ha="right" if m == "smollm2-135m" else "left",
                    fontsize=6, color=INK if m in OURS else INK2)
    ax.set_xscale("log")
    ax.set_xlabel("prefill tokens/s (batch 8, seq 512, RTX 2070S)")
    ax.set_ylabel("mean bpb (lower is better)")
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v/1000:.0f}k"))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_title("Speed vs. quality (area ∝ params)", fontsize=8)
    MF.save(fig, "base_speed_quality")


if __name__ == "__main__":
    fig_bench_vs_size()
    fig_bpb_rank()
    fig_data_efficiency()
    fig_domain_heatmap()
    fig_speed_quality()
    write_tex()
    print("ok", sorted(R))
