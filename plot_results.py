"""Generate benchmark figures for README."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

OUT = Path("figures")
OUT.mkdir(exist_ok=True)

COLORS = {
    "MLX 4-bit": "#4C9BE8",
    "llama.cpp Metal": "#E8864C",
    "Ollama": "#4CE89B",
    "vLLM direct": "#A64CE8",
    "vLLM HTTP": "#E84C4C",
}

plt.rcParams.update({
    "figure.facecolor": "#0d1117",
    "axes.facecolor": "#161b22",
    "axes.edgecolor": "#30363d",
    "axes.labelcolor": "#c9d1d9",
    "xtick.color": "#8b949e",
    "ytick.color": "#8b949e",
    "text.color": "#c9d1d9",
    "grid.color": "#21262d",
    "grid.linewidth": 0.8,
    "font.family": "monospace",
    "font.size": 11,
})


# ── Fig 1: Single-request throughput ─────────────────────────────────────────

backends = ["MLX 4-bit", "llama.cpp Metal", "Ollama", "vLLM direct", "vLLM HTTP"]
tps      = [34.52, 33.10, 29.99, 15.61, 3.24]
ttft     = [256,   344,   126,   341,   366]

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
fig.patch.set_facecolor("#0d1117")

for ax, values, xlabel, title in [
    (axes[0], tps,  "tokens / sec", "Decode Throughput"),
    (axes[1], ttft, "ms",           "Time to First Token (TTFT)"),
]:
    colors = [COLORS[b] for b in backends]
    bars = ax.barh(backends[::-1], values[::-1], color=colors[::-1],
                   height=0.55, edgecolor="none")
    for bar, val in zip(bars, values[::-1]):
        ax.text(val + max(values) * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val}", va="center", fontsize=10, color="#c9d1d9")
    ax.set_xlabel(xlabel)
    ax.set_title(title, pad=10, fontsize=13, color="#e6edf3", fontweight="bold")
    ax.set_xlim(0, max(values) * 1.18)
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    ax.spines[:].set_visible(False)

fig.suptitle("Qwen3.5-9B  |  Apple Silicon  |  Single Request",
             fontsize=14, color="#e6edf3", fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(OUT / "fig1_throughput.png", dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close()
print("fig1 done")


# ── Fig 2: TTFT vs input length ───────────────────────────────────────────────

input_lengths = [64, 512, 2048, 8192, 32768]

# Approximate values from bench_prefill_decode runs
ttft_data = {
    "MLX 4-bit":     [60,  256,  900,  3500, 18000],
    "llama.cpp Metal":[80, 344, 1200,  4500, 22000],
    "Ollama":        [50,  126,  500,  2000, 10000],
}

fig, ax = plt.subplots(figsize=(9, 5))
fig.patch.set_facecolor("#0d1117")

for label, values in ttft_data.items():
    ax.plot(input_lengths, values, marker="o", linewidth=2.2,
            markersize=7, color=COLORS[label], label=label)

# Linear reference from 64-tok baseline (MLX)
ref_base = 60
ref_vals = [ref_base * (l / 64) for l in input_lengths]
ax.plot(input_lengths, ref_vals, linestyle=":", linewidth=1.2,
        color="#555", label="O(n) ideal")

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xticks(input_lengths)
ax.set_xticklabels([str(x) for x in input_lengths])
ax.set_xlabel("Input tokens (log scale)")
ax.set_ylabel("TTFT ms (log scale)")
ax.set_title("Prefill Time vs Input Length\n(GatedDeltaNet O(n) -> SDPA superlinear at 32k)",
             fontsize=12, color="#e6edf3", fontweight="bold")
ax.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="#c9d1d9")
ax.grid(True, which="both", linestyle="--", alpha=0.3)
ax.spines[:].set_visible(False)

plt.tight_layout()
plt.savefig(OUT / "fig2_prefill_scaling.png", dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close()
print("fig2 done")


# ── Fig 3: Decode throughput vs input length ──────────────────────────────────

decode_data = {
    "MLX 4-bit":     [34, 34, 32, 17],
    "llama.cpp Metal":[33, 33, 31, 16],
    "Ollama":        [30, 30, 28, 14],
}
lengths_decode = [512, 2048, 8192, 32768]

fig, ax = plt.subplots(figsize=(9, 5))
fig.patch.set_facecolor("#0d1117")

for label, values in decode_data.items():
    ax.plot(lengths_decode, values, marker="o", linewidth=2.2,
            markersize=7, color=COLORS[label], label=label)

ax.axvspan(8192, 32768, alpha=0.07, color="#E84C4C", label="throughput cliff (32k)")
ax.set_xscale("log")
ax.set_xticks(lengths_decode)
ax.set_xticklabels([str(x) for x in lengths_decode])
ax.set_xlabel("Input tokens (log scale)")
ax.set_ylabel("Decode tokens / sec")
ax.set_title("Decode Throughput vs Input Length\n(stable up to 8k, halved at 32k due to SDPA KV cache)",
             fontsize=12, color="#e6edf3", fontweight="bold")
ax.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="#c9d1d9")
ax.grid(True, which="both", linestyle="--", alpha=0.3)
ax.spines[:].set_visible(False)

plt.tight_layout()
plt.savefig(OUT / "fig3_decode_vs_length.png", dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close()
print("fig3 done")


# ── Fig 4: Concurrency — MLX server vs Ollama ────────────────────────────────

with open("results/concurrency_results.json") as f:
    conc_data = json.load(f)

lengths_c     = [512, 2048, 8192]
levels        = [1, 2, 4]
length_colors = ["#4C9BE8", "#E8864C", "#4CE89B"]
backend_styles = {"mlx": "-", "ollama": "--", "llamacpp": ":"}
backend_labels = {"mlx": "MLX server", "ollama": "Ollama", "llamacpp": "llama-server"}

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
fig.patch.set_facecolor("#0d1117")
fig.suptitle("Concurrency Benchmark  |  MLX server vs Ollama vs llama-server  |  np=4 each",
             fontsize=12, color="#e6edf3", fontweight="bold", y=1.02)

for ax, metric, ylabel, title in [
    (axes[0], "aggregate_tok_per_sec", "Aggregate tok/s",
     "Aggregate Throughput\n(MLX batches; Ollama serializes)"),
    (axes[1], "avg_ttft_ms", "Avg TTFT (ms)",
     "Avg TTFT\n(lower is better)"),
]:
    handles = []
    for backend, linestyle in backend_styles.items():
        for color, length in zip(length_colors, lengths_c):
            rows = sorted(
                [r for r in conc_data
                 if r.get("input_tokens") == length and r["backend"] == backend],
                key=lambda r: r["concurrency"]
            )
            if not rows:
                continue
            xs = [r["concurrency"] for r in rows]
            ys = [r[metric] for r in rows]
            line, = ax.plot(xs, ys, marker="o", linewidth=2.2, markersize=7,
                            color=color, linestyle=linestyle,
                            label=f"{backend_labels[backend]} {length} tok")
            handles.append(line)

    if metric == "aggregate_tok_per_sec":
        mlx_base = next((r["aggregate_tok_per_sec"] for r in conc_data
                         if r.get("input_tokens") == 512 and r["concurrency"] == 1
                         and r["backend"] == "mlx"), None)
        if mlx_base:
            ax.plot(levels, [mlx_base * n for n in levels], linestyle=":",
                    linewidth=1.2, color="#555", label="ideal linear (MLX 512)")

    ax.set_xticks(levels)
    ax.set_xlabel("Concurrency")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11, color="#e6edf3", fontweight="bold")
    ax.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="#c9d1d9",
              fontsize=8, ncol=2)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines[:].set_visible(False)

# Legend: solid=MLX, dashed=Ollama
from matplotlib.lines import Line2D
style_legend = [
    Line2D([0], [0], color="#aaa", linestyle="-",  linewidth=2, label="MLX server (solid)"),
    Line2D([0], [0], color="#aaa", linestyle="--", linewidth=2, label="Ollama (dashed)"),
    Line2D([0], [0], color="#aaa", linestyle=":",  linewidth=2, label="llama-server (dotted)"),
]
axes[0].legend(handles=style_legend + [
    Line2D([0], [0], color=c, marker="o", linewidth=2, label=f"{l} tok")
    for c, l in zip(length_colors, lengths_c)
] + ([Line2D([0], [0], color="#555", linestyle=":", linewidth=1.2, label="ideal linear")] if True else []),
    facecolor="#161b22", edgecolor="#30363d", labelcolor="#c9d1d9", fontsize=8)

plt.tight_layout()
plt.savefig(OUT / "fig4_concurrency.png", dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close()
print("fig4 done")
