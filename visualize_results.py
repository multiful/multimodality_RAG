"""Render data/eval/results.csv (from evaluate.py) as a grouped-bar comparison chart.

Usage:
    python visualize_results.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

RESULTS_CSV = "data/eval/results.csv"
OUT_PATH = Path("data/eval/results_chart.png")

# Fixed model order + categorical colors (palette slots 1-3: blue, orange, aqua).
MODEL_COLORS = {
    "bge-m3": "#2a78d6",
    "bge-m3-ko": "#eb6834",
    "text-embedding-3-small": "#1baf7a",
}
QUERY_TYPE_ORDER = ["financial", "news", "price", "trend", "overall"]

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"


def plot_metric(ax, df: pd.DataFrame, metric: str, title: str):
    models = list(MODEL_COLORS.keys())
    n_models = len(models)
    n_types = len(QUERY_TYPE_ORDER)
    bar_width = 0.8 / n_models
    x = range(n_types)

    for i, model in enumerate(models):
        sub = df[df["model"] == model].set_index("query_type").reindex(QUERY_TYPE_ORDER)
        offsets = [xi + (i - (n_models - 1) / 2) * bar_width for xi in x]
        bars = ax.bar(
            offsets, sub[metric], width=bar_width * 0.9,
            color=MODEL_COLORS[model], label=model, zorder=3,
        )
        for bar, val in zip(bars, sub[metric]):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.2f}", ha="center", va="bottom", fontsize=8, color=INK_SECONDARY,
            )

    ax.set_title(title, color=INK_PRIMARY, fontsize=12, fontweight="bold", loc="left")
    ax.set_xticks(list(x))
    ax.set_xticklabels(QUERY_TYPE_ORDER, color=INK_SECONDARY, fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.tick_params(axis="y", labelcolor=INK_MUTED, labelsize=8, length=0)
    ax.tick_params(axis="x", length=0)
    ax.set_facecolor(SURFACE)
    ax.yaxis.grid(True, color=GRIDLINE, linewidth=1, zorder=0)
    ax.xaxis.grid(False)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)


def main():
    df = pd.read_csv(RESULTS_CSV)
    recall_col = [c for c in df.columns if c.startswith("recall@")][0]

    fig, axes = plt.subplots(2, 1, figsize=(8, 8), facecolor=SURFACE)
    plot_metric(axes[0], df, "mrr", "MRR (순위 품질) — 높을수록 정답을 상위에 랭킹")
    plot_metric(axes[1], df, recall_col, f"{recall_col} — top-5 안에 정답이 있었는지")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.03),
        ncol=3, frameon=False, fontsize=9, labelcolor=INK_SECONDARY,
    )
    fig.suptitle(
        "BGE-M3 vs BGE-m3-ko vs GPT text-embedding-3-small", color=INK_PRIMARY,
        fontsize=13, fontweight="bold", y=1.08,
    )
    fig.tight_layout(rect=[0, 0, 1, 1.0])

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    print(f"saved chart to {OUT_PATH}")


if __name__ == "__main__":
    main()
