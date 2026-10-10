"""Combine idle-probability and DQI backbone validation curves into one 2x2 figure."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from cvn_rl.paper_figure_style import apply_ieee_strict_style, style_full_frame

IDLE_LOG = Path("paper_figures/idle_temporal_backbone_trends_full40_acc/idle_temporal_backbone_training_log.csv")
DQI_LOG = Path("paper_figures/dqi_temporal_backbone_trends_full40_listnet/dqi_temporal_backbone_training_log.csv")
OUTPUT_DIR = Path("paper_figures/combined_temporal_backbone_trends")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS = ["RNN", "GRU", "LSTM", "Attention-LSTM"]
COLORS = {"RNN": "#c27ba0", "GRU": "#e74c3c", "LSTM": "#4c78a8", "Attention-LSTM": "#222222"}
MARKERS = {"RNN": "^", "GRU": "v", "LSTM": "s", "Attention-LSTM": "X"}


def plot_metric(ax: plt.Axes, df: pd.DataFrame, metric: str, ylabel: str) -> None:
    for model in MODELS:
        sub = df[df["model"] == model]
        if sub.empty:
            continue
        ax.plot(
            sub["epoch"],
            sub[metric],
            marker=MARKERS[model],
            markevery=2,
            markersize=3.0,
            linewidth=1.0,
            color=COLORS[model],
            label=model,
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.28, linewidth=0.3)
    ax.tick_params(axis="both", labelsize=6.8, pad=1.5)
    ax.set_box_aspect(1.0)
    style_full_frame(ax)


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        0.5,
        -0.245,
        label,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=7.4,
        fontweight="bold",
        clip_on=False,
    )


def main() -> None:
    apply_ieee_strict_style()
    idle = pd.read_csv(IDLE_LOG)
    dqi = pd.read_csv(DQI_LOG)

    fig, axes = plt.subplots(2, 2, figsize=(5.55, 4.95))
    plot_metric(axes[0, 0], idle, "val_loss", "Idle Val. Loss")
    plot_metric(axes[0, 1], idle, "val_accuracy", "Idle Val. Acc.")
    plot_metric(axes[1, 0], dqi, "val_mse", "DQI Val. MSE")
    plot_metric(axes[1, 1], dqi, "val_listnet_loss", "DQI Val. ListNet")

    add_panel_label(axes[0, 0], "(a)")
    add_panel_label(axes[0, 1], "(b)")
    add_panel_label(axes[1, 0], "(c)")
    add_panel_label(axes[1, 1], "(d)")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=4,
        frameon=True,
        columnspacing=0.9,
        handlelength=1.5,
        fontsize=7.2,
    )
    fig.subplots_adjust(left=0.090, right=0.982, bottom=0.125, top=0.915, wspace=0.015, hspace=0.440)

    stem = OUTPUT_DIR / "fig_combined_idle_dqi_temporal_backbone_trends_square"
    for ext in ("pdf", "png", "jpg"):
        fig.savefig(stem.with_suffix(f".{ext}"), dpi=300, bbox_inches="tight", pad_inches=0.035)
    plt.close(fig)
    print(stem.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
