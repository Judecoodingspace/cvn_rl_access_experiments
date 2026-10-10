"""Create IEEE-style D3-R1 DQI prediction comparison bar chart.

Matches the visual grammar of ``scripts/redraw_p3_heatmap.py``: horizontal bar
chart, goodput gain over Neutral baseline, IEEE paper style, compact single-panel
canvas, zero reference line, and per-bar gain/absolute-goodput annotation.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cvn_rl.paper_figure_style import (
    apply_ieee_style,
    PALETTE,
    save_ieee_figure,
)

# Match the occupancy-prediction panel exactly.  Fixed margins prevent
# label-dependent tight cropping from changing the axes frame in LaTeX.
FIG_SIZE = (4.8, 2.89)
AXES_LAYOUT = {"left": 0.33, "right": 0.98, "bottom": 0.22, "top": 0.96}

VARIANT_LABELS: dict[str, str] = {
    "dqi_future_oracle": "Oracle DQI\n(future next-100 slots)",
    "dqi_attention_frozen": "Frozen predictor\n(Attention-LSTM, causal)",
    "dqi_canonical_persistence": "Persistence\n(stale canonical DQI)",
    "dqi_channel_placebo": "Channel placebo\n(shuffled identity)",
    "dqi_neutral": "Neutral\n(no DQI, DQI = 0.5)",
}

# Conceptual ordering mirrors the central-prediction figure: oracle, learned
# frozen predictor, simple causal baseline, placebo, neutral baseline.
DISPLAY_ORDER = [
    "dqi_future_oracle",
    "dqi_attention_frozen",
    "dqi_canonical_persistence",
    "dqi_channel_placebo",
    "dqi_neutral",
]

BAR_COLORS: dict[str, str] = {
    "dqi_future_oracle": PALETTE["blue"],
    "dqi_attention_frozen": PALETTE["purple"],
    "dqi_canonical_persistence": PALETTE["orange"],
    "dqi_channel_placebo": PALETTE["gray"],
    "dqi_neutral": "#cccccc",
}


def _load_final_metrics(run_dir: Path) -> pd.DataFrame:
    candidates = [
        run_dir / "metrics" / "access_metrics.csv",
        run_dir / "tables" / "final_checkpoint_results.csv",
    ]
    for path in candidates:
        if path.exists():
            data = pd.read_csv(path)
            if "is_final_checkpoint" in data.columns:
                data = data[data["is_final_checkpoint"].astype(bool)]
            return data
    raise FileNotFoundError(
        "could not find metrics/access_metrics.csv or tables/final_checkpoint_results.csv "
        f"under {run_dir}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "run_dir",
        nargs="?",
        default=(
            "results/d3_r1_attention_dqi_frozen/formal_rl/"
            "20260711_102421_multi_vehicle_module_ablation_run"
        ),
        help="D3-R1 formal run directory.",
    )
    ap.add_argument(
        "--output-stem",
        default="paper_figures/fig_d3_r1_dqi_prediction_comparison",
        help="Output path without extension.",
    )
    args = ap.parse_args()
    run = Path(args.run_dir)
    output_stem = Path(args.output_stem)
    output_stem.parent.mkdir(parents=True, exist_ok=True)

    data = _load_final_metrics(run)
    goodput = (
        data.groupby("module_variant")["served_goodput_mbps"]
        .mean()
        .reindex(DISPLAY_ORDER)
    )
    missing = [variant for variant in DISPLAY_ORDER if pd.isna(goodput.get(variant))]
    if missing:
        raise ValueError(f"missing required DQI variants: {missing}")

    neutral_gp = float(goodput["dqi_neutral"])
    display_goodput = np.array([float(goodput[v]) for v in DISPLAY_ORDER])
    display_gain = display_goodput - neutral_gp
    display_labels = [VARIANT_LABELS[v] for v in DISPLAY_ORDER]
    display_colors = [BAR_COLORS[v] for v in DISPLAY_ORDER]

    apply_ieee_style()
    fig, ax = plt.subplots(figsize=FIG_SIZE)

    y_positions = np.arange(len(DISPLAY_ORDER))
    ax.barh(
        y_positions,
        display_gain,
        height=0.58,
        color=display_colors,
        edgecolor="white",
        linewidth=0.4,
        zorder=3,
    )
    ax.axvline(x=0.0, color="#555555", linewidth=0.8, linestyle="--", zorder=2)

    x_min, x_max = ax.get_xlim()
    zero_label_pad = 0.035 * (x_max - x_min)
    for idx, (gain_val, gp_val) in enumerate(zip(display_gain, display_goodput)):
        sign = "+" if gain_val >= 0 else ""
        if abs(gain_val) > 1e-9:
            x_label = gain_val / 2.0
            ha = "center"
        else:
            x_label = zero_label_pad
            ha = "left"
        ax.text(
            x_label,
            idx + 0.34,
            f"{sign}{gain_val:.2f} ({gp_val:.1f} Mbps)",
            va="top",
            ha=ha,
            fontsize=7.0,
            linespacing=0.9,
            color="#222222",
        )

    ax.set_yticks(y_positions)
    ax.set_yticklabels(display_labels, fontsize=7)
    ax.set_xlabel("Goodput gain over Neutral baseline (Mbps)", fontsize=7.5)
    ax.invert_yaxis()
    ax.grid(True, axis="x", alpha=0.3, linewidth=0.3, zorder=0)
    ax.set_axisbelow(True)

    fig.subplots_adjust(**AXES_LAYOUT)

    save_ieee_figure(fig, str(output_stem), formats=("pdf",), tight=False)
    fig.savefig(f"{output_stem}.jpg", dpi=300, bbox_inches=None, pad_inches=0.0)
    plt.close(fig)

    summary = pd.DataFrame(
        {
            "module_variant": DISPLAY_ORDER,
            "served_goodput_mbps": display_goodput,
            "gain_over_neutral_mbps": display_gain,
        }
    )
    summary.to_csv(f"{output_stem}.csv", index=False)
    print(f"Wrote D3-R1 DQI prediction comparison -> {output_stem}.pdf")


if __name__ == "__main__":
    main()
