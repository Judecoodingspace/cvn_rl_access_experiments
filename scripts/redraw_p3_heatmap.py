"""Create IEEE-style central prediction comparison bar chart for e2_p3.

Converts the previously heatmap-based figure to a clean bar chart showing
goodput gain of each theta condition over the neutral (no-prediction) baseline.
Bar charts are the dominant presentation form in cognitive-radio / spectrum-access
literature and are far more familiar to reviewers than heatmaps for this purpose.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cvn_rl.paper_figure_style import (
    apply_ieee_style,
    PALETTE,
    save_ieee_figure,
)

# Keep this plot geometrically identical to the companion DQI panel.  Fixed
# margins prevent label-dependent tight cropping from changing the axes frame.
FIG_SIZE = (4.8, 2.89)
AXES_LAYOUT = {"left": 0.33, "right": 0.98, "bottom": 0.22, "top": 0.96}

# --- descriptive labels for paper readers ---
VARIANT_LABELS: dict[str, str] = {
    'theta_oracle':                 'Oracle\n(perfect PU state)',
    'theta_central_markov':         'Causal Markov\n(2-param formula, causal)',
    'theta_central_frozen':         'Frozen predictor\n(AttentionLSTM, 21 s)',
    'theta_central_frozen_placebo': 'Frozen placebo\n(time-shuffled, no alignment)',
    'theta_neutral':                'Neutral\n(no prediction, $\\theta \\approx 0.5$)',
}

# Display order: best gain at top, neutral at bottom
DISPLAY_ORDER = [
    'theta_oracle',
    'theta_central_frozen',
    'theta_central_markov',
    'theta_central_frozen_placebo',
    'theta_neutral',
]

# Color per variant: semantic grouping
BAR_COLORS: dict[str, str] = {
    'theta_oracle':                 PALETTE['blue'],
    'theta_central_markov':         PALETTE['orange'],
    'theta_central_frozen':         PALETTE['purple'],
    'theta_central_frozen_placebo': PALETTE['gray'],
    'theta_neutral':                '#cccccc',
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run_dir')
    ap.add_argument(
        '--output-stem',
        default=None,
        help='Output path without extension. Defaults to the run figures directory.',
    )
    args = ap.parse_args()
    run = Path(args.run_dir)
    output_stem = (
        Path(args.output_stem)
        if args.output_stem
        else run / 'figures' / 'module_gain_heatmap'
    )
    output_stem.parent.mkdir(parents=True, exist_ok=True)

    # --- load & aggregate ---
    data = pd.read_csv(run / 'tables' / 'final_checkpoint_results.csv')
    data = data[data['is_final_checkpoint'].astype(bool)]
    goodput = (
        data.groupby('module_variant')['served_goodput_mbps']
        .mean()
        .reindex(DISPLAY_ORDER)
    )
    neutral_gp = float(goodput['theta_neutral'])

    display_goodput = np.array([float(goodput[v]) for v in DISPLAY_ORDER])
    display_gain = display_goodput - neutral_gp
    display_labels = [VARIANT_LABELS[v] for v in DISPLAY_ORDER]
    display_colors = [BAR_COLORS[v] for v in DISPLAY_ORDER]

    # --- IEEE style ---
    apply_ieee_style()

    # Horizontal bar chart 鈥?better for reading long condition labels
    # Match the approximately 1.20 axes-frame aspect ratio used by each panel
    # in the final Figure 4/6 two-panel diagnostic style, without enlarging the
    # single-panel figure horizontally.
    fig, ax = plt.subplots(figsize=FIG_SIZE)

    y_positions = np.arange(len(DISPLAY_ORDER))
    bars = ax.barh(
        y_positions,
        display_gain,
        height=0.58,
        color=display_colors,
        edgecolor='white',
        linewidth=0.4,
        zorder=3,
    )

    # Reference line at zero (Neutral baseline)
    ax.axvline(x=0.0, color='#555555', linewidth=0.8, linestyle='--', zorder=2)

    # Annotate each bar below the bar body to avoid clipping at the right edge.
    x_min, x_max = ax.get_xlim()
    x_span = x_max - x_min
    zero_label_pad = 0.035 * x_span
    label_shift = 0.045 * x_span
    for idx, (gain_val, gp_val) in enumerate(zip(display_gain, display_goodput)):
        sign = '+' if gain_val >= 0 else ''
        if abs(gain_val) > 1e-9:
            x_label = gain_val / 2.0 + label_shift
            ha = 'center'
        else:
            x_label = zero_label_pad
            ha = 'left'
        ax.text(
            x_label, idx + 0.34,
            f'{sign}{gain_val:.2f} ({gp_val:.1f} Mbps)',
            va='top', ha=ha,
            fontsize=7.0,
            linespacing=0.9,
            color='#222222',
        )

    # Axes
    ax.set_yticks(y_positions)
    ax.set_yticklabels(display_labels, fontsize=7)
    ax.set_xlabel('Goodput gain over Neutral baseline (Mbps)', fontsize=7.5)
    ax.invert_yaxis()  # best at top

    # Light grid
    ax.grid(True, axis='x', alpha=0.3, linewidth=0.3, zorder=0)
    ax.set_axisbelow(True)


    fig.subplots_adjust(**AXES_LAYOUT)

    # --- save ---
    stem = str(output_stem)
    save_ieee_figure(fig, stem, formats=('pdf',), tight=False)
    fig.savefig(f'{stem}.jpg', dpi=300, bbox_inches=None, pad_inches=0.0)
    plt.close(fig)
    print(f"Wrote IEEE bar chart to {stem}")


if __name__ == '__main__':
    main()
