from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.transforms import Bbox

from cvn_rl.paper_figure_style import (
    apply_ieee_style,
    PALETTE,
    FIG_IEEE_DOUBLE,
    save_ieee_figure,
)


def _add_panel_tags_below(fig, axes, tags, pad_points=4.0):
    """Place (a)(b)... below the subplot label area, not inside the axes frame."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()

    for ax, tag in zip(axes, tags):
        bboxes = [ax.xaxis.label.get_window_extent(renderer=renderer)]
        tick_bboxes = [
            tick.get_window_extent(renderer=renderer)
            for tick in ax.get_xticklabels()
            if tick.get_text()
        ]
        if tick_bboxes:
            bboxes.extend(tick_bboxes)
        anchor = Bbox.union(bboxes)
        x_disp = (ax.bbox.x0 + ax.bbox.x1) / 2.0
        y_disp = anchor.y0 - pad_points * fig.dpi / 72.0
        x_fig, y_fig = inv.transform((x_disp, y_disp))
        fig.text(
            x_fig,
            y_fig,
            tag,
            ha='center',
            va='top',
            fontweight='bold',
            fontsize=8.0,
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run_dir')
    args = ap.parse_args()
    run = Path(args.run_dir)

    data = pd.read_csv(run / 'tables' / 'final_checkpoint_results.csv')
    data = data[data['is_final_checkpoint'].astype(bool)]
    summary = (
        data.groupby(
            ['evaluation_variant', 'prediction_update_slots', 'prediction_age_max_slots'],
            as_index=False,
        )
        .agg(
            served_goodput_mbps_mean=('served_goodput_mbps', 'mean'),
            served_goodput_mbps_std=('served_goodput_mbps', 'std'),
            plr_mean=('plr', 'mean'),
            pu_access_violation_rate_mean=('pu_access_violation_rate', 'mean'),
            broadcast_overhead_kbps_mean=('broadcast_overhead_kbps', 'mean'),
            seeds=('seed', 'nunique'),
        )
        .sort_values('prediction_update_slots')
    )
    summary.to_csv(run / 'tables' / 'p3b_broadcast_sensitivity.csv', index=False)

    expected_variants = {f'prediction_update_{value}' for value in (1, 10, 50, 100, 500)}
    per_seed = data.groupby('seed').agg(
        variants=('evaluation_variant', 'nunique'),
        physical_hashes=('evaluation_physical_trace_hash', 'nunique'),
        policy_hashes=('parameter_freezing_hash', 'nunique'),
        model_hashes=('prediction_model_hash', 'nunique'),
    )
    checks = {
        'three_seeds': data['seed'].nunique() == 3,
        'five_update_variants': set(data['evaluation_variant']) == expected_variants,
        'one_policy_per_seed': bool((per_seed['policy_hashes'] == 1).all()),
        'one_physical_trace_set_per_seed': bool((per_seed['physical_hashes'] == 1).all()),
        'one_model_per_seed': bool((per_seed['model_hashes'] == 1).all()),
        'evaluation_parameters_frozen': bool(
            data['evaluation_parameters_frozen'].astype(bool).all()
        ),
    }
    audit = {
        'passed': all(checks.values()),
        'checks': checks,
        'note': (
            'Policies are frozen and shared across P3-B variants; they were '
            'deterministically retrained rather than loaded from P3-A because '
            'the runner does not persist RL checkpoints.'
        ),
    }
    (run / 'analysis' / 'p3b_broadcast_audit.json').write_text(
        json.dumps(audit, indent=2), encoding='utf-8',
    )

    apply_ieee_style()

    fig, axes_grid = plt.subplots(
        2, 2,
        figsize=(5.55, 4.95),
    )
    axes = axes_grid.ravel()

    x = summary['prediction_update_slots']

    axes[0].plot(x, summary['served_goodput_mbps_mean'], marker='o', color=PALETTE['blue'])
    axes[0].set_ylabel('Goodput (Mbps)')
    axes[0].set_xlabel('Prediction update\nperiod (slots)')
    axes[0].set_xscale('log')
    axes[0].grid(True, alpha=0.35)

    axes[1].plot(x, summary['plr_mean'], marker='o', color=PALETTE['orange'])
    axes[1].set_ylabel('PLR')
    axes[1].set_xlabel('Prediction update\nperiod (slots)')
    axes[1].set_xscale('log')
    axes[1].grid(True, alpha=0.35)

    axes[2].plot(
        x,
        summary['pu_access_violation_rate_mean'],
        marker='o',
        color=PALETTE['red'],
    )
    axes[2].set_ylabel('PU access\nviolation rate')
    axes[2].set_xlabel('Prediction update\nperiod (slots)')
    axes[2].set_xscale('log')
    axes[2].grid(True, alpha=0.35)

    axes[3].plot(
        x,
        summary['broadcast_overhead_kbps_mean'],
        marker='o',
        color=PALETTE['green'],
    )
    axes[3].set_ylabel('Broadcast overhead\n(kbps)')
    axes[3].set_xlabel('Prediction update\nperiod (slots)')
    axes[3].set_xscale('log')
    axes[3].grid(True, alpha=0.35)

    for ax in axes:
        ax.tick_params(axis='x', which='both', bottom=False, top=False)
        ax.set_box_aspect(1.0)

    fig.subplots_adjust(left=0.090, right=0.982, bottom=0.125, top=0.935, wspace=0.120, hspace=0.460)
    _add_panel_tags_below(fig, axes, ('(a)', '(b)', '(c)', '(d)'), pad_points=3.2)

    paper_stem = Path('paper_figures') / 'fig2_broadcast_sensitivity_2x2'
    paper_stem.parent.mkdir(parents=True, exist_ok=True)
    for ext in ('pdf', 'png', 'jpg'):
        fig.savefig(paper_stem.with_suffix(f'.{ext}'), dpi=300, bbox_inches='tight', pad_inches=0.035)
    plt.close(fig)
    print(f"Wrote 2x2 paper figure to {paper_stem.with_suffix('.pdf')}")


if __name__ == '__main__':
    main()
