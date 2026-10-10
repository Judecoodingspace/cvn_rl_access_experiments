from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from cvn_rl.paper_figure_style import PALETTE, apply_ieee_strict_style, save_ieee_figure


VARIANTS = ('full', 'no_dqi', 'no_prediction', 'sensing_only')
LABELS = {
    'full': 'Full',
    'no_dqi': 'No DQI',
    'no_prediction': 'No prediction',
    'sensing_only': 'Sensing only',
}
COLORS = {
    'full': PALETTE['blue'],
    'no_dqi': PALETTE['orange'],
    'no_prediction': PALETTE['green'],
    'sensing_only': PALETTE['gray'],
}
MARKERS = {'full': 'o', 'no_dqi': 's', 'no_prediction': '^', 'sensing_only': 'D'}
SCENARIO_ORDER = ('M3_low_pu', 'M3_high_pu', 'M10_low_pu', 'M10_high_pu')


def write_e2_pilot_figures(run_dir: Path, metrics: pd.DataFrame) -> list[str]:
    required = {
        'scenario',
        'seed',
        'module_variant',
        'checkpoint_slot',
        'is_final_checkpoint',
        'mean_reward',
        'system_vt_throughput',
        'plr',
        'overlay_collision_rate',
    }
    if metrics.empty or not required.issubset(metrics.columns):
        return []
    present_variants = set(metrics['module_variant'])
    if not set(VARIANTS).issubset(present_variants):
        return []
    apply_ieee_strict_style()
    out_dir = run_dir / 'figures'
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    if metrics['checkpoint_slot'].nunique() >= 4:
        written.extend(
            _checkpoint_figure(
                metrics,
                out_dir,
                'mean_reward',
                'Mean held-out reward',
                'e2_pilot_reward_checkpoints',
            )
        )
        written.extend(
            _checkpoint_figure(
                metrics,
                out_dir,
                'system_vt_throughput',
                'System VT throughput (Mbps)',
                'e2_pilot_throughput_checkpoints',
            )
        )
        written.extend(
            _checkpoint_figure(
                metrics,
                out_dir,
                'plr',
                'Packet loss ratio',
                'e2_pilot_plr_checkpoints',
            )
        )
    if metrics['seed'].nunique() >= 2:
        written.extend(_paired_effect_figure(metrics, out_dir))
    plt.close('all')
    return written


def _checkpoint_figure(
    metrics: pd.DataFrame,
    out_dir: Path,
    metric: str,
    ylabel: str,
    stem: str,
) -> list[str]:
    scenarios = [
        scenario for scenario in SCENARIO_ORDER if scenario in set(metrics['scenario'])
    ]
    columns = min(2, len(scenarios))
    rows = int(math.ceil(len(scenarios) / max(columns, 1)))
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(7.1, 2.65 * rows),
        squeeze=False,
        sharex=True,
    )
    for index, scenario in enumerate(scenarios):
        ax = axes.flat[index]
        scenario_data = metrics[metrics['scenario'] == scenario]
        for variant in VARIANTS:
            values = scenario_data[scenario_data['module_variant'] == variant]
            grouped = (
                values.groupby('checkpoint_slot')[metric]
                .agg(['mean', 'std', 'count'])
                .reset_index()
                .sort_values('checkpoint_slot')
            )
            grouped['std'] = grouped['std'].fillna(0.0)
            critical = grouped['count'].apply(
                lambda count: float(student_t.ppf(0.975, count - 1))
                if count > 1
                else 0.0
            )
            ci = critical * grouped['std'] / np.sqrt(grouped['count'].clip(lower=1))
            ax.plot(
                grouped['checkpoint_slot'],
                grouped['mean'],
                color=COLORS[variant],
                marker=MARKERS[variant],
                markevery=max(len(grouped) // 6, 1),
                markersize=3.0,
                label=LABELS[variant],
            )
            ax.fill_between(
                grouped['checkpoint_slot'],
                grouped['mean'] - ci,
                grouped['mean'] + ci,
                color=COLORS[variant],
                alpha=0.12,
                linewidth=0.0,
            )
        ax.set_title(scenario.replace('_', ' '))
        ax.set_xlabel('Training slots')
        ax.set_ylabel(ylabel)
    for index in range(len(scenarios), rows * columns):
        axes.flat[index].set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 0.995))
    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.12, top=0.88, wspace=0.28, hspace=0.35)
    return _save_all(fig, out_dir, stem)


def _paired_effect_figure(metrics: pd.DataFrame, out_dir: Path) -> list[str]:
    data = metrics[metrics['is_final_checkpoint'].astype(bool)].copy()
    scenarios = [
        scenario for scenario in SCENARIO_ORDER if scenario in set(data['scenario'])
    ]
    specs = [
        (
            'system_vt_throughput',
            'no_dqi',
            1.0,
            'DQI throughput gain (Mbps)',
        ),
        ('plr', 'no_prediction', -1.0, 'Prediction PLR reduction'),
        (
            'overlay_collision_rate',
            'no_prediction',
            -1.0,
            'Prediction collision reduction',
        ),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.9))
    for ax, (metric, ablated, sign, ylabel) in zip(axes, specs):
        pivot = data.pivot_table(
            index=['scenario', 'seed'],
            columns='module_variant',
            values=metric,
            aggfunc='mean',
        )
        delta = sign * (pivot['full'] - pivot[ablated])
        for scenario_index, scenario in enumerate(scenarios):
            values = delta.xs(scenario, level='scenario').to_numpy(float)
            offsets = np.linspace(-0.08, 0.08, len(values))
            ax.scatter(
                scenario_index + offsets,
                values,
                color=PALETTE['gray'],
                s=14,
                alpha=0.75,
                zorder=2,
            )
            mean = float(np.mean(values))
            ci = (
                float(
                    student_t.ppf(0.975, len(values) - 1)
                    * np.std(values, ddof=1)
                    / np.sqrt(len(values))
                )
                if len(values) > 1
                else 0.0
            )
            ax.errorbar(
                scenario_index,
                mean,
                yerr=ci,
                color=PALETTE['blue'],
                marker='o',
                capsize=2.0,
                zorder=3,
            )
        ax.axhline(0.0, color=PALETTE['black'], linewidth=0.7, linestyle='--')
        ax.set_xticks(np.arange(len(scenarios)))
        ax.set_xticklabels(
            [scenario.replace('_', '\n') for scenario in scenarios],
            rotation=0,
            fontsize=6.5,
        )
        ax.set_ylabel(ylabel)
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.25, top=0.95, wspace=0.36)
    return _save_all(fig, out_dir, 'e2_pilot_paired_module_effects')


def _save_all(fig: plt.Figure, out_dir: Path, stem: str) -> list[str]:
    output_stem = out_dir / stem
    save_ieee_figure(fig, str(output_stem), formats=('eps', 'pdf'), tight=False)
    jpg = output_stem.with_suffix('.jpg')
    with plt.rc_context({'savefig.bbox': None}):
        fig.savefig(
            jpg,
            format='jpg',
            dpi=600,
            bbox_inches=None,
            pad_inches=0.0,
            pil_kwargs={'quality': 95, 'subsampling': 0},
        )
    return [f'figures/{stem}.{extension}' for extension in ('eps', 'pdf', 'jpg')]
