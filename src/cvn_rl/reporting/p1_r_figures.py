from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from cvn_rl.paper_figure_style import (
    FIG_IEEE_DOUBLE,
    PALETTE,
    apply_ieee_strict_style,
    save_ieee_figure,
)


REPRESENTATIONS = (
    'hard',
    'soft',
    'continuous_separable',
)
REP_LABELS = {
    'hard': 'Hard',
    'soft': 'Soft',
    'continuous_separable': 'Continuous',
}
CONDITIONS = {
    'hard': (
        'theta_oracle',
        'theta_placebo_a',
        'theta_placebo_b',
    ),
    'soft': (
        'theta_soft_oracle',
        'theta_soft_placebo_a',
        'theta_soft_placebo_b',
    ),
    'continuous_separable': (
        'theta_continuous_oracle',
        'theta_continuous_placebo_a',
        'theta_continuous_placebo_b',
    ),
}
ROLE_LABELS = ('Oracle', 'Placebo-A', 'Placebo-B')
ROLE_COLORS = (
    PALETTE['blue'],
    PALETTE['orange'],
    PALETTE['purple'],
)


def write_prediction_representation_bridge_figures(
    run_dir: Path,
    metrics: pd.DataFrame,
) -> list[str]:
    if metrics.empty:
        return []
    expected = {
        condition
        for conditions in CONDITIONS.values()
        for condition in conditions
    }
    if not expected.issubset(set(metrics['module_variant'].astype(str))):
        return []
    apply_ieee_strict_style()
    out_dir = run_dir / 'figures'
    out_dir.mkdir(parents=True, exist_ok=True)
    final = _final_rows(metrics)
    written: list[str] = []
    written.extend(_plot_gate_outcomes(final, out_dir))
    written.extend(_plot_information_effects(final, out_dir))
    written.extend(_plot_theta_audit(final, out_dir))
    written.extend(_plot_goodput_convergence(metrics, out_dir))
    plt.close('all')
    return written


def _plot_gate_outcomes(
    final: pd.DataFrame,
    out_dir: Path,
) -> list[str]:
    specs = [
        ('served_goodput_mbps', 'Served goodput (Mbps)'),
        ('pu_access_violation_rate', 'PU access violation rate'),
        ('overlay_given_busy_legal_backlog', 'Busy-overlay probability'),
    ]
    fig, axes = plt.subplots(1, 3, figsize=FIG_IEEE_DOUBLE)
    x = np.arange(len(REPRESENTATIONS))
    width = 0.23
    for ax, (metric, ylabel) in zip(axes, specs):
        for role in range(3):
            means = []
            errors = []
            for representation in REPRESENTATIONS:
                condition = CONDITIONS[representation][role]
                values = pd.to_numeric(
                    final[
                        final['module_variant'].astype(str) == condition
                    ][metric],
                    errors='coerce',
                ).dropna()
                means.append(float(values.mean()))
                errors.append(_ci95(values))
                jitter = np.linspace(-0.035, 0.035, len(values))
                ax.scatter(
                    np.full(len(values), x[len(means) - 1])
                    + (role - 1) * width
                    + jitter,
                    values,
                    color=PALETTE['black'],
                    s=6,
                    zorder=4,
                )
            ax.bar(
                x + (role - 1) * width,
                means,
                width,
                yerr=errors,
                capsize=1.6,
                color=ROLE_COLORS[role],
                label=ROLE_LABELS[role],
                zorder=2,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(
            [REP_LABELS[name] for name in REPRESENTATIONS],
            rotation=20,
            ha='right',
        )
        ax.set_ylabel(ylabel)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc='upper center',
        ncol=3,
        bbox_to_anchor=(0.5, 0.995),
    )
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.24, top=0.80, wspace=0.36)
    return _save_all(fig, out_dir, 'p1r_gate_outcomes')


def _plot_information_effects(
    final: pd.DataFrame,
    out_dir: Path,
) -> list[str]:
    goodput_gain = []
    violation_reduction = []
    for representation in REPRESENTATIONS:
        oracle_name, placebo_name, _ = CONDITIONS[representation]
        paired = _paired_values(
            final,
            oracle_name,
            placebo_name,
            [
                'served_goodput_mbps',
                'pu_access_violation_rate',
            ],
        )
        goodput_gain.append(
            float(
                np.mean(
                    (
                        paired['oracle:served_goodput_mbps']
                        - paired['placebo:served_goodput_mbps']
                    )
                    / paired['placebo:served_goodput_mbps'].abs().clip(
                        lower=1e-12
                    )
                )
            )
        )
        violation_reduction.append(
            float(
                np.mean(
                    (
                        paired['placebo:pu_access_violation_rate']
                        - paired['oracle:pu_access_violation_rate']
                    )
                    / paired['placebo:pu_access_violation_rate'].abs().clip(
                        lower=1e-12
                    )
                )
            )
        )
    x = np.arange(len(REPRESENTATIONS))
    fig, axes = plt.subplots(1, 2, figsize=FIG_IEEE_DOUBLE)
    axes[0].bar(x, goodput_gain, color=PALETTE['blue'])
    axes[0].axhline(0.05, color=PALETTE['black'], linestyle=':', linewidth=0.8)
    axes[0].set_ylabel('Oracle goodput relative gain')
    axes[1].bar(x, violation_reduction, color=PALETTE['green'])
    axes[1].axhline(0.80, color=PALETTE['black'], linestyle=':', linewidth=0.8)
    axes[1].set_ylabel('PU violation relative reduction')
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(
            [REP_LABELS[name] for name in REPRESENTATIONS],
            rotation=20,
            ha='right',
        )
    fig.subplots_adjust(left=0.10, right=0.99, bottom=0.25, top=0.96, wspace=0.30)
    return _save_all(fig, out_dir, 'p1r_information_effect_by_representation')


def _plot_theta_audit(
    final: pd.DataFrame,
    out_dir: Path,
) -> list[str]:
    oracle_names = [CONDITIONS[name][0] for name in REPRESENTATIONS]
    minimum = []
    maximum = []
    unique_fraction = []
    accuracy = []
    for condition in oracle_names:
        selected = final[
            final['module_variant'].astype(str) == condition
        ]
        minimum.append(float(pd.to_numeric(selected['evaluation_theta_min']).mean()))
        maximum.append(float(pd.to_numeric(selected['evaluation_theta_max']).mean()))
        unique_fraction.append(
            float(
                pd.to_numeric(
                    selected['evaluation_theta_unique_fraction']
                ).mean()
            )
        )
        accuracy.append(
            float(
                pd.to_numeric(
                    selected['evaluation_theta_threshold_accuracy']
                ).mean()
            )
        )
    x = np.arange(len(REPRESENTATIONS))
    fig, axes = plt.subplots(1, 2, figsize=FIG_IEEE_DOUBLE)
    axes[0].vlines(x, minimum, maximum, color=PALETTE['blue'], linewidth=5)
    axes[0].scatter(x, minimum, color=PALETTE['black'], s=10)
    axes[0].scatter(x, maximum, color=PALETTE['black'], s=10)
    axes[0].axhline(0.5, color=PALETTE['red'], linestyle=':', linewidth=0.8)
    axes[0].set_ylabel('Oracle theta value range')
    width = 0.34
    axes[1].bar(
        x - width / 2,
        unique_fraction,
        width,
        color=PALETTE['purple'],
        label='Unique fraction',
    )
    axes[1].bar(
        x + width / 2,
        accuracy,
        width,
        color=PALETTE['green'],
        label='Threshold accuracy',
    )
    axes[1].set_ylim(0.0, 1.05)
    axes[1].set_ylabel('Fraction')
    axes[1].legend(loc='lower right')
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(
            [REP_LABELS[name] for name in REPRESENTATIONS],
            rotation=20,
            ha='right',
        )
    fig.subplots_adjust(left=0.10, right=0.99, bottom=0.25, top=0.96, wspace=0.30)
    return _save_all(fig, out_dir, 'p1r_theta_representation_audit')


def _plot_goodput_convergence(
    metrics: pd.DataFrame,
    out_dir: Path,
) -> list[str]:
    fig, ax = plt.subplots(figsize=FIG_IEEE_DOUBLE)
    colors = (
        PALETTE['blue'],
        PALETTE['orange'],
        PALETTE['green'],
        PALETTE['red'],
        PALETTE['purple'],
        PALETTE['gray'],
    )
    line_index = 0
    for representation in REPRESENTATIONS:
        for role, linestyle in ((0, '-'), (1, '--')):
            condition = CONDITIONS[representation][role]
            selected = metrics[
                metrics['module_variant'].astype(str) == condition
            ]
            grouped = (
                selected.groupby('checkpoint_slot')['system_vt_throughput']
                .mean()
                .sort_index()
            )
            ax.plot(
                grouped.index,
                grouped.values,
                color=colors[line_index],
                linestyle=linestyle,
                marker='o',
                markersize=2.8,
                label=(
                    f'{REP_LABELS[representation]} '
                    f'{ROLE_LABELS[role]}'
                ),
            )
            line_index += 1
    ax.set_xlabel('Training slots')
    ax.set_ylabel('System VT throughput (Mbps)')
    ax.legend(ncol=3, loc='upper center', bbox_to_anchor=(0.5, 1.22))
    fig.subplots_adjust(left=0.10, right=0.99, bottom=0.17, top=0.74)
    return _save_all(fig, out_dir, 'p1r_goodput_checkpoint_convergence')


def _paired_values(
    frame: pd.DataFrame,
    oracle: str,
    placebo: str,
    metrics: list[str],
) -> pd.DataFrame:
    selected = frame[
        frame['module_variant'].astype(str).isin([oracle, placebo])
    ]
    pivot = selected.pivot_table(
        index=['scenario', 'seed'],
        columns='module_variant',
        values=metrics,
        aggfunc='mean',
    )
    output = pd.DataFrame(index=pivot.index)
    for metric in metrics:
        output[f'oracle:{metric}'] = pivot[(metric, oracle)]
        output[f'placebo:{metric}'] = pivot[(metric, placebo)]
    return output.reset_index()


def _ci95(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors='coerce').dropna().to_numpy()
    if numeric.size < 2:
        return 0.0
    critical = float(student_t.ppf(0.975, numeric.size - 1))
    return (
        critical
        * float(np.std(numeric, ddof=1))
        / math.sqrt(numeric.size)
    )


def _final_rows(metrics: pd.DataFrame) -> pd.DataFrame:
    if 'is_final_checkpoint' not in metrics:
        return metrics.copy()
    return metrics[metrics['is_final_checkpoint'].astype(bool)].copy()


def _save_all(fig: plt.Figure, out_dir: Path, stem: str) -> list[str]:
    output_stem = out_dir / stem
    save_ieee_figure(
        fig,
        str(output_stem),
        formats=('eps', 'pdf'),
        tight=False,
    )
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
    plt.close(fig)
    return [
        f'figures/{stem}.{extension}'
        for extension in ('eps', 'pdf', 'jpg')
    ]
