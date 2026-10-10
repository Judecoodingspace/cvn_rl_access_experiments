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

COLOR_POOL = [
    PALETTE["blue"],
    PALETTE["orange"],
    PALETTE["purple"],
    PALETTE["gray"],
    PALETTE["green"],
    PALETTE["red"],
    PALETTE["brown"],
    PALETTE["black"],
]
MARKER_POOL = ["o", "s", "D", "^", "v", "p", "h", "x"]

# Friendly label rules: strip "theta_" prefix, replace underscores
_RENAME = {
    "theta_oracle": "Oracle",
    "theta_placebo_a": "Placebo-A",
    "theta_placebo_b": "Placebo-B",
    "theta_neutral": "Neutral",
    "theta_time_shuffled": "Shuffled",
    "theta_shuffled_p25": "25% shuffled",
    "theta_shuffled_p50": "50% shuffled",
    "theta_shuffled_p75": "75% shuffled",
    "theta_shuffled_p100": "100% shuffled",
    "theta_soft_oracle": "Soft Oracle",
    "theta_soft_placebo_a": "Soft Placebo-A",
    "theta_soft_placebo_b": "Soft Placebo-B",
    "theta_continuous_oracle": "Cont. Oracle",
    "theta_continuous_placebo_a": "Cont. Placebo-A",
    "theta_continuous_placebo_b": "Cont. Placebo-B",
    "theta_stale_s100": "Stale 100",
    "theta_stale_s500": "Stale 500",
    "theta_stale_s1000": "Stale 1000",
    "theta_stale_s5000": "Stale 5000",
    "theta_causal": "Causal",
    "theta_causal_placebo_a": "Causal Placebo-A",
    "theta_causal_placebo_b": "Causal Placebo-B",
    "theta_central_markov": "Central Markov",
    "theta_central_frozen": "Central Frozen",
    "theta_central_frozen_placebo": "Frozen Placebo",
}


def _detect_conditions(metrics: pd.DataFrame) -> dict:
    """Auto-detect module variants and assign colors/markers/labels."""
    variants = sorted(set(metrics["module_variant"].astype(str)))
    labels: dict[str, str] = {}
    colors: dict[str, str] = {}
    markers: dict[str, str] = {}
    for i, variant in enumerate(variants):
        labels[variant] = _RENAME.get(variant, variant.replace("theta_", "").replace("_", " "))
        colors[variant] = COLOR_POOL[i % len(COLOR_POOL)]
        markers[variant] = MARKER_POOL[i % len(MARKER_POOL)]
    return {
        "conditions": tuple(variants),
        "labels": labels,
        "colors": colors,
        "markers": markers,
    }


def write_prediction_positive_control_r2_figures(
    run_dir: Path,
    metrics: pd.DataFrame,
) -> list[str]:
    if metrics.empty or "module_variant" not in metrics:
        return []
    meta = _detect_conditions(metrics)
    conditions = meta["conditions"]
    if len(conditions) < 2:
        return []
    apply_ieee_strict_style()
    out_dir = run_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    final = _final_rows(metrics)
    written: list[str] = []
    written.extend(
        _plot_condition_checkpoint_curves(
            metrics, out_dir, "mean_reward", "Mean evaluation reward",
            "reward_checkpoint_convergence", meta,
        )
    )
    written.extend(
        _plot_condition_checkpoint_curves(
            metrics, out_dir, "system_vt_throughput", "System VT throughput (Mbps)",
            "vt_throughput_checkpoint_convergence", meta,
        )
    )
    written.extend(_plot_safety_throughput(final, out_dir, meta))
    written.extend(_plot_final_diagnostics(final, out_dir, meta))
    written.extend(_plot_underlay_ratio(final, out_dir, meta))
    written.extend(_plot_loss_decomposition(final, out_dir, meta))
    written.extend(_plot_interference_ccdf(final, out_dir, meta))
    written.extend(_plot_condition_heatmap(final, out_dir, meta))
    written.extend(_plot_gate_outcomes(final, out_dir, meta))
    written.extend(_plot_action_response(final, out_dir, meta))
    plt.close("all")
    return written


def _plot_condition_checkpoint_curves(
    metrics: pd.DataFrame,
    out_dir: Path,
    metric: str,
    ylabel: str,
    stem: str,
    meta: dict,
) -> list[str]:
    if metric not in metrics:
        return []
    conditions = meta["conditions"]
    colors = meta["colors"]
    markers = meta["markers"]
    labels = meta["labels"]
    fig, ax = plt.subplots(figsize=FIG_IEEE_DOUBLE)
    for condition in conditions:
        values = metrics[metrics["module_variant"].astype(str) == condition]
        grouped = _aggregate(values, ["checkpoint_slot"], metric)
        grouped = grouped.sort_values("checkpoint_slot")
        ax.plot(
            grouped["checkpoint_slot"],
            grouped["mean"],
            color=colors[condition],
            marker=markers[condition],
            markersize=3.2,
            linewidth=1.0,
            label=labels[condition],
        )
        ax.errorbar(
            grouped["checkpoint_slot"],
            grouped["mean"],
            yerr=grouped["ci95"],
            fmt="none",
            ecolor=colors[condition],
            elinewidth=0.6,
            capsize=1.5,
        )
    final_slot = int(pd.to_numeric(metrics["checkpoint_slot"]).max())
    ax.axvline(final_slot, color=PALETTE["black"], linestyle=":", linewidth=0.7)
    ax.text(
        final_slot, 0.02, "Final long evaluation",
        rotation=90, transform=ax.get_xaxis_transform(),
        ha="right", va="bottom", fontsize=6.2,
    )
    ax.set_xlabel("Training slots")
    ax.set_ylabel(ylabel)
    ncol = min(len(conditions), 4)
    ax.legend(ncol=ncol, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    fig.subplots_adjust(left=0.10, right=0.99, bottom=0.17, top=0.78)
    return _save_all(fig, out_dir, stem)


def _plot_safety_throughput(
    final: pd.DataFrame, out_dir: Path, meta: dict,
) -> list[str]:
    conditions = meta["conditions"]
    colors = meta["colors"]
    markers = meta["markers"]
    labels = meta["labels"]
    fig, ax = plt.subplots(figsize=FIG_IEEE_DOUBLE)
    for condition in conditions:
        values = final[final["module_variant"].astype(str) == condition]
        x = pd.to_numeric(values["pu_access_violation_rate"])
        y = pd.to_numeric(values["served_goodput_mbps"])
        ax.scatter(x, y, color=colors[condition], marker=markers[condition],
                   s=14, alpha=0.45)
        ax.errorbar(
            x.mean(), y.mean(),
            xerr=_ci95(x), yerr=_ci95(y),
            color=colors[condition], marker=markers[condition],
            markerfacecolor=colors[condition], markersize=5.0,
            linestyle="none", capsize=1.8, label=labels[condition],
        )
    ax.set_xlabel("PU access violation rate")
    ax.set_ylabel("Served goodput (Mbps)")
    ncol = min(len(conditions), 4)
    ax.legend(ncol=ncol, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    fig.subplots_adjust(left=0.11, right=0.99, bottom=0.17, top=0.78)
    return _save_all(fig, out_dir, "safety_throughput_pareto")


def _plot_final_diagnostics(
    final: pd.DataFrame, out_dir: Path, meta: dict,
) -> list[str]:
    specs = [
        ("plr", "Packet loss rate"),
        ("completion_rate", "Completion rate"),
        ("mean_queue_occupancy", "Mean queue occupancy"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=FIG_IEEE_DOUBLE)
    for ax, (metric, ylabel) in zip(axes, specs):
        _condition_bars(ax, final, metric, ylabel, meta)
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.30, top=0.94, wspace=0.36)
    return _save_all(fig, out_dir, "performance_vs_load")


def _plot_underlay_ratio(
    final: pd.DataFrame, out_dir: Path, meta: dict,
) -> list[str]:
    fig, ax = plt.subplots(figsize=(3.45, 2.55))
    _condition_bars(ax, final, "underlay_selection_ratio", "Underlay selection ratio", meta)
    ax.set_ylim(-0.02, 1.02)
    fig.subplots_adjust(left=0.18, right=0.98, bottom=0.30, top=0.96)
    return _save_all(fig, out_dir, "underlay_ratio")


def _plot_loss_decomposition(
    final: pd.DataFrame, out_dir: Path, meta: dict,
) -> list[str]:
    conditions = meta["conditions"]
    labels = meta["labels"]
    components = [
        ("queue_overflow_drop_rate", "Queue overflow", PALETTE["red"]),
        ("contention_loss_rate", "Contention loss", PALETTE["orange"]),
        ("pu_interruption_rate", "PU interruption", PALETTE["purple"]),
        ("low_rate_backlog_rate", "Low-rate backlog", PALETTE["gray"]),
    ]
    x = np.arange(len(conditions))
    bottom = np.zeros(len(conditions), dtype=float)
    fig, ax = plt.subplots(figsize=FIG_IEEE_DOUBLE)
    for metric, label, color in components:
        values = np.asarray(
            [
                pd.to_numeric(
                    final[final["module_variant"].astype(str) == condition][metric]
                ).mean()
                for condition in conditions
            ],
            dtype=float,
        )
        ax.bar(x, values, bottom=bottom, color=color, label=label, width=0.68)
        bottom += values
    ax.set_xticks(x)
    ax.set_xticklabels([labels[c] for c in conditions])
    ax.set_ylabel("Event rate")
    ncol = min(len(components), 4)
    ax.legend(ncol=ncol, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    fig.subplots_adjust(left=0.10, right=0.99, bottom=0.17, top=0.78)
    return _save_all(fig, out_dir, "plr_decomposition")


def _plot_interference_ccdf(
    final: pd.DataFrame, out_dir: Path, meta: dict,
) -> list[str]:
    conditions = meta["conditions"]
    colors = meta["colors"]
    markers = meta["markers"]
    labels = meta["labels"]
    thresholds = [
        (0.25, "interference_ccdf_gt_0p25"),
        (0.5, "interference_ccdf_gt_0p5"),
        (0.75, "interference_ccdf_gt_0p75"),
        (1.0, "interference_ccdf_gt_1p0"),
        (1.25, "interference_ccdf_gt_1p25"),
        (1.5, "interference_ccdf_gt_1p5"),
        (2.0, "interference_ccdf_gt_2p0"),
    ]
    fig, ax = plt.subplots(figsize=FIG_IEEE_DOUBLE)
    for condition in conditions:
        values = final[final["module_variant"].astype(str) == condition]
        y = [
            max(float(pd.to_numeric(values[column]).mean()), 1e-5)
            for _, column in thresholds
        ]
        ax.plot(
            [th for th, _ in thresholds], y,
            color=colors[condition], marker=markers[condition],
            markerfacecolor=colors[condition], markersize=3.2, linewidth=1.0,
            label=labels[condition],
        )
    ax.axvline(1.0, color=PALETTE["black"], linewidth=0.7, linestyle=":")
    ax.set_yscale("log")
    ax.set_ylim(1e-5, 1.05)
    ax.set_xlabel(r"Normalized interference $I/Psi$")
    ax.set_ylabel("Exceedance probability")
    ncol = min(len(conditions), 4)
    ax.legend(ncol=ncol, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    fig.subplots_adjust(left=0.11, right=0.99, bottom=0.17, top=0.78)
    return _save_all(fig, out_dir, "interference_ccdf")


def _plot_condition_heatmap(
    final: pd.DataFrame, out_dir: Path, meta: dict,
) -> list[str]:
    conditions = meta["conditions"]
    labels = meta["labels"]
    specs = [
        ("served_goodput_mbps", "Goodput"),
        ("pu_access_violation_rate", "PU violation"),
        ("overlay_given_busy_legal_backlog", "Busy-overlay"),
        ("mean_safe_action_regret_mbps", "Safe regret"),
    ]
    raw = np.asarray(
        [
            [
                pd.to_numeric(
                    final[final["module_variant"].astype(str) == condition][metric]
                ).mean()
                for condition in conditions
            ]
            for metric, _ in specs
        ],
        dtype=float,
    )
    normalized = np.zeros_like(raw)
    for row in range(raw.shape[0]):
        low = float(np.nanmin(raw[row]))
        high = float(np.nanmax(raw[row]))
        if high > low:
            normalized[row] = (raw[row] - low) / (high - low)
    fig, ax = plt.subplots(figsize=FIG_IEEE_DOUBLE)
    image = ax.imshow(normalized, cmap="viridis", aspect="auto", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(conditions)))
    ax.set_xticklabels([labels[c] for c in conditions])
    ax.set_yticks(np.arange(len(specs)))
    ax.set_yticklabels([label for _, label in specs])
    for row in range(raw.shape[0]):
        for col in range(raw.shape[1]):
            ax.text(
                col, row, f"{raw[row, col]:.3f}",
                ha="center", va="center",
                color="white" if normalized[row, col] < 0.45 else "black",
                fontsize=6.5,
            )
    colorbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.025)
    colorbar.set_label("Row-normalized value")
    fig.subplots_adjust(left=0.18, right=0.91, bottom=0.18, top=0.96)
    return _save_all(fig, out_dir, "module_gain_heatmap")


def _plot_gate_outcomes(
    final: pd.DataFrame, out_dir: Path, meta: dict,
) -> list[str]:
    specs = [
        ("served_goodput_mbps", "Served goodput (Mbps)"),
        ("pu_access_violation_rate", "PU access violation rate"),
        ("overlay_given_busy_legal_backlog", "Busy-overlay probability"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=FIG_IEEE_DOUBLE)
    for ax, (metric, ylabel) in zip(axes, specs):
        _condition_bars(ax, final, metric, ylabel, meta)
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.30, top=0.94, wspace=0.36)
    return _save_all(fig, out_dir, "module_performance_vs_load")


def _plot_action_response(
    final: pd.DataFrame, out_dir: Path, meta: dict,
) -> list[str]:
    specs = [
        ("average_selected_theta", "Selected idle probability"),
        ("access_selection_ratio", "Access selection ratio"),
        ("underlay_selection_ratio", "Underlay ratio"),
        ("overlay_given_idle_legal_backlog", "Idle-overlay probability"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=FIG_IEEE_DOUBLE)
    for ax, (metric, ylabel) in zip(axes, specs):
        _condition_bars(ax, final, metric, ylabel, meta)
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.34, top=0.94, wspace=0.46)
    return _save_all(fig, out_dir, "module_action_response")


def _condition_bars(
    ax: plt.Axes,
    data: pd.DataFrame,
    metric: str,
    ylabel: str,
    meta: dict,
) -> None:
    conditions = meta["conditions"]
    colors = meta["colors"]
    labels = meta["labels"]
    x = np.arange(len(conditions))
    means: list[float] = []
    errors: list[float] = []
    for index, condition in enumerate(conditions):
        values = pd.to_numeric(
            data[data["module_variant"].astype(str) == condition][metric],
            errors="coerce",
        ).dropna()
        means.append(float(values.mean()))
        errors.append(_ci95(values))
        jitter = np.linspace(-0.08, 0.08, max(len(values), 1))
        ax.scatter(
            np.full(len(values), index) + jitter[: len(values)],
            values,
            color=PALETTE["black"],
            s=8,
            zorder=3,
        )
    ax.bar(
        x, means, yerr=errors,
        color=[colors[c] for c in conditions],
        width=0.68, capsize=2.0, zorder=2,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(
        [labels[c] for c in conditions],
        rotation=35, ha="right", fontsize=6.2,
    )
    ax.set_ylabel(ylabel)


def _aggregate(
    data: pd.DataFrame, group_columns: list[str], metric: str,
) -> pd.DataFrame:
    grouped = (
        data.groupby(group_columns, dropna=False)[metric]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    grouped["ci95"] = [
        _ci95_from_summary(mean, std, count)
        for mean, std, count in zip(
            grouped["mean"], grouped["std"], grouped["count"],
        )
    ]
    return grouped


def _ci95(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna().to_numpy()
    if numeric.size < 2:
        return 0.0
    return _ci95_from_summary(
        float(np.mean(numeric)),
        float(np.std(numeric, ddof=1)),
        int(numeric.size),
    )


def _ci95_from_summary(mean: float, std: float, count: int) -> float:
    del mean
    if count < 2 or not math.isfinite(float(std)):
        return 0.0
    critical = float(student_t.ppf(0.975, count - 1))
    return critical * float(std) / math.sqrt(count)


def _final_rows(metrics: pd.DataFrame) -> pd.DataFrame:
    if "is_final_checkpoint" not in metrics:
        return metrics.copy()
    return metrics[metrics["is_final_checkpoint"].astype(bool)].copy()


def _save_all(fig: plt.Figure, out_dir: Path, stem: str) -> list[str]:
    output_stem = out_dir / stem
    save_ieee_figure(fig, str(output_stem), formats=("eps", "pdf"), tight=False)
    jpg = output_stem.with_suffix(".jpg")
    with plt.rc_context({"savefig.bbox": None}):
        fig.savefig(
            jpg, format="jpg", dpi=600, bbox_inches=None,
            pad_inches=0.0, pil_kwargs={"quality": 95, "subsampling": 0},
        )
    plt.close(fig)
    return [f"figures/{stem}.{extension}" for extension in ("eps", "pdf", "jpg")]
