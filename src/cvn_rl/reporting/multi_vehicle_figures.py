from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from cvn_rl.paper_figure_style import FIG_IEEE_DOUBLE, PALETTE, apply_ieee_strict_style, save_ieee_figure

AGENT_ORDER = [
    "proposed_esn_ddqn",
    "ddqn_no_esn",
    "independent_ddqn",
    "dqn",
    "ppo_discrete",
    "tabular_q_learning",
    "fixed_overlay",
    "fixed_underlay",
]
LEARNED_AGENTS = [
    'proposed_esn_ddqn',
    'ddqn_no_esn',
    'independent_ddqn',
    'dqn',
    'ppo_discrete',
    'tabular_q_learning',
]
AGENT_LABELS = {
    "proposed_esn_ddqn": "ESN-DDQN",
    "ddqn_no_esn": "DDQN",
    'independent_ddqn': 'Independent DDQN',
    "dqn": "DQN",
    "ppo_discrete": "PPO",
    "tabular_q_learning": "Q-learning",
    "fixed_overlay": "Fixed overlay",
    "fixed_underlay": "Fixed underlay",
}
COLORS = {
    "proposed_esn_ddqn": PALETTE["blue"],
    "ddqn_no_esn": PALETTE["orange"],
    'independent_ddqn': PALETTE['brown'],
    "dqn": PALETTE["green"],
    "ppo_discrete": PALETTE["purple"],
    "tabular_q_learning": PALETTE["gray"],
    "fixed_overlay": PALETTE["red"],
    "fixed_underlay": PALETTE["black"],
}
MARKERS = {
    "proposed_esn_ddqn": "o",
    "ddqn_no_esn": "s",
    'independent_ddqn': 'h',
    "dqn": "^",
    "ppo_discrete": "D",
    "tabular_q_learning": "v",
    "fixed_overlay": "x",
    "fixed_underlay": "P",
}


def write_multi_vehicle_figures(run_dir: Path, metrics: pd.DataFrame) -> list[str]:
    if metrics.empty:
        return []
    if (
        'e2_r1_gate' in metrics
        and set(metrics['e2_r1_gate'].astype(str))
        == {'prediction_representation_bridge'}
    ):
        from cvn_rl.reporting.p1_r_figures import (
            write_prediction_representation_bridge_figures,
        )

        return write_prediction_representation_bridge_figures(
            run_dir,
            metrics,
        )
    if (
        'e2_r1_gate' in metrics
        and any(
            gate in set(metrics['e2_r1_gate'].astype(str))
            for gate in (
                'prediction_positive_control_r2',
                'prediction_representation_bridge',
                'prediction_time_alignment_degradation',
                'prediction_staleness_degradation',
                'prediction_causal_teacher',
                'prediction_frozen_predictor',
                'dqi_positive_control',
            )
        )
    ):
        from cvn_rl.reporting.p0_r2_figures import (
            write_prediction_positive_control_r2_figures,
        )

        return write_prediction_positive_control_r2_figures(
            run_dir,
            metrics,
        )
    apply_ieee_strict_style()
    out_dir = run_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    written.extend(_plot_checkpoint_curves(metrics, out_dir, "mean_reward", "Mean evaluation reward", "reward_checkpoint_convergence"))
    written.extend(_plot_checkpoint_curves(metrics, out_dir, "system_vt_throughput", "System VT throughput (Mbps)", "vt_throughput_checkpoint_convergence"))
    written.extend(_plot_safety_pareto(metrics, out_dir))
    written.extend(_plot_performance_vs_load(metrics, out_dir))
    written.extend(_plot_underlay_ratio(metrics, out_dir))
    written.extend(_plot_plr_decomposition(metrics, out_dir))
    written.extend(_plot_interference_ccdf(metrics, out_dir))
    written.extend(_plot_scale_generalization(metrics, out_dir))
    written.extend(_plot_traffic_comparison(metrics, out_dir))
    if 'module_variant' in metrics.columns and metrics['module_variant'].nunique() > 1:
        written.extend(_plot_module_gain_heatmap(metrics, out_dir))
        written.extend(_plot_module_performance(metrics, out_dir))
        written.extend(_plot_module_action_response(metrics, out_dir))
    if 'descriptor_variant' in metrics.columns and metrics['descriptor_variant'].nunique() > 1:
        written.extend(_plot_descriptor_comparison(metrics, out_dir))
        written.extend(_plot_descriptor_cost_pareto(metrics, out_dir))
        written.extend(_plot_predictor_quality_system_gain(metrics, out_dir))
    if (
        ('descriptor_variant' in metrics.columns and metrics['descriptor_variant'].nunique() > 1)
        or ('evaluation_variant' in metrics.columns and metrics['evaluation_variant'].nunique() > 1)
    ):
        written.extend(_plot_error_staleness(metrics, out_dir))
    if 'protection_variant' in metrics.columns and metrics['protection_variant'].nunique() > 1:
        written.extend(_plot_protection_tradeoff(metrics, out_dir))
    if {'robustness_family', 'robust_x', 'robust_y'}.issubset(metrics.columns):
        written.extend(_plot_robustness_heatmaps(metrics, out_dir))
    plt.close("all")
    return written


def _plot_checkpoint_curves(
    metrics: pd.DataFrame,
    out_dir: Path,
    metric: str,
    ylabel: str,
    stem: str,
) -> list[str]:
    data = metrics[metrics["agent"].isin(LEARNED_AGENTS)]
    scenarios = _main_scenarios(data)
    if not scenarios or metric not in data:
        return []
    fig, axes = plt.subplots(1, len(scenarios), figsize=FIG_IEEE_DOUBLE, squeeze=False, sharey=True)
    for col, scenario in enumerate(scenarios):
        ax = axes[0, col]
        scenario_data = data[data["scenario"] == scenario]
        for agent in AGENT_ORDER:
            agent_data = scenario_data[scenario_data["agent"] == agent]
            if agent_data.empty:
                continue
            grouped = _aggregate(agent_data, ["checkpoint_slot"], metric).sort_values("checkpoint_slot")
            stride = max(len(grouped) // 6, 1)
            error_every = np.arange(0, len(grouped), stride)
            ax.plot(
                grouped["checkpoint_slot"],
                grouped["mean"],
                color=COLORS[agent],
                linestyle="-",
                marker=MARKERS[agent],
                markevery=stride,
                markersize=3.0,
                markerfacecolor=COLORS[agent],
                markeredgecolor=COLORS[agent],
                label=AGENT_LABELS[agent],
            )
            ax.errorbar(
                grouped["checkpoint_slot"].iloc[error_every],
                grouped["mean"].iloc[error_every],
                yerr=grouped["ci95"].iloc[error_every],
                fmt="none",
                ecolor=COLORS[agent],
                elinewidth=0.6,
                capsize=1.4,
            )
        ax.set_title(_scenario_label(scenario, scenario_data))
        ax.set_xlabel("Training slots")
        if col == 0:
            ax.set_ylabel(ylabel)
    _legend(fig, axes[0, 0])
    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.18, top=0.78, wspace=0.24)
    return _save_all(fig, out_dir, stem)


def _plot_safety_pareto(metrics: pd.DataFrame, out_dir: Path) -> list[str]:
    data = _final_rows(metrics)
    scenarios = _main_scenarios(data)
    if not scenarios:
        return []
    fig, axes = plt.subplots(1, len(scenarios), figsize=FIG_IEEE_DOUBLE, squeeze=False)
    for col, scenario in enumerate(scenarios):
        ax = axes[0, col]
        scenario_data = data[data["scenario"] == scenario]
        for agent in AGENT_ORDER:
            values = scenario_data[scenario_data["agent"] == agent]
            if values.empty:
                continue
            x = values["pu_active_violation_rate"].mean()
            y = values["system_vt_throughput"].mean()
            xerr = _ci(values["pu_active_violation_rate"])
            yerr = _ci(values["system_vt_throughput"])
            ax.errorbar(
                x,
                y,
                xerr=xerr,
                yerr=yerr,
                color=COLORS[agent],
                marker=MARKERS[agent],
                markerfacecolor=COLORS[agent],
                markersize=4.0,
                linestyle="none",
                capsize=1.5,
                label=AGENT_LABELS[agent],
            )
        ax.set_title(_scenario_label(scenario, scenario_data))
        ax.set_xlabel("PU-active violation rate")
        if col == 0:
            ax.set_ylabel("System VT throughput (Mbps)")
    _legend(fig, axes[0, 0])
    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.18, top=0.78, wspace=0.26)
    return _save_all(fig, out_dir, "safety_throughput_pareto")


def _plot_performance_vs_load(metrics: pd.DataFrame, out_dir: Path) -> list[str]:
    data = _final_rows(metrics)
    specs = [
        ("plr", "PLR"),
        ("mean_vt_throughput", "Mean VT throughput (Mbps)"),
        ("pu_active_violation_rate", "PU-active violation rate"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=FIG_IEEE_DOUBLE)
    for ax, (metric, ylabel) in zip(axes, specs):
        for agent in AGENT_ORDER:
            values = data[data["agent"] == agent]
            if values.empty:
                continue
            grouped = _aggregate(values, ["load_ratio"], metric).sort_values("load_ratio")
            ax.errorbar(
                grouped["load_ratio"],
                grouped["mean"],
                yerr=grouped["ci95"],
                color=COLORS[agent],
                marker=MARKERS[agent],
                markerfacecolor=COLORS[agent],
                markersize=3.0,
                linestyle="-",
                capsize=1.4,
                label=AGENT_LABELS[agent],
            )
        ax.set_xlabel(r"Load ratio $\rho=N/M$")
        ax.set_ylabel(ylabel)
    _legend(fig, axes[0])
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.19, top=0.78, wspace=0.34)
    return _save_all(fig, out_dir, "performance_vs_load")


def _plot_underlay_ratio(metrics: pd.DataFrame, out_dir: Path) -> list[str]:
    data = _final_rows(metrics)
    fig, ax = plt.subplots(figsize=(3.45, 2.55))
    for agent in AGENT_ORDER:
        values = data[data["agent"] == agent]
        if values.empty:
            continue
        grouped = _aggregate(values, ["load_ratio"], "underlay_selection_ratio").sort_values("load_ratio")
        ax.errorbar(
            grouped["load_ratio"],
            grouped["mean"],
            yerr=grouped["ci95"],
            color=COLORS[agent],
            marker=MARKERS[agent],
            markerfacecolor=COLORS[agent],
            markersize=3.0,
            linestyle="-",
            capsize=1.4,
            label=AGENT_LABELS[agent],
        )
    ax.set_xlabel(r"Load ratio $\rho=N/M$")
    ax.set_ylabel("Underlay selection ratio")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(ncol=2, loc="best")
    fig.subplots_adjust(left=0.18, right=0.98, bottom=0.18, top=0.96)
    return _save_all(fig, out_dir, "underlay_ratio")


def _plot_plr_decomposition(metrics: pd.DataFrame, out_dir: Path) -> list[str]:
    data = _final_rows(metrics)
    scenarios = _main_scenarios(data)
    agents = [agent for agent in AGENT_ORDER if agent in set(data["agent"])]
    if not scenarios or not agents:
        return []
    fig, axes = plt.subplots(1, len(scenarios), figsize=FIG_IEEE_DOUBLE, squeeze=False, sharey=True)
    components = [
        ("queue_overflow_drop_rate", "Queue overflow", PALETTE["red"]),
        ("contention_loss_rate", "Contention loss", PALETTE["orange"]),
        ("pu_interruption_rate", "PU interruption", PALETTE["purple"]),
        ("low_rate_backlog_rate", "Low-rate backlog", PALETTE["gray"]),
    ]
    for col, scenario in enumerate(scenarios):
        ax = axes[0, col]
        scenario_data = data[data["scenario"] == scenario]
        bottom = np.zeros(len(agents), dtype=float)
        for metric, label, color in components:
            values = np.asarray([
                scenario_data[scenario_data["agent"] == agent][metric].mean()
                if not scenario_data[scenario_data["agent"] == agent].empty else 0.0
                for agent in agents
            ])
            ax.bar(np.arange(len(agents)), values, bottom=bottom, color=color, label=label, width=0.72)
            bottom += values
        ax.set_title(_scenario_label(scenario, scenario_data))
        ax.set_xticks(np.arange(len(agents)))
        ax.set_xticklabels([AGENT_LABELS[agent] for agent in agents], rotation=45, ha="right", fontsize=6.5)
        if col == 0:
            ax.set_ylabel("Event rate")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0.995))
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.29, top=0.78, wspace=0.23)
    return _save_all(fig, out_dir, "plr_decomposition")


def _plot_interference_ccdf(metrics: pd.DataFrame, out_dir: Path) -> list[str]:
    data = _final_rows(metrics)
    scenarios = _main_scenarios(data)
    threshold_specs = [
        (0.25, "interference_ccdf_gt_0p25"),
        (0.5, "interference_ccdf_gt_0p5"),
        (0.75, "interference_ccdf_gt_0p75"),
        (1.0, "interference_ccdf_gt_1p0"),
        (1.25, "interference_ccdf_gt_1p25"),
        (1.5, "interference_ccdf_gt_1p5"),
        (2.0, "interference_ccdf_gt_2p0"),
    ]
    if not scenarios or threshold_specs[0][1] not in data.columns:
        return []
    fig, axes = plt.subplots(1, len(scenarios), figsize=FIG_IEEE_DOUBLE, squeeze=False, sharey=True)
    for col, scenario in enumerate(scenarios):
        ax = axes[0, col]
        scenario_data = data[data["scenario"] == scenario]
        for agent in AGENT_ORDER:
            values = scenario_data[scenario_data["agent"] == agent]
            if values.empty:
                continue
            y = [max(float(values[column].mean()), 1e-4) for _, column in threshold_specs]
            ax.plot(
                [threshold for threshold, _ in threshold_specs],
                y,
                color=COLORS[agent],
                marker=MARKERS[agent],
                markerfacecolor=COLORS[agent],
                markersize=3.0,
                linestyle="-",
                label=AGENT_LABELS[agent],
            )
        ax.axvline(1.0, color="#666666", linewidth=0.8)
        ax.set_yscale("log")
        ax.set_ylim(1e-4, 1.05)
        ax.set_title(_scenario_label(scenario, scenario_data))
        ax.set_xlabel(r"Normalized interference $I/\Psi$")
        if col == 0:
            ax.set_ylabel("Exceedance probability")
    _legend(fig, axes[0, 0])
    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.18, top=0.78, wspace=0.20)
    return _save_all(fig, out_dir, "interference_ccdf")


def _plot_scale_generalization(metrics: pd.DataFrame, out_dir: Path) -> list[str]:
    if not metrics["experiment"].astype(str).str.contains("scale_traffic").any():
        return []
    data = _final_rows(metrics)
    if "arrival_probability" in data:
        data = data[np.isclose(data["arrival_probability"], 0.5)]
    fixed = data[data["traffic_family"] == "fixed_markov"] if "traffic_family" in data else data
    if not fixed.empty and "traffic_label" in fixed:
        fixed = fixed[fixed["traffic_label"] == sorted(fixed["traffic_label"].unique())[0]]
    specs = [
        ("system_vt_throughput", "System VT throughput (Mbps)"),
        ("mean_vt_throughput", "Mean VT throughput (Mbps)"),
        ("plr", "PLR"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=FIG_IEEE_DOUBLE)
    for ax, (metric, ylabel) in zip(axes, specs):
        for agent in AGENT_ORDER:
            values = fixed[fixed["agent"] == agent].copy()
            if values.empty:
                continue
            values["system_size"] = values["num_channels"] + values["num_vehicles"]
            grouped = _aggregate(values, ["system_size"], metric).sort_values("system_size")
            ax.errorbar(
                grouped["system_size"],
                grouped["mean"],
                yerr=grouped["ci95"],
                color=COLORS[agent],
                marker=MARKERS[agent],
                markerfacecolor=COLORS[agent],
                markersize=3.0,
                linestyle="-",
                capsize=1.4,
                label=AGENT_LABELS[agent],
            )
        ax.set_xlabel(r"System size $M+N$")
        ax.set_ylabel(ylabel)
    _legend(fig, axes[0])
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.19, top=0.78, wspace=0.34)
    return _save_all(fig, out_dir, "scale_generalization")


def _plot_traffic_comparison(metrics: pd.DataFrame, out_dir: Path) -> list[str]:
    data = _final_rows(metrics)
    label_column = (
        'evaluation_traffic_label'
        if 'evaluation_traffic_label' in data
        and data['evaluation_traffic_label'].nunique() > 1
        else 'traffic_label'
    )
    if label_column not in data or data[label_column].nunique() < 2:
        return []
    if "arrival_probability" in data:
        data = data[np.isclose(data["arrival_probability"], 0.5)]
    preferred_scenario = "scale_medium_rho2"
    if preferred_scenario in set(data["scenario"]):
        data = data[data["scenario"] == preferred_scenario]
    else:
        data = data[data["scenario"] == data["scenario"].iloc[0]]
    labels = list(data[label_column].drop_duplicates())
    specs = [("system_vt_throughput", "System VT throughput (Mbps)"), ("plr", "PLR")]
    fig, axes = plt.subplots(1, 2, figsize=FIG_IEEE_DOUBLE)
    for ax, (metric, ylabel) in zip(axes, specs):
        for agent in ["proposed_esn_ddqn", "ddqn_no_esn", "dqn", "ppo_discrete"]:
            values = data[data["agent"] == agent]
            if values.empty:
                continue
            means = []
            errors = []
            for label in labels:
                series = values[values[label_column] == label][metric]
                means.append(float(series.mean()))
                errors.append(_ci(series))
            ax.errorbar(
                np.arange(len(labels)),
                means,
                yerr=errors,
                color=COLORS[agent],
                marker=MARKERS[agent],
                markerfacecolor=COLORS[agent],
                markersize=3.0,
                linestyle="-",
                capsize=1.4,
                label=AGENT_LABELS[agent],
            )
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=6.2)
        ax.set_ylabel(ylabel)
    _legend(fig, axes[0])
    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.34, top=0.78, wspace=0.28)
    return _save_all(fig, out_dir, "traffic_family_comparison")

def _plot_error_staleness(metrics: pd.DataFrame, out_dir: Path) -> list[str]:
    data = _final_rows(metrics)
    primary_agent = _primary_module_agent(data)
    if primary_agent is None:
        return []
    data = data[data['agent'] == primary_agent]
    variant_column = 'descriptor_variant'
    if (
        data['descriptor_variant'].nunique() <= 1
        and 'evaluation_variant' in data
        and data['evaluation_variant'].nunique() > 1
    ):
        variant_column = 'evaluation_variant'
    names = set(data[variant_column].astype(str))
    if not any('prediction_p' in name or 'dqi_update_' in name for name in names):
        return []
    fig, axes = plt.subplots(1, 3, figsize=FIG_IEEE_DOUBLE)
    for ax, prefix, metric, xlabel, ylabel in [
        (axes[0], 'prediction_pfa_', 'system_vt_throughput', 'False-alarm rate', 'System VT throughput (Mbps)'),
        (axes[1], 'prediction_pmd_', 'pu_active_violation_rate', 'Missed-detection rate', 'PU-active violation rate'),
        (axes[2], 'dqi_update_', 'system_vt_throughput', 'DQI update period (ms)', 'System VT throughput (Mbps)'),
    ]:
        points: list[tuple[float, float]] = []
        for name in names:
            if not name.startswith(prefix):
                continue
            token = name[len(prefix):].replace('ms', '')
            try:
                x_value = float(token)
            except ValueError:
                continue
            y_value = float(data[data[variant_column] == name][metric].mean())
            points.append((x_value, y_value))
        if prefix.startswith('prediction') and 'predicted_baseline' in names:
            baseline = float(data[data[variant_column] == 'predicted_baseline'][metric].mean())
            points.append((0.0, baseline))
        points.sort()
        if points:
            ax.plot(
                [item[0] for item in points], [item[1] for item in points],
                color=COLORS[primary_agent], marker=MARKERS[primary_agent],
                markerfacecolor=COLORS[primary_agent], linestyle='-', markersize=3.0,
            )
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.20, top=0.96, wspace=0.36)
    return _save_all(fig, out_dir, 'error_staleness_curves')


def _plot_descriptor_comparison(metrics: pd.DataFrame, out_dir: Path) -> list[str]:
    data = _final_rows(metrics)
    variants = list(data['descriptor_variant'].drop_duplicates())
    if len(variants) < 2:
        return []
    specs = [
        ('system_vt_throughput', 'System VT throughput (Mbps)'),
        ('plr', 'PLR'),
        ('pu_active_violation_rate', 'PU-active violation rate'),
    ]
    fig, axes = plt.subplots(1, 3, figsize=FIG_IEEE_DOUBLE)
    for ax, (metric, ylabel) in zip(axes, specs):
        for agent in [item for item in LEARNED_AGENTS if item in set(data['agent'])]:
            values = data[data['agent'] == agent]
            means = [float(values[values['descriptor_variant'] == variant][metric].mean()) for variant in variants]
            ax.plot(
                np.arange(len(variants)), means, color=COLORS[agent],
                marker=MARKERS[agent], markerfacecolor=COLORS[agent],
                linestyle='-', markersize=3.0, label=AGENT_LABELS[agent],
            )
        ax.set_xticks(np.arange(len(variants)))
        ax.set_xticklabels(variants, rotation=45, ha='right', fontsize=5.8)
        ax.set_ylabel(ylabel)
    _legend(fig, axes[0])
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.40, top=0.78, wspace=0.34)
    return _save_all(fig, out_dir, 'descriptor_performance_comparison')


def _plot_descriptor_cost_pareto(metrics: pd.DataFrame, out_dir: Path) -> list[str]:
    data = _final_rows(metrics)
    fig, ax = plt.subplots(figsize=(3.45, 2.55))
    for agent in [item for item in LEARNED_AGENTS if item in set(data['agent'])]:
        values = data[data['agent'] == agent]
        grouped = values.groupby('descriptor_variant', dropna=False).agg(
            latency=('descriptor_inference_latency_ms', 'mean'),
            throughput=('system_vt_throughput', 'mean'),
            parameters=('descriptor_parameter_count', 'mean'),
        ).reset_index()
        sizes = 18.0 + 20.0 * grouped['parameters'] / max(float(grouped['parameters'].max()), 1.0)
        ax.scatter(
            grouped['latency'], grouped['throughput'], s=sizes,
            color=COLORS[agent], marker=MARKERS[agent], label=AGENT_LABELS[agent],
        )
    ax.set_xlabel('Descriptor inference latency (ms)')
    ax.set_ylabel('System VT throughput (Mbps)')
    ax.legend(loc='best')
    fig.subplots_adjust(left=0.18, right=0.98, bottom=0.18, top=0.96)
    return _save_all(fig, out_dir, 'complexity_performance_pareto')


def _plot_predictor_quality_system_gain(metrics: pd.DataFrame, out_dir: Path) -> list[str]:
    data = _final_rows(metrics)
    dqi = data[np.isfinite(data['dqi_validation_mse'])]
    pred = data[np.isfinite(data['prediction_validation_brier'])]
    if dqi.empty and pred.empty:
        return []
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.55))
    if not dqi.empty:
        axes[0].scatter(dqi['dqi_validation_mse'], dqi['system_vt_throughput'], color=PALETTE['blue'], s=18)
    axes[0].set_xlabel('DQI validation MSE')
    axes[0].set_ylabel('System VT throughput (Mbps)')
    if not pred.empty:
        axes[1].scatter(pred['prediction_validation_brier'], pred['pu_active_violation_rate'], color=PALETTE['red'], s=18)
    axes[1].set_xlabel('Prediction Brier score')
    axes[1].set_ylabel('PU-active violation rate')
    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.20, top=0.96, wspace=0.30)
    return _save_all(fig, out_dir, 'predictor_quality_vs_system_gain')


def _plot_module_gain_heatmap(metrics: pd.DataFrame, out_dir: Path) -> list[str]:
    data = _final_rows(metrics)
    agents = [agent for agent in LEARNED_AGENTS if agent in set(data['agent'])]
    variants = ['no_dqi', 'no_prediction', 'sensing_only']
    if not agents:
        return []
    gains = np.full((len(agents), len(variants)), np.nan, dtype=float)
    for row, agent in enumerate(agents):
        agent_data = data[data['agent'] == agent]
        full = agent_data[agent_data['module_variant'] == 'full'].groupby(['scenario', 'seed'])['system_vt_throughput'].mean()
        for col, variant in enumerate(variants):
            ablated = agent_data[agent_data['module_variant'] == variant].groupby(['scenario', 'seed'])['system_vt_throughput'].mean()
            common = full.index.intersection(ablated.index)
            if len(common):
                gains[row, col] = float((full.loc[common] - ablated.loc[common]).mean())
    fig, ax = plt.subplots(figsize=(3.45, 2.55))
    image = ax.imshow(gains, cmap='RdBu_r', aspect='auto')
    ax.set_xticks(np.arange(len(variants)))
    ax.set_xticklabels(['No DQI', 'No prediction', 'Sensing only'], rotation=20, ha='right')
    ax.set_yticks(np.arange(len(agents)))
    ax.set_yticklabels([AGENT_LABELS[agent] for agent in agents])
    for row in range(len(agents)):
        for col in range(len(variants)):
            if np.isfinite(gains[row, col]):
                ax.text(col, row, f'{gains[row, col]:.2f}', ha='center', va='center', fontsize=6.5)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.05, pad=0.03)
    colorbar.set_label('Full-model throughput gain (Mbps)')
    fig.subplots_adjust(left=0.25, right=0.90, bottom=0.26, top=0.96)
    return _save_all(fig, out_dir, 'module_gain_heatmap')


def _plot_module_performance(metrics: pd.DataFrame, out_dir: Path) -> list[str]:
    data = _final_rows(metrics)
    primary_agent = _primary_module_agent(data)
    if primary_agent is None:
        return []
    variants = ['full', 'no_dqi', 'no_prediction', 'sensing_only']
    labels = {'full': 'Full', 'no_dqi': 'No DQI', 'no_prediction': 'No prediction', 'sensing_only': 'Sensing only'}
    specs = [
        ('system_vt_throughput', 'System VT throughput (Mbps)'),
        ('plr', 'PLR'),
        ('pu_active_violation_rate', 'PU-active violation rate'),
    ]
    fig, axes = plt.subplots(1, 3, figsize=FIG_IEEE_DOUBLE)
    for ax, (metric, ylabel) in zip(axes, specs):
        for index, variant in enumerate(variants):
            values = data[
                (data['agent'] == primary_agent)
                & (data['module_variant'] == variant)
            ]
            if values.empty:
                continue
            grouped = _aggregate(values, ['load_ratio'], metric).sort_values('load_ratio')
            ax.errorbar(
                grouped['load_ratio'], grouped['mean'], yerr=grouped['ci95'],
                color=list(COLORS.values())[index], marker=list(MARKERS.values())[index],
                markerfacecolor=list(COLORS.values())[index], markersize=3.0,
                linestyle='-', capsize=1.4, label=labels[variant],
            )
        ax.set_xlabel(r'Load ratio $\rho=N/M$')
        ax.set_ylabel(ylabel)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, legend_labels, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 0.995))
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.19, top=0.78, wspace=0.34)
    return _save_all(fig, out_dir, 'module_performance_vs_load')


def _plot_module_action_response(metrics: pd.DataFrame, out_dir: Path) -> list[str]:
    data = _final_rows(metrics)
    primary_agent = _primary_module_agent(data)
    if primary_agent is None:
        return []
    data = data[data['agent'] == primary_agent]
    variants = ['full', 'no_dqi', 'no_prediction', 'sensing_only']
    labels = ['Full', 'No DQI', 'No prediction', 'Sensing only']
    specs = [
        ('average_selected_dqi', 'Selected DQI'),
        ('average_selected_theta', 'Selected idle probability'),
        ('access_selection_ratio', 'Access selection ratio'),
        ('underlay_selection_ratio', 'Underlay ratio'),
    ]
    if data.empty:
        return []
    fig, axes = plt.subplots(1, 4, figsize=FIG_IEEE_DOUBLE)
    for ax, (metric, ylabel) in zip(axes, specs):
        values = [
            float(data[data['module_variant'] == variant][metric].mean())
            for variant in variants
        ]
        ax.bar(
            np.arange(len(variants)),
            values,
            color=[COLORS[agent] for agent in LEARNED_AGENTS[:4]],
        )
        ax.set_xticks(np.arange(len(variants)))
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=6.2)
        ax.set_ylabel(ylabel)
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.33, top=0.96, wspace=0.42)
    return _save_all(fig, out_dir, 'module_action_response')


def _plot_protection_tradeoff(metrics: pd.DataFrame, out_dir: Path) -> list[str]:
    data = _final_rows(metrics)
    primary_agent = _primary_module_agent(data)
    required = {
        'conditional_underlay_violation_rate',
        'system_vt_throughput',
        'interference_ratio_p99',
    }
    if primary_agent is None or not required.issubset(data.columns):
        return []
    data = data[data['agent'] == primary_agent]
    variants = ['gate_only', 'gate_plus_state', 'gate_state_mask']
    labels = {
        'gate_only': 'Gate only',
        'gate_plus_state': 'Gate + state',
        'gate_state_mask': 'Gate + state + mask',
    }
    scenarios = list(data['scenario'].drop_duplicates())
    if not scenarios:
        return []
    fig, axes = plt.subplots(
        1,
        len(scenarios),
        figsize=FIG_IEEE_DOUBLE,
        squeeze=False,
    )
    for column, scenario in enumerate(scenarios):
        ax = axes[0, column]
        values = data[data['scenario'] == scenario]
        for index, variant in enumerate(variants):
            subset = values[values['protection_variant'] == variant]
            if subset.empty:
                continue
            ax.errorbar(
                float(subset['conditional_underlay_violation_rate'].mean()),
                float(subset['system_vt_throughput'].mean()),
                xerr=_ci(subset['conditional_underlay_violation_rate']),
                yerr=_ci(subset['system_vt_throughput']),
                marker=['o', 's', '^'][index],
                color=[PALETTE['orange'], PALETTE['blue'], PALETTE['green']][index],
                linestyle='none',
                capsize=1.4,
                label=labels[variant],
            )
            ax.annotate(
                f"p99={subset['interference_ratio_p99'].mean():.2f}",
                (
                    float(subset['conditional_underlay_violation_rate'].mean()),
                    float(subset['system_vt_throughput'].mean()),
                ),
                xytext=(3, 3),
                textcoords='offset points',
                fontsize=6.0,
            )
        ax.set_title(_scenario_label(scenario, values))
        ax.set_xlabel('Conditional underlay violation')
        if column == 0:
            ax.set_ylabel('System VT throughput (Mbps)')
    _legend(fig, axes[0, 0])
    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.20, top=0.78, wspace=0.28)
    return _save_all(fig, out_dir, 'protection_safety_efficiency')


def _plot_robustness_heatmaps(metrics: pd.DataFrame, out_dir: Path) -> list[str]:
    data = _final_rows(metrics)
    primary_agent = _primary_module_agent(data)
    if primary_agent is None:
        return []
    data = data[data['agent'] == primary_agent]
    specs = [
        ('prediction', 'overlay_collision_rate', 'Prediction error', 'Overlay collision'),
        ('sensing', 'system_vt_throughput', 'Sensing error', 'VT throughput (Mbps)'),
        (
            'underlay_gain',
            'conditional_underlay_violation_rate',
            'Gain uncertainty',
            'Underlay violation',
        ),
    ]
    available = [spec for spec in specs if spec[0] in set(data['robustness_family'])]
    if not available:
        return []
    fig, axes = plt.subplots(1, len(available), figsize=FIG_IEEE_DOUBLE, squeeze=False)
    for column, (family, metric, title, colorbar_label) in enumerate(available):
        ax = axes[0, column]
        subset = data[data['robustness_family'] == family]
        pivot = subset.pivot_table(
            index='robust_y',
            columns='robust_x',
            values=metric,
            aggfunc='mean',
        ).sort_index(ascending=False)
        image = ax.imshow(pivot.to_numpy(float), aspect='auto', cmap='viridis')
        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_xticklabels([f'{value:g}' for value in pivot.columns])
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_yticklabels([f'{value:g}' for value in pivot.index])
        ax.set_xlabel(str(subset['robust_x_label'].iloc[0]))
        ax.set_ylabel(str(subset['robust_y_label'].iloc[0]))
        ax.set_title(title)
        colorbar = fig.colorbar(image, ax=ax, fraction=0.05, pad=0.03)
        colorbar.set_label(colorbar_label)
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.20, top=0.90, wspace=0.45)
    return _save_all(fig, out_dir, 'robustness_three_panel_heatmap')


def _primary_module_agent(data: pd.DataFrame) -> str | None:
    present = set(data['agent']) if 'agent' in data else set()
    for agent in ('ddqn_no_esn', 'independent_ddqn', 'proposed_esn_ddqn', 'ppo_discrete'):
        if agent in present:
            return agent
    return None


def _aggregate(data: pd.DataFrame, group_cols: list[str], metric: str) -> pd.DataFrame:
    grouped = data.groupby(group_cols, dropna=False)[metric].agg(["mean", "std", "count"]).reset_index()
    grouped["std"] = grouped["std"].fillna(0.0)
    critical = grouped['count'].apply(
        lambda count: float(student_t.ppf(0.975, count - 1)) if count > 1 else 0.0
    )
    grouped["ci95"] = critical * grouped["std"] / np.sqrt(grouped["count"].clip(lower=1))
    return grouped


def _final_rows(metrics: pd.DataFrame) -> pd.DataFrame:
    if "is_final_checkpoint" in metrics:
        return metrics[metrics["is_final_checkpoint"].astype(bool)].copy()
    maximum = metrics.groupby(["scenario", "agent", "seed"])["checkpoint_slot"].transform("max")
    return metrics[metrics["checkpoint_slot"] == maximum].copy()


def _main_scenarios(data: pd.DataFrame) -> list[str]:
    present = list(data["scenario"].drop_duplicates())
    controlled = [scenario for scenario in present if str(scenario).startswith("controlled_")]
    if controlled:
        order = ["controlled_abundant", "controlled_balanced", "controlled_scarce"]
        return [scenario for scenario in order if scenario in controlled]
    return present[:3]


def _scenario_label(scenario: str, data: pd.DataFrame) -> str:
    if data.empty:
        return scenario.replace("_", " ")
    m = int(data["num_channels"].iloc[0])
    n = int(data["num_vehicles"].iloc[0])
    return rf"$M={m},\ N={n}$"


def _ci(series: pd.Series) -> float:
    return (
        float(student_t.ppf(0.975, len(series) - 1) * series.std(ddof=1) / math.sqrt(len(series)))
        if len(series) > 1
        else 0.0
    )


def _legend(fig: plt.Figure, ax: plt.Axes) -> None:
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=len(labels),
            bbox_to_anchor=(0.5, 0.995),
            handlelength=1.6,
            handletextpad=0.35,
            columnspacing=0.65,
            borderaxespad=0.2,
        )


def _save_all(fig: plt.Figure, out_dir: Path, stem: str) -> list[str]:
    output_stem = out_dir / stem
    save_ieee_figure(fig, str(output_stem), formats=("eps", "pdf"), tight=False)
    jpg = output_stem.with_suffix(".jpg")
    with plt.rc_context({"savefig.bbox": None}):
        fig.savefig(
            jpg,
            format="jpg",
            dpi=600,
            bbox_inches=None,
            pad_inches=0.0,
            pil_kwargs={"quality": 95, "subsampling": 0},
        )
    return [f"figures/{stem}.{extension}" for extension in ("eps", "pdf", "jpg")]
