from __future__ import annotations

from pathlib import Path

import pandas as pd


def write_expected_figures(
    run_dir: Path,
    metrics: pd.DataFrame,
    expected_outputs: list[str],
    reporting_profile: str = 'formal',
) -> list[str]:
    if reporting_profile == 'smoke':
        return []
    if reporting_profile == 'e2_pilot':
        from cvn_rl.reporting.e2_pilot_figures import write_e2_pilot_figures

        return write_e2_pilot_figures(run_dir, metrics)
    if "checkpoint_slot" in metrics.columns and "system_vt_throughput" in metrics.columns:
        from cvn_rl.reporting.multi_vehicle_figures import write_multi_vehicle_figures

        return write_multi_vehicle_figures(run_dir, metrics)
    figure_outputs = [path for path in expected_outputs if path.startswith("figures/")]
    if not figure_outputs:
        return []
    import matplotlib

    matplotlib.use("Agg")
    from cvn_rl.paper_figure_style import apply_ieee_style

    apply_ieee_style()
    import matplotlib.pyplot as plt

    written: list[str] = []
    for rel_path in figure_outputs:
        path = run_dir / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(3.5, 2.4))
        if "pu_violation" in rel_path or "violation" in rel_path or "heatmap" in rel_path:
            metric = "pu_interference_violation_rate"
        elif "convergence" in rel_path:
            metric = "convergence_slots"
        elif "staleness" in rel_path:
            metric = "broadcast_overhead_kbps"
        else:
            metric = "plr"
        _plot_metric(ax, metrics, metric)
        ax.set_title(metric.replace("_", " "))
        ax.set_xlabel("agent")
        ax.set_ylabel(metric)
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        written.append(rel_path)
    return written


def _plot_metric(ax, metrics: pd.DataFrame, metric: str) -> None:
    if "agent" in metrics.columns and metric in metrics.columns:
        group_cols = [col for col in ["scenario", "agent"] if col in metrics.columns]
        if "scenario" in group_cols:
            data = metrics.groupby(group_cols, dropna=False)[metric].mean().reset_index()
            labels = data["scenario"].astype(str) + "\n" + data["agent"].astype(str)
            values = data[metric].to_numpy()
        else:
            data = metrics.groupby("agent", dropna=False)[metric].mean().sort_values()
            labels = data.index.astype(str)
            values = data.values
        ax.bar(range(len(values)), values)
        ax.set_xticks(range(len(values)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=6)
    elif metric in metrics.columns:
        ax.plot(metrics[metric].to_numpy())
    else:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
