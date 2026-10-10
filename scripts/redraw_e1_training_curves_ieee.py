from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cvn_rl.paper_figure_style import (
    FIG_IEEE_COMPACT_2X2,
    FIG_IEEE_DOUBLE,
    PALETTE,
    apply_ieee_strict_style,
    save_ieee_figure,
)

AGENT_LABELS = {
    "proposed_esn_ddqn": "ESN-DDQN",
    "ddqn_no_esn": "DDQN",
    "dqn": "DQN",
    "ppo_discrete": "PPO",
    "tabular_q_learning": "Q-learning",
    "fixed_overlay": "Fixed overlay",
    "fixed_underlay": "Fixed underlay",
}

AGENT_ORDER = list(AGENT_LABELS)

LINE_STYLES = {
    "proposed_esn_ddqn": {"color": PALETTE["blue"], "linestyle": "-", "marker": "o"},
    "ddqn_no_esn": {"color": PALETTE["orange"], "linestyle": "-", "marker": "s"},
    "dqn": {"color": PALETTE["green"], "linestyle": "-", "marker": "^"},
    "ppo_discrete": {"color": PALETTE["purple"], "linestyle": "-", "marker": "D"},
    "tabular_q_learning": {"color": PALETTE["gray"], "linestyle": "-", "marker": "v"},
    "fixed_overlay": {"color": PALETTE["red"], "linestyle": "-", "marker": "x"},
    "fixed_underlay": {"color": PALETTE["black"], "linestyle": "-", "marker": "P"},
}

SCENARIO_LABELS = {
    "small_scarce": r"Small, $M<N$",
    "small_abundant": r"Small, $M>N$",
    "medium_scarce": r"Medium, $M<N$",
    "medium_abundant": r"Medium, $M>N$",
}

SCENARIO_ORDER = [
    "small_scarce",
    "small_abundant",
    "medium_scarce",
    "medium_abundant",
]

REQUIRED_COLUMNS = {
    "scenario",
    "agent",
    "checkpoint_slot",
    "mean_reward",
    "vt_throughput",
    "plr",
    "pu_interference_violation_rate",
    "underlay_selection_ratio",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Redraw E1 training curves with strict IEEE styling.")
    parser.add_argument(
        "--metrics",
        default="results/e1_training_curves_formal/metrics/e1_training_curves.csv",
    )
    parser.add_argument(
        "--out-dir",
        default="results/e1_training_curves_formal/figures/ieee_strict_redraw",
    )
    args = parser.parse_args(argv)

    metrics_path = Path(args.metrics)
    out_dir = Path(args.out_dir)
    curve_df = pd.read_csv(metrics_path)
    missing = sorted(REQUIRED_COLUMNS - set(curve_df.columns))
    if missing:
        raise ValueError(f"Missing curve columns: {', '.join(missing)}")

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = write_ieee_training_curves(curve_df, out_dir)
    print(f"Source metrics: {metrics_path}")
    for path in paths:
        print(f"Wrote figure: {path}")
    return 0


def write_ieee_training_curves(curve_df: pd.DataFrame, out_dir: Path) -> list[Path]:
    apply_ieee_strict_style()
    paths: list[Path] = []
    paths.extend(_plot_single_metric(curve_df, out_dir, "mean_reward", "Mean reward", "e1_training_reward_convergence"))
    paths.extend(_plot_single_metric(curve_df, out_dir, "vt_throughput", "VT throughput", "e1_training_vt_throughput"))
    paths.extend(_plot_plr_violation(curve_df, out_dir))
    paths.extend(
        _plot_single_metric(
            curve_df,
            out_dir,
            "underlay_selection_ratio",
            "Underlay selection ratio",
            "e1_training_underlay_ratio",
            ylim=(0.0, 1.02),
        )
    )
    plt.close("all")
    return paths


def _aggregate(curve_df: pd.DataFrame, metric: str) -> pd.DataFrame:
    grouped = (
        curve_df.groupby(["scenario", "agent", "checkpoint_slot"], dropna=False)[metric]
        .agg(["mean", "std"])
        .reset_index()
    )
    grouped["std"] = grouped["std"].fillna(0.0)
    return grouped


def _scenario_order(curve_df: pd.DataFrame) -> list[str]:
    present = list(curve_df["scenario"].drop_duplicates())
    return [name for name in SCENARIO_ORDER if name in present] + [
        name for name in present if name not in SCENARIO_ORDER
    ]


def _agent_order(curve_df: pd.DataFrame) -> list[str]:
    present = set(curve_df["agent"].drop_duplicates())
    return [name for name in AGENT_ORDER if name in present] + sorted(present - set(AGENT_ORDER))


def _plot_lines(ax: plt.Axes, data: pd.DataFrame, agents: list[str]) -> None:
    for agent in agents:
        agent_data = data[data["agent"] == agent].sort_values("checkpoint_slot")
        if agent_data.empty:
            continue
        style = LINE_STYLES.get(agent, {"color": PALETTE["black"], "linestyle": "-"})
        ax.plot(
            agent_data["checkpoint_slot"],
            agent_data["mean"],
            label=AGENT_LABELS.get(agent, agent),
            markevery=4,
            markersize=3.2,
            markeredgewidth=0.45,
            markerfacecolor=style["color"],
            markeredgecolor=style["color"],
            **style,
        )


def _add_shared_legend(fig: plt.Figure, ax: plt.Axes) -> None:
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=len(labels),
        bbox_to_anchor=(0.5, 0.995),
        handlelength=1.8,
        handletextpad=0.4,
        columnspacing=0.75,
        borderaxespad=0.2,
    )


def _plot_single_metric(
    curve_df: pd.DataFrame,
    out_dir: Path,
    metric: str,
    ylabel: str,
    stem: str,
    ylim: tuple[float, float] | None = None,
) -> list[Path]:
    data = _aggregate(curve_df, metric)
    scenarios = _scenario_order(curve_df)
    agents = _agent_order(curve_df)
    fig, axes = plt.subplots(
        2,
        2,
        figsize=FIG_IEEE_COMPACT_2X2,
        sharex=True,
        sharey=True,
        constrained_layout=False,
    )
    for ax, scenario in zip(axes.flat, scenarios):
        _plot_lines(ax, data[data["scenario"] == scenario], agents)
        ax.set_title(SCENARIO_LABELS.get(scenario, scenario))
        if ylim is not None:
            ax.set_ylim(*ylim)
    for ax in axes.flat[len(scenarios):]:
        ax.set_visible(False)
    for ax in axes[-1, :]:
        ax.set_xlabel("Training slots")
    for ax in axes[:, 0]:
        ax.set_ylabel(ylabel)
    _add_shared_legend(fig, axes.flat[0])
    fig.align_ylabels(axes[:, 0])
    fig.subplots_adjust(left=0.10, right=0.985, bottom=0.12, top=0.84, wspace=0.16, hspace=0.30)
    return _save_all_formats(fig, out_dir, stem)


def _plot_plr_violation(curve_df: pd.DataFrame, out_dir: Path) -> list[Path]:
    plr = _aggregate(curve_df, "plr")
    violation = _aggregate(curve_df, "pu_interference_violation_rate")
    scenarios = _scenario_order(curve_df)
    agents = _agent_order(curve_df)
    fig, axes = plt.subplots(
        2,
        len(scenarios),
        figsize=FIG_IEEE_DOUBLE,
        sharex=True,
        sharey="row",
        constrained_layout=False,
    )
    if len(scenarios) == 1:
        axes = np.asarray(axes).reshape(2, 1)

    row_specs = (
        (plr, "PLR"),
        (violation, "PU violation rate"),
    )
    for row, (metric_data, ylabel) in enumerate(row_specs):
        max_mean = float(metric_data["mean"].max()) if not metric_data.empty else 0.1
        row_limit = min(1.0, max(0.12, max_mean * 1.15))
        for col, scenario in enumerate(scenarios):
            ax = axes[row, col]
            _plot_lines(ax, metric_data[metric_data["scenario"] == scenario], agents)
            ax.set_ylim(-0.01, row_limit)
            if row == 0:
                ax.set_title(SCENARIO_LABELS.get(scenario, scenario))
            if col == 0:
                ax.set_ylabel(ylabel)
            if row == 1:
                ax.set_xlabel("Training slots")

    _add_shared_legend(fig, axes[0, 0])
    fig.align_ylabels(axes[:, 0])
    fig.subplots_adjust(left=0.085, right=0.99, bottom=0.16, top=0.78, wspace=0.20, hspace=0.26)
    return _save_all_formats(fig, out_dir, "e1_training_plr_pu_violation")


def _save_all_formats(fig: plt.Figure, out_dir: Path, stem: str) -> list[Path]:
    output_stem = out_dir / stem
    save_ieee_figure(fig, str(output_stem), formats=("eps", "pdf"), tight=False)
    jpg_path = output_stem.with_suffix(".jpg")
    with plt.rc_context({"savefig.bbox": None}):
        fig.savefig(
            jpg_path,
            format="jpg",
            dpi=600,
            bbox_inches=None,
            pad_inches=0.0,
            pil_kwargs={"quality": 95, "subsampling": 0},
        )
    return [output_stem.with_suffix(ext) for ext in (".eps", ".pdf", ".jpg")]


if __name__ == "__main__":
    raise SystemExit(main())
