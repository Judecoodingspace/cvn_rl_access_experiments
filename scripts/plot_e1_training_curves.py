from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cvn_rl.agents import create_agent
from cvn_rl.experiments.config import load_config, set_global_seed
from cvn_rl.paper_figure_style import apply_ieee_style
from cvn_rl.simulation.env import AccessEnv
from cvn_rl.simulation.traces import generate_trace
from cvn_rl.simulation.traffic import traffic_label

AGENT_LABELS = {
    "proposed_esn_ddqn": "ESN-DDQN",
    "ddqn_no_esn": "DDQN",
    "dqn": "DQN",
    "ppo_discrete": "PPO",
    "tabular_q_learning": "Q-learning",
    "fixed_overlay": "Fixed overlay",
    "fixed_underlay": "Fixed underlay",
}

AGENT_ORDER = [
    "proposed_esn_ddqn",
    "ddqn_no_esn",
    "dqn",
    "ppo_discrete",
    "tabular_q_learning",
    "fixed_overlay",
    "fixed_underlay",
]

LINE_STYLES = {
    "proposed_esn_ddqn": {"color": "#005BBB", "linestyle": "-", "linewidth": 1.8},
    "ddqn_no_esn": {"color": "#E07A00", "linestyle": "--", "linewidth": 1.2},
    "dqn": {"color": "#2E7D32", "linestyle": "-.", "linewidth": 1.2},
    "ppo_discrete": {"color": "#7B3294", "linestyle": ":", "linewidth": 1.4},
    "tabular_q_learning": {"color": "#666666", "linestyle": "--", "linewidth": 1.0},
    "fixed_overlay": {"color": "#D73027", "linestyle": "-", "linewidth": 1.0},
    "fixed_underlay": {"color": "#1A9850", "linestyle": "-", "linewidth": 1.0},
}

SCENARIO_LABELS = {
    "small_scarce": r"small $M<N$",
    "small_abundant": r"small $M>N$",
    "medium_scarce": r"medium $M<N$",
    "medium_abundant": r"medium $M>N$",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate E1 training-curve CSV and IEEE figures.")
    parser.add_argument("--config", default="configs/e1_baseline_formal.json")
    parser.add_argument("--run-dir", default=None, help="Output directory. Defaults to results/<timestamp>_e1_training_curves.")
    parser.add_argument("--checkpoint-slots", type=int, default=250)
    parser.add_argument("--slots", type=int, default=None, help="Override config slots for a quicker diagnostic curve run.")
    parser.add_argument("--seeds", default=None, help="Comma-separated seed override, e.g. 1,2,3.")
    parser.add_argument("--agents", default=None, help="Comma-separated agent override.")
    parser.add_argument("--device", default="cpu", choices=["cpu", "auto"], help="Device for curve generation. CPU is safer for publication plotting runs.")
    args = parser.parse_args(argv)

    os.environ["CVN_RL_DEVICE"] = args.device
    config = load_config(args.config)
    if args.slots is not None:
        config["slots"] = args.slots
    if args.seeds:
        config["seeds"] = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    if args.agents:
        config["agents"] = [x.strip() for x in args.agents.split(",") if x.strip()]

    run_dir = Path(args.run_dir) if args.run_dir else Path("results") / f"{datetime.now():%Y%m%d_%H%M%S}_e1_training_curves"
    (run_dir / "metrics").mkdir(parents=True, exist_ok=True)
    (run_dir / "figures").mkdir(parents=True, exist_ok=True)

    curve_df = collect_training_curves(config, checkpoint_slots=args.checkpoint_slots)
    csv_path = run_dir / "metrics" / "e1_training_curves.csv"
    curve_df.to_csv(csv_path, index=False)
    figure_paths = write_training_curve_figures(curve_df, run_dir / "figures")
    print(f"Wrote curves: {csv_path}")
    for path in figure_paths:
        print(f"Wrote figure: {path}")
    return 0


def collect_training_curves(config: dict[str, Any], checkpoint_slots: int) -> pd.DataFrame:
    slots = int(config.get("slots", 5000))
    rows: list[dict[str, Any]] = []
    scenarios = config.get("scenarios") or [
        {
            "name": f"M{int(config.get('num_channels', 0))}_N{int(config.get('num_vehicles', 0))}",
            "num_channels": config.get("num_channels"),
            "num_vehicles": config.get("num_vehicles"),
        }
    ]
    for scenario in scenarios:
        case_config = {**config, **{k: v for k, v in scenario.items() if k != "name"}}
        scenario_name = scenario["name"]
        for seed in config.get("seeds", [1]):
            set_global_seed(int(seed))
            trace = generate_trace(case_config, seed=int(seed), slots=slots)
            for agent_name in config.get("agents", AGENT_ORDER):
                agent = create_agent(agent_name, trace.num_channels, seed=int(seed))
                agent.set_eval_mode(False)
                agent.reset()
                env = AccessEnv(trace, case_config)
                observation, info = env.reset()
                window: list[dict[str, Any]] = []
                done = False
                step = 0
                while not done:
                    action = agent.select_action(observation, info)
                    next_observation, reward, done, step_info = env.step(action)
                    next_info = env._info()
                    transition_info = {**info, "next_action_mask": next_info.get("action_mask")}
                    agent.observe(observation, action, reward, next_observation, done, transition_info)
                    window.append(step_info)
                    step += 1
                    if step % checkpoint_slots == 0 or done:
                        rows.append(
                            summarize_window(
                                window,
                                scenario_name=scenario_name,
                                case_config=case_config,
                                agent_name=agent_name,
                                seed=int(seed),
                                checkpoint_slot=step,
                                trace_hash=trace.trace_hash(),
                            )
                        )
                        window.clear()
                    observation, info = next_observation, next_info
    return pd.DataFrame(rows)


def summarize_window(
    records: list[dict[str, Any]],
    scenario_name: str,
    case_config: dict[str, Any],
    agent_name: str,
    seed: int,
    checkpoint_slot: int,
    trace_hash: str,
) -> dict[str, Any]:
    df = pd.DataFrame(records)
    attempted = df["mode"] != "no_access"
    underlay = df["mode"] == "underlay"
    m = int(case_config.get("num_channels", 0))
    n = int(case_config.get("num_vehicles", 0))
    return {
        "scenario": scenario_name,
        "resource_regime": "M_gt_N" if m > n else "M_lt_N" if m < n else "M_eq_N",
        "num_channels": m,
        "num_vehicles": n,
        "agent": agent_name,
        "agent_label": AGENT_LABELS.get(agent_name, agent_name),
        "seed": seed,
        "checkpoint_slot": checkpoint_slot,
        "trace_hash": trace_hash,
        "traffic_family": case_config.get("traffic", {}).get("family", "fixed_markov"),
        "traffic_label": traffic_label(case_config.get("traffic", {})),
        "mean_reward": float(df["reward"].mean()),
        "vt_throughput": float(df["vt_throughput"].mean()),
        "plr": float(df.loc[attempted, "packet_loss"].mean()) if attempted.any() else 0.0,
        "pu_interference_violation_rate": float(df["pu_violation"].mean()),
        "underlay_selection_ratio": float(underlay.mean()),
    }


def write_training_curve_figures(curve_df: pd.DataFrame, out_dir: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    apply_ieee_style()
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    paths.extend(_plot_single_metric(curve_df, out_dir, "mean_reward", "Mean reward", "e1_training_reward_convergence"))
    paths.extend(_plot_single_metric(curve_df, out_dir, "vt_throughput", "VT throughput", "e1_training_vt_throughput"))
    paths.extend(_plot_plr_violation(curve_df, out_dir))
    paths.extend(_plot_single_metric(curve_df, out_dir, "underlay_selection_ratio", "Underlay selection ratio", "e1_training_underlay_ratio", ylim=(0.0, 1.05)))
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
    preferred = ["small_scarce", "small_abundant", "medium_scarce", "medium_abundant"]
    present = list(curve_df["scenario"].drop_duplicates())
    return [s for s in preferred if s in present] + [s for s in present if s not in preferred]


def _agent_order(curve_df: pd.DataFrame) -> list[str]:
    present = set(curve_df["agent"].drop_duplicates())
    return [a for a in AGENT_ORDER if a in present] + sorted(present - set(AGENT_ORDER))


def _plot_single_metric(
    curve_df: pd.DataFrame,
    out_dir: Path,
    metric: str,
    ylabel: str,
    stem: str,
    ylim: tuple[float, float] | None = None,
) -> list[Path]:
    import matplotlib.pyplot as plt

    data = _aggregate(curve_df, metric)
    scenarios = _scenario_order(curve_df)
    agents = _agent_order(curve_df)
    fig, axes = plt.subplots(2, 2, figsize=(7.16, 4.0), sharex=True)
    for ax, scenario in zip(axes.flat, scenarios):
        sdata = data[data["scenario"] == scenario]
        for agent in agents:
            adata = sdata[sdata["agent"] == agent].sort_values("checkpoint_slot")
            if adata.empty:
                continue
            style = LINE_STYLES.get(agent, {})
            ax.plot(
                adata["checkpoint_slot"],
                adata["mean"],
                label=AGENT_LABELS.get(agent, agent),
                **style,
            )
        ax.set_title(SCENARIO_LABELS.get(scenario, scenario))
        ax.grid(True, linewidth=0.3, alpha=0.35)
        if ylim is not None:
            ax.set_ylim(*ylim)
    for ax in axes[-1, :]:
        ax.set_xlabel("Training slot")
    for ax in axes[:, 0]:
        ax.set_ylabel(ylabel)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    return _save_all_formats(fig, out_dir, stem)


def _plot_plr_violation(curve_df: pd.DataFrame, out_dir: Path) -> list[Path]:
    import matplotlib.pyplot as plt

    plr = _aggregate(curve_df, "plr")
    viol = _aggregate(curve_df, "pu_interference_violation_rate")
    scenarios = _scenario_order(curve_df)
    agents = _agent_order(curve_df)
    fig, axes = plt.subplots(2, len(scenarios), figsize=(7.16, 3.6), sharex=True)
    if len(scenarios) == 1:
        axes = np.asarray(axes).reshape(2, 1)
    for col, scenario in enumerate(scenarios):
        for row, (metric_data, ylabel) in enumerate(((plr, "PLR"), (viol, "PU violation"))):
            ax = axes[row, col]
            sdata = metric_data[metric_data["scenario"] == scenario]
            for agent in agents:
                adata = sdata[sdata["agent"] == agent].sort_values("checkpoint_slot")
                if adata.empty:
                    continue
                ax.plot(
                    adata["checkpoint_slot"],
                    adata["mean"],
                    label=AGENT_LABELS.get(agent, agent),
                    **LINE_STYLES.get(agent, {}),
                )
            ax.grid(True, linewidth=0.3, alpha=0.35)
            ax.set_ylim(-0.01, min(1.0, max(0.12, float(sdata["mean"].max()) * 1.25 if not sdata.empty else 0.12)))
            if row == 0:
                ax.set_title(SCENARIO_LABELS.get(scenario, scenario))
            if col == 0:
                ax.set_ylabel(ylabel)
            if row == 1:
                ax.set_xlabel("Training slot")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.91))
    return _save_all_formats(fig, out_dir, "e1_training_plr_pu_violation")


def _save_all_formats(fig, out_dir: Path, stem: str) -> list[Path]:
    paths = []
    for ext in ("pdf", "eps", "jpg"):
        path = out_dir / f"{stem}.{ext}"
        save_kwargs = {"dpi": 300}
        if ext == "jpg":
            save_kwargs["pil_kwargs"] = {"quality": 95}
        fig.savefig(path, **save_kwargs)
        paths.append(path)
    return paths


if __name__ == "__main__":
    raise SystemExit(main())
