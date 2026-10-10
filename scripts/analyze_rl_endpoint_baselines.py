"""Create the preregistered held-out RL endpoint Table 1 and stability audit."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ORDER = [
    "proposed_esn_ddqn", "ddqn_no_esn", "dqn", "lstm_ddqn",
    "sac_discrete", "ppo_discrete", "tabular_q_learning",
    "fixed_overlay", "fixed_underlay",
]
LABELS = {
    "proposed_esn_ddqn":"ESN-DDQN", "ddqn_no_esn":"DDQN",
    "dqn":"DQN", "lstm_ddqn":"LSTM-DDQN",
    "sac_discrete":"Discrete SAC", "ppo_discrete":"PPO",
    "tabular_q_learning":"Tabular Q", "fixed_overlay":"Fixed overlay",
    "fixed_underlay":"Fixed underlay",
}
METRICS = [
    "served_goodput_mbps", "plr", "pu_access_violation_rate", "mean_reward",
    "trainable_parameter_count", "fixed_reservoir_parameter_count",
    "total_stored_parameter_count", "inference_latency_ms",
    "training_time_seconds",
]


def bootstrap(values: np.ndarray, seed: int = 20260706) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = rng.choice(values, (10_000, len(values)), replace=True).mean(axis=1)
    return tuple(float(value) for value in np.quantile(means, [.025, .975]))


def analyze(metrics: pd.DataFrame):
    final = metrics[metrics.is_final_checkpoint.astype(bool)].copy()
    missing = set(ORDER).difference(final.agent)
    if missing:
        raise ValueError(f"formal baseline is missing agents: {sorted(missing)}")
    summary = final.groupby(["scenario", "agent"])[METRICS].agg(["mean", "std"]).reset_index()
    summary.columns = [
        "_".join(part for part in column if part) if isinstance(column, tuple) else column
        for column in summary.columns
    ]
    paired_rows = []
    for scenario, group in final.groupby("scenario"):
        index = group.set_index(["agent", "seed"])
        seeds = sorted(group.seed.unique())
        for agent in ORDER:
            if agent == "ddqn_no_esn":
                continue
            for metric in METRICS[:4]:
                values = np.asarray([
                    index.loc[(agent, seed), metric]
                    - index.loc[("ddqn_no_esn", seed), metric]
                    for seed in seeds
                ], dtype=float)
                low, high = bootstrap(values)
                paired_rows.append({
                    "scenario":scenario, "agent":agent, "reference":"ddqn_no_esn",
                    "metric":metric, "mean_difference":values.mean(),
                    "ci95_low":low, "ci95_high":high,
                    "positive_seeds":int((values > 0).sum()),
                })
    stability_rows = []
    learned = metrics[metrics.agent.isin(ORDER[:7])]
    for (scenario, agent, seed), group in learned.groupby(["scenario", "agent", "seed"]):
        group = group.sort_values("checkpoint_slot")
        x = group.checkpoint_slot.to_numpy(float)
        y = group.served_goodput_mbps.to_numpy(float)
        late = group[group.checkpoint_slot.isin([20_000, 25_000, 30_000])]
        stability_rows.append({
            "scenario":scenario, "agent":agent, "seed":int(seed),
            "checkpoint_auc_mbps_slots":float(np.trapezoid(y, x)),
            "checkpoint_auc_average_mbps":float(np.trapezoid(y, x) / max(x[-1]-x[0], 1)),
            "late_window_goodput_mbps":float(late.served_goodput_mbps.mean()),
            "final_goodput_mbps":float(group.iloc[-1].served_goodput_mbps),
        })
    return final, summary, pd.DataFrame(paired_rows), pd.DataFrame(stability_rows)


def plot(final: pd.DataFrame, output: Path) -> None:
    scenarios = list(final.scenario.unique())
    fig, axes = plt.subplots(1, len(scenarios), figsize=(11, 4), squeeze=False)
    for axis, scenario in zip(axes[0], scenarios):
        group = final[final.scenario.eq(scenario)]
        for x, agent in enumerate(ORDER):
            values = group[group.agent.eq(agent)].served_goodput_mbps.to_numpy()
            axis.scatter(np.full(len(values), x), values, alpha=.65, s=16)
            axis.plot(x, values.mean(), "D", color="black", markersize=4)
        axis.set_xticks(range(len(ORDER)), [LABELS[a] for a in ORDER], rotation=45, ha="right")
        axis.set_title(scenario); axis.set_ylabel("Held-out goodput (Mbps)")
        axis.grid(axis="y", alpha=.2)
    fig.tight_layout(); fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".jpg"), dpi=300, bbox_inches="tight"); plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output-dir", type=Path); args = parser.parse_args()
    out = args.output_dir or args.run_dir / "analysis" / "rl_endpoint_baseline"
    out.mkdir(parents=True, exist_ok=True)
    metrics = pd.read_csv(args.run_dir / "metrics" / "access_metrics.csv")
    final, summary, paired, stability = analyze(metrics)
    final.to_csv(out / "rl_endpoint_raw_seed_results.csv", index=False)
    summary.to_csv(out / "rl_endpoint_table.csv", index=False)
    paired.to_csv(out / "rl_endpoint_paired_effects.csv", index=False)
    stability.to_csv(out / "rl_checkpoint_stability.csv", index=False)
    plot(final, out / "rl_endpoint_baseline")


if __name__ == "__main__":
    main()
