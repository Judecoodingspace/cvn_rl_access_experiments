"""Analyze the preregistered ESN head/reservoir structure diagnostic."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

AGENTS = ["ddqn_mlp", "ddqn_linear", "esn_ddqn_linear", "esn_ddqn_mlp"]
FROZEN = "theta_central_frozen"
PLACEBO = "theta_central_frozen_placebo"
MODEL_SHA = "c0d645858adf54bdcd04b9c0956629a9b11968a60013ff7d21ec5d8381cb7732"


def bootstrap_ci(values: np.ndarray, seed: int = 20260705) -> tuple[float, float]:
    means = np.random.default_rng(seed).choice(values, (10_000, len(values)), replace=True).mean(axis=1)
    return tuple(float(x) for x in np.quantile(means, [0.025, 0.975]))


def validate_rows(final: pd.DataFrame) -> None:
    if len(final) != 24:
        raise ValueError(f"expected 24 final rows, got {len(final)}")
    if set(final.agent) != set(AGENTS) or set(final.module_variant) != {FROZEN, PLACEBO}:
        raise ValueError("agent or condition matrix mismatch")
    counts = final.groupby(["agent", "module_variant"]).seed.agg(["count", "nunique"])
    if not ((counts["count"] == 3) & (counts["nunique"] == 3)).all():
        raise ValueError("every agent-condition pair must contain three unique seeds")
    if final.duplicated(["agent", "scenario", "seed", "module_variant"]).any():
        raise ValueError("duplicate primary key")
    if not final.evaluation_parameters_frozen.astype(bool).all():
        raise ValueError("evaluation changed policy parameters")
    if set(final.prediction_model_hash) != {MODEL_SHA}:
        raise ValueError("central predictor SHA mismatch")
    for seed, group in final.groupby("seed"):
        if group.evaluation_physical_trace_hash.nunique() != 1:
            raise ValueError(f"seed {seed}: physical trace mismatch")
        if group.raw_action_mask_hash.nunique() != 1:
            raise ValueError(f"seed {seed}: action mask mismatch")


def paired_effects(final: pd.DataFrame) -> pd.DataFrame:
    index = final.set_index(["agent", "module_variant", "seed"])
    comparisons = {
        "alignment_ddqn_mlp": (("ddqn_mlp", FROZEN), ("ddqn_mlp", PLACEBO)),
        "alignment_ddqn_linear": (("ddqn_linear", FROZEN), ("ddqn_linear", PLACEBO)),
        "alignment_esn_linear": (("esn_ddqn_linear", FROZEN), ("esn_ddqn_linear", PLACEBO)),
        "alignment_esn_mlp": (("esn_ddqn_mlp", FROZEN), ("esn_ddqn_mlp", PLACEBO)),
        "reservoir_given_linear": (("esn_ddqn_linear", FROZEN), ("ddqn_linear", FROZEN)),
        "mlp_head_given_reservoir": (("esn_ddqn_mlp", FROZEN), ("esn_ddqn_linear", FROZEN)),
        "reservoir_given_mlp": (("esn_ddqn_mlp", FROZEN), ("ddqn_mlp", FROZEN)),
    }
    rows = []
    for effect, (left, right) in comparisons.items():
        for seed in (1, 2, 3):
            for metric in ("served_goodput_mbps", "plr", "pu_access_violation_rate", "mean_reward"):
                value = float(index.loc[(*left, seed), metric] - index.loc[(*right, seed), metric])
                rows.append({"effect": effect, "metric": metric, "seed": seed, "value": value,
                             "left_agent": left[0], "left_condition": left[1],
                             "right_agent": right[0], "right_condition": right[1]})
    return pd.DataFrame(rows)


def summarize_effects(effects: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (effect, metric), group in effects.groupby(["effect", "metric"], sort=False):
        values = group.value.to_numpy(float); low, high = bootstrap_ci(values)
        rows.append({"effect": effect, "metric": metric, "mean": values.mean(), "std": values.std(ddof=1),
                     "ci95_low": low, "ci95_high": high, "positive_seeds": int((values > 0).sum())})
    return pd.DataFrame(rows)


def stability_table(checkpoints: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (agent, condition, seed), group in checkpoints.groupby(["agent", "module_variant", "seed"]):
        group = group.sort_values("checkpoint_slot")
        x = group.checkpoint_slot.to_numpy(float); y = group.served_goodput_mbps.to_numpy(float)
        late = group[group.checkpoint_slot.isin([20_000, 25_000, 30_000])]
        rows.append({"agent": agent, "module_variant": condition, "seed": seed,
                     "checkpoint_auc_mbps_slots": float(np.trapezoid(y, x)),
                     "late_window_goodput_mbps": float(late.served_goodput_mbps.mean()),
                     "final_goodput_mbps": float(group.iloc[-1].served_goodput_mbps)})
    return pd.DataFrame(rows)


def classify(summary: pd.DataFrame) -> dict:
    goodput = summary[summary.metric == "served_goodput_mbps"].set_index("effect")
    head = goodput.loc["mlp_head_given_reservoir"]
    reservoir_linear = goodput.loc["reservoir_given_linear"]
    reservoir_mlp = goodput.loc["reservoir_given_mlp"]
    alignment = {
        name.removeprefix("alignment_"): {
            "mean_goodput_mbps": float(row["mean"]),
            "positive_seeds": int(row["positive_seeds"]),
        }
        for name, row in goodput.iterrows() if name.startswith("alignment_")
    }
    return {
        "diagnostic_only": True,
        "head_capacity_signal": bool(head["mean"] > 0 and head["positive_seeds"] >= 2),
        "reservoir_value_with_linear_head": bool(reservoir_linear["mean"] > 0 and reservoir_linear["positive_seeds"] >= 2),
        "reservoir_value_with_mlp_head": bool(reservoir_mlp["mean"] > 0 and reservoir_mlp["positive_seeds"] >= 2),
        "effects_mbps": {
            "mlp_head_given_reservoir": float(head["mean"]),
            "reservoir_given_linear": float(reservoir_linear["mean"]),
            "reservoir_given_mlp": float(reservoir_mlp["mean"]),
        },
        "alignment_by_structure": alignment,
        "next_stage": "run_leak_rate_diagnostic_only_if_reservoir_effect_is_negative",
    }


def plot_structure(final: pd.DataFrame, effects: pd.DataFrame, output: Path) -> None:
    order = AGENTS
    labels = ["DDQN\nMLP", "DDQN\nLinear", "ESN\nLinear", "ESN\nMLP"]
    colors = ["#2878B5", "#9AC9DB", "#D95319", "#F2A65A"]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.35), constrained_layout=True)
    frozen = final[final.module_variant == FROZEN]
    placebo = final[final.module_variant == PLACEBO]
    for x, (agent, color) in enumerate(zip(order, colors)):
        fv = frozen[frozen.agent == agent].sort_values("seed").served_goodput_mbps.to_numpy()
        pv = placebo[placebo.agent == agent].sort_values("seed").served_goodput_mbps.to_numpy()
        axes[0].scatter(np.full(3, x - .08), fv, color=color, marker="o", alpha=.8)
        axes[0].scatter(np.full(3, x + .08), pv, color=color, marker="x", alpha=.8)
        axes[0].plot([x-.08, x+.08], [fv, pv], color=color, alpha=.25)
        delta = fv - pv; low, high = bootstrap_ci(delta)
        axes[1].scatter(np.full(3, x), delta, color=color, alpha=.8)
        axes[1].errorbar(x, delta.mean(), yerr=[[delta.mean()-low], [high-delta.mean()]], fmt="D", color="black", capsize=3)
    axes[0].set_title("A  Frozen and placebo endpoints"); axes[0].set_ylabel("Goodput (Mbps)")
    axes[1].set_title("B  Frozen - placebo"); axes[1].set_ylabel("Paired goodput effect (Mbps)"); axes[1].axhline(0, color=".4", ls="--")
    counts = final.groupby("agent")[["trainable_parameter_count", "fixed_reservoir_parameter_count"]].first().reindex(order)
    axes[2].bar(range(4), counts.trainable_parameter_count, color=colors, label="Trainable")
    axes[2].bar(range(4), counts.fixed_reservoir_parameter_count, bottom=counts.trainable_parameter_count, color=".75", label="Fixed reservoir")
    axes[2].set_yscale("log"); axes[2].set_title("C  Stored parameters"); axes[2].set_ylabel("Parameter count (log scale)"); axes[2].legend(fontsize=8)
    for axis in axes:
        axis.set_xticks(range(4), labels); axis.grid(axis="y", alpha=.22)
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".jpg"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    out = args.output_dir or args.run_dir / "analysis" / "esn_structure_diagnostic"
    out.mkdir(parents=True, exist_ok=True)
    final = pd.read_csv(args.run_dir / "tables/final_checkpoint_results.csv")
    final = final[final.is_final_checkpoint.astype(bool)].copy()
    checkpoints = pd.read_csv(args.run_dir / "metrics/e2_p3_prediction_checkpoint_metrics.csv")
    validate_rows(final)
    effects = paired_effects(final); summary = summarize_effects(effects); stability = stability_table(checkpoints)
    final.to_csv(out / "esn_structure_final_rows.csv", index=False)
    effects.to_csv(out / "esn_structure_paired_effects.csv", index=False)
    summary.to_csv(out / "esn_structure_effect_summary.csv", index=False)
    stability.to_csv(out / "esn_structure_checkpoint_stability.csv", index=False)
    gate = classify(summary)
    gate["structural_audit_passed"] = True
    gate["model_sha256"] = MODEL_SHA
    (out / "esn_structure_diagnostic_gate.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")
    plot_structure(final, effects, out / "esn_structure_decomposition")
    print(json.dumps(gate, indent=2))


if __name__ == "__main__":
    main()
