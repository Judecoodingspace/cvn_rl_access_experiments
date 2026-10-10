#!/usr/bin/env python3
"""Analyze D4-A joint Prediction-DQI factorial runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cvn_rl.paper_figure_style import (
    FIG_IEEE_DOUBLE,
    PALETTE,
    apply_ieee_style,
    save_ieee_figure,
)


AGENT_LABELS = {
    "proposed_esn_ddqn": "ESN-DDQN",
    "dqn": "DQN",
    "ddqn_no_esn": "DDQN",
}
AGENT_ORDER = list(AGENT_LABELS)
CONDITION_LABELS = {
    "neutral_none": "None",
    "prediction_only": "Prediction",
    "dqi_only": "DQI",
    "prediction_plus_dqi": "Prediction + DQI",
    "prediction_placebo_plus_dqi": "Prediction placebo + DQI",
    "prediction_plus_dqi_placebo": "Prediction + DQI placebo",
}
CONDITION_ORDER = list(CONDITION_LABELS)
FACTORIAL = {
    "none": "neutral_none",
    "prediction": "prediction_only",
    "dqi": "dqi_only",
    "both": "prediction_plus_dqi",
}
PAIRWISE_COMPARISONS = [
    ("prediction_only", "neutral_none", "prediction_gain_vs_none"),
    ("dqi_only", "neutral_none", "dqi_gain_vs_none"),
    ("prediction_plus_dqi", "prediction_only", "dqi_increment_given_prediction"),
    ("prediction_plus_dqi", "dqi_only", "prediction_increment_given_dqi"),
    ("prediction_plus_dqi", "neutral_none", "joint_gain_vs_none"),
    ("prediction_plus_dqi", "prediction_placebo_plus_dqi", "prediction_signal_vs_placebo_with_dqi"),
    ("prediction_plus_dqi", "prediction_plus_dqi_placebo", "dqi_signal_vs_placebo_with_prediction"),
]
EXPECTED_PREDICTION_HASH = "c0d645858adf54bdcd04b9c0956629a9b11968a60013ff7d21ec5d8381cb7732"
EXPECTED_DQI_HASH = "4623d509d776427ca970e640fdfff2d894abb9e3145b717c6bf9c7b77f2c24a4"

SUMMARY_METRICS = [
    "served_goodput_mbps",
    "plr",
    "pu_access_violation_rate",
    "mean_reward",
    "average_selected_theta",
    "underlay_selection_ratio",
    "pu_active_violation_rate",
    "conditional_underlay_violation_rate",
    "average_selected_dqi",
    "mean_selected_future_dqi",
    "mean_channel_selection_regret",
    "oracle_top1_channel_count",
    "switch_count",
    "blackout_loss_bits",
    "rolling_100ms_system_served_q10_mean",
    "mean_queue_length",
    "mean_hol_age_slots",
    "broadcast_overhead_kbps",
    "descriptor_inference_latency_ms",
    "inference_latency_ms",
    "descriptor_parameter_count",
    "parameter_count",
    "training_time_seconds",
]


def _latest_metrics_csv(run_dir: Path) -> Path:
    for name in ("tables/final_checkpoint_results.csv", "access_metrics.csv"):
        candidate = run_dir / name
        if candidate.exists():
            return candidate
    candidates = sorted(run_dir.rglob("access_metrics.csv"))
    if not candidates:
        candidates = sorted(run_dir.rglob("*metrics*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No metrics CSV found under {run_dir}")
    return candidates[-1]


def _bootstrap_ci(diff: np.ndarray, rng: np.random.Generator, n_boot: int = 20000) -> tuple[float, float]:
    if diff.size == 0:
        return np.nan, np.nan
    idx = rng.integers(0, diff.size, size=(n_boot, diff.size))
    boot = diff[idx].mean(axis=1)
    return float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def _final_rows(metrics: pd.DataFrame) -> pd.DataFrame:
    final = metrics.copy()
    if "is_final_checkpoint" in final.columns:
        final = final[final["is_final_checkpoint"].astype(bool)].copy()
    if "evaluation_variant" in final.columns:
        final = final[final["evaluation_variant"].astype(str).eq("nominal")].copy()
    return final[final["module_variant"].astype(str).isin(CONDITION_ORDER)].copy()


def _summarize(final: pd.DataFrame) -> pd.DataFrame:
    available = [m for m in SUMMARY_METRICS if m in final.columns]
    agg: dict[str, tuple[str, str]] = {"n_seeds": ("seed", "nunique")}
    for metric in available:
        agg[f"{metric}_mean"] = (metric, "mean")
        agg[f"{metric}_std"] = (metric, "std")
    summary = final.groupby(["agent", "module_variant"], as_index=False).agg(**agg)
    summary["algorithm"] = summary["agent"].map(AGENT_LABELS).fillna(summary["agent"])
    summary["condition"] = summary["module_variant"].map(CONDITION_LABELS).fillna(summary["module_variant"])
    summary["agent_order"] = summary["agent"].map({a: i for i, a in enumerate(AGENT_ORDER)}).fillna(999)
    summary["condition_order"] = summary["module_variant"].map({c: i for i, c in enumerate(CONDITION_ORDER)}).fillna(999)
    return summary.sort_values(["agent_order", "condition_order"]).drop(columns=["agent_order", "condition_order"])


def _paired_effects(final: pd.DataFrame, metric: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for agent_i, agent in enumerate(AGENT_ORDER):
        group = final[final["agent"].astype(str).eq(agent)]
        if group.empty or metric not in group.columns:
            continue
        pivot = group.pivot_table(index="seed", columns="module_variant", values=metric, aggfunc="mean")
        for comp_i, (left, right, effect) in enumerate(PAIRWISE_COMPARISONS):
            if left not in pivot or right not in pivot:
                continue
            sub = pivot[[left, right]].dropna()
            if sub.empty:
                continue
            diff = sub[left].to_numpy(float) - sub[right].to_numpy(float)
            ci_low, ci_high = _bootstrap_ci(
                diff,
                np.random.default_rng(20260713 + agent_i * 503 + comp_i),
            )
            rows.append({
                "agent": agent,
                "algorithm": AGENT_LABELS.get(agent, agent),
                "metric": metric,
                "effect": effect,
                "left_condition": left,
                "right_condition": right,
                "left_label": CONDITION_LABELS.get(left, left),
                "right_label": CONDITION_LABELS.get(right, right),
                "n_seeds": int(len(diff)),
                "left_mean": float(sub[left].mean()),
                "right_mean": float(sub[right].mean()),
                "mean_difference": float(diff.mean()),
                "ci95_low": ci_low,
                "ci95_high": ci_high,
                "positive_seeds": int((diff > 0).sum()),
            })
    return pd.DataFrame(rows)


def _interactions(final: pd.DataFrame, metric: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for agent_i, agent in enumerate(AGENT_ORDER):
        group = final[final["agent"].astype(str).eq(agent)]
        if group.empty or metric not in group.columns:
            continue
        pivot = group.pivot_table(index="seed", columns="module_variant", values=metric, aggfunc="mean")
        needed = list(FACTORIAL.values())
        if any(condition not in pivot for condition in needed):
            continue
        sub = pivot[needed].dropna()
        if sub.empty:
            continue
        none = sub[FACTORIAL["none"]].to_numpy(float)
        pred = sub[FACTORIAL["prediction"]].to_numpy(float)
        dqi = sub[FACTORIAL["dqi"]].to_numpy(float)
        both = sub[FACTORIAL["both"]].to_numpy(float)
        values = both - pred - dqi + none
        ci_low, ci_high = _bootstrap_ci(values, np.random.default_rng(20260713 + agent_i * 997))
        rows.append({
            "agent": agent,
            "algorithm": AGENT_LABELS.get(agent, agent),
            "metric": metric,
            "n_seeds": int(len(values)),
            "g_none": float(none.mean()),
            "g_prediction": float(pred.mean()),
            "g_dqi": float(dqi.mean()),
            "g_both": float(both.mean()),
            "delta_prediction": float((pred - none).mean()),
            "delta_dqi": float((dqi - none).mean()),
            "delta_both": float((both - none).mean()),
            "interaction": float(values.mean()),
            "ci95_low": ci_low,
            "ci95_high": ci_high,
            "positive_interaction_seeds": int((values > 0).sum()),
            "interpretation": (
                "synergy" if values.mean() > 0.0 and ci_low > 0.0
                else "overlap_or_conflict" if values.mean() < 0.0 and ci_high < 0.0
                else "approximately_additive"
            ),
        })
    return pd.DataFrame(rows)


def _paper_table(summary: pd.DataFrame, effects: pd.DataFrame, interactions: pd.DataFrame) -> pd.DataFrame:
    effect_map = effects[effects["metric"].eq("served_goodput_mbps")].set_index(["agent", "effect"])
    int_map = interactions[interactions["metric"].eq("served_goodput_mbps")].set_index("agent")
    rows: list[dict[str, Any]] = []
    for agent in AGENT_ORDER:
        agent_summary = summary[summary["agent"].astype(str).eq(agent)].set_index("module_variant")
        if agent_summary.empty:
            continue
        def mean(condition: str, metric: str = "served_goodput_mbps_mean") -> float:
            return float(agent_summary.loc[condition, metric]) if condition in agent_summary.index and metric in agent_summary else np.nan
        row: dict[str, Any] = {
            "agent": agent,
            "algorithm": AGENT_LABELS.get(agent, agent),
            "n_seeds": int(agent_summary["n_seeds"].max()),
            "none_goodput_mbps": mean("neutral_none"),
            "prediction_goodput_mbps": mean("prediction_only"),
            "dqi_goodput_mbps": mean("dqi_only"),
            "both_goodput_mbps": mean("prediction_plus_dqi"),
            "prediction_placebo_plus_dqi_goodput_mbps": mean("prediction_placebo_plus_dqi"),
            "prediction_plus_dqi_placebo_goodput_mbps": mean("prediction_plus_dqi_placebo"),
            "both_plr": mean("prediction_plus_dqi", "plr_mean"),
            "both_pu_access_violation_rate": mean("prediction_plus_dqi", "pu_access_violation_rate_mean"),
            "both_selected_future_dqi": mean("prediction_plus_dqi", "mean_selected_future_dqi_mean"),
            "both_channel_selection_regret": mean("prediction_plus_dqi", "mean_channel_selection_regret_mean"),
            "both_switch_count": mean("prediction_plus_dqi", "switch_count_mean"),
            "both_descriptor_parameters": mean("prediction_plus_dqi", "descriptor_parameter_count_mean"),
            "both_descriptor_latency_ms": mean("prediction_plus_dqi", "descriptor_inference_latency_ms_mean"),
            "interaction_mbps": float(int_map.loc[agent, "interaction"]) if agent in int_map.index else np.nan,
            "interaction_ci95_low": float(int_map.loc[agent, "ci95_low"]) if agent in int_map.index else np.nan,
            "interaction_ci95_high": float(int_map.loc[agent, "ci95_high"]) if agent in int_map.index else np.nan,
            "interaction_interpretation": str(int_map.loc[agent, "interpretation"]) if agent in int_map.index else "",
        }
        for effect in [
            "prediction_gain_vs_none",
            "dqi_gain_vs_none",
            "joint_gain_vs_none",
            "dqi_increment_given_prediction",
            "prediction_increment_given_dqi",
        ]:
            key = (agent, effect)
            if key in effect_map.index:
                eff = effect_map.loc[key]
                row[f"{effect}_mbps"] = float(eff["mean_difference"])
                row[f"{effect}_ci95_low"] = float(eff["ci95_low"])
                row[f"{effect}_ci95_high"] = float(eff["ci95_high"])
                row[f"{effect}_wins"] = int(eff["positive_seeds"])
        rows.append(row)
    return pd.DataFrame(rows)


def _audit(final: pd.DataFrame) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    expected_rows = len(AGENT_ORDER) * len(CONDITION_ORDER) * 10
    checks.append({
        "check": "expected_final_rows",
        "passed": int(len(final)) == expected_rows,
        "detail": f"rows={len(final)}, expected={expected_rows}",
    })
    present_agents = set(final["agent"].astype(str).unique()) if "agent" in final else set()
    checks.append({
        "check": "expected_agents_present",
        "passed": present_agents == set(AGENT_ORDER),
        "detail": str(sorted(present_agents)),
    })
    present_conditions = set(final["module_variant"].astype(str).unique()) if "module_variant" in final else set()
    checks.append({
        "check": "expected_conditions_present",
        "passed": present_conditions == set(CONDITION_ORDER),
        "detail": str(sorted(present_conditions)),
    })
    counts = (
        final.groupby(["agent", "module_variant"])["seed"].nunique()
        if {"agent", "module_variant", "seed"} <= set(final.columns)
        else pd.Series(dtype=int)
    )
    count_detail = {
        f"{agent}/{condition}": int(value)
        for (agent, condition), value in counts.to_dict().items()
    }
    checks.append({
        "check": "ten_seeds_per_agent_condition",
        "passed": bool((counts == 10).all()) and int(len(counts)) == len(AGENT_ORDER) * len(CONDITION_ORDER),
        "detail": count_detail,
    })
    if "prediction_model_hash" in final.columns:
        pred_rows = final[final["module_variant"].isin([
            "prediction_only",
            "prediction_plus_dqi",
            "prediction_placebo_plus_dqi",
            "prediction_plus_dqi_placebo",
        ])]
        pred_hashes = set(pred_rows["prediction_model_hash"].astype(str).dropna().unique())
        checks.append({
            "check": "expected_prediction_model_hash",
            "passed": pred_hashes == {EXPECTED_PREDICTION_HASH},
            "detail": str(sorted(pred_hashes)),
        })
    if "dqi_model_hash" in final.columns:
        dqi_rows = final[final["module_variant"].isin([
            "dqi_only",
            "prediction_plus_dqi",
            "prediction_placebo_plus_dqi",
        ])]
        dqi_hashes = set(dqi_rows["dqi_model_hash"].astype(str).dropna().unique())
        checks.append({
            "check": "expected_dqi_model_hash",
            "passed": dqi_hashes == {EXPECTED_DQI_HASH},
            "detail": str(sorted(dqi_hashes)),
        })
    for col in ["observation_dimension", "state_feature_signature"]:
        if col in final.columns:
            checks.append({
                "check": f"{col}_constant",
                "passed": int(final[col].nunique(dropna=False)) == 1,
                "detail": str(final[col].drop_duplicates().tolist()),
            })
    if {"agent", "seed", "evaluation_physical_trace_hash"} <= set(final.columns):
        max_hashes = final.groupby(["agent", "seed"])["evaluation_physical_trace_hash"].nunique().max()
        checks.append({
            "check": "same_physical_trace_across_conditions_per_agent_seed",
            "passed": int(max_hashes) == 1,
            "detail": f"max_hashes={int(max_hashes)}",
        })
    if "evaluation_parameters_frozen" in final.columns:
        checks.append({
            "check": "evaluation_parameters_frozen",
            "passed": bool(final["evaluation_parameters_frozen"].astype(bool).all()),
            "detail": str(final["evaluation_parameters_frozen"].astype(bool).unique().tolist()),
        })
    return {
        "experiment": "d4a_joint_prediction_dqi_factorial",
        "passed": all(bool(c["passed"]) for c in checks),
        "checks": checks,
    }


def _write_markdown(path: Path, table: pd.DataFrame, audit: dict[str, Any]) -> None:
    lines = [
        f"# D4-A Joint Prediction-DQI Factorial: {'PASSED' if audit['passed'] else 'FAILED'}",
        "",
        "| Algorithm | None | Prediction | DQI | Both | Pred gain | DQI gain | Joint gain | Interaction | Interpretation |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for _, r in table.iterrows():
        lines.append(
            f"| {r['algorithm']} | {r['none_goodput_mbps']:.3f} | "
            f"{r['prediction_goodput_mbps']:.3f} | {r['dqi_goodput_mbps']:.3f} | "
            f"{r['both_goodput_mbps']:.3f} | {r.get('prediction_gain_vs_none_mbps', np.nan):+.3f} | "
            f"{r.get('dqi_gain_vs_none_mbps', np.nan):+.3f} | {r.get('joint_gain_vs_none_mbps', np.nan):+.3f} | "
            f"{r['interaction_mbps']:+.3f} [{r['interaction_ci95_low']:+.3f}, {r['interaction_ci95_high']:+.3f}] | "
            f"{r['interaction_interpretation']} |"
        )
    lines += ["", "## Audit", "", "| Check | Pass | Detail |", "|---|:---:|---|"]
    for c in audit["checks"]:
        detail = str(c["detail"]).replace("|", "/")
        lines.append(f"| {c['check']} | {'yes' if c['passed'] else 'no'} | {detail} |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_latex(path: Path, table: pd.DataFrame) -> None:
    view = table[[
        "algorithm",
        "none_goodput_mbps",
        "prediction_goodput_mbps",
        "dqi_goodput_mbps",
        "both_goodput_mbps",
        "prediction_gain_vs_none_mbps",
        "dqi_gain_vs_none_mbps",
        "joint_gain_vs_none_mbps",
        "interaction_mbps",
    ]].copy()
    view.columns = [
        "Algorithm",
        "None",
        "Prediction",
        "DQI",
        "Both",
        "$\\Delta_P$",
        "$\\Delta_D$",
        "$\\Delta_{Both}$",
        "$I$",
    ]
    path.write_text(view.to_latex(index=False, float_format=lambda x: f"{x:.3f}", escape=False), encoding="utf-8")


def _plot(table: pd.DataFrame, output_stem: Path) -> None:
    apply_ieee_style()
    fig, axes = plt.subplots(1, 2, figsize=(FIG_IEEE_DOUBLE[0], 2.55))
    algorithms = table["algorithm"].tolist()
    x = np.arange(len(algorithms))
    width = 0.22
    gain_specs = [
        ("prediction_gain_vs_none_mbps", "Prediction", PALETTE["blue"]),
        ("dqi_gain_vs_none_mbps", "DQI", PALETTE["orange"]),
        ("joint_gain_vs_none_mbps", "Both", PALETTE["green"]),
    ]
    for offset, (col, label, color) in zip([-width, 0.0, width], gain_specs):
        axes[0].bar(x + offset, table[col], width=width, label=label, color=color)
    axes[0].axhline(0.0, color="0.35", linewidth=0.8)
    axes[0].set_xticks(x, algorithms, rotation=12, ha="right")
    axes[0].set_ylabel("Goodput gain (Mbps)")
    axes[0].legend(frameon=False, ncols=3, loc="upper left")
    axes[0].grid(axis="y", alpha=0.3)

    yerr = np.vstack([
        table["interaction_mbps"] - table["interaction_ci95_low"],
        table["interaction_ci95_high"] - table["interaction_mbps"],
    ])
    axes[1].bar(x, table["interaction_mbps"], color=PALETTE["purple"], width=0.42)
    axes[1].errorbar(x, table["interaction_mbps"], yerr=yerr, fmt="none", ecolor="black", capsize=3, linewidth=0.9)
    axes[1].axhline(0.0, color="0.35", linestyle="--", linewidth=0.8)
    axes[1].set_xticks(x, algorithms, rotation=12, ha="right")
    axes[1].set_ylabel("Interaction $I$ (Mbps)")
    axes[1].grid(axis="y", alpha=0.3)

    axes[0].text(0.5, -0.36, "(a)", transform=axes[0].transAxes, ha="center", va="top", fontweight="bold")
    axes[1].text(0.5, -0.36, "(b)", transform=axes[1].transAxes, ha="center", va="top", fontweight="bold")
    fig.tight_layout(rect=(0.0, 0.08, 1.0, 1.0))
    save_ieee_figure(fig, str(output_stem), formats=("pdf", "png"))
    fig.savefig(str(output_stem.with_suffix(".jpg")), dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--metrics-csv", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("paper_table/table_d4a_joint_prediction_dqi_factorial"))
    parser.add_argument("--paper-output-stem", type=Path, default=Path("paper_figures/fig_d4a_joint_factorial_gain"))
    parser.add_argument("--metric", default="served_goodput_mbps")
    args = parser.parse_args()

    metrics = pd.read_csv(args.metrics_csv or _latest_metrics_csv(args.run_dir))
    final = _final_rows(metrics)
    if final.empty:
        raise ValueError("No D4-A final rows found in metrics CSV")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.paper_output_stem.parent.mkdir(parents=True, exist_ok=True)

    summary = _summarize(final)
    effects = _paired_effects(final, args.metric)
    interactions = _interactions(final, args.metric)
    table = _paper_table(summary, effects, interactions)
    audit = _audit(final)

    final.to_csv(args.output_dir / "d4a_raw_seed_results.csv", index=False)
    summary.to_csv(args.output_dir / "d4a_condition_summary.csv", index=False)
    effects.to_csv(args.output_dir / "d4a_pairwise_effects.csv", index=False)
    interactions.to_csv(args.output_dir / "d4a_factorial_interactions.csv", index=False)
    table.to_csv(args.output_dir / "table_d4a_joint_prediction_dqi_factorial.csv", index=False)
    (args.output_dir / "d4a_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    _write_markdown(args.output_dir / "table_d4a_joint_prediction_dqi_factorial.md", table, audit)
    _write_latex(args.output_dir / "table_d4a_joint_prediction_dqi_factorial.tex", table)
    _plot(table, args.paper_output_stem)

    print(f"wrote {args.output_dir / 'table_d4a_joint_prediction_dqi_factorial.md'}")
    print(f"wrote {args.paper_output_stem.with_suffix('.pdf')}")
    print(f"D4-A AUDIT: {'PASSED' if audit['passed'] else 'FAILED'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
