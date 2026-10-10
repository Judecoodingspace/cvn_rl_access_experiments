#!/usr/bin/env python3
"""Analyze D3-R5 Attention Frozen DQI across RL backbones."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


AGENT_LABELS = {
    "proposed_esn_ddqn": "ESN-DDQN",
    "ddqn_no_esn": "DDQN",
    "dqn": "DQN",
    "lstm_ddqn": "LSTM-DDQN",
    "sac_discrete": "Discrete SAC",
    "ppo_discrete": "PPO-discrete",
    "tabular_q_learning": "Tabular Q",
}
AGENT_ORDER = list(AGENT_LABELS)
REFERENCE = "ddqn_no_esn"
EXPECTED_MODEL_HASH = "4623d509d776427ca970e640fdfff2d894abb9e3145b717c6bf9c7b77f2c24a4"

SUMMARY_METRICS = [
    "served_goodput_mbps",
    "plr",
    "mean_reward",
    "average_selected_dqi",
    "mean_selected_future_dqi",
    "mean_channel_selection_regret",
    "dqi_action_disagreement_rate",
    "rolling_100ms_system_served_q10_mean",
    "switch_count",
    "blackout_loss_bits",
    "completion_rate",
    "trainable_parameter_count",
    "fixed_reservoir_parameter_count",
    "total_stored_parameter_count",
    "inference_latency_ms",
    "training_time_seconds",
]


def _latest_metrics_csv(run_dir: Path) -> Path:
    candidates = sorted(run_dir.rglob("access_metrics.csv"))
    if not candidates:
        candidates = sorted(run_dir.rglob("*metrics*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No metrics CSV found under {run_dir}")
    return candidates[-1]


def _bootstrap_ci(diff: np.ndarray, rng: np.random.Generator, n_boot: int = 20000) -> tuple[float, float]:
    idx = rng.integers(0, diff.size, size=(n_boot, diff.size))
    boot = diff[idx].mean(axis=1)
    return float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def _final_rows(metrics: pd.DataFrame) -> pd.DataFrame:
    final = metrics.copy()
    if "is_final_checkpoint" in final.columns:
        final = final[final["is_final_checkpoint"].astype(bool)].copy()
    if "module_variant" in final.columns:
        final = final[final["module_variant"].astype(str).eq("dqi_attention_frozen")].copy()
    return final


def _summarize(final: pd.DataFrame) -> pd.DataFrame:
    available = [m for m in SUMMARY_METRICS if m in final.columns]
    agg: dict[str, tuple[str, str]] = {"n_seeds": ("seed", "nunique")}
    for metric in available:
        agg[f"{metric}_mean"] = (metric, "mean")
        agg[f"{metric}_std"] = (metric, "std")
    summary = final.groupby("agent", as_index=False).agg(**agg)
    summary["algorithm"] = summary["agent"].map(AGENT_LABELS).fillna(summary["agent"])
    summary["agent_order"] = summary["agent"].map({a: i for i, a in enumerate(AGENT_ORDER)}).fillna(999)
    return summary.sort_values("agent_order").drop(columns=["agent_order"])


def _paired_effects(final: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    metrics = [
        "served_goodput_mbps",
        "plr",
        "mean_reward",
        "mean_selected_future_dqi",
        "mean_channel_selection_regret",
        "rolling_100ms_system_served_q10_mean",
    ]
    for metric_i, metric in enumerate([m for m in metrics if m in final.columns]):
        pivot = final.pivot_table(index="seed", columns="agent", values=metric, aggfunc="mean")
        if REFERENCE not in pivot:
            continue
        for agent in AGENT_ORDER:
            if agent == REFERENCE or agent not in pivot:
                continue
            sub = pivot[[agent, REFERENCE]].dropna()
            if sub.empty:
                continue
            diff = sub[agent].to_numpy(float) - sub[REFERENCE].to_numpy(float)
            ci_low, ci_high = _bootstrap_ci(
                diff,
                np.random.default_rng(12500 + metric_i * 101 + AGENT_ORDER.index(agent)),
            )
            rows.append({
                "agent": agent,
                "algorithm": AGENT_LABELS.get(agent, agent),
                "reference": REFERENCE,
                "metric": metric,
                "n_seeds": int(len(diff)),
                "agent_mean": float(sub[agent].mean()),
                "reference_mean": float(sub[REFERENCE].mean()),
                "mean_difference": float(diff.mean()),
                "ci95_low": ci_low,
                "ci95_high": ci_high,
                "positive_seeds": int((diff > 0).sum()),
            })
    return pd.DataFrame(rows)


def _paper_table(summary: pd.DataFrame, effects: pd.DataFrame) -> pd.DataFrame:
    effect_map = effects[effects["metric"].eq("served_goodput_mbps")].set_index("agent")
    rows: list[dict[str, Any]] = []
    for _, row in summary.iterrows():
        agent = str(row["agent"])
        eff = effect_map.loc[agent] if agent in effect_map.index else None
        delta = 0.0 if agent == REFERENCE else float(eff["mean_difference"]) if eff is not None else np.nan
        ci_low = 0.0 if agent == REFERENCE else float(eff["ci95_low"]) if eff is not None else np.nan
        ci_high = 0.0 if agent == REFERENCE else float(eff["ci95_high"]) if eff is not None else np.nan
        rows.append({
            "agent": agent,
            "algorithm": row["algorithm"],
            "n_seeds": int(row["n_seeds"]),
            "goodput_mean_mbps": row.get("served_goodput_mbps_mean", np.nan),
            "goodput_std_mbps": row.get("served_goodput_mbps_std", np.nan),
            "plr_mean": row.get("plr_mean", np.nan),
            "reward_mean": row.get("mean_reward_mean", np.nan),
            "selected_future_dqi_mean": row.get("mean_selected_future_dqi_mean", np.nan),
            "channel_selection_regret_mean": row.get("mean_channel_selection_regret_mean", np.nan),
            "switch_count_mean": row.get("switch_count_mean", np.nan),
            "rolling_q10_mean": row.get("rolling_100ms_system_served_q10_mean_mean", np.nan),
            "trainable_parameters": row.get("trainable_parameter_count_mean", np.nan),
            "fixed_parameters": row.get("fixed_reservoir_parameter_count_mean", np.nan),
            "total_parameters": row.get("total_stored_parameter_count_mean", np.nan),
            "inference_latency_mean_ms": row.get("inference_latency_ms_mean", np.nan),
            "training_time_mean_s": row.get("training_time_seconds_mean", np.nan),
            "delta_goodput_vs_ddqn_mbps": delta,
            "delta_goodput_ci95_low": ci_low,
            "delta_goodput_ci95_high": ci_high,
            "positive_seeds_vs_ddqn": 0 if agent == REFERENCE else int(eff["positive_seeds"]) if eff is not None else np.nan,
            "delta_display": (
                "Reference" if agent == REFERENCE
                else f"{delta:+.3f} [{ci_low:+.3f}, {ci_high:+.3f}]"
            ),
        })
    return pd.DataFrame(rows)


def _audit(final: pd.DataFrame, table: pd.DataFrame) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    agents = set(final["agent"].astype(str).unique()) if "agent" in final else set()
    expected = set(AGENT_ORDER)
    checks.append({
        "check": "all_expected_agents_present",
        "passed": agents == expected,
        "detail": f"present={sorted(agents)}, expected={sorted(expected)}",
    })
    seed_counts = final.groupby("agent")["seed"].nunique().to_dict() if "seed" in final else {}
    checks.append({
        "check": "ten_unique_seeds_per_agent",
        "passed": all(int(seed_counts.get(agent, 0)) == 10 for agent in AGENT_ORDER),
        "detail": str(seed_counts),
    })
    if "dqi_model_hash" in final.columns:
        hashes = set(final["dqi_model_hash"].astype(str).dropna().unique())
        checks.append({
            "check": "expected_attention_dqi_model_hash",
            "passed": hashes == {EXPECTED_MODEL_HASH},
            "detail": str(sorted(hashes)),
        })
    for col in ["observation_dimension", "state_feature_signature"]:
        if col in final.columns:
            checks.append({
                "check": f"{col}_constant",
                "passed": int(final[col].nunique(dropna=False)) == 1,
                "detail": str(final[col].drop_duplicates().tolist()),
            })
    if "evaluation_parameters_frozen" in final.columns:
        checks.append({
            "check": "evaluation_parameters_frozen",
            "passed": bool(final["evaluation_parameters_frozen"].astype(bool).all()),
            "detail": str(final["evaluation_parameters_frozen"].astype(bool).unique().tolist()),
        })
    checks.append({
        "check": "reference_ddqn_present",
        "passed": REFERENCE in set(table["agent"].astype(str)),
        "detail": REFERENCE,
    })
    return {
        "experiment": "d3_r5_dqi_backbone_attention_frozen",
        "passed": all(c["passed"] for c in checks),
        "checks": checks,
    }


def _write_markdown(path: Path, table: pd.DataFrame, audit: dict[str, Any]) -> None:
    lines = [
        f"# D3-R5 DQI Backbone Attention Frozen: {'PASSED' if audit['passed'] else 'FAILED'}",
        "",
        "| Algorithm | Goodput Mbps | Δ vs DDQN | PLR | Reward | Selected future DQI | Regret | Switches | Params | Latency ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in table.iterrows():
        lines.append(
            f"| {r['algorithm']} | {r['goodput_mean_mbps']:.3f} ± {r['goodput_std_mbps']:.3f} | "
            f"{r['delta_display']} | {r['plr_mean']:.4f} | {r['reward_mean']:.4f} | "
            f"{r['selected_future_dqi_mean']:.4f} | {r['channel_selection_regret_mean']:.4f} | "
            f"{r['switch_count_mean']:.1f} | {r['total_parameters']:.0f} | {r['inference_latency_mean_ms']:.3f} |"
        )
    lines += ["", "## Audit", "", "| Check | Pass | Detail |", "|---|:---:|---|"]
    for c in audit["checks"]:
        lines.append(f"| {c['check']} | {'yes' if c['passed'] else 'no'} | {c['detail']} |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_latex(path: Path, table: pd.DataFrame) -> None:
    cols = [
        "algorithm",
        "goodput_mean_mbps",
        "delta_display",
        "plr_mean",
        "selected_future_dqi_mean",
        "channel_selection_regret_mean",
        "total_parameters",
        "inference_latency_mean_ms",
    ]
    view = table[cols].copy()
    view.columns = [
        "Algorithm",
        "Goodput",
        "$\\Delta$ vs DDQN",
        "PLR",
        "Sel. future DQI",
        "Regret",
        "Params",
        "Latency ms",
    ]
    text = view.to_latex(index=False, float_format=lambda x: f"{x:.3f}", escape=False)
    path.write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("paper_table/table_d3_r5_dqi_backbone_attention"))
    parser.add_argument("--metrics-csv", type=Path, default=None)
    args = parser.parse_args()

    metrics = pd.read_csv(args.metrics_csv or _latest_metrics_csv(args.run_dir))
    final = _final_rows(metrics)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = _summarize(final)
    effects = _paired_effects(final)
    table = _paper_table(summary, effects)
    audit = _audit(final, table)

    final.to_csv(args.output_dir / "d3_r5_raw_seed_results.csv", index=False)
    summary.to_csv(args.output_dir / "d3_r5_backbone_summary.csv", index=False)
    effects.to_csv(args.output_dir / "d3_r5_pairwise_effects.csv", index=False)
    table.to_csv(args.output_dir / "table_d3_r5_dqi_backbone_attention.csv", index=False)
    (args.output_dir / "d3_r5_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    _write_markdown(args.output_dir / "table_d3_r5_dqi_backbone_attention.md", table, audit)
    _write_latex(args.output_dir / "table_d3_r5_dqi_backbone_attention.tex", table)
    print(f"wrote {args.output_dir / 'table_d3_r5_dqi_backbone_attention.md'}")
    print(f"D3-R5 AUDIT: {'PASSED' if audit['passed'] else 'FAILED'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
