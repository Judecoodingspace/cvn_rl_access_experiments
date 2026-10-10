#!/usr/bin/env python3
"""Analyze D3-R3 formal RL: Attention DQI vs current-CQI proxies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ORACLE = "dqi_future_oracle"
ATTENTION = "dqi_attention_frozen"
CURRENT_Q10 = "dqi_current_quality_q10"
CURRENT_RANK = "dqi_current_quality_rank"
LAG1_Q10 = "dqi_lag1_quality_q10"
LAG1_RANK = "dqi_lag1_quality_rank"
NEUTRAL = "dqi_neutral"
PERSISTENCE = "dqi_canonical_persistence"
PLACEBO = "dqi_channel_placebo"

PRIMARY_COMPARATORS = [
    CURRENT_Q10,
    CURRENT_RANK,
    LAG1_Q10,
    LAG1_RANK,
    NEUTRAL,
    PERSISTENCE,
    PLACEBO,
]
SECONDARY_OUTCOMES = [
    "plr",
    "rolling_100ms_system_served_q10_mean",
    "switch_count",
    "blackout_loss_bits",
    "mean_selected_future_dqi",
    "mean_channel_selection_regret",
    "dqi_action_disagreement_rate",
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


def _paired_effect(
    metrics: pd.DataFrame,
    condition: str,
    comparator: str,
    outcome: str,
    *,
    higher_is_better: bool = True,
    seed_offset: int = 0,
) -> dict[str, Any] | None:
    pivot = metrics.pivot_table(index="seed", columns="module_variant", values=outcome, aggfunc="mean")
    if condition not in pivot.columns or comparator not in pivot.columns:
        return None
    sub = pivot[[condition, comparator]].dropna()
    if sub.empty:
        return None
    diff = sub[condition].to_numpy(float) - sub[comparator].to_numpy(float)
    rng = np.random.default_rng(9300 + seed_offset)
    ci_low, ci_high = _bootstrap_ci(diff, rng)
    wins = int((diff > 0).sum()) if higher_is_better else int((diff < 0).sum())
    return {
        "condition": condition,
        "comparator": comparator,
        "outcome": outcome,
        "higher_is_better": higher_is_better,
        "n_seeds": int(len(diff)),
        "condition_mean": float(sub[condition].mean()),
        "comparator_mean": float(sub[comparator].mean()),
        "mean_difference": float(diff.mean()),
        "ci95_low": ci_low,
        "ci95_high": ci_high,
        "seed_wins": wins,
    }


def analyze(metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if "is_final_checkpoint" in metrics.columns:
        metrics = metrics[metrics["is_final_checkpoint"].astype(bool)].copy()

    agg_spec: dict[str, tuple[str, str]] = {
        "served_goodput_mbps": ("served_goodput_mbps", "mean"),
        "plr": ("plr", "mean"),
        "rolling_100ms_system_served_q10_mean": ("rolling_100ms_system_served_q10_mean", "mean"),
        "switch_count": ("switch_count", "mean"),
        "blackout_loss_bits": ("blackout_loss_bits", "mean"),
        "n_seeds": ("seed", "nunique"),
    }
    for col in [
        "mean_selected_future_dqi",
        "mean_channel_selection_regret",
        "dqi_action_disagreement_rate",
    ]:
        if col in metrics.columns:
            agg_spec[col] = (col, "mean")
    summary = (
        metrics.groupby("module_variant", as_index=False)
        .agg(**agg_spec)
        .sort_values("served_goodput_mbps", ascending=False)
    )

    rows: list[dict[str, Any]] = []
    offset = 0
    comparisons = [(ORACLE, NEUTRAL)] + [(ATTENTION, c) for c in PRIMARY_COMPARATORS]
    outcomes = ["served_goodput_mbps"] + [c for c in SECONDARY_OUTCOMES if c in metrics.columns]
    for condition, comparator in comparisons:
        for outcome in outcomes:
            offset += 1
            higher_is_better = outcome not in {"plr", "blackout_loss_bits", "mean_channel_selection_regret"}
            row = _paired_effect(
                metrics,
                condition,
                comparator,
                outcome,
                higher_is_better=higher_is_better,
                seed_offset=offset,
            )
            if row is not None:
                rows.append(row)
    effects = pd.DataFrame(rows)

    checks: list[dict[str, Any]] = []
    required = [(ORACLE, NEUTRAL)] + [(ATTENTION, c) for c in PRIMARY_COMPARATORS]
    for condition, comparator in required:
        sub = effects[
            (effects["condition"] == condition)
            & (effects["comparator"] == comparator)
            & (effects["outcome"] == "served_goodput_mbps")
        ]
        if len(sub) != 1:
            checks.append({"check": f"{condition}_vs_{comparator}", "passed": False, "detail": "missing row"})
            continue
        r = sub.iloc[0]
        passed = bool(r["ci95_low"] > 0 and r["seed_wins"] >= 4)
        checks.append({
            "check": f"{condition}_vs_{comparator}",
            "passed": passed,
            "detail": (
                f"diff={r['mean_difference']:.4f}, CI95=[{r['ci95_low']:.4f},{r['ci95_high']:.4f}], "
                f"wins={int(r['seed_wins'])}/{int(r['n_seeds'])}"
            ),
        })
    report = {
        "experiment": "d3_r3_dqi_vs_current_cqi_formal_rl",
        "passed": all(c["passed"] for c in checks),
        "checks": checks,
        "note": "Primary gate is served_goodput_mbps; secondary metrics are reported for diagnosis.",
    }
    return summary, effects, report


def write_markdown(path: Path, summary: pd.DataFrame, effects: pd.DataFrame, report: dict[str, Any]) -> None:
    lines = [
        f"# D3-R3 Formal RL: {'PASSED' if report['passed'] else 'FAILED'}",
        "",
        "## Final checkpoint means",
        "",
        "| Condition | Goodput | PLR | Rolling q10 | Switches | Blackout bits |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for _, r in summary.iterrows():
        lines.append(
            f"| {r['module_variant']} | {r['served_goodput_mbps']:.4f} | "
            f"{r['plr']:.4f} | {r['rolling_100ms_system_served_q10_mean']:.4f} | "
            f"{r['switch_count']:.2f} | {r['blackout_loss_bits']:.2f} |"
        )

    served = effects[effects["outcome"] == "served_goodput_mbps"]
    lines += [
        "",
        "## Primary paired goodput effects",
        "",
        "| Condition | Comparator | Diff Mbps | CI95 | Wins |",
        "|---|---|---:|---:|---:|",
    ]
    for _, r in served.iterrows():
        lines.append(
            f"| {r['condition']} | {r['comparator']} | {r['mean_difference']:.4f} | "
            f"[{r['ci95_low']:.4f}, {r['ci95_high']:.4f}] | {int(r['seed_wins'])}/{int(r['n_seeds'])} |"
        )

    lines += ["", "## Gate checks", "", "| Check | Pass | Detail |", "|---|:---:|---|"]
    for c in report["checks"]:
        lines.append(f"| {c['check']} | {'yes' if c['passed'] else 'no'} | {c['detail']} |")
    lines += ["", report["note"], ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--metrics-csv", type=Path, default=None)
    args = parser.parse_args()

    metrics = pd.read_csv(args.metrics_csv or _latest_metrics_csv(args.run_dir))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary, effects, report = analyze(metrics)
    summary.to_csv(args.output_dir / "d3_r3_formal_summary.csv", index=False)
    effects.to_csv(args.output_dir / "d3_r3_pairwise_effects.csv", index=False)
    (args.output_dir / "d3_r3_gate_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(args.output_dir / "D3_R3_FORMAL_RL.md", summary, effects, report)
    print(f"wrote {args.output_dir / 'D3_R3_FORMAL_RL.md'}")
    print(f"D3-R3 FORMAL RL: {'PASSED' if report['passed'] else 'FAILED'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
