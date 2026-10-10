"""Analyze D3-R2 DQI descriptor update/broadcast sensitivity.

D3-R2 treats ``dqi_prediction_update_slots`` as the RSU/BS-side DQI descriptor
compute-and-broadcast period.  This script compares attention-frozen update
periods against neutral, channel placebo, canonical persistence, and future
oracle controls.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ORACLE = "dqi_future_oracle"
NEUTRAL = "dqi_neutral"
PLACEBO = "dqi_channel_placebo"
PERSISTENCE = "dqi_canonical_persistence"
ATTENTION_CONDITIONS = {
    "dqi_attention_frozen_u1": 1,
    "dqi_attention_frozen_u10": 10,
    "dqi_attention_frozen_u100": 100,
}


def _latest_metrics_csv(run_dir: Path) -> Path:
    candidates = sorted(run_dir.rglob("access_metrics.csv"))
    if not candidates:
        candidates = sorted(run_dir.rglob("*metrics*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No metrics CSV found under {run_dir}")
    return candidates[-1]


def _bootstrap_ci(diff: np.ndarray, rng: np.random.Generator, n_boot: int = 20000) -> tuple[float, float]:
    if diff.size == 0:
        return float("nan"), float("nan")
    idx = rng.integers(0, diff.size, size=(n_boot, diff.size))
    boot = diff[idx].mean(axis=1)
    return float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def _paired_effect(
    metrics: pd.DataFrame,
    condition: str,
    comparator: str,
    outcome: str,
    *,
    seed_offset: int,
) -> dict[str, Any] | None:
    pivot = metrics.pivot_table(index="seed", columns="module_variant", values=outcome, aggfunc="mean")
    if condition not in pivot.columns or comparator not in pivot.columns:
        return None
    sub = pivot[[condition, comparator]].dropna()
    if sub.empty:
        return None
    diff = sub[condition].to_numpy(dtype=float) - sub[comparator].to_numpy(dtype=float)
    rng = np.random.default_rng(7300 + seed_offset)
    ci_low, ci_high = _bootstrap_ci(diff, rng)
    comp_mean = float(sub[comparator].mean())
    return {
        "condition": condition,
        "comparator": comparator,
        "outcome": outcome,
        "n_seeds": int(sub.shape[0]),
        "condition_mean": float(sub[condition].mean()),
        "comparator_mean": comp_mean,
        "mean_difference": float(diff.mean()),
        "relative_gain": float(diff.mean() / comp_mean) if abs(comp_mean) > 1e-12 else float("nan"),
        "ci95_low": ci_low,
        "ci95_high": ci_high,
        "seed_wins": int((diff > 0).sum()),
    }


def analyze(metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if "is_final_checkpoint" in metrics.columns:
        metrics = metrics[metrics["is_final_checkpoint"].astype(bool)].copy()
    switch_col = "access_switch_count" if "access_switch_count" in metrics.columns else "switch_count"
    optional_aggs: dict[str, tuple[str, str]] = {}
    if switch_col in metrics.columns:
        optional_aggs["switch_count"] = (switch_col, "mean")
    if "blackout_loss_bits" in metrics.columns:
        optional_aggs["blackout_loss_bits"] = ("blackout_loss_bits", "mean")
    means = (
        metrics.groupby("module_variant", as_index=False)
        .agg(
            served_goodput_mbps=("served_goodput_mbps", "mean"),
            rolling_100ms_system_served_q10_mean=("rolling_100ms_system_served_q10_mean", "mean"),
            n_seeds=("seed", "nunique"),
            **optional_aggs,
        )
        .sort_values("served_goodput_mbps", ascending=False)
    )
    means["dqi_update_slots"] = means["module_variant"].map(ATTENTION_CONDITIONS)

    rows: list[dict[str, Any]] = []
    comparators = [NEUTRAL, PLACEBO, PERSISTENCE, ORACLE]
    outcomes = ["served_goodput_mbps", "rolling_100ms_system_served_q10_mean"]
    offset = 0
    for cond in ATTENTION_CONDITIONS:
        for comp in comparators:
            for outcome in outcomes:
                offset += 1
                row = _paired_effect(metrics, cond, comp, outcome, seed_offset=offset)
                if row is not None:
                    row["dqi_update_slots"] = ATTENTION_CONDITIONS[cond]
                    rows.append(row)

    effects = pd.DataFrame(rows)
    gate_checks: list[dict[str, Any]] = []
    for cond, update_slots in ATTENTION_CONDITIONS.items():
        for comp in [NEUTRAL, PLACEBO, PERSISTENCE]:
            sub = effects[
                (effects["condition"] == cond)
                & (effects["comparator"] == comp)
                & (effects["outcome"] == "served_goodput_mbps")
            ]
            if len(sub) != 1:
                gate_checks.append({
                    "check": f"u{update_slots}_vs_{comp}",
                    "passed": False,
                    "detail": "missing effect row",
                })
                continue
            r = sub.iloc[0]
            passed = bool(r["ci95_low"] > 0 and r["seed_wins"] >= 4)
            gate_checks.append({
                "check": f"u{update_slots}_vs_{comp}",
                "passed": passed,
                "detail": (
                    f"diff={r['mean_difference']:.4f}, CI95=[{r['ci95_low']:.4f},{r['ci95_high']:.4f}], "
                    f"wins={int(r['seed_wins'])}/{int(r['n_seeds'])}"
                ),
            })

    summary = {
        "experiment": "d3_r2_dqi_descriptor_update_broadcast_sensitivity",
        "passed_strict_all_updates": all(c["passed"] for c in gate_checks),
        "passed_primary_u1_u10": all(
            c["passed"] for c in gate_checks if c["check"].startswith(("u1_", "u10_"))
        ),
        "gate_checks": gate_checks,
        "interpretation": (
            "Update slots are the DQI descriptor compute+broadcast period. "
            "U=1 and U=10 are primary freshness settings; U=100 is a deliberate "
            "staleness stress test and may drop."
        ),
    }
    return means, effects, summary


def write_markdown(means: pd.DataFrame, effects: pd.DataFrame, summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# D3-R2 DQI Update/Broadcast Sensitivity",
        "",
        f"- Strict all-update gate: {'PASSED' if summary['passed_strict_all_updates'] else 'FAILED'}",
        f"- Primary U=1/U=10 gate: {'PASSED' if summary['passed_primary_u1_u10'] else 'FAILED'}",
        "",
        "## Mean outcomes",
        "",
        "| Condition | Update slots | Served goodput Mbps | Rolling q10 | Switches | Blackout bits |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for _, r in means.iterrows():
        update = "" if pd.isna(r.get("dqi_update_slots")) else str(int(r["dqi_update_slots"]))
        lines.append(
            f"| {r['module_variant']} | {update} | {r['served_goodput_mbps']:.4f} | "
            f"{r['rolling_100ms_system_served_q10_mean']:.4f} | {r.get('switch_count', float('nan')):.2f} | "
            f"{r['blackout_loss_bits']:.2f} |"
        )

    lines += ["", "## Paired served-goodput effects", "", "| Condition | Comparator | Diff Mbps | CI95 | Wins |", "|---|---|---:|---:|---:|"]
    served = effects[effects["outcome"] == "served_goodput_mbps"].copy()
    for _, r in served.iterrows():
        lines.append(
            f"| {r['condition']} | {r['comparator']} | {r['mean_difference']:.4f} | "
            f"[{r['ci95_low']:.4f}, {r['ci95_high']:.4f}] | {int(r['seed_wins'])}/{int(r['n_seeds'])} |"
        )
    lines += ["", "## Gate checks", "", "| Check | Pass | Detail |", "|---|:---:|---|"]
    for c in summary["gate_checks"]:
        lines.append(f"| {c['check']} | {'yes' if c['passed'] else 'no'} | {c['detail']} |")
    lines += ["", summary["interpretation"], ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--metrics-csv", type=Path, default=None)
    args = parser.parse_args()

    metrics_csv = args.metrics_csv or _latest_metrics_csv(args.run_dir)
    metrics = pd.read_csv(metrics_csv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    means, effects, summary = analyze(metrics)
    means.to_csv(args.output_dir / "d3_r2_update_summary.csv", index=False)
    effects.to_csv(args.output_dir / "d3_r2_update_pairwise_effects.csv", index=False)
    (args.output_dir / "d3_r2_update_gate_report.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    write_markdown(means, effects, summary, args.output_dir / "D3_R2_UPDATE_SENSITIVITY.md")
    print(f"wrote {args.output_dir / 'D3_R2_UPDATE_SENSITIVITY.md'}")
    print(f"PRIMARY U1/U10 GATE: {'PASSED' if summary['passed_primary_u1_u10'] else 'FAILED'}")
    print(f"STRICT ALL-UPDATES GATE: {'PASSED' if summary['passed_strict_all_updates'] else 'FAILED'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
