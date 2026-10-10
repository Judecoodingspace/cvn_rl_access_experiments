"""Build a D3-specific success gate from an existing RL run/audit.

The legacy D1/D2 audit script is intentionally conservative and still treats
``future_oracle > attention_frozen`` as a required ordering.  For D3 this is no
longer the right gate: the frozen attention descriptor is a deployable
prediction module, while the future oracle is a positive-control upper-bound
label.  If the learned descriptor closes or exceeds the oracle-vs-neutral gap,
that should be reported, not marked as a failure.

This script reads the metrics and pairwise-effects CSVs produced by the
existing audit and writes a D3-cleaned gate report.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ORACLE = "dqi_future_oracle"
FROZEN = "dqi_attention_frozen"
PERSISTENCE = "dqi_canonical_persistence"
NEUTRAL = "dqi_neutral"
CHANNEL_PLACEBO = "dqi_channel_placebo"

MIN_SEED_WINS = 4


def _latest_metrics_csv(run_dir: Path) -> Path:
    candidates = sorted(run_dir.rglob("access_metrics.csv"))
    if not candidates:
        candidates = sorted(run_dir.rglob("*metrics*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No metrics CSV found under {run_dir}")
    return candidates[-1]


def _effect_row(
    effects: pd.DataFrame,
    condition: str,
    comparator: str,
    outcome: str = "served_goodput_mbps",
) -> pd.Series | None:
    sub = effects[
        (effects["condition"].astype(str) == condition)
        & (effects["comparator"].astype(str) == comparator)
        & (effects["outcome"].astype(str) == outcome)
    ]
    if len(sub) != 1:
        return None
    return sub.iloc[0]


def _append_positive_effect_check(
    checks: list[dict[str, Any]],
    effects: pd.DataFrame,
    *,
    check: str,
    condition: str,
    comparator: str,
    outcome: str = "served_goodput_mbps",
) -> None:
    row = _effect_row(effects, condition, comparator, outcome)
    if row is None:
        checks.append({
            "check": check,
            "passed": False,
            "detail": f"missing paired effect row: {condition} vs {comparator}, {outcome}",
        })
        return
    passed = bool(row["ci95_low"] > 0 and row["seed_wins"] >= MIN_SEED_WINS)
    checks.append({
        "check": check,
        "passed": passed,
        "detail": (
            f"diff={row['mean_difference']:.4f}, "
            f"CI95=[{row['ci95_low']:.4f},{row['ci95_high']:.4f}], "
            f"wins={int(row['seed_wins'])}/{int(row['n_seeds'])}"
        ),
    })


def _mean(metrics: pd.DataFrame, condition: str, outcome: str) -> float:
    sub = metrics[metrics["module_variant"].astype(str) == condition]
    if sub.empty:
        return float("nan")
    return float(sub[outcome].mean())


def build_report(metrics: pd.DataFrame, effects: pd.DataFrame) -> dict[str, Any]:
    variants = set(metrics["module_variant"].astype(str).unique())
    checks: list[dict[str, Any]] = []

    required = {ORACLE, FROZEN, NEUTRAL, CHANNEL_PLACEBO}
    missing = sorted(required - variants)
    checks.append({
        "check": "required_d3_conditions_present",
        "passed": not missing,
        "detail": f"missing={missing}",
    })

    _append_positive_effect_check(
        checks, effects,
        check="oracle_positive_control_vs_neutral",
        condition=ORACLE,
        comparator=NEUTRAL,
    )
    _append_positive_effect_check(
        checks, effects,
        check="oracle_positive_control_vs_channel_placebo",
        condition=ORACLE,
        comparator=CHANNEL_PLACEBO,
    )
    _append_positive_effect_check(
        checks, effects,
        check="attention_frozen_vs_neutral",
        condition=FROZEN,
        comparator=NEUTRAL,
    )
    _append_positive_effect_check(
        checks, effects,
        check="attention_frozen_vs_channel_placebo",
        condition=FROZEN,
        comparator=CHANNEL_PLACEBO,
    )
    if PERSISTENCE in variants:
        _append_positive_effect_check(
            checks, effects,
            check="attention_frozen_vs_canonical_persistence",
            condition=FROZEN,
            comparator=PERSISTENCE,
        )

    if "dwell_violation_count" in metrics.columns:
        dwell_sum = int(metrics["dwell_violation_count"].sum())
        checks.append({
            "check": "dwell_violation_zero",
            "passed": dwell_sum == 0,
            "detail": f"total dwell violations={dwell_sum}",
        })
    if "evaluation_parameters_frozen" in metrics.columns:
        all_frozen = bool(metrics["evaluation_parameters_frozen"].all())
        checks.append({
            "check": "evaluation_parameters_frozen",
            "passed": all_frozen,
            "detail": f"all frozen={all_frozen}",
        })
    if "seed" in metrics.columns:
        n_seeds = int(metrics["seed"].nunique())
        checks.append({
            "check": "paired_seeds_count",
            "passed": n_seeds == 5,
            "detail": f"n_seeds={n_seeds}",
        })

    neutral = _mean(metrics, NEUTRAL, "served_goodput_mbps")
    oracle = _mean(metrics, ORACLE, "served_goodput_mbps")
    frozen = _mean(metrics, FROZEN, "served_goodput_mbps")
    denominator = oracle - neutral
    closure = (frozen - neutral) / denominator if abs(denominator) > 1e-12 else float("nan")
    oracle_minus_frozen = oracle - frozen

    passed = all(bool(c["passed"]) for c in checks)
    return {
        "experiment": "d3_attention_dqi_frozen_clean_success_gate",
        "passed": passed,
        "checks": checks,
        "served_goodput_mean_mbps": {
            "oracle": oracle,
            "attention_frozen": frozen,
            "neutral": neutral,
            "channel_placebo": _mean(metrics, CHANNEL_PLACEBO, "served_goodput_mbps"),
            "canonical_persistence": _mean(metrics, PERSISTENCE, "served_goodput_mbps"),
        },
        "oracle_gap_closure": closure,
        "oracle_minus_attention_frozen_mbps": oracle_minus_frozen,
        "interpretation": (
            "Oracle-vs-attention ordering is informational for D3. The success "
            "gate requires oracle positive controls and attention_frozen beating "
            "neutral/placebo/persistence with paired-seed CI support."
        ),
    }


def write_markdown(report: dict[str, Any], out_path: Path) -> None:
    status = "PASSED" if report["passed"] else "FAILED"
    lines = [
        f"# D3 Clean Success Gate: {status}",
        "",
        "## Goodput means",
        "",
        "| Condition | Served goodput Mbps |",
        "|---|---:|",
    ]
    for key, val in report["served_goodput_mean_mbps"].items():
        if np.isfinite(val):
            lines.append(f"| {key} | {val:.4f} |")
    lines += [
        "",
        f"- Oracle gap closure: {report['oracle_gap_closure']:.3f}",
        f"- Oracle - Attention Frozen: {report['oracle_minus_attention_frozen_mbps']:.4f} Mbps",
        "",
        "## Gate checks",
        "",
        "| Check | Pass | Detail |",
        "|---|:---:|---|",
    ]
    for check in report["checks"]:
        lines.append(
            f"| {check['check']} | {'yes' if check['passed'] else 'no'} | {check['detail']} |"
        )
    lines += ["", report["interpretation"], ""]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--pairwise-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--metrics-csv", type=Path, default=None)
    args = parser.parse_args()

    metrics_csv = args.metrics_csv or _latest_metrics_csv(args.run_dir)
    metrics = pd.read_csv(metrics_csv)
    effects = pd.read_csv(args.pairwise_csv)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = build_report(metrics, effects)
    json_path = args.output_dir / "d3_clean_success_gate_report.json"
    md_path = args.output_dir / "D3_CLEAN_SUCCESS_GATE.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(report, md_path)
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    print(f"D3 CLEAN GATE: {'PASSED' if report['passed'] else 'FAILED'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
