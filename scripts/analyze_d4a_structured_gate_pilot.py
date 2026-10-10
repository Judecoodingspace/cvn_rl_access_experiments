#!/usr/bin/env python3
"""Analyze the D4-A mixed-mode structured prediction gate pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


METRICS = [
    "served_goodput_mbps",
    "plr",
    "underlay_selection_ratio",
    "overlay_transmissions",
    "underlay_transmissions",
    "pu_access_violation_rate",
    "pu_active_violation_rate",
    "conditional_underlay_violation_rate",
    "average_selected_theta",
    "average_selected_dqi",
    "mean_selected_future_dqi",
]


def _final_rows(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "tables" / "final_checkpoint_results.csv"
    if not path.exists():
        raise FileNotFoundError(f"missing final checkpoint table: {path}")
    data = pd.read_csv(path)
    if "evaluation_variant" in data:
        data = data[data["evaluation_variant"].astype(str).eq("nominal")].copy()
    return data


def _markdown_table(data: pd.DataFrame) -> str:
    headers = data.columns.tolist()
    rows = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in data.iterrows():
        values = []
        for column in headers:
            value = row[column]
            values.append(f"{value:.4f}" if isinstance(value, float) else str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


def _summary(final: pd.DataFrame) -> pd.DataFrame:
    available = [metric for metric in METRICS if metric in final.columns]
    return (
        final.groupby("module_variant", as_index=False)
        .agg(
            n_seeds=("seed", "nunique"),
            **{f"{metric}_mean": (metric, "mean") for metric in available},
            **{f"{metric}_std": (metric, "std") for metric in available},
        )
        .sort_values("module_variant")
    )


def _gate(summary: pd.DataFrame) -> dict[str, object]:
    by_name = summary.set_index("module_variant")
    required = {"dqi_only", "prediction_only", "full_state_only", "full_structured_gate"}
    if not required.issubset(by_name.index):
        return {"passed": False, "reason": f"missing {sorted(required.difference(by_name.index))}"}
    dqi_goodput = float(by_name.loc["dqi_only", "served_goodput_mbps_mean"])
    pred_violation = float(by_name.loc["prediction_only", "pu_access_violation_rate_mean"])
    structured = by_name.loc["full_structured_gate"]
    state_only = by_name.loc["full_state_only"]
    structured_goodput = float(structured["served_goodput_mbps_mean"])
    structured_violation = float(structured["pu_access_violation_rate_mean"])
    structured_underlay = float(structured["underlay_selection_ratio_mean"])
    state_violation = float(state_only["pu_access_violation_rate_mean"])
    state_goodput = float(state_only["served_goodput_mbps_mean"])
    return {
        "goodput_retention_vs_dqi": structured_goodput / max(dqi_goodput, 1e-9),
        "violation_gap_vs_prediction": structured_violation - pred_violation,
        "violation_reduction_vs_state_only": state_violation - structured_violation,
        "goodput_change_vs_state_only": structured_goodput - state_goodput,
        "structured_underlay_selection_ratio": structured_underlay,
        "keeps_underlay_active": structured_underlay > 0.02,
        "retains_at_least_80pct_dqi_goodput": structured_goodput >= 0.8 * dqi_goodput,
        "safer_than_state_only": structured_violation < state_violation,
        "near_prediction_safety": structured_violation <= pred_violation + 0.005,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    final = _final_rows(args.run_dir)
    summary = _summary(final)
    gate = _gate(summary)
    output_dir = args.output_dir or args.run_dir / "analysis" / "structured_gate_pilot"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "structured_gate_summary.csv", index=False)
    (output_dir / "structured_gate_audit.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")
    cols = [
        "module_variant",
        "n_seeds",
        "served_goodput_mbps_mean",
        "plr_mean",
        "underlay_selection_ratio_mean",
        "pu_access_violation_rate_mean",
        "underlay_transmissions_mean",
        "overlay_transmissions_mean",
    ]
    cols = [column for column in cols if column in summary.columns]
    markdown = "\n".join(
        [
            "# D4-A Structured Gate Pilot",
            "",
            "## Gate",
            "",
            "```json",
            json.dumps(gate, indent=2),
            "```",
            "",
            "## Summary",
            "",
            _markdown_table(summary[cols]),
            "",
        ]
    )
    (output_dir / "structured_gate_summary.md").write_text(markdown, encoding="utf-8")
    print(summary[cols].to_string(index=False))
    print(json.dumps(gate, indent=2))
    print(f"wrote {output_dir / 'structured_gate_summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
