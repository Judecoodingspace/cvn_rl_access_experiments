#!/usr/bin/env python3
"""Compare D4-A mixed-mode pilot runs across underlay theta thresholds."""

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


def _read_final(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "tables" / "final_checkpoint_results.csv"
    if not path.exists():
        raise FileNotFoundError(f"missing final checkpoint table: {path}")
    data = pd.read_csv(path)
    if "evaluation_variant" in data:
        data = data[data["evaluation_variant"].astype(str).eq("nominal")].copy()
    config = json.loads((run_dir / "config.frozen.json").read_text(encoding="utf-8"))
    thresholds = config.get("action_mask_thresholds", {})
    theta = thresholds.get("underlay_theta_min")
    data["run_dir"] = str(run_dir)
    data["underlay_power_cap_min_sweep"] = float(thresholds.get("underlay_power_cap_min", 0.0))
    data["underlay_theta_min_sweep"] = float(theta) if theta is not None else -1.0
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("paper_table/d4a_mixed_mode_theta_sweep"))
    args = parser.parse_args()

    final = pd.concat([_read_final(run_dir) for run_dir in args.run_dirs], ignore_index=True)
    available = [metric for metric in METRICS if metric in final.columns]
    group_cols = ["underlay_power_cap_min_sweep", "underlay_theta_min_sweep", "module_variant"]
    summary = (
        final.groupby(group_cols, as_index=False)
        .agg(
            n_seeds=("seed", "nunique"),
            **{f"{metric}_mean": (metric, "mean") for metric in available},
            **{f"{metric}_std": (metric, "std") for metric in available},
        )
        .sort_values(group_cols)
    )
    gate = (
        summary.groupby(["underlay_power_cap_min_sweep", "underlay_theta_min_sweep"], as_index=False)
        .agg(
            max_pu_access_violation_rate=("pu_access_violation_rate_mean", "max"),
            min_underlay_selection_ratio=("underlay_selection_ratio_mean", "min"),
            max_underlay_selection_ratio=("underlay_selection_ratio_mean", "max"),
            mean_goodput=("served_goodput_mbps_mean", "mean"),
            mean_plr=("plr_mean", "mean"),
        )
        .sort_values(["underlay_power_cap_min_sweep", "underlay_theta_min_sweep"])
    )
    gate["keeps_underlay_active"] = gate["max_underlay_selection_ratio"] > 0.02
    gate["near_zero_pu_violation"] = gate["max_pu_access_violation_rate"] <= 0.005

    args.output_dir.mkdir(parents=True, exist_ok=True)
    final.to_csv(args.output_dir / "mixed_mode_theta_raw_final.csv", index=False)
    summary.to_csv(args.output_dir / "mixed_mode_theta_condition_summary.csv", index=False)
    gate.to_csv(args.output_dir / "mixed_mode_theta_gate_summary.csv", index=False)
    condition_cols = [
        "underlay_power_cap_min_sweep",
        "underlay_theta_min_sweep",
        "module_variant",
        "n_seeds",
        "served_goodput_mbps_mean",
        "plr_mean",
        "underlay_selection_ratio_mean",
        "pu_access_violation_rate_mean",
    ]
    markdown = "\n".join(
        [
            "# D4-A Mixed-Mode Theta Sweep",
            "",
            "## Gate Summary",
            "",
            _markdown_table(gate),
            "",
            "## Condition Summary",
            "",
            _markdown_table(summary[condition_cols]),
            "",
        ]
    )
    (args.output_dir / "mixed_mode_theta_sweep.md").write_text(markdown, encoding="utf-8")
    print(gate.to_string(index=False))
    print(f"wrote {args.output_dir / 'mixed_mode_theta_sweep.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
