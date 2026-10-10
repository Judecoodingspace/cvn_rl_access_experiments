#!/usr/bin/env python3
"""Summarize the D4-A mixed-mode pilot run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


SUMMARY_COLUMNS = [
    "served_goodput_mbps",
    "plr",
    "underlay_selection_ratio",
    "access_selection_ratio",
    "overlay_transmissions",
    "underlay_transmissions",
    "pu_access_violation_rate",
    "pu_active_violation_rate",
    "conditional_underlay_violation_rate",
    "average_selected_theta",
    "average_selected_dqi",
    "mean_selected_future_dqi",
    "mean_channel_selection_regret",
]


def _metrics_path(run_dir: Path) -> Path:
    final = run_dir / "tables" / "final_checkpoint_results.csv"
    if final.exists():
        return final
    metrics = run_dir / "metrics" / "access_metrics.csv"
    if metrics.exists():
        return metrics
    raise FileNotFoundError(f"No metrics CSV found under {run_dir}")


def _final_rows(data: pd.DataFrame) -> pd.DataFrame:
    final = data.copy()
    if "is_final_checkpoint" in final:
        final = final[final["is_final_checkpoint"].astype(bool)].copy()
    if "evaluation_variant" in final:
        final = final[final["evaluation_variant"].astype(str).eq("nominal")].copy()
    return final


def _summary(final: pd.DataFrame) -> pd.DataFrame:
    columns = [column for column in SUMMARY_COLUMNS if column in final.columns]
    grouped = (
        final.groupby(["agent", "module_variant"], as_index=False)
        .agg(
            n_seeds=("seed", "nunique"),
            **{f"{column}_mean": (column, "mean") for column in columns},
            **{f"{column}_std": (column, "std") for column in columns},
        )
    )
    return grouped.sort_values(["agent", "module_variant"])


def _audit(final: pd.DataFrame) -> dict[str, object]:
    underlay = final.get("underlay_transmissions", pd.Series(dtype=float))
    overlay = final.get("overlay_transmissions", pd.Series(dtype=float))
    underlay_ratio = final.get("underlay_selection_ratio", pd.Series(dtype=float))
    pu_violation = final.get("pu_access_violation_rate", pd.Series(dtype=float))
    return {
        "rows": int(len(final)),
        "agents": sorted(final["agent"].astype(str).unique().tolist()) if "agent" in final else [],
        "conditions": sorted(final["module_variant"].astype(str).unique().tolist()) if "module_variant" in final else [],
        "has_underlay_transmissions": bool((pd.to_numeric(underlay, errors="coerce") > 0).any()),
        "has_overlay_transmissions": bool((pd.to_numeric(overlay, errors="coerce") > 0).any()),
        "underlay_ratio_min": float(pd.to_numeric(underlay_ratio, errors="coerce").min()) if len(underlay_ratio) else None,
        "underlay_ratio_max": float(pd.to_numeric(underlay_ratio, errors="coerce").max()) if len(underlay_ratio) else None,
        "pu_access_violation_rate_max": float(pd.to_numeric(pu_violation, errors="coerce").max()) if len(pu_violation) else None,
    }


def _write_markdown(path: Path, summary: pd.DataFrame, audit: dict[str, object]) -> None:
    cols = [
        "agent",
        "module_variant",
        "n_seeds",
        "served_goodput_mbps_mean",
        "plr_mean",
        "underlay_selection_ratio_mean",
        "overlay_transmissions_mean",
        "underlay_transmissions_mean",
        "pu_access_violation_rate_mean",
    ]
    cols = [column for column in cols if column in summary.columns]
    markdown_rows = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in summary[cols].iterrows():
        values = []
        for column in cols:
            value = row[column]
            values.append(f"{value:.4f}" if isinstance(value, float) else str(value))
        markdown_rows.append("| " + " | ".join(values) + " |")
    lines = [
        "# D4-A Mixed-Mode Pilot",
        "",
        "## Audit",
        "",
        "```json",
        json.dumps(audit, indent=2),
        "```",
        "",
        "## Summary",
        "",
        "\n".join(markdown_rows),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    data = pd.read_csv(_metrics_path(args.run_dir))
    final = _final_rows(data)
    summary = _summary(final)
    audit = _audit(final)

    output_dir = args.output_dir or args.run_dir / "analysis" / "mixed_mode_pilot"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "mixed_mode_pilot_summary.csv", index=False)
    (output_dir / "mixed_mode_pilot_audit.json").write_text(
        json.dumps(audit, indent=2),
        encoding="utf-8",
    )
    _write_markdown(output_dir / "mixed_mode_pilot_summary.md", summary, audit)

    print(summary.to_string(index=False))
    print(json.dumps(audit, indent=2))
    print(f"wrote {output_dir / 'mixed_mode_pilot_summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
