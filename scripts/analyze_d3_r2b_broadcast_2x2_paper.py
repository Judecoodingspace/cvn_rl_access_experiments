from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.transforms import Bbox

from cvn_rl.paper_figure_style import (
    FIG_IEEE_DOUBLE,
    PALETTE,
    apply_ieee_style,
    save_ieee_figure,
)


U_RE = re.compile(r"_u(?P<u>\d+)$")


def _latest_metrics_csv(run_dir: Path) -> Path:
    candidates = sorted(run_dir.rglob("access_metrics.csv"))
    if not candidates:
        candidates = sorted(run_dir.rglob("*metrics*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No metrics CSV found under {run_dir}")
    return candidates[-1]


def _parse_update_slots(name: str) -> int | None:
    match = U_RE.search(str(name))
    if not match:
        return None
    return int(match.group("u"))


def _add_panel_tags_below(fig, axes, tags, pad_points=4.0):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    for ax, tag in zip(axes, tags):
        bboxes = [ax.xaxis.label.get_window_extent(renderer=renderer)]
        tick_bboxes = [
            tick.get_window_extent(renderer=renderer)
            for tick in ax.get_xticklabels()
            if tick.get_text()
        ]
        if tick_bboxes:
            bboxes.extend(tick_bboxes)
        anchor = Bbox.union(bboxes)
        x_disp = (ax.bbox.x0 + ax.bbox.x1) / 2.0
        y_disp = anchor.y0 - pad_points * fig.dpi / 72.0
        x_fig, y_fig = inv.transform((x_disp, y_disp))
        fig.text(x_fig, y_fig, tag, ha="center", va="top", fontweight="bold", fontsize=8.0)


def build_summary(data: pd.DataFrame, *, dqi_bits_per_channel: float = 8.0) -> pd.DataFrame:
    if "is_final_checkpoint" in data.columns:
        data = data[data["is_final_checkpoint"].astype(bool)].copy()
    data["dqi_update_slots"] = data["module_variant"].map(_parse_update_slots)
    data = data[data["dqi_update_slots"].notna()].copy()
    data["dqi_update_slots"] = data["dqi_update_slots"].astype(int)
    if data.empty:
        raise ValueError("No dqi_attention_frozen_u* final rows found")

    num_channels = float(data["num_channels"].iloc[0]) if "num_channels" in data.columns else 10.0
    slot_ms = 1.0
    if "slot_duration_ms" in data.columns:
        slot_ms = float(data["slot_duration_ms"].iloc[0])
    data["dqi_broadcast_overhead_kbps_derived"] = (
        num_channels * dqi_bits_per_channel / (data["dqi_update_slots"] * slot_ms)
    )

    agg = (
        data.groupby(["module_variant", "dqi_update_slots"], as_index=False)
        .agg(
            served_goodput_mbps_mean=("served_goodput_mbps", "mean"),
            served_goodput_mbps_std=("served_goodput_mbps", "std"),
            plr_mean=("plr", "mean"),
            plr_std=("plr", "std"),
            pu_access_violation_rate_mean=("pu_access_violation_rate", "mean"),
            pu_access_violation_rate_std=("pu_access_violation_rate", "std"),
            broadcast_overhead_kbps_recorded=("broadcast_overhead_kbps", "mean"),
            dqi_broadcast_overhead_kbps_mean=("dqi_broadcast_overhead_kbps_derived", "mean"),
            rolling_100ms_system_served_q10_mean=("rolling_100ms_system_served_q10_mean", "mean"),
            switch_count_mean=("switch_count", "mean"),
            seeds=("seed", "nunique"),
        )
        .sort_values("dqi_update_slots")
    )
    return agg


def plot(summary: pd.DataFrame, output_stem: Path) -> None:
    apply_ieee_style()
    fig, axes_grid = plt.subplots(2, 2, figsize=(5.55, 4.95))
    axes = axes_grid.ravel()
    x = summary["dqi_update_slots"]

    panels = [
        ("served_goodput_mbps_mean", "Goodput (Mbps)", PALETTE["blue"]),
        ("plr_mean", "PLR", PALETTE["orange"]),
        ("pu_access_violation_rate_mean", "PU access\nviolation rate", PALETTE["red"]),
        ("dqi_broadcast_overhead_kbps_mean", "DQI broadcast\noverhead (kbps)", PALETTE["green"]),
    ]
    for ax, (col, ylabel, color) in zip(axes, panels):
        ax.plot(x, summary[col], marker="o", color=color)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("DQI update\nperiod (slots)")
        ax.set_xscale("log")
        ax.grid(True, alpha=0.35)

    for ax in axes:
        ax.tick_params(axis="x", which="both", bottom=False, top=False)
        ax.set_box_aspect(1.0)

    fig.subplots_adjust(left=0.090, right=0.982, bottom=0.125, top=0.935, wspace=0.120, hspace=0.460)
    _add_panel_tags_below(fig, axes, ("(a)", "(b)", "(c)", "(d)"), pad_points=3.2)
    for ext in ("pdf", "png", "jpg"):
        fig.savefig(output_stem.with_suffix(f".{ext}"), dpi=300, bbox_inches="tight", pad_inches=0.035)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--paper-output-stem", type=Path, default=None)
    parser.add_argument("--dqi-bits-per-channel", type=float, default=8.0)
    args = parser.parse_args()

    run = args.run_dir
    out = args.output_dir or (run / "analysis")
    fig_dir = run / "figures"
    table_dir = run / "tables"
    out.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    metrics = pd.read_csv(_latest_metrics_csv(run))
    summary = build_summary(metrics, dqi_bits_per_channel=args.dqi_bits_per_channel)
    summary_path = table_dir / "d3_r2b_broadcast_sensitivity.csv"
    summary.to_csv(summary_path, index=False)

    audit = {
        "passed": bool(set(summary["dqi_update_slots"]) == {1, 10, 50, 100, 500}),
        "update_slots": summary["dqi_update_slots"].tolist(),
        "dqi_bits_per_channel": args.dqi_bits_per_channel,
        "note": (
            "The plotted overhead is the DQI descriptor payload overhead, derived as "
            "num_channels * bits_per_channel / update_slots for 1 ms slots. This is "
            "the DQI analogue of P3-B theta broadcast overhead."
        ),
    }
    (out / "d3_r2b_broadcast_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")

    stem = fig_dir / "d3_r2b_dqi_broadcast_sensitivity_2x2"
    plot(summary, stem)
    paper_stem = args.paper_output_stem or Path("paper_figures") / "fig_d3_r2b_dqi_broadcast_sensitivity_2x2"
    paper_stem.parent.mkdir(parents=True, exist_ok=True)
    plot(summary, paper_stem)

    print(f"wrote {summary_path}")
    print(f"wrote {stem}.pdf")
    print(f"wrote {paper_stem}.pdf")
    print(f"AUDIT: {'PASSED' if audit['passed'] else 'FAILED'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
