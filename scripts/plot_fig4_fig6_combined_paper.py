from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from matplotlib.transforms import Bbox

from cvn_rl.paper_figure_style import (
    DIAGNOSTIC_PALETTE,
    apply_ieee_style,
    save_ieee_figure,
    style_full_frame,
)

OUTPUT_DIR = Path("paper_figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FIG_SIZE = (7.10, 5.10)
COLORS = DIAGNOSTIC_PALETTE


def _compute_pd_means(df: pd.DataFrame) -> tuple[list[float], list[float], list[float]]:
    pd_map = {
        scenario: float(scenario.split("_")[-1].replace("p", "."))
        for scenario in df["scenario"].unique()
    }
    scenarios = sorted(df["scenario"].unique(), key=lambda scenario: pd_map[scenario], reverse=True)
    pd_vals, full_m, sens_m = [], [], []
    for scenario in scenarios:
        sub = df[df["scenario"] == scenario]
        full_m.append(sub[sub["module_variant"] == "full"]["served_goodput_mbps"].mean())
        sens_m.append(sub[sub["module_variant"] == "sensing_only"]["served_goodput_mbps"].mean())
        pd_vals.append(pd_map[scenario])
    return pd_vals, full_m, sens_m


def _compute_pfa_means(df: pd.DataFrame) -> tuple[list[float], list[float], list[float]]:
    pfa_map = {
        scenario: float(scenario.split("_")[-1].replace("p", "."))
        for scenario in df["scenario"].unique()
    }
    scenarios = sorted(df["scenario"].unique(), key=lambda scenario: pfa_map[scenario])
    pfa_vals, full_m, sens_m = [], [], []
    for scenario in scenarios:
        sub = df[df["scenario"] == scenario]
        full_m.append(sub[sub["module_variant"] == "full"]["served_goodput_mbps"].mean())
        sens_m.append(sub[sub["module_variant"] == "sensing_only"]["served_goodput_mbps"].mean())
        pfa_vals.append(pfa_map[scenario])
    return pfa_vals, full_m, sens_m


def _lookup_pred_neither(
    sup_df: pd.DataFrame,
    ref_pred: float,
    ref_neither: float,
    param_vals: list[float],
    prefix: str,
) -> tuple[list[float], list[float]]:
    pred_m, neither_m = [], []
    for param_val in param_vals:
        scenario = f"P0_M1_{prefix}_{str(param_val).replace('.', 'p')}"
        if scenario in sup_df["scenario"].unique():
            sub = sup_df[sup_df["scenario"] == scenario]
            pred_m.append(sub[sub["module_variant"] == "prediction_only"]["served_goodput_mbps"].mean())
            neither_m.append(sub[sub["module_variant"] == "neither"]["served_goodput_mbps"].mean())
        else:
            pred_m.append(ref_pred)
            neither_m.append(ref_neither)
    return pred_m, neither_m


def _add_panel_tags_below(fig: plt.Figure, axes: list[plt.Axes], tags: list[str], pad_points: float = 4.0) -> None:
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
        fig.text(
            x_fig,
            y_fig,
            tag,
            ha="center",
            va="top",
            fontweight="bold",
            fontsize=9.0,
        )


def main() -> None:
    apply_ieee_style()
    plt.rcParams.update(
        {
            "font.size": 8.75,
            "axes.labelsize": 8.75,
            "xtick.labelsize": 7.75,
            "ytick.labelsize": 7.75,
            "legend.fontsize": 7.8,
            "lines.linewidth": 1.45,
            "lines.markersize": 4.8,
            "lines.markeredgewidth": 0.65,
        }
    )

    base = pd.read_csv(
        "results/e2_prediction_sensing_redundancy_pilot_a/"
        "20260703_142409_e2_r1_oracle_gate_run/tables/final_checkpoint_results.csv"
    )
    df_pd = pd.read_csv(
        "results/20260703_200605_e2_r1_oracle_gate_run/tables/final_checkpoint_results.csv"
    )
    sup_3b = pd.read_csv(
        "results/20260705_113840_e2_r1_oracle_gate_run/tables/final_checkpoint_results.csv"
    )
    df_pfa = pd.read_csv(
        "results/20260705_115119_e2_r1_oracle_gate_run/tables/final_checkpoint_results.csv"
    )
    sup_4b = pd.read_csv(
        "results/20260705_133714_e2_r1_oracle_gate_run/tables/final_checkpoint_results.csv"
    )

    pred_ref = base[base["module_variant"] == "prediction_only"]["served_goodput_mbps"].mean()
    neither_ref = base[base["module_variant"] == "neither"]["served_goodput_mbps"].mean()

    pd_vals, full_pd, sens_pd = _compute_pd_means(df_pd)
    pfa_vals, full_pfa, sens_pfa = _compute_pfa_means(df_pfa)
    pred_pd_all, neither_pd_all = _lookup_pred_neither(
        sup_3b, pred_ref, neither_ref, pd_vals, "pd"
    )
    pred_pfa_all, neither_pfa_all = _lookup_pred_neither(
        sup_4b, pred_ref, neither_ref, pfa_vals, "pfa"
    )

    i_pd = [
        full - sens - pred + neither
        for full, sens, pred, neither in zip(full_pd, sens_pd, pred_pd_all, neither_pd_all)
    ]
    i_pfa = [
        full - sens - pred + neither
        for full, sens, pred, neither in zip(full_pfa, sens_pfa, pred_pfa_all, neither_pfa_all)
    ]

    fig, axes = plt.subplots(2, 2, figsize=FIG_SIZE)
    ((ax_a, ax_b), (ax_c, ax_d)) = axes

    line_full, = ax_a.plot(
        pd_vals, full_pd, "o-", color=COLORS["full"],
        label="Full", linewidth=1.7, markersize=5.2,
    )
    line_sensing, = ax_a.plot(
        pd_vals, sens_pd, "s-", color=COLORS["sensing"],
        label="Sensing-only", linewidth=1.7, markersize=5.0,
    )
    line_pred, = ax_a.plot(
        pd_vals, pred_pd_all, "D-", color=COLORS["pred"],
        label="Prediction-only", linewidth=1.5, markersize=4.9,
    )
    line_neither, = ax_a.plot(
        pd_vals, neither_pd_all, "v-", color=COLORS["neither"],
        label="Neither", linewidth=1.5, markersize=4.9,
    )
    ax_a.axvline(x=0.75, color="black", linestyle=":", linewidth=1.2)
    ax_a.set_xlabel("Sensing $P_d$")
    ax_a.set_ylabel("Served Goodput (Mbps)")
    ax_a.set_xlim(1.08, 0.22)
    ax_a.set_ylim(10, 30)
    ax_a.set_xticks(pd_vals)
    ax_a.set_xticklabels([f"{value:.1f}" for value in pd_vals])
    ax_a.grid(True, alpha=0.35)
    style_full_frame(ax_a)

    pd_bar_colors = [
        COLORS["delta_pos"] if value > 0 else COLORS["delta_neg"]
        for value in i_pd
    ]
    ax_b.bar(
        np.arange(len(pd_vals)),
        i_pd,
        color=pd_bar_colors,
        edgecolor="white",
        linewidth=0.5,
        width=0.62,
    )
    ax_b.axhline(y=0, color="black", linewidth=1.1, linestyle="--")
    ax_b.set_xticks(np.arange(len(pd_vals)))
    ax_b.set_xticklabels([f"{value:.1f}" for value in pd_vals])
    ax_b.set_xlabel("Sensing $P_d$")
    ax_b.set_ylabel("Interaction $I$ (Mbps)")
    ax_b.set_ylim(-12.5, 3.5)
    ax_b.grid(True, alpha=0.3, axis="y")
    style_full_frame(ax_b)

    ax_c.plot(
        pfa_vals, full_pfa, "o-", color=COLORS["full"],
        linewidth=1.7, markersize=5.2,
    )
    ax_c.plot(
        pfa_vals, sens_pfa, "s-", color=COLORS["sensing"],
        linewidth=1.7, markersize=5.0,
    )
    ax_c.plot(
        pfa_vals, pred_pfa_all, "D-", color=COLORS["pred"],
        linewidth=1.5, markersize=4.9,
    )
    ax_c.plot(
        pfa_vals, neither_pfa_all, "v-", color=COLORS["neither"],
        linewidth=1.5, markersize=4.9,
    )
    ax_c.axvline(x=0.15, color="black", linestyle=":", linewidth=1.2)
    ax_c.set_xlabel("Sensing $P_{fa}$")
    ax_c.set_ylabel("Served Goodput (Mbps)")
    ax_c.set_xlim(-0.02, 0.55)
    ax_c.set_ylim(10, 26.5)
    ax_c.set_xticks(pfa_vals)
    ax_c.set_xticklabels([f"{value:.2f}" for value in pfa_vals])
    ax_c.grid(True, alpha=0.35)
    style_full_frame(ax_c)

    pfa_bar_colors = [
        COLORS["delta_pos"] if value > 0 else COLORS["delta_neg"]
        for value in i_pfa
    ]
    ax_d.bar(
        np.arange(len(pfa_vals)),
        i_pfa,
        color=pfa_bar_colors,
        edgecolor="white",
        linewidth=0.5,
        width=0.62,
    )
    ax_d.axhline(y=0, color="black", linewidth=1.1, linestyle="--")
    ax_d.set_xticks(np.arange(len(pfa_vals)))
    ax_d.set_xticklabels([f"{value:.2f}" for value in pfa_vals])
    ax_d.set_xlabel("Sensing $P_{fa}$")
    ax_d.set_ylabel("Interaction $I$ (Mbps)")
    ax_d.set_ylim(-12.5, 3.5)
    ax_d.grid(True, alpha=0.3, axis="y")
    style_full_frame(ax_d)

    fig.legend(
        [line_full, line_sensing, line_pred, line_neither],
        ["Full", "Sensing-only", "Prediction-only", "Neither"],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.935),
        ncol=4,
        frameon=True,
        columnspacing=1.0,
        handlelength=1.6,
        borderaxespad=0.0,
    )
    fig.subplots_adjust(
        left=0.085,
        right=0.99,
        bottom=0.17,
        top=0.82,
        wspace=0.25,
        hspace=0.46,
    )
    _add_panel_tags_below(fig, [ax_a, ax_b, ax_c, ax_d], ["(a)", "(b)", "(c)", "(d)"], pad_points=4.0)

    save_ieee_figure(
        fig,
        str(OUTPUT_DIR / "fig4_fig6_combined_factorial_evolution"),
        formats=("pdf", "png"),
        tight=False,
    )
    plt.close(fig)
    print(f"Combined figure saved -> {OUTPUT_DIR / 'fig4_fig6_combined_factorial_evolution.pdf'}")


if __name__ == "__main__":
    main()
