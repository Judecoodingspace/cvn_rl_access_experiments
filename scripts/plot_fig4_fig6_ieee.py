"""Re-plot Figure 4 and Figure 6 with the project's strict IEEE typography.

Uses the IEEE font configuration from cvn_rl.paper_figure_style while retaining
the requested two-panel proportions and diagnostic colour palette.

User-requested differences from the original fig4/fig6:
- Titles and subplot labels placed below the figure
- All data lines solid (no dashed/dotted/dash-dot)
- No per-bar value annotations on bar charts
- Descriptive sub-figure and main titles
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from cvn_rl.paper_figure_style import (
    DIAGNOSTIC_PALETTE,
    FIG_IEEE_DIAGNOSTIC_TWO_PANEL,
    add_bottom_figure_title,
    add_bottom_subplot_title as _subtitle,
    adjust_diagnostic_two_panel_layout,
    apply_ieee_diagnostic_two_panel_style,
    save_ieee_diagnostic_two_panel_figure,
    style_full_frame as _show_full_frame,
)

OUTPUT_DIR = Path("reports/diagnostic_findings")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

apply_ieee_diagnostic_two_panel_style()

# ---- Identical colour palette as fig3/fig5 ----
COLORS = DIAGNOSTIC_PALETTE

# ---- Figure size: match the wide frame proportion used by fig3/fig5 ----
# Keep the current height so the below-axis subtitles and bottom main title
# retain their existing vertical placement and typography.
FIG_SIZE = FIG_IEEE_DIAGNOSTIC_TWO_PANEL

# ===========================================================================
# Load data
# ===========================================================================
BASE = pd.read_csv(
    "results/e2_prediction_sensing_redundancy_pilot_a/"
    "20260703_142409_e2_r1_oracle_gate_run/tables/final_checkpoint_results.csv"
)
DF_PD = pd.read_csv(
    "results/20260703_200605_e2_r1_oracle_gate_run/tables/final_checkpoint_results.csv"
)
SUP_3B = pd.read_csv(
    "results/20260705_113840_e2_r1_oracle_gate_run/tables/final_checkpoint_results.csv"
)
DF_PFA = pd.read_csv(
    "results/20260705_115119_e2_r1_oracle_gate_run/tables/final_checkpoint_results.csv"
)
SUP_4B = pd.read_csv(
    "results/20260705_133714_e2_r1_oracle_gate_run/tables/final_checkpoint_results.csv"
)

PRED_REF = BASE[BASE["module_variant"] == "prediction_only"]["served_goodput_mbps"].mean()
NEITH_REF = BASE[BASE["module_variant"] == "neither"]["served_goodput_mbps"].mean()


def _compute_pd_means(df):
    pd_map = {s: float(s.split("_")[-1].replace("p", "."))
              for s in df["scenario"].unique()}
    scenarios = sorted(df["scenario"].unique(), key=lambda s: pd_map[s], reverse=True)
    pd_vals, full_m, sens_m = [], [], []
    for sc in scenarios:
        sub = df[df["scenario"] == sc]
        full_m.append(sub[sub["module_variant"] == "full"]["served_goodput_mbps"].mean())
        sens_m.append(sub[sub["module_variant"] == "sensing_only"]["served_goodput_mbps"].mean())
        pd_vals.append(pd_map[sc])
    return pd_vals, full_m, sens_m


def _compute_pfa_means(df):
    pfa_map = {s: float(s.split("_")[-1].replace("p", "."))
               for s in df["scenario"].unique()}
    scenarios = sorted(df["scenario"].unique(), key=lambda s: pfa_map[s])
    pfa_vals, full_m, sens_m = [], [], []
    for sc in scenarios:
        sub = df[df["scenario"] == sc]
        full_m.append(sub[sub["module_variant"] == "full"]["served_goodput_mbps"].mean())
        sens_m.append(sub[sub["module_variant"] == "sensing_only"]["served_goodput_mbps"].mean())
        pfa_vals.append(pfa_map[sc])
    return pfa_vals, full_m, sens_m


def _lookup_pred_neither(sup_df, ref_pred, ref_neith, param_vals, prefix):
    pred_m, neith_m = [], []
    for pv in param_vals:
        scenario = f"P0_M1_{prefix}_{str(pv).replace('.', 'p')}"
        if scenario in sup_df["scenario"].unique():
            sub = sup_df[sup_df["scenario"] == scenario]
            pred_m.append(sub[sub["module_variant"] == "prediction_only"]["served_goodput_mbps"].mean())
            neith_m.append(sub[sub["module_variant"] == "neither"]["served_goodput_mbps"].mean())
        else:
            pred_m.append(ref_pred)
            neith_m.append(ref_neith)
    return pred_m, neith_m


# ===========================================================================
# Pre-compute
# ===========================================================================
pd_vals, full_pd, sens_pd = _compute_pd_means(DF_PD)
pfa_vals, full_pfa, sens_pfa = _compute_pfa_means(DF_PFA)
pred_pd_all, neith_pd_all = _lookup_pred_neither(SUP_3B, PRED_REF, NEITH_REF, pd_vals, "pd")
pred_pfa_all, neith_pfa_all = _lookup_pred_neither(SUP_4B, PRED_REF, NEITH_REF, pfa_vals, "pfa")


# ===========================================================================
# FIGURE 4: 2×2 ablation across Pd (detection probability) gradient
# ===========================================================================
fig4, (ax_a, ax_b) = plt.subplots(1, 2, figsize=FIG_SIZE)

# ---- Panel A: Served goodput of four information conditions vs Pd ----
ax_a.plot(pd_vals, full_pd, "o-", color=COLORS["full"], linewidth=2.5, markersize=9,
          label="Full (sensing + prediction)")
ax_a.plot(pd_vals, sens_pd, "s-", color=COLORS["sensing"], linewidth=2.5, markersize=9,
          label="Sensing-only")
ax_a.plot(pd_vals, pred_pd_all, "D-", color=COLORS["pred"], linewidth=2, markersize=8,
          label="Prediction-only (flat)")
ax_a.plot(pd_vals, neith_pd_all, "v-", color=COLORS["neither"], linewidth=2, markersize=8,
          label="Neither (flat)")

# Crossover marker
ax_a.axvline(x=0.75, color="black", linestyle=":", linewidth=1.5)

ax_a.set_xlabel("Sensing $P_d$ (Detection Probability), $P_{fa}=0.1$ fixed")
ax_a.set_ylabel("Served Goodput (Mbps)")
ax_a.set_xlim(1.08, 0.22)
ax_a.set_ylim(10, 30)
ax_a.legend(
    loc="upper left", ncol=2, columnspacing=0.8, handlelength=1.6,
)
ax_a.grid(True, alpha=0.3)
_show_full_frame(ax_a)

_subtitle(ax_a, "(a) Served goodput of four information conditions\n"
          "under varying detection probability $P_d$")

# ---- Panel B: Factorial interaction I(Pd) ----
I_pd = [f - s - p + n for f, s, p, n in zip(full_pd, sens_pd, pred_pd_all, neith_pd_all)]
bar_colors = [COLORS["delta_neg"]] * 3 + [COLORS["delta_pos"]] * 2
ax_b.bar(np.arange(len(pd_vals)), I_pd, color=bar_colors, edgecolor="white",
         linewidth=0.5, width=0.6)
ax_b.axhline(y=0, color="black", linewidth=1.5, linestyle="--")
ax_b.set_xticks(np.arange(len(pd_vals)))
ax_b.set_xticklabels([f"{v:.1f}" for v in pd_vals])
ax_b.set_xlabel("Sensing $P_d$")
ax_b.set_ylabel("Interaction $I = $ full $-$ sens $-$ pred + neither (Mbps)")
ax_b.set_ylim(-12.5, 3.5)
ax_b.yaxis.label.set_y(0.44)
ax_b.grid(True, alpha=0.3, axis="y")
_show_full_frame(ax_b)

_subtitle(ax_b, "(b) Factorial interaction $I(P_d)$\n"
          "= full $-$ sens $-$ pred + neither")

# Main title at bottom
add_bottom_figure_title(
    fig4,
    "Figure 4: Sensing–prediction $2\\times2$ factorial ablation across "
    "detection probability $P_d$ gradient",
)

adjust_diagnostic_two_panel_layout(fig4)
save_ieee_diagnostic_two_panel_figure(
    fig4, str(OUTPUT_DIR / "fig4_complete_factorial_I_evolution")
)
plt.close(fig4)
print(f"Figure 4 saved → {OUTPUT_DIR / 'fig4_complete_factorial_I_evolution.pdf'}")


# ===========================================================================
# FIGURE 6: 2×2 ablation across Pfa (false-alarm rate) gradient
# ===========================================================================
fig6, (ax_a6, ax_b6) = plt.subplots(1, 2, figsize=FIG_SIZE)

# ---- Panel A: Served goodput of four information conditions vs Pfa ----
ax_a6.plot(pfa_vals, full_pfa, "o-", color=COLORS["full"], linewidth=2.5, markersize=9,
           label="Full (sensing + prediction)")
ax_a6.plot(pfa_vals, sens_pfa, "s-", color=COLORS["sensing"], linewidth=2.5, markersize=9,
           label="Sensing-only")
ax_a6.plot(pfa_vals, pred_pfa_all, "D-", color=COLORS["pred"], linewidth=2, markersize=8,
           label="Prediction-only (flat)")
ax_a6.plot(pfa_vals, neith_pfa_all, "v-", color=COLORS["neither"], linewidth=2, markersize=8,
           label="Neither (flat)")

ax_a6.axvline(x=0.15, color="black", linestyle=":", linewidth=1.5)

ax_a6.set_xlabel("Sensing $P_{fa}$ (False Alarm Rate), $P_d=0.9$ fixed")
ax_a6.set_ylabel("Served Goodput (Mbps)")
ax_a6.set_xlim(-0.02, 0.55)
ax_a6.set_ylim(10, 26.5)
ax_a6.legend(
    loc="lower left", bbox_to_anchor=(0.02, 0.02), ncol=2,
    columnspacing=0.8, handlelength=1.6, borderaxespad=0.0,
)
ax_a6.grid(True, alpha=0.3)
_show_full_frame(ax_a6)

_subtitle(ax_a6, "(a) Served goodput of four information conditions\n"
          "under varying false-alarm rate $P_{fa}$")

# ---- Panel B: Factorial interaction I(Pfa) ----
I_pfa = [f - s - p + n for f, s, p, n in zip(full_pfa, sens_pfa, pred_pfa_all, neith_pfa_all)]
ax_b6.bar(np.arange(len(pfa_vals)), I_pfa, color=COLORS["delta_neg"], edgecolor="white",
          linewidth=0.5, width=0.6)
ax_b6.axhline(y=0, color="black", linewidth=1.5, linestyle="--")
ax_b6.set_xticks(np.arange(len(pfa_vals)))
ax_b6.set_xticklabels([f"{v:.2f}" for v in pfa_vals])
ax_b6.set_xlabel("Sensing $P_{fa}$")
ax_b6.set_ylabel("Interaction $I = $ full $-$ sens $-$ pred + neither (Mbps)")
ax_b6.set_ylim(-12.5, 3.5)
ax_b6.yaxis.label.set_y(0.44)
ax_b6.grid(True, alpha=0.3, axis="y")
_show_full_frame(ax_b6)

_subtitle(ax_b6, "(b) Factorial interaction $I(P_{fa})$\n"
          "= full $-$ sens $-$ pred + neither")

# Main title at bottom
add_bottom_figure_title(
    fig6,
    "Figure 6: Sensing–prediction $2\\times2$ factorial ablation across "
    "false-alarm rate $P_{fa}$ gradient",
)

adjust_diagnostic_two_panel_layout(fig6)
save_ieee_diagnostic_two_panel_figure(
    fig6, str(OUTPUT_DIR / "fig6_pfa_complete_factorial_I_evolution")
)
plt.close(fig6)
print(f"Figure 6 saved → {OUTPUT_DIR / 'fig6_pfa_complete_factorial_I_evolution.pdf'}")

print("\nDone. Both figures use IEEE typography and balanced two-panel proportions.")
