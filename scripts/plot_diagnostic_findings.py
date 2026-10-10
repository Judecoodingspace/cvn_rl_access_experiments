"""Plot key findings from the three diagnostic experiments.

Figure 1: Sensing precision gradient — crossover point
Figure 2: U-staleness effect — bar chart comparison
Figure 3: Full robustness vs sensing collapse
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from pathlib import Path

OUTPUT_DIR = Path("reports/diagnostic_findings")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Style
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

COLORS = {
    "full": "#2196F3",
    "sensing": "#FF9800",
    "delta_pos": "#4CAF50",
    "delta_neg": "#F44336",
    "baseline": "#9E9E9E",
    "diag1": "#FF5722",
    "diag2": "#4CAF50",
}

# ===========================================================================
# Load data
# ===========================================================================
df_grad = pd.read_csv(
    "results/20260703_200605_e2_r1_oracle_gate_run/tables/final_checkpoint_results.csv"
)

# ===========================================================================
# FIGURE 1: Sensing precision gradient — the crossover
# ===========================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

pd_map = {
    s: float(s.split("_")[-1].replace("p", "."))
    for s in df_grad["scenario"].unique()
}
scenarios_sorted = sorted(
    df_grad["scenario"].unique(), key=lambda s: pd_map[s], reverse=True
)

pd_vals = []
full_means, full_stds = [], []
sens_means, sens_stds = [], []
deltas, delta_stds = [], []

for scenario in scenarios_sorted:
    sub = df_grad[df_grad["scenario"] == scenario]
    full_sub = sub[sub["module_variant"] == "full"]
    sens_sub = sub[sub["module_variant"] == "sensing_only"]

    pd_val = pd_map[scenario]
    pd_vals.append(pd_val)
    full_means.append(full_sub["served_goodput_mbps"].mean())
    full_stds.append(full_sub["served_goodput_mbps"].std())
    sens_means.append(sens_sub["served_goodput_mbps"].mean())
    sens_stds.append(sens_sub["served_goodput_mbps"].std())

    # Per-seed deltas
    seed_deltas = []
    for s in [1, 2, 3]:
        fv = full_sub.loc[full_sub["seed"] == s, "served_goodput_mbps"].values
        sv = sens_sub.loc[sens_sub["seed"] == s, "served_goodput_mbps"].values
        if len(fv) and len(sv):
            seed_deltas.append(fv[0] - sv[0])
    deltas.append(np.mean(seed_deltas))
    delta_stds.append(np.std(seed_deltas))

# Left panel: absolute goodput
ax1.plot(pd_vals, full_means, "o-", color=COLORS["full"], linewidth=2, markersize=8, label="Full (sensing + prediction)")
ax1.fill_between(pd_vals, np.array(full_means) - np.array(full_stds), np.array(full_means) + np.array(full_stds), color=COLORS["full"], alpha=0.15)
ax1.plot(pd_vals, sens_means, "s--", color=COLORS["sensing"], linewidth=2, markersize=8, label="Sensing-only")
ax1.fill_between(pd_vals, np.array(sens_means) - np.array(sens_stds), np.array(sens_means) + np.array(sens_stds), color=COLORS["sensing"], alpha=0.15)

ax1.set_xlabel("Sensing $P_d$ (probability of detection)")
ax1.set_ylabel("Served Goodput (Mbps)")
ax1.set_title("(a) Absolute Performance")
ax1.legend(loc="lower left")
ax1.set_ylim(10, 30)
ax1.set_xlim(1.05, 0.25)
ax1.grid(True, alpha=0.3)

# Right panel: Delta (full - sensing)
colors_delta = [COLORS["delta_pos"] if d > 0 else COLORS["delta_neg"] for d in deltas]
ax2.bar(np.arange(len(pd_vals)), deltas, color=colors_delta, edgecolor="white", linewidth=0.5)
ax2.errorbar(np.arange(len(pd_vals)), deltas, yerr=delta_stds, fmt="none", ecolor="black", capsize=5)
ax2.axhline(y=0, color="black", linewidth=1, linestyle="-")
ax2.set_xticks(np.arange(len(pd_vals)))
ax2.set_xticklabels([f"{v:.1f}" for v in pd_vals])
ax2.set_xlabel("Sensing $P_d$")
ax2.set_ylabel("$\\Delta$ Goodput: Full $-$ Sensing-only (Mbps)")
ax2.set_title("(b) Incremental Value of Prediction")

# Crossover annotation
ax2.annotate(
    "Crossover\n$P_d \\approx 0.7$",
    xy=(2, deltas[2]),
    xytext=(2.8, deltas[2] + 2),
    arrowprops=dict(arrowstyle="->", color="black"),
    fontsize=10,
    fontweight="bold",
)
ax2.grid(True, alpha=0.3, axis="y")

fig.suptitle("Figure 1: Prediction compensates for sensing degradation", fontweight="bold", y=1.02)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "fig1_sensing_precision_gradient.pdf")
fig.savefig(OUTPUT_DIR / "fig1_sensing_precision_gradient.png")
plt.close()
print("Figure 1 saved.")

# ===========================================================================
# FIGURE 2: U-staleness effect — sign flip
# ===========================================================================
fig, ax = plt.subplots(figsize=(8, 5))

conditions = ["U=10 (baseline)\nstale prediction", "U=1 (diagnostic 2)\nfresh prediction"]
full_vals = [24.552, 25.371]
sens_vals = [25.087, 25.087]
deltas_u = [full_vals[0] - sens_vals[0], full_vals[1] - sens_vals[1]]

x = np.arange(len(conditions))
width = 0.35

bars1 = ax.bar(x - width / 2, full_vals, width, label="Full (sensing + prediction)", color=COLORS["full"], edgecolor="white")
bars2 = ax.bar(x + width / 2, sens_vals, width, label="Sensing-only", color=COLORS["sensing"], edgecolor="white")

# Add value labels
for bar in bars1:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2, f'{bar.get_height():.1f}', ha="center", va="bottom", fontsize=10, fontweight="bold")
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2, f'{bar.get_height():.1f}', ha="center", va="bottom", fontsize=10)

# Delta annotations
for i, (d, fv, sv) in enumerate(zip(deltas_u, full_vals, sens_vals)):
    color = COLORS["delta_pos"] if d > 0 else COLORS["delta_neg"]
    y = max(fv, sv) + 1.0
    ax.annotate(
        f"$\\Delta = {d:+.2f}$",
        xy=(x[i], y),
        fontsize=12,
        fontweight="bold",
        color=color,
        ha="center",
    )

ax.set_xticks(x)
ax.set_xticklabels(conditions)
ax.set_ylabel("Served Goodput (Mbps)")
ax.set_title("Figure 2: U=1 flips the sign — prediction becomes net positive", fontweight="bold")
ax.legend(loc="upper left")
ax.set_ylim(22, 28)
ax.grid(True, alpha=0.3, axis="y")

# Add "FAIL" / "PASS" badges
ax.text(0, 27.3, "SENSING\nDOMINATES", ha="center", fontsize=10, color=COLORS["delta_neg"], fontweight="bold", bbox=dict(boxstyle="round", facecolor="#FFEBEE", alpha=0.8))
ax.text(1, 27.3, "FULL\nDOMINATES", ha="center", fontsize=10, color=COLORS["delta_pos"], fontweight="bold", bbox=dict(boxstyle="round", facecolor="#E8F5E9", alpha=0.8))

plt.tight_layout()
fig.savefig(OUTPUT_DIR / "fig2_u_staleness_effect.pdf")
fig.savefig(OUTPUT_DIR / "fig2_u_staleness_effect.png")
plt.close()
print("Figure 2 saved.")

# ===========================================================================
# FIGURE 3: Three-panel summary — root cause decomposition
# ===========================================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
width = 0.35

# ---- Panel A: Availability flag leakage ----
conditions_flag = [
    "State includes\nsensing_available flag\n(baseline: agent sees\nwhich source is active)",
    "State excludes\nsensing_available flag\n(diagnostic 1: agent sees\nonly raw sensor values)",
]
full_flag = [24.552, 24.651]
sens_flag = [25.087, 24.613]
deltas_flag = [full_flag[0] - sens_flag[0], full_flag[1] - sens_flag[1]]

x_flag = np.arange(len(conditions_flag))
axes[0].bar(x_flag - width / 2, full_flag, width, label="Full (sensing + prediction)", color=COLORS["full"], edgecolor="white")
axes[0].bar(x_flag + width / 2, sens_flag, width, label="Sensing-only", color=COLORS["sensing"], edgecolor="white")
for i, d in enumerate(deltas_flag):
    color = COLORS["delta_pos"] if d > 0 else COLORS["delta_neg"]
    axes[0].annotate(
        f"$\\Delta={d:+.3f}$",
        xy=(x_flag[i], max(full_flag[i], sens_flag[i]) + 0.5),
        fontsize=11, fontweight="bold", color=color, ha="center",
    )
axes[0].set_xticks(x_flag)
axes[0].set_xticklabels(conditions_flag, fontsize=8)
axes[0].set_ylabel("Served Goodput (Mbps)")
axes[0].set_title(
    "(a) Remove availability flags from state vector\n"
    "$\\Delta$(full$-$sensing) improves by +0.57 Mbps",
    fontsize=10,
)
axes[0].set_ylim(23.3, 26.2)
axes[0].grid(True, alpha=0.3, axis="y")
axes[0].legend(fontsize=8, loc="lower left")

# ---- Panel B: U staleness ----
x_u = np.arange(2)
axes[1].bar(x_u - width / 2, full_vals, width, label="Full (sensing + prediction)", color=COLORS["full"], edgecolor="white")
axes[1].bar(x_u + width / 2, sens_vals, width, label="Sensing-only", color=COLORS["sensing"], edgecolor="white")
for i, d in enumerate(deltas_u):
    color = COLORS["delta_pos"] if d > 0 else COLORS["delta_neg"]
    axes[1].annotate(
        f"$\\Delta={d:+.3f}$",
        xy=(x_u[i], max(full_vals[i], sens_vals[i]) + 0.5),
        fontsize=11, fontweight="bold", color=color, ha="center",
    )
axes[1].set_xticks(x_u)
axes[1].set_xticklabels(
    ["U = 10 slots\n(prediction stale\nup to 9 slots old)",
     "U = 1 slot\n(prediction fresh,\nupdated every slot)"],
    fontsize=8,
)
axes[1].set_title(
    "(b) Reduce broadcast update interval U\n"
    "$\\Delta$(full$-$sensing) improves by +0.82 Mbps",
    fontsize=10,
)
axes[1].set_ylim(23.3, 26.2)
axes[1].grid(True, alpha=0.3, axis="y")
axes[1].legend(fontsize=8, loc="lower left")

# ---- Panel C: Sensing precision gradient (robustness) ----
axes[2].plot(
    pd_vals, full_means, "o-", color=COLORS["full"],
    linewidth=2.5, markersize=8, label="Full (sensing + prediction)",
)
axes[2].fill_between(
    pd_vals,
    np.array(full_means) - np.array(full_stds),
    np.array(full_means) + np.array(full_stds),
    color=COLORS["full"], alpha=0.2,
    label="Full $\\pm 1\\sigma$ (seed variation)",
)
axes[2].plot(
    pd_vals, sens_means, "s--", color=COLORS["sensing"],
    linewidth=2.5, markersize=8, label="Sensing-only",
)
axes[2].fill_between(
    pd_vals,
    np.array(sens_means) - np.array(sens_stds),
    np.array(sens_means) + np.array(sens_stds),
    color=COLORS["sensing"], alpha=0.2,
    label="Sensing $\\pm 1\\sigma$ (seed variation)",
)
# Crossover line
axes[2].axvline(x=0.75, color="black", linestyle=":", linewidth=1.5)
axes[2].annotate(
    "Crossover\n$P_d \\approx 0.7$",
    xy=(0.72, 22), fontsize=9, fontweight="bold", ha="center",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
)
# Region labels
axes[2].annotate(
    "Sensing\nsufficient",
    xy=(0.95, 28.5), fontsize=9, ha="center",
    color=COLORS["delta_neg"], fontstyle="italic",
)
axes[2].annotate(
    "Prediction\ncompensates",
    xy=(0.4, 28.5), fontsize=9, ha="center",
    color=COLORS["delta_pos"], fontstyle="italic",
)

axes[2].set_xlabel(
    "Sensing $P_d$  —  Probability of Detection\n"
    "(detection rate when PU is transmitting; false-alarm $P_{fa}=0.1$ fixed)",
    fontsize=9,
)
axes[2].set_xlim(1.08, 0.22)
axes[2].set_ylabel("Served Goodput (Mbps)")
axes[2].set_title(
    "(c) Prediction provides robustness against sensing degradation\n"
    "Full goodput varies only 1.2 Mbps across the full $P_d$ range",
    fontsize=10,
)
axes[2].legend(fontsize=8, loc="lower left")
axes[2].set_ylim(10, 31)
axes[2].grid(True, alpha=0.3)

fig.suptitle(
    "Figure 3: Root-cause decomposition — three factors behind the 2$\\times$2 factorial failure",
    fontweight="bold", fontsize=13,
)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "fig3_root_cause_decomposition.pdf")
fig.savefig(OUTPUT_DIR / "fig3_root_cause_decomposition.png")
plt.close()
print("Figure 3 saved (annotated).")

print(f"\nAll figures saved to {OUTPUT_DIR.resolve()}")

# ===========================================================================
# FIGURE 4: Complete 4-condition Pd gradient (with prediction_only & neither)
# ===========================================================================
# Load supplement data (prediction_only + neither at Pd=0.5, 0.3)
sup = pd.read_csv(
    "results/20260705_113840_e2_r1_oracle_gate_run/tables/final_checkpoint_results.csv"
)
# Baseline provides prediction_only & neither at Pd=0.9
base = pd.read_csv(
    "results/e2_prediction_sensing_redundancy_pilot_a/"
    "20260703_142409_e2_r1_oracle_gate_run/tables/final_checkpoint_results.csv"
)
pred_base_val = base[base["module_variant"] == "prediction_only"][
    "served_goodput_mbps"
].mean()
neither_base_val = base[base["module_variant"] == "neither"][
    "served_goodput_mbps"
].mean()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

# Panel A: All 4 conditions
pd_vals_all = [1.0, 0.9, 0.7, 0.5, 0.3]
full_all = [24.541, 24.552, 23.365, 23.937, 24.016]
sens_all = [25.586, 25.087, 22.822, 20.254, 14.962]
# pred and neither are flat (empirically verified)
pred_all = [pred_base_val] * 5
neither_all = [neither_base_val] * 5

ax1.plot(pd_vals_all, full_all, "o-", color=COLORS["full"], linewidth=2.5, markersize=9,
         label="Full (sensing + prediction)")
ax1.plot(pd_vals_all, sens_all, "s--", color=COLORS["sensing"], linewidth=2.5, markersize=9,
         label="Sensing-only")
ax1.plot(pd_vals_all, pred_all, "D-.", color="#4CAF50", linewidth=2, markersize=8,
         label="Prediction-only (flat: no sensing used)")
ax1.plot(pd_vals_all, neither_all, "v:", color="#9E9E9E", linewidth=2, markersize=8,
         label="Neither (flat: no information)")

# Crossover shading
ax1.axvspan(0.25, 0.75, alpha=0.06, color=COLORS["delta_pos"])
ax1.axvspan(0.75, 1.05, alpha=0.06, color=COLORS["delta_neg"])
ax1.annotate("Prediction\ncompensates", xy=(0.4, 28.5), fontsize=9, ha="center",
             color=COLORS["delta_pos"], fontstyle="italic", fontweight="bold")
ax1.annotate("Sensing\nsufficient", xy=(0.9, 28.5), fontsize=9, ha="center",
             color=COLORS["delta_neg"], fontstyle="italic", fontweight="bold")
ax1.axvline(x=0.75, color="black", linestyle=":", linewidth=1.5)
ax1.annotate("Crossover\n$P_d \\approx 0.7$", xy=(0.68, 14), fontsize=10,
             fontweight="bold", ha="center",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9))

ax1.set_xlabel("Sensing $P_d$ (Detection Probability), $P_{fa}=0.1$ fixed", fontsize=10)
ax1.set_ylabel("Served Goodput (Mbps)")
ax1.set_title("(a) All four conditions across $P_d$ gradient", fontsize=11)
ax1.set_xlim(1.08, 0.22)
ax1.set_ylim(10, 30)
ax1.legend(fontsize=8, loc="lower left")
ax1.grid(True, alpha=0.3)

# Panel B: Interaction term I(Pd)
I_vals = [-10.2, -9.7, -8.6, -5.5, -0.1]
colors_I = [COLORS["delta_neg"]] * 3 + [COLORS["delta_pos"]] * 2
ax2.bar(np.arange(len(pd_vals_all)), I_vals, color=colors_I, edgecolor="white", linewidth=0.5, width=0.6)
ax2.axhline(y=0, color="black", linewidth=1.5, linestyle="--")
ax2.set_xticks(np.arange(len(pd_vals_all)))
ax2.set_xticklabels([f"{v:.1f}" for v in pd_vals_all])
ax2.set_xlabel("Sensing $P_d$", fontsize=10)
ax2.set_ylabel("Interaction $I = $ full $-$ sens $-$ pred + neither (Mbps)")
ax2.set_title("(b) Interaction $I(P_d)$: from redundant to complementary", fontsize=11)

# Value labels
for i, (val, color) in enumerate(zip(I_vals, colors_I)):
    va = "bottom" if val < 0 else "top"
    offset = -0.5 if val < 0 else 0.3
    ax2.text(i, val + offset, f"{val:+.1f}", ha="center", va=va, fontsize=11,
             fontweight="bold", color=color)

ax2.text(0, -11.2, "REDUNDANT\n$I \\ll 0$\nprediction adds little\nto sensing", ha="center",
         fontsize=8, color=COLORS["delta_neg"])
ax2.text(4, 1.5, "COMPLEMENTARY\n$I \\approx 0$\nprediction independent\nof sensing", ha="center",
         fontsize=8, color=COLORS["delta_pos"])
ax2.grid(True, alpha=0.3, axis="y")

fig.suptitle(
    "Figure 4: Complete 2$\\times$2 factorial — prediction transitions from redundant to complementary",
    fontweight="bold", fontsize=12,
)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "fig4_complete_factorial_I_evolution.pdf")
fig.savefig(OUTPUT_DIR / "fig4_complete_factorial_I_evolution.png")
plt.close()
print("Figure 4 saved (complete factorial + I(Pd) evolution).")

# ===========================================================================
# FIGURE 5: Pfa gradient + Pd vs Pfa comparison
# ===========================================================================
df_pfa = pd.read_csv(
    "results/20260705_115119_e2_r1_oracle_gate_run/tables/final_checkpoint_results.csv"
)
# Also re-read Pd data
df_pd = pd.read_csv(
    "results/20260703_200605_e2_r1_oracle_gate_run/tables/final_checkpoint_results.csv"
)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# ---- Panel A: Pfa gradient goodput ----
pfa_map = {s: float(s.split("_")[-1].replace("p", ".")) for s in df_pfa["scenario"].unique()}
pfa_scenarios = sorted(df_pfa["scenario"].unique(), key=lambda s: pfa_map[s])
pfa_vals = [pfa_map[s] for s in pfa_scenarios]

full_pfa, sens_pfa = [], []
for scenario in pfa_scenarios:
    sub = df_pfa[df_pfa["scenario"] == scenario]
    full_pfa.append(sub[sub["module_variant"] == "full"]["served_goodput_mbps"].mean())
    sens_pfa.append(sub[sub["module_variant"] == "sensing_only"]["served_goodput_mbps"].mean())

axes[0].plot(pfa_vals, full_pfa, "o-", color=COLORS["full"], linewidth=2.5, markersize=8,
             label="Full (sensing + prediction)")
axes[0].plot(pfa_vals, sens_pfa, "s--", color=COLORS["sensing"], linewidth=2.5, markersize=8,
             label="Sensing-only")
axes[0].fill_between(pfa_vals, full_pfa, sens_pfa, alpha=0.1,
                     color=COLORS["delta_pos"] if full_pfa[-1] > sens_pfa[-1] else COLORS["delta_neg"])
axes[0].axvline(x=0.15, color="black", linestyle=":", linewidth=1.5)
axes[0].annotate("Crossover\n$P_{fa} \\approx 0.2$", xy=(0.22, 24.5), fontsize=9,
                 fontweight="bold", ha="left")
axes[0].set_xlabel("Sensing $P_{fa}$ (False Alarm Rate), $P_d=0.9$ fixed", fontsize=9)
axes[0].set_ylabel("Served Goodput (Mbps)")
axes[0].set_title("(a) Pfa gradient: prediction compensates false alarms", fontsize=10)
axes[0].legend(fontsize=8, loc="upper right")
axes[0].set_xlim(-0.02, 0.55)
axes[0].set_ylim(22, 27)
axes[0].grid(True, alpha=0.3)

# ---- Panel B: Delta comparison Pd vs Pfa ----
pd_deltas = []
pd_map_d3 = {s: float(s.split("_")[-1].replace("p", ".")) for s in df_pd["scenario"].unique()}
for scenario in sorted(df_pd["scenario"].unique(), key=lambda s: pd_map_d3[s], reverse=True):
    sub = df_pd[df_pd["scenario"] == scenario]
    f = sub[sub["module_variant"] == "full"]["served_goodput_mbps"].mean()
    s = sub[sub["module_variant"] == "sensing_only"]["served_goodput_mbps"].mean()
    pd_deltas.append(f - s)

pfa_deltas = [f - s for f, s in zip(full_pfa, sens_pfa)]

x_pd = [1.0, 0.9, 0.7, 0.5, 0.3]
x_pfa_pos = np.arange(len(pfa_vals))

# Pd deltas on left y-axis
axes[1].bar(np.arange(len(x_pd)) - 0.15, pd_deltas, 0.3,
           color=[COLORS["delta_pos"] if d > 0 else COLORS["delta_neg"] for d in pd_deltas],
           edgecolor="white", label="$P_d$ gradient (detection)")
# Pfa deltas on right
axes[1].bar(np.arange(len(pfa_vals)) + 0.15, pfa_deltas, 0.3,
           color=[COLORS["delta_pos"] if d > 0 else COLORS["delta_neg"] for d in pfa_deltas],
           edgecolor="black", linewidth=0.5, hatch="//", label="$P_{fa}$ gradient (false alarm)")

axes[1].axhline(y=0, color="black", linewidth=1)
axes[1].set_xticks(np.arange(5))
axes[1].set_xticklabels(["1.0 / 0.05", "0.9 / 0.1", "0.7 / 0.2", "0.5 / 0.3", "0.3 / 0.5"])
axes[1].set_xlabel("$P_d$ / $P_{fa}$ level", fontsize=9)
axes[1].set_ylabel("$\\Delta$ Goodput: Full $-$ Sensing-only (Mbps)")
axes[1].set_title("(b) Pd vs Pfa: both show crossover", fontsize=10)
axes[1].legend(fontsize=8)
axes[1].grid(True, alpha=0.3, axis="y")

# ---- Panel C: Full robustness across both axes ----
# Normalize to sensing-only at best condition
best_sens_pd = max(sens_all)
best_sens_pfa = max(sens_pfa)
full_norm_pd = [f / best_sens_pd * 100 for f in full_all]
sens_norm_pd = [s / best_sens_pd * 100 for s in sens_all]
full_norm_pfa = [f / best_sens_pfa * 100 for f in full_pfa]
sens_norm_pfa = [s / best_sens_pfa * 100 for s in sens_pfa]

axes[2].plot(x_pd, full_norm_pd, "o-", color=COLORS["full"], linewidth=2.5, markersize=8,
             label="Full (Pd axis)")
axes[2].plot(x_pd, sens_norm_pd, "s--", color=COLORS["sensing"], linewidth=2.5, markersize=8,
             label="Sensing-only (Pd axis)")
axes[2].plot(pfa_vals, full_norm_pfa, "o-", color=COLORS["full"], linewidth=2.5, markersize=8,
             alpha=0.5, label="Full (Pfa axis)")
axes[2].plot(pfa_vals, sens_norm_pfa, "s--", color=COLORS["sensing"], linewidth=2.5, markersize=8,
             alpha=0.5, label="Sensing-only (Pfa axis)")

axes[2].set_xlabel("Sensing Quality Parameter ($P_d$ or $1-P_{fa}$)")
axes[2].set_ylabel("Normalized Goodput (% of best sensing-only)")
axes[2].set_title("(c) Full agent is robust to both error types", fontsize=10)
axes[2].legend(fontsize=7, loc="lower left")
axes[2].set_xlim(1.05, 0.0)
axes[2].grid(True, alpha=0.3)

fig.suptitle(
    "Figure 5: Pfa gradient — prediction compensates for false alarms (weaker but confirmed)",
    fontweight="bold", fontsize=12,
)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "fig5_pfa_gradient_dual_axis.pdf")
fig.savefig(OUTPUT_DIR / "fig5_pfa_gradient_dual_axis.png")
plt.close()
print("Figure 5 saved (Pfa gradient + dual-axis comparison).")

# ===========================================================================
# FIGURE 6: Complete Pfa-axis 4-condition factorial + I(Pfa) evolution
# ===========================================================================
# Supplement data (prediction_only + neither at Pfa=0.05, 0.5)
sup_pfa = pd.read_csv(
    "results/20260705_133714_e2_r1_oracle_gate_run/tables/final_checkpoint_results.csv"
)
# Pfa gradient (full + sensing_only at all 5 Pfa levels)
df_pfa_full = pd.read_csv(
    "results/20260705_115119_e2_r1_oracle_gate_run/tables/final_checkpoint_results.csv"
)
# Baseline (prediction_only + neither at Pfa=0.1)
base = pd.read_csv(
    "results/e2_prediction_sensing_redundancy_pilot_a/"
    "20260703_142409_e2_r1_oracle_gate_run/tables/final_checkpoint_results.csv"
)
pred_ref = base[base["module_variant"] == "prediction_only"]["served_goodput_mbps"].mean()
neith_ref = base[base["module_variant"] == "neither"]["served_goodput_mbps"].mean()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

# Panel A: All 4 conditions on Pfa axis
pfa_map = {s: float(s.split("_")[-1].replace("p", ".")) for s in df_pfa_full["scenario"].unique()}
scenarios_sorted = sorted(df_pfa_full["scenario"].unique(), key=lambda s: pfa_map[s])
pfa_vals = [pfa_map[s] for s in scenarios_sorted]

full_pfa, sens_pfa = [], []
for scenario in scenarios_sorted:
    sub = df_pfa_full[df_pfa_full["scenario"] == scenario]
    full_pfa.append(sub[sub["module_variant"] == "full"]["served_goodput_mbps"].mean())
    sens_pfa.append(sub[sub["module_variant"] == "sensing_only"]["served_goodput_mbps"].mean())

# pred + neither: use measured at 0.05 and 0.5, baseline at others
pred_pfa, neith_pfa = [], []
for pfa_val in pfa_vals:
    scenario = f"P0_M1_pfa_{str(pfa_val).replace('.','p')}"
    if scenario in sup_pfa["scenario"].unique():
        sub = sup_pfa[sup_pfa["scenario"] == scenario]
        pred_pfa.append(sub[sub["module_variant"] == "prediction_only"]["served_goodput_mbps"].mean())
        neith_pfa.append(sub[sub["module_variant"] == "neither"]["served_goodput_mbps"].mean())
    else:
        pred_pfa.append(pred_ref)
        neith_pfa.append(neith_ref)

ax1.plot(pfa_vals, full_pfa, "o-", color=COLORS["full"], linewidth=2.5, markersize=9,
         label="Full (sensing + prediction)")
ax1.plot(pfa_vals, sens_pfa, "s--", color=COLORS["sensing"], linewidth=2.5, markersize=9,
         label="Sensing-only")
ax1.plot(pfa_vals, pred_pfa, "D-.", color="#4CAF50", linewidth=2, markersize=8,
         label="Prediction-only (flat: no sensing used)")
ax1.plot(pfa_vals, neith_pfa, "v:", color="#9E9E9E", linewidth=2, markersize=8,
         label="Neither (flat: no information)")

# Mark measured vs baseline points
for x_pfa, y_pred, y_neith in zip(pfa_vals, pred_pfa, neith_pfa):
    scenario = f"P0_M1_pfa_{str(x_pfa).replace('.','p')}"
    if scenario in sup_pfa["scenario"].unique():
        ax1.plot(x_pfa, y_pred, "D", color="#2E7D32", markersize=14, zorder=5)
        ax1.plot(x_pfa, y_neith, "v", color="#424242", markersize=14, zorder=5)

ax1.axvline(x=0.15, color="black", linestyle=":", linewidth=1.5)
ax1.annotate("Crossover\n$P_{fa} \\approx 0.2$", xy=(0.25, 25.5), fontsize=10,
             fontweight="bold", ha="left")
ax1.set_xlabel("Sensing $P_{fa}$ (False Alarm Rate), $P_d=0.9$ fixed", fontsize=10)
ax1.set_ylabel("Served Goodput (Mbps)")
ax1.set_title("(a) All four conditions across $P_{fa}$ gradient", fontsize=11)
ax1.set_xlim(-0.02, 0.55)
ax1.set_ylim(10, 26.5)
ax1.legend(fontsize=8, loc="lower left")
ax1.grid(True, alpha=0.3)

# Panel B: I(Pfa) evolution
I_pfa = []
for i, scenario in enumerate(scenarios_sorted):
    f = full_pfa[i]
    s = sens_pfa[i]
    p = pred_pfa[i]
    n = neith_pfa[i]
    I_pfa.append(f - s - p + n)

colors_I_pfa = [COLORS["delta_neg"]] * 5  # all still negative
positions = np.arange(len(pfa_vals))
ax2.bar(positions, I_pfa, color=colors_I_pfa, edgecolor="white", linewidth=0.5, width=0.6)
ax2.axhline(y=0, color="black", linewidth=1.5, linestyle="--")
ax2.set_xticks(positions)
ax2.set_xticklabels([f"{v:.2f}" for v in pfa_vals])
ax2.set_xlabel("Sensing $P_{fa}$", fontsize=10)
ax2.set_ylabel("Interaction $I = $ full $-$ sens $-$ pred + neither (Mbps)")
ax2.set_title("(b) Interaction $I(P_{fa})$: remains negative, no zero-crossing", fontsize=11)

for i, val in enumerate(I_pfa):
    ax2.text(i, val - 0.3, f"{val:+.1f}", ha="center", va="top", fontsize=11,
             fontweight="bold", color=COLORS["delta_neg"])

ax2.text(2, -10.2, "REDUNDANT\n$I \\ll 0$ across all $P_{fa}$\nFalse alarms don't break\nprediction-sensing redundancy", ha="center",
         fontsize=8, color=COLORS["delta_neg"])
ax2.set_ylim(-11, 1)
ax2.grid(True, alpha=0.3, axis="y")

fig.suptitle(
    "Figure 6: Complete Pfa-axis 2$\\times$2 factorial — false alarms vs missed detections",
    fontweight="bold", fontsize=12,
)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "fig6_pfa_complete_factorial_I_evolution.pdf")
fig.savefig(OUTPUT_DIR / "fig6_pfa_complete_factorial_I_evolution.png")
plt.close()
print("Figure 6 saved (complete Pfa factorial + I(Pfa) evolution).")
