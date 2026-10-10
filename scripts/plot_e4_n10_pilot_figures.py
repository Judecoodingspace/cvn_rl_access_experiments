"""Generate paper figures for the E4 N=10 multi-VT independent-learning pilot.

Fig A: convergence curves (served goodput, mean +/- seed std) in the competitive
regime for independent_ddqn vs shared-parameter ESN-DDQN / DDQN.
Fig B: contention-loss bars with 95% CI across both resource regimes.

Usage:
    python scripts/plot_e4_n10_pilot_figures.py [run_dir]
Default run_dir: results/e4_n10_pilot/20260805_232538_multi_vehicle_baseline_comparison_run
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sps

from cvn_rl.paper_figure_style import (
    FIG_IEEE_DOUBLE,
    FIG_IEEE_SINGLE,
    PALETTE,
    apply_ieee_style,
    save_ieee_figure,
    style_full_frame,
)

RUN_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
    "results/e4_n10_pilot/20260805_232538_multi_vehicle_baseline_comparison_run"
)
OUTPUT_DIR = Path("paper_figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SCEN_COMP = "M10_N10_competitive"
SCEN_ABUN = "M15_N10_abundant"
OFFERED_COMP_MBPS = 10 * 0.5 * 32.0  # N * arrival * packet_bits/slot

AGENT_LABELS = {
    "independent_ddqn": "Independent DDQN",
    "proposed_esn_ddqn": "ESN-DDQN (shared)",
    "ddqn_no_esn": "DDQN (shared)",
    "dqn": "DQN (shared)",
    "lstm_ddqn": "LSTM-DDQN (shared)",
    "sac_discrete": "Discrete SAC (shared)",
    "ppo_discrete": "PPO-discrete (shared)",
    "tabular_q_learning": "Tabular Q (shared)",
    "fixed_overlay": "Fixed overlay",
    "fixed_underlay": "Fixed underlay",
}

T_CRIT_DF4 = sps.t.ppf(0.975, df=4)  # 5 seeds


def plot_fig_a() -> None:
    ck = pd.read_csv(
        RUN_DIR / "tables" / "checkpoint_summary.csv",
        usecols=["scenario", "agent", "checkpoint_slot",
                 "served_goodput_mbps_mean", "served_goodput_mbps_std"],
    )
    ck = ck[ck["scenario"] == SCEN_COMP]

    fig, ax = plt.subplots(figsize=FIG_IEEE_SINGLE)
    series = [
        ("independent_ddqn", PALETTE["red"], 1.9, "-"),
        ("proposed_esn_ddqn", PALETTE["blue"], 1.2, "-"),
        ("ddqn_no_esn", PALETTE["orange"], 1.2, "--"),
    ]
    for agent, color, lw, ls in series:
        g = ck[ck["agent"] == agent].sort_values("checkpoint_slot")
        x = g["checkpoint_slot"].to_numpy() / 1000.0
        m = g["served_goodput_mbps_mean"].to_numpy()
        s = g["served_goodput_mbps_std"].to_numpy()
        ax.plot(x, m, ls, color=color, linewidth=lw, label=AGENT_LABELS[agent])
        ax.fill_between(x, m - s, m + s, color=color, alpha=0.13, linewidth=0.0)

    ax.axhline(OFFERED_COMP_MBPS, color=PALETTE["gray"], linestyle=":",
               linewidth=1.0)
    ax.text(1.2, OFFERED_COMP_MBPS + 3.0, "system offered load",
            color=PALETTE["gray"], fontsize=7.0)

    ax.set_xlabel(r"Training slot ($\times 10^3$)")
    ax.set_ylabel("Served goodput (Mbps)")
    ax.set_xlim(left=1.0)
    ax.set_ylim(0.0, 178.0)
    ax.legend(loc="lower right")
    style_full_frame(fig.axes[0])

    stem = OUTPUT_DIR / "fig_e4_n10_independent_convergence"
    save_ieee_figure(fig, str(stem), formats=("pdf", "png"), tight=True)
    plt.close(fig)
    print(f"saved {stem}.pdf/.png")


OFFERED_ABUN_MBPS = 10 * 0.2 * 32.0  # N * arrival * packet_bits/slot


def _load_checkpoints(scenario: str) -> pd.DataFrame:
    ck = pd.read_csv(
        RUN_DIR / "tables" / "checkpoint_summary.csv",
        usecols=["scenario", "agent", "checkpoint_slot",
                 "served_goodput_mbps_mean", "served_goodput_mbps_std"],
    )
    return ck[ck["scenario"] == scenario]


def plot_fig_a4() -> None:
    """Variant B: 4-line convergence (independent + LSTM + ESN + DDQN)."""
    ck = _load_checkpoints(SCEN_COMP)
    fig, ax = plt.subplots(figsize=FIG_IEEE_SINGLE)
    series = [
        ("independent_ddqn", PALETTE["red"], 1.9, "-", 0.16),
        ("lstm_ddqn", PALETTE["green"], 1.2, "-", 0.10),
        ("proposed_esn_ddqn", PALETTE["blue"], 1.2, "-", 0.10),
        ("ddqn_no_esn", PALETTE["orange"], 1.2, "--", 0.10),
    ]
    for agent, color, lw, ls, alpha in series:
        g = ck[ck["agent"] == agent].sort_values("checkpoint_slot")
        x = g["checkpoint_slot"].to_numpy() / 1000.0
        m = g["served_goodput_mbps_mean"].to_numpy()
        s = g["served_goodput_mbps_std"].to_numpy()
        ax.plot(x, m, ls, color=color, linewidth=lw,
                label=AGENT_LABELS[agent].replace(" (shared)", ""))
        ax.fill_between(x, m - s, m + s, color=color, alpha=alpha, linewidth=0.0)

    ax.axhline(OFFERED_COMP_MBPS, color=PALETTE["gray"], linestyle=":",
               linewidth=1.0)
    ax.text(1.2, OFFERED_COMP_MBPS + 3.0, "system offered load",
            color=PALETTE["gray"], fontsize=7.0)
    ax.set_xlabel(r"Training slot ($\times 10^3$)")
    ax.set_ylabel("Served goodput (Mbps)")
    ax.set_xlim(left=1.0)
    ax.set_ylim(0.0, 178.0)
    ax.legend(loc="lower right")
    style_full_frame(ax)

    stem = OUTPUT_DIR / "fig_e4_n10_independent_convergence_4way"
    save_ieee_figure(fig, str(stem), formats=("pdf", "png"), tight=True)
    plt.close(fig)
    print(f"saved {stem}.pdf/.png")


def plot_fig_c(scenario: str = SCEN_COMP,
               stem: str = "fig_e4_n10_convergence_all_backbones") -> None:
    """Variant C: all 8 learned backbones, no bands, single-column frame
    matching the 4-way version, per-VT served goodput on the y-axis."""
    ck = _load_checkpoints(scenario)
    offered_mbps = {SCEN_COMP: OFFERED_COMP_MBPS,
                    SCEN_ABUN: OFFERED_ABUN_MBPS}[scenario]
    offered_per_vt = offered_mbps / 10.0
    order = [
        "independent_ddqn", "lstm_ddqn", "proposed_esn_ddqn", "ddqn_no_esn",
        "dqn", "sac_discrete", "ppo_discrete", "tabular_q_learning",
    ]
    colors = [PALETTE["red"], PALETTE["green"], PALETTE["blue"],
              PALETTE["orange"], PALETTE["purple"], PALETTE["brown"],
              PALETTE["gray"], PALETTE["black"]]
    markers = ["o", "s", "^", "v", "D", "x", "P", "*"]

    fig, ax = plt.subplots(figsize=FIG_IEEE_SINGLE)
    for i, agent in enumerate(order):
        g = ck[ck["agent"] == agent].sort_values("checkpoint_slot")
        x = g["checkpoint_slot"].to_numpy() / 1000.0
        m = g["served_goodput_mbps_mean"].to_numpy() / 10.0  # per-VT (N=10)
        is_lead = agent == "independent_ddqn"
        ax.plot(
            x, m, color=colors[i], linewidth=1.9 if is_lead else 1.1,
            marker=markers[i], markevery=3, markersize=3.6 if is_lead else 3.0,
            label=AGENT_LABELS[agent].replace(" (shared)", ""),
        )

    ax.axhline(offered_per_vt, color=PALETTE["gray"], linestyle=":",
               linewidth=1.0)
    ax.text(1.2, offered_per_vt * 1.02, "per-VT offered load",
            color=PALETTE["gray"], fontsize=7.0)
    ax.set_xlabel(r"Training slot ($\times 10^3$)")
    ax.set_ylabel("Per-VT served goodput (Mbps)")
    ax.set_xlim(left=1.0)
    ax.set_ylim(0.0, offered_per_vt * 1.11)
    ax.legend(
        loc="lower right", ncol=2, fontsize=6.3,
        labelspacing=0.25, borderpad=0.3, borderaxespad=0.2,
        handletextpad=0.5, columnspacing=0.6, handlelength=1.2,
    )
    style_full_frame(ax)

    stem_path = OUTPUT_DIR / stem
    save_ieee_figure(fig, str(stem_path), formats=("pdf", "png"), tight=True)
    plt.close(fig)
    print(f"saved {stem_path}.pdf/.png")


def plot_fig_b() -> None:
    df = pd.read_csv(
        RUN_DIR / "tables" / "final_checkpoint_results.csv",
        usecols=["scenario", "agent", "seed", "contention_loss_rate"],
    )

    agents = list(AGENT_LABELS)
    stats: dict[str, dict[str, tuple[float, float]]] = {}
    for scen in (SCEN_COMP, SCEN_ABUN):
        sub = df[df["scenario"] == scen]
        for agent in agents:
            vals = sub[sub["agent"] == agent]["contention_loss_rate"].to_numpy() * 100.0
            mean = vals.mean()
            ci = T_CRIT_DF4 * vals.std(ddof=1) / np.sqrt(len(vals))
            stats.setdefault(agent, {})[scen] = (mean, ci)

    # order bars by competitive contention, descending
    order = sorted(agents, key=lambda a: stats[a][SCEN_COMP][0], reverse=True)

    fig, axes = plt.subplots(1, 2, figsize=FIG_IEEE_DOUBLE, sharey=True)
    for ax, scen, title in (
        (axes[0], SCEN_COMP, r"(a) Competitive ($M{=}10$, $N{=}10$)"),
        (axes[1], SCEN_ABUN, r"(b) Abundant ($M{=}15$, $N{=}10$)"),
    ):
        x = np.arange(len(order))
        means = [stats[a][scen][0] for a in order]
        cis = [stats[a][scen][1] for a in order]
        colors = [PALETTE["red"] if a == "independent_ddqn" else PALETTE["blue"]
                  for a in order]
        ax.bar(x, means, yerr=cis, color=colors, edgecolor="black",
               linewidth=0.4, capsize=2.0, error_kw={"linewidth": 0.7}, width=0.72)
        ax.set_xticks(x)
        ax.set_xticklabels([AGENT_LABELS[a].replace(" (shared)", "")
                           .replace("Independent DDQN", "Independent\nDDQN")
                           .replace("Fixed ", "Fixed\n") for a in order],
                           rotation=45, ha="right")
        ax.set_title(title, fontsize=8.5)
        style_full_frame(ax)
        idx = order.index("independent_ddqn")
        ax.annotate(f"{means[idx]:.1f}%", (x[idx], means[idx] + cis[idx] + 1.5),
                    ha="center", fontsize=7.5, color=PALETTE["red"],
                    fontweight="bold")

    axes[0].set_ylabel("Contention loss rate (%)")
    axes[0].set_ylim(0.0, 100.0)
    fig.subplots_adjust(wspace=0.18)

    stem = OUTPUT_DIR / "fig_e4_n10_contention_loss"
    save_ieee_figure(fig, str(stem), formats=("pdf", "png"), tight=True)
    plt.close(fig)
    print(f"saved {stem}.pdf/.png")


def main() -> None:
    apply_ieee_style()
    plot_fig_a()
    plot_fig_a4()
    plot_fig_c()  # competitive
    plot_fig_c(SCEN_ABUN, "fig_e4_n10_convergence_all_backbones_abundant")
    plot_fig_b()


if __name__ == "__main__":
    main()
