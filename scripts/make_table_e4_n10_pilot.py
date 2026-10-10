"""Generate IEEE-style LaTeX tables for the E4 N=10 multi-VT pilot.

One table per resource regime, columns: SR (%), PLR (%), PU access violation (%).
Rows: all eight learned backbones followed by the two fixed policy baselines.
Values: mean +/- std over the five paired seeds, computed from
final_checkpoint_results.csv of the pilot run.

Usage:
    python scripts/make_table_e4_n10_pilot.py [run_dir]
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

RUN_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
    "results/e4_n10_pilot/20260805_232538_multi_vehicle_baseline_comparison_run"
)
OUT_DIR = Path("paper_table/table_e4_n10_multi_vt")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SCENARIOS = {
    "competitive": {
        "scenario": "M10_N10_competitive",
        "offered_mbps": 10 * 0.5 * 32.0,
        "caption": (
            "\\textbf{Multi-VT controller comparison, competitive regime "
            "($M{=}10$, $N{=}10$, 50\\% PU occupancy, arrival rate 0.5).} "
            "SR: service rate (served goodput over offered load); PU violation: "
            "PU access-violation rate. Mean $\\pm$ std over five paired seeds. "
            "All controllers use the Full descriptor configuration. "
            "Fixed policies are non-learning reference rows. "
            "Best learned result in bold."
        ),
        "label": "tab:e4_n10_competitive",
        "file": "table_e4_n10_competitive.tex",
    },
    "abundant": {
        "scenario": "M15_N10_abundant",
        "offered_mbps": 10 * 0.2 * 32.0,
        "caption": (
            "\\textbf{Multi-VT controller comparison, abundant regime "
            "($M{=}15$, $N{=}10$, 25\\% PU occupancy, arrival rate 0.2).} "
            "Definitions as in Table~\\ref{tab:e4_n10_competitive}. "
            "Rows are ordered by the competitive-regime service rate."
        ),
        "label": "tab:e4_n10_abundant",
        "file": "table_e4_n10_abundant.tex",
    },
}

AGENT_ORDER = [  # by competitive-regime service rate, learned first
    "independent_ddqn", "lstm_ddqn", "ddqn_no_esn", "proposed_esn_ddqn",
    "dqn", "sac_discrete", "ppo_discrete", "tabular_q_learning",
    "fixed_overlay", "fixed_underlay",
]
AGENT_LABELS = {
    "independent_ddqn": "Independent DDQN",
    "lstm_ddqn": "LSTM-DDQN",
    "ddqn_no_esn": "DDQN",
    "proposed_esn_ddqn": "ESN-DDQN",
    "dqn": "DQN",
    "sac_discrete": "Discrete SAC",
    "ppo_discrete": "PPO-discrete",
    "tabular_q_learning": "Tabular Q",
    "fixed_overlay": "Fixed overlay",
    "fixed_underlay": "Fixed underlay",
}
FIXED = {"fixed_overlay", "fixed_underlay"}


def fmt(mean: float, std: float, decimals: int) -> str:
    return f"{mean:.{decimals}f} $\\pm$ {std:.{decimals}f}"


def main() -> None:
    df = pd.read_csv(
        RUN_DIR / "tables" / "final_checkpoint_results.csv",
        usecols=["scenario", "agent", "seed", "served_goodput_mbps",
                 "plr", "pu_access_violation_rate"],
    )

    summary_rows = []
    for key, spec in SCENARIOS.items():
        sub = df[df["scenario"] == spec["scenario"]].copy()
        sub["sr"] = sub["served_goodput_mbps"] / spec["offered_mbps"] * 100.0
        sub["plr_pct"] = sub["plr"] * 100.0
        sub["pu_pct"] = sub["pu_access_violation_rate"] * 100.0

        stats = {}
        for agent in AGENT_ORDER:
            ga = sub[sub["agent"] == agent]
            stats[agent] = {
                "sr": (ga["sr"].mean(), ga["sr"].std(ddof=1)),
                "plr": (ga["plr_pct"].mean(), ga["plr_pct"].std(ddof=1)),
                "pu": (ga["pu_pct"].mean(), ga["pu_pct"].std(ddof=1)),
            }
            summary_rows.append({
                "regime": key, "agent": agent,
                "sr_mean": stats[agent]["sr"][0], "sr_std": stats[agent]["sr"][1],
                "plr_mean": stats[agent]["plr"][0], "plr_std": stats[agent]["plr"][1],
                "pu_violation_mean": stats[agent]["pu"][0],
                "pu_violation_std": stats[agent]["pu"][1],
            })

        learned = [a for a in AGENT_ORDER if a not in FIXED]
        best_sr = max(learned, key=lambda a: stats[a]["sr"][0])
        best_plr = min(learned, key=lambda a: stats[a]["plr"][0])
        best_pu = min(learned, key=lambda a: stats[a]["pu"][0])

        lines = [
            "\\begin{table}[!t]",
            "\\centering",
            f"\\caption{{{spec['caption']}}}",
            f"\\label{{{spec['label']}}}",
            "\\small",
            "\\setlength{\\tabcolsep}{4pt}",
            "\\begin{tabular}{lccc}",
            "\\toprule",
            "Controller & SR (\\%) & PLR (\\%) & \\shortstack{PU\\\\violation (\\%)} \\\\",
            "\\midrule",
        ]
        for agent in AGENT_ORDER:
            if agent == "fixed_overlay":
                lines.append("\\midrule")
            sr = fmt(*stats[agent]["sr"], 1)
            plr = fmt(*stats[agent]["plr"], 2)
            pu = fmt(*stats[agent]["pu"], 2)
            if agent == best_sr:
                sr = f"\\textbf{{{sr}}}"
            if agent == best_plr:
                plr = f"\\textbf{{{plr}}}"
            if agent == best_pu:
                pu = f"\\textbf{{{pu}}}"
            lines.append(f"{AGENT_LABELS[agent]} & {sr} & {plr} & {pu} \\\\")
        lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]

        tex_path = OUT_DIR / spec["file"]
        tex_path.write_text("\n".join(lines), encoding="utf-8")
        print(f"saved {tex_path}")

    pd.DataFrame(summary_rows).round(4).to_csv(
        OUT_DIR / "e4_n10_table_source.csv", index=False,
    )
    print(f"saved {OUT_DIR / 'e4_n10_table_source.csv'}")


if __name__ == "__main__":
    main()
