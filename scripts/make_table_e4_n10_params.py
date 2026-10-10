"""Extract per-backbone parameter counts and inference latency from the
E4 N=10 pilot run and emit a LaTeX table for the response letter.

Usage:
    python scripts/make_table_e4_n10_params.py [run_dir]
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

RUN_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
    "results/e4_n10_pilot/20260805_232538_multi_vehicle_baseline_comparison_run"
)
SCENARIO = sys.argv[2] if len(sys.argv) > 2 else "M10_N10_competitive"
OUT_DIR = Path("paper_table/table_e4_n10_multi_vt")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SCENARIO_LABELS = {
    "M10_N10_competitive": "competitive",
    "M15_N10_abundant": "abundant",
}
if SCENARIO not in SCENARIO_LABELS:
    raise ValueError(f"unsupported E4 scenario: {SCENARIO}")
SCENARIO_LABEL = SCENARIO_LABELS[SCENARIO]

AGENT_ORDER = [
    "independent_ddqn", "proposed_esn_ddqn", "ddqn_no_esn", "dqn",
    "lstm_ddqn", "sac_discrete", "ppo_discrete", "tabular_q_learning",
    "fixed_overlay", "fixed_underlay",
]
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

COLS = ["parameter_count", "trainable_parameter_count",
        "fixed_reservoir_parameter_count", "total_stored_parameter_count",
        "inference_latency_ms", "training_time_seconds"]


def main() -> None:
    df = pd.read_csv(RUN_DIR / "tables" / "final_checkpoint_results.csv",
                     usecols=["scenario", "agent", "seed"] + COLS)
    # Parameter counts and latency depend on the action-space dimension, so
    # summarize one explicitly selected scenario at a time.
    df = df[df["scenario"] == SCENARIO]

    rows = []
    for agent in AGENT_ORDER:
        ga = df[df["agent"] == agent]
        if ga.empty:
            continue
        rows.append({
            "agent": agent,
            "label": AGENT_LABELS[agent],
            "params": ga["parameter_count"].mean(),
            "trainable": ga["trainable_parameter_count"].mean(),
            "reservoir_fixed": ga["fixed_reservoir_parameter_count"].mean(),
            "total_stored": ga["total_stored_parameter_count"].mean(),
            "latency_ms": ga["inference_latency_ms"].mean(),
            "latency_std": ga["inference_latency_ms"].std(ddof=1),
            "train_s": ga["training_time_seconds"].mean(),
        })
    out = pd.DataFrame(rows)
    # independent_ddqn's reported parameter_count already aggregates the
    # N=10 per-VT networks (277,970 = 10 x 27,797); derive the per-VT count.
    mask = out["agent"] == "independent_ddqn"
    out["params_per_vt"] = out["params"]
    out.loc[mask, "params_per_vt"] = out.loc[mask, "params"] / 10.0

    out_rounded = out.round(4)
    source_filename = (
        "e4_n10_params_source.csv"
        if SCENARIO_LABEL == "competitive"
        else f"e4_n10_{SCENARIO_LABEL}_params_source.csv"
    )
    tex_filename = (
        "table_e4_n10_params.tex"
        if SCENARIO_LABEL == "competitive"
        else f"table_e4_n10_{SCENARIO_LABEL}_params.tex"
    )
    out_rounded.to_csv(OUT_DIR / source_filename, index=False)

    lines = [
        "\\begin{table}[!t]",
        "\\centering",
        "\\caption{\\textbf{Parameter count and per-decision inference latency "
        "of the compared controllers ($N{=}10$).} For the parameter-shared "
        "controllers, one network serves all VTs. Independent DDQN maintains "
        "one small DDQN per VT; the per-VT trainable count is shown with the "
        "$\\times 10$ network multiplicity. Fixed policies have no learned "
        "parameters. Latency is measured per VT decision on CPU.}",
        f"\\label{{tab:e4_n10_{SCENARIO_LABEL}_params}}",
        "\\small",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "Controller & \\shortstack{Trainable\\\\parameters} & "
        "\\shortstack{Inference\\\\latency (ms)} & "
        "\\shortstack{Training\\\\time (s)} \\\\",
        "\\midrule",
    ]
    for _, r in out.iterrows():
        if r["agent"] in {"fixed_overlay", "fixed_underlay"}:
            row = f"{r['label']} & -- & {r['latency_ms']:.3f} & 0 \\\\"
            if r["agent"] == "fixed_overlay":
                lines.append("\\midrule")
        elif r["agent"] == "independent_ddqn":
            row = (f"{r['label']} & {int(r['params_per_vt']):,} ($\\times$10) & "
                   f"{r['latency_ms']:.3f} & {r['train_s']:.0f} \\\\")
        else:
            row = (f"{r['label']} & {int(r['trainable']):,} & "
                   f"{r['latency_ms']:.3f} & {r['train_s']:.0f} \\\\")
        lines.append(row)
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]

    tex_path = OUT_DIR / tex_filename
    tex_path.write_text("\n".join(lines), encoding="utf-8")
    print(out_rounded.to_string(index=False))
    print(f"saved {tex_path}")
    print(f"saved {OUT_DIR / source_filename}")


if __name__ == "__main__":
    main()
