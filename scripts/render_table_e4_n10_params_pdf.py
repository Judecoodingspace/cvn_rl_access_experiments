"""Render the E4 N=10 parameter/latency table as a PDF for the response letter.

Reads e4_n10_params_source.csv (produced by make_table_e4_n10_params.py) and
renders a booktabs-style table with matplotlib, IEEE fonts.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from cvn_rl.paper_figure_style import PALETTE, apply_ieee_style

DIR = Path("paper_table/table_e4_n10_multi_vt")
SRC = DIR / "e4_n10_params_source.csv"
OUT = DIR / "table_e4_n10_params"

CAPTION = ("Parameter count and per-decision inference latency of the "
           "compared controllers (N = 10)")


def main() -> None:
    apply_ieee_style()
    df = pd.read_csv(SRC)

    header = ["Controller", "Trainable\nparameters", "Inference\nlatency (ms)",
              "Training\ntime (s)"]
    body, fixed_rows = [], []
    for _, r in df.iterrows():
        if r["agent"] in {"fixed_overlay", "fixed_underlay"}:
            row = [r["label"], "--", f"{r['latency_ms']:.3f}", "0"]
            fixed_rows.append(row)
        elif r["agent"] == "independent_ddqn":
            row = [r["label"], f"{int(r['params_per_vt']):,} (×10)",
                   f"{r['latency_ms']:.3f}", f"{r['train_s']:.0f}"]
            body.append(row)
        else:
            row = [r["label"], f"{int(r['trainable']):,}",
                   f"{r['latency_ms']:.3f}", f"{r['train_s']:.0f}"]
            body.append(row)
    n_sep = len(body)
    body += fixed_rows

    fig, ax = plt.subplots(figsize=(5.4, 3.1))
    ax.axis("off")
    table = ax.table(cellText=body, colLabels=header, loc="center",
                     cellLoc="center", colLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)
    table.auto_set_column_width(col=list(range(4)))
    table.scale(1.0, 1.25)

    # booktabs-ish styling: no verticals, horizontal rules only
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("none")
        if row == 0:
            cell.set_text_props(fontweight="bold")
            cell.set_edgecolor("black")
            cell.visible_edges = "TB"
            cell.set_linewidth(0.8)
        elif row == n_sep:
            # thin rule under the last learned row (learned | fixed split)
            cell.visible_edges = "B"
            cell.set_edgecolor("black")
            cell.set_linewidth(0.4)
        elif row == len(body):
            cell.visible_edges = "B"
            cell.set_edgecolor("black")
            cell.set_linewidth(0.8)
        if row > 0 and body[row - 1][0] == "Independent DDQN":
            cell.set_text_props(color=PALETTE["red"])

    ax.set_title(CAPTION, fontsize=8.0, pad=8)
    fig.savefig(f"{OUT}.pdf", format="pdf", dpi=600, bbox_inches="tight",
                pad_inches=0.04)
    fig.savefig(f"{OUT}.png", format="png", dpi=300, bbox_inches="tight",
                pad_inches=0.04)
    plt.close(fig)
    print(f"saved {OUT}.pdf/.png")


if __name__ == "__main__":
    main()
