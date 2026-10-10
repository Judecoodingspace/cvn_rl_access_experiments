"""Paper-ready Matplotlib style helpers for IEEEtran figures."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

PAPER_RCPARAMS: dict = {
    "savefig.format": "eps",
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "savefig.dpi": 300,
    "savefig.transparent": False,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "lines.linewidth": 1.0,
    "lines.markersize": 4,
    "lines.markeredgewidth": 0.5,
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "grid.linewidth": 0.3,
    "grid.alpha": 0.4,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.figsize": (3.5, 2.5),
    "figure.dpi": 150,
    "legend.frameon": True,
    "legend.framealpha": 0.9,
    "legend.edgecolor": "#cccccc",
    "mathtext.fontset": "stix",
    "mathtext.default": "regular",
    "ps.usedistiller": "none",
    "ps.fonttype": 42,
    "pdf.fonttype": 42,
}

FIG_SINGLE = (3.5, 2.5)
FIG_DOUBLE = (7.0, 3.5)
FIG_SINGLE_TALL = (3.5, 3.5)

FIG_IEEE_SINGLE = (3.45, 2.55)
FIG_IEEE_SQUARE = (3.35, 3.10)
FIG_IEEE_DOUBLE = (7.10, 3.20)
FIG_IEEE_COMPACT_2X2 = (6.80, 4.20)
FIG_IEEE_DIAGNOSTIC_TWO_PANEL = (11.8, 5.5)

PALETTE: dict[str, str] = {
    "blue": "#4c78a8",
    "orange": "#f58518",
    "green": "#54a24b",
    "red": "#e45756",
    "purple": "#b279a2",
    "brown": "#9d755d",
    "gray": "#79706e",
    "black": "#333333",
}

DIAGNOSTIC_PALETTE: dict[str, str] = {
    "full": "#2196F3",
    "sensing": "#FF9800",
    "pred": "#4CAF50",
    "neither": "#9E9E9E",
    "delta_pos": "#4CAF50",
    "delta_neg": "#F44336",
}


def apply_paper_style() -> None:
    """Apply paper-ready Matplotlib rcParams globally."""
    plt.rcParams.update(PAPER_RCPARAMS)


IEEE_STRICT_RCPARAMS: dict = {
    **PAPER_RCPARAMS,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "CMU Serif", "DejaVu Serif"],
    "font.size": 8.5,
    "axes.titlesize": 8.5,
    "axes.labelsize": 8.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
    "lines.linewidth": 1.35,
    "lines.markersize": 4.6,
    "lines.markeredgewidth": 0.6,
    "axes.linewidth": 0.9,
    "axes.spines.top": True,
    "axes.spines.right": True,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "grid.linewidth": 0.35,
    "grid.alpha": 0.55,
    "figure.figsize": FIG_IEEE_SINGLE,
    "mathtext.fontset": "cm",
    "ps.fonttype": 42,
    "pdf.fonttype": 42,
}

IEEE_DIAGNOSTIC_TWO_PANEL_RCPARAMS: dict = {
    **IEEE_STRICT_RCPARAMS,
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 10.5,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 10,
    "legend.frameon": True,
    "legend.framealpha": 0.9,
    "legend.edgecolor": "#cccccc",
    "axes.linewidth": 0.9,
    "axes.spines.top": True,
    "axes.spines.right": True,
    "mathtext.fontset": "cm",
    "mathtext.default": "regular",
    "figure.figsize": FIG_IEEE_DIAGNOSTIC_TWO_PANEL,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.06,
    "ps.fonttype": 42,
    "pdf.fonttype": 42,
}

IEEE_DIAGNOSTIC_TWO_PANEL_LAYOUT: dict[str, float] = {
    "left": 0.05,
    "right": 0.99,
    "bottom": 0.20,
    "wspace": 0.28,
    "top": 0.94,
}


def apply_ieee_strict_style() -> None:
    """Apply stricter IEEE-style rcParams for final paper figures."""
    plt.rcParams.update(IEEE_STRICT_RCPARAMS)


def apply_ieee_style() -> None:
    """Backward-compatible alias for the strict IEEE style."""
    apply_ieee_strict_style()


def apply_ieee_diagnostic_two_panel_style() -> None:
    """Apply the reusable Figure 4/6 two-panel diagnostic style."""
    plt.rcParams.update(IEEE_DIAGNOSTIC_TWO_PANEL_RCPARAMS)


def style_full_frame(ax: plt.Axes) -> None:
    """Keep all four axes spines visible above grids and data artists."""
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.9)
        spine.set_color("black")
        spine.set_zorder(10)


def add_bottom_subplot_title(ax: plt.Axes, label: str) -> None:
    """Place a two-line subplot title below an axes."""
    ax.text(
        0.50,
        -0.10,
        label,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=10.5,
        linespacing=1.15,
    )


def add_bottom_figure_title(fig: plt.Figure, label: str) -> None:
    """Place the main title below the subplot-title row."""
    fig.text(
        0.50,
        0.005,
        label,
        ha="center",
        va="bottom",
        fontsize=11.5,
        fontweight="bold",
    )


def adjust_diagnostic_two_panel_layout(fig: plt.Figure) -> None:
    """Apply the Figure 4/6 margins and inter-panel spacing."""
    fig.subplots_adjust(**IEEE_DIAGNOSTIC_TWO_PANEL_LAYOUT)


def save_ieee_diagnostic_two_panel_figure(
    fig: plt.Figure,
    stem: str,
    formats: tuple[str, ...] = ("pdf", "png"),
) -> None:
    """Save with the exact Figure 4/6 crop padding and font embedding."""
    output_stem = Path(stem)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(
            f"{output_stem}.{fmt}",
            format=fmt,
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.06,
            transparent=False,
        )


def save_paper_figure(
    fig: plt.Figure,
    stem: str,
    formats: tuple[str, ...] = ("eps", "pdf"),
) -> None:
    """Save a general paper figure with tight content bounds."""
    for fmt in formats:
        path = f"{stem}.{fmt}"
        if fmt == "eps":
            fig.savefig(
                path,
                format="eps",
                dpi=300,
                bbox_inches="tight",
                pad_inches=0.02,
                transparent=False,
            )
        elif fmt == "pdf":
            fig.savefig(path, format="pdf", dpi=300, bbox_inches="tight", pad_inches=0.02)
        elif fmt == "png":
            fig.savefig(path, format="png", dpi=200, bbox_inches="tight", pad_inches=0.02)


def save_ieee_figure(
    fig: plt.Figure,
    stem: str,
    formats: tuple[str, ...] = ("eps", "pdf", "png"),
    *,
    tight: bool = False,
) -> None:
    """Save an IEEE figure while preserving its requested canvas ratio."""
    bbox_inches = "tight" if tight else None
    pad_inches = 0.02 if tight else 0.0
    for fmt in formats:
        path = f"{stem}.{fmt}"
        with plt.rc_context({"savefig.bbox": None if not tight else "tight"}):
            if fmt == "eps":
                fig.savefig(
                    path,
                    format="eps",
                    dpi=600,
                    bbox_inches=bbox_inches,
                    pad_inches=pad_inches,
                    transparent=False,
                )
            elif fmt == "pdf":
                fig.savefig(path, format="pdf", dpi=600, bbox_inches=bbox_inches, pad_inches=pad_inches)
            elif fmt == "png":
                fig.savefig(path, format="png", dpi=600, bbox_inches=bbox_inches, pad_inches=pad_inches)
