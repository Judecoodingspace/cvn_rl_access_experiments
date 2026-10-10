#!/usr/bin/env python3
"""E0 — Perception validation: offline descriptor quality analysis.

Computes Spearman ρ, Kendall τ (for DQI vs future quality) and Brier score,
ECE (for theta vs actual PU idle) from freshly generated traces.
No RL training required — pure offline analysis.

Outputs:
    figures/e0_dqi_quality_scatter.{eps,pdf,jpg}
    figures/e0_theta_calibration.{eps,pdf,jpg}
    tables/e0_perception_metrics.csv
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC))

from cvn_rl.paper_figure_style import (
    FIG_IEEE_DOUBLE,
    PALETTE,
    apply_ieee_strict_style,
    save_ieee_figure,
)
from cvn_rl.simulation.dqi_labels import latency_constrained_goodput
from cvn_rl.simulation.multi_traces import generate_multi_vehicle_trace


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _load_config(name: str) -> dict:
    path = PROJECT_ROOT / "configs" / name
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _resolve_inherits(config: dict) -> dict:
    """Simple single-level inherits resolution."""
    parent_name = config.get("inherits")
    if not parent_name:
        return config
    parent = _load_config(parent_name)
    resolved = dict(parent)
    resolved.update({k: v for k, v in config.items() if k != "inherits"})
    return resolved


def _expected_calibration_error(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> float:
    """Compute Expected Calibration Error (ECE)."""
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    total = len(y_true)
    for i in range(n_bins):
        mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
        if mask.sum() == 0:
            continue
        bin_acc = y_true[mask].mean()
        bin_conf = y_prob[mask].mean()
        ece += (mask.sum() / total) * abs(bin_acc - bin_conf)
    return float(ece)


def _brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return float(np.mean((y_prob - y_true) ** 2))


# ---------------------------------------------------------------------------
# DQI validation (M=10, causal DQI)
# ---------------------------------------------------------------------------
def validate_dqi(out_dir: Path) -> dict:
    """Generate M=10 traces with causal DQI and compute quality correlation."""
    config = _resolve_inherits(_load_config("e2_r1b_dqi_oracle_gate.json"))
    config["hidden_lookahead_slots"] = 10_000  # ensure 10s future window
    config["slots"] = 5_000
    results: dict[str, list[float]] = {
        "seed": [],
        "spearman_rho": [],
        "kendall_tau": [],
    }

    all_future: list[float] = []
    all_dqi: list[float] = []

    for seed in [1, 2, 3]:
        trace = generate_multi_vehicle_trace(config, seed=seed, slots=5_000, split="rl_eval_0")
        # Sample at every DQI update period to get independent (future, DQI) pairs
        update_slots = max(int(config.get("descriptors", {}).get("dqi_update_slots", 1_000)), 1)
        horizon = 10_000
        lookahead = trace.quality_lookahead_mbps  # (hidden_lookahead, V, M)
        dqi = trace.fresh_dqi  # (slots, M)

        # Sample at each DQI source slot
        for t in range(0, trace.slots, update_slots):
            if t + horizon > lookahead.shape[0]:
                break
            future_window = lookahead[t : t + horizon]  # (horizon, V, M)
            goodput = latency_constrained_goodput(future_window)
            shared = np.quantile(
                goodput, 0.10, axis=1, method='linear'
            )
            future_p10 = np.quantile(
                shared, 0.10, axis=0, method='linear'
            )
            dqi_at_t = dqi[t]
            for ch in range(trace.num_channels):
                fv = float(future_p10[ch])
                dv = float(dqi_at_t[ch])
                if np.isfinite(fv) and np.isfinite(dv):
                    all_future.append(fv)
                    all_dqi.append(dv)

    # Compute correlations across all seeds pooled
    if len(all_future) < 3:
        results["seed"] = [1]
        results["spearman_rho"] = [float("nan")]
        results["kendall_tau"] = [float("nan")]
        return results

    flat_future = np.array(all_future)
    flat_dqi = np.array(all_dqi)
    rho, _ = spearmanr(flat_future, flat_dqi)
    tau, _ = kendalltau(flat_future, flat_dqi)

    # Store per-seed results (we pool here since seeds are IID)
    for seed in [1, 2, 3]:
        results["seed"].append(seed)
        results["spearman_rho"].append(rho)
        results["kendall_tau"].append(tau)

    # ---- Scatter plot ----
    apply_ieee_strict_style()
    fig, ax = plt.subplots(figsize=FIG_IEEE_DOUBLE)
    ax.scatter(
        all_dqi,
        all_future,
        c=PALETTE["blue"],
        s=6,
        alpha=0.35,
        edgecolors="none",
    )
    ax.set_xlabel("DQI (fresh causal quantile estimate)")
    ax.set_ylabel("Future 10s p10 quality (Mbps)")
    ax.set_title(
        f"DQI vs future channel quality  |  "
        f"Spearman $\\rho$ = {rho:+.3f}  |  "
        f"Kendall $\\tau$ = {tau:+.3f}"
    )
    fig.subplots_adjust(left=0.13, right=0.98, bottom=0.15, top=0.90)
    save_ieee_figure(fig, str(out_dir / "e0_dqi_quality_scatter"), formats=("eps", "pdf"), tight=False)
    jpg = out_dir / "e0_dqi_quality_scatter.jpg"
    with plt.rc_context({"savefig.bbox": None}):
        fig.savefig(jpg, format="jpg", dpi=600, bbox_inches=None, pad_inches=0.0,
                    pil_kwargs={"quality": 95, "subsampling": 0})
    plt.close(fig)
    return results


# ---------------------------------------------------------------------------
# Theta validation (M=1, causal theta from P0-R2 config)
# ---------------------------------------------------------------------------
def validate_theta(out_dir: Path) -> dict:
    """Generate M=1 traces with causal theta and compute calibration metrics."""
    config = _resolve_inherits(_load_config("e2_p0_prediction_positive_control_r2.json"))
    config["slots"] = 5_000
    results: dict[str, list[float]] = {
        "seed": [],
        "brier_score": [],
        "ece": [],
    }

    all_idle: list[np.ndarray] = []
    all_theta: list[np.ndarray] = []

    for seed in [1, 2, 3]:
        trace = generate_multi_vehicle_trace(config, seed=seed, slots=5_000, split="rl_eval_0")
        actual_idle = (~trace.occupancy).astype(float).flatten()  # (slots, M) -> flat
        theta = trace.theta.flatten()

        valid = np.isfinite(actual_idle) & np.isfinite(theta)
        brier = _brier_score(actual_idle[valid], theta[valid])
        ece = _expected_calibration_error(actual_idle[valid], theta[valid], n_bins=10)

        results["seed"].append(seed)
        results["brier_score"].append(brier)
        results["ece"].append(ece)
        all_idle.append(actual_idle[valid])
        all_theta.append(theta[valid])

    # ---- Calibration curve ----
    apply_ieee_strict_style()
    fig, ax = plt.subplots(figsize=FIG_IEEE_DOUBLE)
    combined_idle = np.concatenate(all_idle)
    combined_theta = np.concatenate(all_theta)

    # Bin theta predictions into 10 equal-width bins and compute observed idle frequency
    n_bins = 10
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    observed = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins, dtype=int)
    for i in range(n_bins):
        mask = (combined_theta >= bin_edges[i]) & (combined_theta < bin_edges[i + 1])
        bin_counts[i] = mask.sum()
        if mask.sum() > 0:
            observed[i] = combined_idle[mask].mean()

    # Only plot bins with data
    has_data = bin_counts > 0
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Perfect calibration")
    ax.plot(
        bin_centers[has_data],
        observed[has_data],
        color=PALETTE["blue"],
        marker="o",
        markersize=4,
        linewidth=1.2,
        label="Causal theta",
    )
    mean_brier = np.mean(results["brier_score"])
    mean_ece = np.mean(results["ece"])
    ax.set_xlabel("Predicted idle probability (theta)")
    ax.set_ylabel("Observed idle frequency")
    ax.set_title(
        f"Theta reliability calibration  |  "
        f"Brier = {mean_brier:.4f}  |  "
        f"ECE = {mean_ece:.4f}"
    )
    ax.legend(loc="upper left")
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.15, top=0.90)
    save_ieee_figure(fig, str(out_dir / "e0_theta_calibration"), formats=("eps", "pdf"), tight=False)
    jpg = out_dir / "e0_theta_calibration.jpg"
    with plt.rc_context({"savefig.bbox": None}):
        fig.savefig(jpg, format="jpg", dpi=600, bbox_inches=None, pad_inches=0.0,
                    pil_kwargs={"quality": 95, "subsampling": 0})
    plt.close(fig)
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    out_dir = PROJECT_ROOT / "reports" / "e0_perception_validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = out_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    print("=== E0: DQI perception validation (M=10) ===")
    dqi_results = validate_dqi(figures_dir)
    print(f"  Spearman ρ: {np.mean(dqi_results['spearman_rho']):+.4f}")
    print(f"  Kendall τ:  {np.mean(dqi_results['kendall_tau']):+.4f}")

    print()
    print("=== E0: Theta perception validation (M=1) ===")
    theta_results = validate_theta(figures_dir)
    print(f"  Brier score: {np.mean(theta_results['brier_score']):.4f}")
    print(f"  ECE:         {np.mean(theta_results['ece']):.4f}")

    # ---- Metrics table ----
    rows = []
    for seed_idx, seed in enumerate(dqi_results["seed"]):
        rows.append(
            {
                "seed": int(seed),
                "descriptor": "DQI",
                "metric": "spearman_rho",
                "value": dqi_results["spearman_rho"][seed_idx],
            }
        )
        rows.append(
            {
                "seed": int(seed),
                "descriptor": "DQI",
                "metric": "kendall_tau",
                "value": dqi_results["kendall_tau"][seed_idx],
            }
        )
    for seed_idx, seed in enumerate(theta_results["seed"]):
        rows.append(
            {
                "seed": int(seed),
                "descriptor": "theta",
                "metric": "brier_score",
                "value": theta_results["brier_score"][seed_idx],
            }
        )
        rows.append(
            {
                "seed": int(seed),
                "descriptor": "theta",
                "metric": "ece",
                "value": theta_results["ece"][seed_idx],
            }
        )
    df = pd.DataFrame(rows)
    table_path = tables_dir / "e0_perception_metrics.csv"
    df.to_csv(table_path, index=False)
    print(f"\nMetrics saved to {table_path}")
    print("=== E0 complete ===")


if __name__ == "__main__":
    main()
