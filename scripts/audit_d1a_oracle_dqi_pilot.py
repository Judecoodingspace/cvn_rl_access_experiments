#!/usr/bin/env python3
"""D1-A Oracle DQI RL positive-control pilot — standalone audit script.

Reads the RL metrics CSV written by ``run_experiment``, computes paired
bootstrap effects across the three information conditions (oracle / neutral /
channel-placebo), builds a gate report, manifest, and comparison figures.

Usage::

    .venv/Scripts/python.exe scripts/audit_d1a_oracle_dqi_pilot.py \
        --run-dir results/d1a_test/<run_id> \
        --output-dir results/d1a_oracle_dqi_rl_pilot
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
R_REF = 17.377581082644856
GAMMA_D1A = 2.0 ** (-1.0 / 100.0)
ORACLE = "dqi_future_oracle"
CAUSAL = "dqi_predicted_causal"
FROZEN = "dqi_attention_frozen"
PERSISTENCE = "dqi_canonical_persistence"
NEUTRAL = "dqi_neutral"
CHANNEL_PLACEBO = "dqi_channel_placebo"
TIME_PLACEBO = "dqi_time_placebo"
OUTCOMES = [
    "served_goodput_mbps",
    "rolling_100ms_system_served_q10_mean",
]
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED_BASE = 20260708
MIN_SEED_WINS = 4  # at least 4/5 seeds positive
MIN_RELATIVE_GAIN = 0.02


# ---------------------------------------------------------------------------
# Paired bootstrap helpers
# ---------------------------------------------------------------------------

def _paired_bootstrap_ci(
    deltas: np.ndarray,
    *,
    n_samples: int = BOOTSTRAP_SAMPLES,
    seed_offset: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Paired bootstrap 95% CI for the mean of per-seed differences."""
    rng = np.random.default_rng(BOOTSTRAP_SEED_BASE + seed_offset)
    n = len(deltas)
    means = np.empty(n_samples)
    for i in range(n_samples):
        idx = rng.choice(n, size=n, replace=True)
        means[i] = np.mean(deltas[idx])
    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1.0 - alpha / 2))
    return lo, hi


def _paired_effect_row(
    condition: str,
    comparator: str,
    outcome: str,
    oracle_vals: np.ndarray,
    comp_vals: np.ndarray,
    seed_offset: int = 0,
) -> dict[str, Any]:
    """Compute one row of the paired-effects table."""
    diffs = oracle_vals - comp_vals
    rel_gains = diffs / np.maximum(np.abs(comp_vals), 1e-9)
    ci_lo, ci_hi = _paired_bootstrap_ci(diffs, seed_offset=seed_offset)
    rel_lo, rel_hi = _paired_bootstrap_ci(
        rel_gains, seed_offset=seed_offset + 10_000,
    )
    return {
        "condition": condition,
        "comparator": comparator,
        "outcome": outcome,
        "mean_oracle": float(np.mean(oracle_vals)),
        "mean_comparator": float(np.mean(comp_vals)),
        "mean_difference": float(np.mean(diffs)),
        "relative_gain": float(np.mean(rel_gains)),
        "ci95_low": ci_lo,
        "ci95_high": ci_hi,
        "ci95_low_relative": rel_lo,
        "ci95_high_relative": rel_hi,
        "seed_wins": int(np.sum(diffs > 0)),
        "n_seeds": len(diffs),
    }


# ---------------------------------------------------------------------------
# Gate construction
# ---------------------------------------------------------------------------


def _append_effect_check(
    checks: list[dict[str, Any]],
    effects: pd.DataFrame,
    *,
    check: str,
    condition: str,
    comparator: str,
    outcome: str,
    require_relative_gain: bool = False,
) -> None:
    eff = effects[
        (effects["condition"] == condition)
        & (effects["comparator"] == comparator)
        & (effects["outcome"] == outcome)
    ]
    if len(eff) != 1:
        checks.append({
            "check": check,
            "passed": False,
            "detail": "missing paired effect row",
        })
        return
    row = eff.iloc[0]
    passed = bool(
        row["seed_wins"] >= MIN_SEED_WINS
        and row["ci95_low"] > 0
        and (
            not require_relative_gain
            or row["relative_gain"] >= MIN_RELATIVE_GAIN
        )
    )
    gain = f", gain={row['relative_gain']:.4f}" if require_relative_gain else ""
    checks.append({
        "check": check,
        "passed": passed,
        "detail": (
            f"diff={row['mean_difference']:.4f}{gain}, "
            f"wins={row['seed_wins']}/5, "
            f"CI95=[{row['ci95_low']:.4f}, {row['ci95_high']:.4f}]"
        ),
    })

def _build_gate(
    metrics: pd.DataFrame,
    effects: pd.DataFrame,
    config_hash: str,
) -> dict[str, Any]:
    """Build the D1-A pilot gate report."""
    checks: list[dict[str, Any]] = []

    # --- Physical trace identity (training only; eval traces differ by design) ---
    for seed, group in metrics.groupby("seed"):
        train_nunique = group["train_physical_trace_hash"].nunique()
        checks.append({
            "check": f"train_physical_trace_seed_{seed}",
            "passed": bool(train_nunique == 1),
            "detail": f"nunique = {train_nunique}",
        })

    available_variants = set(metrics["module_variant"].astype(str).unique())
    _append_effect_check(
        checks, effects,
        check="goodput_oracle_vs_neutral",
        condition=ORACLE,
        comparator=NEUTRAL,
        outcome="served_goodput_mbps",
        require_relative_gain=True,
    )
    _append_effect_check(
        checks, effects,
        check="rolling_tail_oracle_vs_neutral",
        condition=ORACLE,
        comparator=NEUTRAL,
        outcome="rolling_100ms_system_served_q10_mean",
    )
    _append_effect_check(
        checks, effects,
        check="goodput_oracle_vs_channel_placebo",
        condition=ORACLE,
        comparator=CHANNEL_PLACEBO,
        outcome="served_goodput_mbps",
    )
    _append_effect_check(
        checks, effects,
        check="rolling_tail_oracle_vs_channel_placebo",
        condition=ORACLE,
        comparator=CHANNEL_PLACEBO,
        outcome="rolling_100ms_system_served_q10_mean",
    )
    if TIME_PLACEBO in available_variants:
        _append_effect_check(
            checks, effects,
            check="goodput_oracle_vs_time_placebo",
            condition=ORACLE,
            comparator=TIME_PLACEBO,
            outcome="served_goodput_mbps",
        )
        _append_effect_check(
            checks, effects,
            check="rolling_tail_oracle_vs_time_placebo",
            condition=ORACLE,
            comparator=TIME_PLACEBO,
            outcome="rolling_100ms_system_served_q10_mean",
        )
    if CAUSAL in available_variants:
        _append_effect_check(
            checks, effects,
            check="goodput_oracle_vs_causal_gap",
            condition=ORACLE,
            comparator=CAUSAL,
            outcome="served_goodput_mbps",
        )
        _append_effect_check(
            checks, effects,
            check="goodput_causal_vs_neutral",
            condition=CAUSAL,
            comparator=NEUTRAL,
            outcome="served_goodput_mbps",
        )
        _append_effect_check(
            checks, effects,
            check="goodput_causal_vs_channel_placebo",
            condition=CAUSAL,
            comparator=CHANNEL_PLACEBO,
            outcome="served_goodput_mbps",
        )
        _append_effect_check(
            checks, effects,
            check="rolling_tail_causal_vs_neutral",
            condition=CAUSAL,
            comparator=NEUTRAL,
            outcome="rolling_100ms_system_served_q10_mean",
        )
    if FROZEN in available_variants:
        _append_effect_check(
            checks, effects,
            check="goodput_oracle_vs_attention_frozen_gap",
            condition=ORACLE,
            comparator=FROZEN,
            outcome="served_goodput_mbps",
        )
        _append_effect_check(
            checks, effects,
            check="goodput_attention_frozen_vs_neutral",
            condition=FROZEN,
            comparator=NEUTRAL,
            outcome="served_goodput_mbps",
            require_relative_gain=True,
        )
        _append_effect_check(
            checks, effects,
            check="goodput_attention_frozen_vs_channel_placebo",
            condition=FROZEN,
            comparator=CHANNEL_PLACEBO,
            outcome="served_goodput_mbps",
        )
        if PERSISTENCE in available_variants:
            _append_effect_check(
                checks, effects,
                check="goodput_attention_frozen_vs_persistence",
                condition=FROZEN,
                comparator=PERSISTENCE,
                outcome="served_goodput_mbps",
            )
        _append_effect_check(
            checks, effects,
            check="rolling_tail_attention_frozen_vs_neutral",
            condition=FROZEN,
            comparator=NEUTRAL,
            outcome="rolling_100ms_system_served_q10_mean",
        )


    # --- Dwell violation ---
    dwell_sum = int(metrics["dwell_violation_count"].sum())
    checks.append({
        "check": "dwell_violation_zero",
        "passed": bool(dwell_sum == 0),
        "detail": f"total dwell violations = {dwell_sum}",
    })

    # --- Observation invariants ---
    obs_dim = metrics["observation_dimension"].nunique()
    state_sig = metrics["state_feature_signature"].nunique()
    checks.append({
        "check": "observation_dimension_constant",
        "passed": bool(obs_dim == 1),
        "detail": f"nunique = {obs_dim}",
    })
    checks.append({
        "check": "state_feature_signature_constant",
        "passed": bool(state_sig == 1),
        "detail": f"nunique = {state_sig}",
    })

    # --- Action mask invariant (check per-condition consistency across seeds) ---
    mask_hash_ok = True
    for mv in metrics["module_variant"].astype(str).unique():
        sub = metrics[metrics["module_variant"].astype(str) == mv]
        if sub["raw_action_mask_hash"].nunique() > 1:
            mask_hash_ok = False
    checks.append({
        "check": "action_mask_hash_consistent_per_condition",
        "passed": mask_hash_ok,
        "detail": "mask hash consistent within each condition",
    })

    # --- Evaluation parameters frozen ---
    all_frozen = bool(metrics["evaluation_parameters_frozen"].all())
    checks.append({
        "check": "evaluation_parameters_frozen",
        "passed": all_frozen,
        "detail": f"all frozen = {all_frozen}",
    })

    # --- Gamma matches config ---
    gamma_match = bool(
        np.allclose(metrics["gamma"].dropna().unique(), GAMMA_D1A, rtol=1e-9)
    )
    checks.append({
        "check": "gamma_matches_d1a_config",
        "passed": gamma_match,
        "detail": f"expected {GAMMA_D1A}, got {metrics['gamma'].unique().tolist()}",
    })

    # --- n_seeds check ---
    n_seeds = metrics["seed"].nunique()
    checks.append({
        "check": "paired_seeds_count",
        "passed": bool(n_seeds == 5),
        "detail": f"n_seeds = {n_seeds}",
    })

    passed = all(c["passed"] for c in checks)
    return {
        "experiment": (
            "d3_attention_dqi_frozen"
            if FROZEN in available_variants
            else "d2_causal_dqi_gap"
            if CAUSAL in available_variants
            else "d1a_oracle_dqi_rl_pilot"
        ),
        "passed": passed,
        "config_hash": config_hash,
        "r_ref_mbps": R_REF,
        "gamma_d1a": GAMMA_D1A,
        "checks": checks,
    }


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def _build_manifest(
    metrics: pd.DataFrame,
    config_hash: str,
    run_dir: str,
) -> dict[str, Any]:
    seeds = sorted(metrics["seed"].unique().tolist())
    train_hashes: dict[int, str] = {}
    eval_hashes: dict[int, str] = {}
    for seed in seeds:
        sub = metrics[metrics["seed"] == seed]
        train_hashes[int(seed)] = str(sub["train_physical_trace_hash"].iloc[0])
        eval_hashes[int(seed)] = str(sub["evaluation_physical_trace_hash"].iloc[0])
    return {
        "experiment": (
            "d3_attention_dqi_frozen"
            if FROZEN in set(metrics["module_variant"].astype(str).unique())
            else "d2_causal_dqi_gap"
            if CAUSAL in set(metrics["module_variant"].astype(str).unique())
            else "d1a_oracle_dqi_rl_pilot"
        ),
        "config_hash": config_hash,
        "r_ref_mbps": R_REF,
        "reference_source": "d0_6_train_z_q95",
        "gamma_d1a": GAMMA_D1A,
        "seeds": seeds,
        "conditions": sorted(metrics["module_variant"].astype(str).unique().tolist()),
        "train_physical_trace_hashes": train_hashes,
        "eval_physical_trace_hashes": eval_hashes,
        "rl_run_dir": str(run_dir),
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _make_figures(
    metrics: pd.DataFrame,
    effects: pd.DataFrame,
    output_dir: Path,
    prefix: str = "d1a",
) -> None:
    """Generate the three D1-A comparison PDFs."""
    final = metrics[metrics["is_final_checkpoint"].astype(bool)].copy()
    all_metrics = metrics.copy()
    # Normalise column types
    for df in (final, all_metrics):
        if "module_variant" in df.columns:
            df["module_variant"] = df["module_variant"].astype(str)

    # ---- goodput convergence ----
    fig, ax = plt.subplots(figsize=(8, 5))
    for variant, label, color in [
        (ORACLE, "Oracle DQI", "#1b9e77"),
        (FROZEN, "Attention Frozen", "#3288bd"),
        (PERSISTENCE, "Persistence", "#66a61e"),
        (CAUSAL, "Causal DQI", "#a6d854"),
        (NEUTRAL, "Neutral DQI", "#7570b3"),
        (CHANNEL_PLACEBO, "Channel Placebo", "#d95f02"),
        (TIME_PLACEBO, "Time Placebo", "#e7298a"),
    ]:
        sub = all_metrics[all_metrics["module_variant"].astype(str) == variant]
        if sub.empty:
            continue
        piv = sub.pivot_table(
            values="served_goodput_mbps",
            index="checkpoint_slot",
            aggfunc=["mean", "std"],
        )
        slots_axis = piv.index.values
        # pivot_table with single-value group may drop "std" column
        if isinstance(piv.columns, pd.MultiIndex):
            mean_col = ("mean", "served_goodput_mbps")
            std_col = ("std", "served_goodput_mbps")
            means = piv[mean_col].values if mean_col in piv.columns else piv.iloc[:, 0].values
            stds = piv[std_col].values if std_col in piv.columns else np.zeros_like(means)
        else:
            means = piv.values.ravel()
            stds = np.zeros_like(means)
        ax.plot(slots_axis, means, label=label, color=color)
        if len(slots_axis) > 1:
            ax.fill_between(slots_axis, means - stds, means + stds, alpha=0.15, color=color)
    ax.set_xlabel("Training slot")
    ax.set_ylabel("Served goodput (Mbps)")
    ax.set_title("D1-A Goodput Convergence (mean ± std over seeds)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / f"{prefix}_goodput_convergence.pdf")
    plt.close(fig)

    # ---- mean-tail comparison ----
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax_i, (outcome, title) in enumerate([
        ("served_goodput_mbps", "Mean served goodput"),
        ("rolling_100ms_system_served_q10_mean", "Rolling 100ms q10 mean"),
    ]):
        ax = axes[ax_i]
        variants = [ORACLE, FROZEN, PERSISTENCE, CAUSAL, NEUTRAL, CHANNEL_PLACEBO, TIME_PLACEBO]
        labels = ["Oracle", "Frozen", "Persist", "Causal", "Neutral", "Ch.P.", "T.P."]
        colors = ["#1b9e77", "#3288bd", "#66a61e", "#a6d854", "#7570b3", "#d95f02", "#e7298a"]
        means_list = []
        for variant in variants:
            vals = final[final["module_variant"] == variant][outcome].values
            means_list.append(np.mean(vals) if len(vals) else np.nan)
        x = np.arange(len(variants))
        bars = ax.bar(x, means_list, color=colors, alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis="y")
    fig.suptitle("D1-A Final Evaluation: Mean vs Tail")
    fig.tight_layout()
    fig.savefig(output_dir / f"{prefix}_mean_tail_comparison.pdf")
    plt.close(fig)

    # ---- switching / blackout comparison ----
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, column, title in [
        (axes[0, 0], "switch_count", "Switch count"),
        (axes[0, 1], "blackout_backlog_slots", "Blackout backlog slots"),
        (axes[1, 0], "blackout_loss_bits", "Blackout loss (bits)"),
        (axes[1, 1], "dwell_violation_count", "Dwell violations"),
    ]:
        variants = [ORACLE, FROZEN, PERSISTENCE, CAUSAL, NEUTRAL, CHANNEL_PLACEBO, TIME_PLACEBO]
        labels = ["Oracle", "Frozen", "Persist", "Causal", "Neutral", "Ch.P.", "T.P."]
        colors = ["#1b9e77", "#3288bd", "#66a61e", "#a6d854", "#7570b3", "#d95f02", "#e7298a"]
        means_list = []
        stds_list = []
        for variant in variants:
            vals = final[final["module_variant"] == variant][column].values
            means_list.append(np.mean(vals) if len(vals) else np.nan)
            stds_list.append(np.std(vals) if len(vals) else np.nan)
        x = np.arange(len(variants))
        ax.bar(x, means_list, yerr=stds_list, color=colors, alpha=0.8, capsize=4)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis="y")
    fig.suptitle("D1-A Switching and Blackout Metrics")
    fig.tight_layout()
    fig.savefig(output_dir / f"{prefix}_switching_blackout_comparison.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _paired_effects(metrics: pd.DataFrame) -> pd.DataFrame:
    """Compute paired effects: oracle minus each comparator, per outcome."""
    final = metrics[metrics["is_final_checkpoint"].astype(bool)].copy()
    # Normalise string columns to plain Python str (ArrowStringArray can
    # confuse pivot_table column look-ups).
    for col in ("module_variant",):
        if col in final.columns:
            final[col] = final[col].astype(str)
    rows: list[dict[str, Any]] = []
    for outcome in OUTCOMES:
        if outcome not in final.columns:
            continue
        pivot = final.pivot_table(
            values=outcome,
            index="seed",
            columns="module_variant",
        )
        # Ensure column names are plain str
        pivot.columns = [str(c) for c in pivot.columns]
        if ORACLE not in pivot.columns:
            continue
        oracle_seeds = pivot[ORACLE].values
        for comparator, offset in [
            (NEUTRAL, 0),
            (CHANNEL_PLACEBO, 50),
            (TIME_PLACEBO, 100),
            (CAUSAL, 150),
            (FROZEN, 300),
            (PERSISTENCE, 350),
        ]:
            if comparator not in pivot.columns:
                continue
            comp_seeds = pivot[comparator].values
            rows.append(_paired_effect_row(
                ORACLE, comparator, outcome,
                oracle_seeds, comp_seeds,
                seed_offset=offset,
            ))
        if CAUSAL in pivot.columns:
            causal_seeds = pivot[CAUSAL].values
            for comparator, offset in [(NEUTRAL, 200), (CHANNEL_PLACEBO, 250)]:
                if comparator not in pivot.columns:
                    continue
                rows.append(_paired_effect_row(
                    CAUSAL, comparator, outcome,
                    causal_seeds, pivot[comparator].values,
                    seed_offset=offset,
                ))
        if FROZEN in pivot.columns:
            frozen_seeds = pivot[FROZEN].values
            for comparator, offset in [(NEUTRAL, 400), (CHANNEL_PLACEBO, 450), (PERSISTENCE, 500)]:
                if comparator not in pivot.columns:
                    continue
                rows.append(_paired_effect_row(
                    FROZEN, comparator, outcome,
                    frozen_seeds, pivot[comparator].values,
                    seed_offset=offset,
                ))
    return pd.DataFrame(rows)


def run_audit(run_dir: str, output_dir: str) -> int:
    run_path = Path(run_dir)
    metrics_path = run_path / "metrics" / "access_metrics.csv"
    if not metrics_path.exists():
        print(f"ERROR: {metrics_path} not found", file=sys.stderr)
        return 1

    metrics = pd.read_csv(metrics_path)
    if metrics.empty:
        print("ERROR: empty metrics CSV", file=sys.stderr)
        return 1

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Config hash
    config_path = run_path / "config.frozen.json"
    config_hash = "unknown"
    if config_path.exists():
        import hashlib
        config_hash = hashlib.sha256(
            config_path.read_bytes()
        ).hexdigest()[:16]

    variants = set(metrics["module_variant"].astype(str).unique())
    prefix = (
        "d3"
        if FROZEN in variants
        else "d2"
        if CAUSAL in variants
        else "d1a"
    )

    # Paired effects
    effects = _paired_effects(metrics)
    effects_path = out / f"{prefix}_pairwise_effects.csv"
    effects.to_csv(effects_path, index=False)
    print(f"  -> {effects_path}")

    # Gate
    gate = _build_gate(metrics, effects, config_hash)
    gate_path = out / f"{prefix}_gate_report.json"
    gate_path.write_text(
        json.dumps(gate, indent=2, default=str), encoding="utf-8",
    )
    print(f"  -> {gate_path}")
    print(f"  GATE: {'PASSED' if gate['passed'] else 'FAILED'}")
    if not gate["passed"]:
        for c in gate["checks"]:
            if not c["passed"]:
                print(f"    FAIL: {c['check']} — {c['detail']}")

    # Manifest
    manifest = _build_manifest(metrics, config_hash, run_dir)
    manifest_path = out / f"{prefix}_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8",
    )
    print(f"  -> {manifest_path}")

    # Figures
    _make_figures(metrics, effects, out, prefix)
    print(f"  -> {out / f'{prefix}_goodput_convergence.pdf'}")
    print(f"  -> {out / f'{prefix}_mean_tail_comparison.pdf'}")
    print(f"  -> {out / f'{prefix}_switching_blackout_comparison.pdf'}")

    return 0  # gate pass/fail is in the JSON, exit code is for script errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="D1-A Oracle DQI RL pilot audit",
    )
    parser.add_argument(
        "--run-dir", required=True,
        help="Path to the RL run directory (contains metrics/access_metrics.csv).",
    )
    parser.add_argument(
        "--output-dir",
        default="results/d1a_oracle_dqi_rl_pilot",
        help="Directory for audit artifacts.",
    )
    args = parser.parse_args(argv)
    return run_audit(args.run_dir, args.output_dir)


if __name__ == "__main__":
    raise SystemExit(main())
