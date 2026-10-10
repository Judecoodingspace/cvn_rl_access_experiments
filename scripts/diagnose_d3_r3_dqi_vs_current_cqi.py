#!/usr/bin/env python3
"""D3-R3 offline diagnostic: Attention DQI vs current-CQI proxies.

This diagnostic asks whether the learned D3 Attention DQI descriptor is merely
a substitute for current channel quality.  It uses the runner deployment path
(``_resolve_trace``) for the frozen Attention DQI, and compares its per-channel
ranking against several observable current-quality proxies using the canonical
future-oracle next-100-slot DQI as the target.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cvn_rl.experiments.config import load_config
from cvn_rl.experiments.multi_vehicle_runner import _future_quantile_dqi, _resolve_trace
from cvn_rl.simulation.dqi_labels import normalize_dqi


def _parse_ints(raw: str) -> list[int]:
    return [int(x) for x in raw.split(",") if x.strip()]


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(len(values), dtype=float)
    sorted_vals = values[order]
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and sorted_vals[stop] == sorted_vals[start]:
            stop += 1
        if stop - start > 1:
            ranks[order[start:stop]] = (start + stop - 1) / 2.0
        start = stop
    return ranks


def _spearman_per_row(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.full(a.shape[0], np.nan, dtype=float)
    for i, (x, y) in enumerate(zip(a, b)):
        if np.allclose(x, x[0]) or np.allclose(y, y[0]):
            continue
        rx = _rankdata(x)
        ry = _rankdata(y)
        rx -= rx.mean()
        ry -= ry.mean()
        denom = np.sqrt(float(np.sum(rx * rx) * np.sum(ry * ry)))
        if denom > 0:
            out[i] = float(np.sum(rx * ry) / denom)
    return out


def _finite_mean(values: np.ndarray) -> float:
    return float(np.nanmean(values)) if np.isfinite(values).any() else float("nan")


def _condition_config(config: dict[str, Any], condition_name: str) -> dict[str, Any]:
    for condition in config.get("module_conditions", []):
        if str(condition.get("name")) == condition_name:
            merged = {
                **config,
                **{k: v for k, v in condition.items() if k not in {"name", "descriptors"}},
            }
            merged["module_variant"] = condition_name
            merged["descriptors"] = {
                **config.get("descriptors", {}),
                **condition.get("descriptors", {}),
            }
            return merged
    raise ValueError(f"condition {condition_name!r} not found in config.module_conditions")


def _vehicle_q(values: np.ndarray, quantile: float) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim == 3:
        return np.quantile(values, quantile, axis=1, method="linear")
    if values.ndim == 2:
        return values
    raise ValueError(f"expected [slots, vehicles, channels] or [slots, channels], got {values.shape}")


def _rank_scaled(scores: np.ndarray) -> np.ndarray:
    out = np.empty_like(scores, dtype=float)
    denom = max(scores.shape[1] - 1, 1)
    for i, row in enumerate(scores):
        out[i] = _rankdata(row) / denom
    return out


def _method_metrics(
    *,
    seed: int,
    method: str,
    pred: np.ndarray,
    oracle: np.ndarray,
) -> dict[str, Any]:
    rho = _spearman_per_row(pred, oracle)
    top1 = np.argmax(pred, axis=1) == np.argmax(oracle, axis=1)
    mean_error = float(np.mean(pred) - np.mean(oracle))
    return {
        "seed": int(seed),
        "method": method,
        "samples": int(pred.shape[0]),
        "spearman_mean": _finite_mean(rho),
        "spearman_p50": float(np.nanmedian(rho)) if np.isfinite(rho).any() else float("nan"),
        "valid_spearman_fraction": float(np.isfinite(rho).mean()),
        "top1_agreement": float(np.mean(top1)),
        "mae": float(np.mean(np.abs(pred - oracle))),
        "mse": float(np.mean((pred - oracle) ** 2)),
        "predicted_mean": float(np.mean(pred)),
        "future_mean": float(np.mean(oracle)),
        "mean_error": mean_error,
        "abs_mean_error": abs(mean_error),
        "predicted_std": float(np.std(pred)),
        "future_std": float(np.std(oracle)),
        "predicted_p10": float(np.quantile(pred, 0.10)),
        "predicted_p50": float(np.quantile(pred, 0.50)),
        "predicted_p90": float(np.quantile(pred, 0.90)),
        "future_p10": float(np.quantile(oracle, 0.10)),
        "future_p50": float(np.quantile(oracle, 0.50)),
        "future_p90": float(np.quantile(oracle, 0.90)),
    }


def _paired_advantages(per_seed: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    metrics = ["spearman_mean", "top1_agreement", "mae", "mse", "abs_mean_error"]
    methods = sorted(set(per_seed["method"]) - {"attention_frozen"})
    for metric in metrics:
        pivot = per_seed.pivot_table(index="seed", columns="method", values=metric, aggfunc="mean")
        if "attention_frozen" not in pivot:
            continue
        for method in methods:
            if method not in pivot:
                continue
            sub = pivot[["attention_frozen", method]].dropna()
            if sub.empty:
                continue
            diff = sub["attention_frozen"] - sub[method]
            rows.append({
                "metric": metric,
                "attention_minus_method": method,
                "n_seeds": int(len(diff)),
                "mean_difference": float(diff.mean()),
                "seed_wins": int((diff > 0).sum()) if metric not in {"mae", "mse", "abs_mean_error"} else int((diff < 0).sum()),
                "attention_mean": float(sub["attention_frozen"].mean()),
                "method_mean": float(sub[method].mean()),
            })
    return pd.DataFrame(rows)


def _summarize(per_seed: pd.DataFrame) -> pd.DataFrame:
    return (
        per_seed.groupby("method", as_index=False)
        .agg(
            spearman_mean=("spearman_mean", "mean"),
            spearman_std=("spearman_mean", "std"),
            top1_agreement_mean=("top1_agreement", "mean"),
            top1_agreement_std=("top1_agreement", "std"),
            mae_mean=("mae", "mean"),
            mse_mean=("mse", "mean"),
            abs_mean_error_mean=("abs_mean_error", "mean"),
            predicted_mean=("predicted_mean", "mean"),
            future_mean=("future_mean", "mean"),
            seeds=("seed", "nunique"),
        )
        .sort_values(["spearman_mean", "top1_agreement_mean"], ascending=False)
    )


def _write_markdown(path: Path, payload: dict[str, Any], summary: pd.DataFrame, advantages: pd.DataFrame) -> None:
    lines = [
        "# D3-R3 Offline Diagnostic: Attention DQI vs Current CQI",
        "",
        f"config: `{payload['config']}`",
        f"condition: `{payload['condition']}`",
        f"seeds: `{payload['seeds']}`",
        f"slots: `{payload['slots']}`",
        "",
        "## Summary",
        "",
        "| Method | Spearman | Top1 | MAE | Abs mean error |",
        "|---|---:|---:|---:|---:|",
    ]
    for _, r in summary.iterrows():
        lines.append(
            f"| {r['method']} | {r['spearman_mean']:.4f} | "
            f"{r['top1_agreement_mean']:.4f} | {r['mae_mean']:.4f} | "
            f"{r['abs_mean_error_mean']:.4f} |"
        )
    lines += [
        "",
        "## Attention advantage over current-quality proxies",
        "",
        "| Metric | Comparator | Mean diff | Wins |",
        "|---|---|---:|---:|",
    ]
    for _, r in advantages.iterrows():
        lines.append(
            f"| {r['metric']} | {r['attention_minus_method']} | "
            f"{r['mean_difference']:.4f} | {int(r['seed_wins'])}/{int(r['n_seeds'])} |"
        )
    lines += [
        "",
        "Interpretation: if Attention DQI has materially higher Spearman/Top1 than "
        "current-quality proxies, its value is not simply current CQI substitution.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    cfg = _condition_config(config, args.condition)
    descriptor_cfg = cfg["descriptors"]
    seeds = _parse_ints(args.seeds)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for seed in seeds:
        trace = _resolve_trace(
            cfg,
            args.scenario,
            seed=int(seed),
            slots=int(args.slots),
            split=args.split,
        )
        future = trace.future_dqi_oracle
        if not np.isfinite(future).any():
            future = _future_quantile_dqi(trace, descriptor_cfg)

        stride = max(int(args.sample_stride), 1)
        start = max(int(args.start_slot), 0)
        idx = np.arange(start, trace.slots, stride, dtype=int)
        oracle = future[idx]

        quantile = float(descriptor_cfg.get("dqi_quantile", args.vehicle_quantile))
        r_ref = float(descriptor_cfg.get("r_ref_mbps"))
        current_quality = _vehicle_q(trace.quality_mbps, quantile)
        lag1_quality = np.empty_like(current_quality)
        lag1_quality[0] = current_quality[0]
        lag1_quality[1:] = current_quality[:-1]

        current_dqi = normalize_dqi(current_quality, r_ref)
        lag1_dqi = normalize_dqi(lag1_quality, r_ref)
        methods = {
            "attention_frozen": trace.fresh_dqi[idx],
            "current_quality_q10": current_dqi[idx],
            "lag1_quality_q10": lag1_dqi[idx],
            "current_quality_rank": _rank_scaled(current_quality)[idx],
            "lag1_quality_rank": _rank_scaled(lag1_quality)[idx],
        }
        for method, pred in methods.items():
            rows.append(_method_metrics(seed=seed, method=method, pred=pred, oracle=oracle))
        print(
            f"seed={seed} "
            f"attention_spearman={rows[-5]['spearman_mean']:.4f} "
            f"current_spearman={rows[-4]['spearman_mean']:.4f}",
            flush=True,
        )

    per_seed = pd.DataFrame(rows)
    summary = _summarize(per_seed)
    advantages = _paired_advantages(per_seed)

    payload = {
        "config": str(Path(args.config)),
        "condition": str(args.condition),
        "scenario": str(args.scenario),
        "seeds": seeds,
        "slots": int(args.slots),
        "sample_stride": int(args.sample_stride),
        "start_slot": int(args.start_slot),
        "target": "canonical future_oracle next-100-slot DQI",
        "methods": {
            "attention_frozen": "runner deployment-path frozen Attention DQI in trace.fresh_dqi",
            "current_quality_q10": "current-slot q10 quality_mbps across vehicles, normalized by r_ref",
            "lag1_quality_q10": "previous-slot q10 quality_mbps, normalized by r_ref",
            "current_quality_rank": "rank-only current quality baseline scaled to [0,1]",
            "lag1_quality_rank": "rank-only previous-slot quality baseline scaled to [0,1]",
        },
    }
    per_seed.to_csv(out / "d3_r3_dqi_vs_current_cqi_per_seed.csv", index=False)
    summary.to_csv(out / "d3_r3_dqi_vs_current_cqi_summary.csv", index=False)
    advantages.to_csv(out / "d3_r3_attention_advantage.csv", index=False)
    (out / "d3_r3_dqi_vs_current_cqi_manifest.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    _write_markdown(out / "D3_R3_DQI_VS_CURRENT_CQI.md", payload, summary, advantages)
    print(f"wrote {out / 'D3_R3_DQI_VS_CURRENT_CQI.md'}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/e2_d3_r1_attention_dqi_frozen.json")
    parser.add_argument("--condition", default="dqi_attention_frozen")
    parser.add_argument("--scenario", default="M10_N1")
    parser.add_argument("--seeds", default="301,302,303,304,305")
    parser.add_argument("--slots", type=int, default=50_000)
    parser.add_argument("--sample-stride", type=int, default=10)
    parser.add_argument("--start-slot", type=int, default=200)
    parser.add_argument("--vehicle-quantile", type=float, default=0.10)
    parser.add_argument("--split", default="d3_r3_dqi_vs_current_cqi")
    parser.add_argument("--output-dir", default="results/d3_r3_dqi_vs_current_cqi/offline_diagnostic")
    raise SystemExit(run(parser.parse_args()))


if __name__ == "__main__":
    main()
