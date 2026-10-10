#!/usr/bin/env python3
"""D3 deployment-path gate for frozen Attention DQI.

This audit intentionally calls ``_resolve_trace()`` so it measures the exact DQI
values that the RL runner will inject into ``trace.fresh_dqi``.  It catches
train/deploy feature mismatches that an offline predictor evaluation cannot see.
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


def _parse_ints(raw: str) -> list[int]:
    return [int(x) for x in raw.split(',') if x.strip()]


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind='mergesort')
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
    return float(np.nanmean(values)) if np.isfinite(values).any() else float('nan')


def _condition_config(config: dict[str, Any], condition_name: str) -> dict[str, Any]:
    for condition in config.get('module_conditions', []):
        if str(condition.get('name')) == condition_name:
            merged = {
                **config,
                **{k: v for k, v in condition.items() if k not in {'name', 'descriptors'}},
            }
            merged['module_variant'] = condition_name
            merged['descriptors'] = {
                **config.get('descriptors', {}),
                **condition.get('descriptors', {}),
            }
            return merged
    raise ValueError(f'condition {condition_name!r} not found in config.module_conditions')


def _summarize(seed: int, trace, descriptor_cfg: dict[str, Any], sample_stride: int) -> dict[str, Any]:
    future = trace.future_dqi_oracle
    if not np.isfinite(future).any():
        future = _future_quantile_dqi(trace, descriptor_cfg)
    idx = np.arange(0, trace.slots, max(int(sample_stride), 1), dtype=int)
    pred = trace.fresh_dqi[idx]
    oracle = future[idx]
    rho = _spearman_per_row(pred, oracle)
    mean_error = float(np.mean(pred) - np.mean(oracle))
    return {
        'seed': int(seed),
        'samples': int(len(idx)),
        'spearman_mean': _finite_mean(rho),
        'spearman_p50': float(np.nanmedian(rho)) if np.isfinite(rho).any() else float('nan'),
        'valid_spearman_fraction': float(np.isfinite(rho).mean()),
        'top1_agreement': float((np.argmax(pred, axis=1) == np.argmax(oracle, axis=1)).mean()),
        'mae': float(np.mean(np.abs(pred - oracle))),
        'mse': float(np.mean((pred - oracle) ** 2)),
        'predicted_mean': float(np.mean(pred)),
        'future_mean': float(np.mean(oracle)),
        'mean_error': mean_error,
        'abs_mean_error': abs(mean_error),
        'predicted_std': float(np.std(pred)),
        'future_std': float(np.std(oracle)),
        'predicted_p10': float(np.quantile(pred, 0.10)),
        'predicted_p50': float(np.quantile(pred, 0.50)),
        'predicted_p90': float(np.quantile(pred, 0.90)),
        'future_p10': float(np.quantile(oracle, 0.10)),
        'future_p50': float(np.quantile(oracle, 0.50)),
        'future_p90': float(np.quantile(oracle, 0.90)),
        'dqi_model_id': str(trace.dqi_model_id),
        'dqi_model_hash': str(trace.dqi_model_hash),
        'dqi_validation_mse': float(trace.dqi_validation_mse),
    }


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    summary = payload['summary']
    lines = [
        '# D3-R1 Attention DQI Deployment Gate',
        '',
        f"config: `{payload['config']}`",
        f"condition: `{payload['condition']}`",
        f"model_hash: `{summary.get('dqi_model_hash', '')}`",
        '',
        '| metric | value |',
        '|---|---:|',
        f"| spearman_mean | {summary['spearman_mean']:.4f} |",
        f"| top1_agreement | {summary['top1_agreement']:.4f} |",
        f"| abs_mean_error | {summary['abs_mean_error']:.4f} |",
        f"| predicted_mean | {summary['predicted_mean']:.4f} |",
        f"| future_mean | {summary['future_mean']:.4f} |",
        f"| mae | {summary['mae']:.4f} |",
        '',
        'Gate:',
    ]
    for key, value in payload['gate'].items():
        lines.append(f'- {key}: `{value}`')
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def run(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    cfg = _condition_config(config, args.condition)
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
        row = _summarize(seed, trace, cfg['descriptors'], int(args.sample_stride))
        rows.append(row)
        print(
            f"seed={seed} spearman={row['spearman_mean']:.4f} "
            f"top1={row['top1_agreement']:.4f} abs_mean_error={row['abs_mean_error']:.4f}",
            flush=True,
        )
    per_seed = pd.DataFrame(rows)
    numeric = per_seed.select_dtypes(include=[np.number]).columns.tolist()
    summary = per_seed[numeric].mean(numeric_only=True).to_dict()
    for field in ('dqi_model_id', 'dqi_model_hash'):
        if field in per_seed:
            summary[field] = str(per_seed[field].iloc[0])
    gate = {
        'spearman_ge_threshold': bool(summary['spearman_mean'] >= float(args.gate_spearman)),
        'top1_ge_threshold': bool(summary['top1_agreement'] >= float(args.gate_top1)),
        'calibration_abs_error_le_threshold': bool(summary['abs_mean_error'] <= float(args.gate_calibration_abs)),
    }
    gate['passed'] = bool(all(gate.values()))
    payload = {
        'config': str(Path(args.config)),
        'condition': str(args.condition),
        'scenario': str(args.scenario),
        'seeds': seeds,
        'slots': int(args.slots),
        'sample_stride': int(args.sample_stride),
        'thresholds': {
            'spearman': float(args.gate_spearman),
            'top1': float(args.gate_top1),
            'calibration_abs': float(args.gate_calibration_abs),
        },
        'summary': summary,
        'gate': gate,
    }
    per_seed.to_csv(out / 'd3_attention_dqi_deployment_per_seed.csv', index=False)
    pd.DataFrame([summary]).to_csv(out / 'd3_attention_dqi_deployment_summary.csv', index=False)
    (out / 'd3_attention_dqi_deployment_gate.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
    _write_markdown(out / 'D3_ATTENTION_DQI_DEPLOYMENT_GATE.md', payload)
    print(f"deployment gate {'PASSED' if gate['passed'] else 'FAILED'}")
    print(f"wrote {out / 'd3_attention_dqi_deployment_gate.json'}")
    return 0 if gate['passed'] or bool(args.no_fail) else 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/e2_d3_r1_attention_dqi_frozen.json')
    parser.add_argument('--condition', default='dqi_attention_frozen')
    parser.add_argument('--scenario', default='M10_N1')
    parser.add_argument('--seeds', default='301,302,303,304,305')
    parser.add_argument('--slots', type=int, default=50_000)
    parser.add_argument('--sample-stride', type=int, default=10)
    parser.add_argument('--split', default='d3_attention_dqi_deployment_gate')
    parser.add_argument('--output-dir', default='results/d3_r1_attention_dqi_frozen/deployment_gate')
    parser.add_argument('--gate-spearman', type=float, default=0.45)
    parser.add_argument('--gate-top1', type=float, default=0.17)
    parser.add_argument('--gate-calibration-abs', type=float, default=0.05)
    parser.add_argument('--no-fail', action='store_true')
    raise SystemExit(run(parser.parse_args()))


if __name__ == '__main__':
    main()
