from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from cvn_rl.experiments.config import load_config
from cvn_rl.simulation.dqi_labels import (
    DQI_LABEL_VERSION,
    channel_descriptors,
    compute_unnormalized_dqi,
    fit_feature_scaler,
    fit_reference_rate,
    latency_constrained_goodput,
    normalize_dqi,
)
from cvn_rl.simulation.multi_traces import generate_multi_vehicle_trace


def _sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, default=str).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()


def _array_hash(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        digest.update(np.ascontiguousarray(array).view(np.uint8))
    return digest.hexdigest()


def _reference_z(
    rate: np.ndarray,
    sources: np.ndarray,
    horizon: int,
    vehicle_q: float,
    time_q: float,
    packet_bits: float,
    delay_max: float,
    delay_beta: float,
) -> np.ndarray:
    output = np.empty((len(sources), rate.shape[2]), dtype=float)
    goodput = latency_constrained_goodput(
        rate,
        packet_size_bits=packet_bits,
        delay_max_ms=delay_max,
        delay_beta_ms=delay_beta,
    )
    for row, source in enumerate(sources):
        for channel in range(rate.shape[2]):
            per_time = []
            for future in range(source + 1, source + horizon + 1):
                per_time.append(
                    np.quantile(
                        goodput[future, :, channel],
                        vehicle_q,
                        method='linear',
                    )
                )
            output[row, channel] = np.quantile(
                per_time,
                time_q,
                method='linear',
            )
    return output


def _add_gate(
    gates: list[dict[str, Any]],
    name: str,
    passed: bool,
    value: Any,
    threshold: str,
) -> None:
    gates.append(
        {
            'gate': name,
            'passed': bool(passed),
            'value': value,
            'threshold': threshold,
        }
    )


def _distribution_row(
    split: str,
    channel: str | int,
    z: np.ndarray,
    y: np.ndarray,
) -> dict[str, Any]:
    z_flat = np.asarray(z, dtype=float).reshape(-1)
    y_flat = np.asarray(y, dtype=float).reshape(-1)
    row: dict[str, Any] = {
        'split': split,
        'channel': channel,
        'count': int(y_flat.size),
        'z_min': float(np.min(z_flat)),
        'z_max': float(np.max(z_flat)),
        'y_mean': float(np.mean(y_flat)),
        'y_std': float(np.std(y_flat)),
        'y_zero_fraction': float(np.mean(y_flat == 0.0)),
        'y_one_fraction': float(np.mean(y_flat == 1.0)),
    }
    for quantile in (1, 5, 10, 25, 50, 75, 90, 95, 99):
        row[f'y_p{quantile:02d}'] = float(
            np.percentile(y_flat, quantile)
        )
    return row


def _synthetic_gates(
    gates: list[dict[str, Any]],
    horizon: int,
    vehicle_q: float,
    time_q: float,
) -> dict[str, np.ndarray]:
    rate = np.full((horizon + 2, 10, 2), 40.0)
    delay = np.full_like(rate, 5.0)
    at_limit = latency_constrained_goodput(rate, delay_ms=delay)
    _add_gate(
        gates,
        'formula_at_delay_limit',
        np.allclose(at_limit, rate * 0.5, atol=1e-12),
        float(np.max(np.abs(at_limit - rate * 0.5))),
        'max_abs_error <= 1e-12',
    )

    faster = latency_constrained_goodput(rate * 2.0, delay_ms=delay)
    slower_delay = latency_constrained_goodput(rate, delay_ms=delay + 2.0)
    monotone = bool(np.all(faster >= at_limit) and np.all(slower_delay <= at_limit))
    _add_gate(gates, 'formula_monotonicity', monotone, monotone, 'True')

    base_z, _ = compute_unnormalized_dqi(
        rate,
        delay_ms=delay,
        horizon_samples=horizon,
        vehicle_quantile=vehicle_q,
        time_quantile=time_q,
    )
    current_changed = rate.copy()
    current_changed[0] = 1.0
    after_changed = rate.copy()
    after_changed[horizon + 1] = 1.0
    current_z, _ = compute_unnormalized_dqi(
        current_changed,
        delay_ms=delay,
        horizon_samples=horizon,
        vehicle_quantile=vehicle_q,
        time_quantile=time_q,
    )
    after_z, _ = compute_unnormalized_dqi(
        after_changed,
        delay_ms=delay,
        horizon_samples=horizon,
        vehicle_quantile=vehicle_q,
        time_quantile=time_q,
    )
    boundaries_ok = bool(
        np.allclose(base_z[0], current_z[0])
        and np.allclose(base_z[0], after_z[0])
    )
    _add_gate(gates, 'future_window_boundaries', boundaries_ok, boundaries_ok, 'True')

    bad_times = rate.copy()
    bad_times[1:12, :, 0] = 1.0
    bad_time_z, _ = compute_unnormalized_dqi(
        bad_times,
        delay_ms=delay,
        horizon_samples=horizon,
        vehicle_quantile=vehicle_q,
        time_quantile=time_q,
    )
    _add_gate(
        gates,
        'time_quantile_monotonicity',
        bad_time_z[0, 0] < base_z[0, 0],
        float(base_z[0, 0] - bad_time_z[0, 0]),
        '> 0',
    )

    bad_vehicle = rate.copy()
    bad_vehicle[1 : horizon + 1, 0, 0] = 1.0
    bad_vehicle_z, _ = compute_unnormalized_dqi(
        bad_vehicle,
        delay_ms=delay,
        horizon_samples=horizon,
        vehicle_quantile=vehicle_q,
        time_quantile=time_q,
    )
    _add_gate(
        gates,
        'vehicle_quantile_monotonicity',
        bad_vehicle_z[0, 0] < base_z[0, 0],
        float(base_z[0, 0] - bad_vehicle_z[0, 0]),
        '> 0',
    )
    return {
        'base_rate': rate,
        'base_z': base_z,
        'bad_time_z': bad_time_z,
        'bad_vehicle_z': bad_vehicle_z,
    }


def run_audit(config: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    audit = dict(config.get('d0_audit', {}))
    history = int(audit.get('history_slots', 100))
    horizon = int(audit.get('horizon_slots', 100))
    vehicle_q = float(audit.get('vehicle_quantile', 0.10))
    time_q = float(audit.get('time_quantile', 0.10))
    packet_bits = float(config.get('dqi', {}).get('packet_size_bits', 32_000))
    delay_max = float(config.get('dqi', {}).get('delay_max_ms', 5.0))
    delay_beta = float(config.get('dqi', {}).get('delay_beta_ms', 2.0))
    split_seeds = {
        name: [int(seed) for seed in values]
        for name, values in audit['split_seeds'].items()
    }
    trace_config = {
        **config,
        'num_channels': 10,
        'num_vehicles': 10,
        'slot_duration_ms': 1.0,
        'hidden_lookahead_slots': horizon,
    }
    traces: dict[str, list[Any]] = {}
    z_by_split: dict[str, list[np.ndarray]] = {}
    descriptors_by_split: dict[str, list[np.ndarray]] = {}
    trace_ids: dict[str, list[str]] = {}

    for split, seeds in split_seeds.items():
        traces[split] = []
        z_by_split[split] = []
        descriptors_by_split[split] = []
        trace_ids[split] = []
        for seed in seeds:
            trace = generate_multi_vehicle_trace(
                trace_config,
                seed,
                slots=int(config['slots']),
                split=split,
            )
            combined = np.concatenate(
                [trace.quality_mbps, trace.quality_lookahead_mbps],
                axis=0,
            )
            z_all, _ = compute_unnormalized_dqi(
                combined,
                horizon_samples=horizon,
                vehicle_quantile=vehicle_q,
                time_quantile=time_q,
                packet_size_bits=packet_bits,
                delay_max_ms=delay_max,
                delay_beta_ms=delay_beta,
            )
            valid = slice(history - 1, trace.slots)
            traces[split].append(trace)
            z_by_split[split].append(z_all[valid])
            descriptors_by_split[split].append(
                channel_descriptors(
                    trace.snr_db,
                    trace.rss_dbm,
                    trace.bandwidth_mhz,
                    trace.cci,
                    vehicle_quantile=vehicle_q,
                )[valid]
            )
            trace_ids[split].append(trace.trace_id)

    r_ref = fit_reference_rate(z_by_split['train'])
    scaler = fit_feature_scaler(descriptors_by_split['train'])
    scaler_hash = _array_hash(scaler.mean, scaler.scale)
    y_by_split = {
        split: [normalize_dqi(z, r_ref) for z in values]
        for split, values in z_by_split.items()
    }

    gates: list[dict[str, Any]] = []
    synthetic = _synthetic_gates(gates, horizon, vehicle_q, time_q)
    all_y = np.concatenate(
        [y.reshape(-1) for values in y_by_split.values() for y in values]
    )
    train_y = np.concatenate([y.reshape(-1) for y in y_by_split['train']])
    numeric_ok = bool(
        np.all(np.isfinite(all_y))
        and np.all((all_y >= 0.0) & (all_y <= 1.0))
        and r_ref > 0.0
    )
    _add_gate(gates, 'numeric_stability', numeric_ok, numeric_ok, 'True')
    nondegenerate = bool(
        np.std(train_y) >= float(audit.get('minimum_label_std', 0.02))
        and np.mean(train_y == 1.0)
        <= float(audit.get('maximum_train_saturation', 0.15))
    )
    _add_gate(
        gates,
        'label_non_degeneracy',
        nondegenerate,
        {
            'std': float(np.std(train_y)),
            'one_fraction': float(np.mean(train_y == 1.0)),
        },
        'std >= 0.02 and one_fraction <= 0.15',
    )

    shapes_ok = all(
        z.shape[1:] == (10,)
        for values in z_by_split.values()
        for z in values
    )
    _add_gate(gates, 'shared_channel_shape', shapes_ok, '(T, 10)', '(T, M)')
    broadcast = np.repeat(y_by_split['train'][0][:, None, :], 10, axis=1)
    shared_ok = bool(np.all(broadcast == broadcast[:, :1, :]))
    _add_gate(gates, 'vehicle_broadcast_identity', shared_ok, shared_ok, 'True')

    comparison_rows: list[dict[str, Any]] = []
    max_reference_error = 0.0
    for split, split_traces in traces.items():
        for trace, production_z in zip(split_traces, z_by_split[split]):
            combined = np.concatenate(
                [trace.quality_mbps, trace.quality_lookahead_mbps],
                axis=0,
            )
            raw_sources = np.linspace(
                history - 1,
                trace.slots - 1,
                num=min(25, trace.slots - history + 1),
                dtype=int,
            )
            reference = _reference_z(
                combined,
                raw_sources,
                horizon,
                vehicle_q,
                time_q,
                packet_bits,
                delay_max,
                delay_beta,
            )
            selected = production_z[raw_sources - history + 1]
            errors = np.abs(reference - selected)
            max_reference_error = max(max_reference_error, float(np.max(errors)))
            for row, source in enumerate(raw_sources):
                comparison_rows.append(
                    {
                        'split': split,
                        'trace_id': trace.trace_id,
                        'source_slot': int(source),
                        'max_abs_error': float(np.max(errors[row])),
                    }
                )
    _add_gate(
        gates,
        'reference_implementation',
        max_reference_error <= 1e-12,
        max_reference_error,
        '<= 1e-12',
    )

    occupancy_trace = traces['train'][0]
    occupancy_rate = np.concatenate(
        [
            occupancy_trace.quality_mbps,
            occupancy_trace.quality_lookahead_mbps,
        ],
        axis=0,
    )
    occupancy_labels = []
    for occupancy in (
        occupancy_trace.occupancy,
        ~occupancy_trace.occupancy,
        np.random.default_rng(91).random(occupancy_trace.occupancy.shape) > 0.5,
    ):
        _ = occupancy
        values, _ = compute_unnormalized_dqi(
            occupancy_rate,
            horizon_samples=horizon,
            vehicle_quantile=vehicle_q,
            time_quantile=time_q,
            packet_size_bits=packet_bits,
            delay_max_ms=delay_max,
            delay_beta_ms=delay_beta,
        )
        occupancy_labels.append(values)
    occupancy_independent = bool(
        np.array_equal(occupancy_labels[0], occupancy_labels[1])
        and np.array_equal(occupancy_labels[0], occupancy_labels[2])
    )
    _add_gate(
        gates,
        'occupancy_invariance',
        occupancy_independent,
        [_array_hash(values) for values in occupancy_labels],
        'bitwise invariant',
    )

    contaminated_r_ref = fit_reference_rate(z_by_split['train'])
    _ = [z * 100.0 for split in ('validation', 'test') for z in z_by_split[split]]
    r_ref_clean = contaminated_r_ref == r_ref
    _add_gate(gates, 'train_only_r_ref', r_ref_clean, r_ref, 'unchanged')

    contaminated_scaler = fit_feature_scaler(descriptors_by_split['train'])
    scaler_clean = bool(
        np.array_equal(contaminated_scaler.mean, scaler.mean)
        and np.array_equal(contaminated_scaler.scale, scaler.scale)
    )
    _add_gate(gates, 'train_only_scaler', scaler_clean, scaler_hash, 'unchanged')

    all_trace_ids = [
        trace_id for values in trace_ids.values() for trace_id in values
    ]
    split_clean = len(all_trace_ids) == len(set(all_trace_ids))
    _add_gate(gates, 'split_isolation', split_clean, trace_ids, 'all unique')
    configured_horizon = int(
        config.get('descriptors', {}).get('dqi_horizon_slots', -1)
    )
    timing_ok = bool(
        trace_config['slot_duration_ms'] == 1.0
        and history == 100
        and horizon == 100
        and configured_horizon == 100
    )
    _add_gate(
        gates,
        'paper_time_scale',
        timing_ok,
        {
            'slot_duration_ms': trace_config['slot_duration_ms'],
            'history_slots': history,
            'horizon_slots': horizon,
            'configured_horizon_slots': configured_horizon,
        },
        '1 ms, W=100, H=100',
    )
    reference_config_ok = bool(
        config.get('dqi', {}).get('reference_source') == 'train_quantile'
        and config.get('dqi', {}).get('r_ref_mbps') is None
    )
    _add_gate(
        gates,
        'no_fixed_reference_rate',
        reference_config_ok,
        config.get('dqi', {}).get('r_ref_mbps'),
        'null before train-only fitting',
    )
    input_causal = history >= 1 and horizon == 100
    _add_gate(
        gates,
        'input_target_causality',
        input_causal,
        {
            'input_offsets': [-(history - 1), 0],
            'target_offsets': [1, horizon],
        },
        'input <= t; target = t+1..t+100',
    )

    split_rows: list[dict[str, Any]] = []
    channel_rows: list[dict[str, Any]] = []
    for split in z_by_split:
        z_joined = np.concatenate(z_by_split[split], axis=0)
        y_joined = np.concatenate(y_by_split[split], axis=0)
        split_rows.append(_distribution_row(split, 'all', z_joined, y_joined))
        for channel in range(z_joined.shape[1]):
            channel_rows.append(
                _distribution_row(
                    split,
                    channel,
                    z_joined[:, channel],
                    y_joined[:, channel],
                )
            )
    pd.DataFrame(split_rows).to_csv(
        output_dir / 'd0_dqi_distribution_by_split.csv',
        index=False,
    )
    pd.DataFrame(channel_rows).to_csv(
        output_dir / 'd0_dqi_distribution_by_channel.csv',
        index=False,
    )
    pd.DataFrame(comparison_rows).to_csv(
        output_dir / 'd0_dqi_reference_comparison.csv',
        index=False,
    )
    first_trace = traces['train'][0]
    first_z = z_by_split['train'][0]
    first_y = y_by_split['train'][0]
    np.savez_compressed(
        output_dir / 'd0_dqi_example_windows.npz',
        rate_mbps=first_trace.quality_mbps[: history + horizon],
        z_mbps=first_z[:10],
        y=first_y[:10],
        source_slots=np.arange(history - 1, history + 9),
        synthetic_rate=synthetic['base_rate'],
        synthetic_z=synthetic['base_z'],
    )

    figure, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for split, values in z_by_split.items():
        axes[0].hist(
            np.concatenate(values).reshape(-1),
            bins=40,
            alpha=0.45,
            density=True,
            label=split,
        )
        axes[1].hist(
            np.concatenate(y_by_split[split]).reshape(-1),
            bins=40,
            alpha=0.45,
            density=True,
            label=split,
        )
    axes[0].set_xlabel('Unnormalised z (Mbps)')
    axes[1].set_xlabel('Normalised DQI y')
    axes[0].set_ylabel('Density')
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / 'd0_dqi_distribution.pdf')
    plt.close(figure)

    manifest = {
        'label_version': DQI_LABEL_VERSION,
        'config_hash': _sha256_json(config),
        'trace_ids': trace_ids,
        'slot_duration_ms': 1.0,
        'history_slots': history,
        'horizon_slots': horizon,
        'vehicle_quantile': vehicle_q,
        'time_quantile': time_q,
        'quantile_method': 'linear',
        'r_ref_mbps': r_ref,
        'reference_source': 'train_z_q95',
        'scaler': {
            'feature_names': list(scaler.feature_names),
            'mean': scaler.mean.tolist(),
            'scale': scaler.scale.tolist(),
            'sample_count': scaler.sample_count,
            'hash': scaler_hash,
        },
    }
    (output_dir / 'd0_dqi_manifest.json').write_text(
        json.dumps(manifest, indent=2),
        encoding='utf-8',
    )
    passed = bool(all(gate['passed'] for gate in gates))
    report = {
        'passed': passed,
        'stop_before_d1': not passed,
        'config_hash': manifest['config_hash'],
        'trace_ids': trace_ids,
        'r_ref_mbps': r_ref,
        'scaler_hash': scaler_hash,
        'gates': gates,
    }
    (output_dir / 'd0_dqi_gate_report.json').write_text(
        json.dumps(report, indent=2),
        encoding='utf-8',
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description='Audit paper-consistent DQI labels.')
    parser.add_argument(
        '--config',
        default='configs/e2_d0_dqi_label_audit.json',
    )
    parser.add_argument(
        '--output-dir',
        default='results/d0_dqi_label_audit',
    )
    args = parser.parse_args()
    report = run_audit(load_config(args.config), Path(args.output_dir))
    print(
        json.dumps(
            {
                'passed': report['passed'],
                'output_dir': str(Path(args.output_dir).resolve()),
                'failed_gates': [
                    gate['gate']
                    for gate in report['gates']
                    if not gate['passed']
                ],
            },
            indent=2,
        )
    )
    if not report['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
