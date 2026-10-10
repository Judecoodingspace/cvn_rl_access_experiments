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
    compute_unnormalized_dqi,
    fit_feature_scaler,
    fit_reference_rate,
    latency_constrained_goodput,
    normalize_dqi,
    transmission_delay_ms,
)
from cvn_rl.simulation.dqi_policy_value import (
    PolicyEvaluation,
    SwitchingCost,
    evaluate_channel_policy,
    future_window_mean,
    make_channel_placebo,
)
from cvn_rl.simulation.multi_traces import generate_smooth_semi_markov_quality


POLICIES = (
    'current_cqi_greedy',
    'current_cqi_hysteresis',
    'oracle_dqi',
    'channel_placebo_dqi',
)


def _json_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, default=str).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()


def _array_hash(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        digest.update(np.ascontiguousarray(array).view(np.uint8))
    return digest.hexdigest()


def _gate(
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


def _descriptor_array(
    quality: np.ndarray,
    bandwidth: np.ndarray,
    vehicle_quantile: float,
) -> np.ndarray:
    efficiency = quality / np.maximum(bandwidth[None, None, :], 1e-12)
    snr_db = 10.0 * np.log10(np.maximum(np.exp2(efficiency) - 1.0, 1e-12))
    shared_snr = np.quantile(
        snr_db, vehicle_quantile, axis=1, method='linear'
    )
    cci = np.ones_like(shared_snr)
    cci[1:] = np.exp(-np.abs(np.diff(shared_snr, axis=0)) / 10.0)
    shared_rss = shared_snr - 95.0
    bandwidth_time = np.repeat(bandwidth[None, :], quality.shape[0], axis=0)
    return np.stack((shared_snr, shared_rss, bandwidth_time, cci), axis=-1)


def _generate_payload(
    config: dict[str, Any],
    seed: int,
    *,
    need_descriptors: bool,
    audit_key: str = 'd0_5_audit',
) -> dict[str, Any]:
    audit = config[audit_key]
    slots = int(config['slots'])
    horizon = int(audit['horizon_slots'])
    vehicles = int(config['num_vehicles'])
    channels = int(config['num_channels'])
    vehicle_q = float(config['dqi']['vehicle_quantile'])
    time_q = float(config['dqi']['time_quantile'])
    rng = np.random.default_rng(seed)
    bandwidth, combined_quality = generate_smooth_semi_markov_quality(
        slots + horizon,
        vehicles,
        channels,
        config,
        rng,
    )
    z_mbps, source_slots = compute_unnormalized_dqi(
        combined_quality,
        horizon_samples=horizon,
        vehicle_quantile=vehicle_q,
        time_quantile=time_q,
        quantile_method=str(config['dqi']['quantile_method']),
        packet_size_bits=float(config['dqi']['packet_size_bits']),
        delay_max_ms=float(config['dqi']['delay_max_ms']),
        delay_beta_ms=float(config['dqi']['delay_beta_ms']),
    )
    combined_delay = transmission_delay_ms(
        combined_quality, float(config['dqi']['packet_size_bits'])
    )
    combined_goodput = latency_constrained_goodput(
        combined_quality,
        delay_ms=combined_delay,
        packet_size_bits=float(config['dqi']['packet_size_bits']),
        delay_max_ms=float(config['dqi']['delay_max_ms']),
        delay_beta_ms=float(config['dqi']['delay_beta_ms']),
    )
    quality = combined_quality[:slots]
    delay = combined_delay[:slots]
    goodput = combined_goodput[:slots]
    combined_shared_goodput = np.quantile(
        combined_goodput, vehicle_q, axis=1, method='linear'
    )
    shared_goodput = np.quantile(
        goodput, vehicle_q, axis=1, method='linear'
    )
    shared_raw = np.quantile(
        quality, vehicle_q, axis=1, method='linear'
    )
    delay_violation = delay > float(config['dqi']['delay_max_ms'])
    descriptors = None
    if need_descriptors:
        descriptors = _descriptor_array(quality, bandwidth, vehicle_q)
    return {
        'bandwidth_mhz': bandwidth,
        'quality_mbps': quality,
        'z_mbps': z_mbps,
        'source_slots': source_slots,
        'shared_goodput_mbps': shared_goodput,
        'future_mean_mbps': future_window_mean(
            combined_shared_goodput, horizon
        ),
        'shared_raw_rate_mbps': shared_raw,
        'delay_violation': delay_violation,
        'descriptors': descriptors,
        'trace_hash': _array_hash(combined_quality, bandwidth),
    }


def _readiness_row(
    split: str,
    seed: int,
    payload: dict[str, Any],
    y: np.ndarray,
    start: int,
) -> dict[str, Any]:
    current = np.empty_like(payload['shared_raw_rate_mbps'])
    current[0] = payload['shared_raw_rate_mbps'][0]
    current[1:] = payload['shared_raw_rate_mbps'][:-1]
    oracle_top = np.argmax(y[start:], axis=1)
    current_top = np.argmax(current[start:], axis=1)
    counts = np.bincount(oracle_top, minlength=y.shape[1])
    return {
        'split': split,
        'seed': int(seed),
        'trace_id': f'{split}-s{seed}',
        'trace_hash': payload['trace_hash'],
        'delay_violation_fraction': float(
            np.mean(payload['delay_violation'][start:])
        ),
        'distinct_top1_channels': int(np.count_nonzero(counts)),
        'maximum_top1_channel_share': float(np.max(counts) / len(oracle_top)),
        'oracle_current_top1_match': float(np.mean(oracle_top == current_top)),
        'label_std': float(np.std(y[start:])),
        'label_one_fraction': float(np.mean(y[start:] == 1.0)),
        'top1_turnover_rate': float(np.mean(oracle_top[1:] != oracle_top[:-1])),
        'bandwidth_range_mhz': float(np.ptp(payload['bandwidth_mhz'])),
    }


def _policy_metric_row(
    split: str,
    seed: int,
    trace_hash: str,
    condition: str,
    cost: SwitchingCost,
    policy: str,
    evaluation: PolicyEvaluation,
    disagreement: float,
) -> dict[str, Any]:
    duration_seconds = len(evaluation.selected_channel) / 1000.0
    return {
        'split': split,
        'seed': int(seed),
        'trace_hash': trace_hash,
        'switching_condition': condition,
        'blackout_slots': cost.blackout_slots,
        'min_dwell_slots': cost.min_dwell_slots,
        'policy': policy,
        'net_latency_constrained_goodput_mbps': float(
            np.mean(evaluation.net_shared_goodput_mbps)
        ),
        'gross_latency_constrained_goodput_mbps': float(
            np.mean(evaluation.gross_shared_goodput_mbps)
        ),
        'net_raw_throughput_mbps': float(np.mean(evaluation.net_raw_rate_mbps)),
        'gross_raw_throughput_mbps': float(
            np.mean(evaluation.gross_raw_rate_mbps)
        ),
        'delay_violation_rate': float(np.mean(evaluation.delay_violation)),
        'switch_count': int(np.sum(evaluation.switched)),
        'switch_count_per_second': float(
            np.sum(evaluation.switched) / duration_seconds
        ),
        'blackout_fraction': float(np.mean(evaluation.blackout)),
        'blackout_loss_mbps': float(
            np.mean(
                evaluation.gross_shared_goodput_mbps
                - evaluation.net_shared_goodput_mbps
            )
        ),
        'instantaneous_channel_regret_mbps': float(
            np.mean(evaluation.instantaneous_regret_mbps)
        ),
        'dqi_channel_regret_mbps': float(np.mean(evaluation.dqi_regret_mbps)),
        'oracle_current_action_disagreement': float(disagreement),
    }


def _evaluate_policies(
    payload: dict[str, Any],
    y: np.ndarray,
    r_ref: float,
    config: dict[str, Any],
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, PolicyEvaluation]]]:
    audit = config['d0_5_audit']
    start = int(audit['history_slots']) - 1
    horizon = int(audit['horizon_slots'])
    shared_raw = payload['shared_raw_rate_mbps']
    shared_goodput = payload['shared_goodput_mbps']
    current_raw = np.empty_like(shared_raw)
    current_goodput = np.empty_like(shared_goodput)
    current_raw[0], current_goodput[0] = shared_raw[0], shared_goodput[0]
    current_raw[1:], current_goodput[1:] = shared_raw[:-1], shared_goodput[:-1]
    oracle_values = r_ref * y
    placebo, _ = make_channel_placebo(
        y,
        block_slots=int(audit['placebo_block_slots']),
        rng=np.random.default_rng(900_000 + seed),
    )
    strategy = {
        'current_cqi_greedy': (current_raw, horizon * current_goodput, False),
        'current_cqi_hysteresis': (current_raw, horizon * current_goodput, True),
        'oracle_dqi': (y, horizon * oracle_values, True),
        'channel_placebo_dqi': (
            placebo,
            horizon * r_ref * placebo,
            True,
        ),
    }
    rows: list[dict[str, Any]] = []
    all_evaluations: dict[str, dict[str, PolicyEvaluation]] = {}
    for condition, values in audit['switching_conditions'].items():
        cost = SwitchingCost(
            blackout_slots=int(values['blackout_slots']),
            min_dwell_slots=int(values['min_dwell_slots']),
        )
        evaluations: dict[str, PolicyEvaluation] = {}
        for policy, (ranking, projected, hysteresis) in strategy.items():
            evaluations[policy] = evaluate_channel_policy(
                shared_goodput,
                shared_raw,
                payload['delay_violation'],
                ranking,
                projected,
                oracle_values,
                cost,
                start_slot=start,
                use_hysteresis=hysteresis,
            )
        disagreement = float(
            np.mean(
                evaluations['oracle_dqi'].selected_channel
                != evaluations['current_cqi_hysteresis'].selected_channel
            )
        )
        for policy, evaluation in evaluations.items():
            rows.append(
                _policy_metric_row(
                    'test', seed, payload['trace_hash'], condition, cost,
                    policy, evaluation, disagreement,
                )
            )
        all_evaluations[condition] = evaluations
    return rows, all_evaluations


def _bootstrap_ci(values: np.ndarray, samples: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(samples, len(values)))
    means = np.mean(values[indices], axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _pairwise_effects(
    metrics: pd.DataFrame,
    bootstrap_samples: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    metric = 'net_latency_constrained_goodput_mbps'
    for condition in sorted(metrics['switching_condition'].unique()):
        subset = metrics[metrics['switching_condition'] == condition]
        pivot = subset.pivot(index='seed', columns='policy', values=metric)
        regret = subset.pivot(
            index='seed', columns='policy', values='dqi_channel_regret_mbps'
        )
        for comparator in ('current_cqi_hysteresis', 'channel_placebo_dqi'):
            difference = pivot['oracle_dqi'] - pivot[comparator]
            relative = difference / np.maximum(pivot[comparator], 1e-12)
            regret_difference = regret['oracle_dqi'] - regret[comparator]
            for seed in difference.index:
                rows.append(
                    {
                        'switching_condition': condition,
                        'comparator': comparator,
                        'seed': int(seed),
                        'oracle_minus_comparator_mbps': float(difference.loc[seed]),
                        'relative_gain': float(relative.loc[seed]),
                        'oracle_minus_comparator_dqi_regret_mbps': float(
                            regret_difference.loc[seed]
                        ),
                        'ci95_low_relative_gain': np.nan,
                        'ci95_high_relative_gain': np.nan,
                        'seed_wins': int(difference.loc[seed] > 0.0),
                    }
                )
            low, high = _bootstrap_ci(
                relative.to_numpy(), bootstrap_samples,
                seed=70_000 + sum(ord(c) for c in condition + comparator),
            )
            rows.append(
                {
                    'switching_condition': condition,
                    'comparator': comparator,
                    'seed': 'ALL',
                    'oracle_minus_comparator_mbps': float(np.mean(difference)),
                    'relative_gain': float(np.mean(relative)),
                    'oracle_minus_comparator_dqi_regret_mbps': float(
                        np.mean(regret_difference)
                    ),
                    'ci95_low_relative_gain': low,
                    'ci95_high_relative_gain': high,
                    'seed_wins': int(np.sum(difference > 0.0)),
                }
            )
    return pd.DataFrame(rows)


def _make_figures(
    metrics: pd.DataFrame,
    timeline: dict[str, Any],
    output_dir: Path,
) -> None:
    summary = metrics.groupby(
        ['switching_condition', 'policy'], as_index=False
    )['net_latency_constrained_goodput_mbps'].agg(['mean', 'std']).reset_index()
    figure, axis = plt.subplots(figsize=(8.4, 4.2))
    for policy in POLICIES:
        values = summary[summary['policy'] == policy]
        axis.errorbar(
            values['switching_condition'], values['mean'], yerr=values['std'],
            marker='o', capsize=3, label=policy,
        )
    axis.set_ylabel('Net latency-constrained goodput (Mbps)')
    axis.set_xlabel('Switching condition')
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output_dir / 'd0_5_switching_tradeoff.pdf')
    plt.close(figure)

    length = len(timeline['oracle_channel'])
    x = np.arange(length)
    figure, axes = plt.subplots(2, 1, figsize=(9, 5.2), sharex=True)
    axes[0].step(x, timeline['oracle_channel'], where='post', label='Oracle DQI')
    axes[0].step(
        x, timeline['current_channel'], where='post', alpha=0.8,
        label='Current-CQI hysteresis',
    )
    axes[0].set_ylabel('Selected channel')
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.2)
    axes[1].plot(x, timeline['oracle_net_goodput'], label='Oracle DQI')
    axes[1].plot(
        x, timeline['current_net_goodput'], alpha=0.8,
        label='Current-CQI hysteresis',
    )
    axes[1].fill_between(
        x, 0, 1, where=timeline['oracle_blackout'], alpha=0.15,
        transform=axes[1].get_xaxis_transform(), label='Oracle blackout',
    )
    axes[1].set_xlabel('Evaluation slot (1 ms)')
    axes[1].set_ylabel('Net LC-goodput (Mbps)')
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(output_dir / 'd0_5_example_timeline.pdf')
    plt.close(figure)


def run_audit(config: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    audit = dict(config['d0_5_audit'])
    dqi = dict(config['dqi'])
    history = int(audit['history_slots'])
    start = history - 1
    split_seeds = {
        split: [int(seed) for seed in seeds]
        for split, seeds in audit['split_seeds'].items()
    }
    train_z: list[np.ndarray] = []
    train_descriptors: list[np.ndarray] = []
    compact: dict[tuple[str, int], dict[str, Any]] = {}
    trace_ids: dict[str, list[str]] = {
        split: [f'{split}-s{seed}' for seed in seeds]
        for split, seeds in split_seeds.items()
    }

    for seed in split_seeds['train']:
        payload = _generate_payload(config, seed, need_descriptors=True)
        train_z.append(payload['z_mbps'][start:])
        train_descriptors.append(payload['descriptors'][start:])
        compact[('train', seed)] = {
            'z_mbps': payload['z_mbps'],
            'source_slots': payload['source_slots'],
            'bandwidth_mhz': payload['bandwidth_mhz'],
            'shared_raw_rate_mbps': payload['shared_raw_rate_mbps'],
            'delay_violation': payload['delay_violation'],
            'trace_hash': payload['trace_hash'],
        }

    r_ref = fit_reference_rate(
        train_z, quantile=0.95, quantile_method=str(dqi['quantile_method'])
    )
    scaler = fit_feature_scaler(train_descriptors)
    scaler_hash = _array_hash(scaler.mean, scaler.scale)
    readiness_rows: list[dict[str, Any]] = []
    for seed in split_seeds['train']:
        payload = compact[('train', seed)]
        y = normalize_dqi(payload['z_mbps'], r_ref)
        readiness_rows.append(_readiness_row('train', seed, payload, y, start))

    policy_rows: list[dict[str, Any]] = []
    test_evaluations: dict[int, dict[str, dict[str, PolicyEvaluation]]] = {}
    for split in ('validation', 'test'):
        for seed in split_seeds[split]:
            payload = _generate_payload(config, seed, need_descriptors=False)
            y = normalize_dqi(payload['z_mbps'], r_ref)
            readiness_rows.append(_readiness_row(split, seed, payload, y, start))
            if split == 'test':
                rows, evaluations = _evaluate_policies(
                    payload, y, r_ref, config, seed
                )
                policy_rows.extend(rows)
                test_evaluations[seed] = evaluations

    readiness = pd.DataFrame(readiness_rows)
    metrics = pd.DataFrame(policy_rows)
    effects = _pairwise_effects(metrics, int(audit['bootstrap_samples']))
    readiness.to_csv(output_dir / 'd0_5_trace_readiness.csv', index=False)
    metrics.to_csv(output_dir / 'd0_5_policy_metrics.csv', index=False)
    effects.to_csv(output_dir / 'd0_5_policy_pairwise_effects.csv', index=False)

    gates: list[dict[str, Any]] = []
    formula_rate = np.full((2, 2), 40.0)
    formula_delay = np.full((2, 2), float(dqi['delay_max_ms']))
    half = latency_constrained_goodput(formula_rate, delay_ms=formula_delay)
    _gate(
        gates, 'd0_formula_contract', np.allclose(half, 0.5 * formula_rate),
        float(np.max(np.abs(half - 0.5 * formula_rate))), '<= 1e-12',
    )
    source_contract = all(
        np.array_equal(
            compact[('train', seed)]['source_slots'],
            np.arange(int(config['slots'])),
        )
        for seed in split_seeds['train']
    )
    _gate(
        gates, 'd0_label_shape_and_timing', source_contract,
        {'shape': [int(config['slots']), int(config['num_channels'])],
         'target_offsets': [1, int(audit['horizon_slots'])]},
        'shared (T,M), target t+1:t+100',
    )
    train_y = np.concatenate([normalize_dqi(z, r_ref).reshape(-1) for z in train_z])
    _gate(
        gates, 'd0_train_only_reference', np.isfinite(r_ref) and r_ref > 0.0,
        {'r_ref_mbps': r_ref, 'source': 'train_z_q95'}, 'finite train-only q95',
    )
    scaler_ok = bool(
        np.all(np.isfinite(scaler.mean)) and np.all(np.isfinite(scaler.scale))
    )
    _gate(
        gates, 'd0_train_only_scaler', scaler_ok,
        {'hash': scaler_hash, 'sample_count': scaler.sample_count},
        'finite parameters fitted on train descriptors',
    )
    _gate(
        gates, 'd0_occupancy_exclusion', True,
        'canonical label API accepts only link rate/delay', 'no occupancy input',
    )
    readiness_cfg = audit['readiness']
    delay_ok = readiness['delay_violation_fraction'].between(
        float(readiness_cfg['minimum_delay_violation_fraction']),
        float(readiness_cfg['maximum_delay_violation_fraction']),
        inclusive='both',
    )
    _gate(
        gates, 'readiness_delay_violation', bool(delay_ok.all()),
        readiness[['trace_id', 'delay_violation_fraction']].to_dict('records'),
        'every trace in [0.05, 0.20]',
    )
    distinct_ok = readiness['distinct_top1_channels'] >= int(
        readiness_cfg['minimum_distinct_top1_channels']
    )
    _gate(
        gates, 'readiness_top1_diversity', bool(distinct_ok.all()),
        readiness[['trace_id', 'distinct_top1_channels']].to_dict('records'),
        'every trace >= 3 channels',
    )
    share_ok = readiness['maximum_top1_channel_share'] <= float(
        readiness_cfg['maximum_top1_channel_share']
    )
    _gate(
        gates, 'readiness_no_dominant_channel', bool(share_ok.all()),
        readiness[['trace_id', 'maximum_top1_channel_share']].to_dict('records'),
        'every trace <= 0.80',
    )
    match_ok = readiness['oracle_current_top1_match'] < float(
        readiness_cfg['maximum_oracle_current_top1_match']
    )
    _gate(
        gates, 'readiness_oracle_not_current_duplicate', bool(match_ok.all()),
        readiness[['trace_id', 'oracle_current_top1_match']].to_dict('records'),
        'every trace < 0.90',
    )
    label_std_ok = readiness['label_std'] >= float(
        readiness_cfg['minimum_label_std']
    )
    _gate(
        gates, 'readiness_label_variation', bool(label_std_ok.all()),
        float(readiness['label_std'].min()), 'every trace std >= 0.02',
    )
    train_saturation = float(np.mean(train_y == 1.0))
    _gate(
        gates, 'readiness_train_saturation',
        train_saturation <= float(readiness_cfg['maximum_train_saturation']),
        train_saturation, '<= 0.15',
    )
    bandwidth_ok = bool(np.all(readiness['bandwidth_range_mhz'] == 0.0))
    _gate(
        gates, 'readiness_equal_bandwidth', bandwidth_ok,
        float(readiness['bandwidth_range_mhz'].max()),
        'range = 0; DQI-bandwidth correlation undefined',
    )

    primary = str(audit['primary_condition'])
    minimum_gain = float(audit['acceptance']['minimum_primary_relative_gain'])
    primary_effects = effects[
        (effects['switching_condition'] == primary) & (effects['seed'] == 'ALL')
    ].set_index('comparator')
    for comparator, gate_name in (
        ('current_cqi_hysteresis', 'primary_oracle_over_current_cqi'),
        ('channel_placebo_dqi', 'primary_oracle_over_placebo'),
    ):
        row = primary_effects.loc[comparator]
        passed = bool(row['seed_wins'] == len(split_seeds['test'])) and bool(
            row['relative_gain'] >= minimum_gain
        )
        _gate(
            gates, gate_name, passed,
            {'seed_wins': int(row['seed_wins']),
             'mean_relative_gain': float(row['relative_gain'])},
            f'{len(split_seeds["test"])}/{len(split_seeds["test"])} wins and gain >= {minimum_gain}',
        )
    primary_seed = effects[
        (effects['switching_condition'] == primary)
        & (effects['comparator'] == 'current_cqi_hysteresis')
        & (effects['seed'] != 'ALL')
    ]
    regret_ok = bool(
        (primary_seed['oracle_minus_comparator_dqi_regret_mbps'] < 0.0).all()
    )
    _gate(
        gates, 'primary_oracle_lower_regret', regret_ok,
        primary_seed[['seed', 'oracle_minus_comparator_dqi_regret_mbps']].to_dict('records'),
        'oracle regret lower on every seed',
    )
    required_wins = int(audit['acceptance']['minimum_sensitivity_seed_wins'])
    for condition in ('C0', 'C2'):
        sensitivity = effects[
            (effects['switching_condition'] == condition)
            & (effects['comparator'] == 'current_cqi_hysteresis')
            & (effects['seed'] != 'ALL')
        ]
        wins = int(np.sum(sensitivity['oracle_minus_comparator_mbps'] >= 0.0))
        _gate(
            gates, f'sensitivity_{condition}_nonnegative', wins >= required_wins,
            {'nonnegative_seeds': wins}, f'>= {required_wins}/3 seeds',
        )

    first_seed = split_seeds['test'][0]
    primary_eval = test_evaluations[first_seed][primary]
    timeline_length = min(1000, len(primary_eval['oracle_dqi'].selected_channel))
    timeline = {
        'oracle_channel': primary_eval['oracle_dqi'].selected_channel[:timeline_length],
        'current_channel': primary_eval['current_cqi_hysteresis'].selected_channel[:timeline_length],
        'oracle_net_goodput': primary_eval['oracle_dqi'].net_shared_goodput_mbps[:timeline_length],
        'current_net_goodput': primary_eval['current_cqi_hysteresis'].net_shared_goodput_mbps[:timeline_length],
        'oracle_blackout': primary_eval['oracle_dqi'].blackout[:timeline_length],
    }
    _make_figures(metrics, timeline, output_dir)

    config_hash = _json_hash(config)
    manifest = {
        'experiment': 'd0_5_dqi_policy_value_gate',
        'config_hash': config_hash,
        'label_version': DQI_LABEL_VERSION,
        'trace_ids': trace_ids,
        'trace_hashes': {
            row['trace_id']: row['trace_hash'] for row in readiness_rows
        },
        'split_seeds': split_seeds,
        'slot_duration_ms': float(config['slot_duration_ms']),
        'history_slots': history,
        'horizon_slots': int(audit['horizon_slots']),
        'r_ref_mbps': r_ref,
        'reference_source': 'train_z_q95',
        'scaler_hash': scaler_hash,
        'quality_family': config['channel']['quality_family'],
        'quality_parameters': config['channel'],
        'policies': list(POLICIES),
        'switching_conditions': audit['switching_conditions'],
        'primary_condition': primary,
        'initial_selection_free': True,
        'minimum_dwell_begins_at_initial_selection_or_switch': True,
        'blackout_is_only_switching_penalty': True,
    }
    (output_dir / 'd0_5_manifest.json').write_text(
        json.dumps(manifest, indent=2), encoding='utf-8'
    )
    passed = bool(all(gate['passed'] for gate in gates))
    report = {
        'passed': passed,
        'stop_before_d1': not passed,
        'config_hash': config_hash,
        'r_ref_mbps': r_ref,
        'scaler_hash': scaler_hash,
        'gates': gates,
        'interpretation': (
            'D0.5 passed: proceed to Oracle RL.'
            if passed
            else 'D0.5 failed: inspect readiness and policy-value gates before D1.'
        ),
    }
    (output_dir / 'd0_5_gate_report.json').write_text(
        json.dumps(report, indent=2), encoding='utf-8'
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description='Run the D0.5 DQI policy-value gate.')
    parser.add_argument(
        '--config', default='configs/e2_d0_5_dqi_policy_value_gate.json'
    )
    parser.add_argument(
        '--output-dir', default='results/d0_5_dqi_policy_value'
    )
    args = parser.parse_args()
    report = run_audit(load_config(args.config), Path(args.output_dir))
    print(
        json.dumps(
            {
                'passed': report['passed'],
                'stop_before_d1': report['stop_before_d1'],
                'output_dir': str(Path(args.output_dir).resolve()),
            },
            indent=2,
        )
    )


if __name__ == '__main__':
    main()
