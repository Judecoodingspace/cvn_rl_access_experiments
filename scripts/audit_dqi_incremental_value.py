from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

try:
    from scripts import audit_dqi_policy_value as d05
except ImportError:  # Direct execution from the scripts directory.
    import audit_dqi_policy_value as d05

from cvn_rl.experiments.config import load_config
from cvn_rl.simulation.dqi_labels import (
    DQI_LABEL_VERSION,
    fit_feature_scaler,
    fit_reference_rate,
    normalize_dqi,
)
from cvn_rl.simulation.dqi_policy_value import (
    PolicyEvaluation,
    SwitchingCost,
    evaluate_channel_policy,
    make_channel_placebo,
    make_time_block_placebo,
    rolling_window_quantile,
)


POLICIES = (
    'current_cqi_hysteresis',
    'current_lcg_hysteresis',
    'oracle_dqi',
    'channel_placebo_dqi',
    'time_placebo_dqi',
    'future_mean_oracle',
)
OUTCOMES = (
    'net_latency_constrained_goodput_mbps',
    'rolling_100ms_q10_mean',
)


def _compact_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        'z_mbps': payload['z_mbps'],
        'source_slots': payload['source_slots'],
        'bandwidth_mhz': payload['bandwidth_mhz'],
        'shared_raw_rate_mbps': payload['shared_raw_rate_mbps'],
        'delay_violation': payload['delay_violation'],
        'trace_hash': payload['trace_hash'],
    }


def _readiness_pass(
    rows: pd.DataFrame,
    train_saturation: float,
    config: dict[str, Any],
    *,
    source_contract: bool,
    scaler_finite: bool,
    r_ref_mbps: float,
) -> tuple[bool, dict[str, bool]]:
    limits = config['d0_6_audit']['readiness']
    checks = {
        'delay_violation': bool(
            rows['delay_violation_fraction'].between(
                float(limits['minimum_delay_violation_fraction']),
                float(limits['maximum_delay_violation_fraction']),
                inclusive='both',
            ).all()
        ),
        'top1_diversity': bool(
            (rows['distinct_top1_channels'] >= int(
                limits['minimum_distinct_top1_channels']
            )).all()
        ),
        'no_dominant_channel': bool(
            (rows['maximum_top1_channel_share'] <= float(
                limits['maximum_top1_channel_share']
            )).all()
        ),
        'oracle_not_current_duplicate': bool(
            (rows['oracle_current_top1_match'] < float(
                limits['maximum_oracle_current_top1_match']
            )).all()
        ),
        'label_variation': bool(
            (rows['label_std'] >= float(limits['minimum_label_std'])).all()
        ),
        'train_saturation': bool(
            train_saturation <= float(limits['maximum_train_saturation'])
        ),
        'equal_bandwidth': bool((rows['bandwidth_range_mhz'] == 0.0).all()),
        'd0_source_contract': bool(source_contract),
        'd0_reference_contract': bool(np.isfinite(r_ref_mbps) and r_ref_mbps > 0.0),
        'd0_scaler_contract': bool(scaler_finite),
    }
    return bool(all(checks.values())), checks


def calibrate_readiness(
    config: dict[str, Any],
) -> tuple[dict[str, Any] | None, pd.DataFrame, dict[str, Any]]:
    """Select physics using calibration seeds only; never evaluate policies."""

    audit = config['d0_6_audit']
    history_start = int(audit['history_slots']) - 1
    train_seeds = [int(seed) for seed in audit['calibration_seeds']['train']]
    validation_seeds = [
        int(seed) for seed in audit['calibration_seeds']['validation']
    ]
    records: list[dict[str, Any]] = []
    selected: dict[str, Any] | None = None
    selected_summary: dict[str, Any] = {}

    for candidate_index, candidate in enumerate(audit['bad_dwell_candidates']):
        candidate_config = copy.deepcopy(config)
        candidate_config['channel']['semi_markov_bad_dwell_slots'] = [
            int(candidate[0]), int(candidate[1])
        ]
        train_z: list[np.ndarray] = []
        train_descriptors: list[np.ndarray] = []
        train_payloads: dict[int, dict[str, Any]] = {}
        for seed in train_seeds:
            payload = d05._generate_payload(
                candidate_config,
                seed,
                need_descriptors=True,
                audit_key='d0_6_audit',
            )
            train_z.append(payload['z_mbps'][history_start:])
            train_descriptors.append(payload['descriptors'][history_start:])
            train_payloads[seed] = _compact_payload(payload)

        r_ref = fit_reference_rate(
            train_z,
            quantile=0.95,
            quantile_method=str(config['dqi']['quantile_method']),
        )
        scaler = fit_feature_scaler(train_descriptors)
        scaler_hash = d05._array_hash(scaler.mean, scaler.scale)
        train_y_flat = np.concatenate(
            [normalize_dqi(z, r_ref).reshape(-1) for z in train_z]
        )
        train_saturation = float(np.mean(train_y_flat == 1.0))
        readiness_rows: list[dict[str, Any]] = []
        for seed, payload in train_payloads.items():
            readiness_rows.append(
                d05._readiness_row(
                    'train', seed, payload,
                    normalize_dqi(payload['z_mbps'], r_ref), history_start,
                )
            )
        for seed in validation_seeds:
            payload = d05._generate_payload(
                candidate_config,
                seed,
                need_descriptors=False,
                audit_key='d0_6_audit',
            )
            readiness_rows.append(
                d05._readiness_row(
                    'validation', seed, payload,
                    normalize_dqi(payload['z_mbps'], r_ref), history_start,
                )
            )
        readiness = pd.DataFrame(readiness_rows)
        source_contract = all(
            np.array_equal(
                payload['source_slots'], np.arange(int(config['slots']))
            )
            for payload in train_payloads.values()
        )
        scaler_finite = bool(
            np.all(np.isfinite(scaler.mean))
            and np.all(np.isfinite(scaler.scale))
        )
        candidate_passed, checks = _readiness_pass(
            readiness,
            train_saturation,
            candidate_config,
            source_contract=source_contract,
            scaler_finite=scaler_finite,
            r_ref_mbps=r_ref,
        )
        for row in readiness_rows:
            records.append(
                {
                    'candidate_index': candidate_index,
                    'bad_dwell_low_slots': int(candidate[0]),
                    'bad_dwell_high_slots': int(candidate[1]),
                    **row,
                    'train_saturation': train_saturation,
                    'r_ref_mbps': r_ref,
                    'scaler_hash': scaler_hash,
                    'candidate_passed': candidate_passed,
                    **{f'check_{key}': value for key, value in checks.items()},
                }
            )
        if candidate_passed:
            selected = candidate_config
            selected_summary = {
                'candidate_index': candidate_index,
                'bad_dwell_slots': [int(candidate[0]), int(candidate[1])],
                'calibration_r_ref_mbps': r_ref,
                'calibration_scaler_hash': scaler_hash,
                'checks': checks,
            }
            break
    return selected, pd.DataFrame(records), selected_summary


def _fit_frozen_training_statistics(
    config: dict[str, Any],
) -> tuple[float, str, dict[str, Any]]:
    audit = config['d0_6_audit']
    start = int(audit['history_slots']) - 1
    train_z: list[np.ndarray] = []
    descriptors: list[np.ndarray] = []
    trace_hashes: dict[str, str] = {}
    for seed in audit['calibration_seeds']['train']:
        payload = d05._generate_payload(
            config, int(seed), need_descriptors=True, audit_key='d0_6_audit'
        )
        train_z.append(payload['z_mbps'][start:])
        descriptors.append(payload['descriptors'][start:])
        trace_hashes[f'train-s{seed}'] = payload['trace_hash']
    r_ref = fit_reference_rate(
        train_z,
        quantile=0.95,
        quantile_method=str(config['dqi']['quantile_method']),
    )
    scaler = fit_feature_scaler(descriptors)
    return r_ref, d05._array_hash(scaler.mean, scaler.scale), {
        'feature_names': list(scaler.feature_names),
        'mean': scaler.mean.tolist(),
        'scale': scaler.scale.tolist(),
        'sample_count': scaler.sample_count,
        'trace_hashes': trace_hashes,
    }


def _lag_one(values: np.ndarray) -> np.ndarray:
    source = np.asarray(values, dtype=float)
    output = np.empty_like(source)
    output[0] = source[0]
    output[1:] = source[:-1]
    return output


def build_strategy_inputs(
    payload: dict[str, Any],
    y: np.ndarray,
    r_ref_mbps: float,
    config: dict[str, Any],
    seed: int,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Build ranking and projected-value arrays for all D0.6 policies."""

    audit = config['d0_6_audit']
    horizon = int(audit['horizon_slots'])
    current_raw = _lag_one(payload['shared_raw_rate_mbps'])
    current_lcg = _lag_one(payload['shared_goodput_mbps'])
    channel_placebo, _ = make_channel_placebo(
        y,
        block_slots=int(audit['placebo_block_slots']),
        rng=np.random.default_rng(1_100_000 + seed),
    )
    time_placebo, _ = make_time_block_placebo(
        y,
        block_slots=int(audit['placebo_block_slots']),
        rng=np.random.default_rng(1_200_000 + seed),
    )
    oracle_values = float(r_ref_mbps) * y
    future_mean = payload['future_mean_mbps']
    return {
        'current_cqi_hysteresis': (
            current_raw, horizon * current_lcg
        ),
        'current_lcg_hysteresis': (
            current_lcg, horizon * current_lcg
        ),
        'oracle_dqi': (y, horizon * oracle_values),
        'channel_placebo_dqi': (
            channel_placebo, horizon * r_ref_mbps * channel_placebo
        ),
        'time_placebo_dqi': (
            time_placebo, horizon * r_ref_mbps * time_placebo
        ),
        'future_mean_oracle': (
            future_mean, horizon * future_mean
        ),
    }


def _metric_row(
    seed: int,
    trace_hash: str,
    condition: str,
    cost: SwitchingCost,
    policy: str,
    evaluation: PolicyEvaluation,
    disagreement: float,
    horizon: int,
) -> dict[str, Any]:
    row = d05._policy_metric_row(
        'test', seed, trace_hash, condition, cost, policy,
        evaluation, disagreement,
    )
    rolling = rolling_window_quantile(
        evaluation.net_shared_goodput_mbps,
        window_samples=horizon,
        quantile=0.10,
        quantile_method='linear',
    )
    row.update(
        {
            'rolling_100ms_q10_mean': float(np.mean(rolling)),
            'rolling_100ms_q10_p10': float(
                np.quantile(rolling, 0.10, method='linear')
            ),
            'zero_service_fraction': float(
                np.mean(evaluation.net_shared_goodput_mbps == 0.0)
            ),
        }
    )
    return row


def _evaluate_test_seed(
    config: dict[str, Any],
    seed: int,
    r_ref_mbps: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    audit = config['d0_6_audit']
    start = int(audit['history_slots']) - 1
    horizon = int(audit['horizon_slots'])
    payload = d05._generate_payload(
        config, seed, need_descriptors=False, audit_key='d0_6_audit'
    )
    y = normalize_dqi(payload['z_mbps'], r_ref_mbps)
    strategy = build_strategy_inputs(payload, y, r_ref_mbps, config, seed)
    rows: list[dict[str, Any]] = []
    evaluations_by_condition: dict[str, dict[str, PolicyEvaluation]] = {}
    for condition, values in audit['switching_conditions'].items():
        cost = SwitchingCost(
            blackout_slots=int(values['blackout_slots']),
            min_dwell_slots=int(values['min_dwell_slots']),
        )
        evaluations: dict[str, PolicyEvaluation] = {}
        for policy, (ranking, projected) in strategy.items():
            evaluations[policy] = evaluate_channel_policy(
                payload['shared_goodput_mbps'],
                payload['shared_raw_rate_mbps'],
                payload['delay_violation'],
                ranking,
                projected,
                r_ref_mbps * y,
                cost,
                start_slot=start,
                use_hysteresis=True,
            )
        disagreement = float(
            np.mean(
                evaluations['oracle_dqi'].selected_channel
                != evaluations['current_cqi_hysteresis'].selected_channel
            )
        )
        for policy, evaluation in evaluations.items():
            rows.append(
                _metric_row(
                    seed, payload['trace_hash'], condition, cost, policy,
                    evaluation, disagreement, horizon,
                )
            )
        evaluations_by_condition[condition] = evaluations
    readiness = d05._readiness_row('test', seed, payload, y, start)
    return rows, readiness


def _pairwise_effects(
    metrics: pd.DataFrame,
    bootstrap_samples: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for condition in sorted(metrics['switching_condition'].unique()):
        subset = metrics[metrics['switching_condition'] == condition]
        for outcome in OUTCOMES:
            pivot = subset.pivot(index='seed', columns='policy', values=outcome)
            for comparator in [policy for policy in POLICIES if policy != 'oracle_dqi']:
                difference = pivot['oracle_dqi'] - pivot[comparator]
                relative = difference / np.maximum(
                    np.abs(pivot[comparator]), 1e-12
                )
                for seed in difference.index:
                    rows.append(
                        {
                            'switching_condition': condition,
                            'outcome': outcome,
                            'comparator': comparator,
                            'seed': int(seed),
                            'oracle_minus_comparator': float(difference.loc[seed]),
                            'relative_gain': float(relative.loc[seed]),
                            'ci95_low': np.nan,
                            'ci95_high': np.nan,
                            'ci95_low_relative': np.nan,
                            'ci95_high_relative': np.nan,
                            'seed_wins': int(difference.loc[seed] > 0.0),
                        }
                    )
                seed_offset = sum(ord(c) for c in condition + outcome + comparator)
                low, high = d05._bootstrap_ci(
                    difference.to_numpy(), bootstrap_samples, 1_300_000 + seed_offset
                )
                low_rel, high_rel = d05._bootstrap_ci(
                    relative.to_numpy(), bootstrap_samples, 1_400_000 + seed_offset
                )
                rows.append(
                    {
                        'switching_condition': condition,
                        'outcome': outcome,
                        'comparator': comparator,
                        'seed': 'ALL',
                        'oracle_minus_comparator': float(np.mean(difference)),
                        'relative_gain': float(np.mean(relative)),
                        'ci95_low': low,
                        'ci95_high': high,
                        'ci95_low_relative': low_rel,
                        'ci95_high_relative': high_rel,
                        'seed_wins': int(np.sum(difference > 0.0)),
                    }
                )
    return pd.DataFrame(rows)


def _aggregate_effect(
    effects: pd.DataFrame,
    condition: str,
    outcome: str,
    comparator: str,
) -> pd.Series:
    selected = effects[
        (effects['switching_condition'] == condition)
        & (effects['outcome'] == outcome)
        & (effects['comparator'] == comparator)
        & (effects['seed'] == 'ALL')
    ]
    if len(selected) != 1:
        raise RuntimeError('missing unique aggregate pairwise effect')
    return selected.iloc[0]


def _build_gates(
    config: dict[str, Any],
    readiness: pd.DataFrame,
    effects: pd.DataFrame,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    audit = config['d0_6_audit']
    limits = audit['readiness']
    acceptance = audit['acceptance']
    gates: list[dict[str, Any]] = []
    test_readiness = bool(
        readiness['delay_violation_fraction'].between(
            float(limits['minimum_delay_violation_fraction']),
            float(limits['maximum_delay_violation_fraction']),
            inclusive='both',
        ).all()
        and (readiness['distinct_top1_channels'] >= int(
            limits['minimum_distinct_top1_channels']
        )).all()
        and (readiness['maximum_top1_channel_share'] <= float(
            limits['maximum_top1_channel_share']
        )).all()
        and (readiness['oracle_current_top1_match'] < float(
            limits['maximum_oracle_current_top1_match']
        )).all()
        and (readiness['label_std'] >= float(
            limits['minimum_label_std']
        )).all()
        and (readiness['bandwidth_range_mhz'] == 0.0).all()
    )
    d05._gate(
        gates, 'heldout_test_readiness', test_readiness,
        readiness.to_dict('records'), 'all heldout readiness limits pass',
    )
    condition_status: dict[str, dict[str, Any]] = {}
    minimum_wins = int(acceptance['minimum_seed_wins'])
    minimum_gain = float(acceptance['minimum_practical_relative_gain'])
    for condition in acceptance['operational_conditions']:
        mechanism_checks: dict[str, bool] = {}
        for comparator in (
            'current_lcg_hysteresis',
            'channel_placebo_dqi',
            'time_placebo_dqi',
        ):
            effect = _aggregate_effect(
                effects, condition, 'rolling_100ms_q10_mean', comparator
            )
            passed = bool(
                int(effect['seed_wins']) >= minimum_wins
                and float(effect['ci95_low']) > 0.0
            )
            mechanism_checks[comparator] = passed
            d05._gate(
                gates, f'{condition}_tail_oracle_over_{comparator}', passed,
                {
                    'seed_wins': int(effect['seed_wins']),
                    'mean_difference': float(effect['oracle_minus_comparator']),
                    'ci95': [float(effect['ci95_low']), float(effect['ci95_high'])],
                },
                f'>= {minimum_wins}/10 wins and paired CI low > 0',
            )
        practical = _aggregate_effect(
            effects,
            condition,
            'net_latency_constrained_goodput_mbps',
            'current_cqi_hysteresis',
        )
        practical_pass = bool(
            int(practical['seed_wins']) >= minimum_wins
            and float(practical['ci95_low']) > 0.0
            and float(practical['relative_gain']) >= minimum_gain
        )
        d05._gate(
            gates, f'{condition}_practical_oracle_over_current_cqi',
            practical_pass,
            {
                'seed_wins': int(practical['seed_wins']),
                'relative_gain': float(practical['relative_gain']),
                'ci95': [float(practical['ci95_low']), float(practical['ci95_high'])],
            },
            f'>= {minimum_wins}/10 wins, CI low > 0, gain >= {minimum_gain}',
        )
        condition_status[condition] = {
            'mechanism_passed': bool(all(mechanism_checks.values())),
            'mechanism_checks': mechanism_checks,
            'practical_passed': practical_pass,
            'fully_passed': bool(all(mechanism_checks.values()) and practical_pass),
        }
    return gates, condition_status


def _make_figures(metrics: pd.DataFrame, output_dir: Path) -> None:
    summary = metrics.groupby(
        ['switching_condition', 'policy'], as_index=False
    ).agg(
        mean_goodput=('net_latency_constrained_goodput_mbps', 'mean'),
        sd_goodput=('net_latency_constrained_goodput_mbps', 'std'),
        mean_tail=('rolling_100ms_q10_mean', 'mean'),
        sd_tail=('rolling_100ms_q10_mean', 'std'),
    )
    labels = {
        'current_cqi_hysteresis': 'Current CQI',
        'current_lcg_hysteresis': 'Current LCG',
        'oracle_dqi': 'Oracle DQI',
        'channel_placebo_dqi': 'Channel placebo',
        'time_placebo_dqi': 'Time placebo',
        'future_mean_oracle': 'Future-mean oracle',
    }
    colors = {
        policy: plt.get_cmap('tab10')(index)
        for index, policy in enumerate(POLICIES)
    }
    markers = {'C0': 'o', 'C1': 's', 'C2': '^'}
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.8))
    for policy in POLICIES:
        for condition in ('C0', 'C1', 'C2'):
            row = summary[
                (summary['policy'] == policy)
                & (summary['switching_condition'] == condition)
            ].iloc[0]
            axes[0].scatter(
                row['mean_goodput'], row['mean_tail'],
                marker=markers[condition], color=colors[policy], s=42,
            )
    axes[0].set_xlabel('Mean net LC-goodput (Mbps)')
    axes[0].set_ylabel('Mean rolling 100 ms q10 (Mbps)')
    axes[0].grid(alpha=0.2)
    condition_handles = [
        Line2D(
            [0], [0], marker=marker, linestyle='none', color='black',
            label=condition,
        )
        for condition, marker in markers.items()
    ]
    axes[0].legend(
        handles=condition_handles, title='Switching', fontsize=8,
        title_fontsize=8, loc='upper left',
    )
    for policy in POLICIES:
        values = summary[summary['policy'] == policy]
        axes[1].plot(
            values['switching_condition'], values['mean_goodput'],
            marker='o', color=colors[policy], label=labels[policy],
        )
    axes[1].set_xlabel('Switching condition')
    axes[1].set_ylabel('Mean net LC-goodput (Mbps)')
    axes[1].grid(alpha=0.2)
    policy_handles = [
        Line2D([0], [0], color=colors[policy], marker='o', label=labels[policy])
        for policy in POLICIES
    ]
    figure.legend(
        handles=policy_handles, loc='lower center', ncol=3,
        fontsize=8, frameon=False,
    )
    figure.tight_layout(rect=(0, 0.15, 1, 1))
    figure.savefig(output_dir / 'd0_6_mean_tail_tradeoff.pdf')
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))
    for policy in POLICIES:
        values = summary[summary['policy'] == policy]
        axes[0].plot(
            values['switching_condition'], values['mean_goodput'],
            marker='o', color=colors[policy], label=labels[policy],
        )
        axes[1].plot(
            values['switching_condition'], values['mean_tail'],
            marker='o', color=colors[policy], label=labels[policy],
        )
    axes[0].set_ylabel('Mean net LC-goodput (Mbps)')
    axes[1].set_ylabel('Mean rolling 100 ms q10 (Mbps)')
    for axis in axes:
        axis.set_xlabel('Switching condition')
        axis.grid(alpha=0.2)
    figure.legend(
        handles=policy_handles, loc='lower center', ncol=3,
        fontsize=8, frameon=False,
    )
    figure.tight_layout(rect=(0, 0.15, 1, 1))
    figure.savefig(output_dir / 'd0_6_switching_cost_profile.pdf')
    plt.close(figure)

    placebo_policies = (
        'current_lcg_hysteresis', 'oracle_dqi',
        'channel_placebo_dqi', 'time_placebo_dqi',
    )
    subset = summary[
        summary['switching_condition'].isin(['C1', 'C2'])
        & summary['policy'].isin(placebo_policies)
    ]
    figure, axis = plt.subplots(figsize=(8.5, 4.2))
    width = 0.18
    x = np.arange(2)
    for offset, policy in enumerate(placebo_policies):
        values = subset[subset['policy'] == policy].set_index(
            'switching_condition'
        ).loc[['C1', 'C2']]
        axis.bar(
            x + (offset - 1.5) * width, values['mean_tail'], width,
            label=labels[policy],
        )
    axis.set_xticks(x, ['C1', 'C2'])
    axis.set_ylabel('Mean rolling 100 ms q10 (Mbps)')
    axis.grid(axis='y', alpha=0.2)
    axis.legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(output_dir / 'd0_6_placebo_comparison.pdf')
    plt.close(figure)


def run_audit(config: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_config, calibration, calibration_summary = calibrate_readiness(config)
    calibration.to_csv(
        output_dir / 'd0_6_readiness_calibration.csv', index=False
    )
    if selected_config is None:
        selected_payload = {
            'selected': False,
            'reason': 'no preregistered bad-dwell candidate passed readiness',
        }
        (output_dir / 'd0_6_selected_physical_config.json').write_text(
            json.dumps(selected_payload, indent=2), encoding='utf-8'
        )
        report = {
            'passed': False,
            'stop_before_d1': True,
            'decision': 'calibration_failed',
            'gates': [{
                'gate': 'readiness_calibration', 'passed': False,
                'value': selected_payload['reason'],
                'threshold': 'first preregistered candidate passes',
            }],
        }
        (output_dir / 'd0_6_gate_report.json').write_text(
            json.dumps(report, indent=2), encoding='utf-8'
        )
        return report

    selected_payload = {
        'selected': True,
        **calibration_summary,
        'channel': selected_config['channel'],
        'selection_used_policy_metrics': False,
    }
    (output_dir / 'd0_6_selected_physical_config.json').write_text(
        json.dumps(selected_payload, indent=2), encoding='utf-8'
    )
    r_ref, scaler_hash, scaler = _fit_frozen_training_statistics(selected_config)
    policy_rows: list[dict[str, Any]] = []
    readiness_rows: list[dict[str, Any]] = []
    for seed in selected_config['d0_6_audit']['test_seeds']:
        rows, readiness = _evaluate_test_seed(
            selected_config, int(seed), r_ref
        )
        policy_rows.extend(rows)
        readiness_rows.append(readiness)
    metrics = pd.DataFrame(policy_rows)
    readiness = pd.DataFrame(readiness_rows)
    effects = _pairwise_effects(
        metrics, int(selected_config['d0_6_audit']['bootstrap_samples'])
    )
    readiness.to_csv(output_dir / 'd0_6_trace_readiness.csv', index=False)
    metrics.to_csv(output_dir / 'd0_6_policy_metrics.csv', index=False)
    effects.to_csv(output_dir / 'd0_6_pairwise_effects.csv', index=False)
    gates, condition_status = _build_gates(
        selected_config, readiness, effects
    )
    calibration_gate = {
        'gate': 'readiness_calibration',
        'passed': True,
        'value': calibration_summary['bad_dwell_slots'],
        'threshold': 'first preregistered candidate passes without policy metrics',
    }
    gates.insert(0, calibration_gate)
    c1_full = condition_status['C1']['fully_passed']
    c2_full = condition_status['C2']['fully_passed']
    heldout_ready = bool(gates[1]['passed'])
    if heldout_ready and c1_full and c2_full:
        decision = 'full_incremental_value'
        stop_before_d1 = False
    elif heldout_ready and c2_full:
        decision = 'c2_switching_cost_conditional_value'
        stop_before_d1 = False
    elif heldout_ready and any(
        status['mechanism_passed'] for status in condition_status.values()
    ):
        decision = 'mechanism_only_no_practical_value'
        stop_before_d1 = True
    else:
        decision = 'no_independent_future_dqi_value'
        stop_before_d1 = True
    passed = not stop_before_d1
    _make_figures(metrics, output_dir)

    manifest = {
        'experiment': 'd0_6_dqi_incremental_value',
        'config_hash': d05._json_hash(selected_config),
        'label_version': DQI_LABEL_VERSION,
        'selected_bad_dwell_slots': calibration_summary['bad_dwell_slots'],
        'calibration_seeds': selected_config['d0_6_audit']['calibration_seeds'],
        'test_seeds': selected_config['d0_6_audit']['test_seeds'],
        'r_ref_mbps': r_ref,
        'reference_source': 'train_z_q95',
        'scaler_hash': scaler_hash,
        'scaler': scaler,
        'policies': list(POLICIES),
        'outcomes': list(OUTCOMES),
        'switching_conditions': selected_config['d0_6_audit']['switching_conditions'],
        'test_trace_hashes': {
            row['trace_id']: row['trace_hash'] for row in readiness_rows
        },
        'calibration_used_policy_metrics': False,
        'd0_5_results_immutable': True,
    }
    (output_dir / 'd0_6_manifest.json').write_text(
        json.dumps(manifest, indent=2), encoding='utf-8'
    )
    report = {
        'passed': passed,
        'stop_before_d1': stop_before_d1,
        'decision': decision,
        'r_ref_mbps': r_ref,
        'scaler_hash': scaler_hash,
        'condition_status': condition_status,
        'gates': gates,
    }
    (output_dir / 'd0_6_gate_report.json').write_text(
        json.dumps(report, indent=2), encoding='utf-8'
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Run D0.6 DQI incremental-value and mechanism confirmation.'
    )
    parser.add_argument(
        '--config', default='configs/e2_d0_6_dqi_incremental_value.json'
    )
    parser.add_argument(
        '--output-dir', default='results/d0_6_dqi_incremental_value'
    )
    args = parser.parse_args()
    report = run_audit(load_config(args.config), Path(args.output_dir))
    print(json.dumps({
        'passed': report['passed'],
        'stop_before_d1': report['stop_before_d1'],
        'decision': report['decision'],
        'output_dir': str(Path(args.output_dir).resolve()),
    }, indent=2))


if __name__ == '__main__':
    main()
