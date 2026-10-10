from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cvn_rl.experiments.multi_vehicle_runner import run_multi_vehicle_experiment
from cvn_rl.multi_metrics import summarize_checkpoint_table
from cvn_rl.simulation.multi_traces import generate_multi_vehicle_trace
from cvn_rl.simulation.rate_model import (
    underlay_rate_mbps,
    underlay_transmit_power_mw,
)


def run_e2_r1_experiment(
    config: dict[str, Any],
    dry_run: bool = False,
    quick: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run power/load calibration before the preregistered E2-R1 matrix."""

    resolved = copy.deepcopy(config)
    run_dir = Path(str(config.get('_run_dir', '.')))
    tables_dir = run_dir / 'tables'
    analysis_dir = run_dir / 'analysis'
    tables_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    power_table, selected_fraction = calibrate_underlay_power_fraction(resolved)
    power_table.to_csv(tables_dir / 'e2_r1_power_calibration.csv', index=False)
    diagnostic = bool(dry_run or quick)
    if selected_fraction is None:
        _write_calibration_failure(
            analysis_dir,
            'No underlay power candidate produced an idle overlay/underlay '
            'median rate ratio in [2, 4].',
        )
        if not diagnostic:
            return pd.DataFrame(), pd.DataFrame()
        selected_fraction = float(
            resolved.get('power_calibration', {}).get(
                'diagnostic_fallback_fraction',
                0.2,
            )
        )
    resolved.setdefault('underlay_power', {})[
        'service_power_ceiling_fraction'
    ] = float(selected_fraction)

    final_calibration, selected_arrival = _run_or_skip_load_calibration(
        resolved,
        dry_run=dry_run,
        quick=quick,
    )
    final_calibration['calibration_selected'] = (
        np.isclose(
            pd.to_numeric(
                final_calibration.get('arrival_probability'),
                errors='coerce',
            ),
            float(selected_arrival),
        )
        if selected_arrival is not None and not final_calibration.empty
        else False
    )
    final_calibration.to_csv(
        tables_dir / 'e2_r1_load_calibration.csv',
        index=False,
    )
    if selected_arrival is None:
        _write_calibration_failure(
            analysis_dir,
            'No arrival probability satisfied the preregistered queue and '
            'completion-rate calibration criteria.',
        )
        return pd.DataFrame(), pd.DataFrame()

    resolved['arrival_probabilities'] = [float(selected_arrival)]
    resolved.setdefault('packet_traffic', {})[
        'arrival_probability'
    ] = float(selected_arrival)
    resolved['_run_scope'] = 'formal'
    resolved_hash = _config_hash(resolved)
    resolved['_resolved_config_hash'] = resolved_hash
    (analysis_dir / 'e2_r1_resolved_config.json').write_text(
        json.dumps(
            {
                **_serializable_config(resolved),
                'selected_arrival_probability': float(selected_arrival),
                'selected_underlay_service_power_ceiling_fraction': float(
                    selected_fraction
                ),
                'resolved_config_hash': resolved_hash,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding='utf-8',
    )
    metrics, summary = run_multi_vehicle_experiment(
        resolved,
        dry_run=dry_run,
        quick=quick,
    )
    return metrics, summary


def calibrate_underlay_power_fraction(
    config: dict[str, Any],
) -> tuple[pd.DataFrame, float | None]:
    calibration_cfg = config.get('power_calibration', {})
    if not bool(calibration_cfg.get('enabled', True)):
        selected = float(
            config.get('underlay_power', {}).get(
                'service_power_ceiling_fraction',
                1.0,
            )
        )
        return pd.DataFrame(
            [
                {
                    'candidate_fraction': selected,
                    'median_idle_overlay_underlay_rate_ratio': np.nan,
                    'accepted': True,
                    'distance_to_target': np.nan,
                    'sample_count': 0,
                    'selected': True,
                    'calibration_skipped': True,
                }
            ]
        ), selected
    candidates = [
        float(value)
        for value in calibration_cfg.get('candidates', [0.1, 0.2, 0.3])
    ]
    target = float(calibration_cfg.get('target_ratio', 3.0))
    ratio_min, ratio_max = calibration_cfg.get('accepted_ratio', [2.0, 4.0])
    scenario_config = _first_scenario_config(config)
    slots = max(int(calibration_cfg.get('slots', 5_000)), 10)
    seed = int(calibration_cfg.get('seed', 9_000))
    trace = generate_multi_vehicle_trace(
        scenario_config,
        seed=seed,
        slots=slots,
        split='power_calibration',
    )
    idle = ~trace.occupancy[:, None, :]
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        candidate_config = copy.deepcopy(scenario_config)
        candidate_config.setdefault('underlay_power', {})[
            'service_power_ceiling_fraction'
        ] = candidate
        ratios: list[float] = []
        for t, vehicle, channel in np.argwhere(idle):
            quality = float(trace.quality_mbps[t, vehicle, channel])
            transmit_power = underlay_transmit_power_mw(
                float(trace.transmit_power_cap_mw[t, vehicle, channel]),
                candidate_config,
            )
            rate = underlay_rate_mbps(
                quality,
                float(trace.bandwidth_mhz[t, channel]),
                transmit_power,
                False,
                candidate_config,
            )
            if rate > 0.0:
                ratios.append(quality / rate)
        median_ratio = float(np.median(ratios)) if ratios else np.inf
        accepted = bool(float(ratio_min) <= median_ratio <= float(ratio_max))
        rows.append(
            {
                'candidate_fraction': candidate,
                'median_idle_overlay_underlay_rate_ratio': median_ratio,
                'accepted': accepted,
                'distance_to_target': abs(median_ratio - target),
                'sample_count': len(ratios),
                'selected': False,
            }
        )
    table = pd.DataFrame(rows)
    accepted = table[table['accepted']].sort_values(
        ['distance_to_target', 'candidate_fraction'],
        ascending=[True, True],
    )
    selected = None if accepted.empty else float(accepted.iloc[0]['candidate_fraction'])
    if selected is not None:
        table.loc[np.isclose(table['candidate_fraction'], selected), 'selected'] = True
    return table, selected


def select_calibrated_arrival_probability(
    metrics: pd.DataFrame,
    config: dict[str, Any],
) -> float | None:
    if metrics.empty:
        return None
    calibration_cfg = config.get('load_calibration', {})
    neutral_name = str(calibration_cfg['neutral_condition'])
    oracle_name = str(calibration_cfg['oracle_condition'])
    rows: list[tuple[float, bool]] = []
    for arrival, group in metrics.groupby('arrival_probability'):
        neutral = group[group['module_variant'].astype(str) == neutral_name]
        oracle = group[group['module_variant'].astype(str) == oracle_name]
        if neutral.empty or oracle.empty:
            rows.append((float(arrival), False))
            continue
        neutral_queue = float(neutral['mean_queue_occupancy'].mean())
        neutral_completion = float(neutral['completion_rate'].mean())
        oracle_completion = float(oracle['completion_rate'].mean())
        accepted = bool(
            0.20 <= neutral_queue <= 0.60
            and 0.80 <= neutral_completion <= 0.95
            and oracle_completion >= 0.90
        )
        rows.append((float(arrival), accepted))
    valid = sorted(arrival for arrival, accepted in rows if accepted)
    return valid[0] if valid else None


def _run_or_skip_load_calibration(
    config: dict[str, Any],
    dry_run: bool,
    quick: bool,
) -> tuple[pd.DataFrame, float | None]:
    calibration = config.get('load_calibration', {})
    if not bool(calibration.get('enabled', True)):
        configured = config.get('arrival_probabilities') or [
            config.get('packet_traffic', {}).get('arrival_probability', 0.5)
        ]
        selected = float(configured[0])
        return pd.DataFrame(
            [
                {
                    'arrival_probability': selected,
                    'calibration_selected': True,
                    'calibration_skipped': True,
                }
            ]
        ), selected

    calibration_cfg = _load_calibration_config(config)
    calibration_metrics, _ = run_multi_vehicle_experiment(
        calibration_cfg,
        dry_run=dry_run,
        quick=quick,
    )
    final = _final_rows(calibration_metrics)
    selected = select_calibrated_arrival_probability(final, config)
    if dry_run or quick:
        selected = float(
            calibration.get('diagnostic_arrival_probability', 0.2)
        )
    return final, selected


def evaluate_e2_r1_gate(
    metrics: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame]:
    gate = str(config.get('e2_r1_gate', '')).lower()
    final = _final_rows(metrics)
    if final.empty or 'module_variant' not in final:
        checks = pd.DataFrame(
            [
                {
                    'check': 'formal_matrix_available',
                    'passed': False,
                    'value': 0,
                    'requirement': (
                        'load calibration must select a preregistered arrival '
                        'probability before formal training'
                    ),
                }
            ]
        )
        payload = {
            'passed': False,
            'gate': gate,
            'resolved_config_hash': '',
            'stop_reason': 'calibration_failed_or_formal_matrix_empty',
            'checks': json.loads(checks.to_json(orient='records')),
        }
        return payload, checks
    if gate == 'prediction_positive_control':
        payload, checks = _evaluate_prediction_positive_control(final, config)
    elif gate == 'prediction_positive_control_r2':
        payload, checks = _evaluate_prediction_positive_control_r2(final, config)
    elif gate == 'prediction_representation_bridge':
        payload, checks = _evaluate_prediction_representation_bridge(
            final,
            config,
        )
    elif gate == 'prediction_time_alignment_degradation':
        payload, checks = _evaluate_prediction_time_alignment_degradation(
            final,
            config,
        )
    elif gate == 'prediction_staleness_degradation':
        payload, checks = _evaluate_prediction_staleness_degradation(
            final,
            config,
        )
    elif gate == 'prediction_causal_teacher':
        payload, checks = _evaluate_prediction_causal_teacher(
            final,
            config,
        )
    elif gate == 'prediction_frozen_predictor':
        payload, checks = _evaluate_prediction_frozen_predictor(
            final,
            config,
        )
    elif gate == 'prediction_sensing_redundancy':
        payload, checks = _evaluate_prediction_sensing_redundancy(final, config)
    elif gate == 'dqi_positive_control':
        payload, checks = _evaluate_dqi_positive_control(final, config)
    elif gate == 'prediction':
        payload, checks = _evaluate_prediction_gate(final, config)
    elif gate == 'dqi':
        payload, checks = _evaluate_dqi_gate(final, config)
    else:
        raise ValueError(f'unsupported E2-R1 gate: {gate}')
    payload['gate'] = gate
    payload['resolved_config_hash'] = (
        str(final['resolved_config_hash'].iloc[0])
        if not final.empty and 'resolved_config_hash' in final
        else ''
    )
    return payload, checks


def render_e2_r1_report(payload: dict[str, Any]) -> str:
    status = 'PASS' if payload.get('passed') else 'FAIL'
    gate = str(payload.get('gate', ''))
    positive_control = 'positive_control' in gate
    title = (
        f'# {gate} 最大作用正对照报告'
        if positive_control
        else f'# E2-R1 {gate.upper()} Oracle Gate'
    )
    lines = [
        title,
        '',
        f'- Decision（判定）: **{status}**',
        (
            '- Resolved config hash（解析后配置哈希）: '
            f'`{payload.get("resolved_config_hash", "")}`'
        ),
        (
            '- Evidence class（证据类别）: privileged diagnostic upper bound'
            '（特权诊断上界），不可作为可部署模型证据。'
        ),
        '',
        '## Checks（门槛检查）',
        '',
        '| Check | Passed | Value | Requirement |',
        '|---|---:|---:|---|',
    ]
    if payload.get('stop_reason'):
        lines[4:4] = [f"- Stop reason: `{payload['stop_reason']}`"]
    for check in payload.get('checks', []):
        lines.append(
            f"| {check['check']} | {str(bool(check['passed']))} | "
            f"{check.get('value', '')} | {check.get('requirement', '')} |"
        )
    lines.extend(
        [
            '',
            'Oracle（特权真值上界）成功只说明信息价值在该环境中可识别，'
            '不证明 causal surrogate（因果替代描述子）或 frozen ML model'
            '（冻结机器学习模型）有效。',
        ]
    )
    return '\n'.join(lines) + '\n'


def e2_r1_paired_seed_effects(
    metrics: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    final = _final_rows(metrics)
    if final.empty:
        return pd.DataFrame()
    if str(config.get('e2_r1_gate', '')).lower() == (
        'prediction_representation_bridge'
    ):
        return _representation_bridge_effects(final, config)
    names = config.get('gate_conditions', {})
    oracle = str(names.get('oracle', ''))
    r2 = str(config.get('e2_r1_gate', '')).lower() == (
        'prediction_positive_control_r2'
    )
    shuffled = str(
        names.get('primary_placebo' if r2 else 'shuffled', '')
    )
    neutral = str(
        names.get('replicate_placebo' if r2 else 'neutral', '')
    )
    oracle_effect_label = (
        'oracle_minus_primary_placebo' if r2 else 'oracle_minus_shuffled'
    )
    placebo_effect_label = (
        'primary_minus_replicate_placebo'
        if r2
        else 'shuffled_minus_neutral'
    )
    index = [
        column
        for column in ['scenario', 'arrival_probability', 'agent', 'seed']
        if column in final
    ]
    metrics_to_pair = [
        column
        for column in [
            'served_goodput_mbps',
            'overlay_collision_rate',
            'pu_access_violation_rate',
            'mean_safe_action_regret_mbps',
            'mean_channel_selection_regret',
            'switch_count',
            'blackout_loss_bits',
            'dqi_action_disagreement_rate',
        ]
        if column in final
    ]
    pivot = final.pivot_table(
        index=index,
        columns='module_variant',
        values=metrics_to_pair,
        aggfunc='mean',
    )
    pivot.columns = [
        f'{condition}:{metric}' for metric, condition in pivot.columns
    ]
    pivot = pivot.reset_index()
    for metric in metrics_to_pair:
        oracle_col = f'{oracle}:{metric}'
        shuffled_col = f'{shuffled}:{metric}'
        neutral_col = f'{neutral}:{metric}'
        if oracle_col in pivot and shuffled_col in pivot:
            pivot[f'{oracle_effect_label}:{metric}'] = (
                pivot[oracle_col] - pivot[shuffled_col]
            )
        if shuffled_col in pivot and neutral_col in pivot:
            pivot[f'{placebo_effect_label}:{metric}'] = (
                pivot[shuffled_col] - pivot[neutral_col]
            )
    return pivot


def _representation_bridge_effects(
    final: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    mappings = config.get('gate_conditions', {}).get(
        'representations',
        {},
    )
    index = [
        column
        for column in ['scenario', 'arrival_probability', 'agent', 'seed']
        if column in final
    ]
    metrics_to_pair = [
        column
        for column in [
            'served_goodput_mbps',
            'pu_access_violation_rate',
            'overlay_given_busy_legal_backlog',
            'mean_safe_action_regret_mbps',
            'underlay_selection_ratio',
        ]
        if column in final
    ]
    pivot = final.pivot_table(
        index=index,
        columns='module_variant',
        values=metrics_to_pair,
        aggfunc='mean',
    )
    pivot.columns = [
        f'{condition}:{metric}'
        for metric, condition in pivot.columns
    ]
    pivot = pivot.reset_index()
    for representation, names in mappings.items():
        oracle = str(names['oracle'])
        primary = str(names['primary_placebo'])
        replicate = str(names['replicate_placebo'])
        for metric in metrics_to_pair:
            oracle_column = f'{oracle}:{metric}'
            primary_column = f'{primary}:{metric}'
            replicate_column = f'{replicate}:{metric}'
            if oracle_column in pivot and primary_column in pivot:
                pivot[
                    f'{representation}_oracle_minus_placebo_a:{metric}'
                ] = pivot[oracle_column] - pivot[primary_column]
            if primary_column in pivot and replicate_column in pivot:
                pivot[
                    f'{representation}_placebo_a_minus_b:{metric}'
                ] = pivot[primary_column] - pivot[replicate_column]
    return pivot


def _evaluate_prediction_gate(
    final: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame]:
    names = config.get('gate_conditions', {})
    oracle_name = str(names.get('oracle', 'theta_oracle'))
    shuffled_name = str(names.get('shuffled', 'theta_time_shuffled'))
    paired = _paired(final, oracle_name, shuffled_name)
    goodput_delta = _relative_delta(paired, 'served_goodput_mbps')
    collision_reduction = _relative_reduction(paired, 'overlay_collision_rate')
    goodput_direction = bool((paired[f'{oracle_name}:served_goodput_mbps'] > paired[f'{shuffled_name}:served_goodput_mbps']).all()) if not paired.empty else False
    collision_direction = bool((paired[f'{oracle_name}:overlay_collision_rate'] < paired[f'{shuffled_name}:overlay_collision_rate']).all()) if not paired.empty else False
    throughput_path = goodput_direction and _mean(goodput_delta) >= 0.03
    safety_path = bool(
        collision_direction
        and _mean(collision_reduction) >= 0.30
        and _mean(goodput_delta) >= -0.02
    )
    oracle = final[final['module_variant'].astype(str) == oracle_name]
    shuffled = final[final['module_variant'].astype(str) == shuffled_name]
    common = {
        'paired_seeds': len(paired) == 3,
        'effect_path': throughput_path or safety_path,
        'pu_safety': _mean(oracle.get('pu_access_violation_rate')) <= 1e-3,
        'regret_direction': _all_less(
            paired,
            oracle_name,
            shuffled_name,
            'mean_safe_action_regret_mbps',
        ),
        'busy_overlay_direction': _all_less(
            paired,
            oracle_name,
            shuffled_name,
            'overlay_given_busy_legal_backlog',
        ),
        'idle_overlay_noninferiority': (
            _mean(oracle.get('overlay_given_idle_legal_backlog'))
            - _mean(shuffled.get('overlay_given_idle_legal_backlog'))
            >= -0.02
        ),
        'paired_physical_trace_and_mask': _paired_structure(final),
    }
    values = {
        'paired_seeds': len(paired),
        'effect_path': f'throughput={throughput_path}, safety={safety_path}',
        'pu_safety': _mean(oracle.get('pu_access_violation_rate')),
        'regret_direction': 'oracle < shuffled in every paired seed',
        'busy_overlay_direction': 'oracle < shuffled in every paired seed',
        'idle_overlay_noninferiority': (
            _mean(oracle.get('overlay_given_idle_legal_backlog'))
            - _mean(shuffled.get('overlay_given_idle_legal_backlog'))
        ),
        'paired_physical_trace_and_mask': common['paired_physical_trace_and_mask'],
    }
    requirements = {
        'paired_seeds': 'exactly 3 paired seeds',
        'effect_path': 'goodput +3% or collision -30% with goodput >= -2%',
        'pu_safety': '<= 1e-3',
        'regret_direction': '3/3 seeds',
        'busy_overlay_direction': '3/3 seeds',
        'idle_overlay_noninferiority': 'mean difference >= -0.02',
        'paired_physical_trace_and_mask': 'identical within each seed',
    }
    return _gate_payload(common, values, requirements)


def _evaluate_prediction_positive_control(
    final: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame]:
    names = config.get('gate_conditions', {})
    oracle_name = str(names.get('oracle', 'theta_oracle'))
    shuffled_name = str(names.get('shuffled', 'theta_time_shuffled'))
    neutral_name = str(names.get('neutral', 'theta_neutral'))
    paired = _paired(final, oracle_name, shuffled_name)
    oracle = final[final['module_variant'].astype(str) == oracle_name]
    shuffled = final[final['module_variant'].astype(str) == shuffled_name]
    neutral = final[final['module_variant'].astype(str) == neutral_name]
    violation_reduction = _relative_reduction(
        paired,
        'pu_access_violation_rate',
    )
    goodput_delta = _relative_delta(paired, 'served_goodput_mbps')
    shuffled_neutral_goodput = _relative_group_difference(
        shuffled,
        neutral,
        'served_goodput_mbps',
    )
    shuffled_neutral_violation = abs(
        _mean(shuffled.get('pu_access_violation_rate'))
        - _mean(neutral.get('pu_access_violation_rate'))
    )
    checks = {
        'formal_run': (
            not bool(final.get('diagnostic_run', pd.Series([True])).astype(bool).any())
            and set(final.get('run_scope', pd.Series(dtype=str)).astype(str)) == {'formal'}
        ),
        'paired_seeds': len(paired) == 3,
        'pu_violation_effect': (
            _all_less(
                paired,
                oracle_name,
                shuffled_name,
                'pu_access_violation_rate',
            )
            and _mean(violation_reduction) >= 0.80
            and _mean(oracle.get('pu_access_violation_rate')) <= 1e-3
        ),
        'goodput_effect': (
            _all_greater(
                paired,
                oracle_name,
                shuffled_name,
                'served_goodput_mbps',
            )
            and _mean(goodput_delta) >= 0.05
        ),
        'busy_overlay_effect': _all_less(
            paired,
            oracle_name,
            shuffled_name,
            'overlay_given_busy_legal_backlog',
        ),
        'placebo_equivalence': (
            abs(shuffled_neutral_goodput) <= 0.02
            and shuffled_neutral_violation <= 0.01
        ),
        'paired_physical_trace_and_mask': _paired_structure(final),
    }
    values = {
        'formal_run': not bool(
            final.get('diagnostic_run', pd.Series([True])).astype(bool).any()
        ),
        'paired_seeds': len(paired),
        'pu_violation_effect': (
            f'reduction={_mean(violation_reduction):.6f}, '
            f'oracle={_mean(oracle.get("pu_access_violation_rate")):.6f}'
        ),
        'goodput_effect': _mean(goodput_delta),
        'busy_overlay_effect': 'oracle < shuffled in every paired seed',
        'placebo_equivalence': (
            f'goodput={shuffled_neutral_goodput:.6f}, '
            f'violation={shuffled_neutral_violation:.6f}'
        ),
        'paired_physical_trace_and_mask': checks[
            'paired_physical_trace_and_mask'
        ],
    }
    requirements = {
        'formal_run': 'not --dry-run or --quick; run_scope=formal',
        'paired_seeds': 'exactly 3 paired seeds',
        'pu_violation_effect': '3/3 lower, mean reduction >= 80%, oracle <= 1e-3',
        'goodput_effect': '3/3 higher, mean relative gain >= 5%',
        'busy_overlay_effect': '3/3 lower',
        'placebo_equivalence': '|goodput| <= 2%, violation difference <= 0.01',
        'paired_physical_trace_and_mask': 'identical within each paired seed',
    }
    return _gate_payload(checks, values, requirements)


def _evaluate_prediction_positive_control_r2(
    final: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame]:
    names = config.get('gate_conditions', {})
    oracle_name = str(names.get('oracle', 'theta_oracle'))
    primary_name = str(names.get('primary_placebo', 'theta_placebo_a'))
    replicate_name = str(names.get('replicate_placebo', 'theta_placebo_b'))
    audit_names = [oracle_name, primary_name, replicate_name]
    paired = _paired(final, oracle_name, primary_name)
    placebo_paired = _paired(final, primary_name, replicate_name)
    oracle = final[final['module_variant'].astype(str) == oracle_name]
    primary = final[final['module_variant'].astype(str) == primary_name]
    replicate = final[final['module_variant'].astype(str) == replicate_name]
    violation_reduction = _relative_reduction(
        paired,
        'pu_access_violation_rate',
    )
    goodput_delta = _relative_delta(paired, 'served_goodput_mbps')
    placebo_goodput = _relative_group_difference(
        primary,
        replicate,
        'served_goodput_mbps',
    )
    placebo_violation = abs(
        _mean(primary.get('pu_access_violation_rate'))
        - _mean(replicate.get('pu_access_violation_rate'))
    )
    max_alignment = float(
        config.get('placebo_max_abs_theta_idle_correlation', 0.05)
    )
    alignment_columns = [
        'train_theta_idle_correlation',
        'evaluation_theta_idle_correlation',
    ]
    alignment_values = _finite_abs_values(
        final,
        [primary_name, replicate_name],
        alignment_columns,
    )
    max_observed_alignment = max(
        alignment_values,
        default=float('inf'),
    )
    oracle_violation = _mean(oracle.get('pu_access_violation_rate'))
    checks = {
        'formal_run': (
            not bool(final.get('diagnostic_run', pd.Series([True])).astype(bool).any())
            and set(final.get('run_scope', pd.Series(dtype=str)).astype(str)) == {'formal'}
        ),
        'paired_seeds': len(paired) == 3 and len(placebo_paired) == 3,
        'placebo_marginal_distribution': _variants_share_audit_values(
            final,
            audit_names,
            [
                'train_theta_marginal_hash',
                'evaluation_theta_marginal_hash',
            ],
        ),
        'placebo_sequences_distinct': _variants_have_distinct_values(
            final,
            audit_names,
            'theta_observation_hash',
        ),
        'placebo_alignment_broken': (
            len(alignment_values) == 12
            and max_observed_alignment <= max_alignment
        ),
        'pu_violation_effect': (
            _all_less(
                paired,
                oracle_name,
                primary_name,
                'pu_access_violation_rate',
            )
            and _mean(violation_reduction) >= 0.80
            and oracle_violation <= 1e-3
        ),
        'goodput_effect': (
            _all_greater(
                paired,
                oracle_name,
                primary_name,
                'served_goodput_mbps',
            )
            and _mean(goodput_delta) >= 0.05
        ),
        'busy_overlay_effect': _all_less(
            paired,
            oracle_name,
            primary_name,
            'overlay_given_busy_legal_backlog',
        ),
        'placebo_equivalence': (
            abs(placebo_goodput) <= 0.02
            and placebo_violation <= 0.01
        ),
        'paired_physical_trace_and_mask': _paired_structure(final),
    }
    values = {
        'formal_run': not bool(
            final.get('diagnostic_run', pd.Series([True])).astype(bool).any()
        ),
        'paired_seeds': (
            f'oracle={len(paired)}, placebo={len(placebo_paired)}'
        ),
        'placebo_marginal_distribution': (
            'oracle/placebo-A/placebo-B multiset hashes match per paired seed'
        ),
        'placebo_sequences_distinct': (
            'oracle/placebo-A/placebo-B observation hashes differ per paired seed'
        ),
        'placebo_alignment_broken': (
            f'max_abs_correlation={max_observed_alignment:.6f}'
        ),
        'pu_violation_effect': (
            f'reduction={_mean(violation_reduction):.6f}, '
            f'oracle={oracle_violation:.6f}'
        ),
        'goodput_effect': _mean(goodput_delta),
        'busy_overlay_effect': 'oracle < placebo-A in every paired seed',
        'placebo_equivalence': (
            f'goodput={placebo_goodput:.6f}, '
            f'violation={placebo_violation:.6f}'
        ),
        'paired_physical_trace_and_mask': checks[
            'paired_physical_trace_and_mask'
        ],
    }
    requirements = {
        'formal_run': 'not --dry-run or --quick; run_scope=formal',
        'paired_seeds': 'exactly 3 paired seeds in both comparisons',
        'placebo_marginal_distribution': (
            'exact train/evaluation theta multiset match within each paired seed'
        ),
        'placebo_sequences_distinct': (
            'three distinct evaluation theta sequences within each paired seed'
        ),
        'placebo_alignment_broken': (
            f'|theta-idle correlation| <= {max_alignment:.3f} '
            'for both placebos in train and evaluation'
        ),
        'pu_violation_effect': (
            '3/3 lower, mean reduction >= 80%, oracle <= 1e-3'
        ),
        'goodput_effect': '3/3 higher, mean relative gain >= 5%',
        'busy_overlay_effect': '3/3 lower',
        'placebo_equivalence': (
            '|goodput| <= 2%, violation difference <= 0.01'
        ),
        'paired_physical_trace_and_mask': (
            'identical within each paired seed'
        ),
    }
    return _gate_payload(checks, values, requirements)


def _evaluate_prediction_representation_bridge(
    final: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame]:
    representation_names = (
        config.get('gate_conditions', {}).get('representations', {})
    )
    required_representations = (
        'hard',
        'soft',
        'continuous_separable',
    )
    if not all(name in representation_names for name in required_representations):
        raise ValueError(
            'representation bridge requires hard, soft, and '
            'continuous_separable Gate mappings'
        )
    max_alignment = float(
        config.get('placebo_max_abs_theta_idle_correlation', 0.05)
    )
    checks: dict[str, bool] = {
        'formal_run': (
            not bool(
                final.get(
                    'diagnostic_run',
                    pd.Series([True]),
                ).astype(bool).any()
            )
            and set(
                final.get(
                    'run_scope',
                    pd.Series(dtype=str),
                ).astype(str)
            )
            == {'formal'}
        ),
    }
    values: dict[str, Any] = {
        'formal_run': checks['formal_run'],
    }
    requirements: dict[str, str] = {
        'formal_run': 'not --dry-run or --quick; run_scope=formal',
    }

    all_alignment_values: list[float] = []
    paired_seed_counts: list[int] = []
    for representation in required_representations:
        names = representation_names[representation]
        oracle_name = str(names['oracle'])
        primary_name = str(names['primary_placebo'])
        replicate_name = str(names['replicate_placebo'])
        audit_names = [oracle_name, primary_name, replicate_name]
        paired = _paired(final, oracle_name, primary_name)
        placebo_paired = _paired(final, primary_name, replicate_name)
        paired_seed_counts.extend([len(paired), len(placebo_paired)])
        oracle = final[
            final['module_variant'].astype(str) == oracle_name
        ]
        primary = final[
            final['module_variant'].astype(str) == primary_name
        ]
        replicate = final[
            final['module_variant'].astype(str) == replicate_name
        ]
        goodput_delta = _relative_delta(
            paired,
            'served_goodput_mbps',
        )
        violation_reduction = _relative_reduction(
            paired,
            'pu_access_violation_rate',
        )
        placebo_goodput = _relative_group_difference(
            primary,
            replicate,
            'served_goodput_mbps',
        )
        placebo_violation = abs(
            _mean(primary.get('pu_access_violation_rate'))
            - _mean(replicate.get('pu_access_violation_rate'))
        )
        oracle_violation = _mean(
            oracle.get('pu_access_violation_rate')
        )
        prefix = representation.replace('_separable', '')
        checks[f'{prefix}_marginal_distribution'] = (
            _variants_share_audit_values(
                final,
                audit_names,
                [
                    'train_theta_marginal_hash',
                    'evaluation_theta_marginal_hash',
                ],
            )
        )
        checks[f'{prefix}_sequences_distinct'] = (
            _variants_have_distinct_values(
                final,
                audit_names,
                'theta_observation_hash',
            )
        )
        checks[f'{prefix}_goodput_effect'] = (
            _all_greater(
                paired,
                oracle_name,
                primary_name,
                'served_goodput_mbps',
            )
            and _mean(goodput_delta) >= 0.05
        )
        checks[f'{prefix}_pu_violation_effect'] = (
            _all_less(
                paired,
                oracle_name,
                primary_name,
                'pu_access_violation_rate',
            )
            and _mean(violation_reduction) >= 0.80
            and oracle_violation <= 1e-3
        )
        checks[f'{prefix}_busy_overlay_effect'] = _all_less(
            paired,
            oracle_name,
            primary_name,
            'overlay_given_busy_legal_backlog',
        )
        checks[f'{prefix}_placebo_equivalence'] = (
            abs(placebo_goodput) <= 0.02
            and placebo_violation <= 0.01
        )
        values[f'{prefix}_marginal_distribution'] = (
            'oracle/placebo-A/placebo-B train and evaluation hashes match'
        )
        values[f'{prefix}_sequences_distinct'] = (
            'three evaluation theta observation hashes differ'
        )
        values[f'{prefix}_goodput_effect'] = _mean(goodput_delta)
        values[f'{prefix}_pu_violation_effect'] = (
            f'reduction={_mean(violation_reduction):.6f}, '
            f'oracle={oracle_violation:.6f}'
        )
        values[f'{prefix}_busy_overlay_effect'] = (
            'oracle < placebo-A in every paired seed'
        )
        values[f'{prefix}_placebo_equivalence'] = (
            f'goodput={placebo_goodput:.6f}, '
            f'violation={placebo_violation:.6f}'
        )
        requirements[f'{prefix}_marginal_distribution'] = (
            'exact theta multiset match within every paired seed'
        )
        requirements[f'{prefix}_sequences_distinct'] = (
            'three distinct evaluation theta sequences per seed'
        )
        requirements[f'{prefix}_goodput_effect'] = (
            '3/3 higher; mean relative gain >= 5%'
        )
        requirements[f'{prefix}_pu_violation_effect'] = (
            '3/3 lower; mean reduction >= 80%; oracle <= 1e-3'
        )
        requirements[f'{prefix}_busy_overlay_effect'] = '3/3 lower'
        requirements[f'{prefix}_placebo_equivalence'] = (
            '|goodput| <= 2%; violation difference <= 0.01'
        )
        all_alignment_values.extend(
            _finite_abs_values(
                final,
                [primary_name, replicate_name],
                [
                    'train_theta_idle_correlation',
                    'evaluation_theta_idle_correlation',
                ],
            )
        )

    checks['paired_seeds'] = bool(
        paired_seed_counts
        and all(count == 3 for count in paired_seed_counts)
    )
    values['paired_seeds'] = ','.join(
        str(count) for count in paired_seed_counts
    )
    requirements['paired_seeds'] = (
        'exactly 3 seeds for all oracle/placebo and placebo/placebo pairs'
    )
    max_observed_alignment = max(
        all_alignment_values,
        default=float('inf'),
    )
    checks['placebo_alignment_broken'] = (
        len(all_alignment_values) == 36
        and max_observed_alignment <= max_alignment
    )
    values['placebo_alignment_broken'] = (
        f'max_abs_correlation={max_observed_alignment:.6f}'
    )
    requirements['placebo_alignment_broken'] = (
        f'36 train/evaluation values; max <= {max_alignment:.3f}'
    )

    hard_oracle = str(representation_names['hard']['oracle'])
    soft_oracle = str(representation_names['soft']['oracle'])
    continuous_oracle = str(
        representation_names['continuous_separable']['oracle']
    )
    checks['hard_schema'] = _theta_schema_check(
        final,
        hard_oracle,
        expected_unique_count=2,
        expected_min=0.0,
        expected_max=1.0,
    )
    checks['soft_schema'] = _theta_schema_check(
        final,
        soft_oracle,
        expected_unique_count=2,
        expected_min=0.1,
        expected_max=0.9,
    )
    checks['continuous_schema'] = _continuous_theta_schema_check(
        final,
        continuous_oracle,
    )
    values['hard_schema'] = 'unique=2, min=0, max=1, accuracy=1'
    values['soft_schema'] = 'unique=2, min=0.1, max=0.9, accuracy=1'
    values['continuous_schema'] = (
        'busy<=0.45, idle>=0.55, unique_fraction>=0.95, accuracy=1'
    )
    requirements['hard_schema'] = (
        'train/evaluation values are exactly 0/1 and perfectly separable'
    )
    requirements['soft_schema'] = (
        'train/evaluation values are exactly 0.1/0.9 and perfectly separable'
    )
    requirements['continuous_schema'] = (
        'train/evaluation ranges disjoint at 0.5 and unique fraction >= 95%'
    )

    hard_frame = final[
        final['module_variant'].astype(str) == hard_oracle
    ]
    soft_frame = final[
        final['module_variant'].astype(str) == soft_oracle
    ]
    continuous_frame = final[
        final['module_variant'].astype(str) == continuous_oracle
    ]
    hard_goodput = _mean(hard_frame.get('served_goodput_mbps'))
    soft_noninferiority = (
        _mean(soft_frame.get('served_goodput_mbps')) - hard_goodput
    ) / max(abs(hard_goodput), 1e-12)
    continuous_noninferiority = (
        _mean(continuous_frame.get('served_goodput_mbps')) - hard_goodput
    ) / max(abs(hard_goodput), 1e-12)
    soft_violation = _mean(
        soft_frame.get('pu_access_violation_rate')
    )
    continuous_violation = _mean(
        continuous_frame.get('pu_access_violation_rate')
    )
    checks['representation_noninferiority'] = (
        soft_noninferiority >= -0.02
        and continuous_noninferiority >= -0.02
        and soft_violation <= 1e-3
        and continuous_violation <= 1e-3
    )
    values['representation_noninferiority'] = (
        f'soft_goodput={soft_noninferiority:.6f}, '
        f'continuous_goodput={continuous_noninferiority:.6f}, '
        f'soft_violation={soft_violation:.6f}, '
        f'continuous_violation={continuous_violation:.6f}'
    )
    requirements['representation_noninferiority'] = (
        'soft/continuous goodput >= -2% vs hard; violations <= 1e-3'
    )

    hard_names = {
        str(value)
        for value in representation_names['hard'].values()
    }
    neutral_name = str(
        config.get('gate_conditions', {}).get('neutral', 'theta_neutral')
    )
    reused_expected = hard_names | {neutral_name}
    reused_flag = final.get(
        'reused_curve',
        pd.Series(False, index=final.index),
    ).fillna(False).astype(bool)
    reused_rows = final[
        final['module_variant'].astype(str).isin(reused_expected)
    ]
    new_rows = final[
        ~final['module_variant'].astype(str).isin(reused_expected)
    ]
    requirement = config.get('reuse_gate_requirement', {})
    expected_source_hash = str(
        requirement.get('resolved_config_hash', '')
    )
    expected_source_gate = str(requirement.get('gate', ''))
    reuse_columns_available = {
        'reuse_source_resolved_config_hash',
        'reuse_source_e2_r1_gate',
        'reuse_source_run',
    }.issubset(reused_rows.columns)
    checks['hard_reuse_provenance'] = bool(
        len(reused_rows) == 12
        and reused_flag.loc[reused_rows.index].all()
        and not reused_flag.loc[new_rows.index].any()
        and reuse_columns_available
        and (
            reused_rows['reuse_source_resolved_config_hash']
            .astype(str)
            .eq(expected_source_hash)
            .all()
        )
        and (
            reused_rows['reuse_source_e2_r1_gate']
            .astype(str)
            .eq(expected_source_gate)
            .all()
        )
        and reused_rows['reuse_source_run'].astype(str).str.len().gt(0).all()
    )
    values['hard_reuse_provenance'] = (
        f'rows={len(reused_rows)}, source_hash={expected_source_hash}'
    )
    requirements['hard_reuse_provenance'] = (
        '12 hard/neutral final rows reused from the frozen passing P0-R2 run'
    )
    checks['paired_physical_trace_and_mask'] = _paired_structure(final)
    values['paired_physical_trace_and_mask'] = checks[
        'paired_physical_trace_and_mask'
    ]
    requirements['paired_physical_trace_and_mask'] = (
        'identical within each paired seed across all conditions'
    )
    return _gate_payload(checks, values, requirements)


def _theta_schema_check(
    frame: pd.DataFrame,
    variant: str,
    *,
    expected_unique_count: int,
    expected_min: float,
    expected_max: float,
) -> bool:
    selected = frame[frame['module_variant'].astype(str) == variant]
    if len(selected) != 3:
        return False
    for prefix in ('train', 'evaluation'):
        required = [
            f'{prefix}_theta_unique_count',
            f'{prefix}_theta_min',
            f'{prefix}_theta_max',
            f'{prefix}_theta_threshold_accuracy',
        ]
        if not set(required).issubset(selected.columns):
            return False
        if not (
            pd.to_numeric(selected[required[0]], errors='coerce')
            .eq(expected_unique_count)
            .all()
            and np.isclose(
                pd.to_numeric(selected[required[1]], errors='coerce'),
                expected_min,
            ).all()
            and np.isclose(
                pd.to_numeric(selected[required[2]], errors='coerce'),
                expected_max,
            ).all()
            and np.isclose(
                pd.to_numeric(selected[required[3]], errors='coerce'),
                1.0,
            ).all()
        ):
            return False
    return True


def _continuous_theta_schema_check(
    frame: pd.DataFrame,
    variant: str,
) -> bool:
    selected = frame[frame['module_variant'].astype(str) == variant]
    if len(selected) != 3:
        return False
    for prefix in ('train', 'evaluation'):
        required = [
            f'{prefix}_theta_unique_fraction',
            f'{prefix}_theta_busy_max',
            f'{prefix}_theta_idle_min',
            f'{prefix}_theta_threshold_accuracy',
        ]
        if not set(required).issubset(selected.columns):
            return False
        if not (
            pd.to_numeric(selected[required[0]], errors='coerce')
            .ge(0.95)
            .all()
            and pd.to_numeric(selected[required[1]], errors='coerce')
            .le(0.45)
            .all()
            and pd.to_numeric(selected[required[2]], errors='coerce')
            .ge(0.55)
            .all()
            and np.isclose(
                pd.to_numeric(selected[required[3]], errors='coerce'),
                1.0,
            ).all()
        ):
            return False
    return True


def _evaluate_prediction_time_alignment_degradation(
    final: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame]:
    gate_conditions = config.get("gate_conditions", {})
    oracle_name = str(gate_conditions.get("oracle", "theta_oracle"))
    degradation_conditions: list[dict[str, Any]] = list(
        gate_conditions.get("degradation_conditions", [])
    )
    neutral_name = str(gate_conditions.get("neutral", "theta_neutral"))
    if len(degradation_conditions) < 2:
        raise ValueError(
            "time_alignment_degradation gate requires at least 2 degradation conditions"
        )
    all_condition_names = [oracle_name] + [
        str(cond["name"]) for cond in degradation_conditions
    ] + [neutral_name]
    paired = _paired(final, oracle_name, degradation_conditions[0]["name"])
    checks: dict[str, bool] = {
        "formal_run": (
            not bool(
                final.get("diagnostic_run", pd.Series([True]))
                .astype(bool)
                .any()
            )
            and set(final.get("run_scope", pd.Series(dtype=str)).astype(str))
            == {"formal"}
        ),
    }
    values: dict[str, Any] = {
        "formal_run": checks["formal_run"],
    }
    requirements: dict[str, str] = {
        "formal_run": "not --dry-run or --quick; run_scope=formal",
    }

    # Marginal distribution: all conditions share the same theta multiset
    checks["marginal_distribution"] = _variants_share_audit_values(
        final,
        all_condition_names,
        ["train_theta_marginal_hash", "evaluation_theta_marginal_hash"],
    )
    values["marginal_distribution"] = checks["marginal_distribution"]
    requirements["marginal_distribution"] = (
        "identical train/evaluation theta marginal hash across all conditions"
    )

    # Paired seeds
    checks["paired_seeds"] = len(paired) == 3
    values["paired_seeds"] = len(paired)
    requirements["paired_seeds"] = "exactly 3 paired seeds"

    # Compute goodput means across conditions for monotonicity check
    condition_goodputs: list[tuple[str, float, float]] = []
    for cond_name in all_condition_names:
        data = final[final["module_variant"].astype(str) == cond_name]
        if data.empty:
            continue
        goodput = pd.to_numeric(data["served_goodput_mbps"], errors="coerce")
        violation = pd.to_numeric(
            data.get("pu_access_violation_rate", pd.Series([0.0])),
            errors="coerce",
        )
        condition_goodputs.append(
            (cond_name, float(goodput.mean()), float(violation.mean()))
        )

    # Oracle should have highest goodput
    oracle_goodput = _mean(
        final[final["module_variant"].astype(str) == oracle_name].get(
            "served_goodput_mbps"
        )
    )
    degraded_goodput_max = max(
        _mean(
            final[
                final["module_variant"].astype(str) == str(cond["name"])
            ].get("served_goodput_mbps")
        )
        for cond in degradation_conditions
    )
    checks["oracle_goodput_superior"] = (
        oracle_goodput > degraded_goodput_max
        and all(
            _all_greater(
                _paired(final, oracle_name, str(cond["name"])),
                oracle_name,
                str(cond["name"]),
                "served_goodput_mbps",
            )
            for cond in degradation_conditions
        )
    )
    values["oracle_goodput_superior"] = (
        f"oracle={oracle_goodput:.4f}, best_degraded={degraded_goodput_max:.4f}"
    )
    requirements["oracle_goodput_superior"] = (
        "Oracle goodput > every degradation-condition goodput in all 3/3 seeds"
    )

    # Monotonic decrease: goodput decreases as shuffle fraction increases
    fractions = [0.0] + [float(cond["fraction"]) for cond in degradation_conditions]
    # Build goodput list: oracle (0%) + degradation conditions
    goodput_by_fraction: list[tuple[float, float]] = []
    # Oracle = fraction 0
    goodput_by_fraction.append((0.0, oracle_goodput))
    for cond in degradation_conditions:
        frac = float(cond["fraction"])
        gp = _mean(
            final[
                final["module_variant"].astype(str) == str(cond["name"])
            ].get("served_goodput_mbps")
        )
        goodput_by_fraction.append((frac, gp))

    goodput_by_fraction.sort(key=lambda x: x[0])
    sorted_goodputs = [gp for _, gp in goodput_by_fraction]
    # Check monotonic decrease: all adjacent pairs should decrease
    monotonic_checks = [
        sorted_goodputs[i] > sorted_goodputs[i + 1]
        for i in range(len(sorted_goodputs) - 1)
    ]
    checks["monotonic_goodput_decrease"] = all(monotonic_checks)
    values["monotonic_goodput_decrease"] = (
        f"goodput sequence: "
        + " → ".join(f"{gp:.4f}" for gp in sorted_goodputs)
        + f"; all_steps_decrease={all(monotonic_checks)}"
    )
    requirements["monotonic_goodput_decrease"] = (
        "goodput monotonically decreases as shuffle fraction increases (0%->25%->...->100%)"
    )

    # 100% shuffled should be close to neutral
    p100_name = str(degradation_conditions[-1]["name"])
    p100_goodput = _mean(
        final[final["module_variant"].astype(str) == p100_name].get(
            "served_goodput_mbps"
        )
    )
    neutral_goodput = _mean(
        final[final["module_variant"].astype(str) == neutral_name].get(
            "served_goodput_mbps"
        )
    )
    neutral_diff = (
        abs(p100_goodput - neutral_goodput) / max(neutral_goodput, 1e-9)
        if neutral_goodput > 0
        else 0.0
    )
    checks["p100_near_neutral"] = neutral_diff <= 0.05
    values["p100_near_neutral"] = (
        f"p100={p100_goodput:.4f}, neutral={neutral_goodput:.4f}, "
        f"relative_diff={neutral_diff:.4f}"
    )
    requirements["p100_near_neutral"] = (
        "100% shuffled goodput relative difference from neutral <= 5%"
    )

    # Oracle PU violation should be zero
    oracle_violation = _mean(
        final[final["module_variant"].astype(str) == oracle_name].get(
            "pu_access_violation_rate"
        )
    )
    checks["oracle_violation_zero"] = oracle_violation <= 1e-3
    values["oracle_violation_zero"] = f"{oracle_violation:.6f}"
    requirements["oracle_violation_zero"] = "Oracle PU access violation <= 1e-3"

    # Physical trace and mask paired
    checks["paired_physical_trace_and_mask"] = _paired_structure(final)
    values["paired_physical_trace_and_mask"] = checks[
        "paired_physical_trace_and_mask"
    ]
    requirements["paired_physical_trace_and_mask"] = (
        "identical within each paired seed"
    )

    return _gate_payload(checks, values, requirements)


def _evaluate_prediction_staleness_degradation(
    final: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame]:
    gate_conditions = config.get("gate_conditions", {})
    oracle_name = str(gate_conditions.get("oracle", "theta_oracle"))
    degradation_conditions: list[dict[str, Any]] = list(
        gate_conditions.get("degradation_conditions", [])
    )
    neutral_name = str(gate_conditions.get("neutral", "theta_neutral"))
    if len(degradation_conditions) < 2:
        raise ValueError(
            "staleness_degradation gate requires at least 2 degradation conditions"
        )
    all_condition_names = [oracle_name] + [
        str(cond["name"]) for cond in degradation_conditions
    ] + [neutral_name]
    paired = _paired(final, oracle_name, degradation_conditions[0]["name"])

    checks: dict[str, bool] = {
        "formal_run": (
            not bool(
                final.get("diagnostic_run", pd.Series([True]))
                .astype(bool)
                .any()
            )
            and set(final.get("run_scope", pd.Series(dtype=str)).astype(str))
            == {"formal"}
        ),
    }
    values: dict[str, Any] = {"formal_run": checks["formal_run"]}
    requirements: dict[str, str] = {
        "formal_run": "not --dry-run or --quick; run_scope=formal",
    }

    checks["paired_seeds"] = len(paired) == 3
    values["paired_seeds"] = len(paired)
    requirements["paired_seeds"] = "exactly 3 paired seeds"

    # Build goodput sequence: oracle (staleness=0) + degradation conditions
    goodput_by_staleness: list[tuple[int, float]] = []
    oracle_data = final[final["module_variant"].astype(str) == oracle_name]
    oracle_goodput = _mean(oracle_data.get("served_goodput_mbps"))
    goodput_by_staleness.append((0, oracle_goodput))
    for cond in degradation_conditions:
        staleness = int(cond.get("staleness_slots", 0))
        gp = _mean(
            final[
                final["module_variant"].astype(str) == str(cond["name"])
            ].get("served_goodput_mbps")
        )
        goodput_by_staleness.append((staleness, gp))

    goodput_by_staleness.sort(key=lambda x: x[0])
    stalenesses = [s for s, _ in goodput_by_staleness]
    goodputs = [gp for _, gp in goodput_by_staleness]

    # Oracle should have highest goodput
    checks["oracle_goodput_superior"] = all(
        oracle_goodput > gp for _, gp in goodput_by_staleness[1:]
    ) and all(
        _all_greater(
            _paired(final, oracle_name, str(cond["name"])),
            oracle_name,
            str(cond["name"]),
            "served_goodput_mbps",
        )
        for cond in degradation_conditions
    )
    values["oracle_goodput_superior"] = (
        f"oracle={oracle_goodput:.4f}, "
        + ", ".join(
            f"s{s}={gp:.4f}" for s, gp in goodput_by_staleness[1:]
        )
    )
    requirements["oracle_goodput_superior"] = (
        "Oracle goodput > every stale condition in all 3/3 seeds"
    )

    # Monotonic decrease as staleness increases
    monotonic_checks = [
        goodputs[i] > goodputs[i + 1]
        for i in range(len(goodputs) - 1)
    ]
    checks["monotonic_goodput_decrease"] = all(monotonic_checks)
    values["monotonic_goodput_decrease"] = (
        "goodput: "
        + " → ".join(
            f"s{s}={gp:.4f}" for s, gp in goodput_by_staleness
        )
        + f"; all_decrease={all(monotonic_checks)}"
    )
    requirements["monotonic_goodput_decrease"] = (
        "goodput monotonically decreases as staleness increases (0→100→...→5000 slots)"
    )

    # Max staleness should be close to neutral
    max_stale_name = str(degradation_conditions[-1]["name"])
    max_stale_goodput = _mean(
        final[
            final["module_variant"].astype(str) == max_stale_name
        ].get("served_goodput_mbps")
    )
    neutral_goodput = _mean(
        final[
            final["module_variant"].astype(str) == neutral_name
        ].get("served_goodput_mbps")
    )
    neutral_diff = (
        abs(max_stale_goodput - neutral_goodput)
        / max(neutral_goodput, 1e-9)
        if neutral_goodput > 0
        else 0.0
    )
    max_slots = int(degradation_conditions[-1].get("staleness_slots", 5000))
    checks["max_stale_near_neutral"] = neutral_diff <= 0.05
    values["max_stale_near_neutral"] = (
        f"s{max_slots}={max_stale_goodput:.4f}, neutral={neutral_goodput:.4f}, "
        f"diff={neutral_diff:.4f}"
    )
    requirements["max_stale_near_neutral"] = (
        f"staleness={max_slots} goodput relative diff from neutral <= 5%"
    )

    # Oracle violation should be zero
    oracle_violation = _mean(oracle_data.get("pu_access_violation_rate"))
    checks["oracle_violation_zero"] = oracle_violation <= 1e-3
    values["oracle_violation_zero"] = f"{oracle_violation:.6f}"
    requirements["oracle_violation_zero"] = "Oracle PU access violation <= 1e-3"

    # Physical trace and mask
    checks["paired_physical_trace_and_mask"] = _paired_structure(final)
    values["paired_physical_trace_and_mask"] = checks[
        "paired_physical_trace_and_mask"
    ]
    requirements["paired_physical_trace_and_mask"] = (
        "identical within each paired seed"
    )

    return _gate_payload(checks, values, requirements)


def _evaluate_prediction_causal_teacher(
    final: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame]:
    names = config.get("gate_conditions", {})
    causal_name = str(names.get("causal", "theta_causal"))
    primary_name = str(names.get("primary_placebo", "theta_causal_placebo_a"))
    replicate_name = str(names.get("replicate_placebo", "theta_causal_placebo_b"))
    neutral_name = str(names.get("neutral", "theta_neutral"))
    audit_names = [causal_name, primary_name, replicate_name]
    paired = _paired(final, causal_name, primary_name)
    placebo_paired = _paired(final, primary_name, replicate_name)
    causal = final[final["module_variant"].astype(str) == causal_name]
    primary = final[final["module_variant"].astype(str) == primary_name]
    replicate = final[final["module_variant"].astype(str) == replicate_name]
    neutral = final[final["module_variant"].astype(str) == neutral_name]

    checks: dict[str, bool] = {
        "formal_run": (
            not bool(
                final.get("diagnostic_run", pd.Series([True]))
                .astype(bool).any()
            )
            and set(final.get("run_scope", pd.Series(dtype=str)).astype(str))
            == {"formal"}
        ),
    }
    values: dict[str, Any] = {"formal_run": checks["formal_run"]}
    requirements: dict[str, str] = {
        "formal_run": "not --dry-run or --quick; run_scope=formal",
    }

    checks["paired_seeds"] = len(paired) == 3 and len(placebo_paired) == 3
    values["paired_seeds"] = f"{len(paired)}/{len(placebo_paired)}"
    requirements["paired_seeds"] = "exactly 3 paired seeds"

    # Marginal distribution match
    checks["marginal_distribution"] = _variants_share_audit_values(
        final,
        audit_names,
        ["train_theta_marginal_hash", "evaluation_theta_marginal_hash"],
    )
    values["marginal_distribution"] = checks["marginal_distribution"]
    requirements["marginal_distribution"] = (
        "identical train/evaluation theta marginal hash across causal + placebos"
    )

    # Placebo equivalence
    placebo_goodput_diff = _relative_group_difference(
        primary, replicate, "served_goodput_mbps"
    )
    placebo_violation_diff = abs(
        _mean(primary.get("pu_access_violation_rate"))
        - _mean(replicate.get("pu_access_violation_rate"))
    )
    checks["placebo_equivalence"] = (
        abs(placebo_goodput_diff) <= 0.02 and placebo_violation_diff <= 0.01
    )
    values["placebo_equivalence"] = (
        f"goodput_diff={placebo_goodput_diff:.4f}, "
        f"violation_diff={placebo_violation_diff:.4f}"
    )
    requirements["placebo_equivalence"] = (
        "|goodput| <= 2%, violation difference <= 0.01"
    )

    # Causal goodput > placebo in all 3/3 seeds
    causal_goodput_effect = _relative_delta(paired, "served_goodput_mbps")
    causal_mean_goodput = _mean(causal.get("served_goodput_mbps"))
    placebo_mean_goodput = _mean(primary.get("served_goodput_mbps"))
    checks["causal_goodput_effect"] = (
        _all_greater(
            paired,
            causal_name,
            primary_name,
            "served_goodput_mbps",
        )
        and _mean(causal_goodput_effect) >= 0.01
    )
    values["causal_goodput_effect"] = (
        f"causal={causal_mean_goodput:.4f}, placebo={placebo_mean_goodput:.4f}, "
        f"gain={_mean(causal_goodput_effect):.4f}"
    )
    requirements["causal_goodput_effect"] = (
        "causal goodput > placebo in 3/3 seeds; mean gain >= 1%"
    )

    # Causal goodput > neutral
    causal_neutral_effect = _relative_delta(
        _paired(final, causal_name, neutral_name),
        "served_goodput_mbps",
    )
    neutral_mean_goodput = _mean(neutral.get("served_goodput_mbps"))
    checks["causal_beats_neutral"] = (
        _all_greater(
            _paired(final, causal_name, neutral_name),
            causal_name,
            neutral_name,
            "served_goodput_mbps",
        )
        and _mean(causal_neutral_effect) >= 0.01
    )
    values["causal_beats_neutral"] = (
        f"causal={causal_mean_goodput:.4f}, neutral={neutral_mean_goodput:.4f}, "
        f"gain={_mean(causal_neutral_effect):.4f}"
    )
    requirements["causal_beats_neutral"] = (
        "causal goodput > neutral in 3/3 seeds; mean gain >= 1%"
    )

    # Causal PU violation: must be acceptable (not zero like oracle, but bounded)
    causal_violation = _mean(causal.get("pu_access_violation_rate"))
    checks["causal_violation_bounded"] = causal_violation <= 0.05
    values["causal_violation_bounded"] = f"{causal_violation:.4f}"
    requirements["causal_violation_bounded"] = (
        "causal PU access violation <= 5%"
    )

    # Physical trace and mask
    checks["paired_physical_trace_and_mask"] = _paired_structure(final)
    values["paired_physical_trace_and_mask"] = checks[
        "paired_physical_trace_and_mask"
    ]
    requirements["paired_physical_trace_and_mask"] = (
        "identical within each paired seed"
    )

    return _gate_payload(checks, values, requirements)


def _evaluate_prediction_frozen_predictor(
    final: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame]:
    names = config.get("gate_conditions", {})
    oracle_name = str(names.get("oracle", "theta_oracle"))
    causal_name = str(names.get("causal", "theta_causal"))
    frozen_name = str(names.get("frozen", "theta_frozen"))
    frozen_placebo_name = str(names.get("frozen_placebo", "theta_frozen_placebo"))
    neutral_name = str(names.get("neutral", "theta_neutral"))

    oracle = final[final["module_variant"].astype(str) == oracle_name]
    causal = final[final["module_variant"].astype(str) == causal_name]
    frozen = final[final["module_variant"].astype(str) == frozen_name]
    frozen_placebo = final[final["module_variant"].astype(str) == frozen_placebo_name]
    neutral = final[final["module_variant"].astype(str) == neutral_name]
    paired_frozen = _paired(final, frozen_name, frozen_placebo_name)

    checks: dict[str, bool] = {
        "formal_run": (
            not bool(
                final.get("diagnostic_run", pd.Series([True]))
                .astype(bool).any()
            )
            and set(final.get("run_scope", pd.Series(dtype=str)).astype(str))
            == {"formal"}
        ),
    }
    values: dict[str, Any] = {"formal_run": checks["formal_run"]}
    requirements: dict[str, str] = {
        "formal_run": "not --dry-run or --quick; run_scope=formal",
    }

    checks["paired_seeds"] = len(paired_frozen) == 3
    values["paired_seeds"] = len(paired_frozen)
    requirements["paired_seeds"] = "exactly 3 paired seeds"

    # Extract goodputs
    oracle_gp = _mean(oracle.get("served_goodput_mbps"))
    causal_gp = _mean(causal.get("served_goodput_mbps"))
    frozen_gp = _mean(frozen.get("served_goodput_mbps"))
    frozen_pl_gp = _mean(frozen_placebo.get("served_goodput_mbps"))
    neutral_gp = _mean(neutral.get("served_goodput_mbps"))

    # Frozen > Placebo (time-alignment value of frozen model)
    frozen_vs_placebo = _relative_delta(paired_frozen, "served_goodput_mbps")
    checks["frozen_beats_placebo"] = (
        _all_greater(
            paired_frozen,
            frozen_name,
            frozen_placebo_name,
            "served_goodput_mbps",
        )
        and _mean(frozen_vs_placebo) >= 0.02
    )
    values["frozen_beats_placebo"] = (
        f"frozen={frozen_gp:.4f}, placebo={frozen_pl_gp:.4f}, "
        f"gain={_mean(frozen_vs_placebo):.4f}"
    )
    requirements["frozen_beats_placebo"] = (
        "frozen goodput > placebo in 3/3 seeds; mean gain >= 2%"
    )

    # Frozen/Markov ratio is diagnostic only; neither predictor is ordered a priori.
    frozen_causal_ratio = frozen_gp / max(causal_gp, 1e-9)
    values["frozen_noninferior_to_causal"] = (
        f"frozen={frozen_gp:.4f}, causal={causal_gp:.4f}, "
        f"ratio={frozen_causal_ratio:.4f}"
    )
    requirements["frozen_noninferior_to_causal"] = "informational only"

    checks["frozen_not_below_neutral"] = frozen_gp >= neutral_gp
    values["frozen_not_below_neutral"] = f"frozen={frozen_gp:.4f}, neutral={neutral_gp:.4f}"
    requirements["frozen_not_below_neutral"] = "mean frozen goodput >= mean neutral goodput"
    values["predictor_ordering"] = f"oracle={oracle_gp:.4f}, markov={causal_gp:.4f}, frozen={frozen_gp:.4f}"
    requirements["predictor_ordering"] = "informational only; no Markov/Frozen ordering"

    # Frozen violation bounded
    frozen_violation = _mean(frozen.get("pu_access_violation_rate"))
    checks["frozen_violation_bounded"] = frozen_violation <= 0.05
    values["frozen_violation_bounded"] = f"{frozen_violation:.4f}"
    requirements["frozen_violation_bounded"] = (
        "frozen PU access violation <= 5%"
    )

    # Structure and central-model provenance
    checks["paired_physical_trace_and_mask"] = _paired_structure(final)
    values["paired_physical_trace_and_mask"] = checks[
        "paired_physical_trace_and_mask"
    ]
    requirements["paired_physical_trace_and_mask"] = (
        "identical within each paired seed"
    )

    frozen_sources = set(frozen.get("prediction_input_source", pd.Series(dtype=str)).astype(str))
    frozen_hashes = set(frozen.get("prediction_model_hash", pd.Series(dtype=str)).astype(str))
    placebo_hashes = set(frozen_placebo.get("prediction_model_hash", pd.Series(dtype=str)).astype(str))
    checks["central_predictor_provenance"] = (
        frozen_sources == {"bs_true_idle_history"}
        and len(frozen_hashes) == 1
        and frozen_hashes == placebo_hashes
        and all(len(value) == 64 for value in frozen_hashes)
    )
    values["central_predictor_provenance"] = f"sources={sorted(frozen_sources)}, hashes={sorted(frozen_hashes)}"
    requirements["central_predictor_provenance"] = "BS true-idle input; one full SHA256 shared by frozen and placebo"

    return _gate_payload(checks, values, requirements)


def _evaluate_prediction_sensing_redundancy(
    final: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame]:
    names = config.get('gate_conditions', {})
    full_name = str(names.get('full', 'full'))
    sensing_name = str(names.get('sensing_only', 'sensing_only'))
    prediction_name = str(names.get('prediction_only', 'prediction_only'))
    neither_name = str(names.get('neither', 'neither'))
    variants = [full_name, sensing_name, prediction_name, neither_name]
    pred_pair = _paired(final, full_name, sensing_name)
    sense_pair = _paired(final, full_name, prediction_name)

    pred_gain = _relative_delta(pred_pair, 'served_goodput_mbps', reference=full_name)
    pred_violation = (
        pred_pair[f'{full_name}:pu_access_violation_rate']
        - pred_pair[f'{sensing_name}:pu_access_violation_rate']
    )
    sensing_goodput = _relative_delta(
        sense_pair, 'served_goodput_mbps', reference=full_name
    )
    plr_improvement = (
        sense_pair[f'{prediction_name}:plr'] - sense_pair[f'{full_name}:plr']
    )
    violation_improvement = (
        sense_pair[f'{prediction_name}:pu_access_violation_rate']
        - sense_pair[f'{full_name}:pu_access_violation_rate']
    )

    means = final.groupby('module_variant')[
        ['served_goodput_mbps', 'plr', 'pu_access_violation_rate']
    ].mean()
    full_values = means.loc[full_name]
    dominated = False
    for other in (sensing_name, prediction_name):
        candidate = means.loc[other]
        weak = (
            candidate['served_goodput_mbps'] >= full_values['served_goodput_mbps']
            and candidate['plr'] <= full_values['plr']
            and candidate['pu_access_violation_rate']
            <= full_values['pu_access_violation_rate']
        )
        strict = (
            candidate['served_goodput_mbps'] > full_values['served_goodput_mbps']
            or candidate['plr'] < full_values['plr']
            or candidate['pu_access_violation_rate']
            < full_values['pu_access_violation_rate']
        )
        dominated = dominated or (weak and strict)

    expected_sources = {
        full_name: ('real', 'real'),
        sensing_name: ('real', 'neutral'),
        prediction_name: ('neutral', 'real'),
        neither_name: ('neutral', 'neutral'),
    }
    source_ok = True
    for variant, (sensing_source, prediction_source) in expected_sources.items():
        rows = final[final['module_variant'].astype(str) == variant]
        source_ok = source_ok and (
            set(rows['sensing_info_source'].astype(str)) == {sensing_source}
            and set(rows['prediction_info_source'].astype(str)) == {prediction_source}
        )

    checks = {
        'formal_run': (
            not final.get('diagnostic_run', pd.Series([True])).astype(bool).any()
            and set(final.get('run_scope', pd.Series(dtype=str)).astype(str)) == {'formal'}
        ),
        'paired_seeds': len(pred_pair) == 3 and len(sense_pair) == 3,
        'prediction_conditional_value': (
            _all_greater(pred_pair, full_name, sensing_name, 'served_goodput_mbps')
            and _mean(pred_gain) >= 0.02
            and _mean(pred_violation) <= 0.01
        ),
        'sensing_conditional_value': (
            (
                (_all_less(sense_pair, full_name, prediction_name, 'plr') and _mean(plr_improvement) >= 0.01)
                or (_all_less(sense_pair, full_name, prediction_name, 'pu_access_violation_rate') and _mean(violation_improvement) >= 0.01)
            )
            and _mean(sensing_goodput) >= -0.02
        ),
        'full_not_pareto_dominated': not dominated,
        'paired_physical_trace_and_mask': (
            _paired_structure(final)
            and _variants_share_audit_values(final, variants, ['raw_action_mask_hash'])
        ),
        'factor_sources_and_dimension': (
            source_ok
            and final['observation_dimension'].astype(int).nunique() == 1
        ),
        'evaluation_parameters_frozen': bool(
            final['evaluation_parameters_frozen'].astype(bool).all()
        ),
    }
    values = {
        'formal_run': checks['formal_run'],
        'paired_seeds': f'prediction={len(pred_pair)}, sensing={len(sense_pair)}',
        'prediction_conditional_value': f'mean_goodput_gain={_mean(pred_gain):.4f}, violation_change={_mean(pred_violation):.4f}',
        'sensing_conditional_value': f'plr_improvement={_mean(plr_improvement):.4f}, violation_improvement={_mean(violation_improvement):.4f}, goodput_gain={_mean(sensing_goodput):.4f}',
        'full_not_pareto_dominated': not dominated,
        'paired_physical_trace_and_mask': checks['paired_physical_trace_and_mask'],
        'factor_sources_and_dimension': source_ok,
        'evaluation_parameters_frozen': checks['evaluation_parameters_frozen'],
    }
    requirements = {
        'formal_run': 'not quick/dry; run_scope=formal',
        'paired_seeds': 'exactly 3 paired seeds for both conditional effects',
        'prediction_conditional_value': 'full>sensing-only 3/3; mean goodput gain>=2%; violation worsening<=1pp',
        'sensing_conditional_value': 'PLR or violation improves 3/3 by >=1pp; mean goodput noninferior within 2%',
        'full_not_pareto_dominated': 'no single-module condition dominates full',
        'paired_physical_trace_and_mask': 'same physical traces and raw masks across four factors',
        'factor_sources_and_dimension': 'registered 2x2 sources; one observation dimension',
        'evaluation_parameters_frozen': 'all evaluation parameter hashes unchanged',
    }
    return _gate_payload(checks, values, requirements)


def _evaluate_dqi_gate(
    final: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame]:
    names = config.get('gate_conditions', {})
    oracle_name = str(names.get('oracle', 'dqi_future_oracle'))
    shuffled_name = str(names.get('shuffled', 'dqi_channel_shuffled'))
    neutral_name = str(names.get('neutral', 'dqi_neutral'))
    paired = _paired(final, oracle_name, shuffled_name)
    goodput_delta = _relative_delta(paired, 'served_goodput_mbps')
    regret_reduction = _relative_reduction(
        paired,
        'mean_channel_selection_regret',
    )
    goodput_direction = _all_greater(
        paired,
        oracle_name,
        shuffled_name,
        'served_goodput_mbps',
    )
    regret_direction = _all_less(
        paired,
        oracle_name,
        shuffled_name,
        'mean_channel_selection_regret',
    )
    throughput_path = goodput_direction and _mean(goodput_delta) >= 0.05
    ranking_path = bool(
        regret_direction
        and _mean(regret_reduction) >= 0.30
        and goodput_direction
        and _mean(goodput_delta) >= 0.02
    )
    oracle = final[final['module_variant'].astype(str) == oracle_name]
    shuffled = final[final['module_variant'].astype(str) == shuffled_name]
    neutral = final[final['module_variant'].astype(str) == neutral_name]
    shuffled_neutral_goodput = _relative_group_difference(
        shuffled,
        neutral,
        'served_goodput_mbps',
    )
    shuffled_neutral_regret = _relative_group_difference(
        shuffled,
        neutral,
        'mean_channel_selection_regret',
    )
    safety_difference = abs(
        _mean(oracle.get('pu_access_violation_rate'))
        - _mean(neutral.get('pu_access_violation_rate'))
    )
    common = {
        'paired_seeds': len(paired) == 3,
        'effect_path': throughput_path or ranking_path,
        'pu_safety_equivalence': safety_difference <= 1e-3,
        'top1_diversity': (
            not oracle.empty
            and int(oracle['oracle_top1_channel_count'].min()) >= 2
        ),
        'shuffled_neutral_equivalence': (
            abs(shuffled_neutral_goodput) <= 0.02
            and abs(shuffled_neutral_regret) <= 0.10
        ),
        'dwell_invariant': (
            not final.empty
            and int(final['dwell_violation_count'].sum()) == 0
        ),
        'dqi_action_sensitivity': (
            not oracle.empty
            and float(oracle['dqi_action_disagreement_rate'].mean()) > 0.0
        ),
        'paired_physical_trace_and_mask': _paired_structure(final),
    }
    values = {
        'paired_seeds': len(paired),
        'effect_path': f'throughput={throughput_path}, ranking={ranking_path}',
        'pu_safety_equivalence': safety_difference,
        'top1_diversity': int(oracle['oracle_top1_channel_count'].min()) if not oracle.empty else 0,
        'shuffled_neutral_equivalence': f'goodput={shuffled_neutral_goodput:.6f}, regret={shuffled_neutral_regret:.6f}',
        'dwell_invariant': int(final['dwell_violation_count'].sum()) if not final.empty else -1,
        'dqi_action_sensitivity': _mean(oracle.get('dqi_action_disagreement_rate')),
        'paired_physical_trace_and_mask': common['paired_physical_trace_and_mask'],
    }
    requirements = {
        'paired_seeds': 'exactly 3 paired seeds',
        'effect_path': 'goodput +5% or regret -30% with goodput +2%',
        'pu_safety_equivalence': 'absolute difference <= 1e-3',
        'top1_diversity': 'at least 2 channels per oracle evaluation row',
        'shuffled_neutral_equivalence': '|goodput| <= 2%, |regret| <= 10%',
        'dwell_invariant': 'zero violations',
        'dqi_action_sensitivity': '> 0',
        'paired_physical_trace_and_mask': 'identical within each seed',
    }
    return _gate_payload(common, values, requirements)


def _evaluate_dqi_positive_control(
    final: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame]:
    names = config.get('gate_conditions', {})
    oracle_name = str(names.get('oracle', 'dqi_future_oracle'))
    shuffled_name = str(names.get('shuffled', 'dqi_channel_shuffled'))
    neutral_name = str(names.get('neutral', 'dqi_neutral'))
    paired = _paired(final, oracle_name, shuffled_name)
    oracle = final[final['module_variant'].astype(str) == oracle_name]
    shuffled = final[final['module_variant'].astype(str) == shuffled_name]
    neutral = final[final['module_variant'].astype(str) == neutral_name]
    goodput_delta = _relative_delta(paired, 'served_goodput_mbps')
    regret_reduction = _relative_reduction(
        paired,
        'mean_channel_selection_regret',
    )
    shuffled_neutral_goodput = _relative_group_difference(
        shuffled,
        neutral,
        'served_goodput_mbps',
    )
    shuffled_neutral_regret = _relative_group_difference(
        shuffled,
        neutral,
        'mean_channel_selection_regret',
    )
    checks = {
        'formal_run': (
            not bool(final.get('diagnostic_run', pd.Series([True])).astype(bool).any())
            and set(final.get('run_scope', pd.Series(dtype=str)).astype(str)) == {'formal'}
        ),
        'paired_seeds': len(paired) == 3,
        'channel_regret_effect': (
            _all_less(
                paired,
                oracle_name,
                shuffled_name,
                'mean_channel_selection_regret',
            )
            and _mean(regret_reduction) >= 0.30
        ),
        'goodput_effect': (
            _all_greater(
                paired,
                oracle_name,
                shuffled_name,
                'served_goodput_mbps',
            )
            and _mean(goodput_delta) >= 0.05
        ),
        'dqi_action_sensitivity': (
            not oracle.empty
            and _mean(oracle.get('dqi_action_disagreement_rate')) >= 0.10
        ),
        'top1_diversity': (
            not oracle.empty
            and int(oracle['oracle_top1_channel_count'].min()) >= 3
        ),
        'dwell_invariant': (
            not final.empty
            and int(final['dwell_violation_count'].sum()) == 0
        ),
        'placebo_equivalence': (
            abs(shuffled_neutral_goodput) <= 0.02
            and abs(shuffled_neutral_regret) <= 0.10
        ),
        'paired_physical_trace_and_mask': _paired_structure(final),
    }
    values = {
        'formal_run': not bool(
            final.get('diagnostic_run', pd.Series([True])).astype(bool).any()
        ),
        'paired_seeds': len(paired),
        'channel_regret_effect': _mean(regret_reduction),
        'goodput_effect': _mean(goodput_delta),
        'dqi_action_sensitivity': _mean(
            oracle.get('dqi_action_disagreement_rate')
        ),
        'top1_diversity': (
            int(oracle['oracle_top1_channel_count'].min())
            if not oracle.empty
            else 0
        ),
        'dwell_invariant': (
            int(final['dwell_violation_count'].sum()) if not final.empty else -1
        ),
        'placebo_equivalence': (
            f'goodput={shuffled_neutral_goodput:.6f}, '
            f'regret={shuffled_neutral_regret:.6f}'
        ),
        'paired_physical_trace_and_mask': checks[
            'paired_physical_trace_and_mask'
        ],
    }
    requirements = {
        'formal_run': 'not --dry-run or --quick; run_scope=formal',
        'paired_seeds': 'exactly 3 paired seeds',
        'channel_regret_effect': '3/3 lower, mean reduction >= 30%',
        'goodput_effect': '3/3 higher, mean relative gain >= 5%',
        'dqi_action_sensitivity': 'mean action disagreement >= 10%',
        'top1_diversity': 'at least 3 channels in every oracle evaluation row',
        'dwell_invariant': 'zero violations',
        'placebo_equivalence': '|goodput| <= 2%, |regret| <= 10%',
        'paired_physical_trace_and_mask': 'identical within each paired seed',
    }
    return _gate_payload(checks, values, requirements)


def _gate_payload(
    checks: dict[str, bool],
    values: dict[str, Any],
    requirements: dict[str, str],
) -> tuple[dict[str, Any], pd.DataFrame]:
    table = pd.DataFrame(
        [
            {
                'check': name,
                'passed': bool(passed),
                'value': values.get(name),
                'requirement': requirements.get(name, ''),
            }
            for name, passed in checks.items()
        ]
    )
    payload = {
        'passed': bool(all(checks.values())),
        'checks': json.loads(table.to_json(orient='records')),
    }
    return payload, table


def _load_calibration_config(config: dict[str, Any]) -> dict[str, Any]:
    calibration = config.get('load_calibration', {})
    names = {
        str(calibration['neutral_condition']),
        str(calibration['oracle_condition']),
    }
    conditions = [
        copy.deepcopy(condition)
        for condition in config.get('module_conditions', [])
        if str(condition.get('name')) in names
    ]
    if len(conditions) != 2:
        raise ValueError('load calibration requires exactly neutral and oracle conditions')
    output = copy.deepcopy(config)
    output['module_conditions'] = conditions
    output['arrival_probabilities'] = list(
        calibration.get('arrival_probabilities', [0.20, 0.35, 0.50, 0.65])
    )
    output['seeds'] = [int(calibration.get('seed', 9_001))]
    output['slots'] = int(calibration.get('slots', 20_000))
    output['quick_slots'] = min(
        int(output.get('quick_slots', 500)),
        output['slots'],
    )
    output['checkpoint_evaluation'] = {
        **output.get('checkpoint_evaluation', {}),
        **calibration.get('checkpoint_evaluation', {}),
    }
    output['_run_scope'] = 'load_calibration'
    output['power_calibration'] = {'enabled': False}
    return output


def _paired(
    frame: pd.DataFrame,
    first: str,
    second: str,
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    index = [
        column
        for column in ['scenario', 'arrival_probability', 'agent', 'seed']
        if column in frame
    ]
    values = [
        'served_goodput_mbps',
        'plr',
        'overlay_collision_rate',
        'pu_access_violation_rate',
        'mean_safe_action_regret_mbps',
        'overlay_given_busy_legal_backlog',
        'mean_channel_selection_regret',
    ]
    subset = frame[frame['module_variant'].astype(str).isin([first, second])]
    pivot = subset.pivot_table(
        index=index,
        columns='module_variant',
        values=[value for value in values if value in subset],
        aggfunc='mean',
    )
    pivot.columns = [f'{condition}:{metric}' for metric, condition in pivot.columns]
    required = [f'{name}:served_goodput_mbps' for name in (first, second)]
    if not all(column in pivot for column in required):
        return pd.DataFrame()
    return pivot.dropna(subset=required).reset_index()


def _relative_delta(
    frame: pd.DataFrame, metric: str, reference: str = "oracle"
) -> pd.Series:
    columns = [column for column in frame if column.endswith(f":{metric}")]
    if len(columns) != 2:
        return pd.Series(dtype=float)
    ref_col = next(
        (column for column in columns if reference in column),
        columns[0],
    )
    other = next(column for column in columns if column != ref_col)
    return (frame[ref_col] - frame[other]) / frame[other].abs().clip(lower=1e-12)


def _relative_reduction(
    frame: pd.DataFrame, metric: str, reference: str = "oracle"
) -> pd.Series:
    columns = [column for column in frame if column.endswith(f":{metric}")]
    if len(columns) != 2:
        return pd.Series(dtype=float)
    ref_col = next(
        (column for column in columns if reference in column),
        columns[0],
    )
    other = next(column for column in columns if column != ref_col)
    return (frame[other] - frame[ref_col]) / frame[other].abs().clip(lower=1e-12)


def _all_less(
    frame: pd.DataFrame,
    first: str,
    second: str,
    metric: str,
) -> bool:
    first_col = f'{first}:{metric}'
    second_col = f'{second}:{metric}'
    return bool(
        not frame.empty
        and first_col in frame
        and second_col in frame
        and (frame[first_col] < frame[second_col]).all()
    )


def _all_greater(
    frame: pd.DataFrame,
    first: str,
    second: str,
    metric: str,
) -> bool:
    first_col = f'{first}:{metric}'
    second_col = f'{second}:{metric}'
    return bool(
        not frame.empty
        and first_col in frame
        and second_col in frame
        and (frame[first_col] > frame[second_col]).all()
    )


def _paired_structure(frame: pd.DataFrame) -> bool:
    if frame.empty:
        return False
    columns = [
        column
        for column in [
            'train_physical_trace_hash',
            'evaluation_physical_trace_hash',
            'raw_action_mask_hash',
        ]
        if column in frame
    ]
    grouped = frame.groupby(
        [column for column in ['scenario', 'arrival_probability', 'agent', 'seed'] if column in frame]
    )
    return bool(
        columns
        and all(
            group[column].astype(str).nunique() == 1
            for _, group in grouped
            for column in columns
        )
    )


def _variants_share_audit_values(
    frame: pd.DataFrame,
    variants: list[str],
    columns: list[str],
) -> bool:
    if frame.empty or not set(columns).issubset(frame.columns):
        return False
    subset = frame[frame['module_variant'].astype(str).isin(variants)]
    group_columns = [
        column
        for column in ['scenario', 'arrival_probability', 'agent', 'seed']
        if column in subset
    ]
    if subset.empty or not group_columns:
        return False
    expected = set(variants)
    for _, group in subset.groupby(group_columns):
        if set(group['module_variant'].astype(str)) != expected:
            return False
        if any(group[column].astype(str).nunique() != 1 for column in columns):
            return False
    return True


def _variants_have_distinct_values(
    frame: pd.DataFrame,
    variants: list[str],
    column: str,
) -> bool:
    if frame.empty or column not in frame:
        return False
    subset = frame[frame['module_variant'].astype(str).isin(variants)]
    group_columns = [
        key
        for key in ['scenario', 'arrival_probability', 'agent', 'seed']
        if key in subset
    ]
    if subset.empty or not group_columns:
        return False
    expected = set(variants)
    for _, group in subset.groupby(group_columns):
        if set(group['module_variant'].astype(str)) != expected:
            return False
        if group[column].astype(str).nunique() != len(variants):
            return False
    return True


def _finite_abs_values(
    frame: pd.DataFrame,
    variants: list[str],
    columns: list[str],
) -> list[float]:
    if frame.empty or not set(columns).issubset(frame.columns):
        return []
    subset = frame[frame['module_variant'].astype(str).isin(variants)]
    values: list[float] = []
    for column in columns:
        numeric = pd.to_numeric(subset[column], errors='coerce').to_numpy()
        values.extend(np.abs(numeric[np.isfinite(numeric)]).tolist())
    return values


def _relative_group_difference(
    first: pd.DataFrame,
    second: pd.DataFrame,
    metric: str,
) -> float:
    denominator = abs(_mean(second.get(metric)))
    return (_mean(first.get(metric)) - _mean(second.get(metric))) / max(
        denominator,
        1e-12,
    )


def _mean(values: Any) -> float:
    if values is None:
        return float('nan')
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    return float(np.mean(array)) if array.size else float('nan')


def _final_rows(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return metrics.copy()
    if 'is_final_checkpoint' in metrics:
        return metrics[metrics['is_final_checkpoint'].astype(bool)].copy()
    return metrics.copy()


def _first_scenario_config(config: dict[str, Any]) -> dict[str, Any]:
    scenario = (config.get('scenarios') or [{}])[0]
    return {
        **copy.deepcopy(config),
        **{key: copy.deepcopy(value) for key, value in scenario.items() if key != 'name'},
    }


def _config_hash(config: dict[str, Any]) -> str:
    payload = json.dumps(
        _serializable_config(config),
        sort_keys=True,
        separators=(',', ':'),
    )
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()[:16]


def _serializable_config(config: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in config.items()
        if not key.startswith('_') and key != '_config_path'
    }


def _write_calibration_failure(analysis_dir: Path, reason: str) -> None:
    (analysis_dir / 'e2_r1_calibration_failure.json').write_text(
        json.dumps({'passed': False, 'reason': reason}, indent=2),
        encoding='utf-8',
    )
