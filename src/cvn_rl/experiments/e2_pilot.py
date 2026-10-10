from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


E2_SCENARIOS = ('M3_low_pu', 'M3_high_pu', 'M10_low_pu', 'M10_high_pu')
E2_VARIANTS = ('sensing_only', 'no_dqi', 'no_prediction', 'full')
E2_SEEDS = (1, 2, 3)
PRIMARY_METRICS = (
    'mean_reward',
    'system_vt_throughput',
    'plr',
    'overlay_collision_rate',
    'conditional_underlay_violation_rate',
    'contention_loss_rate',
    'jain_fairness',
)


def validate_e2_pilot_metrics(
    metrics: pd.DataFrame,
    *,
    slots: int = 30_000,
    checkpoint_interval: int = 1_000,
) -> tuple[pd.DataFrame, bool]:
    expected_checkpoints = tuple(range(checkpoint_interval, slots + 1, checkpoint_interval))
    expected_rows = (
        len(E2_SCENARIOS)
        * len(E2_VARIANTS)
        * len(E2_SEEDS)
        * len(expected_checkpoints)
    )
    expected_final = len(E2_SCENARIOS) * len(E2_VARIANTS) * len(E2_SEEDS)
    checks: list[dict[str, object]] = []

    def add(check: str, passed: bool, value: object, expected: object) -> None:
        checks.append(
            {
                'check': check,
                'passed': bool(passed),
                'value': value,
                'expected': expected,
            }
        )

    required = {
        'scenario',
        'seed',
        'agent',
        'module_variant',
        'checkpoint_slot',
        'is_final_checkpoint',
        'train_trace_hash',
        'evaluation_trace_hash',
        'parameter_hash_before_eval',
        'parameter_hash_after_eval',
        'observation_dimension',
        'state_feature_signature',
        *PRIMARY_METRICS,
    }
    missing = sorted(required.difference(metrics.columns))
    add('required_columns', not missing, '|'.join(missing) if missing else 'complete', 'complete')
    if missing:
        table = pd.DataFrame(checks)
        return table, False

    key_columns = [
        'scenario',
        'seed',
        'agent',
        'module_variant',
        'checkpoint_slot',
        'evaluation_variant',
    ]
    key_columns = [column for column in key_columns if column in metrics]
    duplicate_count = int(metrics.duplicated(key_columns, keep=False).sum())
    add('unique_checkpoint_key', duplicate_count == 0, duplicate_count, 0)
    add('row_count', len(metrics) == expected_rows, len(metrics), expected_rows)

    finals = metrics[metrics['is_final_checkpoint'].astype(bool)]
    add('final_row_count', len(finals) == expected_final, len(finals), expected_final)
    add(
        'scenario_coverage',
        set(metrics['scenario']) == set(E2_SCENARIOS),
        '|'.join(sorted(set(metrics['scenario']))),
        '|'.join(E2_SCENARIOS),
    )
    add(
        'seed_coverage',
        set(metrics['seed'].astype(int)) == set(E2_SEEDS),
        '|'.join(map(str, sorted(set(metrics['seed'].astype(int))))),
        '|'.join(map(str, E2_SEEDS)),
    )
    add(
        'variant_coverage',
        set(metrics['module_variant']) == set(E2_VARIANTS),
        '|'.join(sorted(set(metrics['module_variant']))),
        '|'.join(E2_VARIANTS),
    )
    add(
        'checkpoint_coverage',
        set(metrics['checkpoint_slot'].astype(int)) == set(expected_checkpoints),
        len(set(metrics['checkpoint_slot'].astype(int))),
        len(expected_checkpoints),
    )
    add(
        'agent_is_ddqn',
        set(metrics['agent']) == {'ddqn_no_esn'},
        '|'.join(sorted(set(metrics['agent']))),
        'ddqn_no_esn',
    )

    expected_combinations = pd.MultiIndex.from_product(
        [E2_SCENARIOS, E2_SEEDS, E2_VARIANTS, expected_checkpoints],
        names=['scenario', 'seed', 'module_variant', 'checkpoint_slot'],
    )
    actual_combinations = pd.MultiIndex.from_frame(
        metrics[['scenario', 'seed', 'module_variant', 'checkpoint_slot']].drop_duplicates()
    )
    missing_combinations = expected_combinations.difference(actual_combinations)
    add(
        'matrix_complete',
        len(missing_combinations) == 0,
        len(missing_combinations),
        0,
    )

    train_hash_counts = metrics.groupby(['scenario', 'seed'])['train_trace_hash'].nunique()
    add(
        'shared_training_trace',
        bool((train_hash_counts == 1).all()),
        int(train_hash_counts.max()),
        1,
    )
    eval_hash_counts = metrics.groupby(
        ['scenario', 'seed', 'checkpoint_slot']
    )['evaluation_trace_hash'].nunique()
    add(
        'shared_evaluation_trace',
        bool((eval_hash_counts == 1).all()),
        int(eval_hash_counts.max()),
        1,
    )
    frozen = (
        metrics['parameter_hash_before_eval'].astype(str)
        == metrics['parameter_hash_after_eval'].astype(str)
    )
    add('evaluation_parameters_frozen', bool(frozen.all()), int((~frozen).sum()), 0)

    dimension_counts = metrics.groupby(
        ['scenario', 'seed', 'checkpoint_slot']
    )['observation_dimension'].nunique()
    signature_counts = metrics.groupby(
        ['scenario', 'seed', 'checkpoint_slot']
    )['state_feature_signature'].nunique()
    schema_ok = bool((dimension_counts == 1).all() and (signature_counts == 1).all())
    add(
        'module_observation_schema_equal',
        schema_ok,
        max(int(dimension_counts.max()), int(signature_counts.max())),
        1,
    )
    numeric = metrics[list(PRIMARY_METRICS)].apply(pd.to_numeric, errors='coerce')
    nonfinite = int((~np.isfinite(numeric.to_numpy(float))).sum())
    add('primary_metrics_finite', nonfinite == 0, nonfinite, 0)

    final_slots = set(finals['checkpoint_slot'].astype(int))
    add('final_checkpoint_is_max_slots', final_slots == {slots}, '|'.join(map(str, final_slots)), slots)
    table = pd.DataFrame(checks)
    return table, bool(table['passed'].all())


def require_e2_pilot_integrity(metrics: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
    table, passed = validate_e2_pilot_metrics(metrics, **kwargs)
    if not passed:
        failed = table[~table['passed'].astype(bool)]
        raise ValueError(
            'E2 pilot integrity failed: '
            + '; '.join(f"{row.check}={row.value}" for row in failed.itertuples())
        )
    return table


def evaluate_e2_pilot_signal(
    metrics: pd.DataFrame,
    config: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, bool]:
    gate = (config or {}).get('pilot_signal_gate', {})
    agent = str(gate.get('agent', 'ddqn_no_esn'))
    hard_scenario = str(gate.get('hard_scenario', 'M3_high_pu'))
    minimum_scenarios = int(gate.get('minimum_direction_scenarios', 3))
    minimum_hard_seeds = int(gate.get('minimum_hard_seed_directions', 2))
    data = metrics[metrics['agent'] == agent].copy()
    if 'evaluation_variant' in data and (data['evaluation_variant'] == 'nominal').any():
        data = data[data['evaluation_variant'] == 'nominal']
    if 'is_final_checkpoint' in data:
        data = data[data['is_final_checkpoint'].astype(bool)]
    pivot = data.pivot_table(
        index=['scenario', 'seed'],
        columns='module_variant',
        values=['system_vt_throughput', 'plr', 'overlay_collision_rate'],
        aggfunc='mean',
    )
    required_columns = [
        (metric, variant)
        for metric in ('system_vt_throughput', 'plr', 'overlay_collision_rate')
        for variant in E2_VARIANTS
    ]
    rows: list[dict[str, object]] = []
    if pivot.empty or any(column not in pivot.columns for column in required_columns):
        table = pd.DataFrame(
            [{'check': 'complete_signal_matrix', 'passed': False, 'detail': 'missing variants'}]
        )
        return table, False

    dqi = (
        pivot[('system_vt_throughput', 'full')]
        - pivot[('system_vt_throughput', 'no_dqi')]
    )
    dqi_by_scenario = dqi.groupby(level='scenario').mean()
    dqi_hard = dqi.xs(hard_scenario, level='scenario')
    rows.extend(
        [
            {
                'check': 'dqi_hard_mean_positive',
                'passed': float(dqi_hard.mean()) > 0.0,
                'detail': f'mean_delta={dqi_hard.mean():.6g}',
            },
            {
                'check': 'dqi_scenario_direction',
                'passed': int((dqi_by_scenario > 0.0).sum()) >= minimum_scenarios,
                'detail': f'{int((dqi_by_scenario > 0.0).sum())}/{len(dqi_by_scenario)} positive',
            },
            {
                'check': 'dqi_hard_seed_direction',
                'passed': int((dqi_hard > 0.0).sum()) >= minimum_hard_seeds,
                'detail': f'{int((dqi_hard > 0.0).sum())}/{len(dqi_hard)} positive',
            },
        ]
    )

    candidates: list[tuple[int, int, float, str, pd.Series]] = []
    for metric in ('plr', 'overlay_collision_rate'):
        gain = pivot[(metric, 'no_prediction')] - pivot[(metric, 'full')]
        by_scenario = gain.groupby(level='scenario').mean()
        hard = gain.xs(hard_scenario, level='scenario')
        candidates.append(
            (
                int((by_scenario > 0.0).sum()),
                int((hard > 0.0).sum()),
                float(hard.mean()),
                metric,
                gain,
            )
        )
    candidates.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
    scenario_positive, seed_positive, hard_mean, prediction_metric, prediction_gain = candidates[0]
    rows.extend(
        [
            {
                'check': 'prediction_hard_mean_positive',
                'passed': hard_mean > 0.0,
                'detail': f'{prediction_metric}: mean_gain={hard_mean:.6g}',
            },
            {
                'check': 'prediction_scenario_direction',
                'passed': scenario_positive >= minimum_scenarios,
                'detail': f'{prediction_metric}: {scenario_positive}/{len(E2_SCENARIOS)} positive',
            },
            {
                'check': 'prediction_hard_seed_direction',
                'passed': seed_positive >= minimum_hard_seeds,
                'detail': f'{prediction_metric}: {seed_positive}/{len(E2_SEEDS)} positive',
            },
        ]
    )

    means = data.groupby(['scenario', 'module_variant'])[
        ['system_vt_throughput', 'plr', 'overlay_collision_rate']
    ].mean()
    dominated: list[str] = []
    for scenario in E2_SCENARIOS:
        full = means.loc[(scenario, 'full')]
        for variant in ('no_dqi', 'no_prediction'):
            candidate = means.loc[(scenario, variant)]
            no_worse = (
                candidate['system_vt_throughput'] >= full['system_vt_throughput']
                and candidate['plr'] <= full['plr']
                and candidate['overlay_collision_rate'] <= full['overlay_collision_rate']
            )
            strictly_better = (
                candidate['system_vt_throughput'] > full['system_vt_throughput']
                or candidate['plr'] < full['plr']
                or candidate['overlay_collision_rate'] < full['overlay_collision_rate']
            )
            if no_worse and strictly_better:
                dominated.append(f'{scenario}:{variant}')
    rows.append(
        {
            'check': 'full_not_dominated',
            'passed': not dominated,
            'detail': 'none' if not dominated else '|'.join(dominated),
        }
    )
    table = pd.DataFrame(rows)
    return table, bool(table['passed'].all())


def module_seed_values_table(metrics: pd.DataFrame) -> pd.DataFrame:
    data = (
        metrics[metrics['is_final_checkpoint'].astype(bool)].copy()
        if 'is_final_checkpoint' in metrics
        else metrics.copy()
    )
    columns = [
        'scenario',
        'seed',
        'agent',
        'module_variant',
        *PRIMARY_METRICS,
        'access_selection_ratio',
        'underlay_selection_ratio',
        'mean_queue_length',
        'mean_hol_age_slots',
    ]
    return data[[column for column in columns if column in data]].sort_values(
        ['scenario', 'seed', 'module_variant']
    )


def render_e2_report(title: str, table: pd.DataFrame, passed: bool) -> str:
    lines = [f'# {title}', '', f'- Overall: **{"PASS" if passed else "FAIL"}**', '']
    lines.extend(f'    {line}' for line in table.to_string(index=False).splitlines())
    return '\n'.join(lines) + '\n'
