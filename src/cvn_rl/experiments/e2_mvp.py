from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def evaluate_e2_mvp_gate(
    metrics: pd.DataFrame,
    phase: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    phase = phase.upper()
    final = metrics.copy()
    if 'evaluation_variant' in final:
        final = final[final['evaluation_variant'].astype(str) == 'nominal']
    if 'is_final_checkpoint' in final:
        final = final[final['is_final_checkpoint'].astype(bool)]
    checks: list[dict[str, Any]] = []

    def add(
        name: str,
        passed: bool,
        observed: Any,
        criterion: str,
        category: str,
    ) -> None:
        checks.append(
            {
                'check': name,
                'category': category,
                'passed': bool(passed),
                'observed': observed,
                'criterion': criterion,
            }
        )

    if phase == 'A':
        payload = _phase_a(metrics, final, add)
    elif phase in {'C', 'D'}:
        payload = _phase_c_or_d(final, add, phase)
    else:
        raise ValueError(f'unknown E2 MVP phase: {phase}')
    table = pd.DataFrame(checks)
    payload.update(
        {
            'phase': phase,
            'passed': bool(table['passed'].all()) if not table.empty else False,
            'checks': checks,
        }
    )
    if phase == 'A':
        structure = table[table['category'].isin(['matrix', 'structure'])]
        safety = table[table['category'] == 'safety']
        performance = table[table['category'] == 'performance']
        payload['structure_passed'] = bool(structure['passed'].all())
        payload['safety_passed'] = bool(safety['passed'].all())
        payload['performance_passed'] = bool(performance['passed'].all())
        payload['weak_performance_signal'] = not payload['performance_passed']
        payload['passed'] = bool(
            payload['structure_passed'] and payload['safety_passed']
        )
    elif phase == 'C':
        structure = table[table['category'].isin(['matrix', 'structure'])]
        science = table[table['category'] == 'science']
        payload['structure_passed'] = bool(structure['passed'].all())
        payload['science_passed'] = bool(science['passed'].all())
        payload['passed'] = bool(
            payload['structure_passed'] and payload['science_passed']
        )
    return payload, table


def _phase_a(
    metrics: pd.DataFrame,
    final: pd.DataFrame,
    add: Any,
) -> dict[str, Any]:
    variants = {
        'legacy_full',
        'revised_full',
        'no_prediction_info',
        'prediction_mask_ablation',
    }
    relevant = final[
        final['module_variant'].astype(str).isin(variants)
        & (final['agent'].astype(str) == 'ddqn_no_esn')
    ]
    key_count = relevant[
        ['scenario', 'module_variant', 'seed', 'arrival_probability']
    ].drop_duplicates()
    add(
        'phase_a_matrix',
        len(key_count) == 12 and len(relevant) == 12,
        {'unique': len(key_count), 'rows': len(relevant)},
        '12 unique, non-duplicated final curves',
        'matrix',
    )
    add(
        'phase_a_variants',
        set(relevant['module_variant'].astype(str)) == variants,
        sorted(relevant['module_variant'].astype(str).unique().tolist()),
        'four preregistered conditions',
        'matrix',
    )
    expected_metadata = {
        'legacy_full': ('real', 'real', 'legacy_v1'),
        'revised_full': ('real', 'real', 'e2_revised_v2'),
        'no_prediction_info': ('neutral', 'real', 'e2_revised_v2'),
        'prediction_mask_ablation': ('real', 'neutral', 'e2_revised_v2'),
    }
    metadata_ok = all(
        (
            str(row.ablation_info_source),
            str(row.ablation_mask_source),
            str(row.mask_policy_version),
        ) == expected_metadata[str(row.module_variant)]
        for row in relevant.itertuples()
    )
    add('phase_a_source_metadata', metadata_ok, metadata_ok, 'registered info/mask/version tuple', 'structure')
    paired = _paired(final, ['scenario', 'arrival_probability', 'seed'])
    mask_equal = _pair_column_equal(
        paired, 'revised_full', 'no_prediction_info', 'raw_action_mask_hash'
    )
    theta_equal = _pair_column_equal(
        paired, 'revised_full', 'prediction_mask_ablation',
        'theta_observation_hash',
    )
    add(
        'info_only_mask_identical',
        mask_equal,
        mask_equal,
        'revised_full == no_prediction_info per seed',
        'structure',
    )
    add(
        'mask_only_theta_identical',
        theta_equal,
        theta_equal,
        'revised_full == prediction_mask_ablation per seed',
        'structure',
    )
    trace_ok = all(
        group['evaluation_trace_hash'].nunique() == 1
        and group['train_trace_hash'].nunique() == 1
        for _, group in relevant.groupby(['scenario', 'seed', 'arrival_probability'])
    )
    add('paired_trace_identity', trace_ok, trace_ok, 'one train/eval trace hash per seed', 'structure')

    revised = relevant[relevant['module_variant'] == 'revised_full']
    legacy = relevant[relevant['module_variant'] == 'legacy_full']
    gain = float(
        revised['busy_underlay_allowed_rate'].mean()
        - legacy['busy_underlay_allowed_rate'].mean()
    )
    add('busy_underlay_gain', gain >= 0.50, gain, '>= 0.50', 'structure')
    max_pu = float(revised['pu_active_violation_rate'].max())
    max_ccdf = float(revised['interference_ccdf_gt_1p0'].max())
    add('pu_active_safety', max_pu <= 0.001, max_pu, '<= 0.001', 'safety')
    add('interference_ccdf_safety', max_ccdf <= 0.001, max_ccdf, '<= 0.001', 'safety')

    gp = relevant.pivot_table(
        index='seed', columns='module_variant', values='served_goodput_mbps', aggfunc='first'
    )
    paired_gp = gp.dropna(subset=['revised_full', 'legacy_full'])
    relative = (
        (paired_gp['revised_full'] - paired_gp['legacy_full'])
        / paired_gp['legacy_full'].abs().clip(lower=1e-12)
    )
    mean_relative = float(relative.mean()) if len(relative) else -np.inf
    noninferior = int((relative >= 0.0).sum())
    add('mean_goodput_noninferiority', mean_relative >= -0.02, mean_relative, '>= -0.02', 'performance')
    add('seed_goodput_direction', noninferior >= 2, noninferior, '>= 2 of 3 seeds', 'performance')
    revised_all = metrics[
        (metrics['module_variant'].astype(str) == 'revised_full')
        & (metrics['agent'].astype(str) == 'ddqn_no_esn')
    ]
    spreads: list[float] = []
    for _, group in revised_all.groupby('seed'):
        values = group.sort_values('checkpoint_slot').tail(3)['served_goodput_mbps'].to_numpy(float)
        if len(values) == 3:
            spreads.append(float((values.max() - values.min()) / max(abs(values.mean()), 1e-12)))
    stable = len(spreads) == 3 and max(spreads) <= 0.05
    add('last_three_checkpoint_stability', stable, spreads, '<= 0.05 for every seed', 'performance')
    return {}


def _paired(frame: pd.DataFrame, keys: list[str]) -> dict[tuple[Any, ...], pd.DataFrame]:
    return {key: group.set_index('module_variant') for key, group in frame.groupby(keys)}


def _pair_column_equal(
    groups: dict[tuple[Any, ...], pd.DataFrame],
    left: str,
    right: str,
    column: str,
) -> bool:
    comparisons = []
    for group in groups.values():
        if left in group.index and right in group.index and column in group:
            comparisons.append(str(group.loc[left, column]) == str(group.loc[right, column]))
    return len(comparisons) == 3 and all(comparisons)


def _phase_c_or_d(
    final: pd.DataFrame,
    add: Any,
    phase: str,
) -> dict[str, Any]:
    agent = 'ddqn_no_esn' if phase == 'C' else 'proposed_esn_ddqn'
    conditions = {
        'full', 'no_dqi', 'no_prediction_info', 'state_sensing_only',
    }
    loads = [0.20, 0.35] if phase == 'C' else [0.20]
    core = final[
        (final['agent'].astype(str) == agent)
        & final['module_variant'].astype(str).isin(conditions)
        & final['scenario'].astype(str).isin({'M3_high_pu', 'M10_low_pu'})
        & final['arrival_probability'].apply(
            lambda value: any(np.isclose(float(value), load) for load in loads)
        )
    ].copy()
    unique = core[
        ['scenario', 'arrival_probability', 'module_variant', 'seed', 'agent']
    ].drop_duplicates()
    expected = 48 if phase == 'C' else 24
    add(
        'phase_matrix',
        len(unique) == expected and len(core) == expected,
        {'unique': len(unique), 'rows': len(core)},
        f'{expected} unique, non-duplicated final curves',
        'matrix',
    )
    add(
        'condition_matrix',
        set(core['module_variant'].astype(str)) == conditions,
        sorted(core['module_variant'].astype(str).unique().tolist()),
        'four preregistered conditions',
        'matrix',
    )
    required_metrics = [
        'served_goodput_mbps', 'noncompletion_rate',
        'pu_active_violation_rate', 'mean_queue_occupancy',
    ]
    finite = all(
        column in core and np.isfinite(pd.to_numeric(core[column], errors='coerce')).all()
        for column in required_metrics
    )
    add('finite_metrics', finite, finite, 'all required metrics finite', 'structure')
    if phase == 'C' and 'reused_curve' in final.columns:
        extra_load = ~final['arrival_probability'].astype(float).apply(
            lambda value: np.isclose(value, 0.20)
            or np.isclose(value, 0.35)
        )
        has_fallback = bool(extra_load.any())
        core_reused = int(
            core['reused_curve'].fillna(False).astype(bool).sum()
        )
        fallback_reused = int(
            final.loc[extra_load, 'reused_curve']
            .fillna(False)
            .astype(bool)
            .sum()
        )
        expected_core_reused = 48 if has_fallback else 6
        reuse_ok = (
            core_reused == expected_core_reused
            and (not has_fallback or fallback_reused == 0)
        )
        add(
            'curve_reuse_budget',
            reuse_ok,
            {
                'core_reused': core_reused,
                'fallback_reused': fallback_reused,
            },
            (
                f'core reused={expected_core_reused}; '
                'fallback reused=0 when present'
            ),
            'structure',
        )
    trace_ok = all(
        group['evaluation_trace_hash'].nunique() == 1
        and group['train_trace_hash'].nunique() == 1
        for _, group in core.groupby(['scenario', 'arrival_probability', 'seed'])
    )
    add('paired_trace_identity', trace_ok, trace_ok, 'one train/eval trace per paired cell', 'structure')
    policy_ok = (
        core['mask_policy_version'].astype(str).nunique() == 1
        and core['mask_policy_version'].astype(str).iloc[0] == 'e2_revised_v2'
    )
    add('mask_policy_identity', policy_ok, sorted(core['mask_policy_version'].astype(str).unique()), 'e2_revised_v2 only', 'structure')
    expected_sources = {
        'full': ('real', 'real'),
        'no_dqi': ('real', 'real'),
        'no_prediction_info': ('neutral', 'real'),
        'state_sensing_only': ('neutral', 'real'),
    }
    source_ok = all(
        (
            str(row.ablation_info_source),
            str(row.ablation_mask_source),
        ) == expected_sources[str(row.module_variant)]
        for row in core.itertuples()
    )
    add('ablation_source_metadata', source_ok, source_ok, 'registered info/mask source tuple', 'structure')
    grouped = _paired(core, ['scenario', 'arrival_probability', 'seed'])
    expected_groups = 12 if phase == 'C' else 6
    mask_checks: list[bool] = []
    observation_checks: list[bool] = []
    for group in grouped.values():
        if not conditions.issubset(group.index):
            continue
        mask_checks.append(group.loc[list(conditions), 'raw_action_mask_hash'].astype(str).nunique() == 1)
        observation_checks.append(
            str(group.loc['full', 'theta_observation_hash'])
            == str(group.loc['no_dqi', 'theta_observation_hash'])
            and str(group.loc['full', 'theta_observation_hash'])
            != str(group.loc['no_prediction_info', 'theta_observation_hash'])
            and str(group.loc['full', 'dqi_observation_hash'])
            == str(group.loc['no_prediction_info', 'dqi_observation_hash'])
            and str(group.loc['full', 'dqi_observation_hash'])
            != str(group.loc['no_dqi', 'dqi_observation_hash'])
        )
    add(
        'all_masks_identical',
        len(mask_checks) == expected_groups and all(mask_checks),
        f'{sum(mask_checks)}/{len(mask_checks)}',
        f'{expected_groups}/{expected_groups} paired cells',
        'structure',
    )
    add(
        'single_information_change',
        len(observation_checks) == expected_groups and all(observation_checks),
        f'{sum(observation_checks)}/{len(observation_checks)}',
        'only the ablated observation block changes',
        'structure',
    )

    nominal = core[np.isclose(core['arrival_probability'].astype(float), 0.20)]
    effects = nominal.pivot_table(
        index=['scenario', 'seed'],
        columns='module_variant',
        values='served_goodput_mbps',
        aggfunc='first',
    )
    m10 = effects.loc['M10_low_pu'] if 'M10_low_pu' in effects.index.get_level_values(0) else pd.DataFrame()
    m3 = effects.loc['M3_high_pu'] if 'M3_high_pu' in effects.index.get_level_values(0) else pd.DataFrame()
    dqi_positive = int((m10.get('full', pd.Series(dtype=float)) - m10.get('no_dqi', pd.Series(dtype=float)) > 0).sum())
    pred_positive = int((m3.get('full', pd.Series(dtype=float)) - m3.get('no_prediction_info', pd.Series(dtype=float)) > 0).sum())
    add('dqi_direction_m10', dqi_positive >= 2, dqi_positive, '>= 2 of 3 seeds', 'science')
    add('prediction_direction_m3', pred_positive >= 2, pred_positive, '>= 2 of 3 seeds', 'science')
    combined_effects: dict[str, float] = {}
    for scenario, group in nominal.groupby('scenario'):
        means = group.groupby('module_variant')['served_goodput_mbps'].mean()
        combined_effects[str(scenario)] = float(
            means.get('full', np.nan) - means.get('state_sensing_only', np.nan)
        )
    combined_ok = (
        set(combined_effects) == {'M3_high_pu', 'M10_low_pu'}
        and all(value > 0 for value in combined_effects.values())
    )
    add('combined_ablation_direction', combined_ok, combined_effects, '> 0 in both scenarios', 'science')
    dominated_by: list[str] = []
    for scenario, group in nominal.groupby('scenario'):
        means = group.groupby('module_variant')[
            ['served_goodput_mbps', 'noncompletion_rate', 'pu_active_violation_rate']
        ].mean()
        if 'full' not in means.index:
            continue
        full = means.loc['full']
        for condition in conditions - {'full'}:
            other = means.loc[condition]
            no_worse = (
                other['served_goodput_mbps'] >= full['served_goodput_mbps']
                and other['noncompletion_rate'] <= full['noncompletion_rate']
                and other['pu_active_violation_rate'] <= full['pu_active_violation_rate']
            )
            strict = (
                other['served_goodput_mbps'] > full['served_goodput_mbps']
                or other['noncompletion_rate'] < full['noncompletion_rate']
                or other['pu_active_violation_rate'] < full['pu_active_violation_rate']
            )
            if no_worse and strict:
                dominated_by.append(f'{scenario}:{condition}')
    add('full_not_dominated', not dominated_by, dominated_by, 'no ablation dominates full', 'science')

    payload: dict[str, Any] = {}
    if phase == 'C':
        high = core[np.isclose(core['arrival_probability'].astype(float), 0.35)]
        full_high = high[high['module_variant'].astype(str) == 'full']
        by_scenario = full_high.groupby('scenario')[
            ['noncompletion_rate', 'mean_queue_occupancy']
        ].mean()
        noncompletion = by_scenario['noncompletion_rate'].to_dict()
        occupancy = by_scenario['mean_queue_occupancy'].to_dict()
        if len(noncompletion) == 2 and any(value > 0.50 for value in noncompletion.values()):
            decision = 'fallback_0.30'
        elif len(noncompletion) == 2 and all(value < 0.01 for value in noncompletion.values()):
            decision = 'fallback_0.40'
        elif (
            len(noncompletion) == 2
            and any(0.01 <= value <= 0.30 for value in noncompletion.values())
            and all(value <= 0.50 for value in noncompletion.values())
            and any(0.20 <= value <= 0.80 for value in occupancy.values())
        ):
            decision = 'accept_0.35'
        else:
            decision = 'indeterminate_0.35'
        payload['load_decision'] = decision
        payload['load_observed'] = {
            'noncompletion_rate': noncompletion,
            'mean_queue_occupancy': occupancy,
        }
        fallback_probabilities = sorted(
            {
                float(value)
                for value in final['arrival_probability'].astype(float).unique()
                if not np.isclose(float(value), 0.20)
                and not np.isclose(float(value), 0.35)
            }
        )
        if fallback_probabilities:
            fallback_probability = fallback_probabilities[0]
            fallback = final[
                (final['agent'].astype(str) == 'ddqn_no_esn')
                & final['module_variant'].astype(str).isin(conditions)
                & final['scenario'].astype(str).isin(
                    {'M3_high_pu', 'M10_low_pu'}
                )
                & np.isclose(
                    final['arrival_probability'].astype(float),
                    fallback_probability,
                )
            ].copy()
            fallback_keys = fallback[
                ['scenario', 'module_variant', 'seed', 'arrival_probability']
            ].drop_duplicates()
            fallback_complete = (
                len(fallback_probabilities) == 1
                and len(fallback) == 24
                and len(fallback_keys) == 24
            )
            add(
                'fallback_matrix',
                fallback_complete,
                {
                    'probabilities': fallback_probabilities,
                    'unique': len(fallback_keys),
                    'rows': len(fallback),
                },
                'one fallback probability and 24 non-duplicated curves',
                'structure',
            )
            summary_columns = [
                'served_goodput_mbps',
                'completion_rate',
                'noncompletion_rate',
                'mean_queue_occupancy',
                'pu_active_violation_rate',
            ]
            fallback_summary: list[dict[str, Any]] = []
            for (scenario, variant), group in fallback.groupby(
                ['scenario', 'module_variant']
            ):
                item: dict[str, Any] = {
                    'scenario': str(scenario),
                    'module_variant': str(variant),
                }
                for column in summary_columns:
                    item[column] = float(group[column].mean())
                fallback_summary.append(item)
            fallback_full = fallback[
                fallback['module_variant'].astype(str) == 'full'
            ].groupby('scenario')[
                ['noncompletion_rate', 'mean_queue_occupancy']
            ].mean()
            fallback_noncompletion = (
                fallback_full['noncompletion_rate'].to_dict()
            )
            fallback_occupancy = (
                fallback_full['mean_queue_occupancy'].to_dict()
            )
            if (
                len(fallback_noncompletion) == 2
                and any(value > 0.50 for value in fallback_noncompletion.values())
            ):
                fallback_assessment = 'still_too_heavy'
            elif (
                len(fallback_noncompletion) == 2
                and all(value < 0.01 for value in fallback_noncompletion.values())
            ):
                fallback_assessment = 'still_too_light'
            elif (
                len(fallback_noncompletion) == 2
                and any(
                    0.01 <= value <= 0.30
                    for value in fallback_noncompletion.values()
                )
                and all(
                    value <= 0.50
                    for value in fallback_noncompletion.values()
                )
                and any(
                    0.20 <= value <= 0.80
                    for value in fallback_occupancy.values()
                )
            ):
                fallback_assessment = 'accepted'
            else:
                fallback_assessment = 'indeterminate'
            payload['fallback_probe'] = {
                'probability': fallback_probability,
                'assessment': fallback_assessment,
                'search_exhausted': True,
                'further_search_allowed': False,
                'full_noncompletion_rate': fallback_noncompletion,
                'full_mean_queue_occupancy': fallback_occupancy,
                'condition_summary': fallback_summary,
            }
    return payload


def render_e2_mvp_report(payload: dict[str, Any]) -> str:
    status = 'PASS' if payload.get('passed') else 'FAIL'
    lines = [
        f'# E2 MVP Phase {payload.get("phase")} Gate',
        '',
        f'Overall: **{status}**',
    ]
    if 'load_decision' in payload:
        lines.extend(['', f'Load decision: **{payload["load_decision"]}**'])
    if 'fallback_probe' in payload:
        fallback = payload['fallback_probe']
        lines.extend(
            [
                '',
                (
                    'Fallback probe: '
                    f'**p={fallback["probability"]:.2f}, '
                    f'{fallback["assessment"]}** '
                    '(one-shot search exhausted)'
                ),
            ]
        )
    lines.extend(['', '| Check | Category | Pass | Observed | Criterion |', '|---|---|---:|---|---|'])
    for check in payload.get('checks', []):
        observed = str(check['observed']).replace('|', '/')
        lines.append(
            f'| {check["check"]} | {check["category"]} | '
            f'{"yes" if check["passed"] else "no"} | {observed} | '
            f'{check["criterion"]} |'
        )
    return '\n'.join(lines) + '\n'
