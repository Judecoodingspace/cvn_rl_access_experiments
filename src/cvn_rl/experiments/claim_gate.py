from __future__ import annotations

from typing import Any

import pandas as pd


def evaluate_claim_gates(
    effects: pd.DataFrame,
    metrics: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, bool]:
    if effects.empty or metrics.empty:
        table = pd.DataFrame(
            [{'claim': 'module_effects_available', 'passed': False, 'detail': 'no effects'}]
        )
        return table, False
    gate = config.get('claim_gate', {})
    final = (
        metrics[metrics['is_final_checkpoint'].astype(bool)].copy()
        if 'is_final_checkpoint' in metrics
        else metrics.copy()
    )
    if 'evaluation_variant' in effects and (effects['evaluation_variant'] == 'nominal').any():
        effects = effects[effects['evaluation_variant'] == 'nominal']
    if 'evaluation_variant' in final and (final['evaluation_variant'] == 'nominal').any():
        final = final[final['evaluation_variant'] == 'nominal']
    agent = str(
        gate.get(
            'agent',
            'independent_ddqn'
            if 'independent_ddqn' in set(final['agent'])
            else 'ddqn_no_esn',
        )
    )
    effects = effects[effects['agent'] == agent]
    final = final[final['agent'] == agent]
    rows: list[dict[str, object]] = []
    is_multi = bool(not final.empty and final['num_vehicles'].min() > 1)

    dqi = effects[
        (effects['ablated_variant'] == 'no_dqi')
        & (effects['metric'] == 'system_vt_throughput')
    ]
    prediction_candidates = effects[
        (effects['ablated_variant'] == 'no_prediction')
        & (effects['metric'].isin(['plr', 'overlay_collision_rate']))
    ]
    minimum_directions = int(gate.get('minimum_direction_scenarios', 2 if is_multi else 3))
    dqi_positive = int((dqi['oriented_module_gain'] > 0.0).sum())
    rows.append(
        {
            'claim': 'dqi_direction_consistency',
            'passed': dqi_positive >= minimum_directions,
            'detail': f'{dqi_positive}/{dqi["scenario"].nunique()} scenarios positive',
        }
    )

    prediction_scores = []
    for metric, group in prediction_candidates.groupby('metric'):
        prediction_scores.append(
            (
                int((group['oriented_module_gain'] > 0.0).sum()),
                str(metric),
                group,
            )
        )
    prediction_scores.sort(key=lambda item: (item[0], item[1]), reverse=True)
    prediction_positive = prediction_scores[0][0] if prediction_scores else 0
    prediction_metric = prediction_scores[0][1] if prediction_scores else 'none'
    prediction_group = prediction_scores[0][2] if prediction_scores else pd.DataFrame()
    rows.append(
        {
            'claim': 'prediction_direction_consistency',
            'passed': prediction_positive >= minimum_directions,
            'detail': (
                f'{prediction_positive}/'
                f'{prediction_group["scenario"].nunique() if not prediction_group.empty else 0} '
                f'scenarios positive using {prediction_metric}'
            ),
        }
    )

    if not is_multi:
        hard_scenario = str(gate.get('hard_scenario', 'M3_high_pu'))
        hard_dqi = dqi[dqi['scenario'] == hard_scenario]
        dqi_hard_pass = bool(
            not hard_dqi.empty
            and float(hard_dqi.iloc[0]['ci95_low']) > 0.0
            and bool(hard_dqi.iloc[0].get('holm_significant_0p05', True))
        )
        rows.append(
            {
                'claim': 'dqi_hard_scenario_ci',
                'passed': dqi_hard_pass,
                'detail': f'{hard_scenario}: throughput paired CI lower bound > 0',
            }
        )
        hard_prediction = prediction_candidates[
            prediction_candidates['scenario'] == hard_scenario
        ]
        prediction_significant = (
            hard_prediction['holm_significant_0p05'].astype(bool)
            if 'holm_significant_0p05' in hard_prediction
            else pd.Series(True, index=hard_prediction.index)
        )
        prediction_hard_pass = bool(
            not hard_prediction.empty
            and (
                (hard_prediction['ci95_low'].astype(float) > 0.0)
                & prediction_significant
            ).any()
        )
        rows.append(
            {
                'claim': 'prediction_hard_scenario_ci',
                'passed': prediction_hard_pass,
                'detail': f'{hard_scenario}: PLR or overlay-collision paired CI lower bound > 0',
            }
        )

    metric_names = [
        name
        for name in ['system_vt_throughput', 'plr', 'overlay_collision_rate']
        if name in final
    ]
    dominated = []
    if len(metric_names) == 3:
        means = final.groupby(['scenario', 'module_variant'])[metric_names].mean()
        for scenario in final['scenario'].drop_duplicates():
            try:
                full = means.loc[(scenario, 'full')]
            except KeyError:
                continue
            for variant in ('no_dqi', 'no_prediction'):
                try:
                    candidate = means.loc[(scenario, variant)]
                except KeyError:
                    continue
                no_worse = (
                    candidate['system_vt_throughput'] >= full['system_vt_throughput']
                    and candidate['plr'] <= full['plr']
                    and candidate['overlay_collision_rate']
                    <= full['overlay_collision_rate']
                )
                strictly_better = (
                    candidate['system_vt_throughput'] > full['system_vt_throughput']
                    or candidate['plr'] < full['plr']
                    or candidate['overlay_collision_rate']
                    < full['overlay_collision_rate']
                )
                if no_worse and strictly_better:
                    dominated.append(f'{scenario}:{variant}')
    rows.append(
        {
            'claim': 'full_not_dominated_by_single_module',
            'passed': not dominated,
            'detail': 'none' if not dominated else '|'.join(dominated),
        }
    )
    table = pd.DataFrame(rows)
    return table, bool(table['passed'].all())


def render_claim_gate_report(table: pd.DataFrame, passed: bool) -> str:
    status = 'PASS' if passed else 'FAIL'
    lines = [
        '# Pre-registered Claim Gate',
        '',
        f'- Overall: **{status}**',
        '',
    ]
    lines.extend(f'    {line}' for line in table.to_string(index=False).splitlines())
    lines.extend(
        [
            '',
            'A failed check means the corresponding paper claim must be weakened or omitted.',
            '',
        ]
    )
    return '\n'.join(lines)
