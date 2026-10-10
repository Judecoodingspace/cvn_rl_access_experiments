from __future__ import annotations

import pandas as pd

from cvn_rl.experiments.claim_gate import evaluate_claim_gates


def test_claim_gate_enforces_ci_direction_and_non_domination() -> None:
    effects = []
    metrics = []
    for scenario in ('M3_high_pu', 'M3_low_pu', 'M10_high_pu', 'M10_low_pu'):
        effects.extend(
            [
                {
                    'scenario': scenario,
                    'agent': 'ddqn_no_esn',
                    'evaluation_variant': 'nominal',
                    'ablated_variant': 'no_dqi',
                    'metric': 'system_vt_throughput',
                    'oriented_module_gain': 1.0,
                    'ci95_low': 0.2,
                },
                {
                    'scenario': scenario,
                    'agent': 'ddqn_no_esn',
                    'evaluation_variant': 'nominal',
                    'ablated_variant': 'no_prediction',
                    'metric': 'overlay_collision_rate',
                    'oriented_module_gain': 0.1,
                    'ci95_low': 0.02,
                },
            ]
        )
        for variant, throughput, plr, collision in [
            ('full', 10.0, 0.1, 0.1),
            ('no_dqi', 9.0, 0.1, 0.1),
            ('no_prediction', 10.0, 0.2, 0.2),
        ]:
            metrics.append(
                {
                    'scenario': scenario,
                    'agent': 'ddqn_no_esn',
                    'evaluation_variant': 'nominal',
                    'module_variant': variant,
                    'num_vehicles': 1,
                    'is_final_checkpoint': True,
                    'system_vt_throughput': throughput,
                    'plr': plr,
                    'overlay_collision_rate': collision,
                }
            )
    table, passed = evaluate_claim_gates(
        pd.DataFrame(effects),
        pd.DataFrame(metrics),
        {
            'claim_gate': {
                'agent': 'ddqn_no_esn',
                'hard_scenario': 'M3_high_pu',
                'minimum_direction_scenarios': 3,
            }
        },
    )
    assert passed
    assert table['passed'].all()
