from __future__ import annotations

import numpy as np
import pandas as pd

from cvn_rl.experiments.statistics import (
    module_ablation_bootstrap_table,
    module_interaction_bootstrap_table,
    select_training_horizon,
)


def _module_rows() -> pd.DataFrame:
    values = {
        'sensing_only': 4.0,
        'no_dqi': 6.0,
        'no_prediction': 7.0,
        'full': 10.0,
    }
    rows = []
    for seed in (1, 2, 3):
        for variant, throughput in values.items():
            rows.append(
                {
                    'experiment': 'synthetic',
                    'scenario': 'hard',
                    'agent': 'ddqn_no_esn',
                    'seed': seed,
                    'module_variant': variant,
                    'checkpoint_slot': 30000,
                    'is_final_checkpoint': True,
                    'system_vt_throughput': throughput + seed * 0.01,
                    'mean_vt_throughput': throughput + seed * 0.01,
                    'plr': 1.0 - throughput / 20.0,
                    'overlay_collision_rate': 1.0 - throughput / 20.0,
                    'pu_active_violation_rate': 0.0,
                }
            )
    return pd.DataFrame(rows)


def test_module_bootstrap_and_interaction_match_synthetic_effects() -> None:
    data = _module_rows()
    effects = module_ablation_bootstrap_table(data, bootstrap_samples=1000)
    no_dqi = effects[
        (effects['ablated_variant'] == 'no_dqi')
        & (effects['metric'] == 'system_vt_throughput')
    ].iloc[0]
    assert no_dqi['raw_full_minus_ablated'] == 4.0
    assert no_dqi['ci95_low'] > 0.0
    assert 'holm_p_value' in effects
    interaction = module_interaction_bootstrap_table(data, bootstrap_samples=1000)
    throughput = interaction[
        interaction['metric'] == 'system_vt_throughput'
    ].iloc[0]
    assert np.isclose(throughput['interaction_mean'], 1.0)


def test_training_horizon_requires_three_stable_intervals() -> None:
    rows = []
    for checkpoint, reward, throughput, plr in [
        (7000, 1.0, 10.0, 0.20),
        (8000, 1.01, 10.1, 0.198),
        (9000, 1.02, 10.2, 0.196),
        (10000, 1.03, 10.3, 0.194),
        (11000, 1.04, 10.4, 0.192),
        (12000, 1.05, 10.5, 0.190),
    ]:
        rows.append(
            {
                'scenario': 'stable',
                'agent': 'ddqn_no_esn',
                'module_variant': 'full',
                'checkpoint_slot': checkpoint,
                'mean_reward': reward,
                'system_vt_throughput': throughput,
                'plr': plr,
            }
        )
    table = select_training_horizon(pd.DataFrame(rows))
    scenario = table[table['scenario'] == 'stable'].iloc[0]
    assert bool(scenario['converged'])
    assert scenario['selected_checkpoint'] == 10000
