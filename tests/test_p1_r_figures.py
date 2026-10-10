from __future__ import annotations

from pathlib import Path

import pandas as pd

from cvn_rl.reporting.multi_vehicle_figures import (
    write_multi_vehicle_figures,
)


def _metrics() -> pd.DataFrame:
    groups = {
        'hard': (
            'theta_oracle',
            'theta_placebo_a',
            'theta_placebo_b',
        ),
        'soft': (
            'theta_soft_oracle',
            'theta_soft_placebo_a',
            'theta_soft_placebo_b',
        ),
        'continuous_separable': (
            'theta_continuous_oracle',
            'theta_continuous_placebo_a',
            'theta_continuous_placebo_b',
        ),
    }
    rows = []
    for checkpoint in (5_000, 10_000):
        for seed in (1, 2, 3):
            for rep_index, (representation, conditions) in enumerate(
                groups.items()
            ):
                for role, condition in enumerate(conditions):
                    oracle = role == 0
                    rows.append(
                        {
                            'e2_r1_gate': (
                                'prediction_representation_bridge'
                            ),
                            'module_variant': condition,
                            'scenario': 'bridge',
                            'seed': seed,
                            'checkpoint_slot': checkpoint,
                            'is_final_checkpoint': checkpoint == 10_000,
                            'system_vt_throughput': (
                                25.0 - 0.1 * rep_index
                                if oracle
                                else 23.0 - 0.1 * role
                            ),
                            'served_goodput_mbps': (
                                25.0 - 0.1 * rep_index
                                if oracle
                                else 23.0 - 0.1 * role
                            ),
                            'pu_access_violation_rate': (
                                0.0005 if oracle else 0.05
                            ),
                            'overlay_given_busy_legal_backlog': (
                                0.01 if oracle else 0.08
                            ),
                            'evaluation_theta_min': (
                                0.0
                                if representation == 'hard'
                                else 0.1
                                if representation == 'soft'
                                else 0.05
                            ),
                            'evaluation_theta_max': (
                                1.0
                                if representation == 'hard'
                                else 0.9
                                if representation == 'soft'
                                else 0.95
                            ),
                            'evaluation_theta_unique_fraction': (
                                1.0
                                if representation
                                == 'continuous_separable'
                                else 0.001
                            ),
                            'evaluation_theta_threshold_accuracy': (
                                1.0 if oracle else 0.5
                            ),
                        }
                    )
    return pd.DataFrame(rows)


def test_p1_r_dispatch_writes_four_specialized_figures(
    tmp_path: Path,
) -> None:
    outputs = write_multi_vehicle_figures(tmp_path, _metrics())
    assert len(outputs) == 12
    assert len({Path(path).stem for path in outputs}) == 4
    for relative in outputs:
        path = tmp_path / relative
        assert path.exists()
        assert path.stat().st_size > 0
