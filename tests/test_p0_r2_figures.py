from __future__ import annotations

from pathlib import Path

import pandas as pd

from cvn_rl.reporting.multi_vehicle_figures import (
    write_multi_vehicle_figures,
)


def _metrics() -> pd.DataFrame:
    rows = []
    conditions = (
        'theta_oracle',
        'theta_placebo_a',
        'theta_placebo_b',
        'theta_neutral',
    )
    for checkpoint in (5_000, 10_000):
        for seed in (1, 2, 3):
            for index, condition in enumerate(conditions):
                oracle = condition == 'theta_oracle'
                violation = 0.0 if oracle else 0.04 + 0.005 * index
                goodput = 25.0 if oracle else 23.0 - 0.2 * index
                rows.append(
                    {
                        'e2_r1_gate': 'prediction_positive_control_r2',
                        'module_variant': condition,
                        'checkpoint_slot': checkpoint,
                        'is_final_checkpoint': checkpoint == 10_000,
                        'seed': seed,
                        'mean_reward': 0.3 + 0.01 * seed - 0.02 * index,
                        'system_vt_throughput': goodput,
                        'served_goodput_mbps': goodput,
                        'pu_access_violation_rate': violation,
                        'plr': 0.20 + 0.01 * index,
                        'completion_rate': 0.80 - 0.01 * index,
                        'mean_queue_occupancy': 5.0 + index,
                        'underlay_selection_ratio': 0.3 + 0.1 * index,
                        'queue_overflow_drop_rate': 0.1 + 0.01 * index,
                        'contention_loss_rate': 0.0,
                        'pu_interruption_rate': violation,
                        'low_rate_backlog_rate': 0.2 + 0.01 * index,
                        'interference_ccdf_gt_0p25': violation + 0.10,
                        'interference_ccdf_gt_0p5': violation + 0.07,
                        'interference_ccdf_gt_0p75': violation + 0.04,
                        'interference_ccdf_gt_1p0': violation,
                        'interference_ccdf_gt_1p25': violation * 0.7,
                        'interference_ccdf_gt_1p5': violation * 0.5,
                        'interference_ccdf_gt_2p0': violation * 0.2,
                        'overlay_given_busy_legal_backlog': violation,
                        'mean_safe_action_regret_mbps': 1.0 + index,
                        'average_selected_theta': 0.8 - 0.1 * index,
                        'access_selection_ratio': 0.9 - 0.02 * index,
                        'overlay_given_idle_legal_backlog': 0.8 - 0.05 * index,
                    }
                )
    return pd.DataFrame(rows)


def test_p0_r2_dispatch_writes_ten_condition_separated_figures(
    tmp_path: Path,
) -> None:
    outputs = write_multi_vehicle_figures(tmp_path, _metrics())
    assert len(outputs) == 30
    assert len({Path(path).stem for path in outputs}) == 10
    for relative in outputs:
        path = tmp_path / relative
        assert path.exists()
        assert path.stat().st_size > 0


def test_p3_frozen_dispatch_keeps_prediction_conditions_separate(tmp_path: Path) -> None:
    metrics = _metrics().copy()
    metrics['e2_r1_gate'] = 'prediction_frozen_predictor'
    rename = {
        'theta_placebo_a': 'theta_central_markov',
        'theta_placebo_b': 'theta_central_frozen',
        'theta_neutral': 'theta_central_frozen_placebo',
    }
    metrics['module_variant'] = metrics['module_variant'].replace(rename)
    neutral = metrics[metrics['module_variant'] == 'theta_central_frozen_placebo'].copy()
    neutral['module_variant'] = 'theta_neutral'
    metrics = pd.concat([metrics, neutral], ignore_index=True)
    outputs = write_multi_vehicle_figures(tmp_path, metrics)
    assert len(outputs) == 30
    assert (tmp_path / 'figures' / 'vt_throughput_checkpoint_convergence.pdf').exists()
