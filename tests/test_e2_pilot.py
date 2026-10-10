from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cvn_rl.experiments.config import load_config
from cvn_rl.experiments.e2_pilot import (
    E2_SCENARIOS,
    E2_SEEDS,
    E2_VARIANTS,
    evaluate_e2_pilot_signal,
    require_e2_pilot_integrity,
    validate_e2_pilot_metrics,
)


def _pilot_metrics() -> pd.DataFrame:
    rows = []
    throughput = {
        'sensing_only': 7.0,
        'no_dqi': 8.0,
        'no_prediction': 9.0,
        'full': 10.0,
    }
    for scenario_index, scenario in enumerate(E2_SCENARIOS):
        for seed in E2_SEEDS:
            train_hash = f'train-{scenario}-{seed}'
            eval_hash = f'eval-{scenario}-{seed}'
            for variant in E2_VARIANTS:
                for checkpoint in range(1000, 30001, 1000):
                    rows.append(
                        {
                            'scenario': scenario,
                            'seed': seed,
                            'agent': 'ddqn_no_esn',
                            'module_variant': variant,
                            'evaluation_variant': 'nominal',
                            'checkpoint_slot': checkpoint,
                            'is_final_checkpoint': checkpoint == 30000,
                            'train_trace_hash': train_hash,
                            'evaluation_trace_hash': eval_hash,
                            'parameter_hash_before_eval': f'params-{variant}-{seed}',
                            'parameter_hash_after_eval': f'params-{variant}-{seed}',
                            'observation_dimension': 40 + scenario_index,
                            'state_feature_signature': 'delta|fresh_dqi|theta',
                            'mean_reward': throughput[variant] / 10.0,
                            'system_vt_throughput': (
                                throughput[variant] + scenario_index + seed * 0.01
                            ),
                            'plr': (
                                0.20
                                if variant == 'no_prediction'
                                else 0.10
                                if variant == 'full'
                                else 0.14
                            ),
                            'overlay_collision_rate': (
                                0.15
                                if variant == 'no_prediction'
                                else 0.05
                                if variant == 'full'
                                else 0.10
                            ),
                            'conditional_underlay_violation_rate': 0.0,
                            'contention_loss_rate': 0.0,
                            'jain_fairness': 1.0,
                            'access_selection_ratio': 0.5,
                            'underlay_selection_ratio': 0.5,
                            'mean_queue_length': 0.2,
                            'mean_hol_age_slots': 1.0,
                        }
                    )
    return pd.DataFrame(rows)


def test_shard_configs_expand_to_exact_e2_matrix() -> None:
    configs = [
        load_config(f'configs/e2_pilot_{scenario}.json')
        for scenario in E2_SCENARIOS
    ]
    combinations = {
        (scenario['name'], int(seed), str(variant))
        for config in configs
        for scenario in config['scenarios']
        for seed in config['seeds']
        for variant in config['module_variants']
    }
    assert len(combinations) == 48
    assert all(config['checkpoint_evaluation']['eval_episodes'] == 5 for config in configs)


def test_e2_integrity_and_signal_gate_pass_complete_synthetic_data() -> None:
    metrics = _pilot_metrics()
    checks, passed = validate_e2_pilot_metrics(metrics)
    assert passed, checks.to_string(index=False)
    signal, signal_passed = evaluate_e2_pilot_signal(metrics)
    assert signal_passed, signal.to_string(index=False)


@pytest.mark.parametrize(
    'mutation',
    ['duplicate', 'missing', 'trace_mismatch', 'parameter_changed', 'schema_mismatch'],
)
def test_e2_integrity_rejects_invalid_shards(mutation: str) -> None:
    metrics = _pilot_metrics()
    if mutation == 'duplicate':
        metrics = pd.concat([metrics, metrics.iloc[[0]]], ignore_index=True)
    elif mutation == 'missing':
        metrics = metrics.iloc[1:].copy()
    elif mutation == 'trace_mismatch':
        metrics.loc[0, 'train_trace_hash'] = 'wrong-trace'
    elif mutation == 'parameter_changed':
        metrics.loc[0, 'parameter_hash_after_eval'] = 'changed'
    elif mutation == 'schema_mismatch':
        metrics.loc[0, 'observation_dimension'] += 1
    with pytest.raises(ValueError, match='E2 pilot integrity failed'):
        require_e2_pilot_integrity(metrics)


def test_formal_config_restores_ten_evaluation_episodes() -> None:
    formal = load_config('configs/e2_single_vt_module_ablation_formal.json')
    assert formal['checkpoint_evaluation']['eval_episodes'] == 10
    assert formal['claim_gate']['enabled'] is True
    assert formal['pilot_signal_gate']['enabled'] is False
