from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from cvn_rl.experiments.config import load_config
from cvn_rl.experiments.e2_r1 import (
    _run_or_skip_load_calibration,
    calibrate_underlay_power_fraction,
    evaluate_e2_r1_gate,
)
from cvn_rl.experiments.multi_vehicle_runner import (
    _build_cases,
    _resolve_trace,
    run_multi_vehicle_experiment,
)
from cvn_rl.simulation.actions import encode_action
from cvn_rl.simulation.descriptor_bundle import load_descriptor_bundle, save_descriptor_bundle
from cvn_rl.simulation.multi_env import MultiVehicleAccessEnv
from cvn_rl.simulation.multi_traces import generate_multi_vehicle_trace
from cvn_rl.simulation.rate_model import underlay_rate_mbps
from cvn_rl.simulation.state_builder import feature_slices


def _config(slots: int = 8, channels: int = 3) -> dict:
    config = load_config('configs/base_multi_vehicle.json')
    config.update(
        {
            'slots': slots,
            'num_channels': channels,
            'num_vehicles': 1,
            'underlay_rate_model': 'power_aware_v1',
            'protection_variant': 'gate_state_mask',
            'action_mask_thresholds': {
                'sensing_idle_min': 0.5,
                'overlay_theta_min': None,
                'underlay_power_cap_min': 0.0,
                'underlay_theta_min': None,
            },
        }
    )
    config['underlay_power'] = {
        **config['underlay_power'],
        'service_power_ceiling_fraction': 0.2,
        'pu_to_su_interference_ratio': 1.0,
    }
    config['packet_traffic'] = {
        'arrival_probability': 1.0,
        'queue_capacity_packets': 20,
        'packet_size_bits': 1_000,
    }
    return config


def test_r1_observation_rejects_occupancy_and_mask_ignores_theta() -> None:
    config = _config()
    trace = generate_multi_vehicle_trace(config, seed=1, slots=8)
    trace.sensing_idle[:] = 1.0
    trace.theta[:] = 0.0
    env = MultiVehicleAccessEnv(trace, config)
    observation, infos = env.reset()
    assert 'occupancy_type' not in env.state_features
    overlay = encode_action(0, 'overlay', trace.num_channels)
    assert infos[0]['raw_action_mask'][overlay]
    with pytest.raises(ValueError, match='privileged simulator state'):
        MultiVehicleAccessEnv(
            trace,
            config,
            state_features=['delta', 'occupancy_type'],
        )
    assert observation.shape[1] > 0


def test_power_aware_underlay_rate_is_monotone_and_bounded() -> None:
    config = _config()
    rates = [
        underlay_rate_mbps(80.0, 10.0, power, False, config)
        for power in (0.0, 0.1, 0.2)
    ]
    assert 0.0 <= rates[0] < rates[1] < rates[2] <= 80.0
    busy = underlay_rate_mbps(80.0, 10.0, 0.2, True, config)
    assert 0.0 <= busy < rates[2]


def test_enabled_modes_can_disable_underlay() -> None:
    config = _config()
    config['action'] = {'enabled_modes': ['overlay']}
    trace = generate_multi_vehicle_trace(config, seed=11, slots=8)
    trace.sensing_idle[:] = 1.0
    _, infos = MultiVehicleAccessEnv(trace, config).reset()
    for channel in range(trace.num_channels):
        assert infos[0]['raw_action_mask'][encode_action(channel, 'overlay', 3)]
        assert not infos[0]['raw_action_mask'][encode_action(channel, 'underlay', 3)]


def test_positive_control_can_skip_load_calibration() -> None:
    config = _config()
    config['arrival_probabilities'] = [1.0]
    config['load_calibration'] = {'enabled': False}
    table, selected = _run_or_skip_load_calibration(
        config,
        dry_run=False,
        quick=False,
    )
    assert selected == 1.0
    assert bool(table.loc[0, 'calibration_skipped'])


def test_overlay_only_control_can_skip_power_calibration() -> None:
    config = _config()
    config['power_calibration'] = {'enabled': False}
    config['underlay_power']['service_power_ceiling_fraction'] = 0.17
    table, selected = calibrate_underlay_power_fraction(config)
    assert selected == 0.17
    assert bool(table.loc[0, 'calibration_skipped'])


def test_current_cqi_uses_previous_slot_and_hidden_quality_is_not_observed() -> None:
    config = _config(slots=4, channels=2)
    config['hidden_lookahead_slots'] = 3
    config['protection_variant'] = 'gate_only'
    config['state_features'] = ['current_cqi']
    trace = generate_multi_vehicle_trace(config, seed=2, slots=4)
    trace.quality_mbps[0] = [[20.0, 40.0]]
    trace.quality_mbps[1] = [[80.0, 80.0]]
    env = MultiVehicleAccessEnv(trace, config)
    first, _ = env.reset()
    assert np.allclose(first, 0.5)
    second, _, _, _ = env.step(np.array([0]))
    assert np.allclose(second, [[0.25, 0.50]])
    assert trace.quality_lookahead_mbps.shape == (3, 1, 2)


def test_switching_blackout_and_dwell_boundaries_legacy() -> None:
    """Legacy semantics: dwell starts after blackout ends. Total lock = blackout + dwell."""
    config = _config(slots=10, channels=2)
    config['switching'] = {
        'blackout_slots': 2,
        'min_dwell_slots': 3,
        'dwell_semantics': 'legacy_after_blackout',
    }
    config['state_features'] = [
        'current_channel',
        'dwell_remaining',
        'blackout_remaining',
    ]
    trace = generate_multi_vehicle_trace(config, seed=3, slots=10)
    trace.occupancy[:] = False
    trace.sensing_idle[:] = 1.0
    env = MultiVehicleAccessEnv(trace, config)
    env.reset()
    channel0 = encode_action(0, 'overlay', 2)
    channel1 = encode_action(1, 'overlay', 2)

    _, _, _, first = env.step(np.array([channel0]))
    assert not first['switch_started'][0]
    assert first['dwell_remaining'][0] == 3
    assert env.dwell_remaining[0] == 2
    for _ in range(2):
        env.step(np.array([0]))
    assert env.dwell_remaining[0] == 0
    assert env.agent_infos()[0]['action_mask'][channel1]

    _, _, _, switch = env.step(np.array([channel1]))
    assert switch['switch_started'][0]
    assert switch['blackout_active'][0]
    assert env.blackout_remaining[0] == 1
    blackout_mask = env.agent_infos()[0]['action_mask']
    assert blackout_mask.tolist() == [True, False, False, False, False]
    _, _, _, second_blackout = env.step(np.array([0]))
    assert second_blackout['blackout_active'][0]
    assert env.blackout_remaining[0] == 0
    # Legacy: dwell set to min_dwell AFTER blackout expires
    assert env.dwell_remaining[0] == 3
    assert second_blackout['dwell_violation'][0] == 0


def test_switching_total_from_switch_dwell_includes_blackout() -> None:
    """total_from_switch: dwell armed at switch, blackout counted inside dwell.
    Total lock = min_dwell_slots (matches offline evaluate_channel_policy)."""
    config = _config(slots=10, channels=2)
    config['switching'] = {
        'blackout_slots': 2,
        'min_dwell_slots': 3,
        'dwell_semantics': 'total_from_switch',
    }
    config['state_features'] = [
        'current_channel',
        'dwell_remaining',
        'blackout_remaining',
    ]
    trace = generate_multi_vehicle_trace(config, seed=3, slots=10)
    trace.occupancy[:] = False
    trace.sensing_idle[:] = 1.0
    env = MultiVehicleAccessEnv(trace, config)
    env.reset()
    channel0 = encode_action(0, 'overlay', 2)
    channel1 = encode_action(1, 'overlay', 2)

    # Initial selection: free, dwell armed
    _, _, _, first = env.step(np.array([channel0]))
    assert not first['switch_started'][0]
    assert first['dwell_remaining'][0] == 3
    assert env.dwell_remaining[0] == 2
    assert env.blackout_remaining[0] == 0
    for _ in range(2):
        env.step(np.array([0]))
    assert env.dwell_remaining[0] == 0  # dwell expired

    # Switch to channel 1 — dwell AND blackout armed simultaneously
    _, _, _, switch = env.step(np.array([channel1]))
    assert switch['switch_started'][0]
    assert switch['dwell_remaining'][0] == 3  # dwell armed at switch
    assert switch['blackout_active'][0]       # blackout active this slot
    assert env.blackout_remaining[0] == 1     # blackout decremented 2→1
    assert env.dwell_remaining[0] == 2        # dwell decremented 3→2
    blackout_mask = env.agent_infos()[0]['action_mask']
    assert blackout_mask.tolist() == [True, False, False, False, False]

    # Second slot of blackout; both timers decrement
    _, _, _, second = env.step(np.array([0]))
    assert second['blackout_active'][0]
    assert env.blackout_remaining[0] == 0     # blackout expired
    assert env.dwell_remaining[0] == 1        # dwell still active

    # Third slot: blackout over, dwell active — can transmit, cannot switch
    _, _, _, third = env.step(np.array([channel0]))
    assert not third['blackout_active'][0]
    assert not third['switch_started'][0]     # cannot switch yet
    assert env.dwell_remaining[0] == 0        # dwell expired

    # Fourth slot after switch: dwell=0, switching permitted again
    assert env.agent_infos()[0]['action_mask'][channel0]
    # Total lock = 3 slots = min_dwell (matches D0.6 / offline evaluator)


def test_dwell_total_from_switch_matches_offline_policy_evaluator() -> None:
    """Cross-check: RL env total_from_switch lock length == offline
    evaluate_channel_policy lock length on the same ranking sequence."""
    from cvn_rl.simulation.dqi_policy_value import (
        SwitchingCost,
        evaluate_channel_policy,
    )

    blackout, dwell = 2, 5
    slots = 20
    channels = 3
    rng = np.random.default_rng(42)

    # Build a trace where channel 0 and channel 1 alternate as best
    goodput = np.zeros((slots, channels), dtype=float)
    raw = np.zeros((slots, channels), dtype=float)
    ranking = np.zeros((slots, channels), dtype=float)
    projected = np.zeros((slots, channels), dtype=float)
    oracle = np.zeros((slots, channels), dtype=float)
    violations = np.zeros((slots, 1, channels), dtype=bool)

    # Channel 0 best for first half, channel 1 best for second half
    goodput[:10, 0] = 100.0
    goodput[:10, 1] = 50.0
    goodput[10:, 1] = 100.0
    goodput[10:, 0] = 50.0
    goodput[:, 2] = 10.0
    raw[:] = goodput
    ranking[:] = goodput
    projected[:] = goodput * 100  # large projection to bypass hysteresis
    oracle[:] = goodput

    # Offline evaluation
    cost = SwitchingCost(blackout_slots=blackout, min_dwell_slots=dwell)
    offline = evaluate_channel_policy(
        goodput, raw, violations, ranking, projected, oracle, cost,
        start_slot=1, use_hysteresis=False,
    )
    offline_switches = np.flatnonzero(offline.switched)
    assert len(offline_switches) >= 1, "test trace should trigger at least one switch"

    # RL env on the same ranking-driven actions
    config = _config(slots=slots, channels=channels)
    config['switching'] = {
        'blackout_slots': blackout,
        'min_dwell_slots': dwell,
        'dwell_semantics': 'total_from_switch',
    }
    config['state_features'] = [
        'current_channel', 'dwell_remaining', 'blackout_remaining',
    ]
    trace = generate_multi_vehicle_trace(config, seed=7, slots=slots)
    trace.occupancy[:] = False
    trace.sensing_idle[:] = 1.0
    env = MultiVehicleAccessEnv(trace, config)
    env.reset()

    # Follow the ranking-driven channel selections (argmax ranking each slot)
    # At t=0 the env auto-selects best ranking channel; from t=1 onward we feed
    # the argmax ranking as the action
    env_switch_slots: list[int] = []
    for t in range(slots):
        best = int(np.argmax(ranking[t]))
        action = encode_action(best, 'overlay', channels)
        _, _, _, info = env.step(np.array([action]))
        if info['switch_started'][0]:
            env_switch_slots.append(t)

    assert len(env_switch_slots) >= 1, "RL env should trigger at least one switch"

    # Verify: for each switch, the offline and RL lock lengths match
    for offline_idx, env_t in zip(offline_switches, env_switch_slots):
        # Offline: next permitted switch = last_switch_slot + min_dwell
        # Find the next offline switch
        next_offline = offline_switches[offline_switches > offline_idx]
        if len(next_offline) == 0:
            continue
        offline_gap = int(next_offline[0]) - int(offline_idx)

        # RL: find the next RL switch
        next_rl = [s for s in env_switch_slots if s > env_t]
        if len(next_rl) == 0:
            continue
        rl_gap = int(next_rl[0]) - int(env_t)

        assert rl_gap >= dwell, (
            f"RL lock {rl_gap} < min_dwell {dwell} "
            f"(offline lock = {offline_gap})"
        )
        assert rl_gap == offline_gap, (
            f"RL lock {rl_gap} != offline lock {offline_gap} "
            f"at switch slot {env_t}"
        )


def test_future_dqi_uses_hidden_lookahead_and_shuffle_preserves_physics() -> None:
    base = _config(slots=6, channels=3)
    base.update({'hidden_lookahead_slots': 4, 'diagnostic_privileged': True})
    base['descriptors'] = {
        **base['descriptors'],
        'diagnostic_privileged': True,
        'theta_condition': 'neutral',
        'dqi_update_slots': 2,
        'dqi_horizon_slots': 4,
        'dqi_quantile': 0.10,
        'r_ref_mbps': 80.0,
        'compute_future_dqi_for_metrics': True,
    }
    oracle_cfg = {
        **base,
        'descriptors': {**base['descriptors'], 'dqi_condition': 'future_oracle'},
    }
    shuffled_cfg = {
        **base,
        'descriptors': {**base['descriptors'], 'dqi_condition': 'channel_shuffled'},
    }
    oracle = _resolve_trace(oracle_cfg, 'test', seed=5, slots=6, split='rl_train')
    shuffled = _resolve_trace(shuffled_cfg, 'test', seed=5, slots=6, split='rl_train')
    repeated = _resolve_trace(shuffled_cfg, 'test', seed=5, slots=6, split='rl_train')
    assert oracle.physical_trace_hash() == shuffled.physical_trace_hash()
    assert np.array_equal(oracle.future_dqi_oracle, shuffled.future_dqi_oracle)
    assert not np.array_equal(oracle.fresh_dqi, shuffled.fresh_dqi)
    assert np.array_equal(shuffled.fresh_dqi, repeated.fresh_dqi)
    assert np.all(oracle.dqi_input_cutoff_slot > oracle.dqi_source_slot)


def test_formal_loader_rejects_privileged_bundle(tmp_path) -> None:
    trace = generate_multi_vehicle_trace(_config(slots=4), seed=7, slots=4)
    frozen = replace(
        trace,
        dqi_model_id='frozen_dqi',
        prediction_model_id='frozen_theta',
        dqi_model_hash='a' * 64,
        prediction_model_hash='b' * 64,
    )
    save_descriptor_bundle(
        tmp_path,
        frozen,
        extra_manifest={'diagnostic_privileged': True},
    )
    with pytest.raises(ValueError, match='privileged'):
        load_descriptor_bundle(tmp_path, require_frozen_models=True)


def test_r1_configs_expand_to_three_conditions() -> None:
    for path in (
        'configs/e2_r1a_prediction_oracle_gate.json',
        'configs/e2_r1b_dqi_oracle_gate.json',
    ):
        config = load_config(path)
        cases = _build_cases(config, list(config['scenarios']))
        assert len(cases) == 3
        assert all(case['diagnostic_privileged'] for _, case in cases)

    positive = load_config('configs/e2_p0_prediction_positive_control.json')
    cases = _build_cases(positive, list(positive['scenarios']))
    assert len(cases) == 3
    assert positive['load_calibration']['enabled'] is False

    r2 = load_config('configs/e2_p0_prediction_positive_control_r2.json')
    r2_cases = _build_cases(r2, list(r2['scenarios']))
    assert len(r2_cases) == 4
    assert {
        case['module_variant'] for _, case in r2_cases
    } == {
        'theta_oracle',
        'theta_placebo_a',
        'theta_placebo_b',
        'theta_neutral',
    }

    bridge = load_config(
        'configs/e2_p1r_prediction_representation_bridge.json'
    )
    bridge_cases = _build_cases(bridge, list(bridge['scenarios']))
    assert len(bridge_cases) == 10


def test_prediction_r2_placebos_match_marginal_but_use_distinct_orders() -> None:
    config = load_config('configs/e2_p0_prediction_positive_control_r2.json')
    cases = {
        case['module_variant']: case
        for _, case in _build_cases(config, list(config['scenarios']))
    }
    traces = {
        name: _resolve_trace(
            case,
            'P0_M1_prediction_max',
            seed=17,
            slots=5_000,
            split='audit',
        )
        for name, case in cases.items()
        if name != 'theta_neutral'
    }
    oracle = traces['theta_oracle'].theta
    placebo_a = traces['theta_placebo_a'].theta
    placebo_b = traces['theta_placebo_b'].theta
    assert np.array_equal(np.sort(oracle, axis=0), np.sort(placebo_a, axis=0))
    assert np.array_equal(np.sort(oracle, axis=0), np.sort(placebo_b, axis=0))
    assert not np.array_equal(oracle, placebo_a)
    assert not np.array_equal(oracle, placebo_b)
    assert not np.array_equal(placebo_a, placebo_b)


def test_prediction_representation_bridge_encodings_and_placebos() -> None:
    config = load_config(
        'configs/e2_p1r_prediction_representation_bridge.json'
    )
    cases = {
        case['module_variant']: case
        for _, case in _build_cases(config, list(config['scenarios']))
    }
    traces = {
        name: _resolve_trace(
            case,
            'P0_M1_prediction_max',
            seed=17,
            slots=5_000,
            split='representation_audit',
        )
        for name, case in cases.items()
    }
    hard = traces['theta_oracle']
    expected_hard = (~hard.occupancy).astype(float)
    assert np.array_equal(hard.theta, expected_hard)

    soft = traces['theta_soft_oracle'].theta
    assert np.allclose(np.unique(soft), [0.1, 0.9])

    continuous_trace = traces['theta_continuous_oracle']
    continuous = continuous_trace.theta
    busy = continuous_trace.occupancy
    assert float(np.max(continuous[busy])) <= 0.45
    assert float(np.min(continuous[~busy])) >= 0.55
    assert np.array_equal(continuous >= 0.5, ~busy)
    assert np.unique(continuous).size / continuous.size >= 0.95

    for prefix in ('theta', 'theta_soft', 'theta_continuous'):
        oracle = traces[f'{prefix}_oracle'].theta
        placebo_a = traces[f'{prefix}_placebo_a'].theta
        placebo_b = traces[f'{prefix}_placebo_b'].theta
        assert np.array_equal(
            np.sort(oracle, axis=0),
            np.sort(placebo_a, axis=0),
        )
        assert np.array_equal(
            np.sort(oracle, axis=0),
            np.sort(placebo_b, axis=0),
        )
        assert not np.array_equal(oracle, placebo_a)
        assert not np.array_equal(oracle, placebo_b)
        assert not np.array_equal(placebo_a, placebo_b)


def test_final_checkpoint_uses_independent_long_evaluation() -> None:
    config = _config(slots=4, channels=1)
    config.update(
        {
            'experiment': 'e2_r1_oracle_gate',
            'seeds': [1],
            'agents': ['ddqn_no_esn'],
            'module_conditions': [
                {
                    'name': 'theta_neutral',
                    'descriptors': {
                        'theta_condition': 'neutral',
                        'dqi_condition': 'neutral',
                    },
                }
            ],
            'checkpoint_evaluation': {
                'checkpoint_interval': 2,
                'eval_episodes': 1,
                'eval_slots': 2,
            },
            'final_checkpoint_evaluation': {
                'eval_episodes': 2,
                'eval_slots': 3,
            },
        }
    )
    metrics, _ = run_multi_vehicle_experiment(config)
    intermediate = metrics[~metrics['is_final_checkpoint'].astype(bool)].iloc[-1]
    final = metrics[metrics['is_final_checkpoint'].astype(bool)].iloc[-1]
    assert int(intermediate['evaluation_episodes']) == 1
    assert int(intermediate['evaluation_slots']) == 2
    assert int(final['evaluation_episodes']) == 2
    assert int(final['evaluation_slots']) == 3
    assert intermediate['evaluation_physical_trace_hash'] != final[
        'evaluation_physical_trace_hash'
    ]


def test_reuse_preserves_independent_final_evaluation_context() -> None:
    config = _config(slots=4, channels=1)
    config.update(
        {
            'experiment': 'e2_r1_oracle_gate',
            'seeds': [1],
            'agents': ['ddqn_no_esn'],
            'module_conditions': [
                {
                    'name': 'theta_oracle',
                    'descriptors': {
                        'theta_condition': 'oracle',
                        'dqi_condition': 'neutral',
                    },
                }
            ],
            'diagnostic_privileged': True,
            'checkpoint_evaluation': {
                'checkpoint_interval': 2,
                'eval_episodes': 1,
                'eval_slots': 2,
            },
            'final_checkpoint_evaluation': {
                'eval_episodes': 2,
                'eval_slots': 3,
            },
        }
    )
    source, _ = run_multi_vehicle_experiment(config)
    source['_reuse_source_run'] = 'source-run'
    target_config = {
        **config,
        '_reuse_metrics': source,
        'require_reuse_cases': [
            {'module_variant': 'theta_oracle'}
        ],
    }
    reused, _ = run_multi_vehicle_experiment(target_config)
    assert reused['reused_curve'].astype(bool).all()
    assert (
        reused['evaluation_trace_hash'].astype(str).tolist()
        == source['evaluation_trace_hash'].astype(str).tolist()
    )
    source_final = source[source['is_final_checkpoint'].astype(bool)].iloc[0]
    reused_final = reused[reused['is_final_checkpoint'].astype(bool)].iloc[0]
    assert (
        reused_final['evaluation_physical_trace_hash']
        == source_final['evaluation_physical_trace_hash']
    )
    assert reused_final['reuse_source_run'] == 'source-run'


def test_prediction_gate_accepts_preregistered_synthetic_effect() -> None:
    rows = []
    for seed in (1, 2, 3):
        for condition in ('theta_oracle', 'theta_time_shuffled', 'theta_neutral'):
            oracle = condition == 'theta_oracle'
            rows.append(
                {
                    'scenario': 'r1a',
                    'arrival_probability': 0.35,
                    'agent': 'ddqn_no_esn',
                    'seed': seed,
                    'module_variant': condition,
                    'is_final_checkpoint': True,
                    'served_goodput_mbps': 10.5 if oracle else 10.0,
                    'overlay_collision_rate': 0.01 if oracle else 0.02,
                    'pu_access_violation_rate': 0.0005 if oracle else 0.01,
                    'mean_safe_action_regret_mbps': 1.0 if oracle else 2.0,
                    'overlay_given_busy_legal_backlog': 0.01 if oracle else 0.05,
                    'overlay_given_idle_legal_backlog': 0.20,
                    'train_physical_trace_hash': f'train-{seed}',
                    'evaluation_physical_trace_hash': f'eval-{seed}',
                    'raw_action_mask_hash': f'mask-{seed}',
                    'resolved_config_hash': 'abc',
                }
            )
    config = load_config('configs/e2_r1a_prediction_oracle_gate.json')
    payload, checks = evaluate_e2_r1_gate(pd.DataFrame(rows), config)
    assert payload['passed']
    assert checks['passed'].all()


def test_empty_formal_matrix_yields_clean_gate_failure() -> None:
    config = load_config('configs/e2_r1a_prediction_oracle_gate.json')
    payload, checks = evaluate_e2_r1_gate(pd.DataFrame(), config)
    assert not payload['passed']
    assert payload['stop_reason'] == 'calibration_failed_or_formal_matrix_empty'
    assert checks['check'].tolist() == ['formal_matrix_available']


def _positive_control_rows(kind: str, diagnostic: bool = False) -> pd.DataFrame:
    rows = []
    if kind == 'prediction':
        conditions = ('theta_oracle', 'theta_time_shuffled', 'theta_neutral')
    else:
        conditions = ('dqi_future_oracle', 'dqi_channel_shuffled', 'dqi_neutral')
    for seed in (1, 2, 3):
        for condition in conditions:
            oracle = 'oracle' in condition
            row = {
                'scenario': kind,
                'arrival_probability': 1.0,
                'agent': 'ddqn_no_esn',
                'seed': seed,
                'module_variant': condition,
                'is_final_checkpoint': True,
                'diagnostic_run': diagnostic,
                'run_scope': 'formal',
                'served_goodput_mbps': 10.6 if oracle else 10.0,
                'pu_access_violation_rate': 0.0005 if oracle else 0.01,
                'overlay_given_busy_legal_backlog': 0.01 if oracle else 0.05,
                'mean_channel_selection_regret': 0.5 if oracle else 1.0,
                'dqi_action_disagreement_rate': 0.2 if oracle else 0.0,
                'oracle_top1_channel_count': 3,
                'dwell_violation_count': 0,
                'train_physical_trace_hash': f'train-{seed}',
                'evaluation_physical_trace_hash': f'eval-{seed}',
                'raw_action_mask_hash': f'mask-{seed}',
                'resolved_config_hash': 'positive',
            }
            rows.append(row)
    return pd.DataFrame(rows)


@pytest.mark.parametrize(
    ('config_path', 'gate_type', 'kind'),
    [
        (
            'configs/e2_p0_prediction_positive_control.json',
            'prediction_positive_control',
            'prediction',
        ),
        (
            'configs/e2_r1b_dqi_oracle_gate.json',
            'dqi_positive_control',
            'dqi',
        ),
    ],
)
def test_positive_control_gates_have_synthetic_pass_and_diagnostic_fail(
    config_path: str,
    gate_type: str,
    kind: str,
) -> None:
    config = load_config(config_path)
    config['e2_r1_gate'] = gate_type
    payload, checks = evaluate_e2_r1_gate(
        _positive_control_rows(kind),
        config,
    )
    assert payload['passed']
    assert checks['passed'].all()

    diagnostic, diagnostic_checks = evaluate_e2_r1_gate(
        _positive_control_rows(kind, diagnostic=True),
        config,
    )
    assert not diagnostic['passed']
    formal = diagnostic_checks[
        diagnostic_checks['check'].astype(str) == 'formal_run'
    ]
    assert not bool(formal.iloc[0]['passed'])


def _prediction_r2_rows(diagnostic: bool = False) -> pd.DataFrame:
    rows = []
    conditions = (
        'theta_oracle',
        'theta_placebo_a',
        'theta_placebo_b',
        'theta_neutral',
    )
    for seed in (1, 2, 3):
        for condition in conditions:
            oracle = condition == 'theta_oracle'
            placebo = condition in {'theta_placebo_a', 'theta_placebo_b'}
            rows.append(
                {
                    'scenario': 'prediction_r2',
                    'arrival_probability': 1.0,
                    'agent': 'ddqn_no_esn',
                    'seed': seed,
                    'module_variant': condition,
                    'is_final_checkpoint': True,
                    'diagnostic_run': diagnostic,
                    'run_scope': 'formal',
                    'served_goodput_mbps': 10.6 if oracle else 10.0,
                    'pu_access_violation_rate': 0.0005 if oracle else 0.01,
                    'overlay_given_busy_legal_backlog': (
                        0.01 if oracle else 0.05
                    ),
                    'train_physical_trace_hash': f'train-{seed}',
                    'evaluation_physical_trace_hash': f'eval-{seed}',
                    'raw_action_mask_hash': f'mask-{seed}',
                    'train_theta_marginal_hash': (
                        f'marginal-train-{seed}'
                        if oracle or placebo
                        else 'neutral'
                    ),
                    'evaluation_theta_marginal_hash': (
                        f'marginal-eval-{seed}'
                        if oracle or placebo
                        else 'neutral'
                    ),
                    'train_theta_idle_correlation': (
                        1.0 if oracle else 0.01 if placebo else np.nan
                    ),
                    'evaluation_theta_idle_correlation': (
                        1.0 if oracle else -0.01 if placebo else np.nan
                    ),
                    'theta_observation_hash': f'{condition}-{seed}',
                    'resolved_config_hash': 'positive-r2',
                }
            )
    return pd.DataFrame(rows)


def test_prediction_r2_gate_has_synthetic_pass_and_audit_fail() -> None:
    config = load_config('configs/e2_p0_prediction_positive_control_r2.json')
    payload, checks = evaluate_e2_r1_gate(_prediction_r2_rows(), config)
    assert payload['passed']
    assert checks['passed'].all()

    broken = _prediction_r2_rows()
    broken.loc[
        (broken['module_variant'] == 'theta_placebo_b')
        & (broken['seed'] == 2),
        'evaluation_theta_marginal_hash',
    ] = 'mismatch'
    failed, failed_checks = evaluate_e2_r1_gate(broken, config)
    assert not failed['passed']
    marginal = failed_checks[
        failed_checks['check'] == 'placebo_marginal_distribution'
    ]
    assert not bool(marginal.iloc[0]['passed'])


def _representation_bridge_rows(
    diagnostic: bool = False,
) -> pd.DataFrame:
    representation_conditions = {
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
    for seed in (1, 2, 3):
        for representation, conditions in representation_conditions.items():
            for role, condition in enumerate(conditions):
                oracle = role == 0
                hard = representation == 'hard'
                soft = representation == 'soft'
                if hard:
                    goodput = 10.6 if oracle else 10.0
                    theta_min, theta_max = 0.0, 1.0
                    unique_count, unique_fraction = 2, 0.001
                elif soft:
                    goodput = 10.5 if oracle else 9.9
                    theta_min, theta_max = 0.1, 0.9
                    unique_count, unique_fraction = 2, 0.001
                else:
                    goodput = 10.45 if oracle else 9.85
                    theta_min, theta_max = 0.05, 0.95
                    unique_count, unique_fraction = 10_000, 1.0
                reused = hard
                rows.append(
                    {
                        'scenario': 'prediction_bridge',
                        'arrival_probability': 1.0,
                        'agent': 'ddqn_no_esn',
                        'seed': seed,
                        'module_variant': condition,
                        'is_final_checkpoint': True,
                        'diagnostic_run': diagnostic,
                        'run_scope': 'formal',
                        'served_goodput_mbps': goodput,
                        'pu_access_violation_rate': (
                            0.0005 if oracle else 0.01
                        ),
                        'overlay_given_busy_legal_backlog': (
                            0.01 if oracle else 0.05
                        ),
                        'train_physical_trace_hash': f'train-{seed}',
                        'evaluation_physical_trace_hash': f'eval-{seed}',
                        'raw_action_mask_hash': f'mask-{seed}',
                        'train_theta_marginal_hash': (
                            f'{representation}-train-{seed}'
                        ),
                        'evaluation_theta_marginal_hash': (
                            f'{representation}-eval-{seed}'
                        ),
                        'theta_observation_hash': f'{condition}-{seed}',
                        'train_theta_idle_correlation': (
                            1.0 if oracle else 0.01
                        ),
                        'evaluation_theta_idle_correlation': (
                            1.0 if oracle else -0.01
                        ),
                        'train_theta_unique_count': unique_count,
                        'evaluation_theta_unique_count': unique_count,
                        'train_theta_unique_fraction': unique_fraction,
                        'evaluation_theta_unique_fraction': unique_fraction,
                        'train_theta_min': theta_min,
                        'evaluation_theta_min': theta_min,
                        'train_theta_max': theta_max,
                        'evaluation_theta_max': theta_max,
                        'train_theta_busy_max': (
                            0.0 if hard else 0.1 if soft else 0.45
                        ),
                        'evaluation_theta_busy_max': (
                            0.0 if hard else 0.1 if soft else 0.45
                        ),
                        'train_theta_idle_min': (
                            1.0 if hard else 0.9 if soft else 0.55
                        ),
                        'evaluation_theta_idle_min': (
                            1.0 if hard else 0.9 if soft else 0.55
                        ),
                        'train_theta_threshold_accuracy': (
                            1.0 if oracle else 0.5
                        ),
                        'evaluation_theta_threshold_accuracy': (
                            1.0 if oracle else 0.5
                        ),
                        'reused_curve': reused,
                        'reuse_source_run': (
                            'p0-r2-run' if reused else ''
                        ),
                        'reuse_source_resolved_config_hash': (
                            '55a19b96cda05bf4' if reused else ''
                        ),
                        'reuse_source_e2_r1_gate': (
                            'prediction_positive_control_r2'
                            if reused
                            else ''
                        ),
                        'resolved_config_hash': 'bridge',
                    }
                )
        rows.append(
            {
                **rows[-1],
                'module_variant': 'theta_neutral',
                'served_goodput_mbps': 9.5,
                'pu_access_violation_rate': 0.02,
                'overlay_given_busy_legal_backlog': 0.1,
                'train_theta_marginal_hash': 'neutral-train',
                'evaluation_theta_marginal_hash': 'neutral-eval',
                'theta_observation_hash': f'neutral-{seed}',
                'train_theta_idle_correlation': np.nan,
                'evaluation_theta_idle_correlation': np.nan,
                'train_theta_unique_count': 1,
                'evaluation_theta_unique_count': 1,
                'train_theta_unique_fraction': 0.001,
                'evaluation_theta_unique_fraction': 0.001,
                'train_theta_min': 0.5,
                'evaluation_theta_min': 0.5,
                'train_theta_max': 0.5,
                'evaluation_theta_max': 0.5,
                'train_theta_threshold_accuracy': 0.5,
                'evaluation_theta_threshold_accuracy': 0.5,
                'reused_curve': True,
                'reuse_source_run': 'p0-r2-run',
                'reuse_source_resolved_config_hash': '55a19b96cda05bf4',
                'reuse_source_e2_r1_gate': (
                    'prediction_positive_control_r2'
                ),
            }
        )
    return pd.DataFrame(rows)


def _bridge_check(
    checks: pd.DataFrame,
    name: str,
) -> bool:
    selected = checks[checks['check'] == name]
    return bool(selected.iloc[0]['passed'])


def test_representation_bridge_gate_synthetic_pass_and_failures() -> None:
    config = load_config(
        'configs/e2_p1r_prediction_representation_bridge.json'
    )
    payload, checks = evaluate_e2_r1_gate(
        _representation_bridge_rows(),
        config,
    )
    assert payload['passed']
    assert checks['passed'].all()

    diagnostic, diagnostic_checks = evaluate_e2_r1_gate(
        _representation_bridge_rows(diagnostic=True),
        config,
    )
    assert not diagnostic['passed']
    assert not _bridge_check(diagnostic_checks, 'formal_run')

    broken_reuse = _representation_bridge_rows()
    broken_reuse.loc[
        broken_reuse['module_variant'] == 'theta_oracle',
        'reuse_source_resolved_config_hash',
    ] = 'wrong'
    payload, checks = evaluate_e2_r1_gate(broken_reuse, config)
    assert not payload['passed']
    assert not _bridge_check(checks, 'hard_reuse_provenance')

    broken_range = _representation_bridge_rows()
    broken_range.loc[
        broken_range['module_variant'] == 'theta_continuous_oracle',
        'evaluation_theta_busy_max',
    ] = 0.51
    payload, checks = evaluate_e2_r1_gate(broken_range, config)
    assert not payload['passed']
    assert not _bridge_check(checks, 'continuous_schema')

    broken_alignment = _representation_bridge_rows()
    broken_alignment.loc[
        broken_alignment['module_variant']
        == 'theta_soft_placebo_a',
        'train_theta_idle_correlation',
    ] = 0.2
    payload, checks = evaluate_e2_r1_gate(broken_alignment, config)
    assert not payload['passed']
    assert not _bridge_check(checks, 'placebo_alignment_broken')

    broken_effect = _representation_bridge_rows()
    broken_effect.loc[
        broken_effect['module_variant'] == 'theta_soft_oracle',
        'served_goodput_mbps',
    ] = 9.91
    payload, checks = evaluate_e2_r1_gate(broken_effect, config)
    assert not payload['passed']
    assert not _bridge_check(checks, 'soft_goodput_effect')

    broken_noninferiority = _representation_bridge_rows()
    broken_noninferiority.loc[
        broken_noninferiority['module_variant']
        == 'theta_continuous_oracle',
        'served_goodput_mbps',
    ] = 10.0
    payload, checks = evaluate_e2_r1_gate(
        broken_noninferiority,
        config,
    )
    assert not payload['passed']
    assert not _bridge_check(checks, 'representation_noninferiority')
