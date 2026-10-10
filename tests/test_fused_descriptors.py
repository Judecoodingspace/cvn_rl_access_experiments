from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from cvn_rl.experiments.config import load_config
from cvn_rl.experiments.multi_vehicle_runner import _resolve_trace
from cvn_rl.simulation.actions import encode_action
from cvn_rl.simulation.descriptor_bundle import (
    load_descriptor_bundle,
    save_descriptor_bundle,
    validate_descriptor_causality,
)
from cvn_rl.simulation.multi_env import MultiVehicleAccessEnv
from cvn_rl.simulation.multi_traces import generate_multi_vehicle_trace
from cvn_rl.simulation.state_builder import feature_slices


def _config(channels: int = 2, vehicles: int = 2, slots: int = 4) -> dict:
    config = load_config('configs/base_multi_vehicle.json')
    config.update({'num_channels': channels, 'num_vehicles': vehicles, 'slots': slots})
    config['packet_traffic'] = {
        **config['packet_traffic'],
        'arrival_probability': 1.0,
        'packet_size_bits': 1_000,
    }
    return config


def test_generated_descriptors_are_causal() -> None:
    trace = generate_multi_vehicle_trace(_config(), seed=3, slots=8)
    validate_descriptor_causality(trace)
    assert np.array_equal(trace.theta_target_slot, np.arange(8))
    assert np.all(trace.theta_input_cutoff_slot < trace.theta_target_slot)
    assert np.all(trace.dqi_source_slot <= np.arange(8))
    assert trace.dqi_model_hash.startswith('surrogate-')


def test_descriptor_bundle_round_trip(tmp_path) -> None:
    trace = generate_multi_vehicle_trace(_config(), seed=5, slots=6, split='rl_eval_0')
    manifest = save_descriptor_bundle(tmp_path, trace)
    loaded = load_descriptor_bundle(tmp_path)
    assert manifest['trace_hash'] == trace.trace_hash()
    assert loaded.trace_hash() == trace.trace_hash()
    assert loaded.split == 'rl_eval_0'
    assert np.array_equal(loaded.contention_priority, trace.contention_priority)
    with pytest.raises(ValueError, match='frozen upstream model hash'):
        load_descriptor_bundle(tmp_path, require_frozen_models=True)


def test_module_masking_keeps_dimensions_and_physical_reward() -> None:
    config = _config()
    trace = generate_multi_vehicle_trace(config, seed=7, slots=4)
    trace.occupancy[:] = False
    trace.quality_mbps[:] = 100.0
    full_env = MultiVehicleAccessEnv(trace, {**config, 'module_variant': 'full'})
    no_dqi_env = MultiVehicleAccessEnv(trace, {**config, 'module_variant': 'no_dqi'})
    full_obs, _ = full_env.reset()
    no_dqi_obs, _ = no_dqi_env.reset()
    assert full_obs.shape == no_dqi_obs.shape
    slices = feature_slices(trace.num_channels, full_env.state_features)
    assert np.allclose(no_dqi_obs[:, slices['fresh_dqi']], 0.5)
    actions = np.asarray([
        encode_action(0, 'overlay', 2),
        encode_action(1, 'overlay', 2),
    ])
    _, full_reward, _, _ = full_env.step(actions)
    _, no_dqi_reward, _, _ = no_dqi_env.step(actions)
    assert np.allclose(full_reward, no_dqi_reward)


def test_contention_loser_does_not_create_interference() -> None:
    config = _config(channels=1, vehicles=2, slots=2)
    trace = generate_multi_vehicle_trace(config, seed=9, slots=2)
    trace.occupancy[:] = True
    trace.su_pu_gain_true[0, :, 0] = [1.0, 100.0]
    trace.transmit_power_cap_mw[0, :, 0] = 0.1
    trace.contention_priority[0, :, 0] = [0.0, 1.0]
    env = MultiVehicleAccessEnv(trace, config)
    env.reset()
    action = encode_action(0, 'underlay', 1)
    _, _, _, info = env.step(np.asarray([action, action]))
    assert np.isclose(info['actual_interference_mw'][0], 0.1)
    assert info['transmitted'].tolist() == [True, False]


def test_protection_variants_change_state_and_action_mask() -> None:
    config = _config(channels=2, vehicles=1, slots=2)
    trace = generate_multi_vehicle_trace(config, seed=13, slots=2)
    trace.sensing_idle[:] = 0.0
    trace.theta[:] = 0.0
    trace.power_cap[:] = 0.0
    gate = MultiVehicleAccessEnv(trace, {**config, 'protection_variant': 'gate_only'})
    state = MultiVehicleAccessEnv(
        trace,
        {**config, 'protection_variant': 'gate_plus_state'},
    )
    masked = MultiVehicleAccessEnv(
        trace,
        {**config, 'protection_variant': 'gate_state_mask'},
    )
    gate_obs, gate_info = gate.reset()
    state_obs, state_info = state.reset()
    masked_obs, masked_info = masked.reset()
    assert gate_obs.shape[1] < state_obs.shape[1]
    assert state_obs.shape == masked_obs.shape
    assert gate_info[0]['action_mask'].all()
    assert state_info[0]['action_mask'].all()
    assert masked_info[0]['action_mask'].tolist() == [True, False, False, False, False]


def test_formal_descriptor_bundle_accepts_sha256_and_rejects_diagnostic(tmp_path) -> None:
    trace = generate_multi_vehicle_trace(_config(), seed=19, slots=4, split='rl_train')
    trace = replace(
        trace,
        dqi_model_id='gru_core_mse_v20',
        prediction_model_id='attention_lstm_h50_e80',
        dqi_model_hash='a' * 64,
        prediction_model_hash='b' * 64,
    )
    save_descriptor_bundle(
        tmp_path,
        trace,
        extra_manifest={
            'diagnostic_surrogate': False,
            'validation_type': 'hybrid_berlin_link_synthetic_pu',
        },
    )
    loaded = load_descriptor_bundle(tmp_path, require_frozen_models=True)
    assert loaded.dqi_model_hash == 'a' * 64


def test_formal_bundle_rejects_future_leakage_and_split_mismatch(tmp_path) -> None:
    trace = generate_multi_vehicle_trace(_config(), seed=23, slots=4, split='rl_train')
    leaking = replace(
        trace,
        theta_input_cutoff_slot=trace.theta_target_slot.copy(),
    )
    with pytest.raises(ValueError, match='prediction input'):
        validate_descriptor_causality(leaking)

    trace = replace(
        trace,
        dqi_model_id='gru_core_mse_v20',
        prediction_model_id='attention_lstm_h50_e80',
        dqi_model_hash='c' * 64,
        prediction_model_hash='d' * 64,
    )
    save_descriptor_bundle(
        tmp_path,
        trace,
        extra_manifest={'diagnostic_surrogate': False},
    )
    config = _config()
    config['descriptors'] = {
        'source': 'artifact',
        'require_frozen_models': True,
        'bundle_path_template': str(tmp_path),
    }
    with pytest.raises(ValueError, match='descriptor split mismatch'):
        _resolve_trace(
            config,
            'test',
            seed=23,
            slots=4,
            split='rl_eval_0',
        )
