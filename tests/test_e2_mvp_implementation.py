from __future__ import annotations

import numpy as np
import pandas as pd

from cvn_rl.experiments.config import load_config
from cvn_rl.experiments.e2_mvp import evaluate_e2_mvp_gate
from cvn_rl.experiments.multi_vehicle_runner import _build_cases
from cvn_rl.multi_metrics import summarize_multi_vehicle_records
from cvn_rl.simulation.actions import build_action_mask, encode_action
from cvn_rl.simulation.multi_env import MultiVehicleAccessEnv
from cvn_rl.simulation.multi_traces import generate_multi_vehicle_trace
from cvn_rl.simulation.state_builder import feature_slices


def _small_config() -> dict:
    config = load_config('configs/base_multi_vehicle.json')
    config.update(
        {
            'num_channels': 2,
            'num_vehicles': 1,
            'slots': 3,
            'protection_variant': 'gate_state_mask',
        }
    )
    config['packet_traffic'] = {
        'arrival_probability': 1.0,
        'queue_capacity_packets': 10,
        'packet_size_bits': 1_000,
    }
    return config


def test_legacy_mask_is_default_and_revised_underlay_ignores_theta() -> None:
    sensing = np.array([1.0])
    theta = np.array([0.1])
    power = np.array([0.5])
    legacy = build_action_mask(sensing, theta, power, ['action_mask'])
    explicit_legacy = build_action_mask(
        sensing,
        theta,
        power,
        ['action_mask'],
        {'underlay_theta_min': 0.2},
    )
    revised = build_action_mask(
        sensing,
        theta,
        power,
        ['action_mask'],
        {'underlay_theta_min': None},
    )
    underlay = encode_action(0, 'underlay', 1)
    assert np.array_equal(legacy, explicit_legacy)
    assert not legacy[underlay]
    assert revised[underlay]


def test_information_and_mask_sources_are_independent() -> None:
    config = _small_config()
    trace = generate_multi_vehicle_trace(config, seed=21, slots=3)
    trace.theta[:] = 0.1
    trace.sensing_idle[:] = 1.0
    trace.power_cap[:] = 0.5
    common = {
        **config,
        'mask_policy_version': 'e2_revised_v2',
        'action_mask_thresholds': {'underlay_theta_min': None},
    }
    full = MultiVehicleAccessEnv(
        trace,
        {
            **common,
            'module_variant': 'revised_full',
            'ablation_info_source': 'real',
            'ablation_mask_source': 'real',
        },
    )
    info_only = MultiVehicleAccessEnv(
        trace,
        {
            **common,
            'module_variant': 'no_prediction_info',
            'ablation_info_source': 'neutral',
            'ablation_mask_source': 'real',
        },
    )
    mask_only = MultiVehicleAccessEnv(
        trace,
        {
            **common,
            'module_variant': 'prediction_mask_ablation',
            'ablation_info_source': 'real',
            'ablation_mask_source': 'neutral',
        },
    )
    full_obs, full_info = full.reset()
    info_obs, info_info = info_only.reset()
    mask_obs, mask_info = mask_only.reset()
    slices = feature_slices(trace.num_channels, full.state_features)
    assert np.array_equal(
        full_info[0]['raw_action_mask'],
        info_info[0]['raw_action_mask'],
    )
    assert not np.array_equal(
        full_obs[:, slices['theta']],
        info_obs[:, slices['theta']],
    )
    assert np.array_equal(
        full_obs[:, slices['theta']],
        mask_obs[:, slices['theta']],
    )
    assert not np.array_equal(
        full_info[0]['raw_action_mask'],
        mask_info[0]['raw_action_mask'],
    )


def test_served_goodput_and_completion_accounting() -> None:
    config = _small_config()
    trace = generate_multi_vehicle_trace(config, seed=22, slots=3)
    trace.packet_arrivals[:] = True
    trace.occupancy[:] = False
    trace.quality_mbps[:] = 100.0
    env = MultiVehicleAccessEnv(trace, config)
    env.reset()
    records = []
    done = False
    action = encode_action(0, 'overlay', trace.num_channels)
    while not done:
        _, _, done, record = env.step(np.array([action]))
        records.append(record)
    summary = summarize_multi_vehicle_records(records, {}, config)
    served = sum(float(np.sum(record['served_bits'])) for record in records)
    assert np.isclose(summary['total_served_bits'], served)
    assert np.isclose(
        summary['served_goodput_mbps'],
        served / (3 * config['slot_duration_ms'] / 1_000) / 1_000_000,
    )
    assert (
        summary['completed_packets']
        + summary['dropped_packets']
        + summary['unfinished_packets']
        == summary['generated_packets']
    )
    assert np.isclose(
        summary['completion_rate'] + summary['noncompletion_rate'],
        1.0,
    )


def test_mvp_configs_expand_to_preregistered_curve_budget() -> None:
    expected_case_counts = {
        'configs/e2_mvp_phase_a.json': 4,
        'configs/e2_mvp_phase_c.json': 16,
        'configs/e2_mvp_phase_c_fallback_0p30.json': 24,
        'configs/e2_mvp_phase_c_fallback_0p40.json': 24,
        'configs/e2_mvp_phase_d_esn.json': 8,
    }
    for path, expected in expected_case_counts.items():
        config = load_config(path)
        cases = _build_cases(config, list(config['scenarios']))
        assert len(cases) == expected
        assert len(cases) * len(config['seeds']) <= 72
    phase_c = load_config('configs/e2_mvp_phase_c.json')
    assert len(phase_c['require_reuse_cases']) * len(phase_c['seeds']) == 6
    for probability in ('0p30', '0p40'):
        fallback = load_config(
            f'configs/e2_mvp_phase_c_fallback_{probability}.json'
        )
        assert fallback['required_gate_field'] == 'structure_passed'
    phase_d = load_config('configs/e2_mvp_phase_d_esn.json')
    assert phase_d.get('required_gate_field', 'passed') == 'passed'


def test_phase_a_gate_accepts_structurally_valid_synthetic_metrics() -> None:
    rows = []
    variants = [
        'legacy_full', 'revised_full', 'no_prediction_info',
        'prediction_mask_ablation',
    ]
    metadata = {
        'legacy_full': ('real', 'real', 'legacy_v1'),
        'revised_full': ('real', 'real', 'e2_revised_v2'),
        'no_prediction_info': ('neutral', 'real', 'e2_revised_v2'),
        'prediction_mask_ablation': ('real', 'neutral', 'e2_revised_v2'),
    }
    for seed in (1, 2, 3):
        for variant in variants:
            info_source, mask_source, version = metadata[variant]
            rows.append(
                {
                    'scenario': 'M3_high_pu',
                    'arrival_probability': 0.2,
                    'seed': seed,
                    'agent': 'ddqn_no_esn',
                    'module_variant': variant,
                    'ablation_info_source': info_source,
                    'ablation_mask_source': mask_source,
                    'mask_policy_version': version,
                    'evaluation_variant': 'nominal',
                    'is_final_checkpoint': True,
                    'checkpoint_slot': 30_000,
                    'raw_action_mask_hash': (
                        'real-mask' if mask_source == 'real' else 'neutral-mask'
                    ),
                    'theta_observation_hash': (
                        'real-theta' if info_source == 'real' else 'neutral-theta'
                    ),
                    'train_trace_hash': f'train-{seed}',
                    'evaluation_trace_hash': f'eval-{seed}',
                    'busy_underlay_allowed_rate': 0.1 if variant == 'legacy_full' else 0.8,
                    'pu_active_violation_rate': 0.0,
                    'interference_ccdf_gt_1p0': 0.0,
                    'served_goodput_mbps': 10.0,
                }
            )
        base_revised = dict(rows[-3])
        for checkpoint, goodput in ((10_000, 9.9), (20_000, 10.0)):
            row = dict(base_revised)
            row.update(
                {
                    'module_variant': 'revised_full',
                    'ablation_info_source': 'real',
                    'ablation_mask_source': 'real',
                    'is_final_checkpoint': False,
                    'checkpoint_slot': checkpoint,
                    'served_goodput_mbps': goodput,
                }
            )
            rows.append(row)
    metrics = pd.DataFrame(rows)
    payload, _ = evaluate_e2_mvp_gate(metrics, 'A')
    assert payload['passed']
    assert payload['structure_passed']
    assert payload['safety_passed']
    assert payload['performance_passed']
    assert not payload['weak_performance_signal']

    weak = metrics.copy()
    weak.loc[
        weak['module_variant'].eq('revised_full')
        & weak['is_final_checkpoint'].astype(bool),
        'served_goodput_mbps',
    ] = 5.0
    weak_payload, _ = evaluate_e2_mvp_gate(weak, 'A')
    assert weak_payload['passed']
    assert weak_payload['structure_passed']
    assert weak_payload['safety_passed']
    assert not weak_payload['performance_passed']
    assert weak_payload['weak_performance_signal']


def test_phase_c_gate_accepts_complete_internal_validity_matrix() -> None:
    rows = []
    sources = {
        'full': ('real', 'real', 't-real', 'd-real'),
        'no_dqi': ('real', 'real', 't-real', 'd-neutral'),
        'no_prediction_info': ('neutral', 'real', 't-neutral', 'd-real'),
        'state_sensing_only': ('neutral', 'real', 't-neutral', 'd-neutral'),
    }
    for scenario in ('M3_high_pu', 'M10_low_pu'):
        for load in (0.20, 0.35):
            for seed in (1, 2, 3):
                for variant, values in sources.items():
                    info_source, mask_source, theta_hash, dqi_hash = values
                    goodput = 10.0
                    if variant == 'no_dqi':
                        goodput = 8.0 if scenario == 'M10_low_pu' else 9.0
                    elif variant == 'no_prediction_info':
                        goodput = 8.0 if scenario == 'M3_high_pu' else 9.0
                    elif variant == 'state_sensing_only':
                        goodput = 7.0
                    rows.append(
                        {
                            'scenario': scenario,
                            'arrival_probability': load,
                            'seed': seed,
                            'agent': 'ddqn_no_esn',
                            'module_variant': variant,
                            'ablation_info_source': info_source,
                            'ablation_mask_source': mask_source,
                            'mask_policy_version': 'e2_revised_v2',
                            'evaluation_variant': 'nominal',
                            'is_final_checkpoint': True,
                            'raw_action_mask_hash': f'mask-{scenario}-{load}-{seed}',
                            'theta_observation_hash': theta_hash,
                            'dqi_observation_hash': dqi_hash,
                            'train_trace_hash': f'train-{scenario}-{load}-{seed}',
                            'evaluation_trace_hash': f'eval-{scenario}-{load}-{seed}',
                            'served_goodput_mbps': goodput,
                            'completion_rate': 0.90,
                            'noncompletion_rate': 0.10 if load == 0.35 else 0.02,
                            'pu_active_violation_rate': 0.0,
                            'mean_queue_occupancy': 0.4,
                        }
                    )
    payload, _ = evaluate_e2_mvp_gate(pd.DataFrame(rows), 'C')
    assert payload['passed']
    assert payload['structure_passed']
    assert payload['science_passed']
    assert payload['load_decision'] == 'accept_0.35'

    fallback_rows = []
    for row in rows:
        if np.isclose(float(row['arrival_probability']), 0.35):
            fallback = dict(row)
            fallback['arrival_probability'] = 0.30
            fallback_rows.append(fallback)
    fallback_payload, fallback_checks = evaluate_e2_mvp_gate(
        pd.DataFrame([*rows, *fallback_rows]),
        'C',
    )
    assert fallback_payload['passed']
    assert fallback_payload['fallback_probe']['probability'] == 0.30
    assert fallback_payload['fallback_probe']['assessment'] == 'accepted'
    assert fallback_payload['fallback_probe']['search_exhausted']
    assert not fallback_payload['fallback_probe']['further_search_allowed']
    assert len(fallback_payload['fallback_probe']['condition_summary']) == 8
    assert fallback_checks.set_index('check').loc['fallback_matrix', 'passed']
