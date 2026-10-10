from copy import deepcopy
import numpy as np
import pytest
from cvn_rl.experiments.r5c2_v2.protocol import ROOT, read_json
from cvn_rl.experiments.r5c2_v2.physics_spec import validate as validate_r5c2
from cvn_rl.simulation.revision_physics_spec import validate_physics
from cvn_rl.simulation.revision_physics import generate, umi_los, umi_nlos
from cvn_rl.simulation.revision_scenarios import (
    observation_variant, component_hashes, scenario_entry, build_frozen_evaluation_case,
)


@pytest.fixture
def entry():
    return read_json(ROOT / 'configs/r2_r05c02_dqi_sensitivity_v2.json')


@pytest.mark.parametrize('patch', [
    {'protection_margin': 3.},
    {'position_error_std_m': 5.},
    {'position_error_std_m': 20., 'position_error_distribution': 'uniform'},
    {'estimated_gain_bias_db': -3.},
    {'estimated_gain_error_std_db': 3.},
    {'additional_report_delay_slots': 50},
    {'estimator_pathloss': 'UMi_NLOS_Table7.4.1-1'},
])
def test_observation_errors_preserve_true_world(entry, patch):
    base = generate(entry, 101, 240, 10, 'test')
    original = base.physical_hash()
    variant, altered = observation_variant(base, entry, patch)
    a, b = component_hashes(base), component_hashes(variant)
    assert a['truth_hash'] == b['truth_hash']
    assert a['exogenous_hash'] == b['exogenous_hash']
    assert a['observation_hash'] != b['observation_hash']
    assert base.physical_hash() == original
    replay = generate(altered, 101, 240, 10, 'test')
    for key, value in variant.arrays.items():
        np.testing.assert_array_equal(value, replay.arrays[key], err_msg=key)


def test_underestimate_and_margin_effect(entry):
    base = generate(entry, 102, 200, 10, 'test')
    low, _ = observation_variant(base, entry, {'estimated_gain_bias_db': -6.})
    safe, _ = observation_variant(base, entry, {'protection_margin': 4.})
    np.testing.assert_allclose(low.arrays['gain_pu_estimated'], base.arrays['gain_pu_estimated'] * 10 ** (-.6))
    assert np.all(low.arrays['cap_mw'] >= base.arrays['cap_mw'])
    assert np.all(safe.arrays['cap_mw'] <= base.arrays['cap_mw'])


@pytest.mark.parametrize('override', [{'shadow_sigma_db': 6.}, {'rician_k_db': 0.}, {'small_scale_fading': 'rayleigh'}])
def test_pu_uncertainty_does_not_change_bs_service_source(entry, override):
    base = generate(entry, 103, 200, 10, 'test')
    changed = scenario_entry(entry, {'physics': {'link_overrides': {'vt_pu': override}}})
    variant = generate(changed, 103, 200, 10, 'test')
    for key in ['gain_bs_true', 'rate_mbps', 'snr_db', 'rss_dbm', 'gain_pu_estimated', 'cap_mw',
                'occupancy', 'sensing_idle', 'arrivals', 'contention_priority']:
        np.testing.assert_array_equal(base.arrays[key], variant.arrays[key], err_msg=key)
    assert not np.array_equal(base.arrays['gain_pu_true'], variant.arrays['gain_pu_true'])


def test_common_validation_and_contract_are_separate(entry):
    changed = deepcopy(entry['physics']); changed['shadow_sigma_db'] = 6.
    validate_physics(changed)
    with pytest.raises(ValueError, match='paper fc/K/shadow'):
        validate_r5c2(changed)
    changed['protection_margin'] = np.nan
    with pytest.raises(ValueError):
        validate_physics(changed)


def test_los_nlos_formulas_and_breakpoint():
    fc = 5.9e9; bp = 4 * 9 * .5 * fc / 3e8
    d = np.array([100., bp, 1000.]); d3 = np.sqrt(d ** 2 + 8.5 ** 2)
    los = umi_los(d, d3, fc); nlos = umi_nlos(d, d3, fc)
    expected = 22.4 + 35.3 * np.log10(d3) + 21.3 * np.log10(5.9)
    np.testing.assert_allclose(nlos, np.maximum(los, expected))
    first = 32.4 + 21 * np.log10(d3[1]) + 20 * np.log10(5.9)
    second = 32.4 + 40 * np.log10(d3[1]) + 20 * np.log10(5.9) - 9.5 * np.log10(bp ** 2 + 8.5 ** 2)
    assert first == pytest.approx(second)
    assert np.all(nlos >= los)


def test_speed_schedule_and_frozen_normalizer(entry):
    changed = scenario_entry(entry, {'physics': {'speed_schedule': [{'start_slot': 100, 'speed_kmh': 90.}]}})
    b = generate(changed, 104, 240, 10, 'test')
    np.testing.assert_allclose(np.abs(b.arrays['velocity_m_s'][99, :, 0]), 60 / 3.6)
    np.testing.assert_allclose(np.abs(b.arrays['velocity_m_s'][100, :, 0]), 90 / 3.6)
    np.testing.assert_allclose(np.abs(np.diff(b.arrays['position_m'][:, :, 0], axis=0)),
                               np.abs(b.arrays['velocity_m_s'][:-1, :, 0]) * .001)
    assert changed['physics']['reference_max_speed_kmh'] == entry['physics']['reference_max_speed_kmh']


def test_ood_case_keeps_frozen_artifacts_and_training_entry(entry):
    references = {k: 'a' * 64 for k in ['policy_sha256', 'dqi_model_sha256', 'occupancy_model_sha256', 'preprocessing_sha256']}
    references['dqi_reference_mbps'] = 20.
    original = deepcopy(entry)
    case = build_frozen_evaluation_case(entry, {'traffic': {'sensing_pd': .7, 'p01': .3},
                                              'packet_traffic': {'arrival_probability': .8}}, references,
                                        105, 200, 10, 'ood_example')
    assert entry == original
    assert case.manifest['training_distribution_hash'] != case.manifest['evaluation_distribution_hash']
    case.assert_frozen_references(references)
    bad = dict(references, dqi_reference_mbps=30.)
    with pytest.raises(ValueError):
        case.assert_frozen_references(bad)
    with pytest.raises(TypeError):
        case.references['dqi_reference_mbps'] = 30.
    assert not case.manifest['controller_training_performed']


def test_generic_dimensions_and_unknown_variants(entry):
    e = deepcopy(entry); e['num_channels'] = 3
    b = generate(e, 106, 20, 2, 'test')
    assert b.trace().quality_mbps.shape == (20, 2, 3)
    with pytest.raises(ValueError):
        observation_variant(b, e, {'shadow_sigma_db': 8.})
    with pytest.raises(ValueError):
        scenario_entry(entry, {'predictor_recipe': {'epochs': 1}})


def test_public_protocol_does_not_require_private_audit(monkeypatch, tmp_path):
    from cvn_rl.experiments.r5c2_v2 import protocol
    monkeypatch.setattr(protocol, 'protection_snapshot_path', lambda: tmp_path / 'not_published.json')
    assert protocol.check_protected()['mode'] == 'repository_only_no_author_snapshot'
    p = protocol.resolve('configs/r2_r05c02_dqi_sensitivity_v2.json')
    assert 'src/cvn_rl/simulation/revision_physics.py' in p['source_hashes']
    assert 'src/cvn_rl/agents/shared_multi.py' in p['source_hashes']
