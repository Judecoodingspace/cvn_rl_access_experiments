import numpy as np
import pytest

from r5c2_fixtures import small_protocol
from cvn_rl.experiments.r5c2_data import prepare_data, prepared_training_caches
from cvn_rl.experiments.r5c2_protocol import condition_matrix, bundle_root
from cvn_rl.simulation.dqi_labels import compute_unnormalized_dqi
from cvn_rl.predictors.dqi_features import build_dqi_feature_tensor
from cvn_rl.simulation.multi_traces import generate_multi_vehicle_trace


def test_selected_indices_match_canonical_quantiles():
    rates = np.random.default_rng(42).uniform(1, 30, (40, 5, 2))
    indices = np.array([3, 8, 12])
    dense, _ = compute_unnormalized_dqi(rates, horizon_samples=10, vehicle_quantile=.05, time_quantile=.2)
    sparse, actual = compute_unnormalized_dqi(rates, horizon_samples=10, vehicle_quantile=.05,
                                              time_quantile=.2, source_indices=indices)
    assert np.allclose(sparse, dense[indices])
    assert np.array_equal(actual, indices)
    a, _ = compute_unnormalized_dqi(rates, horizon_samples=10, vehicle_quantile=.2, time_quantile=.05)
    assert not np.allclose(a[indices], sparse)


def test_prepared_targets_and_reference_are_shared_inputs(tmp_path):
    protocol, _ = small_protocol(tmp_path)
    manifest = prepare_data(protocol)
    assert len(manifest['targets']) == 11
    assert len(condition_matrix()) == 14
    assert len(manifest['entries']) == 3
    assert manifest['history_slots'] == 200
    for tid in manifest['targets']:
        caches, _, target = prepared_training_caches(bundle_root(protocol) / 'data', tid, 320)
        train = caches['train'][0]
        assert len(train.sample_slots) == 12
        z = np.load(bundle_root(protocol) / 'data/targets' / tid / 'train/seed_1001.npy')
        assert np.isclose(target['r_ref_mbps'], np.quantile(z, .95), rtol=1e-6)
        assert np.allclose(train.labels[train.sample_slots], np.clip(z / target['r_ref_mbps'], 0, 1))
    assert prepare_data(protocol, resume=True)['protocol_hash'] == protocol['protocol_hash']


def test_fixed_scaling_is_causal_to_future_changes():
    cfg = dict(num_vehicles=15, num_channels=10, slots=40, hidden_lookahead_slots=0,
               descriptors={'skip_causal_dqi': True}, channel={'fixed_bandwidth_mhz': 10})
    trace = generate_multi_vehicle_trace(cfg, seed=42)
    scale = dict(quality_mbps=20, bandwidth_mhz=10)
    before = build_dqi_feature_tensor(trace, physical_scaling=scale).copy()
    trace.quality_mbps[20:] *= 100
    trace.bandwidth_mhz[20:] *= 2
    after = build_dqi_feature_tensor(trace, physical_scaling=scale)
    assert np.array_equal(before[:20], after[:20])


def test_formal_protocol_rejects_short_dataset(tmp_path):
    from cvn_rl.experiments.r5c2_protocol import read_json, write_json, resolve_protocol
    _, path = small_protocol(tmp_path)
    entry = read_json(path)
    entry['execution_scope'] = 'formal'
    write_json(path, entry)
    with pytest.raises(ValueError):
        resolve_protocol(path)
