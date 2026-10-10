from __future__ import annotations

import numpy as np

from cvn_rl.experiments.config import load_config
from cvn_rl.simulation.descriptor_bundle import (
    load_descriptor_bundle,
    save_descriptor_bundle,
)
from cvn_rl.simulation.dqi_labels import (
    channel_descriptors,
    compute_unnormalized_dqi,
    fit_feature_scaler,
    fit_reference_rate,
    latency_constrained_goodput,
)
from cvn_rl.simulation.multi_env import MultiVehicleAccessEnv
from cvn_rl.simulation.multi_traces import generate_multi_vehicle_trace


def test_paper_dqi_uses_vehicle_then_future_time_quantiles() -> None:
    rng = np.random.default_rng(4)
    rate = rng.uniform(1.0, 100.0, size=(8, 4, 3))
    z, sources = compute_unnormalized_dqi(
        rate,
        horizon_samples=3,
        vehicle_quantile=0.10,
        time_quantile=0.10,
    )
    goodput = latency_constrained_goodput(rate)
    expected = np.empty_like(z)
    for source in sources:
        for channel in range(3):
            per_time = [
                np.quantile(
                    goodput[t, :, channel],
                    0.10,
                    method='linear',
                )
                for t in range(source + 1, source + 4)
            ]
            expected[source, channel] = np.quantile(
                per_time,
                0.10,
                method='linear',
            )
    assert z.shape == (5, 3)
    assert np.allclose(z, expected, atol=1e-12)


def test_reference_rate_and_scaler_ignore_nontraining_arrays() -> None:
    train_z = [np.arange(1.0, 21.0).reshape(10, 2)]
    reference = fit_reference_rate(train_z)
    validation = np.full((10, 2), 1e9)
    assert fit_reference_rate(train_z) == reference
    assert validation.max() > reference

    train = [np.arange(80.0).reshape(5, 4, 4)]
    scaler = fit_feature_scaler(train)
    test = np.full((5, 4, 4), 1e9)
    repeated = fit_feature_scaler(train)
    assert np.array_equal(scaler.mean, repeated.mean)
    assert np.array_equal(scaler.scale, repeated.scale)
    assert test.max() > scaler.mean.max()


def test_trace_bundle_and_environment_use_shared_channel_dqi(tmp_path) -> None:
    config = load_config('configs/base_multi_vehicle.json')
    config.update(
        {
            'num_channels': 3,
            'num_vehicles': 4,
            'slots': 8,
        }
    )
    trace = generate_multi_vehicle_trace(config, seed=9, slots=8)
    assert trace.fresh_dqi.shape == (8, 3)
    assert trace.future_dqi_oracle.shape == (8, 3)
    assert trace.bandwidth_mhz.shape == (8, 3)
    assert trace.snr_db.shape == (8, 4, 3)
    assert trace.rss_dbm.shape == (8, 4, 3)
    assert trace.cci.shape == (8, 3)

    trace.fresh_dqi[0] = [0.1, 0.2, 0.3]
    observation, info = MultiVehicleAccessEnv(
        trace,
        config,
        state_features=['fresh_dqi'],
    ).reset()
    assert observation.shape == (4, 3)
    assert np.all(observation == observation[:1])
    assert info[0]['num_channels'] == 3

    manifest = save_descriptor_bundle(tmp_path, trace)
    loaded = load_descriptor_bundle(tmp_path)
    assert manifest['schema_version'] == '2.0'
    assert manifest['dqi_label_spec']['label_version'] == 'paper_dqi_v1'
    assert loaded.fresh_dqi.shape == (8, 3)


def test_channel_descriptor_contract() -> None:
    snr = np.arange(60.0).reshape(5, 4, 3)
    rss = snr - 95.0
    bandwidth = np.full((5, 3), 10.0)
    cci = np.full((5, 3), 0.9)
    descriptors = channel_descriptors(snr, rss, bandwidth, cci)
    assert descriptors.shape == (5, 3, 4)
    assert np.allclose(descriptors[..., 2], 10.0)
    assert np.allclose(descriptors[..., 3], 0.9)
