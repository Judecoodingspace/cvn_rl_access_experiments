from __future__ import annotations

import numpy as np

from scripts.audit_dqi_incremental_value import build_strategy_inputs


def test_current_scores_are_lagged_but_oracles_use_supplied_future_values() -> None:
    slots, channels = 6, 3
    raw = np.arange(slots * channels, dtype=float).reshape(slots, channels) + 1.0
    goodput = raw * 0.5
    future_mean = raw + 100.0
    y = np.linspace(0.1, 0.9, slots * channels).reshape(slots, channels)
    payload = {
        'shared_raw_rate_mbps': raw,
        'shared_goodput_mbps': goodput,
        'future_mean_mbps': future_mean,
    }
    config = {
        'd0_6_audit': {
            'horizon_slots': 2,
            'placebo_block_slots': 2,
        }
    }
    strategies = build_strategy_inputs(payload, y, 20.0, config, seed=4)
    current_cqi, _ = strategies['current_cqi_hysteresis']
    current_lcg, _ = strategies['current_lcg_hysteresis']
    mean_oracle, _ = strategies['future_mean_oracle']
    assert np.array_equal(current_cqi[1:], raw[:-1])
    assert np.array_equal(current_lcg[1:], goodput[:-1])
    assert np.array_equal(mean_oracle, future_mean)


def test_strategy_placebos_preserve_shape_and_determinism() -> None:
    slots, channels = 8, 4
    raw = np.arange(slots * channels, dtype=float).reshape(slots, channels) + 1.0
    payload = {
        'shared_raw_rate_mbps': raw,
        'shared_goodput_mbps': raw * 0.5,
        'future_mean_mbps': raw * 0.75,
    }
    y = raw / raw.max()
    config = {
        'd0_6_audit': {
            'horizon_slots': 2,
            'placebo_block_slots': 2,
        }
    }
    first = build_strategy_inputs(payload, y, 20.0, config, seed=7)
    second = build_strategy_inputs(payload, y, 20.0, config, seed=7)
    for policy in ('channel_placebo_dqi', 'time_placebo_dqi'):
        assert first[policy][0].shape == y.shape
        assert np.array_equal(first[policy][0], second[policy][0])
