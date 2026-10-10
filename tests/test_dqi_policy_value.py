from __future__ import annotations

import numpy as np

from cvn_rl.simulation.dqi_policy_value import (
    SwitchingCost,
    evaluate_channel_policy,
    future_window_mean,
    make_channel_placebo,
    make_time_block_placebo,
    rolling_window_quantile,
)
from cvn_rl.simulation.multi_traces import generate_smooth_semi_markov_quality


def _inputs(slots: int = 8) -> tuple[np.ndarray, ...]:
    goodput = np.tile(np.array([[10.0, 20.0, 5.0]]), (slots, 1))
    raw = goodput + 1.0
    violation = np.zeros((slots, 2, 3), dtype=bool)
    ranking = np.zeros_like(goodput)
    ranking[:, 0] = 1.0
    projected = 100.0 * ranking
    oracle = goodput.copy()
    return goodput, raw, violation, ranking, projected, oracle


def test_smooth_semi_markov_is_reproducible_and_equal_bandwidth() -> None:
    config = {
        'channel': {
            'quality_family': 'smooth_semi_markov',
            'fixed_bandwidth_mhz': 10.0,
            'semi_markov_state_snr_db': [-6.0, 0.0, 8.0],
            'semi_markov_bad_dwell_slots': [6, 10],
            'semi_markov_medium_good_dwell_slots': [12, 20],
            'semi_markov_transition_slots': 3,
            'shadowing_ar_coefficient': 0.995,
            'shadowing_stationary_std_db': 2.0,
            'fast_fading_std_db': 1.0,
        }
    }
    bandwidth_a, quality_a = generate_smooth_semi_markov_quality(
        100, 4, 5, config, np.random.default_rng(7)
    )
    bandwidth_b, quality_b = generate_smooth_semi_markov_quality(
        100, 4, 5, config, np.random.default_rng(7)
    )
    assert quality_a.shape == (100, 4, 5)
    assert np.all(bandwidth_a == 10.0)
    assert np.array_equal(bandwidth_a, bandwidth_b)
    assert np.array_equal(quality_a, quality_b)


def test_placebo_is_blockwise_derangement_and_preserves_values() -> None:
    values = np.arange(120.0).reshape(20, 6)
    placebo, permutations = make_channel_placebo(
        values, block_slots=5, rng=np.random.default_rng(8)
    )
    assert len(permutations) == 4
    for permutation in permutations:
        assert np.all(permutation != np.arange(6))
    assert np.array_equal(np.sort(placebo, axis=1), np.sort(values, axis=1))


def test_time_placebo_deranges_blocks_and_preserves_internal_order() -> None:
    values = np.arange(48.0).reshape(12, 4)
    placebo, permutation = make_time_block_placebo(
        values, block_slots=3, rng=np.random.default_rng(9)
    )
    assert np.all(permutation != np.arange(4))
    source_blocks = values.reshape(4, 3, 4)
    placebo_blocks = placebo.reshape(4, 3, 4)
    assert np.array_equal(placebo_blocks, source_blocks[permutation])


def test_future_mean_uses_strict_t_plus_one_through_horizon() -> None:
    values = np.arange(12.0).reshape(6, 2)
    result = future_window_mean(values, 2)
    assert result.shape == (4, 2)
    assert np.array_equal(result[0], np.mean(values[1:3], axis=0))
    changed = values.copy()
    changed[0] = 1e9
    assert np.array_equal(future_window_mean(changed, 2), result)


def test_rolling_quantile_includes_blackout_zeros() -> None:
    service = np.array([10.0, 0.0, 20.0, 30.0, 0.0])
    result = rolling_window_quantile(
        service, window_samples=3, quantile=0.10
    )
    expected = np.array([
        np.quantile(service[index:index + 3], 0.10, method='linear')
        for index in range(3)
    ])
    assert np.array_equal(result, expected)


def test_initial_selection_is_free_and_blackout_has_exact_length() -> None:
    goodput, raw, violation, ranking, projected, oracle = _inputs()
    ranking[2:, 0] = 0.0
    ranking[2:, 1] = 2.0
    projected = 100.0 * ranking
    result = evaluate_channel_policy(
        goodput, raw, violation, ranking, projected, oracle,
        SwitchingCost(blackout_slots=2, min_dwell_slots=0),
        start_slot=1, use_hysteresis=False,
    )
    assert not result.blackout[0]
    assert result.switched.tolist() == [False, True, False, False, False, False, False]
    assert result.blackout.tolist() == [False, True, True, False, False, False, False]
    assert np.all(result.net_shared_goodput_mbps[result.blackout] == 0.0)
    assert np.all(
        result.gross_shared_goodput_mbps[~result.blackout]
        == result.net_shared_goodput_mbps[~result.blackout]
    )


def test_minimum_dwell_blocks_early_switch() -> None:
    goodput, raw, violation, ranking, projected, oracle = _inputs()
    ranking[2:, 0] = 0.0
    ranking[2:, 1] = 2.0
    projected = 100.0 * ranking
    result = evaluate_channel_policy(
        goodput, raw, violation, ranking, projected, oracle,
        SwitchingCost(blackout_slots=0, min_dwell_slots=3),
        start_slot=1, use_hysteresis=False,
    )
    assert result.selected_channel[:3].tolist() == [0, 0, 0]
    assert result.switched[3]
    assert result.selected_channel[3] == 1


def test_hysteresis_requires_gain_above_blackout_loss() -> None:
    goodput, raw, violation, ranking, projected, oracle = _inputs()
    ranking[2:, 1] = 2.0
    projected[:, 0] = 100.0
    projected[:, 1] = 105.0
    result = evaluate_channel_policy(
        goodput, raw, violation, ranking, projected, oracle,
        SwitchingCost(blackout_slots=2, min_dwell_slots=0),
        start_slot=1, use_hysteresis=True,
    )
    assert not np.any(result.switched)
    projected[:, 1] = 125.0
    switched = evaluate_channel_policy(
        goodput, raw, violation, ranking, projected, oracle,
        SwitchingCost(blackout_slots=2, min_dwell_slots=0),
        start_slot=1, use_hysteresis=True,
    )
    assert switched.switched[1]
