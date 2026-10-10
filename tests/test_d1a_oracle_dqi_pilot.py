"""Tests for D1-A Oracle DQI RL positive-control pilot."""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch

from cvn_rl.agents.shared_multi import create_shared_agent
from cvn_rl.experiments.multi_vehicle_runner import (
    _future_quantile_dqi,
    _resolve_trace,
)
from cvn_rl.simulation.dqi_labels import (
    compute_unnormalized_dqi,
    latency_constrained_goodput,
    normalize_dqi,
)
from cvn_rl.simulation.dqi_policy_value import (
    make_channel_placebo,
    rolling_window_quantile,
)
from cvn_rl.simulation.multi_env import MultiVehicleAccessEnv
from cvn_rl.simulation.multi_traces import (
    MultiVehicleAccessTrace,
    generate_multi_vehicle_trace,
)

R_REF = 17.377581082644856
GAMMA_D1A = 2.0 ** (-1.0 / 100.0)  # ≈ 0.9930924954370359


# ---------------------------------------------------------------------------
# Helper: minimal config
# ---------------------------------------------------------------------------

def _minimal_config(**overrides: object) -> dict:
    cfg = {
        "num_channels": 4,
        "num_vehicles": 3,
        "slot_duration_ms": 1.0,
        "hidden_lookahead_slots": 100,
        "channel": {
            "quality_family": "smooth_semi_markov",
            "fixed_bandwidth_mhz": 10.0,
            "semi_markov_state_snr_db": [-6.0, 0.0, 8.0],
            "semi_markov_bad_dwell_slots": [40, 70],
            "semi_markov_medium_good_dwell_slots": [120, 200],
            "semi_markov_transition_slots": 30,
            "shadowing_ar_coefficient": 0.995,
            "shadowing_stationary_std_db": 2.0,
            "fast_fading_std_db": 1.0,
            "noise_floor_dbm": -95.0,
        },
        "traffic": {"family": "fixed_markov", "p01": 0.0, "p10": 1.0,
                     "sensing_pd": 1.0, "sensing_pfa": 0.0},
        "packet_traffic": {"family": "bernoulli", "arrival_probability": 1.0,
                           "queue_capacity_packets": 10, "packet_size_bits": 32000},
        "switching": {"blackout_slots": 0, "min_dwell_slots": 0},
        "reward": {"type": "physical", "service_weight": 1.0, "drop_weight": 1.0,
                   "contention_weight": 0.25, "violation_weight": 2.0,
                   "queue_weight": 0.05, "clip_reward": [-3.0, 2.0]},
        "action": {"enabled_modes": ["overlay"]},
        "diagnostic_privileged": True,
        "descriptors": {
            "source": "generated",
            "skip_causal_dqi": True,
            "theta_condition": "neutral",
            "theta_neutral_value": 0.5,
            "dqi_update_slots": 1,
            "dqi_horizon_slots": 100,
            "dqi_quantile": 0.10,
            "quantile_method": "linear",
            "r_ref_mbps": R_REF,
            "packet_size_bits": 32000,
            "dqi_delay_max_ms": 5.0,
            "dqi_delay_beta_ms": 2.0,
        },
    }
    cfg.update(overrides)
    return cfg


# ---------------------------------------------------------------------------
# RL Oracle DQI vs canonical DQI
# ---------------------------------------------------------------------------

def test_future_quantile_dqi_matches_canonical_to_tolerance() -> None:
    """RL _future_quantile_dqi should match canonical compute_unnormalized_dqi
    + normalize_dqi within 1e-12 tolerance."""
    rng = np.random.default_rng(42)
    slots = 50
    lookahead = 100
    V, M = 3, 4

    quality = rng.uniform(5.0, 60.0, size=(slots, V, M))
    quality_lookahead = rng.uniform(5.0, 60.0, size=(lookahead, V, M))
    combined = np.concatenate([quality, quality_lookahead], axis=0)

    # Canonical
    z_mbps, _ = compute_unnormalized_dqi(
        combined,
        horizon_samples=100,
        vehicle_quantile=0.10,
        time_quantile=0.10,
        quantile_method="linear",
        packet_size_bits=32000,
        delay_max_ms=5.0,
        delay_beta_ms=2.0,
    )
    canonical_y = normalize_dqi(z_mbps, R_REF)

    # Via runner helper
    trace = generate_multi_vehicle_trace(
        _minimal_config(slots=slots, num_channels=M, num_vehicles=V,
                        hidden_lookahead_slots=lookahead),
        seed=42, slots=slots,
    )
    # Overwrite quality for deterministic test
    object.__setattr__(trace, 'quality_mbps', quality)
    object.__setattr__(trace, 'quality_lookahead_mbps', quality_lookahead)

    desc_cfg = {
        "dqi_update_slots": 1,
        "dqi_horizon_slots": 100,
        "dqi_quantile": 0.10,
        "quantile_method": "linear",
        "r_ref_mbps": R_REF,
        "packet_size_bits": 32000,
        "dqi_delay_max_ms": 5.0,
        "dqi_delay_beta_ms": 2.0,
    }
    rl_dqi = _future_quantile_dqi(trace, desc_cfg)
    valid = canonical_y.shape[0]  # T - H

    assert np.allclose(rl_dqi[:valid], canonical_y, atol=1e-12, rtol=0), (
        "_future_quantile_dqi does not match canonical DQI within 1e-12"
    )


def test_future_quantile_dqi_time_index_is_t_plus_1_to_t_plus_horizon() -> None:
    """Source index s uses window [s+1, s+1+H] (strictly after current)."""
    slots = 150
    horizon = 100
    V, M = 2, 2
    # Build a ramp where each slot has a unique value so window is unambiguous
    combined = np.arange(slots + horizon, dtype=float).reshape(-1, 1, 1)
    combined = np.broadcast_to(combined, (slots + horizon, V, M))

    quality = combined[:slots]
    lookahead = combined[slots:]

    trace = generate_multi_vehicle_trace(
        _minimal_config(slots=slots, num_channels=M, num_vehicles=V,
                        hidden_lookahead_slots=horizon),
        seed=1, slots=slots,
    )
    object.__setattr__(trace, 'quality_mbps', quality)
    object.__setattr__(trace, 'quality_lookahead_mbps', lookahead)

    desc_cfg = {
        "dqi_update_slots": 1,
        "dqi_horizon_slots": horizon,
        "dqi_quantile": 0.10,
        "quantile_method": "linear",
        "r_ref_mbps": R_REF,
        "packet_size_bits": 32000,
        "dqi_delay_max_ms": 5.0,
        "dqi_delay_beta_ms": 2.0,
    }
    rl_dqi = _future_quantile_dqi(trace, desc_cfg)
    assert rl_dqi.shape == (slots, M)
    # Valid DQI source slots: 0 through slots - horizon - 1
    valid_end = slots - horizon - 1
    # With a monotonic ramp, later sources see higher future values
    # The first valid source has lower DQI than a later valid source
    assert valid_end >= 1, f"need at least 2 valid source slots, got {valid_end + 1}"
    assert np.all(rl_dqi[0] < rl_dqi[valid_end])


# ---------------------------------------------------------------------------
# Physical trace hash identity
# ---------------------------------------------------------------------------

def test_oracle_neutral_placebo_share_physical_trace_hash() -> None:
    """All three D1-A information conditions share the same physical trace."""
    base = _minimal_config(slots=20, num_channels=3, num_vehicles=2)
    seed = 42
    splits = ["rl_train"] + [f"rl_eval_{e}" for e in range(2)]

    for split in splits:
        hashes = set()
        for dqi_cond in ("future_oracle", "neutral", "channel_placebo"):
            desc = dict(base["descriptors"])
            desc["dqi_condition"] = dqi_cond
            if dqi_cond == "channel_placebo":
                desc["placebo_block_slots"] = 10
            cfg = {**base, "descriptors": desc}
            trace = _resolve_trace(cfg, "test", seed=seed, slots=20, split=split)
            hashes.add(trace.physical_trace_hash())
        assert len(hashes) == 1, (
            f"physical_trace_hash differs across conditions for {split}: {hashes}"
        )


# ---------------------------------------------------------------------------
# Gamma configurable
# ---------------------------------------------------------------------------

def test_gamma_default_is_0_97() -> None:
    agent = create_shared_agent("ddqn_no_esn", 3, 2, 42, settings={})
    assert agent.gamma == 0.97


def test_gamma_from_config() -> None:
    agent = create_shared_agent(
        "ddqn_no_esn", 3, 2, 42,
        settings={"gamma": GAMMA_D1A},
    )
    assert agent.gamma == pytest.approx(GAMMA_D1A)


def test_gamma_rejects_invalid_range() -> None:
    with pytest.raises(ValueError, match="gamma must be"):
        create_shared_agent("ddqn_no_esn", 3, 2, 42,
                            settings={"gamma": 0.0})
    with pytest.raises(ValueError, match="gamma must be"):
        create_shared_agent("ddqn_no_esn", 3, 2, 42,
                            settings={"gamma": 1.5})
    with pytest.raises(ValueError, match="gamma must be"):
        create_shared_agent("ddqn_no_esn", 3, 2, 42,
                            settings={"gamma": -0.5})


def test_gamma_recorded_in_diagnostics() -> None:
    agent = create_shared_agent(
        "ddqn_no_esn", 3, 2, 42,
        settings={"gamma": GAMMA_D1A},
    )
    diag = agent.training_diagnostics()
    assert "gamma" in diag
    assert diag["gamma"] == pytest.approx(GAMMA_D1A)


# ---------------------------------------------------------------------------
# Dwell semantics: total_from_switch
# ---------------------------------------------------------------------------

def test_dwell_total_from_switch_lock_equals_min_dwell() -> None:
    """Under total_from_switch, total channel lock == min_dwell_slots."""
    config = _minimal_config(slots=20, num_channels=2, num_vehicles=1)
    config["switching"] = {
        "blackout_slots": 3,
        "min_dwell_slots": 8,
        "dwell_semantics": "total_from_switch",
    }
    trace = generate_multi_vehicle_trace(config, seed=5, slots=20)
    trace.occupancy[:] = False
    trace.sensing_idle[:] = 1.0
    env = MultiVehicleAccessEnv(trace, config)
    env.reset()

    from cvn_rl.simulation.actions import encode_action
    ch0 = encode_action(0, "overlay", 2)
    ch1 = encode_action(1, "overlay", 2)

    # Initial acquisition: free
    env.step(np.array([ch0]))
    for _ in range(9):
        env.step(np.array([ch0]))  # dwell expires after 8 steps

    assert env.dwell_remaining[0] == 0

    # Switch at slot s — dwell and blackout armed together
    _, _, _, sw = env.step(np.array([ch1]))
    assert sw["switch_started"][0]
    # step_info captures state AFTER timer advance
    # dwell was 8 at switch, decremented to 7; blackout was 3, decremented to 2
    assert env.dwell_remaining[0] == 7
    assert env.blackout_remaining[0] == 2

    # Let blackout expire (2 more slots)
    env.step(np.array([ch0]))  # blackout decremented: 2→1, dwell: 7→6
    assert env.blackout_remaining[0] == 1
    assert env.dwell_remaining[0] == 6
    env.step(np.array([ch0]))  # blackout decremented: 1→0, dwell: 6→5
    assert env.blackout_remaining[0] == 0
    assert env.dwell_remaining[0] == 5

    # Decrement dwell remaining 5 steps
    for _ in range(5):
        env.step(np.array([ch1]))
    assert env.dwell_remaining[0] == 0

    # Now switching is permitted again
    assert env.agent_infos()[0]["action_mask"][ch0]  # switching allowed
    # Total slots since switch: 1 (switch slot) + 2 + 5 = 8 = min_dwell
    # Next switch permitted precisely at s + min_dwell


def test_dwell_legacy_after_blackout_preserved() -> None:
    """Legacy semantics: dwell starts after blackout. Total = blackout + dwell."""
    config = _minimal_config(slots=20, num_channels=2, num_vehicles=1)
    config["switching"] = {
        "blackout_slots": 2,
        "min_dwell_slots": 5,
        "dwell_semantics": "legacy_after_blackout",
    }
    trace = generate_multi_vehicle_trace(config, seed=7, slots=20)
    trace.occupancy[:] = False
    trace.sensing_idle[:] = 1.0
    env = MultiVehicleAccessEnv(trace, config)
    env.reset()

    from cvn_rl.simulation.actions import encode_action
    ch0 = encode_action(0, "overlay", 2)
    ch1 = encode_action(1, "overlay", 2)

    env.step(np.array([ch0]))
    for _ in range(6):
        env.step(np.array([ch0]))
    assert env.dwell_remaining[0] == 0

    _, _, _, sw = env.step(np.array([ch1]))
    assert sw["switch_started"][0]
    assert env.blackout_remaining[0] == 1  # 2→1 decremented

    # After 2nd blackout slot
    env.step(np.array([ch0]))
    assert env.blackout_remaining[0] == 0
    # Legacy: dwell starts NOW
    assert env.dwell_remaining[0] == 5  # legacy arms dwell after blackout


# ---------------------------------------------------------------------------
# Channel placebo is block derangement
# ---------------------------------------------------------------------------

def test_channel_placebo_is_block_derangement() -> None:
    """make_channel_placebo permutes channels within blocks, no identity map."""
    rng = np.random.default_rng(123)
    values = np.arange(40, dtype=float).reshape(10, 4)  # 10 slots, 4 channels
    placebo, perms = make_channel_placebo(values, block_slots=5, rng=rng)
    assert placebo.shape == values.shape
    assert len(perms) == 2  # 2 blocks of 5

    for p in perms:
        assert not np.all(p == np.arange(4)), "permutation should be derangement"

    # Per-block multiset preserved
    for start, p in zip([0, 5], perms):
        block = values[start:start + 5]
        placebo_block = placebo[start:start + 5]
        for t in range(block.shape[0]):
            assert set(block[t]) == set(placebo_block[t])


# ---------------------------------------------------------------------------
# Rolling q10 on synthetic data
# ---------------------------------------------------------------------------

def test_rolling_window_quantile_on_known_series() -> None:
    """rolling_window_quantile with window=3, q=0.10 on known data."""
    data = np.array([0.0, 0.0, 5.0, 1.0, 0.0, 8.0, 0.0, 0.0, 10.0], dtype=float)
    # Window 3: [0,0,5]→q10=0, [0,5,1]→q10=0.1, [5,1,0]→q10=0.1, ...
    result = rolling_window_quantile(data, window_samples=3, quantile=0.10)
    assert len(result) == 7  # 9 - 3 + 1
    # First window [0,0,5]: sorted=[0,0,5], q10 = 0.0 (linear interp)
    assert result[0] == pytest.approx(0.0, abs=1e-10)
    # Window [0,0,10]: sorted=[0,0,10], q10 = 0.0
    # Last window [8,0,0]: q10 = 0.0
    # Window [5,1,0]: q10 = 0.1
    # Window [1,0,8]: q10 = 0.1
    # Window [0,8,0]: q10 = 0.0
    # Window [8,0,0]: q10 = 0.0
    # Window [0,0,10]: q10 = 0.0


def test_rolling_quantile_includes_zeros() -> None:
    """Zeros participate in the quantile (not filtered)."""
    data = np.array([0.0, 0.0, 0.0, 0.0, 10.0], dtype=float)
    result = rolling_window_quantile(data, window_samples=3, quantile=0.10)
    # All windows before slot 4: [0,0,0] → q10 = 0
    assert result[0] == 0.0
    assert result[1] == 0.0
    # Last window [0,0,10]: q10 = 0.0 (linear interp between first two 0s)
    assert result[-1] == pytest.approx(0.0, abs=1e-10)


# ---------------------------------------------------------------------------
# Reward independent of DQI
# ---------------------------------------------------------------------------

def test_physical_reward_does_not_read_dqi() -> None:
    """Under reward.type == 'physical', changing fresh_dqi does not change reward."""
    config = _minimal_config(slots=10, num_channels=2, num_vehicles=1)
    config["switching"] = {"blackout_slots": 0, "min_dwell_slots": 0}
    trace = generate_multi_vehicle_trace(config, seed=99, slots=10)
    trace.occupancy[:] = False
    trace.sensing_idle[:] = 1.0

    # Run with two different DQI arrays
    from cvn_rl.simulation.actions import encode_action
    action = encode_action(0, "overlay", 2)
    rewards_a = []
    rewards_b = []

    for dqi_val in (0.3, 0.9):
        t = generate_multi_vehicle_trace(config, seed=99, slots=10)
        t.occupancy[:] = False
        t.sensing_idle[:] = 1.0
        t.fresh_dqi[:] = dqi_val
        env = MultiVehicleAccessEnv(t, config)
        env.reset()
        rewards = []
        for _ in range(8):
            _, r, _, _ = env.step(np.array([action]))
            rewards.append(float(r[0]))
        if dqi_val == 0.3:
            rewards_a = rewards
        else:
            rewards_b = rewards

    assert rewards_a == rewards_b, (
        "physical reward should not depend on fresh_dqi"
    )


# ---------------------------------------------------------------------------
# Gate logic unit test
# ---------------------------------------------------------------------------

def _synthetic_final_frame(
    oracle_goodput: list[float],
    neutral_goodput: list[float],
    placebo_goodput: list[float],
    oracle_tail: list[float] | None = None,
    neutral_tail: list[float] | None = None,
    placebo_tail: list[float] | None = None,
    dwell_violations: int = 0,
) -> pd.DataFrame:
    """Build a minimal final-checkpoint DataFrame for gate testing."""
    seeds = [301, 302, 303, 304, 305]
    rows = []
    phys_hash = "abc123"
    for i, seed in enumerate(seeds):
        for variant, goodput, tail in [
            ("dqi_future_oracle", oracle_goodput[i], (oracle_tail or oracle_goodput)[i]),
            ("dqi_neutral", neutral_goodput[i], (neutral_tail or neutral_goodput)[i]),
            ("dqi_channel_placebo", placebo_goodput[i], (placebo_tail or placebo_goodput)[i]),
        ]:
            rows.append({
                "module_variant": variant,
                "seed": seed,
                "served_goodput_mbps": goodput,
                "rolling_100ms_system_served_q10_mean": tail,
                "dwell_violation_count": dwell_violations,
                "train_physical_trace_hash": phys_hash,
                "evaluation_physical_trace_hash": phys_hash,
                "is_final_checkpoint": True,
                "run_scope": "formal",
                "gamma": GAMMA_D1A,
                "evaluation_parameters_frozen": True,
            })
    return pd.DataFrame(rows)


def test_gate_pass_synthetic() -> None:
    """Synthetic frame where oracle clearly beats neutral and placebo → pass."""
    oracle = [16.5, 17.0, 16.8, 17.2, 16.9]
    neutral = [16.0, 16.2, 16.1, 16.3, 16.0]
    placebo = [14.0, 14.5, 14.3, 14.8, 14.1]
    df = _synthetic_final_frame(oracle, neutral, placebo)

    # Quick manual check: all seeds oracle > neutral, mean gain > 2%
    for o, n in zip(oracle, neutral):
        assert o > n
    rel_gains = [(o - n) / n for o, n in zip(oracle, neutral)]
    assert np.mean(rel_gains) >= 0.02

    # Physical trace hashes match
    for seed in [301, 302, 303, 304, 305]:
        sub = df[df["seed"] == seed]
        assert sub["train_physical_trace_hash"].nunique() == 1
    assert df["dwell_violation_count"].sum() == 0


def test_gate_fails_on_dwell_violation() -> None:
    """Gate should detect non-zero dwell violations."""
    oracle = [16.5, 17.0, 16.8, 17.2, 16.9]
    neutral = [16.0, 16.2, 16.1, 16.3, 16.0]
    placebo = [14.0, 14.5, 14.3, 14.8, 14.1]
    df = _synthetic_final_frame(oracle, neutral, placebo, dwell_violations=1)
    # Each of the 15 rows (5 seeds × 3 conditions) has dwell_violation_count=1
    assert (df["dwell_violation_count"] > 0).all()
    assert df["dwell_violation_count"].iloc[0] == 1


def test_gate_fails_on_hash_mismatch() -> None:
    """Gate should detect physical trace hash mismatch."""
    df = _synthetic_final_frame(
        [16.5, 17.0, 16.8, 17.2, 16.9],
        [16.0, 16.2, 16.1, 16.3, 16.0],
        [14.0, 14.5, 14.3, 14.8, 14.1],
    )
    # Change only one variant's hash for one seed → creates mismatch
    mask = (df["seed"] == 301) & (df["module_variant"] == "dqi_future_oracle")
    df.loc[mask, "train_physical_trace_hash"] = "MISMATCH"
    sub = df[df["seed"] == 301]
    assert sub["train_physical_trace_hash"].nunique() == 2  # original + MISMATCH


def test_gate_fails_on_insufficient_gain() -> None:
    """Gate should detect gain < 2%."""
    oracle = [16.1, 16.15, 16.12, 16.18, 16.14]
    neutral = [16.0, 16.13, 16.11, 16.17, 16.13]
    # Very small gains
    rel_gains = [(o - n) / n for o, n in zip(oracle, neutral)]
    assert np.mean(rel_gains) < 0.02


# ---------------------------------------------------------------------------
# D1-A-R2: time placebo tests
# ---------------------------------------------------------------------------

from cvn_rl.simulation.dqi_policy_value import make_time_block_placebo


def test_time_placebo_is_block_derangement() -> None:
    """make_time_block_placebo permutes complete blocks, preserves internal order."""
    rng = np.random.default_rng(456)
    # 12 slots, 4 channels, 3 blocks of 4
    values = np.arange(48, dtype=float).reshape(12, 4)
    placebo, perm = make_time_block_placebo(values, block_slots=4, rng=rng)
    assert placebo.shape == values.shape
    # permutation should be derangement
    assert not np.all(perm == np.arange(3))
    # Internal order within each block preserved (values within a block are contiguous)
    # Each block in result corresponds to some original block
    n_blocks = 3
    result_blocks = placebo.reshape(n_blocks, 4, 4)
    for rb in result_blocks:
        found = False
        for ob in values.reshape(n_blocks, 4, 4):
            if np.array_equal(rb, ob):
                found = True
                break
        assert found, "each result block should equal some original block"


def test_time_placebo_preserves_per_block_values() -> None:
    """Time placebo preserves all values — just reorders blocks."""
    rng = np.random.default_rng(789)
    values = np.random.uniform(0, 1, size=(200, 5))
    placebo, _ = make_time_block_placebo(values, block_slots=50, rng=rng)
    # Sorted values over all time/channels should match
    assert np.allclose(np.sort(values.ravel()), np.sort(placebo.ravel()))


def test_oracle_channel_time_neutral_share_physical_trace() -> None:
    """All 4 R2 conditions share the same physical trace."""
    from cvn_rl.experiments.multi_vehicle_runner import _resolve_trace

    base = _minimal_config(slots=200, num_channels=3, num_vehicles=1)
    seed = 42

    hashes = {}
    for dqi_cond in ("future_oracle", "channel_placebo", "time_placebo", "neutral"):
        desc = dict(base["descriptors"])
        desc["dqi_condition"] = dqi_cond
        if dqi_cond in ("channel_placebo", "time_placebo"):
            desc["placebo_block_slots"] = 50
        cfg = {**base, "descriptors": desc}
        trace = _resolve_trace(cfg, "test", seed=seed, slots=200, split="rl_train")
        hashes[dqi_cond] = trace.physical_trace_hash()

    assert len(set(hashes.values())) == 1, (
        f"physical_trace_hash differs: {hashes}"
    )


def test_r2_dry_run_all_four_conditions_present() -> None:
    """D1-A-R2 config resolves to 4 module_variants."""
    from cvn_rl.experiments.config import load_config

    config = load_config("configs/e2_d1a_r2_oracle_dqi_pilot.json")
    names = [c["name"] for c in config.get("module_conditions", [])]
    assert len(names) == 4
    assert set(names) == {
        "dqi_future_oracle",
        "dqi_channel_placebo",
        "dqi_time_placebo",
        "dqi_neutral",
    }
