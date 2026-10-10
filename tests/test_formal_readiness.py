from __future__ import annotations

import numpy as np

from cvn_rl.experiments.config import load_config
from cvn_rl.simulation.actions import encode_action
from cvn_rl.simulation.reward import compute_step_outcome
from cvn_rl.simulation.traces import generate_trace, recompute_underlay_power_cap
from cvn_rl.simulation.traffic import traffic_label


def test_trace_hash_is_stable_for_same_seed() -> None:
    cfg = {**load_config("configs/base.json"), "num_channels": 4}
    a = generate_trace(cfg, seed=11, slots=20)
    b = generate_trace(cfg, seed=11, slots=20)
    c = generate_trace(cfg, seed=12, slots=20)
    assert a.trace_hash() == b.trace_hash()
    assert a.trace_hash() != c.trace_hash()


def test_underlay_power_cap_uses_estimated_gain_and_threshold() -> None:
    cfg = {"underlay_power": {"p_max_mw": 1.0, "fading_margin": 2.0}}
    gain_hat = np.array([[0.5, 2.0]])
    threshold = np.array([[0.5, 0.5]])
    power_cap, transmit_cap = recompute_underlay_power_cap(gain_hat, threshold, cfg)
    assert np.allclose(transmit_cap, [[0.5, 0.125]])
    assert np.allclose(power_cap, [[0.5, 0.125]])


def test_underlay_violation_uses_actual_interference() -> None:
    cfg = {**load_config("configs/base.json"), "num_channels": 1}
    trace = generate_trace(cfg, seed=3, slots=2)
    trace.occupancy[:, 0] = True
    trace.su_pu_gain_true[:, 0] = 10.0
    trace.pu_threshold_mw[:, 0] = 0.1
    trace.transmit_power_cap_mw[:, 0] = 0.05
    trace.power_cap[:, 0] = 0.05
    action = encode_action(0, "underlay", 1)
    outcome = compute_step_outcome(trace, 0, __import__("cvn_rl.simulation.actions", fromlist=["decode_action"]).decode_action(action, 1), cfg)
    assert outcome.actual_interference_mw == 0.5
    assert outcome.pu_threshold_mw == 0.1
    assert outcome.pu_violation is True


def test_fixed_markov_traffic_label_is_unique() -> None:
    a = traffic_label({"family": "fixed_markov", "p01": 0.1, "p10": 0.3})
    b = traffic_label({"family": "fixed_markov", "p01": 0.3, "p10": 0.1})
    assert a != b
    assert "p01_0.1" in a
