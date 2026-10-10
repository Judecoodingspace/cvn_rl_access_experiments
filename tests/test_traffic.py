from __future__ import annotations

import numpy as np

from cvn_rl.simulation.traffic import generate_occupancy


def test_traffic_families_are_reproducible() -> None:
    families = [
        {"family": "fixed_markov", "p01": 0.2, "p10": 0.3},
        {"family": "time_varying_markov", "blocks": 3, "p01_range": [0.1, 0.2], "p10_range": [0.2, 0.3]},
        {"family": "bursty_on_off", "idle_mean_slots": 5, "busy_mean_slots": 4},
        {"family": "periodic_with_jitter", "period_slots": 10, "busy_slots": 3, "jitter_slots": 1},
        {"family": "mixed_by_channel"},
    ]
    for family in families:
        a = generate_occupancy(40, 4, family, np.random.default_rng(7))
        b = generate_occupancy(40, 4, family, np.random.default_rng(7))
        assert a.shape == (40, 4)
        assert np.array_equal(a, b)
