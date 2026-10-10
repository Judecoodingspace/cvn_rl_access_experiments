from __future__ import annotations

import numpy as np

from cvn_rl.experiments.config import load_config
from cvn_rl.simulation.actions import encode_action
from cvn_rl.simulation.env import AccessEnv
from cvn_rl.simulation.traces import generate_trace


def test_env_step_returns_reward_and_safety_info() -> None:
    cfg = load_config("configs/base.json")
    trace = generate_trace({**cfg, "num_channels": 3}, seed=1, slots=8)
    env = AccessEnv(trace, cfg, reward_layers=["reward_gate", "action_mask"])
    obs, info = env.reset()
    action = encode_action(0, "overlay", trace.num_channels)
    next_obs, reward, done, step_info = env.step(action)
    assert obs.size > 0
    assert next_obs.size == obs.size
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert "pu_violation" in step_info
    assert info["action_mask"].dtype == np.bool_
