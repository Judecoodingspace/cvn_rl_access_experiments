from __future__ import annotations

from typing import Any

import numpy as np

from cvn_rl.agents.base import BaseAgent
from cvn_rl.simulation.actions import encode_action
from cvn_rl.simulation.state_builder import observation_feature


class ScorePolicyAgent(BaseAgent):
    """Fast value-policy scaffold used until full GPU trainers are plugged in."""

    def __init__(self, name: str, num_channels: int, seed: int = 0, style: str = "ddqn") -> None:
        super().__init__(name=name, num_channels=num_channels, seed=seed)
        self.style = style
        self._reservoir = np.zeros(num_channels, dtype=float)

    def select_action(self, observation: np.ndarray, info: dict[str, Any]) -> int:
        features = info.get("state_features")
        dqi = observation_feature(observation, "fresh_dqi", self.num_channels, features, default=0.5)
        theta = observation_feature(observation, "theta", self.num_channels, features, default=0.5)
        delta = observation_feature(observation, "delta", self.num_channels, features, default=0.5)
        cap = observation_feature(observation, "power_cap", self.num_channels, features, default=0.5)

        if self.style == "esn_ddqn":
            self._reservoir = 0.85 * self._reservoir + 0.15 * (dqi * theta)
            score = 0.55 * dqi + 0.35 * theta + 0.10 * self._reservoir
        elif self.style == "ddqn":
            score = 0.55 * dqi + 0.45 * theta
        elif self.style == "dqn":
            score = 0.70 * dqi + 0.30 * delta
        elif self.style == "ppo":
            score = 0.45 * dqi + 0.45 * theta + 0.10 * cap
        else:
            score = dqi * theta

        channel = int(np.argmax(score))
        mode = "overlay" if delta[channel] > 0.5 and theta[channel] > 0.45 else "underlay"
        if mode == "underlay" and cap[channel] < 0.25:
            mode = "overlay"
        action = encode_action(channel, mode, self.num_channels)
        return _masked_or_best(action, score, mode, info, self.num_channels)


def _masked_or_best(
    preferred_action: int,
    channel_score: np.ndarray,
    preferred_mode: str,
    info: dict[str, Any],
    num_channels: int,
) -> int:
    mask = info.get("action_mask")
    if mask is None or bool(mask[preferred_action]):
        return preferred_action
    ranked = np.argsort(channel_score)[::-1]
    for channel in ranked:
        for mode in (preferred_mode, "overlay", "underlay"):
            candidate = encode_action(int(channel), mode, num_channels)
            if bool(mask[candidate]):
                return candidate
    return 0
