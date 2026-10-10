from __future__ import annotations

from typing import Any

import numpy as np

from cvn_rl.agents.base import BaseAgent
from cvn_rl.simulation.actions import encode_action
from cvn_rl.simulation.state_builder import observation_feature


class FixedOverlayAgent(BaseAgent):
    """Always use overlay on the best visible idle-quality channel."""

    def select_action(self, observation: np.ndarray, info: dict[str, Any]) -> int:
        features = info.get("state_features")
        delta = observation_feature(observation, "delta", self.num_channels, features, default=1.0)
        dqi = observation_feature(observation, "fresh_dqi", self.num_channels, features, default=0.5)
        theta = observation_feature(observation, "theta", self.num_channels, features, default=0.5)
        scores = dqi * theta * delta
        channel = int(np.argmax(scores))
        if scores[channel] <= 0:
            channel = int(np.argmax(dqi * theta))
        return _masked_or_fallback(encode_action(channel, "overlay", self.num_channels), info)


class FixedUnderlayAgent(BaseAgent):
    """Always use underlay on the best visible power-cap channel."""

    def select_action(self, observation: np.ndarray, info: dict[str, Any]) -> int:
        features = info.get("state_features")
        dqi = observation_feature(observation, "fresh_dqi", self.num_channels, features, default=0.5)
        cap = observation_feature(observation, "power_cap", self.num_channels, features, default=1.0)
        theta = observation_feature(observation, "theta", self.num_channels, features, default=0.5)
        scores = dqi * cap * np.maximum(theta, 0.2)
        channel = int(np.argmax(scores))
        return _masked_or_fallback(encode_action(channel, "underlay", self.num_channels), info)


def _masked_or_fallback(action: int, info: dict[str, Any]) -> int:
    mask = info.get("action_mask")
    if mask is None or bool(mask[action]):
        return action
    valid = np.flatnonzero(mask)
    return int(valid[0]) if len(valid) else 0
