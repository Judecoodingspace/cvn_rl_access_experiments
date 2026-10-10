from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from cvn_rl.agents.base import BaseAgent
from cvn_rl.simulation.state_builder import observation_feature


class TabularQLearningAgent(BaseAgent):
    """Small online Q-learning baseline for smoke tests and ablations."""

    def __init__(
        self,
        name: str,
        num_channels: int,
        seed: int = 0,
        learning_rate: float = 0.15,
        discount: float = 0.95,
        epsilon: float = 0.08,
    ) -> None:
        super().__init__(name=name, num_channels=num_channels, seed=seed)
        self.learning_rate = learning_rate
        self.discount = discount
        self.epsilon = epsilon
        self.num_actions = 1 + 2 * num_channels
        self.q_values: defaultdict[tuple[int, ...], np.ndarray] = defaultdict(
            lambda: np.zeros(self.num_actions, dtype=float)
        )
        self._last_state: tuple[int, ...] | None = None

    def select_action(self, observation: np.ndarray, info: dict[str, Any]) -> int:
        state = self._state_key(observation, info)
        self._last_state = state
        mask = info.get("action_mask")
        if (not self.eval_mode) and self.rng.random() < self.epsilon:
            valid = np.flatnonzero(mask) if mask is not None else np.arange(self.num_actions)
            return int(self.rng.choice(valid)) if len(valid) else 0
        q = self.q_values[state].copy()
        if mask is not None:
            q[~mask] = -np.inf
        return int(np.argmax(q))

    def observe(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        next_observation: np.ndarray,
        done: bool,
        info: dict[str, Any],
    ) -> None:
        if self.eval_mode:
            return
        state = self._last_state or self._state_key(observation, info)
        next_state = self._state_key(next_observation, info)
        future = 0.0 if done else float(np.max(self.q_values[next_state]))
        target = reward + self.discount * future
        self.q_values[state][action] += self.learning_rate * (target - self.q_values[state][action])

    def _state_key(self, observation: np.ndarray, info: dict[str, Any]) -> tuple[int, ...]:
        features = info.get("state_features")
        delta = observation_feature(observation, "delta", self.num_channels, features, default=0.5)
        dqi = observation_feature(observation, "fresh_dqi", self.num_channels, features, default=0.5)
        theta = observation_feature(observation, "theta", self.num_channels, features, default=0.5)
        cap = observation_feature(observation, "power_cap", self.num_channels, features, default=0.5)
        return (
            int(np.mean(delta) > 0.5),
            int(np.max(dqi) * 4),
            int(np.max(theta) * 4),
            int(np.max(cap) * 4),
        )
