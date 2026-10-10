from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class BaseAgent:
    """Common interface used by the experiment runners."""

    name: str
    num_channels: int
    seed: int = 0
    rng: np.random.Generator = field(init=False)
    training_time_seconds: float = 0.0
    inference_latency_ms: float = 0.0
    eval_mode: bool = False

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)

    def reset(self) -> None:
        """Reset episode-local state."""

    def set_eval_mode(self, enabled: bool) -> None:
        self.eval_mode = enabled

    def select_action(self, observation: np.ndarray, info: dict[str, Any]) -> int:
        raise NotImplementedError

    def observe(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        next_observation: np.ndarray,
        done: bool,
        info: dict[str, Any],
    ) -> None:
        """Receive a transition after the environment step."""
