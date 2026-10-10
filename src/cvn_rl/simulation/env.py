from __future__ import annotations

from typing import Any

import numpy as np

from cvn_rl.simulation.actions import build_action_mask, decode_action
from cvn_rl.simulation.reward import compute_step_outcome
from cvn_rl.simulation.state_builder import DEFAULT_FEATURES, build_state, feature_slices
from cvn_rl.simulation.traces import AccessTrace


class AccessEnv:
    """Minimal discrete channel-mode environment over a frozen trace."""

    def __init__(
        self,
        trace: AccessTrace,
        config: dict[str, Any],
        state_features: list[str] | None = None,
        reward_layers: list[str] | None = None,
    ) -> None:
        self.trace = trace
        self.config = config
        self.state_features = state_features or list(DEFAULT_FEATURES)
        self.reward_layers = reward_layers or ["reward_gate"]
        self.t = 0

    def reset(self) -> tuple[np.ndarray, dict[str, Any]]:
        self.t = 0
        return self._observation(), self._info()

    def step(self, action_index: int) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        action = decode_action(action_index, self.trace.num_channels)
        outcome = compute_step_outcome(self.trace, self.t, action, self.config)
        info = self._info()
        info.update(
            {
                "action_index": action_index,
                "channel": action.channel,
                "mode": action.mode,
                "vt_throughput": outcome.vt_throughput,
                "pu_throughput": outcome.pu_throughput,
                "pu_violation": outcome.pu_violation,
                "packet_loss": outcome.packet_loss,
                "selected_dqi": outcome.selected_dqi,
                "underlay_selected": outcome.underlay_selected,
                "reward": outcome.reward,
                "actual_interference_mw": outcome.actual_interference_mw,
                "pu_threshold_mw": outcome.pu_threshold_mw,
                "transmit_power_mw": outcome.transmit_power_mw,
                "interference_margin_mw": outcome.interference_margin_mw,
            }
        )
        self.t += 1
        done = self.t >= self.trace.slots
        return self._observation(), outcome.reward, done, info

    def _observation(self) -> np.ndarray:
        t = min(self.t, self.trace.slots - 1)
        return build_state(self.trace, t, self.state_features)

    def _info(self) -> dict[str, Any]:
        t = min(self.t, self.trace.slots - 1)
        return {
            "t": t,
            "num_channels": self.trace.num_channels,
            "state_features": list(self.state_features),
            "feature_slices": feature_slices(self.trace.num_channels, self.state_features),
            "action_mask": build_action_mask(
                self.trace.sensing_idle[t],
                self.trace.theta[t],
                self.trace.power_cap[t],
                self.reward_layers,
                self.config.get("action_mask_thresholds"),
            ),
        }
