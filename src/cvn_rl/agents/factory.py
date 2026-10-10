from __future__ import annotations

from cvn_rl.agents.base import BaseAgent
from cvn_rl.agents.fixed import FixedOverlayAgent, FixedUnderlayAgent
from cvn_rl.agents.tabular_q import TabularQLearningAgent
from cvn_rl.agents.torch_agents import TorchDQNAgent, TorchPPOAgent


def create_agent(name: str, num_channels: int, seed: int = 0) -> BaseAgent:
    if name == "fixed_overlay":
        return FixedOverlayAgent(name=name, num_channels=num_channels, seed=seed)
    if name == "fixed_underlay":
        return FixedUnderlayAgent(name=name, num_channels=num_channels, seed=seed)
    if name == "tabular_q_learning":
        return TabularQLearningAgent(name=name, num_channels=num_channels, seed=seed)
    if name == "proposed_esn_ddqn":
        return TorchDQNAgent(name=name, num_channels=num_channels, seed=seed, double_q=True, esn=True)
    if name == "ddqn_no_esn":
        return TorchDQNAgent(name=name, num_channels=num_channels, seed=seed, double_q=True, esn=False)
    if name == "dqn":
        return TorchDQNAgent(name=name, num_channels=num_channels, seed=seed, double_q=False, esn=False)
    if name == "ppo_discrete":
        return TorchPPOAgent(name=name, num_channels=num_channels, seed=seed)
    raise ValueError(f"Unknown agent: {name}")
