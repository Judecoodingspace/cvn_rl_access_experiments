"""Agent registry and baseline policy implementations."""

from cvn_rl.agents.base import BaseAgent
from cvn_rl.agents.factory import create_agent

__all__ = ["BaseAgent", "create_agent"]
