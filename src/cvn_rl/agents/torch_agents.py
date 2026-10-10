from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import os
from typing import Any

import numpy as np

from cvn_rl.agents.base import BaseAgent

try:
    import torch
    from torch import nn
    import torch.nn.functional as F
except Exception as exc:  # pragma: no cover - exercised only when torch is unavailable.
    torch = None
    nn = None
    F = None
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None


@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    action_mask: np.ndarray
    next_action_mask: np.ndarray



def _select_device() -> "torch.device":
    requested = os.environ.get("CVN_RL_DEVICE", "auto").lower()
    if requested == "cpu":
        return torch.device("cpu")
    if requested.startswith("cuda") and torch.cuda.is_available():
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ReplayBuffer:
    def __init__(self, capacity: int, rng: np.random.Generator) -> None:
        self.capacity = capacity
        self.rng = rng
        self.data: deque[Transition] = deque(maxlen=capacity)

    def add(self, transition: Transition) -> None:
        self.data.append(transition)

    def __len__(self) -> int:
        return len(self.data)

    def sample(self, batch_size: int) -> list[Transition]:
        indices = self.rng.choice(len(self.data), size=batch_size, replace=False)
        return [self.data[int(i)] for i in indices]


class MLPQNetwork(nn.Module):
    def __init__(self, input_dim: int, num_actions: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LinearQHead(nn.Module):
    def __init__(self, input_dim: int, num_actions: int) -> None:
        super().__init__()
        self.head = nn.Linear(input_dim, num_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class ActorCritic(nn.Module):
    def __init__(self, input_dim: int, num_actions: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.actor = nn.Linear(hidden_dim, num_actions)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.body(x)
        return self.actor(h), self.critic(h).squeeze(-1)


class TorchDQNAgent(BaseAgent):
    """Online DQN/DDQN trainer with replay buffer and action masks."""

    def __init__(
        self,
        name: str,
        num_channels: int,
        seed: int = 0,
        double_q: bool = False,
        esn: bool = False,
    ) -> None:
        super().__init__(name=name, num_channels=num_channels, seed=seed)
        if torch is None:
            raise RuntimeError(f"PyTorch is required for {name}: {_TORCH_IMPORT_ERROR}")
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        self.double_q = double_q
        self.esn = esn
        self.num_actions = 1 + 2 * num_channels
        self.device = _select_device()
        self.gamma = 0.97
        self.learning_rate = 1e-3 if not esn else 3e-3
        self.batch_size = 128
        self.warmup = 64
        self.update_every = 4
        self.target_update_every = 200
        self.epsilon_start = 0.25
        self.epsilon_end = 0.02
        self.epsilon_decay_steps = 3000
        self.replay = ReplayBuffer(100_000, self.rng)
        self.steps = 0
        self.input_dim: int | None = None
        self.train_dim: int | None = None
        self.q_net: nn.Module | None = None
        self.target_net: nn.Module | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.reservoir_size = 128
        self.reservoir_leak = 0.35
        self._reservoir_state: np.ndarray | None = None
        self._win: np.ndarray | None = None
        self._wres: np.ndarray | None = None
        self._last_train_state: np.ndarray | None = None

    def reset(self) -> None:
        if self.esn and self._reservoir_state is not None:
            self._reservoir_state.fill(0.0)

    def select_action(self, observation: np.ndarray, info: dict[str, Any]) -> int:
        self._ensure_initialized(observation)
        train_state = self._train_state(observation, update_reservoir=True)
        self._last_train_state = train_state
        mask = info.get("action_mask")
        epsilon = 0.0 if self.eval_mode else self._epsilon()
        if self.rng.random() < epsilon:
            valid = np.flatnonzero(mask) if mask is not None else np.arange(self.num_actions)
            return int(self.rng.choice(valid)) if len(valid) else 0
        with torch.no_grad():
            x = torch.as_tensor(train_state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.q_net(x).squeeze(0).detach().cpu().numpy()  # type: ignore[operator]
        if mask is not None:
            q_values = q_values.copy()
            q_values[~mask] = -np.inf
        return int(np.argmax(q_values))

    def observe(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        next_observation: np.ndarray,
        done: bool,
        info: dict[str, Any],
    ) -> None:
        self._ensure_initialized(observation)
        state = self._last_train_state
        if state is None:
            state = self._train_state(observation, update_reservoir=False)
        next_state = self._train_state(next_observation, update_reservoir=False)
        action_mask = np.asarray(info.get("action_mask", np.ones(self.num_actions, dtype=bool)), dtype=bool)
        next_action_mask = np.asarray(info.get("next_action_mask", np.ones(self.num_actions, dtype=bool)), dtype=bool)
        if self.eval_mode:
            return
        self.replay.add(
            Transition(
                state=state.astype(np.float32),
                action=action,
                reward=float(reward),
                next_state=next_state.astype(np.float32),
                done=bool(done),
                action_mask=action_mask,
                next_action_mask=next_action_mask,
            )
        )
        self.steps += 1
        if len(self.replay) >= max(self.warmup, self.batch_size) and self.steps % self.update_every == 0:
            self._update_q_network()
        if self.steps % self.target_update_every == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())  # type: ignore[union-attr]

    def _ensure_initialized(self, observation: np.ndarray) -> None:
        if self.q_net is not None:
            return
        self.input_dim = int(observation.size)
        if self.esn:
            self._reservoir_state = np.zeros(self.reservoir_size, dtype=np.float32)
            scale = 1.0 / max(self.input_dim, 1) ** 0.5
            self._win = self.rng.normal(0.0, scale, size=(self.reservoir_size, self.input_dim)).astype(np.float32)
            raw_wres = self.rng.normal(0.0, 0.08, size=(self.reservoir_size, self.reservoir_size)).astype(np.float32)
            eigvals = np.linalg.eigvals(raw_wres)
            radius = max(float(np.max(np.abs(eigvals))), 1e-6)
            self._wres = (raw_wres / radius * 0.85).astype(np.float32)
            self.train_dim = self.input_dim + self.reservoir_size
            self.q_net = LinearQHead(self.train_dim, self.num_actions).to(self.device)
            self.target_net = LinearQHead(self.train_dim, self.num_actions).to(self.device)
        else:
            self.train_dim = self.input_dim
            self.q_net = MLPQNetwork(self.train_dim, self.num_actions).to(self.device)
            self.target_net = MLPQNetwork(self.train_dim, self.num_actions).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=self.learning_rate)

    def _train_state(self, observation: np.ndarray, update_reservoir: bool) -> np.ndarray:
        obs = observation.astype(np.float32)
        if not self.esn:
            return obs
        assert self._reservoir_state is not None and self._win is not None and self._wres is not None
        previous = self._reservoir_state
        candidate = np.tanh(self._win @ obs + self._wres @ previous)
        next_reservoir = (1.0 - self.reservoir_leak) * previous + self.reservoir_leak * candidate
        if update_reservoir:
            self._reservoir_state = next_reservoir.astype(np.float32)
        return np.concatenate([obs, next_reservoir.astype(np.float32)])

    def _update_q_network(self) -> None:
        batch = self.replay.sample(self.batch_size)
        states = torch.as_tensor(np.stack([b.state for b in batch]), dtype=torch.float32, device=self.device)
        actions = torch.as_tensor([b.action for b in batch], dtype=torch.long, device=self.device)
        rewards = torch.as_tensor([b.reward for b in batch], dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(np.stack([b.next_state for b in batch]), dtype=torch.float32, device=self.device)
        dones = torch.as_tensor([b.done for b in batch], dtype=torch.float32, device=self.device)
        next_masks = torch.as_tensor(np.stack([b.next_action_mask for b in batch]), dtype=torch.bool, device=self.device)

        q = self.q_net(states).gather(1, actions[:, None]).squeeze(1)  # type: ignore[operator]
        with torch.no_grad():
            if self.double_q:
                online_next = self.q_net(next_states)  # type: ignore[operator]
                online_next = online_next.masked_fill(~next_masks, -1e9)
                next_actions = online_next.argmax(dim=1)
                target_next = self.target_net(next_states).gather(1, next_actions[:, None]).squeeze(1)  # type: ignore[operator]
            else:
                target_values = self.target_net(next_states)  # type: ignore[operator]
                target_values = target_values.masked_fill(~next_masks, -1e9)
                target_next = target_values.max(dim=1).values
            target = rewards + self.gamma * (1.0 - dones) * target_next
        loss = F.smooth_l1_loss(q, target)
        self.optimizer.zero_grad()  # type: ignore[union-attr]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 5.0)  # type: ignore[union-attr]
        self.optimizer.step()  # type: ignore[union-attr]

    def _epsilon(self) -> float:
        frac = min(self.steps / max(self.epsilon_decay_steps, 1), 1.0)
        return self.epsilon_start + frac * (self.epsilon_end - self.epsilon_start)


class TorchPPOAgent(BaseAgent):
    """Discrete PPO-style actor-critic trainer with action masks."""

    def __init__(self, name: str, num_channels: int, seed: int = 0) -> None:
        super().__init__(name=name, num_channels=num_channels, seed=seed)
        if torch is None:
            raise RuntimeError(f"PyTorch is required for {name}: {_TORCH_IMPORT_ERROR}")
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        self.num_actions = 1 + 2 * num_channels
        self.device = _select_device()
        self.gamma = 0.97
        self.lam = 0.92
        self.clip_range = 0.2
        self.learning_rate = 7e-4
        self.rollout_length = 128
        self.update_epochs = 3
        self.minibatch_size = 64
        self.model: ActorCritic | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.storage: list[dict[str, Any]] = []
        self._last_log_prob: float | None = None
        self._last_value: float | None = None

    def select_action(self, observation: np.ndarray, info: dict[str, Any]) -> int:
        self._ensure_initialized(observation)
        obs_t = torch.as_tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        mask = np.asarray(info.get("action_mask", np.ones(self.num_actions, dtype=bool)), dtype=bool)
        mask_t = torch.as_tensor(mask, dtype=torch.bool, device=self.device).unsqueeze(0)
        with torch.no_grad():
            logits, value = self.model(obs_t)  # type: ignore[operator]
            logits = logits.masked_fill(~mask_t, -1e9)
            dist = torch.distributions.Categorical(logits=logits)
            action = torch.argmax(logits, dim=1) if self.eval_mode else dist.sample()
            log_prob = dist.log_prob(action)
        self._last_log_prob = float(log_prob.item())
        self._last_value = float(value.item())
        return int(action.item())

    def observe(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        next_observation: np.ndarray,
        done: bool,
        info: dict[str, Any],
    ) -> None:
        self._ensure_initialized(observation)
        if self.eval_mode:
            return
        self.storage.append(
            {
                "obs": observation.astype(np.float32),
                "action": int(action),
                "reward": float(reward),
                "done": bool(done),
                "mask": np.asarray(info.get("action_mask", np.ones(self.num_actions, dtype=bool)), dtype=bool),
                "log_prob": float(self._last_log_prob or 0.0),
                "value": float(self._last_value or 0.0),
            }
        )
        if len(self.storage) >= self.rollout_length or done:
            self._update_policy(next_observation, done)
            self.storage.clear()

    def _ensure_initialized(self, observation: np.ndarray) -> None:
        if self.model is not None:
            return
        self.model = ActorCritic(int(observation.size), self.num_actions).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def _update_policy(self, next_observation: np.ndarray, done: bool) -> None:
        if not self.storage:
            return
        with torch.no_grad():
            if done:
                next_value = 0.0
            else:
                obs_t = torch.as_tensor(next_observation, dtype=torch.float32, device=self.device).unsqueeze(0)
                _, value_t = self.model(obs_t)  # type: ignore[operator]
                next_value = float(value_t.item())
        rewards = np.asarray([s["reward"] for s in self.storage], dtype=np.float32)
        dones = np.asarray([s["done"] for s in self.storage], dtype=np.float32)
        values = np.asarray([s["value"] for s in self.storage] + [next_value], dtype=np.float32)
        advantages = np.zeros_like(rewards)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            nonterminal = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * values[t + 1] * nonterminal - values[t]
            gae = delta + self.gamma * self.lam * nonterminal * gae
            advantages[t] = gae
        returns = advantages + values[:-1]
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        obs = torch.as_tensor(np.stack([s["obs"] for s in self.storage]), dtype=torch.float32, device=self.device)
        actions = torch.as_tensor([s["action"] for s in self.storage], dtype=torch.long, device=self.device)
        old_log_probs = torch.as_tensor([s["log_prob"] for s in self.storage], dtype=torch.float32, device=self.device)
        masks = torch.as_tensor(np.stack([s["mask"] for s in self.storage]), dtype=torch.bool, device=self.device)
        adv = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)
        ret = torch.as_tensor(returns, dtype=torch.float32, device=self.device)
        n = len(self.storage)
        for _ in range(self.update_epochs):
            order = self.rng.permutation(n)
            for start in range(0, n, self.minibatch_size):
                idx_np = order[start : start + self.minibatch_size]
                idx = torch.as_tensor(idx_np, dtype=torch.long, device=self.device)
                logits, values_t = self.model(obs[idx])  # type: ignore[operator]
                logits = logits.masked_fill(~masks[idx], -1e9)
                dist = torch.distributions.Categorical(logits=logits)
                log_probs = dist.log_prob(actions[idx])
                ratio = torch.exp(log_probs - old_log_probs[idx])
                pg1 = ratio * adv[idx]
                pg2 = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * adv[idx]
                policy_loss = -torch.min(pg1, pg2).mean()
                value_loss = F.mse_loss(values_t, ret[idx])
                entropy = dist.entropy().mean()
                loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
                self.optimizer.zero_grad()  # type: ignore[union-attr]
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)  # type: ignore[union-attr]
                self.optimizer.step()  # type: ignore[union-attr]
