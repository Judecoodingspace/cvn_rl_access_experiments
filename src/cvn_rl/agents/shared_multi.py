from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np

from cvn_rl.agents.fixed import FixedOverlayAgent, FixedUnderlayAgent
from cvn_rl.agents.torch_agents import (
    ActorCritic,
    LinearQHead,
    MLPQNetwork,
    ReplayBuffer,
    Transition,
    _select_device,
    torch,
    F,
)


@dataclass
class RuntimeSnapshot:
    eval_mode: bool
    rng_state: dict[str, Any]
    payload: dict[str, Any]


class SharedPolicyAgent:
    def __init__(self, name: str, num_channels: int, num_vehicles: int, seed: int) -> None:
        self.name = name
        self.num_channels = num_channels
        self.num_vehicles = num_vehicles
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.eval_mode = False
        self.env_steps = 0

    @property
    def learned(self) -> bool:
        return True

    def set_eval_mode(self, enabled: bool) -> None:
        self.eval_mode = enabled

    def reset_episode(self) -> None:
        pass

    def select_actions(self, observations: np.ndarray, infos: list[dict[str, Any]]) -> np.ndarray:
        raise NotImplementedError

    def observe_batch(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_observations: np.ndarray,
        done: bool,
        infos: list[dict[str, Any]],
        next_infos: list[dict[str, Any]],
    ) -> None:
        raise NotImplementedError

    def snapshot_runtime_state(self) -> RuntimeSnapshot:
        return RuntimeSnapshot(
            eval_mode=self.eval_mode,
            rng_state=copy.deepcopy(self.rng.bit_generator.state),
            payload={},
        )

    def restore_runtime_state(self, snapshot: RuntimeSnapshot) -> None:
        self.eval_mode = snapshot.eval_mode
        self.rng.bit_generator.state = copy.deepcopy(snapshot.rng_state)

    def parameter_hash(self) -> str:
        return hashlib.sha256(self.name.encode("utf-8")).hexdigest()[:16]

    def training_diagnostics(self) -> dict[str, Any]:
        return {}


@dataclass
class SequenceTransition:
    observation: np.ndarray
    action: int
    reward: float
    next_observation: np.ndarray
    done: bool
    action_mask: np.ndarray
    next_action_mask: np.ndarray
    episode_id: int


class SequenceReplayBuffer:
    """Replay buffer of contiguous, episode-safe per-vehicle windows."""

    def __init__(self, capacity: int, window_length: int, num_streams: int,
                 rng: np.random.Generator) -> None:
        self.capacity = int(capacity)
        self.window_length = int(window_length)
        self.num_streams = int(num_streams)
        self.rng = rng
        self.streams: dict[int, list[SequenceTransition]] = {
            stream: [] for stream in range(self.num_streams)
        }
        self.candidates: list[tuple[int, int]] = []

    def __len__(self) -> int:
        return sum(len(stream) for stream in self.streams.values())

    @property
    def num_windows(self) -> int:
        return len(self.candidates)

    def add(self, stream_id: int, transition: SequenceTransition) -> None:
        stream_id = int(stream_id)
        if stream_id not in self.streams:
            raise ValueError(f"unknown replay stream: {stream_id}")
        if len(self) >= self.capacity:
            self._evict_oldest()
        stream = self.streams[stream_id]
        stream.append(transition)
        start = len(stream) - self.window_length
        if start >= 0 and self._valid_window(stream, start):
            self.candidates.append((stream_id, start))

    def sample(self, batch_size: int) -> list[list[SequenceTransition]]:
        if not self.candidates:
            raise ValueError("sequence replay has no complete windows")
        indices = self.rng.choice(
            len(self.candidates), size=int(batch_size),
            replace=len(self.candidates) < int(batch_size),
        )
        windows = []
        for index in np.atleast_1d(indices):
            stream_id, start = self.candidates[int(index)]
            windows.append(self.streams[stream_id][start:start + self.window_length])
        return windows

    def _valid_window(self, stream: list[SequenceTransition], start: int) -> bool:
        window = stream[start:start + self.window_length]
        if len(window) != self.window_length:
            return False
        episode = window[0].episode_id
        if any(item.episode_id != episode for item in window):
            return False
        return not any(item.done for item in window[:-1])

    def _evict_oldest(self) -> None:
        nonempty = [key for key, stream in self.streams.items() if stream]
        if not nonempty:
            return
        # Evict in chunks of ~1% of capacity so candidate rebuilds are
        # amortized. With per-stream writes (N streams => N adds per env
        # slot), one-transition-per-add eviction forces a full
        # O(capacity x window_length) rebuild on every add once the buffer
        # is full, which made N=10 LSTM training effectively stall.
        chunk = max(1, self.capacity // 100)
        for _ in range(chunk):
            nonempty = [key for key, stream in self.streams.items() if stream]
            if not nonempty:
                break
            stream_id = max(nonempty, key=lambda key: len(self.streams[key]))
            self.streams[stream_id].pop(0)
        self._rebuild_candidates()

    def _rebuild_candidates(self) -> None:
        self.candidates.clear()
        for stream_id, stream in self.streams.items():
            limit = len(stream) - self.window_length + 1
            for start in range(max(limit, 0)):
                if self._valid_window(stream, start):
                    self.candidates.append((stream_id, start))


class LSTMQNetwork(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_size: int, num_actions: int) -> None:
        super().__init__()
        self.lstm = torch.nn.LSTM(
            input_size=int(input_dim), hidden_size=int(hidden_size),
            num_layers=1, batch_first=True,
        )
        self.head = torch.nn.Linear(int(hidden_size), int(num_actions))

    def forward(self, inputs, hidden=None):
        encoded, hidden_out = self.lstm(inputs, hidden)
        return self.head(encoded), hidden_out


class DiscreteActor(torch.nn.Module):
    def __init__(self, input_dim: int, num_actions: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim), torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim), torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, num_actions),
        )

    def forward(self, inputs):
        return self.net(inputs)


class DiscreteQNetwork(DiscreteActor):
    pass


class SharedDQNAgent(SharedPolicyAgent):
    def __init__(
        self,
        name: str,
        num_channels: int,
        num_vehicles: int,
        seed: int,
        *,
        double_q: bool,
        esn: bool,
        q_head: str = "auto",
        learning_rate: float | None = None,
        reservoir_size: int = 128,
        reservoir_leak: float = 0.35,
        reservoir_radius: float = 0.85,
        gamma: float = 0.97,
    ) -> None:
        super().__init__(name, num_channels, num_vehicles, seed)
        if torch is None:
            raise RuntimeError("PyTorch is required for shared DQN agents")
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        self.double_q = double_q
        self.esn = esn
        self.q_head = ("linear" if esn else "mlp") if q_head == "auto" else q_head
        if self.q_head not in {"linear", "mlp"}:
            raise ValueError(f"unsupported Q head: {self.q_head}")
        self.num_actions = 1 + 2 * num_channels
        self.device = _select_device()
        gamma_float = float(gamma)
        if not (0.0 < gamma_float <= 1.0):
            raise ValueError(f"gamma must be in (0, 1], got {gamma_float}")
        self.gamma = gamma_float
        self.learning_rate = float(learning_rate if learning_rate is not None else (3e-3 if esn else 1e-3))
        self.batch_size = 128
        self.warmup = 64
        self.update_every = 4
        self.target_update_every = 200
        self.epsilon_start = 0.25
        self.epsilon_end = 0.02
        self.epsilon_decay_steps = 10_000
        self.replay = ReplayBuffer(100_000, self.rng)
        self.input_dim: int | None = None
        self.train_dim: int | None = None
        self.q_net = None
        self.target_net = None
        self.optimizer = None
        self.reservoir_size = int(reservoir_size)
        self.reservoir_leak = float(reservoir_leak)
        self.reservoir_radius = float(reservoir_radius)
        self.reservoir_states: np.ndarray | None = None
        self.win: np.ndarray | None = None
        self.wres: np.ndarray | None = None
        self.last_train_states: np.ndarray | None = None

    def reset_episode(self) -> None:
        if self.reservoir_states is not None:
            self.reservoir_states.fill(0.0)
        self.last_train_states = None

    def select_actions(self, observations: np.ndarray, infos: list[dict[str, Any]]) -> np.ndarray:
        self._ensure_initialized(observations)
        train_states = self._encode_states(observations, update_reservoir=True)
        self.last_train_states = train_states.copy()
        masks = np.stack([np.asarray(info["action_mask"], dtype=bool) for info in infos])
        with torch.no_grad():
            inputs = torch.as_tensor(train_states, dtype=torch.float32, device=self.device)
            q_values = self.q_net(inputs).detach().cpu().numpy()
        q_values = q_values.copy()
        q_values[~masks] = -np.inf
        actions = np.argmax(q_values, axis=1).astype(int)
        epsilon = 0.0 if self.eval_mode else self._epsilon()
        if epsilon > 0.0:
            explore = self.rng.random(self.num_vehicles) < epsilon
            for vehicle in np.flatnonzero(explore):
                valid = np.flatnonzero(masks[vehicle])
                actions[vehicle] = int(self.rng.choice(valid)) if len(valid) else 0
        return actions

    def observe_batch(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_observations: np.ndarray,
        done: bool,
        infos: list[dict[str, Any]],
        next_infos: list[dict[str, Any]],
    ) -> None:
        if self.eval_mode:
            return
        self._ensure_initialized(observations)
        states = self.last_train_states
        if states is None:
            states = self._encode_states(observations, update_reservoir=False)
        next_states = self._encode_states(next_observations, update_reservoir=False)
        for vehicle in range(self.num_vehicles):
            self.replay.add(
                Transition(
                    state=states[vehicle].astype(np.float32),
                    action=int(actions[vehicle]),
                    reward=float(rewards[vehicle]),
                    next_state=next_states[vehicle].astype(np.float32),
                    done=bool(done),
                    action_mask=np.asarray(infos[vehicle]["action_mask"], dtype=bool),
                    next_action_mask=np.asarray(next_infos[vehicle]["action_mask"], dtype=bool),
                )
            )
        self.env_steps += 1
        if len(self.replay) >= max(self.warmup, self.batch_size) and self.env_steps % self.update_every == 0:
            self._update_network()
        if self.env_steps % self.target_update_every == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

    def snapshot_runtime_state(self) -> RuntimeSnapshot:
        snapshot = super().snapshot_runtime_state()
        snapshot.payload = {
            "reservoir_states": None if self.reservoir_states is None else self.reservoir_states.copy(),
            "last_train_states": None if self.last_train_states is None else self.last_train_states.copy(),
        }
        return snapshot

    def restore_runtime_state(self, snapshot: RuntimeSnapshot) -> None:
        super().restore_runtime_state(snapshot)
        reservoir = snapshot.payload.get("reservoir_states")
        last = snapshot.payload.get("last_train_states")
        self.reservoir_states = None if reservoir is None else reservoir.copy()
        self.last_train_states = None if last is None else last.copy()

    def parameter_hash(self) -> str:
        return _torch_module_hash(self.q_net, self.name)

    def _ensure_initialized(self, observations: np.ndarray) -> None:
        if self.q_net is not None:
            return
        # Construction is lazy, so restore this policy's own torch seed here.
        # This keeps independently created per-VT networks deterministic and
        # prevents them from inheriting another policy's global RNG state.
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        self.input_dim = int(observations.shape[1])
        if self.esn:
            self.reservoir_states = np.zeros((self.num_vehicles, self.reservoir_size), dtype=np.float32)
            scale = 1.0 / max(self.input_dim, 1) ** 0.5
            self.win = self.rng.normal(0.0, scale, size=(self.reservoir_size, self.input_dim)).astype(np.float32)
            raw = self.rng.normal(0.0, 0.08, size=(self.reservoir_size, self.reservoir_size)).astype(np.float32)
            vector = self.rng.normal(size=self.reservoir_size).astype(np.float32)
            vector /= max(float(np.linalg.norm(vector)), 1e-6)
            for _ in range(24):
                vector = raw @ vector
                vector /= max(float(np.linalg.norm(vector)), 1e-6)
            radius = max(float(np.linalg.norm(raw @ vector)), 1e-6)
            self.wres = (raw / radius * self.reservoir_radius).astype(np.float32)
            self.train_dim = self.input_dim + self.reservoir_size
        else:
            self.train_dim = self.input_dim
        if self.q_head == "linear":
            self.q_net = LinearQHead(self.train_dim, self.num_actions).to(self.device)
            self.target_net = LinearQHead(self.train_dim, self.num_actions).to(self.device)
        else:
            self.q_net = MLPQNetwork(self.train_dim, self.num_actions).to(self.device)
            self.target_net = MLPQNetwork(self.train_dim, self.num_actions).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=self.learning_rate)

    def _encode_states(self, observations: np.ndarray, update_reservoir: bool) -> np.ndarray:
        obs = observations.astype(np.float32)
        if not self.esn:
            return obs
        assert self.reservoir_states is not None and self.win is not None and self.wres is not None
        candidate = np.tanh(obs @ self.win.T + self.reservoir_states @ self.wres.T)
        next_states = (1.0 - self.reservoir_leak) * self.reservoir_states + self.reservoir_leak * candidate
        if update_reservoir:
            self.reservoir_states = next_states.astype(np.float32)
        return np.concatenate([obs, next_states.astype(np.float32)], axis=1)

    def _update_network(self) -> None:
        batch = self.replay.sample(self.batch_size)
        states = torch.as_tensor(np.stack([item.state for item in batch]), dtype=torch.float32, device=self.device)
        actions = torch.as_tensor([item.action for item in batch], dtype=torch.long, device=self.device)
        rewards = torch.as_tensor([item.reward for item in batch], dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(np.stack([item.next_state for item in batch]), dtype=torch.float32, device=self.device)
        dones = torch.as_tensor([item.done for item in batch], dtype=torch.float32, device=self.device)
        next_masks = torch.as_tensor(np.stack([item.next_action_mask for item in batch]), dtype=torch.bool, device=self.device)
        q_values = self.q_net(states).gather(1, actions[:, None]).squeeze(1)
        with torch.no_grad():
            if self.double_q:
                online = self.q_net(next_states).masked_fill(~next_masks, -1e9)
                next_actions = online.argmax(dim=1)
                target_next = self.target_net(next_states).gather(1, next_actions[:, None]).squeeze(1)
            else:
                target_next = self.target_net(next_states).masked_fill(~next_masks, -1e9).max(dim=1).values
            target = rewards + self.gamma * (1.0 - dones) * target_next
        loss = F.smooth_l1_loss(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 5.0)
        self.optimizer.step()

    def _epsilon(self) -> float:
        fraction = min(self.env_steps / max(self.epsilon_decay_steps, 1), 1.0)
        return self.epsilon_start + fraction * (self.epsilon_end - self.epsilon_start)

    def training_diagnostics(self) -> dict[str, Any]:
        return {
            "gamma": self.gamma,
            "epsilon": float(self._epsilon()),
        }


class SharedLSTMDDQNAgent(SharedPolicyAgent):
    """Double DQN with a trainable recurrent encoder and sequence replay."""

    def __init__(self, name: str, num_channels: int, num_vehicles: int, seed: int,
                 *, hidden_size: int = 128, sequence_length: int = 32,
                 burn_in: int = 8, learning_rate: float = 1e-3,
                 warmup: int = 512, batch_size: int = 128) -> None:
        super().__init__(name, num_channels, num_vehicles, seed)
        if torch is None:
            raise RuntimeError("PyTorch is required for LSTM-DDQN")
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        self.num_actions = 1 + 2 * num_channels
        self.device = _select_device()
        self.hidden_size = int(hidden_size)
        self.sequence_length = int(sequence_length)
        self.burn_in = int(burn_in)
        self.window_length = self.sequence_length + self.burn_in
        self.gamma = 0.97
        self.learning_rate = float(learning_rate)
        self.batch_size = int(batch_size)
        self.warmup = int(warmup)
        self.update_every = 4
        self.target_update_every = 200
        self.epsilon_start = 0.25
        self.epsilon_end = 0.02
        self.epsilon_decay_steps = 10_000
        self.replay = SequenceReplayBuffer(
            100_000, self.window_length, num_vehicles, self.rng,
        )
        self.input_dim: int | None = None
        self.q_net = None
        self.target_net = None
        self.optimizer = None
        self.hidden = None
        self.episode_id = 0
        self.initial_parameter_hash = ""
        self.last_q_loss = float("nan")
        self.gradient_updates = 0

    def reset_episode(self) -> None:
        self.hidden = None
        self.episode_id += 1

    def select_actions(self, observations: np.ndarray,
                       infos: list[dict[str, Any]]) -> np.ndarray:
        self._ensure_initialized(observations)
        masks = np.stack([np.asarray(info["action_mask"], dtype=bool) for info in infos])
        inputs = torch.as_tensor(
            observations[:, None, :], dtype=torch.float32, device=self.device,
        )
        with torch.no_grad():
            q_sequence, hidden = self.q_net(inputs, self.hidden)
        self.hidden = tuple(value.detach() for value in hidden)
        q_values = q_sequence[:, 0, :].detach().cpu().numpy()
        q_values[~masks] = -np.inf
        actions = np.argmax(q_values, axis=1).astype(int)
        epsilon = 0.0 if self.eval_mode else self._epsilon()
        for vehicle in np.flatnonzero(self.rng.random(self.num_vehicles) < epsilon):
            valid = np.flatnonzero(masks[vehicle])
            actions[vehicle] = int(self.rng.choice(valid)) if len(valid) else 0
        return actions

    def observe_batch(self, observations: np.ndarray, actions: np.ndarray,
                      rewards: np.ndarray, next_observations: np.ndarray,
                      done: bool, infos: list[dict[str, Any]],
                      next_infos: list[dict[str, Any]]) -> None:
        if self.eval_mode:
            return
        for vehicle in range(self.num_vehicles):
            self.replay.add(vehicle, SequenceTransition(
                observation=observations[vehicle].astype(np.float32),
                action=int(actions[vehicle]), reward=float(rewards[vehicle]),
                next_observation=next_observations[vehicle].astype(np.float32),
                done=bool(done),
                action_mask=np.asarray(infos[vehicle]["action_mask"], dtype=bool),
                next_action_mask=np.asarray(next_infos[vehicle]["action_mask"], dtype=bool),
                episode_id=self.episode_id,
            ))
        self.env_steps += 1
        if (len(self.replay) >= self.warmup and self.replay.num_windows > 0
                and self.env_steps % self.update_every == 0):
            self._update_network()
        if self.env_steps % self.target_update_every == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

    def snapshot_runtime_state(self) -> RuntimeSnapshot:
        snapshot = super().snapshot_runtime_state()
        snapshot.payload = {
            "episode_id": self.episode_id,
            "hidden": None if self.hidden is None else tuple(
                value.detach().cpu().numpy().copy() for value in self.hidden
            ),
        }
        return snapshot

    def restore_runtime_state(self, snapshot: RuntimeSnapshot) -> None:
        super().restore_runtime_state(snapshot)
        self.episode_id = int(snapshot.payload.get("episode_id", self.episode_id))
        hidden = snapshot.payload.get("hidden")
        self.hidden = None if hidden is None else tuple(
            torch.as_tensor(value, dtype=torch.float32, device=self.device)
            for value in hidden
        )

    def parameter_hash(self) -> str:
        return _torch_module_hash(self.q_net, self.name)

    def training_diagnostics(self) -> dict[str, Any]:
        return {
            "training_initial_parameter_hash": self.initial_parameter_hash,
            "q_loss": float(self.last_q_loss),
            "gradient_updates": int(self.gradient_updates),
            "sequence_replay_windows": int(self.replay.num_windows),
        }

    def _ensure_initialized(self, observations: np.ndarray) -> None:
        if self.q_net is not None:
            return
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        self.input_dim = int(observations.shape[1])
        self.q_net = LSTMQNetwork(
            self.input_dim, self.hidden_size, self.num_actions,
        ).to(self.device)
        self.target_net = LSTMQNetwork(
            self.input_dim, self.hidden_size, self.num_actions,
        ).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = torch.optim.Adam(
            self.q_net.parameters(), lr=self.learning_rate,
        )
        self.initial_parameter_hash = _torch_module_hash(self.q_net, self.name)

    def _update_network(self) -> None:
        windows = self.replay.sample(self.batch_size)
        observations = np.stack([
            np.stack([item.observation for item in window]) for window in windows
        ])
        final_next = np.stack([window[-1].next_observation for window in windows])
        state_sequence = np.concatenate(
            [observations, final_next[:, None, :]], axis=1,
        )
        actions = np.asarray([
            [item.action for item in window[self.burn_in:]] for window in windows
        ], dtype=np.int64)
        rewards = np.asarray([
            [item.reward for item in window[self.burn_in:]] for window in windows
        ], dtype=np.float32)
        dones = np.asarray([
            [item.done for item in window[self.burn_in:]] for window in windows
        ], dtype=np.float32)
        next_masks = np.asarray([
            [item.next_action_mask for item in window[self.burn_in:]]
            for window in windows
        ], dtype=bool)
        states_t = torch.as_tensor(state_sequence, dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.long, device=self.device)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        dones_t = torch.as_tensor(dones, dtype=torch.float32, device=self.device)
        masks_t = torch.as_tensor(next_masks, dtype=torch.bool, device=self.device)
        with torch.no_grad():
            _, online_hidden = self.q_net(states_t[:, :self.burn_in, :])
            _, target_hidden = self.target_net(states_t[:, :self.burn_in, :])
        online_values, _ = self.q_net(
            states_t[:, self.burn_in:, :],
            tuple(value.detach() for value in online_hidden),
        )
        current_q = online_values[:, :-1, :].gather(
            2, actions_t.unsqueeze(-1),
        ).squeeze(-1)
        with torch.no_grad():
            online_next = online_values[:, 1:, :].detach().masked_fill(~masks_t, -1e9)
            next_actions = online_next.argmax(dim=2, keepdim=True)
            target_values, _ = self.target_net(
                states_t[:, self.burn_in:, :],
                tuple(value.detach() for value in target_hidden),
            )
            target_next = target_values[:, 1:, :].gather(2, next_actions).squeeze(-1)
            target = rewards_t + self.gamma * (1.0 - dones_t) * target_next
        loss = F.smooth_l1_loss(current_q, target)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 5.0)
        self.optimizer.step()
        self.last_q_loss = float(loss.detach().cpu().item())
        self.gradient_updates += 1

    def _epsilon(self) -> float:
        fraction = min(self.env_steps / max(self.epsilon_decay_steps, 1), 1.0)
        return self.epsilon_start + fraction * (self.epsilon_end - self.epsilon_start)


class SharedDiscreteSACAgent(SharedPolicyAgent):
    """Categorical Soft Actor-Critic for the masked discrete action space."""

    def __init__(self, name: str, num_channels: int, num_vehicles: int, seed: int,
                 *, hidden_dim: int = 128, learning_rate: float = 3e-4,
                 tau: float = 0.005, initial_alpha: float = 0.2,
                 target_entropy_scale: float = 0.98,
                 warmup: int = 512, batch_size: int = 128,
                 update_every: int = 4) -> None:
        super().__init__(name, num_channels, num_vehicles, seed)
        if torch is None:
            raise RuntimeError("PyTorch is required for discrete SAC")
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        self.num_actions = 1 + 2 * num_channels
        self.device = _select_device()
        self.hidden_dim = int(hidden_dim)
        self.learning_rate = float(learning_rate)
        self.tau = float(tau)
        self.target_entropy_scale = float(target_entropy_scale)
        self.gamma = 0.97
        self.batch_size = int(batch_size)
        self.warmup = int(warmup)
        self.update_every = int(update_every)
        self.replay = ReplayBuffer(100_000, self.rng)
        self.actor = None
        self.q1 = None
        self.q2 = None
        self.target_q1 = None
        self.target_q2 = None
        self.model = None
        self.actor_optimizer = None
        self.q1_optimizer = None
        self.q2_optimizer = None
        self.log_alpha = torch.tensor(
            np.log(initial_alpha), dtype=torch.float32, device=self.device,
            requires_grad=True,
        )
        self.alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=self.learning_rate,
        )
        self.initial_parameter_hash = ""
        self.last_actor_loss = float("nan")
        self.last_critic_loss = float("nan")
        self.last_policy_entropy = float("nan")
        self.gradient_updates = 0

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_actions(self, observations: np.ndarray,
                       infos: list[dict[str, Any]]) -> np.ndarray:
        self._ensure_initialized(observations)
        masks = np.stack([np.asarray(info["action_mask"], dtype=bool) for info in infos])
        obs_t = torch.as_tensor(observations, dtype=torch.float32, device=self.device)
        mask_t = torch.as_tensor(masks, dtype=torch.bool, device=self.device)
        with torch.no_grad():
            logits = self.actor(obs_t).masked_fill(~mask_t, -1e9)
            if self.eval_mode:
                actions = logits.argmax(dim=1)
            else:
                actions = torch.distributions.Categorical(logits=logits).sample()
        return actions.detach().cpu().numpy().astype(int)

    def observe_batch(self, observations: np.ndarray, actions: np.ndarray,
                      rewards: np.ndarray, next_observations: np.ndarray,
                      done: bool, infos: list[dict[str, Any]],
                      next_infos: list[dict[str, Any]]) -> None:
        if self.eval_mode:
            return
        self._ensure_initialized(observations)
        for vehicle in range(self.num_vehicles):
            self.replay.add(Transition(
                state=observations[vehicle].astype(np.float32),
                action=int(actions[vehicle]), reward=float(rewards[vehicle]),
                next_state=next_observations[vehicle].astype(np.float32),
                done=bool(done),
                action_mask=np.asarray(infos[vehicle]["action_mask"], dtype=bool),
                next_action_mask=np.asarray(next_infos[vehicle]["action_mask"], dtype=bool),
            ))
        self.env_steps += 1
        if (len(self.replay) >= max(self.warmup, self.batch_size)
                and self.env_steps % self.update_every == 0):
            self._update_networks()

    def parameter_hash(self) -> str:
        digest = hashlib.sha256()
        digest.update(_torch_module_hash(self.model, self.name).encode("utf-8"))
        digest.update(self.log_alpha.detach().cpu().numpy().tobytes())
        return digest.hexdigest()[:16]

    def training_diagnostics(self) -> dict[str, Any]:
        return {
            "training_initial_parameter_hash": self.initial_parameter_hash,
            "actor_loss": float(self.last_actor_loss),
            "critic_loss": float(self.last_critic_loss),
            "policy_entropy": float(self.last_policy_entropy),
            "entropy_temperature": float(self.alpha.detach().cpu().item()),
            "gradient_updates": int(self.gradient_updates),
        }

    def _ensure_initialized(self, observations: np.ndarray) -> None:
        if self.actor is not None:
            return
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        input_dim = int(observations.shape[1])
        self.actor = DiscreteActor(input_dim, self.num_actions, self.hidden_dim).to(self.device)
        self.q1 = DiscreteQNetwork(input_dim, self.num_actions, self.hidden_dim).to(self.device)
        self.q2 = DiscreteQNetwork(input_dim, self.num_actions, self.hidden_dim).to(self.device)
        self.target_q1 = DiscreteQNetwork(input_dim, self.num_actions, self.hidden_dim).to(self.device)
        self.target_q2 = DiscreteQNetwork(input_dim, self.num_actions, self.hidden_dim).to(self.device)
        self.target_q1.load_state_dict(self.q1.state_dict())
        self.target_q2.load_state_dict(self.q2.state_dict())
        self.model = torch.nn.ModuleList([self.actor, self.q1, self.q2])
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=self.learning_rate,
        )
        self.q1_optimizer = torch.optim.Adam(self.q1.parameters(), lr=self.learning_rate)
        self.q2_optimizer = torch.optim.Adam(self.q2.parameters(), lr=self.learning_rate)
        self.initial_parameter_hash = self.parameter_hash()

    @staticmethod
    def _masked_policy(logits, masks):
        masked_logits = logits.masked_fill(~masks, -1e9)
        log_probabilities = torch.log_softmax(masked_logits, dim=-1)
        probabilities = torch.softmax(masked_logits, dim=-1)
        probabilities = probabilities * masks.to(probabilities.dtype)
        probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        return probabilities, log_probabilities

    def _update_networks(self) -> None:
        batch = self.replay.sample(self.batch_size)
        states = torch.as_tensor(
            np.stack([item.state for item in batch]), dtype=torch.float32, device=self.device,
        )
        actions = torch.as_tensor(
            [item.action for item in batch], dtype=torch.long, device=self.device,
        )
        rewards = torch.as_tensor(
            [item.reward for item in batch], dtype=torch.float32, device=self.device,
        )
        next_states = torch.as_tensor(
            np.stack([item.next_state for item in batch]), dtype=torch.float32, device=self.device,
        )
        dones = torch.as_tensor(
            [item.done for item in batch], dtype=torch.float32, device=self.device,
        )
        masks = torch.as_tensor(
            np.stack([item.action_mask for item in batch]), dtype=torch.bool, device=self.device,
        )
        next_masks = torch.as_tensor(
            np.stack([item.next_action_mask for item in batch]), dtype=torch.bool, device=self.device,
        )
        with torch.no_grad():
            next_prob, next_log_prob = self._masked_policy(
                self.actor(next_states), next_masks,
            )
            target_q = torch.minimum(
                self.target_q1(next_states), self.target_q2(next_states),
            )
            next_value = (
                next_prob * (target_q - self.alpha.detach() * next_log_prob)
            ).sum(dim=1)
            target = rewards + self.gamma * (1.0 - dones) * next_value
        q1_selected = self.q1(states).gather(1, actions[:, None]).squeeze(1)
        q2_selected = self.q2(states).gather(1, actions[:, None]).squeeze(1)
        q1_loss = F.smooth_l1_loss(q1_selected, target)
        q2_loss = F.smooth_l1_loss(q2_selected, target)
        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q1.parameters(), 5.0)
        self.q1_optimizer.step()
        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q2.parameters(), 5.0)
        self.q2_optimizer.step()

        probabilities, log_probabilities = self._masked_policy(
            self.actor(states), masks,
        )
        with torch.no_grad():
            minimum_q = torch.minimum(self.q1(states), self.q2(states))
        actor_loss = (
            probabilities * (self.alpha.detach() * log_probabilities - minimum_q)
        ).sum(dim=1).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 5.0)
        self.actor_optimizer.step()

        entropy = -(probabilities.detach() * log_probabilities.detach()).sum(dim=1)
        valid_actions = masks.sum(dim=1).to(torch.float32).clamp_min(1.0)
        target_entropy = self.target_entropy_scale * torch.log(valid_actions)
        alpha_loss = (self.log_alpha * (entropy - target_entropy)).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        with torch.no_grad():
            self.log_alpha.clamp_(np.log(1e-4), np.log(100.0))

        with torch.no_grad():
            for target_parameter, parameter in zip(
                self.target_q1.parameters(), self.q1.parameters(),
            ):
                target_parameter.mul_(1.0 - self.tau).add_(parameter, alpha=self.tau)
            for target_parameter, parameter in zip(
                self.target_q2.parameters(), self.q2.parameters(),
            ):
                target_parameter.mul_(1.0 - self.tau).add_(parameter, alpha=self.tau)
        self.last_actor_loss = float(actor_loss.detach().cpu().item())
        self.last_critic_loss = float((q1_loss + q2_loss).detach().cpu().item())
        self.last_policy_entropy = float(entropy.mean().cpu().item())
        self.gradient_updates += 1


class SharedPPOAgent(SharedPolicyAgent):
    def __init__(self, name: str, num_channels: int, num_vehicles: int, seed: int) -> None:
        super().__init__(name, num_channels, num_vehicles, seed)
        if torch is None:
            raise RuntimeError("PyTorch is required for shared PPO")
        torch.manual_seed(seed)
        self.num_actions = 1 + 2 * num_channels
        self.device = _select_device()
        self.gamma = 0.97
        self.lam = 0.92
        self.clip_range = 0.2
        self.learning_rate = 7e-4
        self.rollout_length = 128
        self.update_epochs = 3
        self.minibatch_size = 64
        self.model = None
        self.optimizer = None
        self.storage: list[dict[str, Any]] = []
        self.last_log_probs: np.ndarray | None = None
        self.last_values: np.ndarray | None = None

    def reset_episode(self) -> None:
        self.last_log_probs = None
        self.last_values = None

    def select_actions(self, observations: np.ndarray, infos: list[dict[str, Any]]) -> np.ndarray:
        self._ensure_initialized(observations)
        masks = np.stack([np.asarray(info["action_mask"], dtype=bool) for info in infos])
        obs_t = torch.as_tensor(observations, dtype=torch.float32, device=self.device)
        mask_t = torch.as_tensor(masks, dtype=torch.bool, device=self.device)
        with torch.no_grad():
            logits, values = self.model(obs_t)
            logits = logits.masked_fill(~mask_t, -1e9)
            distribution = torch.distributions.Categorical(logits=logits)
            actions = torch.argmax(logits, dim=1) if self.eval_mode else distribution.sample()
            log_probs = distribution.log_prob(actions)
        self.last_log_probs = log_probs.detach().cpu().numpy()
        self.last_values = values.detach().cpu().numpy()
        return actions.detach().cpu().numpy().astype(int)

    def observe_batch(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_observations: np.ndarray,
        done: bool,
        infos: list[dict[str, Any]],
        next_infos: list[dict[str, Any]],
    ) -> None:
        if self.eval_mode:
            return
        assert self.last_log_probs is not None and self.last_values is not None
        self.storage.append(
            {
                "obs": observations.astype(np.float32),
                "actions": actions.astype(int),
                "rewards": rewards.astype(np.float32),
                "done": bool(done),
                "masks": np.stack([np.asarray(info["action_mask"], dtype=bool) for info in infos]),
                "log_probs": self.last_log_probs.astype(np.float32),
                "values": self.last_values.astype(np.float32),
            }
        )
        self.env_steps += 1
        if len(self.storage) >= self.rollout_length or done:
            self._update_policy(next_observations, done)
            self.storage.clear()

    def snapshot_runtime_state(self) -> RuntimeSnapshot:
        snapshot = super().snapshot_runtime_state()
        snapshot.payload = {
            "last_log_probs": None if self.last_log_probs is None else self.last_log_probs.copy(),
            "last_values": None if self.last_values is None else self.last_values.copy(),
        }
        return snapshot

    def restore_runtime_state(self, snapshot: RuntimeSnapshot) -> None:
        super().restore_runtime_state(snapshot)
        log_probs = snapshot.payload.get("last_log_probs")
        values = snapshot.payload.get("last_values")
        self.last_log_probs = None if log_probs is None else log_probs.copy()
        self.last_values = None if values is None else values.copy()

    def parameter_hash(self) -> str:
        return _torch_module_hash(self.model, self.name)

    def _ensure_initialized(self, observations: np.ndarray) -> None:
        if self.model is not None:
            return
        self.model = ActorCritic(int(observations.shape[1]), self.num_actions).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def _update_policy(self, next_observations: np.ndarray, done: bool) -> None:
        if not self.storage:
            return
        with torch.no_grad():
            if done:
                bootstrap = np.zeros(self.num_vehicles, dtype=np.float32)
            else:
                obs_t = torch.as_tensor(next_observations, dtype=torch.float32, device=self.device)
                _, values = self.model(obs_t)
                bootstrap = values.detach().cpu().numpy().astype(np.float32)
        rewards = np.stack([slot["rewards"] for slot in self.storage])
        values = np.stack([slot["values"] for slot in self.storage])
        dones = np.asarray([slot["done"] for slot in self.storage], dtype=np.float32)
        advantages = np.zeros_like(rewards)
        gae = np.zeros(self.num_vehicles, dtype=np.float32)
        next_values = bootstrap
        for t in reversed(range(len(self.storage))):
            nonterminal = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * next_values * nonterminal - values[t]
            gae = delta + self.gamma * self.lam * nonterminal * gae
            advantages[t] = gae
            next_values = values[t]
        returns = advantages + values
        flat_advantages = advantages.reshape(-1)
        if flat_advantages.size > 1:
            flat_advantages = (flat_advantages - flat_advantages.mean()) / (flat_advantages.std() + 1e-8)
        obs = torch.as_tensor(np.stack([slot["obs"] for slot in self.storage]).reshape(-1, self.storage[0]["obs"].shape[1]), dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(np.stack([slot["actions"] for slot in self.storage]).reshape(-1), dtype=torch.long, device=self.device)
        old_log_probs = torch.as_tensor(np.stack([slot["log_probs"] for slot in self.storage]).reshape(-1), dtype=torch.float32, device=self.device)
        masks = torch.as_tensor(np.stack([slot["masks"] for slot in self.storage]).reshape(-1, self.num_actions), dtype=torch.bool, device=self.device)
        adv = torch.as_tensor(flat_advantages, dtype=torch.float32, device=self.device)
        ret = torch.as_tensor(returns.reshape(-1), dtype=torch.float32, device=self.device)
        total = len(actions)
        for _ in range(self.update_epochs):
            order = self.rng.permutation(total)
            for start in range(0, total, self.minibatch_size):
                idx = torch.as_tensor(order[start : start + self.minibatch_size], dtype=torch.long, device=self.device)
                logits, predicted_values = self.model(obs[idx])
                logits = logits.masked_fill(~masks[idx], -1e9)
                distribution = torch.distributions.Categorical(logits=logits)
                ratio = torch.exp(distribution.log_prob(actions[idx]) - old_log_probs[idx])
                clipped = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range)
                policy_loss = -torch.min(ratio * adv[idx], clipped * adv[idx]).mean()
                value_loss = F.mse_loss(predicted_values, ret[idx])
                loss = policy_loss + 0.5 * value_loss - 0.01 * distribution.entropy().mean()
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                self.optimizer.step()


class SharedTabularAgent(SharedPolicyAgent):
    def __init__(self, name: str, num_channels: int, num_vehicles: int, seed: int) -> None:
        super().__init__(name, num_channels, num_vehicles, seed)
        self.num_actions = 1 + 2 * num_channels
        self.q_table: dict[tuple[int, ...], np.ndarray] = {}
        self.alpha = 0.15
        self.gamma = 0.95
        self.epsilon_start = 0.25
        self.epsilon_end = 0.02
        self.epsilon_decay_steps = 10_000

    def select_actions(self, observations: np.ndarray, infos: list[dict[str, Any]]) -> np.ndarray:
        actions = np.zeros(self.num_vehicles, dtype=int)
        epsilon = 0.0 if self.eval_mode else self._epsilon()
        for vehicle in range(self.num_vehicles):
            key = self._state_key(observations[vehicle])
            base_q = (
                self.q_table.get(key)
                if self.eval_mode
                else self.q_table.setdefault(key, self._new_q_values())
            )
            q_values = (self._new_q_values() if base_q is None else base_q).copy()
            mask = np.asarray(infos[vehicle]["action_mask"], dtype=bool)
            q_values[~mask] = -np.inf
            if epsilon > 0.0 and self.rng.random() < epsilon:
                valid = np.flatnonzero(mask)
                actions[vehicle] = int(self.rng.choice(valid)) if len(valid) else 0
            else:
                actions[vehicle] = int(np.argmax(q_values))
        return actions

    def observe_batch(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_observations: np.ndarray,
        done: bool,
        infos: list[dict[str, Any]],
        next_infos: list[dict[str, Any]],
    ) -> None:
        if self.eval_mode:
            return
        for vehicle in range(self.num_vehicles):
            key = self._state_key(observations[vehicle])
            next_key = self._state_key(next_observations[vehicle])
            q_values = self.q_table.setdefault(key, self._new_q_values())
            next_q = self.q_table.setdefault(next_key, self._new_q_values()).copy()
            next_q[~np.asarray(next_infos[vehicle]["action_mask"], dtype=bool)] = -np.inf
            target = float(rewards[vehicle])
            if not done and np.any(np.isfinite(next_q)):
                target += self.gamma * float(np.max(next_q))
            action = int(actions[vehicle])
            q_values[action] += self.alpha * (target - q_values[action])
        self.env_steps += 1

    def parameter_hash(self) -> str:
        digest = hashlib.sha256()
        for key in sorted(self.q_table):
            digest.update(np.asarray(key, dtype=np.int8).tobytes())
            digest.update(self.q_table[key].tobytes())
        return digest.hexdigest()[:16]

    def _new_q_values(self) -> np.ndarray:
        values = np.zeros(self.num_actions, dtype=float)
        values[0] = -0.05
        return values

    def _state_key(self, observation: np.ndarray) -> tuple[int, ...]:
        return tuple(np.clip(np.floor(observation * 4.0), 0, 4).astype(np.int8).tolist())

    def _epsilon(self) -> float:
        fraction = min(self.env_steps / max(self.epsilon_decay_steps, 1), 1.0)
        return self.epsilon_start + fraction * (self.epsilon_end - self.epsilon_start)


class IndependentDQNAgent(SharedPolicyAgent):
    '''A separate DDQN policy for every VT.'''

    SEED_STRIDE = 1009

    def __init__(self, name: str, num_channels: int, num_vehicles: int, seed: int) -> None:
        super().__init__(name, num_channels, num_vehicles, seed)
        self.vehicle_agents = []
        for vehicle in range(num_vehicles):
            agent = SharedDQNAgent(
                f'{name}_vt{vehicle}',
                num_channels,
                1,
                seed + self.SEED_STRIDE * vehicle,
                double_q=True,
                esn=False,
            )
            self.vehicle_agents.append(agent)

    def set_eval_mode(self, enabled: bool) -> None:
        super().set_eval_mode(enabled)
        for agent in self.vehicle_agents:
            agent.set_eval_mode(enabled)

    def reset_episode(self) -> None:
        for agent in self.vehicle_agents:
            agent.reset_episode()

    def select_actions(self, observations: np.ndarray, infos: list[dict[str, Any]]) -> np.ndarray:
        if observations.shape[0] != self.num_vehicles or len(infos) != self.num_vehicles:
            raise ValueError('independent DDQN expects one observation and info per vehicle')
        actions = [
            agent.select_actions(observations[vehicle : vehicle + 1], [infos[vehicle]])[0]
            for vehicle, agent in enumerate(self.vehicle_agents)
        ]
        return np.asarray(actions, dtype=int)

    def observe_batch(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_observations: np.ndarray,
        done: bool,
        infos: list[dict[str, Any]],
        next_infos: list[dict[str, Any]],
    ) -> None:
        if self.eval_mode:
            return
        for vehicle, agent in enumerate(self.vehicle_agents):
            agent.observe_batch(
                observations[vehicle : vehicle + 1],
                actions[vehicle : vehicle + 1],
                rewards[vehicle : vehicle + 1],
                next_observations[vehicle : vehicle + 1],
                done,
                [infos[vehicle]],
                [next_infos[vehicle]],
            )
        self.env_steps += 1

    def snapshot_runtime_state(self) -> RuntimeSnapshot:
        snapshot = super().snapshot_runtime_state()
        snapshot.payload = {
            'vehicle_snapshots': [agent.snapshot_runtime_state() for agent in self.vehicle_agents]
        }
        return snapshot

    def restore_runtime_state(self, snapshot: RuntimeSnapshot) -> None:
        super().restore_runtime_state(snapshot)
        children = snapshot.payload.get('vehicle_snapshots', [])
        if len(children) != len(self.vehicle_agents):
            raise ValueError('independent DDQN snapshot vehicle count mismatch')
        for agent, child_snapshot in zip(self.vehicle_agents, children):
            agent.restore_runtime_state(child_snapshot)

    def parameter_hash(self) -> str:
        digest = hashlib.sha256()
        for vehicle, agent in enumerate(self.vehicle_agents):
            digest.update(str(vehicle).encode('utf-8'))
            digest.update(agent.parameter_hash().encode('utf-8'))
        return digest.hexdigest()[:16]


class SharedFixedAgent(SharedPolicyAgent):
    def __init__(self, name: str, num_channels: int, num_vehicles: int, seed: int, mode: str) -> None:
        super().__init__(name, num_channels, num_vehicles, seed)
        agent_type = FixedOverlayAgent if mode == "overlay" else FixedUnderlayAgent
        self.policy = agent_type(name=name, num_channels=num_channels, seed=seed)

    @property
    def learned(self) -> bool:
        return False

    def select_actions(self, observations: np.ndarray, infos: list[dict[str, Any]]) -> np.ndarray:
        return np.asarray(
            [self.policy.select_action(observations[i], infos[i]) for i in range(self.num_vehicles)],
            dtype=int,
        )

    def observe_batch(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_observations: np.ndarray,
        done: bool,
        infos: list[dict[str, Any]],
        next_infos: list[dict[str, Any]],
    ) -> None:
        return


def create_shared_agent(
    name: str,
    num_channels: int,
    num_vehicles: int,
    seed: int,
    settings: dict[str, Any] | None = None,
) -> SharedPolicyAgent:
    settings = dict(settings or {})
    if name == 'independent_ddqn':
        return IndependentDQNAgent(name, num_channels, num_vehicles, seed)
    if name == "proposed_esn_ddqn":
        return SharedDQNAgent(name, num_channels, num_vehicles, seed, double_q=True, esn=True, **settings)
    if name == "ddqn_no_esn":
        return SharedDQNAgent(name, num_channels, num_vehicles, seed, double_q=True, esn=False, **settings)
    if name == "ddqn_mlp":
        return SharedDQNAgent(name, num_channels, num_vehicles, seed, double_q=True, esn=False, q_head="mlp", **settings)
    if name == "ddqn_linear":
        return SharedDQNAgent(name, num_channels, num_vehicles, seed, double_q=True, esn=False, q_head="linear", **settings)
    if name == "esn_ddqn_linear":
        return SharedDQNAgent(name, num_channels, num_vehicles, seed, double_q=True, esn=True, q_head="linear", **settings)
    if name == "esn_ddqn_mlp":
        return SharedDQNAgent(name, num_channels, num_vehicles, seed, double_q=True, esn=True, q_head="mlp", **settings)
    if name.startswith("esn_ddqn_linear_"):
        return SharedDQNAgent(name, num_channels, num_vehicles, seed, double_q=True, esn=True, q_head="linear", **settings)
    if name.startswith("ddqn_mlp_"):
        return SharedDQNAgent(name, num_channels, num_vehicles, seed, double_q=True, esn=False, q_head="mlp", **settings)
    if name == "lstm_ddqn":
        return SharedLSTMDDQNAgent(
            name, num_channels, num_vehicles, seed, **settings,
        )
    if name == "sac_discrete":
        return SharedDiscreteSACAgent(
            name, num_channels, num_vehicles, seed, **settings,
        )
    if name == "dqn":
        return SharedDQNAgent(name, num_channels, num_vehicles, seed, double_q=False, esn=False)
    if name == "ppo_discrete":
        return SharedPPOAgent(name, num_channels, num_vehicles, seed)
    if name == "tabular_q_learning":
        return SharedTabularAgent(name, num_channels, num_vehicles, seed)
    if name == "fixed_overlay":
        return SharedFixedAgent(name, num_channels, num_vehicles, seed, mode="overlay")
    if name == "fixed_underlay":
        return SharedFixedAgent(name, num_channels, num_vehicles, seed, mode="underlay")
    raise ValueError(f"unknown shared agent: {name}")


def _torch_module_hash(module: Any, fallback: str) -> str:
    if module is None:
        return hashlib.sha256(fallback.encode("utf-8")).hexdigest()[:16]
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()[:16]
