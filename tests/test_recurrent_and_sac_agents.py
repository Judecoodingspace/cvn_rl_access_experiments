from pathlib import Path

import numpy as np
import torch

from cvn_rl.agents.shared_multi import (
    SequenceReplayBuffer,
    SequenceTransition,
    create_shared_agent,
)
from cvn_rl.experiments.config import load_config
from cvn_rl.experiments.multi_vehicle_runner import _parameter_count_breakdown


def transition(value: float, episode: int, done: bool = False):
    mask = np.array([True, False, True])
    return SequenceTransition(
        observation=np.full(4, value, dtype=np.float32), action=0,
        reward=value, next_observation=np.full(4, value + 1, dtype=np.float32),
        done=done, action_mask=mask, next_action_mask=mask,
        episode_id=episode,
    )


def test_sequence_replay_never_crosses_episode_boundary():
    replay = SequenceReplayBuffer(100, 4, 1, np.random.default_rng(3))
    for index in range(5):
        replay.add(0, transition(float(index), 1, done=index == 4))
    for index in range(5):
        replay.add(0, transition(float(index), 2, done=index == 4))
    assert replay.num_windows == 4
    for window in replay.sample(20):
        assert len({item.episode_id for item in window}) == 1
        assert not any(item.done for item in window[:-1])


def exercise(agent, steps: int = 24):
    observation = np.zeros((1, 4), dtype=np.float32)
    infos = [{"action_mask": np.array([True, False, True])}]
    agent.reset_episode(); agent.set_eval_mode(False)
    for step in range(steps):
        action = agent.select_actions(observation, infos)
        assert action[0] in {0, 2}
        next_observation = np.full_like(observation, (step % 5) / 5)
        agent.observe_batch(
            observation, action, np.array([.1], dtype=np.float32),
            next_observation, step == steps - 1, infos, infos,
        )
        observation = next_observation
    return agent


def test_lstm_ddqn_updates_and_restores_hidden_state(monkeypatch):
    monkeypatch.setenv("CVN_RL_DEVICE", "cpu")
    agent = create_shared_agent(
        "lstm_ddqn", 1, 1, 5,
        settings={"hidden_size": 16, "sequence_length": 4, "burn_in": 2,
                  "warmup": 8, "batch_size": 4},
    )
    exercise(agent)
    assert agent.gradient_updates > 0
    assert np.isfinite(agent.last_q_loss)
    snapshot = agent.snapshot_runtime_state()
    hidden_before = tuple(value.clone() for value in agent.hidden)
    agent.select_actions(
        np.ones((1, 4), dtype=np.float32),
        [{"action_mask": np.ones(3, dtype=bool)}],
    )
    agent.restore_runtime_state(snapshot)
    assert all(torch.equal(left, right) for left, right in zip(agent.hidden, hidden_before))


def test_discrete_sac_masks_actions_and_updates(monkeypatch):
    monkeypatch.setenv("CVN_RL_DEVICE", "cpu")
    agent = create_shared_agent(
        "sac_discrete", 1, 1, 7,
        settings={"hidden_dim": 16, "warmup": 8, "batch_size": 8},
    )
    exercise(agent)
    assert agent.gradient_updates > 0
    assert np.isfinite(agent.last_actor_loss)
    assert np.isfinite(agent.last_critic_loss)
    assert 1e-4 <= float(agent.alpha.detach()) <= 100.0


def test_new_controller_configs_and_parameter_counts(monkeypatch):
    monkeypatch.setenv("CVN_RL_DEVICE", "cpu")
    for config_name, expected in [
        ("e2_p3x_lstm_ddqn_frozen_predictor.json", "lstm_ddqn"),
        ("e2_p3x_sac_discrete_frozen_predictor.json", "sac_discrete"),
    ]:
        config = load_config(Path("configs") / config_name)
        assert config["agents"] == [expected]
        assert len(config["module_conditions"]) == 5
    for name in ("lstm_ddqn", "sac_discrete"):
        agent = create_shared_agent(name, 1, 1, 11)
        agent.set_eval_mode(True)
        agent.select_actions(
            np.zeros((1, 4), dtype=np.float32),
            [{"action_mask": np.ones(3, dtype=bool)}],
        )
        counts = _parameter_count_breakdown(agent)
        assert counts["trainable"] > 0
        assert counts["fixed_reservoir"] == 0
        assert counts["total"] == counts["trainable"]
