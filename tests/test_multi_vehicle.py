from __future__ import annotations

import numpy as np
import pytest

from cvn_rl.agents.shared_multi import create_shared_agent
from cvn_rl.experiments.config import load_config
from cvn_rl.experiments.multi_vehicle_runner import run_multi_vehicle_experiment
from cvn_rl.multi_metrics import summarize_multi_vehicle_records
from cvn_rl.simulation.actions import encode_action
from cvn_rl.simulation.multi_env import MultiVehicleAccessEnv
from cvn_rl.simulation.multi_traces import generate_multi_vehicle_trace


def _config(num_channels: int, num_vehicles: int, slots: int = 4) -> dict:
    base = load_config("configs/base_multi_vehicle.json")
    return {
        **base,
        "num_channels": num_channels,
        "num_vehicles": num_vehicles,
        "slots": slots,
        "packet_traffic": {
            "arrival_probability": 1.0,
            "queue_capacity_packets": 10,
            "packet_size_bits": 1_000,
        },
    }


def test_multi_trace_is_reproducible_and_uses_vehicle_dimension() -> None:
    config = _config(3, 5, slots=6)
    first = generate_multi_vehicle_trace(config, seed=4, slots=6)
    second = generate_multi_vehicle_trace(config, seed=4, slots=6)
    assert first.occupancy.shape == (6, 3)
    assert first.quality_mbps.shape == (6, 5, 3)
    assert first.packet_arrivals.shape == (6, 5)
    assert first.trace_hash() == second.trace_hash()


def test_arrival_first_queue_overflow_drives_plr() -> None:
    config = _config(1, 1, slots=3)
    config["packet_traffic"]["queue_capacity_packets"] = 1
    trace = generate_multi_vehicle_trace(config, seed=1, slots=3)
    trace.packet_arrivals[:] = True
    env = MultiVehicleAccessEnv(trace, config)
    _, _ = env.reset()
    records = []
    done = False
    while not done:
        _, _, done, info = env.step(np.array([0]))
        records.append(info)
    summary = summarize_multi_vehicle_records(records, {}, config)
    assert summary["generated_packets"] == 3
    assert summary["dropped_packets"] == 2
    assert np.isclose(summary["plr"], 2.0 / 3.0)


def test_same_channel_contention_selects_one_transmitter() -> None:
    config = _config(1, 2, slots=2)
    trace = generate_multi_vehicle_trace(config, seed=2, slots=2)
    trace.occupancy[:] = False
    trace.packet_arrivals[:] = True
    env = MultiVehicleAccessEnv(trace, config)
    env.reset()
    action = encode_action(0, "overlay", 1)
    _, _, _, info = env.step(np.array([action, action]))
    assert int(np.sum(info["contention_lost"])) == 1
    assert int(np.sum(info["transmitted"])) == 1
    assert not info["pu_violation_per_channel"].any()
    assert info["system_vt_throughput"] > 0.0


def test_different_channels_complete_packets() -> None:
    config = _config(2, 2, slots=2)
    trace = generate_multi_vehicle_trace(config, seed=3, slots=2)
    trace.occupancy[:] = False
    trace.packet_arrivals[:] = True
    trace.quality_mbps[:] = 100.0
    env = MultiVehicleAccessEnv(trace, config)
    env.reset()
    actions = np.array([
        encode_action(0, "overlay", 2),
        encode_action(1, "overlay", 2),
    ])
    _, _, _, info = env.step(actions)
    assert not info["vt_collision"].any()
    assert info["packet_completed"].tolist() == [1, 1]
    assert info["system_vt_throughput"] > 0.0


def test_packet_success_and_pu_violation_are_independent() -> None:
    config = _config(1, 1, slots=2)
    trace = generate_multi_vehicle_trace(config, seed=5, slots=2)
    trace.occupancy[:] = True
    trace.packet_arrivals[:] = True
    trace.quality_mbps[:] = 100.0
    trace.su_pu_gain_true[:] = 10.0
    trace.pu_threshold_mw[:] = 0.1
    trace.transmit_power_cap_mw[:] = 0.1
    trace.power_cap[:] = 0.1
    env = MultiVehicleAccessEnv(trace, config)
    env.reset()
    action = encode_action(0, "underlay", 1)
    _, _, _, info = env.step(np.array([action]))
    assert info["packet_completed"][0] == 1
    assert bool(info["pu_violation_per_channel"][0]) is True


def test_shared_esn_keeps_independent_vehicle_states(monkeypatch) -> None:
    monkeypatch.setenv("CVN_RL_DEVICE", "cpu")
    agent = create_shared_agent("proposed_esn_ddqn", num_channels=2, num_vehicles=2, seed=7)
    observations = np.asarray([[0.0] * 6, [1.0] * 6], dtype=float)
    mask = np.ones(5, dtype=bool)
    infos = [{"action_mask": mask, "vehicle_id": 0}, {"action_mask": mask, "vehicle_id": 1}]
    agent.select_actions(observations, infos)
    assert agent.reservoir_states.shape == (2, 128)
    assert not np.allclose(agent.reservoir_states[0], agent.reservoir_states[1])


def test_checkpoint_evaluation_keeps_parameters_and_common_trace(monkeypatch) -> None:
    monkeypatch.setenv("CVN_RL_DEVICE", "cpu")
    config = _config(2, 2, slots=8)
    config.update(
        {
            "experiment": "multi_vehicle_baseline_comparison",
            "seeds": [1],
            "agents": ["proposed_esn_ddqn", "fixed_overlay", "fixed_underlay"],
            "scenarios": [{"name": "tiny", "num_channels": 2, "num_vehicles": 2}],
            "checkpoint_evaluation": {"checkpoint_interval": 8, "eval_episodes": 1, "eval_slots": 4},
        }
    )
    metrics, _ = run_multi_vehicle_experiment(config)
    assert metrics["evaluation_trace_hash"].nunique() == 1
    assert (metrics["parameter_hash_before_eval"] == metrics["parameter_hash_after_eval"]).all()
    assert (metrics['observation_dimension'] > 0).all()
    assert metrics['state_feature_signature'].str.len().gt(0).all()
    ratios = metrics.set_index("agent")["underlay_selection_ratio"]
    assert ratios["fixed_overlay"] == 0.0
    assert ratios["fixed_underlay"] == 1.0


def test_independent_ddqn_has_private_networks_replay_and_seeds(monkeypatch) -> None:
    monkeypatch.setenv('CVN_RL_DEVICE', 'cpu')
    torch = pytest.importorskip('torch')
    agent = create_shared_agent(
        'independent_ddqn',
        num_channels=2,
        num_vehicles=2,
        seed=11,
    )
    children = agent.vehicle_agents
    assert [child.seed for child in children] == [11, 1020]
    assert children[0].replay is not children[1].replay
    observations = np.zeros((2, 6), dtype=float)
    mask = np.ones(5, dtype=bool)
    infos = [
        {'action_mask': mask, 'vehicle_id': 0},
        {'action_mask': mask, 'vehicle_id': 1},
    ]
    agent.set_eval_mode(True)
    agent.select_actions(observations, infos)
    with torch.no_grad():
        for vehicle, child in enumerate(children):
            for parameter in child.q_net.parameters():
                parameter.zero_()
            child.q_net.net[-1].bias[vehicle + 1] = 10.0
    actions = agent.select_actions(observations, infos)
    assert actions.tolist() == [1, 2]
    assert children[0].parameter_hash() != children[1].parameter_hash()


def test_safety_events_are_reported_separately() -> None:
    config = _config(1, 1, slots=2)

    overlay_trace = generate_multi_vehicle_trace(config, seed=31, slots=2)
    overlay_trace.occupancy[:] = True
    overlay_trace.packet_arrivals[:] = True
    overlay_env = MultiVehicleAccessEnv(overlay_trace, config)
    overlay_env.reset()
    _, _, _, overlay_record = overlay_env.step(
        np.array([encode_action(0, 'overlay', 1)])
    )
    overlay = summarize_multi_vehicle_records([overlay_record], {}, config)
    assert overlay['overlay_collision_rate'] == 1.0
    assert overlay['conditional_underlay_violation_rate'] == 0.0
    assert overlay['contention_loss_rate'] == 0.0

    underlay_trace = generate_multi_vehicle_trace(config, seed=32, slots=2)
    underlay_trace.occupancy[:] = True
    underlay_trace.packet_arrivals[:] = True
    underlay_trace.su_pu_gain_true[:] = 10.0
    underlay_trace.pu_threshold_mw[:] = 0.01
    underlay_trace.transmit_power_cap_mw[:] = 1.0
    underlay_env = MultiVehicleAccessEnv(underlay_trace, config)
    underlay_env.reset()
    _, _, _, underlay_record = underlay_env.step(
        np.array([encode_action(0, 'underlay', 1)])
    )
    underlay = summarize_multi_vehicle_records([underlay_record], {}, config)
    assert underlay['overlay_collision_rate'] == 0.0
    assert underlay['conditional_underlay_violation_rate'] == 1.0
    assert underlay['contention_loss_rate'] == 0.0

    contention_config = _config(1, 2, slots=2)
    contention_trace = generate_multi_vehicle_trace(contention_config, seed=33, slots=2)
    contention_trace.occupancy[:] = False
    contention_trace.packet_arrivals[:] = True
    contention_env = MultiVehicleAccessEnv(contention_trace, contention_config)
    contention_env.reset()
    action = encode_action(0, 'overlay', 1)
    _, _, _, contention_record = contention_env.step(np.array([action, action]))
    contention = summarize_multi_vehicle_records(
        [contention_record],
        {},
        contention_config,
    )
    assert contention['contention_loss_rate'] == 0.5
    assert contention['overlay_collision_rate'] == 0.0
    assert contention['conditional_underlay_violation_rate'] == 0.0


def test_evaluation_variants_share_seed_but_have_distinct_trace_hashes(monkeypatch) -> None:
    monkeypatch.setenv('CVN_RL_DEVICE', 'cpu')
    config = _config(2, 1, slots=8)
    config.update(
        {
            'experiment': 'multi_vehicle_evaluation_robustness',
            'seeds': [1],
            'agents': ['ddqn_no_esn'],
            'scenarios': [{'name': 'tiny', 'num_channels': 2, 'num_vehicles': 1}],
            'checkpoint_evaluation': {
                'checkpoint_interval': 8,
                'eval_episodes': 1,
                'eval_slots': 4,
            },
            'evaluation_variants': [
                {'name': 'clean'},
                {
                    'name': 'noisy',
                    'descriptors': {'controlled_prediction_pfa': 1.0},
                },
            ],
        }
    )
    metrics, _ = run_multi_vehicle_experiment(config)
    assert set(metrics['evaluation_variant']) == {'clean', 'noisy'}
    assert metrics['evaluation_trace_hash'].nunique() == 2
    assert metrics['parameter_freezing_hash'].nunique() == 1
    assert metrics['evaluation_parameters_frozen'].all()
