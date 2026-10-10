from pathlib import Path

import numpy as np

from cvn_rl.agents.shared_multi import create_shared_agent
from cvn_rl.experiments.config import load_config
from cvn_rl.experiments.multi_vehicle_runner import (
    _build_cases,
    _parameter_count_breakdown,
)
from scripts.analyze_esn_structure_diagnostic import (
    FROZEN,
    MODEL_SHA,
    PLACEBO,
    classify,
    paired_effects,
    summarize_effects,
    validate_rows,
)


AGENTS = ["ddqn_mlp", "ddqn_linear", "esn_ddqn_linear", "esn_ddqn_mlp"]


def _initialize(agent):
    observations = np.zeros((1, 11), dtype=float)
    infos = [{"action_mask": np.ones(3, dtype=bool), "vehicle_id": 0}]
    agent.set_eval_mode(True)
    agent.select_actions(observations, infos)
    return agent


def test_structure_config_is_four_by_two_by_three():
    config = load_config(Path("configs/e2_p3x_esn_structure_diagnostic.json"))
    assert config["agents"] == AGENTS
    assert config["seeds"] == [1, 2, 3]
    assert {item["name"] for item in config["module_conditions"]} == {
        "theta_central_frozen", "theta_central_frozen_placebo"
    }
    cases = _build_cases(config, config["scenarios"])
    assert len(cases) == 2
    assert len(AGENTS) * len(cases) * len(config["seeds"]) == 24


def test_four_structures_have_expected_encoders_and_heads(monkeypatch):
    monkeypatch.setenv("CVN_RL_DEVICE", "cpu")
    agents = {name: _initialize(create_shared_agent(name, 1, 1, 7)) for name in AGENTS}
    assert agents["ddqn_mlp"].esn is False and agents["ddqn_mlp"].q_head == "mlp"
    assert agents["ddqn_linear"].esn is False and agents["ddqn_linear"].q_head == "linear"
    assert agents["esn_ddqn_linear"].esn is True and agents["esn_ddqn_linear"].q_head == "linear"
    assert agents["esn_ddqn_mlp"].esn is True and agents["esn_ddqn_mlp"].q_head == "mlp"


def test_parameter_breakdown_separates_fixed_reservoir(monkeypatch):
    monkeypatch.setenv("CVN_RL_DEVICE", "cpu")
    counts = {
        name: _parameter_count_breakdown(_initialize(create_shared_agent(name, 1, 1, 9)))
        for name in AGENTS
    }
    assert counts["ddqn_mlp"]["fixed_reservoir"] == 0
    assert counts["ddqn_linear"]["fixed_reservoir"] == 0
    assert counts["esn_ddqn_linear"]["fixed_reservoir"] > 0
    assert counts["esn_ddqn_mlp"]["fixed_reservoir"] == counts["esn_ddqn_linear"]["fixed_reservoir"]
    assert counts["ddqn_linear"]["trainable"] < counts["ddqn_mlp"]["trainable"]
    assert counts["esn_ddqn_linear"]["trainable"] < counts["esn_ddqn_mlp"]["trainable"]
    for value in counts.values():
        assert value["total"] == value["trainable"] + value["fixed_reservoir"]


def test_agent_settings_override_lr_and_leak(monkeypatch):
    monkeypatch.setenv("CVN_RL_DEVICE", "cpu")
    agent = create_shared_agent(
        "esn_ddqn_linear", 1, 1, 3,
        settings={"learning_rate": 1e-3, "reservoir_leak": 1.0},
    )
    assert agent.learning_rate == 1e-3
    assert agent.reservoir_leak == 1.0


def _synthetic_final():
    rows = []
    base = {"ddqn_mlp": 10.0, "ddqn_linear": 6.0, "esn_ddqn_linear": 7.0, "esn_ddqn_mlp": 9.0}
    for agent in AGENTS:
        for condition in (FROZEN, PLACEBO):
            for seed in (1, 2, 3):
                rows.append({"agent": agent, "scenario": "p3x", "module_variant": condition, "seed": seed,
                             "is_final_checkpoint": True, "evaluation_parameters_frozen": True,
                             "prediction_model_hash": MODEL_SHA, "evaluation_physical_trace_hash": f"trace-{seed}",
                             "raw_action_mask_hash": f"mask-{seed}",
                             "served_goodput_mbps": base[agent] - (1.0 if condition == PLACEBO else 0.0),
                             "plr": .2, "pu_access_violation_rate": .01, "mean_reward": .5})
    return __import__("pandas").DataFrame(rows)


def test_synthetic_structure_effects_and_classification():
    final = _synthetic_final()
    validate_rows(final)
    summary = summarize_effects(paired_effects(final))
    gate = classify(summary)
    assert gate["head_capacity_signal"] is True
    assert gate["reservoir_value_with_linear_head"] is True
    assert gate["reservoir_value_with_mlp_head"] is False


def test_structure_validator_rejects_trace_and_hash_faults():
    bad_trace = _synthetic_final()
    bad_trace.loc[0, "evaluation_physical_trace_hash"] = "wrong"
    with __import__("pytest").raises(ValueError, match="physical trace"):
        validate_rows(bad_trace)
    bad_hash = _synthetic_final()
    bad_hash.loc[0, "prediction_model_hash"] = "wrong"
    with __import__("pytest").raises(ValueError, match="SHA"):
        validate_rows(bad_hash)


def test_followup_configs_only_add_preregistered_controls():
    leak = load_config(Path("configs/e2_p3x_esn_leak_diagnostic.json"))
    assert leak["agents"] == ["esn_ddqn_linear_leak_0p1", "esn_ddqn_linear_leak_1p0"]
    assert {leak["agent_settings"][name]["reservoir_leak"] for name in leak["agents"]} == {0.1, 1.0}
    assert len(_build_cases(leak, leak["scenarios"])) == 2
    lr = load_config(Path("configs/e2_p3x_esn_lr_control.json"))
    assert lr["agents"] == ["esn_ddqn_linear_lr_1e3", "ddqn_mlp_lr_3e3"]
    assert len(lr["module_conditions"]) == 1
    assert len(_build_cases(lr, lr["scenarios"])) == 1
