from __future__ import annotations

import time
from typing import Any

import pandas as pd

from cvn_rl.agents import create_agent
from cvn_rl.experiments.config import set_global_seed
from cvn_rl.experiments.sweeps import (
    overhead_cases,
    robustness_cases,
    scale_traffic_cases,
    seeds,
    state_reward_cases,
)
from cvn_rl.metrics import summarize_episode, summarize_table
from cvn_rl.simulation.env import AccessEnv
from cvn_rl.simulation.perturbations import apply_trace_perturbations
from cvn_rl.simulation.traces import generate_trace
from cvn_rl.simulation.traffic import traffic_label


def run_experiment_matrix(config: dict[str, Any], dry_run: bool = False, quick: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    experiment = config.get("experiment", "rl_baseline_comparison")
    slots = int(config.get("quick_slots" if quick or dry_run else "slots", config.get("slots", 5000)))
    if dry_run:
        slots = min(slots, 120)

    if experiment == "state_reward_ablation":
        rows = _run_state_reward_ablation(config, slots, dry_run)
    elif experiment == "prediction_sensing_robustness":
        rows = _run_robustness(config, slots, dry_run)
    elif experiment == "scale_and_traffic_sensitivity":
        rows = _run_scale_traffic(config, slots, dry_run)
    elif experiment == "overhead_staleness_sensitivity":
        rows = _run_overhead(config, slots, dry_run)
    else:
        rows = _run_baseline(config, slots, dry_run)

    metrics = pd.DataFrame(rows)
    summary = summarize_table(metrics)
    return metrics, summary


def _run_baseline(config: dict[str, Any], slots: int, dry_run: bool) -> list[dict[str, Any]]:
    rows = []
    scenarios = config.get("scenarios") or [{"name": _scenario_name(config), "num_channels": config.get("num_channels"), "num_vehicles": config.get("num_vehicles")}]
    for scenario in scenarios:
        case_config = {**config, **{k: v for k, v in scenario.items() if k != "name"}}
        for seed in seeds(config, dry_run):
            set_global_seed(seed)
            trace = generate_trace(case_config, seed=seed, slots=slots)
            context = _base_context(case_config, trace, config.get("experiment", "rl_baseline_comparison"), scenario.get("name"))
            for agent_name in config.get("agents", ["fixed_overlay", "fixed_underlay"]):
                rows.append(_run_agent(case_config, trace, agent_name, seed, context))
    return rows


def _run_state_reward_ablation(config: dict[str, Any], slots: int, dry_run: bool) -> list[dict[str, Any]]:
    rows = []
    for seed in seeds(config, dry_run):
        trace = generate_trace(config, seed=seed, slots=slots)
        base_context = _base_context(config, trace, config["experiment"], _scenario_name(config))
        for case in state_reward_cases(config, dry_run):
            state = case["state"]
            reward = case["reward"]
            for agent_name in config.get("agents", ["proposed_esn_ddqn"]):
                rows.append(
                    _run_agent(
                        config,
                        trace,
                        agent_name,
                        seed,
                        {
                            **base_context,
                            "state_variant": state["name"],
                            "reward_variant": reward["name"],
                        },
                        state_features=state["features"],
                        reward_layers=reward["layers"],
                    )
                )
    return rows


def _run_robustness(config: dict[str, Any], slots: int, dry_run: bool) -> list[dict[str, Any]]:
    rows = []
    for seed in seeds(config, dry_run):
        base_trace = generate_trace(config, seed=seed, slots=slots)
        for case in robustness_cases(config, dry_run):
            trace = apply_trace_perturbations(base_trace, case, seed=seed + 17)
            base_context = _base_context(config, trace, config["experiment"], _scenario_name(config))
            for agent_name in config.get("agents", ["proposed_esn_ddqn"]):
                rows.append(_run_agent(config, trace, agent_name, seed, {**base_context, **case}))
    return rows


def _run_scale_traffic(config: dict[str, Any], slots: int, dry_run: bool) -> list[dict[str, Any]]:
    rows = []
    for case in scale_traffic_cases(config, dry_run):
        scenario = case["scenario"]
        traffic = case["traffic"]
        case_config = {**config, **scenario, "traffic": traffic}
        for seed in seeds(config, dry_run):
            trace = generate_trace(case_config, seed=seed, slots=slots)
            base_context = _base_context(case_config, trace, config["experiment"], scenario["name"])
            base_context.update({"traffic_family": traffic["family"], "traffic_label": traffic_label(traffic)})
            for agent_name in config.get("agents", ["proposed_esn_ddqn"]):
                rows.append(_run_agent(case_config, trace, agent_name, seed, base_context))
    return rows


def _run_overhead(config: dict[str, Any], slots: int, dry_run: bool) -> list[dict[str, Any]]:
    rows = []
    for case in overhead_cases(config, dry_run):
        case_config = {
            **config,
            "broadcast_update_slots": case["broadcast_update_slots"],
            "dqi": {
                **config.get("dqi", {}),
                "validity": {
                    **config.get("dqi", {}).get("validity", {}),
                    "t_max_slots": case["dqi_validity_slots"],
                },
            },
        }
        case_config.setdefault("underlay_power", dict(config.get("underlay_power", {})))
        case_config["underlay_power"] = {
            **case_config["underlay_power"],
            "broadcast_update_slots": case["broadcast_update_slots"],
        }
        for seed in seeds(config, dry_run):
            trace = generate_trace(case_config, seed=seed, slots=slots)
            base_context = _base_context(case_config, trace, config["experiment"], _scenario_name(case_config))
            for agent_name in config.get("agents", ["proposed_esn_ddqn"]):
                rows.append(
                    _run_agent(
                        case_config,
                        trace,
                        agent_name,
                        seed,
                        {
                            **base_context,
                            "broadcast_update_slots": case["broadcast_update_slots"],
                            "dqi_validity_slots": case["dqi_validity_slots"],
                        },
                    )
                )
    return rows


def _run_agent(
    config: dict[str, Any],
    trace: Any,
    agent_name: str,
    seed: int,
    context: dict[str, Any],
    state_features: list[str] | None = None,
    reward_layers: list[str] | None = None,
) -> dict[str, Any]:
    agent = create_agent(agent_name, num_channels=trace.num_channels, seed=seed)

    train_env = AccessEnv(trace, config, state_features=state_features, reward_layers=reward_layers)
    train_start = time.perf_counter()
    agent.set_eval_mode(False)
    agent.reset()
    _rollout_agent(agent, train_env, collect_records=False)
    training_time = time.perf_counter() - train_start

    eval_env = AccessEnv(trace, config, state_features=state_features, reward_layers=reward_layers)
    agent.set_eval_mode(True)
    agent.reset()
    records, inference_seconds = _rollout_agent(agent, eval_env, collect_records=True)
    latency_ms = inference_seconds / max(len(records), 1) * 1000.0
    return summarize_episode(
        records,
        {"agent": agent_name, "seed": seed, **context},
        config,
        training_time_seconds=training_time,
        inference_latency_ms=latency_ms,
    )


def _rollout_agent(agent: Any, env: AccessEnv, collect_records: bool) -> tuple[list[dict[str, Any]], float]:
    observation, info = env.reset()
    records: list[dict[str, Any]] = []
    inference_seconds = 0.0
    done = False
    while not done:
        t0 = time.perf_counter()
        action = agent.select_action(observation, info)
        inference_seconds += time.perf_counter() - t0
        next_observation, reward, done, step_info = env.step(action)
        next_info = env._info()
        transition_info = {**info, "next_action_mask": next_info.get("action_mask")}
        agent.observe(observation, action, reward, next_observation, done, transition_info)
        if collect_records:
            records.append(step_info)
        observation, info = next_observation, next_info
    return records, inference_seconds
def _base_context(config: dict[str, Any], trace: Any, experiment: str, scenario: str | None) -> dict[str, Any]:
    traffic = config.get("traffic", {})
    return {
        "experiment": experiment,
        "scenario": scenario or _scenario_name(config),
        "resource_regime": _resource_regime(config),
        "num_channels": int(config.get("num_channels", trace.num_channels)),
        "num_vehicles": int(config.get("num_vehicles", 1)),
        "trace_hash": trace.trace_hash(),
        "traffic_family": traffic.get("family", "fixed_markov"),
        "traffic_label": traffic_label(traffic),
    }


def _scenario_name(config: dict[str, Any]) -> str:
    return f"M{int(config.get('num_channels', 0))}_N{int(config.get('num_vehicles', 0))}"


def _resource_regime(config: dict[str, Any]) -> str:
    m = int(config.get("num_channels", 0))
    n = int(config.get("num_vehicles", 0))
    if m > n:
        return "M_gt_N"
    if m < n:
        return "M_lt_N"
    return "M_eq_N"
