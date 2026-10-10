from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cvn_rl.agents.shared_multi import SharedPolicyAgent, create_shared_agent
from cvn_rl.agents.torch_agents import _select_device
from cvn_rl.experiments.config import set_global_seed
from cvn_rl.experiments.sweeps import seeds
from cvn_rl.multi_metrics import summarize_checkpoint_table, summarize_multi_vehicle_records
from cvn_rl.simulation.actions import DEFAULT_ACTION_MASK_THRESHOLDS, encode_action
from cvn_rl.simulation.descriptor_bundle import load_descriptor_bundle, save_descriptor_bundle
from cvn_rl.simulation.multi_env import MultiVehicleAccessEnv
from cvn_rl.simulation.dqi_labels import (
    compute_unnormalized_dqi,
    normalize_dqi,
)
from cvn_rl.simulation.dqi_policy_value import (
    make_channel_placebo,
    make_time_block_placebo,
)
from cvn_rl.simulation.multi_traces import (
    MultiVehicleAccessTrace,
    _latency_constrained_goodput,
    generate_multi_vehicle_trace,
)
from cvn_rl.simulation.state_builder import feature_slices
from cvn_rl.simulation.traffic import traffic_label

EvaluationSet = tuple[
    str,
    dict[str, Any],
    list[MultiVehicleAccessTrace],
    dict[str, Any],
]


def run_multi_vehicle_experiment(
    config: dict[str, Any],
    dry_run: bool = False,
    quick: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    config = {**config, '_diagnostic_run': bool(dry_run or quick)}
    training_cfg = config.get("checkpoint_evaluation", {})
    train_slots = int(config.get("quick_slots" if quick or dry_run else "slots", config.get("slots", 30_000)))
    checkpoint_interval = int(training_cfg.get("checkpoint_interval", 1_000))
    eval_episodes = int(training_cfg.get("eval_episodes", 10))
    eval_slots = int(training_cfg.get("eval_slots", 1_000))
    final_evaluation_cfg = config.get("final_checkpoint_evaluation")
    final_eval_episodes = int(
        (final_evaluation_cfg or {}).get("eval_episodes", eval_episodes)
    )
    final_eval_slots = int(
        (final_evaluation_cfg or {}).get("eval_slots", eval_slots)
    )
    scenarios = list(config.get("scenarios") or [_default_scenario(config)])
    cases = _build_cases(config, scenarios)
    agent_names = list(config.get("agents", ["proposed_esn_ddqn", "ddqn_no_esn"]))
    run_seeds = seeds(config, dry_run=False)

    if dry_run:
        train_slots = min(train_slots, 80)
        checkpoint_interval = train_slots
        eval_episodes = 1
        eval_slots = min(eval_slots, 40)
        final_eval_episodes = 1
        final_eval_slots = min(final_eval_slots, 40)
        if config.get('experiment') in {
            'multi_vehicle_module_ablation',
            'e2_r1_oracle_gate',
        } and cases:
            first_scenario = cases[0][0].get('name')
            cases = [
                case
                for case in cases
                if case[0].get('name') == first_scenario
            ][:4]
        else:
            cases = cases[:2]
        run_seeds = run_seeds[:1]
        preferred = ["proposed_esn_ddqn", "fixed_overlay", "fixed_underlay"]
        agent_names = [name for name in preferred if name in agent_names] or agent_names[:1]
    elif quick:
        checkpoint_interval = min(checkpoint_interval, max(train_slots, 1))
        eval_episodes = min(eval_episodes, 2)
        eval_slots = min(eval_slots, 200)
        final_eval_episodes = min(final_eval_episodes, 2)
        final_eval_slots = min(final_eval_slots, 200)

    rows: list[dict[str, Any]] = []
    total_curves = len(cases) * len(run_seeds) * len(agent_names)
    curve_index = 0
    for scenario, case_config in cases:
        for seed in run_seeds:
            set_global_seed(int(seed))
            scenario_name = str(scenario.get('name') or _default_scenario(case_config)['name'])
            train_trace = _resolve_trace(
                case_config,
                scenario_name,
                seed=int(seed),
                slots=train_slots,
                split='rl_train',
            )
            evaluation_traces = [
                _resolve_trace(
                    case_config,
                    scenario_name,
                    seed=_evaluation_seed(int(seed), episode),
                    slots=eval_slots,
                    split=f'rl_eval_{episode}',
                )
                for episode in range(eval_episodes)
            ]
            evaluation_sets = _build_evaluation_sets(
                case_config,
                scenario_name,
                int(seed),
                eval_episodes,
                eval_slots,
                evaluation_traces,
            )
            if final_evaluation_cfg:
                final_evaluation_traces = [
                    _resolve_trace(
                        case_config,
                        scenario_name,
                        seed=_final_evaluation_seed(int(seed), episode),
                        slots=final_eval_slots,
                        split=f'rl_final_eval_{episode}',
                    )
                    for episode in range(final_eval_episodes)
                ]
                final_evaluation_sets = _build_evaluation_sets(
                    case_config,
                    scenario_name,
                    int(seed),
                    final_eval_episodes,
                    final_eval_slots,
                    final_evaluation_traces,
                    final_checkpoint=True,
                )
            else:
                final_evaluation_sets = evaluation_sets
            context = _base_context(case_config, scenario.get("name"), train_trace, evaluation_traces)
            for agent_name in agent_names:
                curve_index += 1
                print(
                    "[multi_vehicle] "
                    f"curve {curve_index}/{total_curves} "
                    f"scenario={scenario_name} "
                    f"condition={context.get('module_variant', '')} "
                    f"seed={int(seed)} "
                    f"agent={agent_name}",
                    flush=True,
                )
                reused_rows = _reuse_rows_for_case(
                    case_config,
                    context,
                    agent_name,
                    int(seed),
                    evaluation_sets,
                    final_evaluation_sets,
                )
                if reused_rows:
                    print(
                        "[multi_vehicle] "
                        f"reused {len(reused_rows)} rows "
                        f"condition={context.get('module_variant', '')} "
                        f"seed={int(seed)} "
                        f"agent={agent_name}",
                        flush=True,
                    )
                    rows.extend(reused_rows)
                    continue
                if (
                    not bool(case_config.get('_diagnostic_run', False))
                    and _case_requires_reuse(case_config, context)
                ):
                    raise ValueError(
                        'required Phase A reuse row was not found or trace hashes '
                        f'did not match: {context["scenario"]}, '
                        f'{context["module_variant"]}, seed={seed}, '
                        f'p_a={context["arrival_probability"]}'
                    )
                rows.extend(
                    _train_and_evaluate(
                        case_config,
                        train_trace,
                        evaluation_sets,
                        final_evaluation_sets,
                        agent_name,
                        int(seed),
                        checkpoint_interval,
                        context,
                    )
                )

    metrics = pd.DataFrame(rows)
    summary = summarize_checkpoint_table(metrics) if not metrics.empty else pd.DataFrame()
    return metrics, summary


def _case_requires_reuse(
    config: dict[str, Any],
    context: dict[str, Any],
) -> bool:
    for required in config.get('require_reuse_cases', []):
        if any(
            (
                not np.isclose(
                    float(context[key]),
                    float(value),
                )
                if key == 'arrival_probability'
                else str(context.get(key)) != str(value)
            )
            for key, value in required.items()
        ):
            continue
        return True
    return False


def _reuse_rows_for_case(
    config: dict[str, Any],
    context: dict[str, Any],
    agent_name: str,
    seed: int,
    evaluation_sets: list[EvaluationSet],
    final_evaluation_sets: list[EvaluationSet],
) -> list[dict[str, Any]]:
    reuse = config.get('_reuse_metrics')
    if not isinstance(reuse, pd.DataFrame) or reuse.empty:
        return []
    aliases = config.get('reuse_module_aliases', {})
    source_variant = str(
        aliases.get(context['module_variant'], context['module_variant'])
    )
    required_columns = {
        'scenario', 'module_variant', 'arrival_probability', 'agent', 'seed',
        'train_trace_hash', 'evaluation_trace_hash',
    }
    if not required_columns.issubset(reuse.columns):
        return []
    selected = reuse[
        (reuse['scenario'].astype(str) == str(context['scenario']))
        & (reuse['module_variant'].astype(str) == source_variant)
        & np.isclose(
            pd.to_numeric(reuse['arrival_probability'], errors='coerce'),
            float(context['arrival_probability']),
        )
        & (reuse['agent'].astype(str) == str(agent_name))
        & (pd.to_numeric(reuse['seed'], errors='coerce') == int(seed))
    ].copy()
    if 'evaluation_variant' in selected:
        selected = selected[selected['evaluation_variant'].astype(str) == 'nominal']
    for field in (
        'mask_policy_version',
        'ablation_info_source',
        'ablation_mask_source',
    ):
        if field in selected and field in context:
            selected = selected[
                selected[field].astype(str) == str(context[field])
            ]
    if selected.empty:
        return []
    if (
        selected['train_trace_hash'].astype(str).nunique() != 1
        or str(selected['train_trace_hash'].iloc[0])
        != str(context['train_trace_hash'])
    ):
        return []
    regular_evaluation = _reuse_evaluation_context(evaluation_sets)
    final_evaluation = _reuse_evaluation_context(final_evaluation_sets)
    for _, row in selected.iterrows():
        expected = (
            final_evaluation
            if bool(row.get('is_final_checkpoint', False))
            else regular_evaluation
        )
        if str(row['evaluation_trace_hash']) != str(
            expected['evaluation_trace_hash']
        ):
            return []
    selected = selected.sort_values('checkpoint_slot')
    records = selected.to_dict(orient='records')
    for record in records:
        source_run = str(record.get('_reuse_source_run', ''))
        source_hash = str(record.get('resolved_config_hash', ''))
        source_gate = str(record.get('e2_r1_gate', ''))
        source_variant_record = str(record.get('module_variant', ''))
        expected_evaluation = (
            final_evaluation
            if bool(record.get('is_final_checkpoint', False))
            else regular_evaluation
        )
        record.update(context)
        record.update(expected_evaluation)
        record['agent'] = agent_name
        record['seed'] = int(seed)
        record['reused_curve'] = True
        record['reuse_source_run'] = source_run
        record['reuse_source_resolved_config_hash'] = source_hash
        record['reuse_source_e2_r1_gate'] = source_gate
        record['reused_from_module_variant'] = source_variant_record
    return records


def _reuse_evaluation_context(
    evaluation_sets: list[EvaluationSet],
) -> dict[str, Any]:
    if not evaluation_sets:
        return {}
    variant_name, _, traces, variant_context = evaluation_sets[0]
    evaluation_hash = hashlib.sha256(
        '|'.join(trace.trace_hash() for trace in traces).encode('utf-8')
    ).hexdigest()[:16]
    physical_hash = hashlib.sha256(
        '|'.join(
            trace.physical_trace_hash() for trace in traces
        ).encode('utf-8')
    ).hexdigest()[:16]
    return {
        **variant_context,
        **_theta_audit_context(traces, 'evaluation'),
        'evaluation_variant': variant_name,
        'evaluation_trace_ids': '|'.join(
            trace.trace_id for trace in traces
        ),
        'evaluation_trace_hash': evaluation_hash,
        'evaluation_physical_trace_hash': physical_hash,
    }


def _build_cases(
    config: dict[str, Any],
    scenarios: list[dict[str, Any]],
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    if config.get('experiment') == 'multi_vehicle_protection_ablation':
        variants = list(
            config.get(
                'protection_variants',
                ['gate_only', 'gate_plus_state', 'gate_state_mask'],
            )
        )
        return [
            (
                scenario,
                {
                    **config,
                    **{key: value for key, value in scenario.items() if key != 'name'},
                    'module_variant': 'full',
                    'protection_variant': str(variant),
                },
            )
            for scenario in scenarios
            for variant in variants
        ]
    if config.get('experiment') in {
        'multi_vehicle_descriptor_sensitivity',
        'multi_vehicle_external_validation',
    }:
        variants = list(config.get('descriptor_variants', [{'name': 'baseline'}]))
        cases: list[tuple[dict[str, Any], dict[str, Any]]] = []
        for scenario in scenarios:
            for variant in variants:
                case_config = {
                    **config,
                    **{key: value for key, value in scenario.items() if key != 'name'},
                    'descriptor_variant': str(variant.get('name', 'baseline')),
                    'module_variant': str(variant.get('module_variant', 'full')),
                    'descriptors': {
                        **config.get('descriptors', {}),
                        **variant.get('descriptors', {}),
                    },
                }
                cases.append((scenario, case_config))
        return cases
    if config.get('experiment') in {
        'multi_vehicle_module_ablation',
        'e2_r1_oracle_gate',
    }:
        configured_conditions = config.get('module_conditions')
        if configured_conditions is None:
            configured_conditions = [
                {'name': str(variant)}
                for variant in config.get(
                    'module_variants',
                    ['full', 'no_dqi', 'no_prediction', 'sensing_only'],
                )
            ]
        arrival_probabilities = list(
            config.get(
                'arrival_probabilities',
                [config.get('packet_traffic', {}).get('arrival_probability', 0.5)],
            )
        )
        cases: list[tuple[dict[str, Any], dict[str, Any]]] = []
        for scenario in scenarios:
            for arrival_probability in arrival_probabilities:
                for condition in configured_conditions:
                    condition = dict(condition)
                    name = str(condition.pop('name'))
                    condition_descriptors = condition.pop('descriptors', {})
                    case_config = {
                        **config,
                        **{key: value for key, value in scenario.items() if key != 'name'},
                        **condition,
                        'module_variant': name,
                        'descriptors': {
                            **config.get('descriptors', {}),
                            **condition_descriptors,
                        },
                        'packet_traffic': {
                            **config.get('packet_traffic', {}),
                            'arrival_probability': float(arrival_probability),
                        },
                    }
                    cases.append((scenario, case_config))
        return cases
    if config.get("experiment") != "multi_vehicle_scale_traffic":
        return [
            (scenario, {**config, **{key: value for key, value in scenario.items() if key != "name"}})
            for scenario in scenarios
        ]
    traffic_families = list(config.get("traffic_families", [config.get("traffic", {})]))
    arrival_probabilities = list(
        config.get(
            "arrival_probabilities",
            [config.get("packet_traffic", {}).get("arrival_probability", 0.5)],
        )
    )
    cases: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for scenario in scenarios:
        for traffic in traffic_families:
            for arrival_probability in arrival_probabilities:
                case_config = {
                    **config,
                    **{key: value for key, value in scenario.items() if key != "name"},
                    "traffic": traffic,
                    "packet_traffic": {
                        **config.get("packet_traffic", {}),
                        "arrival_probability": float(arrival_probability),
                    },
                }
                cases.append((scenario, case_config))
    return cases

def _build_evaluation_sets(
    config: dict[str, Any],
    scenario_name: str,
    seed: int,
    eval_episodes: int,
    eval_slots: int,
    nominal_traces: list[MultiVehicleAccessTrace],
    final_checkpoint: bool = False,
) -> list[EvaluationSet]:
    variants = list(config.get("evaluation_variants", []))
    if bool(config.get("_diagnostic_run", False)):
        variants = variants[:2]
    if not variants:
        return [("nominal", config, nominal_traces, {})]

    evaluation_sets: list[EvaluationSet] = []
    for variant in variants:
        name = str(variant.get("name", "nominal"))
        variant_config = {**config}
        for section in (
            "descriptors",
            "traffic",
            "underlay_power",
            "broadcast_component_update_slots",
        ):
            variant_config[section] = {
                **config.get(section, {}),
                **variant.get(section, {}),
            }
        for field in ("module_variant", "protection_variant", "broadcast_update_slots"):
            if field in variant:
                variant_config[field] = variant[field]
        is_nominal = name == "nominal" and not any(
            section in variant for section in ("descriptors", "traffic", "underlay_power")
        )
        traces = nominal_traces if is_nominal else [
            _resolve_trace(
                variant_config,
                scenario_name,
                seed=(
                    _final_evaluation_seed(seed, episode)
                    if final_checkpoint
                    else _evaluation_seed(seed, episode)
                ),
                slots=eval_slots,
                split=(
                    f"rl_final_eval_{episode}"
                    if final_checkpoint
                    else f"rl_eval_{episode}"
                ),
            )
            for episode in range(eval_episodes)
        ]
        traffic = variant_config.get("traffic", {})
        metadata = {
            **variant.get("metadata", {}),
            "evaluation_traffic_family": traffic.get("family", "fixed_markov"),
            "evaluation_traffic_label": traffic_label(traffic),
        }
        evaluation_sets.append((name, variant_config, traces, metadata))
    return evaluation_sets


def _train_and_evaluate(
    config: dict[str, Any],
    train_trace: MultiVehicleAccessTrace,
    evaluation_sets: list[EvaluationSet],
    final_evaluation_sets: list[EvaluationSet],
    agent_name: str,
    seed: int,
    checkpoint_interval: int,
    context: dict[str, Any],
) -> list[dict[str, Any]]:
    agent = create_shared_agent(
        agent_name,
        train_trace.num_channels,
        train_trace.num_vehicles,
        seed,
        settings=config.get("agent_settings", {}).get(agent_name, {}),
    )
    final_slot = train_trace.slots
    if not agent.learned:
        fixed_rows = _evaluate_sets(
            agent,
            final_evaluation_sets,
            final_slot,
            0.0,
            context,
        )
        for row in fixed_rows:
            row["is_final_checkpoint"] = True
        return fixed_rows

    env = MultiVehicleAccessEnv(train_trace, config)
    observations, infos = env.reset()
    agent.set_eval_mode(False)
    agent.reset_episode()
    rows: list[dict[str, Any]] = []
    training_seconds = 0.0
    done = False
    while not done:
        start = time.perf_counter()
        actions = agent.select_actions(observations, infos)
        next_observations, rewards, done, _ = env.step(actions)
        next_infos = env.agent_infos()
        agent.observe_batch(
            observations,
            actions,
            rewards,
            next_observations,
            done,
            infos,
            next_infos,
        )
        training_seconds += time.perf_counter() - start
        observations, infos = next_observations, next_infos
        checkpoint = env.t
        if checkpoint % checkpoint_interval == 0 or done:
            selected_evaluation_sets = (
                final_evaluation_sets if checkpoint == final_slot else evaluation_sets
            )
            print(
                "[multi_vehicle] "
                f"checkpoint {checkpoint}/{final_slot} "
                f"condition={context.get('module_variant', '')} "
                f"seed={seed} "
                f"agent={agent_name} "
                f"train_seconds={training_seconds:.1f}",
                flush=True,
            )
            checkpoint_rows = _evaluate_sets(
                agent,
                selected_evaluation_sets,
                checkpoint,
                training_seconds,
                context,
            )
            for row in checkpoint_rows:
                row["is_final_checkpoint"] = checkpoint == final_slot
            rows.extend(checkpoint_rows)
    return rows


def _evaluate_sets(
    agent: SharedPolicyAgent,
    evaluation_sets: list[EvaluationSet],
    checkpoint_slot: int,
    training_seconds: float,
    context: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for variant_name, variant_config, traces, variant_context in evaluation_sets:
        evaluation_hash = hashlib.sha256(
            '|'.join(trace.trace_hash() for trace in traces).encode('utf-8')
        ).hexdigest()[:16]
        physical_hash = hashlib.sha256(
            '|'.join(trace.physical_trace_hash() for trace in traces).encode('utf-8')
        ).hexdigest()[:16]
        rows.append(
            _evaluate_checkpoint(
                agent,
                traces,
                variant_config,
                checkpoint_slot,
                training_seconds,
                {
                    **context,
                    **variant_context,
                    **_theta_audit_context(traces, 'evaluation'),
                    'prediction_update_slots': int(
                        variant_config.get('descriptors', {}).get(
                            'prediction_update_slots', 1
                        )
                    ),
                    'prediction_age_max_slots': int(
                        max(
                            (
                                np.max(
                                    (trace.theta_target_slot[trace.theta_input_cutoff_slot >= 0]
                                     - trace.theta_input_cutoff_slot[trace.theta_input_cutoff_slot >= 0]
                                     - 1)
                                )
                                if np.any(trace.theta_input_cutoff_slot >= 0)
                                else 0
                            )
                            for trace in traces
                        )
                    ),
                    "evaluation_variant": variant_name,
                    'evaluation_trace_ids': '|'.join(trace.trace_id for trace in traces),
                    'evaluation_trace_hash': evaluation_hash,
                    'evaluation_physical_trace_hash': physical_hash,
                },
            )
        )
    return rows


def _evaluate_checkpoint(
    agent: SharedPolicyAgent,
    traces: list[MultiVehicleAccessTrace],
    config: dict[str, Any],
    checkpoint_slot: int,
    training_seconds: float,
    context: dict[str, Any],
) -> dict[str, Any]:
    snapshot = agent.snapshot_runtime_state()
    before_hash = agent.parameter_hash()
    agent.set_eval_mode(True)
    records: list[dict[str, Any]] = []
    inference_seconds = 0.0
    action_count = 0
    observation_dimension: int | None = None
    state_feature_signature: str | None = None
    mask_hasher = hashlib.sha256()
    theta_hasher = hashlib.sha256()
    dqi_hasher = hashlib.sha256()
    raw_allowed = 0
    raw_action_slots = 0
    busy_underlay_allowed = 0
    busy_underlay_slots = 0
    dqi_disagreements = 0
    dqi_disagreement_opportunities = 0
    invalid_action_count = 0
    for trace in traces:
        env = MultiVehicleAccessEnv(trace, config)
        observations, infos = env.reset()
        current_dimension = int(observations.shape[1])
        current_signature = '|'.join(env.state_features)
        if observation_dimension is None:
            observation_dimension = current_dimension
            state_feature_signature = current_signature
        elif (
            observation_dimension != current_dimension
            or state_feature_signature != current_signature
        ):
            raise ValueError('evaluation episodes disagree on observation schema')
        agent.reset_episode()
        done = False
        while not done:
            slices = feature_slices(trace.num_channels, env.state_features)
            warmup = int(config.get('r5c2_eval_warmup_slots', 0))
            if env.t < warmup:
                actions = agent.select_actions(observations, infos)
                observations, _, done, _ = env.step(actions)
                infos = env.agent_infos()
                continue
            if 'theta' in slices:
                theta_hasher.update(
                    np.ascontiguousarray(
                        observations[:, slices['theta']], dtype=np.float64
                    ).tobytes()
                )
            if 'fresh_dqi' in slices:
                dqi_hasher.update(
                    np.ascontiguousarray(
                        observations[:, slices['fresh_dqi']], dtype=np.float64
                    ).tobytes()
                )
            busy = np.asarray(trace.occupancy[env.t], dtype=bool)
            for info in infos:
                raw_mask = np.asarray(
                    info.get('raw_action_mask', info['action_mask']),
                    dtype=bool,
                )
                mask_hasher.update(np.ascontiguousarray(raw_mask).tobytes())
                raw_allowed += int(np.sum(raw_mask[1:]))
                raw_action_slots += int(raw_mask.size - 1)
                for channel in np.flatnonzero(busy):
                    underlay = encode_action(
                        int(channel), 'underlay', trace.num_channels
                    )
                    busy_underlay_allowed += int(raw_mask[underlay])
                    busy_underlay_slots += 1
            runtime_before_action = agent.snapshot_runtime_state()
            start = time.perf_counter()
            actions = agent.select_actions(observations, infos)
            inference_seconds += time.perf_counter() - start
            runtime_after_action = agent.snapshot_runtime_state()
            for action, info in zip(actions, infos):
                action_mask = np.asarray(info["action_mask"], dtype=bool)
                invalid_action_count += int(
                    int(action) < 0
                    or int(action) >= action_mask.size
                    or not action_mask[int(action)]
                )
            if (
                str(config.get('counterfactual_descriptor_disagreement', ''))
                == 'dqi_channel_permutation'
                and 'fresh_dqi' in slices
                and trace.num_channels > 1
            ):
                counterfactual = observations.copy()
                dqi_slice = slices['fresh_dqi']
                counterfactual[:, dqi_slice] = np.roll(
                    counterfactual[:, dqi_slice],
                    shift=1,
                    axis=1,
                )
                agent.restore_runtime_state(runtime_before_action)
                counterfactual_actions = agent.select_actions(counterfactual, infos)
                eligible = np.asarray(
                    [int(info.get('queue_length', 0)) > 0 for info in infos],
                    dtype=bool,
                )
                dqi_disagreements += int(
                    np.sum((counterfactual_actions != actions) & eligible)
                )
                dqi_disagreement_opportunities += int(np.sum(eligible))
                agent.restore_runtime_state(runtime_after_action)
            action_count += trace.num_vehicles
            next_observations, rewards, done, step_info = env.step(actions)
            next_infos = env.agent_infos()
            agent.observe_batch(
                observations,
                actions,
                rewards,
                next_observations,
                done,
                infos,
                next_infos,
            )
            records.append(step_info)
            observations, infos = next_observations, next_infos
    after_hash = agent.parameter_hash()
    agent.restore_runtime_state(snapshot)
    row = summarize_multi_vehicle_records(
        records,
        {
            **context,
            "agent": agent.name,
            "seed": agent.seed,
            "checkpoint_slot": int(checkpoint_slot),
            "parameter_hash_before_eval": before_hash,
            "parameter_hash_after_eval": after_hash,
            "evaluation_episodes": len(traces),
            "evaluation_slots": max(traces[0].slots - int(config.get('r5c2_eval_warmup_slots', 0)), 0) if traces else 0,
            "evaluation_warmup_slots": int(config.get('r5c2_eval_warmup_slots', 0)),
            "observation_dimension": int(observation_dimension or 0),
            "state_feature_signature": state_feature_signature or '',
        },
        config,
        training_time_seconds=training_seconds,
        inference_latency_ms=inference_seconds / max(action_count, 1) * 1000.0,
    )
    parameter_counts = _parameter_count_breakdown(agent)
    row["parameter_count"] = parameter_counts["trainable"]
    row["trainable_parameter_count"] = parameter_counts["trainable"]
    row["fixed_reservoir_parameter_count"] = parameter_counts["fixed_reservoir"]
    row["total_stored_parameter_count"] = parameter_counts["total"]
    row['parameter_freezing_hash'] = before_hash
    row['evaluation_parameters_frozen'] = bool(before_hash == after_hash)
    row['raw_action_mask_hash'] = mask_hasher.hexdigest()[:16]
    row['theta_observation_hash'] = theta_hasher.hexdigest()[:16]
    row['dqi_observation_hash'] = dqi_hasher.hexdigest()[:16]
    row['raw_allowed_action_fraction'] = float(
        raw_allowed / max(raw_action_slots, 1)
    )
    row['busy_underlay_allowed_rate'] = float(
        busy_underlay_allowed / max(busy_underlay_slots, 1)
    )
    row['dqi_action_disagreement_rate'] = float(
        dqi_disagreements / max(dqi_disagreement_opportunities, 1)
    )
    row['invalid_action_count'] = int(invalid_action_count)
    row['training_parameter_hash'] = before_hash
    if config.get('r5c2_protocol_version'):
        row['scored_physical_hash'] = hashlib.sha256(
            '|'.join(trace.physical_trace_hash() for trace in traces).encode()).hexdigest()
        row['r5c2_execution_stage'] = config.get('r5c2_stage', '')
        row['scored_slot_records'] = len(records)
        for name, field in (
            ('generated_packets', 'packet_generated'), ('dropped_packets', 'packet_dropped'),
            ('completed_packets', 'packet_completed'), ('transmitted_count', 'transmitted'),
            ('pu_access_violation_count', 'pu_access_violation'),
            ('pu_interference_violation_count', 'pu_violation_per_channel'),
            ('pu_busy_channel_slots', 'pu_busy_channels'),
        ):
            row[name] = sum(int(np.sum(record[field])) for record in records)
        row['total_channel_slots'] = sum(int(np.size(record['pu_violation_per_channel'])) for record in records)
        payloads = [getattr(trace, 'r5c2_freshness_audit', None) for trace in traces]
        if all(p is not None for p in payloads):
            warmup = int(config.get('r5c2_eval_warmup_slots', 0))
            for name, field in [('freshness_expired_fraction', 'expired'), ('freshness_discount_mean', 'discount')]:
                row[name] = float(np.mean(np.concatenate([p[field][warmup:].ravel() for p in payloads])))
    diagnostics = agent.training_diagnostics()
    row.update(diagnostics)
    initial_hash = str(diagnostics.get('training_initial_parameter_hash', ''))
    row['training_parameters_changed'] = bool(
        initial_hash and initial_hash != before_hash
    )
    return row


def _base_context(
    config: dict[str, Any],
    scenario_name: str | None,
    train_trace: MultiVehicleAccessTrace,
    evaluation_traces: list[MultiVehicleAccessTrace],
) -> dict[str, Any]:
    m = train_trace.num_channels
    n = train_trace.num_vehicles
    traffic = config.get("traffic", {})
    eval_hash = hashlib.sha256(
        "|".join(trace.trace_hash() for trace in evaluation_traces).encode("utf-8")
    ).hexdigest()[:16]
    eval_physical_hash = hashlib.sha256(
        "|".join(
            trace.physical_trace_hash() for trace in evaluation_traces
        ).encode("utf-8")
    ).hexdigest()[:16]
    mask_thresholds = {
        **DEFAULT_ACTION_MASK_THRESHOLDS,
        **config.get('action_mask_thresholds', {}),
    }
    return {
        'condition_id': str(config.get('r5c2_condition_id', '')),
        'target_id': str(config.get('r5c2_target_id', '')),
        'r5c2_protocol_hash': str(config.get('r5c2_protocol_hash', '')),
        'module_variant': str(config.get('module_variant', 'full')),
        'ablation_info_source': str(config.get('ablation_info_source', 'derived')),
        'ablation_mask_source': str(config.get('ablation_mask_source', 'derived')),
        'mask_policy_version': str(config.get('mask_policy_version', 'legacy_v1')),
        'sensing_idle_min': mask_thresholds['sensing_idle_min'],
        'overlay_theta_min': mask_thresholds['overlay_theta_min'],
        'underlay_power_cap_min': mask_thresholds['underlay_power_cap_min'],
        'underlay_theta_min': mask_thresholds['underlay_theta_min'],
        'protection_variant': str(config.get('protection_variant', 'gate_only')),
        'descriptor_variant': str(config.get('descriptor_variant', 'baseline')),
        'descriptor_source': str(config.get('descriptors', {}).get('source', 'generated')),
        'theta_information_condition': str(
            config.get('descriptors', {}).get(
                'theta_condition',
                'predicted',
            )
        ),
        'theta_representation': str(
            config.get('descriptors', {}).get(
                'theta_representation',
                'hard',
            )
        ),
        'validation_type': str(config.get('validation_type', 'synthetic')),
        'diagnostic_privileged': bool(config.get('diagnostic_privileged', False)),
        'run_scope': str(config.get('_run_scope', 'formal')),
        'diagnostic_run': bool(config.get('_diagnostic_run', False)),
        'e2_r1_gate': str(config.get('e2_r1_gate', '')),
        'resolved_config_hash': str(config.get('_resolved_config_hash', '')),
        **_theta_audit_context([train_trace], 'train'),
        **_theta_audit_context(evaluation_traces, 'evaluation'),
        'train_trace_id': train_trace.trace_id,
        'evaluation_trace_ids': '|'.join(trace.trace_id for trace in evaluation_traces),
        'dqi_model_id': train_trace.dqi_model_id,
        'dqi_model_hash': train_trace.dqi_model_hash,
        'prediction_model_id': train_trace.prediction_model_id,
        'prediction_model_hash': train_trace.prediction_model_hash,
        'sensing_info_source': str(config.get('sensing_info_source', 'real')),
        'prediction_info_source': str(
            config.get('prediction_info_source', config.get('ablation_info_source', 'real'))
        ),
        'sensing_mask_source': str(config.get('sensing_mask_source', 'real')),
        'prediction_mask_source': str(
            config.get('prediction_mask_source', config.get('ablation_mask_source', 'real'))
        ),
        'prediction_input_source': str(
            config.get('descriptors', {}).get('prediction_input_source', '')
        ),
        'prediction_horizon_slots': int(
            config.get('descriptors', {}).get('prediction_horizon_slots', 1)
        ),
        'prediction_update_slots': int(
            config.get('descriptors', {}).get('prediction_update_slots', 1)
        ),
        'prediction_age_max_slots': int(
            np.max(
                train_trace.theta_target_slot
                - train_trace.theta_input_cutoff_slot
                - 1
            )
        ),
        'descriptor_inference_latency_ms': (
            train_trace.dqi_inference_latency_ms
            + train_trace.prediction_inference_latency_ms
        ),
        'descriptor_parameter_count': (
            train_trace.dqi_parameter_count
            + train_trace.prediction_parameter_count
        ),
        'dqi_validation_mse': train_trace.dqi_validation_mse,
        'prediction_validation_brier': train_trace.prediction_validation_brier,
        "experiment": config.get("experiment", "multi_vehicle_baseline_comparison"),
        "scenario": scenario_name or f"M{m}_N{n}",
        "resource_regime": "M_gt_N" if m > n else "M_lt_N" if m < n else "M_eq_N",
        "load_ratio": float(n / max(m, 1)),
        "num_channels": m,
        "num_vehicles": n,
        "train_trace_hash": train_trace.trace_hash(),
        "train_physical_trace_hash": train_trace.physical_trace_hash(),
        "evaluation_trace_hash": eval_hash,
        "evaluation_physical_trace_hash": eval_physical_hash,
        "traffic_family": traffic.get("family", "fixed_markov"),
        "traffic_label": traffic_label(traffic),
        "arrival_probability": float(config.get("packet_traffic", {}).get("arrival_probability", 0.5)),
        "queue_capacity_packets": int(config.get("packet_traffic", {}).get("queue_capacity_packets", 10)),
        "packet_size_bits": int(config.get("packet_traffic", {}).get("packet_size_bits", 32_000)),
        'underlay_rate_model': str(config.get('underlay_rate_model', 'legacy_v1')),
        'enabled_modes': '|'.join(
            config.get('action', {}).get(
                'enabled_modes',
                ['overlay', 'underlay'],
            )
        ),
        'underlay_service_power_ceiling_fraction': float(
            config.get('underlay_power', {}).get(
                'service_power_ceiling_fraction',
                1.0,
            )
        ),
    }


def _default_scenario(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": f"M{int(config.get('num_channels', 0))}_N{int(config.get('num_vehicles', 0))}",
        "num_channels": int(config.get("num_channels", 10)),
        "num_vehicles": int(config.get("num_vehicles", 20)),
    }


def _parameter_count(agent: SharedPolicyAgent) -> int:
    children = getattr(agent, 'vehicle_agents', None)
    if children is not None:
        return int(sum(_parameter_count(child) for child in children))
    module = getattr(agent, "q_net", None) or getattr(agent, "model", None)
    if module is not None:
        count = int(sum(parameter.numel() for parameter in module.parameters()))
        log_alpha = getattr(agent, 'log_alpha', None)
        if log_alpha is not None and bool(getattr(log_alpha, 'requires_grad', False)):
            count += int(log_alpha.numel())
        return count
    q_table = getattr(agent, "q_table", None)
    if q_table is not None:
        return int(sum(np.asarray(values).size for values in q_table.values()))
    return 0


def _parameter_count_breakdown(agent: SharedPolicyAgent) -> dict[str, int]:
    children = getattr(agent, 'vehicle_agents', None)
    if children is not None:
        parts = [_parameter_count_breakdown(child) for child in children]
        return {
            key: int(sum(part[key] for part in parts))
            for key in ('trainable', 'fixed_reservoir', 'total')
        }
    trainable = _parameter_count(agent)
    fixed = 0
    for name in ('win', 'wres', '_win', '_wres'):
        value = getattr(agent, name, None)
        if value is not None:
            fixed += int(np.asarray(value).size)
    return {
        'trainable': int(trainable),
        'fixed_reservoir': int(fixed),
        'total': int(trainable + fixed),
    }

def _theta_audit_context(
    traces: list[MultiVehicleAccessTrace],
    prefix: str,
) -> dict[str, Any]:
    # Summarize theta without retaining privileged slot-level values.
    if not traces:
        return {
            f'{prefix}_theta_marginal_hash': '',
            f'{prefix}_theta_idle_correlation': float('nan'),
            f'{prefix}_theta_unique_count': 0,
            f'{prefix}_theta_unique_fraction': float('nan'),
            f'{prefix}_theta_min': float('nan'),
            f'{prefix}_theta_max': float('nan'),
            f'{prefix}_theta_busy_max': float('nan'),
            f'{prefix}_theta_idle_min': float('nan'),
            f'{prefix}_theta_threshold_accuracy': float('nan'),
        }
    marginal_hasher = hashlib.sha256()
    theta_values: list[np.ndarray] = []
    idle_values: list[np.ndarray] = []
    for trace in traces:
        theta = np.asarray(trace.theta, dtype=np.float64)
        idle = (~np.asarray(trace.occupancy, dtype=bool)).astype(np.float64)
        marginal_hasher.update(str(theta.shape).encode('utf-8'))
        marginal_hasher.update(
            np.ascontiguousarray(np.sort(theta, axis=0)).tobytes()
        )
        theta_values.append(theta.reshape(-1))
        idle_values.append(idle.reshape(-1))
    theta_flat = np.concatenate(theta_values)
    idle_flat = np.concatenate(idle_values)
    if np.std(theta_flat) <= 1e-12 or np.std(idle_flat) <= 1e-12:
        correlation = float('nan')
    else:
        correlation = float(np.corrcoef(theta_flat, idle_flat)[0, 1])
    unique_count = int(np.unique(theta_flat).size)
    busy_values = theta_flat[idle_flat < 0.5]
    idle_theta_values = theta_flat[idle_flat > 0.5]
    threshold_accuracy = float(
        np.mean((theta_flat >= 0.5) == (idle_flat > 0.5))
    )
    return {
        f'{prefix}_theta_marginal_hash': marginal_hasher.hexdigest()[:16],
        f'{prefix}_theta_idle_correlation': correlation,
        f'{prefix}_theta_unique_count': unique_count,
        f'{prefix}_theta_unique_fraction': float(
            unique_count / max(theta_flat.size, 1)
        ),
        f'{prefix}_theta_min': float(np.min(theta_flat)),
        f'{prefix}_theta_max': float(np.max(theta_flat)),
        f'{prefix}_theta_busy_max': (
            float(np.max(busy_values)) if busy_values.size else float('nan')
        ),
        f'{prefix}_theta_idle_min': (
            float(np.min(idle_theta_values))
            if idle_theta_values.size
            else float('nan')
        ),
        f'{prefix}_theta_threshold_accuracy': threshold_accuracy,
    }


def _resolve_trace(
    config: dict[str, Any],
    scenario: str,
    *,
    seed: int,
    slots: int,
    split: str,
) -> MultiVehicleAccessTrace:
    descriptor_cfg = dict(config.get('descriptors', {}))
    privileged_conditions = {
        str(descriptor_cfg.get('theta_condition', 'predicted')),
        str(descriptor_cfg.get('dqi_condition', 'predicted')),
    } & {'oracle', 'time_shuffled', 'future_oracle', 'channel_shuffled', 'channel_placebo', 'time_placebo'}
    if privileged_conditions and not bool(
        config.get(
            'diagnostic_privileged',
            descriptor_cfg.get('diagnostic_privileged', False),
        )
    ):
        raise ValueError(
            'oracle and shuffled-oracle descriptors require diagnostic_privileged=true'
        )
    # Cross-validate r_ref_mbps when both dqi and descriptors sections set it
    dqi_r_ref = config.get('dqi', {}).get('r_ref_mbps')
    desc_r_ref = descriptor_cfg.get('r_ref_mbps')
    if dqi_r_ref is not None and desc_r_ref is not None:
        if not np.isclose(float(dqi_r_ref), float(desc_r_ref), rtol=1e-12):
            raise ValueError(
                f'dqi.r_ref_mbps ({dqi_r_ref}) and '
                f'descriptors.r_ref_mbps ({desc_r_ref}) must be equal'
            )
    source = str(descriptor_cfg.get('source', 'generated'))
    if source == 'r5c2_physical_cache':
        from cvn_rl.experiments.r5c2_data import load_rl_physical
        trace = load_rl_physical(config, seed, slots, split)
        return _apply_descriptor_perturbations(trace, descriptor_cfg, seed)
    require_frozen = bool(descriptor_cfg.get('require_frozen_models', False))
    diagnostic = bool(config.get('_diagnostic_run', False))
    if source == 'artifact':
        template = str(descriptor_cfg.get('bundle_path_template', ''))
        if not template:
            raise ValueError('artifact descriptor source requires bundle_path_template')
        bundle_path = Path(
            template.format(
                scenario=scenario,
                split=split,
                seed=seed,
                descriptor_variant=config.get('descriptor_variant', 'baseline'),
            )
        )
        if bundle_path.exists():
            trace = load_descriptor_bundle(bundle_path, require_frozen_models=require_frozen)
            if trace.split != split:
                raise ValueError(
                    f'descriptor split mismatch: requested {split}, bundle has {trace.split}'
                )
            if trace.seed != seed:
                raise ValueError(
                    f'descriptor seed mismatch: requested {seed}, bundle has {trace.seed}'
                )
            expected_shape = (
                int(config.get('num_channels', trace.num_channels)),
                int(config.get('num_vehicles', trace.num_vehicles)),
            )
            if (trace.num_channels, trace.num_vehicles) != expected_shape:
                raise ValueError(
                    'descriptor scenario mismatch: '
                    f'requested M={expected_shape[0]}, N={expected_shape[1]}, '
                    f'bundle has M={trace.num_channels}, N={trace.num_vehicles}'
                )
            return _apply_descriptor_perturbations(trace, descriptor_cfg, seed)
        if not diagnostic:
            raise FileNotFoundError(f'missing formal descriptor bundle: {bundle_path}')
        source = 'generated'
    if source != 'generated':
        raise ValueError(f'unknown descriptor source: {source}')
    if require_frozen and not diagnostic:
        raise ValueError('formal run requires artifact descriptor bundles from frozen upstream models')

    trace = generate_multi_vehicle_trace(config, seed=seed, slots=slots, split=split)
    if bool(descriptor_cfg.get('persist_generated_bundles', False)):
        run_dir = config.get('_run_dir')
        if run_dir:
            bundle_path = Path(run_dir) / 'descriptors' / scenario / split / f'seed_{seed}'
            save_descriptor_bundle(
                bundle_path,
                trace,
                extra_manifest={
                    'diagnostic_surrogate': True,
                    'diagnostic_privileged': bool(
                        config.get(
                            'diagnostic_privileged',
                            descriptor_cfg.get('diagnostic_privileged', False),
                        )
                    ),
                },
            )
    return _apply_descriptor_perturbations(trace, descriptor_cfg, seed)


def _apply_descriptor_perturbations(
    trace: MultiVehicleAccessTrace,
    descriptor_cfg: dict[str, Any],
    seed: int,
) -> MultiVehicleAccessTrace:
    rng = np.random.default_rng(seed + 73_001)
    theta_shuffle_seed_offset = int(
        descriptor_cfg.get('theta_shuffle_seed_offset', 0)
    )
    theta_rng = (
        rng
        if theta_shuffle_seed_offset == 0
        else np.random.default_rng(seed + 73_001 + theta_shuffle_seed_offset)
    )
    theta_value_seed_offset = int(
        descriptor_cfg.get('theta_value_seed_offset', 0)
    )
    theta_value_rng = np.random.default_rng(
        seed + 173_001 + theta_value_seed_offset
    )
    mode = str(descriptor_cfg.get('diagnostic_mode', 'predicted'))
    if mode == 'oracle':
        r_ref_raw = descriptor_cfg.get('r_ref_mbps')
        if r_ref_raw is None:
            raise ValueError(
                'diagnostic_mode "oracle" requires an explicit '
                'descriptors.r_ref_mbps; refusing to use a hard-coded default'
            )
        r_ref = max(float(r_ref_raw), 1e-9)
        goodput = _latency_constrained_goodput(
            trace.quality_mbps,
            packet_size_bits=float(
                descriptor_cfg.get('packet_size_bits', 32_000)
            ),
            delay_max_ms=float(
                descriptor_cfg.get('dqi_delay_max_ms', 5.0)
            ),
            delay_beta_ms=float(
                descriptor_cfg.get('dqi_delay_beta_ms', 2.0)
            ),
        )
        shared_goodput = np.quantile(
            goodput, 0.10, axis=1, method='linear'
        )
        oracle_dqi = np.clip(shared_goodput / r_ref, 0.0, 1.0)
        trace.dqi_predicted[:] = oracle_dqi
        trace.fresh_dqi[:] = oracle_dqi
        trace.dqi_age[:] = 0.0
        trace.theta[:] = (~trace.occupancy).astype(float)
    elif mode == 'no_module':
        trace.fresh_dqi[:] = 0.5
        trace.theta[:] = 0.5

    theta_condition = str(descriptor_cfg.get('theta_condition', 'predicted'))
    if theta_condition in {'oracle', 'time_shuffled'}:
        theta_hard_oracle = (~trace.occupancy).astype(float)
        theta_oracle = _encode_theta_representation(
            theta_hard_oracle,
            descriptor_cfg,
            theta_value_rng,
        )
        trace.theta[:] = theta_oracle
        trace.theta_input_cutoff_slot[:] = np.arange(trace.slots, dtype=int)
        if theta_condition == 'time_shuffled':
            shuffle_fraction = float(descriptor_cfg.get('theta_shuffle_fraction', 1.0))
            for channel in range(trace.num_channels):
                if shuffle_fraction >= 1.0:
                    permutation = theta_rng.permutation(trace.slots)
                    trace.theta[:, channel] = theta_oracle[permutation, channel]
                else:
                    n_slots = trace.slots
                    n_shuffle = max(int(n_slots * shuffle_fraction), 0)
                    if n_shuffle == 0:
                        continue
                    shuffle_indices = theta_rng.choice(
                        n_slots, size=n_shuffle, replace=False
                    )
                    shuffled_values = theta_oracle[
                        theta_rng.permutation(n_shuffle), channel
                    ]
                    result = theta_oracle[:, channel].copy()
                    result[shuffle_indices] = shuffled_values
                    trace.theta[:, channel] = result
    elif theta_condition in {'frozen_model', 'central_frozen_model'}:
        from cvn_rl.predictors.frozen_predictor import FrozenPredictor

        model_path = str(descriptor_cfg.get('frozen_model_path', 'models/attention_lstm_periodic.pt'))
        if not Path(model_path).is_absolute():
            project_root = Path(__file__).resolve().parents[3]
            model_path = str(project_root / model_path)
        input_source = str(descriptor_cfg.get('prediction_input_source', ''))
        if input_source != 'bs_true_idle_history':
            raise ValueError('central frozen predictor requires prediction_input_source=bs_true_idle_history')
        if int(descriptor_cfg.get('prediction_horizon_slots', 1)) != 1:
            raise ValueError('central frozen predictor currently supports horizon 1 only')
        predictor = FrozenPredictor(
            model_path,
            device=str(_select_device()),
            expected_sha256=str(descriptor_cfg.get('expected_model_sha256', '')) or None,
            expected_input_source=input_source,
        )
        history_window = predictor.history_window
        neutral_val = float(descriptor_cfg.get('theta_neutral_value', 0.5))
        bs_idle = (~trace.occupancy).astype(np.float32)
        update_slots = max(int(descriptor_cfg.get('prediction_update_slots', 1)), 1)
        last_prediction = np.full(trace.num_channels, neutral_val, dtype=float)
        last_cutoff = -1
        for t in range(trace.slots):
            if t >= history_window and t % update_slots == 0:
                last_prediction = predictor.predict(bs_idle[t - history_window : t, :])
                last_cutoff = t - 1
            trace.theta[t] = last_prediction
            trace.theta_input_cutoff_slot[t] = last_cutoff
            trace.theta_target_slot[t] = t
        object.__setattr__(trace, 'prediction_model_id', str(predictor.meta.get('model', 'AttentionLSTMIdlePredictor')))
        object.__setattr__(trace, 'prediction_model_hash', predictor.sha256)
        object.__setattr__(trace, 'prediction_validation_brier', float(predictor.meta.get('metrics', {}).get('brier_score', np.nan)))
        object.__setattr__(trace, 'prediction_parameter_count', predictor.parameter_count)
        object.__setattr__(trace, 'prediction_inference_latency_ms', predictor.inference_latency_ms)
    elif theta_condition in {'frozen_time_shuffled', 'central_frozen_time_shuffled'}:
        # Frozen model predictions, then time-shuffled (placebo)
        from cvn_rl.predictors.frozen_predictor import FrozenPredictor

        model_path = str(descriptor_cfg.get('frozen_model_path', 'models/attention_lstm_periodic.pt'))
        if not Path(model_path).is_absolute():
            project_root = Path(__file__).resolve().parents[3]
            model_path = str(project_root / model_path)
        input_source = str(descriptor_cfg.get('prediction_input_source', ''))
        if input_source != 'bs_true_idle_history':
            raise ValueError('central frozen placebo requires BS true idle history')
        predictor = FrozenPredictor(
            model_path,
            device=str(_select_device()),
            expected_sha256=str(descriptor_cfg.get('expected_model_sha256', '')) or None,
            expected_input_source=input_source,
        )
        history_window = predictor.history_window
        neutral_val = float(descriptor_cfg.get('theta_neutral_value', 0.5))
        bs_idle = (~trace.occupancy).astype(np.float32)
        update_slots = max(int(descriptor_cfg.get('prediction_update_slots', 1)), 1)
        last_prediction = np.full(trace.num_channels, neutral_val, dtype=float)
        last_cutoff = -1
        for t in range(trace.slots):
            if t >= history_window and t % update_slots == 0:
                last_prediction = predictor.predict(bs_idle[t - history_window : t, :])
                last_cutoff = t - 1
            trace.theta[t] = last_prediction
            trace.theta_input_cutoff_slot[t] = last_cutoff
            trace.theta_target_slot[t] = t
        # Shuffle in time
        shuffle_seed_offset = int(descriptor_cfg.get('theta_shuffle_seed_offset', 0))
        shuffle_rng = (
            theta_rng
            if shuffle_seed_offset == 0
            else np.random.default_rng(seed + 73_001 + shuffle_seed_offset)
        )
        for channel in range(trace.num_channels):
            permutation = shuffle_rng.permutation(trace.slots)
            trace.theta[:, channel] = trace.theta[permutation, channel]
        object.__setattr__(trace, 'prediction_model_id', str(predictor.meta.get('model', 'AttentionLSTMIdlePredictor')))
        object.__setattr__(trace, 'prediction_model_hash', predictor.sha256)
        object.__setattr__(trace, 'prediction_validation_brier', float(predictor.meta.get('metrics', {}).get('brier_score', np.nan)))
        object.__setattr__(trace, 'prediction_parameter_count', predictor.parameter_count)
        object.__setattr__(trace, 'prediction_inference_latency_ms', predictor.inference_latency_ms)
    elif theta_condition == 'neutral':
        trace.theta[:] = float(descriptor_cfg.get('theta_neutral_value', 0.5))
    elif theta_condition == 'causal_time_shuffled':
        # Shuffle the existing causal theta in time (preserves marginal distribution)
        shuffle_seed_offset = int(descriptor_cfg.get('theta_shuffle_seed_offset', 0))
        causal_rng = (
            theta_rng
            if shuffle_seed_offset == 0
            else np.random.default_rng(seed + 73_001 + shuffle_seed_offset)
        )
        for channel in range(trace.num_channels):
            permutation = causal_rng.permutation(trace.slots)
            trace.theta[:, channel] = trace.theta[permutation, channel]

    # Apply staleness: shift theta backward, pad front with neutral
    staleness = int(descriptor_cfg.get('theta_staleness_slots', 0))
    if staleness > 0:
        neutral_val = float(descriptor_cfg.get('theta_neutral_value', 0.5))
        channels = trace.theta.shape[1]
        shifted = np.full_like(trace.theta, neutral_val)
        if staleness < trace.slots:
            shifted[staleness:] = trace.theta[:-staleness, :]
        trace.theta[:] = shifted

    dqi_condition = str(descriptor_cfg.get('dqi_condition', 'predicted'))
    if dqi_condition in {
        'future_oracle', 'channel_shuffled', 'channel_placebo', 'time_placebo', 'neutral', 'attention_frozen',
        'current_quality_q10', 'current_quality_rank', 'lag1_quality_q10', 'lag1_quality_rank',
    }:
        if dqi_condition in {
            'future_oracle', 'channel_shuffled', 'channel_placebo', 'time_placebo',
        } or bool(
            descriptor_cfg.get('compute_future_dqi_for_metrics', False)
        ):
            future_oracle = _future_quantile_dqi(trace, descriptor_cfg)
            trace.future_dqi_oracle[:] = future_oracle
        else:
            future_oracle = trace.future_dqi_oracle
        if dqi_condition == 'future_oracle':
            trace.dqi_predicted[:] = future_oracle
            trace.fresh_dqi[:] = future_oracle
        elif dqi_condition == 'attention_frozen':
            from cvn_rl.predictors.dqi_features import build_dqi_feature_tensor
            from cvn_rl.predictors.frozen_predictor import FrozenDQIPredictor

            model_path = str(descriptor_cfg.get('dqi_frozen_model_path', 'models/attention_lstm_dqi.pt'))
            if not Path(model_path).is_absolute():
                project_root = Path(__file__).resolve().parents[3]
                model_path = str(project_root / model_path)
            input_source = str(descriptor_cfg.get('dqi_prediction_input_source', 'dqi_channel_history'))
            predictor = FrozenDQIPredictor(
                model_path,
                device=str(_select_device()),
                expected_sha256=str(descriptor_cfg.get('expected_dqi_model_sha256', '')) or None,
                expected_input_source=input_source,
            )
            history_window = predictor.history_window
            r5c2 = descriptor_cfg.get('r5c2_protocol_hash')
            if r5c2:
                predictor.validate_target(descriptor_cfg['r5c2_target_parameters'], r5c2,
                                          float(descriptor_cfg['r_ref_mbps']))
            if predictor.num_channels != trace.num_channels:
                raise ValueError(
                    f'frozen DQI predictor has M={predictor.num_channels}, trace has M={trace.num_channels}'
                )
            features = build_dqi_feature_tensor(
                trace,
                predictor.feature_names,
                vehicle_quantile=float(descriptor_cfg.get('dqi_feature_vehicle_quantile', 0.10)),
                physical_scaling=predictor.meta.get('physical_scaling') if r5c2 else None,
            )
            neutral_value = float(descriptor_cfg.get('dqi_neutral_value', 0.5))
            update = max(int(descriptor_cfg.get('dqi_prediction_update_slots', descriptor_cfg.get('dqi_update_slots', 1))), 1)
            last_prediction = np.full(trace.num_channels, neutral_value, dtype=float)
            last_source = -1
            last_cutoff = -1
            for t in range(trace.slots):
                if t >= history_window and t % update == 0:
                    last_prediction = predictor.predict(features[t - history_window : t])
                    last_source = t
                    last_cutoff = t - 1
                trace.dqi_predicted[t] = last_prediction
                trace.fresh_dqi[t] = last_prediction
                trace.dqi_source_slot[t] = last_source
                trace.dqi_input_cutoff_slot[t] = last_cutoff
                trace.dqi_age[t] = max(t - last_source, 0) if last_source >= 0 else 0
                trace.dqi_validity_slots[t] = max(float(descriptor_cfg.get('dqi_prediction_validity_slots', history_window)), 1.0)
            if r5c2:
                from cvn_rl.simulation.dqi_freshness import apply_dqi_freshness
                if descriptor_cfg.get('resample_dqi_update_slots') is not None:
                    raise ValueError('R5-C2 forbids a second resample/discount path')
                freshness = descriptor_cfg['r5c2_freshness']
                coherence = np.vstack([np.ones((1, trace.num_channels)), trace.cci[:-1]])
                payload = apply_dqi_freshness(
                    trace.dqi_predicted, trace.dqi_age * freshness['slot_duration_ms'], coherence,
                    freshness['t_min_ms'], freshness['t_max_ms'], freshness['scale'], freshness['enabled'])
                available = trace.dqi_source_slot >= 0
                trace.fresh_dqi[:] = np.where(available[:, None], payload['fresh'], neutral_value)
                trace.dqi_validity_slots[:] = payload['validity_ms'] / freshness['slot_duration_ms']
                object.__setattr__(trace, 'r5c2_freshness_audit', payload)
                trace.dqi_metadata.update(target_parameters=descriptor_cfg['r5c2_target_parameters'],
                                          r_ref_mbps=float(descriptor_cfg['r_ref_mbps']))
                trace.dqi_metadata.update(
                    horizon_slots=int(descriptor_cfg['r5c2_target_parameters']['horizon_ms']),
                    vehicle_quantile=descriptor_cfg['r5c2_target_parameters']['vehicle_quantile'],
                    time_quantile=descriptor_cfg['r5c2_target_parameters']['time_quantile'],
                    delay_max_ms=descriptor_cfg['r5c2_target_parameters']['delay_max_ms'])
            object.__setattr__(trace, 'dqi_model_id', str(predictor.meta.get('model', 'AttentionLSTMDQIPredictor')))
            object.__setattr__(trace, 'dqi_model_hash', predictor.sha256)
            object.__setattr__(trace, 'dqi_validation_mse', predictor.validation_mse)
            object.__setattr__(trace, 'dqi_parameter_count', predictor.parameter_count)
            object.__setattr__(trace, 'dqi_inference_latency_ms', predictor.inference_latency_ms)
        elif dqi_condition in {
            'current_quality_q10', 'current_quality_rank', 'lag1_quality_q10', 'lag1_quality_rank',
        }:
            quantile = float(descriptor_cfg.get('dqi_quantile', 0.10))
            current_quality = np.quantile(
                trace.quality_mbps,
                quantile,
                axis=1,
                method=str(descriptor_cfg.get('quantile_method', 'linear')),
            )
            if dqi_condition.startswith('lag1_'):
                source_quality = np.empty_like(current_quality)
                source_quality[0] = current_quality[0]
                source_quality[1:] = current_quality[:-1]
            else:
                source_quality = current_quality
            if dqi_condition.endswith('_rank'):
                denom = max(trace.num_channels - 1, 1)
                ranked = np.empty_like(source_quality, dtype=float)
                for t in range(trace.slots):
                    order = np.argsort(source_quality[t], kind='mergesort')
                    ranks = np.empty(trace.num_channels, dtype=float)
                    ranks[order] = np.arange(trace.num_channels, dtype=float)
                    sorted_vals = source_quality[t, order]
                    start = 0
                    while start < trace.num_channels:
                        stop = start + 1
                        while stop < trace.num_channels and sorted_vals[stop] == sorted_vals[start]:
                            stop += 1
                        if stop - start > 1:
                            ranks[order[start:stop]] = (start + stop - 1) / 2.0
                        start = stop
                    ranked[t] = ranks / denom
                proxy_dqi = np.clip(ranked, 0.0, 1.0)
            else:
                r_ref = float(descriptor_cfg.get('r_ref_mbps'))
                if not (r_ref > 0 and np.isfinite(r_ref)):
                    raise ValueError(
                        'descriptors.r_ref_mbps must be a finite positive float for '
                        f'current-quality DQI proxy, got {descriptor_cfg.get("r_ref_mbps")!r}'
                    )
                proxy_dqi = normalize_dqi(source_quality, r_ref)
            trace.dqi_predicted[:] = proxy_dqi
            trace.fresh_dqi[:] = proxy_dqi
            trace.dqi_age[:] = 0.0 if dqi_condition.startswith('current_') else 1.0
            trace.dqi_source_slot[:] = np.arange(trace.slots, dtype=int)
            trace.dqi_input_cutoff_slot[:] = np.arange(trace.slots, dtype=int)
            if dqi_condition.startswith('lag1_'):
                trace.dqi_source_slot[1:] = np.arange(trace.slots - 1, dtype=int)
                trace.dqi_input_cutoff_slot[1:] = np.arange(trace.slots - 1, dtype=int)
            trace.dqi_validity_slots[:] = 1.0
            object.__setattr__(trace, 'dqi_model_id', dqi_condition)
            object.__setattr__(trace, 'dqi_model_hash', 'current_cqi_proxy')
            object.__setattr__(trace, 'dqi_validation_mse', float('nan'))
            object.__setattr__(trace, 'dqi_parameter_count', 0)
            object.__setattr__(trace, 'dqi_inference_latency_ms', 0.0)
        elif dqi_condition == 'channel_shuffled':
            update = max(int(descriptor_cfg.get('dqi_update_slots', 1_000)), 1)
            shuffled = np.empty_like(future_oracle)
            base = np.arange(trace.num_channels)
            # Optionally also shuffle quality_mbps so that current_cqi
            # (derived from quality_mbps) is de-correlated from the true
            # channel ordering.  Controlled by the descriptor flag
            # "shuffle_current_cqi" (default False = backward-compatible).
            shuffle_cqi = bool(descriptor_cfg.get('shuffle_current_cqi', False))
            if shuffle_cqi:
                shuffled_quality = np.empty_like(trace.quality_mbps)
            for source in range(0, trace.slots, update):
                stop = min(source + update, trace.slots)
                offset = int(rng.integers(1, trace.num_channels))
                permutation = np.roll(base, offset)
                shuffled[source:stop] = future_oracle[
                    source:stop,
                    permutation,
                ]
                if shuffle_cqi:
                    shuffled_quality[source:stop] = trace.quality_mbps[
                        source:stop, :, permutation
                    ]
            trace.dqi_predicted[:] = shuffled
            trace.fresh_dqi[:] = shuffled
            if shuffle_cqi:
                object.__setattr__(trace, '_shuffled_quality_mbps', shuffled_quality)
        elif dqi_condition == 'channel_placebo':
            block_slots = max(
                int(descriptor_cfg.get('placebo_block_slots', 100)), 1,
            )
            placebo, _ = make_channel_placebo(
                future_oracle,
                block_slots=block_slots,
                rng=rng,
            )
            trace.dqi_predicted[:] = placebo
            trace.fresh_dqi[:] = placebo
        elif dqi_condition == 'time_placebo':
            configured_block = max(
                int(descriptor_cfg.get('placebo_block_slots', 100)), 1,
            )
            # Auto-adjust block size: need 鈮?2 complete blocks
            block_slots = configured_block
            while block_slots > 1 and trace.slots // block_slots < 2:
                block_slots -= 1
            if trace.slots // block_slots < 2:
                raise ValueError(
                    f'time_placebo: {trace.slots} slots too few for '
                    f'block 鈮?{configured_block} (need 鈮?2 blocks)'
                )
            placebo, _ = make_time_block_placebo(
                future_oracle,
                block_slots=block_slots,
                rng=rng,
            )
            trace.dqi_predicted[:] = placebo
            trace.fresh_dqi[:] = placebo
        else:
            neutral_value = float(descriptor_cfg.get('dqi_neutral_value', 0.5))
            trace.dqi_predicted[:] = neutral_value
            trace.fresh_dqi[:] = neutral_value
        if dqi_condition in {'future_oracle', 'channel_shuffled', 'channel_placebo', 'time_placebo'}:
            update = max(int(descriptor_cfg.get('dqi_update_slots', 1_000)), 1)
            horizon = max(int(descriptor_cfg.get('dqi_horizon_slots', 10_000)), 1)
            for source in range(0, trace.slots, update):
                stop = min(source + update, trace.slots)
                trace.dqi_age[source:stop] = np.arange(
                    stop - source,
                    dtype=float,
                )[:, None]
                trace.dqi_validity_slots[source:stop] = horizon
                trace.dqi_source_slot[source:stop] = source
                trace.dqi_input_cutoff_slot[source:stop] = source + horizon

    pfa = float(descriptor_cfg.get('controlled_prediction_pfa', 0.0))
    pmd = float(descriptor_cfg.get('controlled_prediction_pmd', 0.0))
    if pfa > 0.0:
        false_alarm = (~trace.occupancy) & (rng.random(trace.occupancy.shape) < pfa)
        trace.theta[false_alarm] = np.minimum(trace.theta[false_alarm], 0.05)
    if pmd > 0.0:
        missed = trace.occupancy & (rng.random(trace.occupancy.shape) < pmd)
        trace.theta[missed] = np.maximum(trace.theta[missed], 0.95)
    drop_rate = float(descriptor_cfg.get('prediction_drop_rate', 0.0))
    if drop_rate > 0.0:
        dropped = rng.random(trace.theta.shape) < drop_rate
        trace.theta[dropped] = 0.5

    if (
        'controlled_sensing_pd' in descriptor_cfg
        or 'controlled_sensing_pfa' in descriptor_cfg
    ):
        pd_value = float(descriptor_cfg.get('controlled_sensing_pd', 0.9))
        pfa_value = float(descriptor_cfg.get('controlled_sensing_pfa', 0.1))
        busy = np.repeat(
            trace.occupancy[:, None, :],
            trace.num_vehicles,
            axis=1,
        )
        draws = rng.random(busy.shape)
        sensed_busy = np.where(busy, draws < pd_value, draws < pfa_value)
        trace.sensing_idle[:] = (~sensed_busy).astype(float)

    gain_error = descriptor_cfg.get('path_loss_error_db')
    margin_override = descriptor_cfg.get('fading_margin')
    if gain_error is not None or margin_override is not None:
        if gain_error is not None:
            factor = float(np.power(10.0, float(gain_error) / 10.0))
            trace.su_pu_gain_hat[:] = np.clip(
                trace.su_pu_gain_hat * factor,
                1e-12,
                None,
            )
        p_max = float(descriptor_cfg.get('p_max_mw', 1.0))
        margin = float(
            margin_override
            if margin_override is not None
            else descriptor_cfg.get('nominal_fading_margin', 2.0)
        )
        denominator = np.maximum(trace.su_pu_gain_hat * margin, 1e-12)
        transmit_cap = np.minimum(
            p_max,
            trace.pu_threshold_mw[:, None, :] / denominator,
        )
        trace.transmit_power_cap_mw[:] = transmit_cap
        trace.power_cap[:] = np.clip(transmit_cap / max(p_max, 1e-12), 0.0, 1.0)

    update_slots = descriptor_cfg.get('resample_dqi_update_slots')
    if update_slots is not None:
        update = max(int(update_slots), 1)
        original = trace.dqi_predicted.copy()
        for source in range(0, trace.slots, update):
            stop = min(source + update, trace.slots)
            prediction = original[source]
            validity = trace.dqi_validity_slots[source]
            age = np.arange(stop - source, dtype=float)[:, None]
            discount = np.exp(-age / np.maximum(validity, 1.0))
            trace.dqi_predicted[source:stop] = prediction
            trace.dqi_age[source:stop] = age
            trace.dqi_source_slot[source:stop] = source
            trace.dqi_input_cutoff_slot[source:stop] = source
            trace.fresh_dqi[source:stop] = np.where(
                age <= validity,
                prediction * discount,
                0.0,
            )
    return trace


def _encode_theta_representation(
    theta_hard_oracle: np.ndarray,
    descriptor_cfg: dict[str, Any],
    rng: np.random.Generator,
) -> np.ndarray:
    representation = str(
        descriptor_cfg.get('theta_representation', 'hard')
    ).lower()
    if representation == 'hard':
        return theta_hard_oracle.copy()
    if representation == 'soft':
        epsilon = float(descriptor_cfg.get('theta_soft_epsilon', 0.1))
        if not 0.0 < epsilon < 0.5:
            raise ValueError('theta_soft_epsilon must be in (0, 0.5)')
        return epsilon + (1.0 - 2.0 * epsilon) * theta_hard_oracle
    if representation == 'continuous_separable':
        busy_low, busy_high = [
            float(value)
            for value in descriptor_cfg.get(
                'theta_continuous_busy_range',
                [0.05, 0.45],
            )
        ]
        idle_low, idle_high = [
            float(value)
            for value in descriptor_cfg.get(
                'theta_continuous_idle_range',
                [0.55, 0.95],
            )
        ]
        if not (
            0.0 <= busy_low < busy_high < 0.5
            < idle_low < idle_high <= 1.0
        ):
            raise ValueError(
                'continuous theta ranges must be disjoint around 0.5'
            )
        unit = rng.random(theta_hard_oracle.shape)
        busy_values = busy_low + (busy_high - busy_low) * unit
        idle_values = idle_low + (idle_high - idle_low) * unit
        return np.where(theta_hard_oracle > 0.5, idle_values, busy_values)
    raise ValueError(f'unknown theta_representation: {representation}')


def _future_quantile_dqi(
    trace: MultiVehicleAccessTrace,
    descriptor_cfg: dict[str, Any],
) -> np.ndarray:
    """Compute future-oracle DQI using the canonical double-quantile algorithm.

    Delegates to ``dqi_labels.compute_unnormalized_dqi`` for algorithmic
    identity with the offline-audit DQI labels, then normalizes with the
    frozen *R_ref* supplied in ``descriptor_cfg``.
    """
    update = max(int(descriptor_cfg.get('dqi_update_slots', 1_000)), 1)
    horizon = max(int(descriptor_cfg.get('dqi_horizon_slots', 100)), 1)
    quantile = float(descriptor_cfg.get('dqi_quantile', 0.10))
    r_ref = float(descriptor_cfg.get('r_ref_mbps'))
    if not (r_ref > 0 and np.isfinite(r_ref)):
        raise ValueError(
            'descriptors.r_ref_mbps must be a finite positive float for '
            f'future-oracle DQI, got {descriptor_cfg.get("r_ref_mbps")!r}'
        )
    combined = np.concatenate(
        [trace.quality_mbps, trace.quality_lookahead_mbps],
        axis=0,
    )
    # Canonical DQI: q10 across vehicles, then q10 over future window t+1:t+H+1
    z_mbps, _source_slots = compute_unnormalized_dqi(
        combined,
        horizon_samples=horizon,
        vehicle_quantile=float(descriptor_cfg.get('dqi_vehicle_quantile', quantile)),
        time_quantile=float(descriptor_cfg.get('dqi_time_quantile', quantile)),
        quantile_method=str(descriptor_cfg.get('quantile_method', 'linear')),
        packet_size_bits=float(descriptor_cfg.get('packet_size_bits', 32_000)),
        delay_max_ms=float(descriptor_cfg.get('dqi_delay_max_ms', 5.0)),
        delay_beta_ms=float(descriptor_cfg.get('dqi_delay_beta_ms', 2.0)),
    )
    y = normalize_dqi(z_mbps, r_ref)  # [T-H, M], clipped to [0, 1]

    # Broadcast to trace.slots: for update > 1, hold last value between updates
    valid_slots = y.shape[0]
    output = np.empty((trace.slots, trace.num_channels), dtype=float)
    for source in range(0, trace.slots, update):
        stop = min(source + update, trace.slots)
        if source < valid_slots:
            output[source:stop] = y[source]
        else:
            # Past the last valid source slot: hold last available DQI
            output[source:stop] = y[-1]
    return output


def _evaluation_seed(seed: int, episode: int) -> int:
    return 1_000_003 + seed * 10_007 + episode * 97


def _final_evaluation_seed(seed: int, episode: int) -> int:
    return 2_000_033 + seed * 20_011 + episode * 193
