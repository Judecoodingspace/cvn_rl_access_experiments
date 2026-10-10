"""Explicit, opt-in protocol for the R5-C2 sensitivity experiment."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from cvn_rl.experiments.config import deep_merge, load_config

ROOT = Path(__file__).resolve().parents[3]
VERSION = 'r5c2_v1'


def fingerprint(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def file_hash(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def project_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def condition_matrix() -> list[dict[str, Any]]:
    values = [
        (.10, .10, 100, 5), (.05, .10, 100, 5), (.20, .10, 100, 5),
        (.10, .05, 100, 5), (.10, .20, 100, 5), (.10, .10, 50, 5),
        (.10, .10, 200, 5), (.10, .10, 100, 2.5), (.10, .10, 100, 10),
        (.10, .05, 200, 5), (.10, .20, 50, 5),
    ]
    rows = []
    for index, (qv, qt, horizon, delay) in enumerate(values):
        rows.append(dict(condition_id=f'C{index:02}', target_id=f'T{index:02}',
                         vehicle_quantile=qv, time_quantile=qt, horizon_ms=horizon,
                         delay_max_ms=delay, freshness_scale=1.0, freshness_enabled=True))
    for index, scale, enabled in [(11, .5, True), (12, 2., True), (13, 1., False)]:
        rows.append({**rows[0], 'condition_id': f'C{index:02}',
                     'freshness_scale': scale, 'freshness_enabled': enabled})
    return rows


def target_parameters(condition: dict[str, Any]) -> dict[str, Any]:
    return {key: condition[key] for key in (
        'target_id', 'vehicle_quantile', 'time_quantile', 'horizon_ms', 'delay_max_ms')}


def resolve_protocol(path: str | Path) -> dict[str, Any]:
    entry = load_config(project_path(path))
    entry.pop('_config_path', None)
    if entry.get('protocol_version') != VERSION:
        raise ValueError('unsupported R5-C2 protocol_version')
    entry.setdefault('execution_scope', 'formal')
    if entry['execution_scope'] not in {'formal', 'smoke'}:
        raise ValueError('execution_scope must be formal or smoke')
    if entry['execution_scope'] == 'formal':
        if entry['rl_seeds'] != [1, 2, 3, 4, 5]:
            raise ValueError('formal contract requires all five paired RL seeds')
        if entry['label_data']['slots'] != 50000:
            raise ValueError('short label data must be marked execution_scope=smoke')
        if [len(entry['label_data']['split_seeds'][s]) for s in ('train', 'val', 'test')] != [20, 5, 5]:
            raise ValueError('formal label split counts differ from the contract')
    if entry['agent'] != 'proposed_esn_ddqn' or len(entry['rl_seeds']) != len(set(entry['rl_seeds'])):
        raise ValueError('one contracted controller and unique RL seeds required')
    if entry['slot_duration_ms'] != entry['dqi_sampling_ms'] or entry['dqi_sampling_ms'] != 1:
        raise ValueError('R5-C2 v1 requires a common 1 ms simulation/DQI grid')
    if (entry['label_num_vehicles'], entry['rl_num_vehicles'], entry['num_channels']) != (15, 10, 10):
        raise ValueError('R5-C2 v1 requires N15 labels and N10/M10 access')
    if entry['history_slots'] != 200 or entry['max_horizon_slots'] != 200:
        raise ValueError('R5-C2 v1 requires history200/max future200')
    if entry['input_vehicle_quantile'] != .1 or entry['reference_quantile'] != .95:
        raise ValueError('input quantile and reference rule are fixed by contract')
    source = load_config(project_path(entry['scenario_source']))
    scene = next((s for s in source['scenarios'] if s['name'] == entry['scenario_name']), None)
    if scene is None:
        raise ValueError('requested source scenario not found')
    base = deep_merge(source, {k: v for k, v in scene.items() if k != 'name'})
    for key in ('_config_path', 'scenarios', 'pilot_gate', 'claim_gate', 'requires_pilot_gate',
                'requires_e2_r1_gate', 'reuse_gate_requirement', 'module_conditions',
                'module_variants', 'evaluation_variants', 'expected_outputs'):
        base.pop(key, None)
    base.update(num_vehicles=10, num_channels=10, agents=[entry['agent']],
                module_variant='full', experiment='multi_vehicle_baseline_comparison',
                hidden_lookahead_slots=200, slot_duration_ms=1,
                r5c2_protocol_version=VERSION, diagnostic_privileged=False)
    base['channel']['quality_family'] = base['channel'].get('quality_family', 'ar1')
    guidance = load_config(project_path(entry['guidance_source']))
    full = next(c for c in guidance['module_conditions'] if c['name'] == 'prediction_plus_dqi')
    # Use the common observation schema, not the source's physics or traffic.
    base['state_features'] = list(guidance['state_features'])
    descriptors = copy.deepcopy(full['descriptors'])
    descriptors.update(source='generated', skip_causal_dqi=True,
                       compute_future_dqi_for_metrics=False, require_frozen_models=False,
                       dqi_feature_vehicle_quantile=.1)
    for key in ('dqi_frozen_model_path', 'expected_dqi_model_sha256', 'dqi_model_id',
                'dqi_prediction_validity_slots'):
        descriptors.pop(key, None)
    base['descriptors'] = descriptors
    recipe = copy.deepcopy(entry['predictor_recipe'])
    if recipe.get('source_meta'):
        meta = read_json(project_path(recipe['source_meta']))
        for key in ('hidden_size', 'num_layers', 'attention_size', 'dropout'):
            recipe.setdefault(key, meta[key])
        for key in ('batch_size', 'learning_rate', 'rank_loss_weight', 'seed'):
            recipe.setdefault(key, meta['training'][key])
        recipe.setdefault('feature_names', meta['feature_names'])
    for key, default in [('epochs', 40), ('patience', 8), ('weight_decay', 1e-5), ('sample_stride', 10)]:
        recipe.setdefault(key, default)
    split_seeds = entry['label_data']['split_seeds']
    flattened = [int(s) for group in split_seeds.values() for s in group]
    if len(flattened) != len(set(flattened)) or any(not group for group in split_seeds.values()):
        raise ValueError('label split seeds must be nonempty and disjoint')
    if entry['label_data']['slots'] <= 200:
        raise ValueError('label traces must exceed history window')
    if entry['rl_budget']['checkpoint_interval'] != entry['rl_budget']['train_slots']:
        raise ValueError('this protocol evaluates the final checkpoint only')
    fresh = entry['freshness']
    if not 0 < fresh['t_min_ms'] <= fresh['t_max_ms'] or fresh['score_lag_slots'] != 1:
        raise ValueError('invalid freshness endpoints or noncausal score lag')
    if fresh['score_source'] != 'causal_trace_cci' or fresh['validity_timing'] != 'per_decision':
        raise ValueError('unsupported coherence source or validity timing')
    conditions = condition_matrix()
    source_hashes = {name: file_hash(project_path(entry[name]))
                     for name in ('scenario_source', 'guidance_source')}
    code_hashes = {name: file_hash(ROOT / name) for name in (
        'src/cvn_rl/experiments/r5c2_protocol.py', 'src/cvn_rl/experiments/r5c2_data.py',
        'src/cvn_rl/experiments/r5c2_execution.py', 'src/cvn_rl/experiments/r5c2_analysis.py',
        'scripts/r5c2_dqi_sensitivity.py',
        'src/cvn_rl/experiments/run_experiment.py', 'src/cvn_rl/multi_metrics.py',
        'src/cvn_rl/experiments/artifacts.py',
        'src/cvn_rl/simulation/multi_traces.py', 'src/cvn_rl/simulation/descriptor_bundle.py',
        'src/cvn_rl/agents/shared_multi.py', 'src/cvn_rl/agents/torch_agents.py',
        'src/cvn_rl/simulation/dqi_freshness.py', 'src/cvn_rl/simulation/dqi_labels.py',
        'src/cvn_rl/predictors/dqi_features.py', 'src/cvn_rl/predictors/frozen_predictor.py',
        'src/cvn_rl/experiments/multi_vehicle_runner.py', 'src/cvn_rl/simulation/multi_env.py',
        'scripts/train_dqi_attention_predictor.py') if (ROOT / name).exists()}
    resolved = dict(entry=entry, base_config=base, predictor_recipe=recipe,
                    conditions=conditions, targets=[target_parameters(c) for c in conditions[:11]],
                    source_hashes=source_hashes, code_hashes=code_hashes)
    resolved['protocol_hash'] = fingerprint(resolved)
    return resolved


def bundle_root(protocol: dict[str, Any]) -> Path:
    return project_path(protocol['entry']['output_root'])


def write_resolution(protocol: dict[str, Any]) -> Path:
    import pandas as pd
    root = bundle_root(protocol)
    existing = root / 'resolved_protocol.json'
    if existing.exists() and read_json(existing)['protocol_hash'] != protocol['protocol_hash']:
        raise ValueError('output bundle belongs to a different protocol; use a new output_root')
    write_json(existing, protocol)
    pd.DataFrame(protocol['conditions']).to_csv(root / 'conditions.csv', index=False)
    pd.DataFrame(protocol['targets']).to_csv(root / 'targets.csv', index=False)
    return root
