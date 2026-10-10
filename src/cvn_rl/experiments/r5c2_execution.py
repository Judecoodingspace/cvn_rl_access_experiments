"""Staged and restartable execution of the explicitly resolved R5-C2 protocol."""
from __future__ import annotations

import copy
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cvn_rl.experiments.r5c2_protocol import (
    ROOT, bundle_root, file_hash, fingerprint, project_path, read_json, write_json,
)
from cvn_rl.experiments.r5c2_data import ensure_raw_cache, ensure_rl_caches, raw_config
from cvn_rl.simulation.dqi_freshness import apply_dqi_freshness


def process_is_running(pid: int) -> bool:
    if os.name == 'nt':
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
        kernel.GetExitCodeProcess.restype = wintypes.BOOL
        kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            return ctypes.get_last_error() == 5  # access denied: treat as still active
        try:
            code = wintypes.DWORD()
            return not kernel.GetExitCodeProcess(handle, ctypes.byref(code)) or code.value == 259
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _execute(arguments: list[str], log: Path, on_start=None) -> float:
    log.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env['PYTHONPATH'] = str(ROOT / 'src') + os.pathsep + env.get('PYTHONPATH', '')
    started = time.perf_counter()
    print('running:', ' '.join(arguments), flush=True)
    with log.open('w', encoding='utf-8') as stream:
        process = subprocess.Popen([sys.executable, *arguments], cwd=ROOT, env=env,
                                   stdout=stream, stderr=subprocess.STDOUT)
        if on_start:
            on_start(process.pid)
        code = process.wait()
    if code:
        raise RuntimeError(f'command failed ({code}); see {log}')
    return time.perf_counter() - started


def pilot_freshness(protocol: dict[str, Any]) -> dict[str, Any]:
    entry = protocol['entry']
    root = bundle_root(protocol)
    slots = int(entry['pilot']['age_slots'])
    cfg = raw_config(protocol['base_config'], 10)
    trace = ensure_raw_cache(root / 'pilot' / 'age_physical', cfg, 91001, slots, 'r5c2_age_pilot')
    fresh = entry['freshness']
    coherence = np.vstack([np.ones((1, 10)), trace.cci[:-1]])
    rows = []
    selected = None
    candidates = list(fresh['update_candidates_slots']) + list(fresh.get('expanded_candidates_slots', []))
    for period in sorted(set(candidates)):
        first = math.ceil(entry['history_slots'] / period) * period
        if first >= slots:
            rows.append(dict(update_slots=period, passed=False, reason='pilot shorter than first update'))
            continue
        age = (np.arange(slots) % period)[:, None] * entry['slot_duration_ms']
        nominal = apply_dqi_freshness(np.ones_like(coherence), age, coherence,
                                     fresh['t_min_ms'], fresh['t_max_ms'], 1)
        half = apply_dqi_freshness(np.ones_like(coherence), age, coherence,
                                  fresh['t_min_ms'], fresh['t_max_ms'], .5)
        ratio = age[first:] / nominal['validity_ms'][first:]
        inside = float(np.mean((ratio > 0) & (ratio <= 1)))
        expiry = float(np.mean(half['expired'][first:]))
        passed = inside >= .10 and expiry >= .01
        rows.append(dict(update_slots=period, first_update=first, channel_decisions=ratio.size,
                         nonzero_fraction=float(np.mean(ratio > 0)), inside_fraction=inside,
                         half_expired_fraction=expiry, passed=passed,
                         per_channel_inside=np.mean((ratio > 0) & (ratio <= 1), axis=0).tolist(),
                         per_channel_half_expired=np.mean(half['expired'][first:], axis=0).tolist()))
        if passed and selected is None:
            selected = period
    payload = dict(protocol_hash=protocol['protocol_hash'], selected_update_slots=selected,
                   selection_basis='age_coverage_only', candidates=rows, physical_hash=trace.physical_trace_hash())
    write_json(root / 'pilot' / 'freshness_selection.json', payload)
    if selected is None:
        raise ValueError('no update candidate meets preregistered age coverage; inspect pilot report')
    return payload


def train_predictors(protocol: dict[str, Any], stage: str, targets: list[str] | None, resume: bool):
    root = bundle_root(protocol)
    if stage == 'formal':
        frozen = read_json(root / 'protocol_frozen.json')
        if frozen['protocol_hash'] != protocol['protocol_hash']:
            raise ValueError('formal predictor training requires matching frozen protocol')
    manifest = read_json(root / 'data' / 'prepared_manifest.json')
    recipe = protocol['predictor_recipe']
    index_path = root / ('model_index.json' if stage == 'formal' else 'pilot/model_index.json')
    index = read_json(index_path) if index_path.exists() else {}
    selected = targets or list(manifest['targets'])
    if not set(selected) <= set(manifest['targets']):
        raise ValueError('unknown target ID')
    for tid in selected:
        output = root / ('models' if stage == 'formal' else 'pilot/models') / tid
        training = dict(recipe)
        training.update(slots=protocol['entry']['label_data']['slots'] if stage == 'formal'
                        else min(protocol['entry']['pilot']['predictor_slots'], protocol['entry']['label_data']['slots']),
                        epochs=recipe['epochs'] if stage == 'formal' else protocol['entry']['pilot']['predictor_epochs'])
        signature = fingerprint(dict(protocol_hash=protocol['protocol_hash'], stage=stage, target=tid,
                                     training=training, dataset=manifest['files']))
        marker = output / 'training_task.json'
        if marker.exists():
            previous = read_json(marker)
            if previous['signature'] != signature:
                raise ValueError('completed model has different inputs; choose a new bundle')
            if resume and file_hash(output / 'attention_lstm_dqi.pt') == previous['model_sha256'] and file_hash(output / 'attention_lstm_dqi.meta.json') == previous['meta_sha256']:
                index[tid] = previous
                continue
            raise FileExistsError(f'model output already complete: {output}')
        config_path = root / 'training_configs' / f'{stage}_{tid}.json'
        cfg = copy.deepcopy(protocol['base_config'])
        cfg['num_vehicles'] = 15
        write_json(config_path, cfg)
        args = ['scripts/train_dqi_attention_predictor.py', '--config', str(config_path),
                '--output-dir', str(output), '--prepared-data', str(root / 'data'),
                '--target-id', tid, '--protocol-file', str(root / 'resolved_protocol.json'),
                '--slots', str(training['slots']), '--epochs', str(training['epochs']),
                '--history-slots', '200']
        for key in ('hidden_size', 'num_layers', 'attention_size', 'dropout', 'batch_size',
                    'learning_rate', 'rank_loss_weight', 'patience', 'weight_decay', 'sample_stride', 'seed'):
            args += ['--' + key.replace('_', '-'), str(training[key])]
        if stage == 'pilot':
            args += ['--prepared-train-seed-count', str(protocol['entry']['pilot']['predictor_train_seed_count'])]
        elapsed = _execute(args, output / 'train.log')
        meta = read_json(output / 'attention_lstm_dqi.meta.json')
        record = dict(signature=signature, stage=stage, target_id=tid, model_path=str(output / 'attention_lstm_dqi.pt'),
                      model_sha256=meta['model_sha256'], meta_sha256=file_hash(output / 'attention_lstm_dqi.meta.json'),
                      target_parameters=manifest['targets'][tid]['parameters'], r_ref_mbps=meta['r_ref_mbps'],
                      wall_seconds=elapsed, status='completed', protocol_hash=protocol['protocol_hash'])
        write_json(marker, record)
        index[tid] = record
        write_json(index_path, index)
    write_json(index_path, index)
    return index


def selected_period(protocol):
    payload = read_json(bundle_root(protocol) / 'pilot' / 'freshness_selection.json')
    if payload['protocol_hash'] != protocol['protocol_hash'] or payload['selected_update_slots'] is None:
        raise ValueError('matching age pilot required')
    return int(payload['selected_update_slots'])


def case_config(protocol, condition_id: str, seed: int, stage='formal', full_runtime=False):
    root = bundle_root(protocol)
    condition = (next(c for c in protocol['conditions'] if c['condition_id'] == condition_id)
                 if condition_id != 'B00' else {**protocol['conditions'][0], 'condition_id': 'B00'})
    models = read_json(root / ('model_index.json' if stage == 'formal' else 'pilot/model_index.json'))
    model = models[condition['target_id']]
    if (model['stage'] != stage or model['protocol_hash'] != protocol['protocol_hash'] or file_hash(model['model_path']) != model['model_sha256']
            or file_hash(Path(model['model_path']).with_suffix('.meta.json')) != model['meta_sha256']):
        raise ValueError('model index/hash mismatch')
    period = selected_period(protocol)
    warmup = math.ceil(200 / period) * period
    entry = protocol['entry']
    budget = entry['rl_budget']
    train_slots = budget['train_slots'] if stage == 'formal' or full_runtime else entry['pilot']['rl_train_slots']
    scored = budget['final_scored_slots'] if stage == 'formal' else entry['pilot']['scored_slots']
    episodes = budget['final_eval_episodes'] if stage == 'formal' else entry['pilot']['eval_episodes']
    cfg = copy.deepcopy(protocol['base_config'])
    cfg.update(seeds=[seed], slots=train_slots, quick_slots=train_slots,
               r5c2_condition_id=condition_id, r5c2_target_id=condition['target_id'],
               r5c2_protocol_hash=protocol['protocol_hash'], r5c2_eval_warmup_slots=warmup,
               module_variant='no_dqi' if condition_id == 'B00' else 'full',
               current_cqi_reference_mbps=models['T00']['r_ref_mbps'],
               r5c2_age_feature_scale_ms=entry['freshness']['age_feature_scale_ms'],
               expected_outputs=['metrics/access_metrics.csv', 'tables/checkpoint_summary.csv'])
    cfg['checkpoint_evaluation'] = dict(checkpoint_interval=train_slots, eval_episodes=1, eval_slots=warmup + scored)
    cfg['final_checkpoint_evaluation'] = dict(eval_episodes=episodes, eval_slots=warmup + scored)
    cfg['dqi']['r_ref_mbps'] = model['r_ref_mbps']
    cfg['dqi'].update(vehicle_quantile=condition['vehicle_quantile'], time_quantile=condition['time_quantile'],
                      horizon_ms=condition['horizon_ms'], delay_max_ms=condition['delay_max_ms'], history_window=200)
    cfg['dqi'].pop('horizon_seconds', None)
    descriptor = cfg['descriptors']
    descriptor.update(source='r5c2_physical_cache',
                      physical_cache_root=str(root / 'data' / f'rl_physical_{stage}'),
                      dqi_condition='neutral' if condition_id == 'B00' else 'attention_frozen',
                      dqi_neutral_value=.5, dqi_frozen_model_path=model['model_path'],
                      expected_dqi_model_sha256=model['model_sha256'],
                      dqi_prediction_update_slots=period, r_ref_mbps=model['r_ref_mbps'],
                      dqi_horizon_slots=int(condition['horizon_ms']),
                      dqi_vehicle_quantile=condition['vehicle_quantile'], dqi_time_quantile=condition['time_quantile'],
                      dqi_delay_max_ms=condition['delay_max_ms'], dqi_delay_beta_ms=2., packet_size_bits=32000,
                      dqi_apply_freshness_discount=condition['freshness_enabled'],
                      r5c2_target_parameters=model['target_parameters'],
                      r5c2_protocol_hash=protocol['protocol_hash'],
                      r5c2_freshness=dict(slot_duration_ms=1., t_min_ms=entry['freshness']['t_min_ms'],
                                          t_max_ms=entry['freshness']['t_max_ms'],
                                          scale=condition['freshness_scale'], enabled=condition['freshness_enabled']))
    cfg['broadcast_component_update_slots'] = dict(dqi=period, prediction=descriptor['prediction_update_slots'])
    cfg['r5c2_physical_config'] = raw_config(protocol['base_config'], 10)
    cfg['r5c2_stage'] = stage
    cfg['r5c2_execution_scope'] = entry['execution_scope']
    cfg['_run_scope'] = 'formal' if stage == 'formal' and entry['execution_scope'] == 'formal' else 'smoke'
    cfg['reporting_profile'] = cfg['_run_scope']
    return cfg


def freeze(protocol, phase):
    root = bundle_root(protocol)
    period = selected_period(protocol)
    data = read_json(root / 'data' / 'prepared_manifest.json')
    if data['protocol_hash'] != protocol['protocol_hash']:
        raise ValueError('prepared data protocol mismatch')
    if phase == 'protocol':
        runtime = read_json(root / 'pilot' / 'runtime_summary.json')
        if runtime['protocol_hash'] != protocol['protocol_hash'] or not runtime.get('full_training_probe'):
            raise ValueError('full-length runtime probe required before protocol freeze')
        payload = dict(protocol_hash=protocol['protocol_hash'], selected_update_slots=period,
                       data_manifest_hash=file_hash(root / 'data' / 'prepared_manifest.json'),
                       runtime_estimate=runtime, rl_budget=protocol['entry']['rl_budget'])
        write_json(root / 'protocol_frozen.json', payload)
        return payload
    frozen = read_json(root / 'protocol_frozen.json')
    if frozen['protocol_hash'] != protocol['protocol_hash']:
        raise ValueError('protocol freeze mismatch')
    models = read_json(root / 'model_index.json')
    if set(models) != {t['target_id'] for t in protocol['targets']}:
        raise ValueError('all eleven formal targets required')
    rows = []
    for condition in [c['condition_id'] for c in protocol['conditions']] + ['B00']:
        for seed in protocol['entry']['rl_seeds']:
            cfg = case_config(protocol, condition, seed)
            path = root / 'resolved_cases' / f'{condition}_s{seed}.json'
            write_json(path, cfg)
            rows.append(dict(condition_id=condition, seed=seed, path=str(path), config_hash=fingerprint(cfg)))
    write_json(root / 'runs_frozen.json', dict(protocol_hash=protocol['protocol_hash'], tasks=rows,
                                            model_index_hash=file_hash(root / 'model_index.json')))
    return rows


def run_cases(protocol, stage='formal', conditions=None, seeds=None, resume=False, full_runtime=False):
    root = bundle_root(protocol)
    known = {c['condition_id'] for c in protocol['conditions']} | {'B00'}
    if conditions and not set(conditions) <= known:
        raise ValueError('unknown condition ID')
    if seeds and not set(seeds) <= set(protocol['entry']['rl_seeds']):
        raise ValueError('seed outside contracted seed list')
    if stage == 'formal':
        frozen = read_json(root / 'runs_frozen.json')
        if frozen['protocol_hash'] != protocol['protocol_hash'] or frozen['model_index_hash'] != file_hash(root / 'model_index.json'):
            raise ValueError('formal run freeze/model index mismatch')
        all_tasks = frozen['tasks']
    else:
        all_tasks = []
        for cid in conditions or ['C00', 'C11', 'B00']:
            for seed in seeds or protocol['entry']['rl_seeds'][:2]:
                cfg = case_config(protocol, cid, seed, 'pilot', full_runtime)
                path = root / 'pilot' / 'resolved_cases' / f'{cid}_s{seed}_{"full" if full_runtime else "short"}.json'
                write_json(path, cfg)
                all_tasks.append(dict(condition_id=cid, seed=seed, path=str(path), config_hash=fingerprint(cfg)))
    if conditions:
        all_tasks = [t for t in all_tasks if t['condition_id'] in conditions]
    if seeds:
        all_tasks = [t for t in all_tasks if t['seed'] in seeds]
    index_path = root / ('run_index.json' if stage == 'formal' else 'pilot/run_index.json')
    index = read_json(index_path) if index_path.exists() else {}
    results_root = root / ('rl_runs' if stage == 'formal' else 'pilot/rl_runs')
    results_root.mkdir(parents=True, exist_ok=True)
    for task in all_tasks:
        key = f"{task['condition_id']}_s{task['seed']}" + ('_full' if full_runtime else '')
        if fingerprint(read_json(task['path'])) != task['config_hash']:
            raise ValueError('resolved case changed after freeze')
        previous = index.get(key)
        if previous and previous['status'] == 'running':
            if not previous.get('pid') or process_is_running(previous['pid']):
                raise RuntimeError(f'task process still running or unresolved: {key}, pid={previous.get("pid")}')
        if previous and previous['status'] == 'completed':
            if previous['config_hash'] != task['config_hash']:
                raise ValueError('completed task configuration mismatch')
            if resume and file_hash(Path(previous['run_dir']) / 'metrics/access_metrics.csv') == previous['metrics_hash']:
                continue
            raise FileExistsError(f'task already complete: {key}')
        cfg = read_json(task['path'])
        ensure_rl_caches(cfg)
        before = {p for p in results_root.iterdir() if p.is_dir()}
        index[key] = {**task, 'status': 'running'}
        write_json(index_path, index)
        try:
            def record_pid(pid):
                index[key]['pid'] = pid
                write_json(index_path, index)
            elapsed = _execute(['-m', 'cvn_rl.experiments.run_experiment', '--config', task['path'],
                                '--results-root', str(results_root), '--run-label', f'r2_r05c02_{key.lower()}'],
                               root / ('logs' if stage == 'formal' else 'pilot/logs') / f'{key}.log', record_pid)
            new = [p for p in results_root.iterdir() if p.is_dir() and p not in before]
            if len(new) != 1:
                raise ValueError('cannot uniquely identify task output directory')
            metrics_path = new[0] / 'metrics/access_metrics.csv'
            metrics = pd.read_csv(metrics_path)
            final = metrics[metrics['is_final_checkpoint'].astype(bool)]
            if len(final) != 1 or not bool(final.iloc[0]['evaluation_parameters_frozen']):
                raise ValueError('incomplete final evaluation or parameter mutation')
            index[key] = {**task, 'status': 'completed', 'run_dir': str(new[0]), 'wall_seconds': elapsed,
                          'metrics_hash': file_hash(metrics_path), 'stage': stage, 'full_runtime': full_runtime}
        except Exception as exc:
            index[key] = {**task, 'status': 'failed', 'error': str(exc)}
            write_json(index_path, index)
            raise
        write_json(index_path, index)
    return index


def pilot_runtime(protocol, conditions=None, seeds=None, full_probe=True, resume=False):
    root = bundle_root(protocol)
    index = run_cases(protocol, 'pilot', conditions, seeds, resume)
    if full_probe:
        index = run_cases(protocol, 'pilot', ['C00'], protocol['entry']['rl_seeds'][:1], resume, full_runtime=True)
    completed = [v for v in index.values() if v['status'] == 'completed']
    full = [v for v in completed if v.get('full_runtime')]
    model_index = read_json(root / 'pilot/model_index.json')
    import platform
    import torch
    environment = dict(python=sys.version.split()[0], platform=platform.system(), torch=torch.__version__,
                       cuda_available=torch.cuda.is_available(),
                       device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')
    training_slots = protocol['entry']['rl_budget']['train_slots']
    period = selected_period(protocol)
    warmup = math.ceil(200 / period) * period
    budget = protocol['entry']['rl_budget']
    pilot = protocol['entry']['pilot']
    work_ratio = (training_slots + budget['final_eval_episodes'] * (warmup + budget['final_scored_slots'])) / (
        training_slots + pilot['eval_episodes'] * (warmup + pilot['scored_slots']))
    formal_examples = len(protocol['entry']['label_data']['split_seeds']['train']) * max(
        protocol['entry']['label_data']['slots'] - 200, 1)
    pilot_examples = min(pilot['predictor_train_seed_count'], len(protocol['entry']['label_data']['split_seeds']['train'])) * max(
        min(pilot['predictor_slots'], protocol['entry']['label_data']['slots']) - 200, 1)
    predictor_ratio = formal_examples / pilot_examples * protocol['predictor_recipe']['epochs'] / max(pilot['predictor_epochs'], 1)
    payload = dict(protocol_hash=protocol['protocol_hash'], full_training_probe=bool(full), environment=environment,
                   rl_probes=completed, predictor_probes=model_index,
                   rl_75_task_wall_seconds_lower_bound=(75 * full[-1]['wall_seconds'] if full else None),
                   rl_75_task_scaled_estimate_seconds=(75 * full[-1]['wall_seconds'] * work_ratio if full else None),
                   predictor_11_scaled_estimate_seconds=11 * float(np.mean([m['wall_seconds'] for m in model_index.values()])) * predictor_ratio,
                   estimate_note='work-volume extrapolation only; startup, caching, early stopping and device saturation introduce uncertainty')
    write_json(root / 'pilot/runtime_summary.json', payload)
    return payload
