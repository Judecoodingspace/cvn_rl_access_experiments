"""Shared physical caches and train-fitted targets for R5-C2."""
from __future__ import annotations

import copy
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cvn_rl.experiments.r5c2_protocol import (
    bundle_root, file_hash, fingerprint, read_json, write_json,
)
from cvn_rl.predictors.dqi_features import build_dqi_feature_tensor, _shared_vehicle_quantile
from cvn_rl.simulation.descriptor_bundle import load_descriptor_bundle, save_descriptor_bundle
from cvn_rl.simulation.dqi_labels import compute_unnormalized_dqi, fit_reference_rate
from cvn_rl.simulation.multi_traces import generate_multi_vehicle_trace


def raw_config(base: dict[str, Any], vehicles: int) -> dict[str, Any]:
    cfg = copy.deepcopy(base)
    cfg['num_vehicles'] = vehicles
    cfg['hidden_lookahead_slots'] = 200
    cfg['descriptors'] = {'source': 'generated', 'skip_causal_dqi': True,
                          'theta_condition': 'neutral', 'compute_future_dqi_for_metrics': False}
    return cfg


def ensure_raw_cache(path: Path, cfg: dict[str, Any], seed: int, slots: int, split: str):
    signature = fingerprint({'config': cfg, 'seed': seed, 'slots': slots, 'split': split})
    manifest_path = path / 'manifest.json'
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        if manifest.get('physical_config_hash') != signature or manifest.get('cache_kind') != 'physical':
            raise ValueError(f'raw cache configuration mismatch: {path}')
        return load_descriptor_bundle(path)
    if path.exists() and any(path.iterdir()):
        raise ValueError(f'incomplete raw cache; use a new cache path or remove only this partial cache: {path}')
    trace = generate_multi_vehicle_trace(cfg, seed=seed, slots=slots, split=split)
    trace.dqi_predicted[:] = .5
    trace.fresh_dqi[:] = .5
    trace.dqi_age[:] = 0
    trace.dqi_source_slot[:] = -1
    trace.dqi_input_cutoff_slot[:] = -1
    trace.theta[:] = .5
    trace.theta_input_cutoff_slot[:] = -1
    save_descriptor_bundle(path, trace, extra_manifest={
        'cache_kind': 'physical', 'physical_config_hash': signature,
        'physical_trace_hash': trace.physical_trace_hash(),
    })
    return trace


def _validate_prepared_files(root: Path, manifest: dict[str, Any]) -> None:
    for relative, expected in manifest['files'].items():
        path = root / relative
        if not path.is_file() or file_hash(path) != expected:
            raise ValueError(f'prepared file missing or changed: {path}')


def prepare_data(protocol: dict[str, Any], resume: bool = False) -> dict[str, Any]:
    started = time.perf_counter()
    root = bundle_root(protocol) / 'data'
    marker = root / 'prepared_manifest.json'
    if marker.exists():
        old = read_json(marker)
        if old['protocol_hash'] != protocol['protocol_hash']:
            raise ValueError('prepared dataset belongs to another protocol')
        _validate_prepared_files(root, old)
        if resume:
            return old
        raise FileExistsError('prepared data already complete; use --resume or a new output_root')
    root.mkdir(parents=True, exist_ok=True)
    entry = protocol['entry']
    cfg = raw_config(protocol['base_config'], entry['label_num_vehicles'])
    total = entry['label_data']['slots']
    stride = protocol['predictor_recipe']['sample_stride']
    source_slots = np.arange(entry['history_slots'], total, stride, dtype=int)
    if not len(source_slots):
        raise ValueError('no prepared history/target samples')
    names = protocol['predictor_recipe']['feature_names']
    entries = []
    quality_train = []
    bandwidth_max = 0.0
    for split, seeds in entry['label_data']['split_seeds'].items():
        for seed in seeds:
            relative = Path('raw_labels') / split / f'seed_{seed}'
            trace = ensure_raw_cache(root / relative, cfg, seed, total, f'r5c2_labels_{split}')
            entries.append(dict(split=split, seed=seed, raw_path=str(relative),
                                physical_hash=trace.physical_trace_hash()))
            if split == 'train':
                quality_train.append(_shared_vehicle_quantile(trace.quality_mbps, .1).astype(np.float32))
                bandwidth_max = max(bandwidth_max, float(np.max(trace.bandwidth_mhz)))
    scaling = dict(quality_mbps=max(float(np.quantile(np.concatenate(quality_train).ravel(), .95)), 1e-6),
                   bandwidth_mhz=max(bandwidth_max, 1e-6))
    del quality_train
    sums = np.zeros(entry['num_channels'] * len(names))
    sums2 = np.zeros_like(sums)
    count = 0
    target_values = {t['target_id']: {'train': []} for t in protocol['targets']}
    files: dict[str, str] = {}
    for row in entries:
        trace = load_descriptor_bundle(root / row['raw_path'])
        features = build_dqi_feature_tensor(trace, names, vehicle_quantile=.1, physical_scaling=scaling)
        feature_path = Path('features') / row['split'] / f"seed_{row['seed']}.npy"
        (root / feature_path).parent.mkdir(parents=True, exist_ok=True)
        np.save(root / feature_path, features)
        row['features_path'] = str(feature_path)
        if row['split'] == 'train':
            flat = features.reshape(len(features), -1).astype(float)
            sums += flat.sum(axis=0)
            sums2 += (flat * flat).sum(axis=0)
            count += len(flat)
        rates = np.concatenate([trace.quality_mbps, trace.quality_lookahead_mbps])
        for target in protocol['targets']:
            tid = target['target_id']
            z, indices = compute_unnormalized_dqi(
                rates, source_indices=source_slots,
                horizon_samples=int(target['horizon_ms'] / entry['dqi_sampling_ms']),
                vehicle_quantile=target['vehicle_quantile'], time_quantile=target['time_quantile'],
                delay_max_ms=target['delay_max_ms'], packet_size_bits=32000, delay_beta_ms=2,
            )
            relative = Path('targets') / tid / row['split'] / f"seed_{row['seed']}.npy"
            (root / relative).parent.mkdir(parents=True, exist_ok=True)
            np.save(root / relative, z.astype(np.float32))
            if row['split'] == 'train':
                target_values[tid]['train'].append(z)
        print(f"prepared split={row['split']} seed={row['seed']} samples={len(source_slots)}", flush=True)
    mean = sums / count
    std = np.sqrt(np.maximum(sums2 / count - mean * mean, 1e-8))
    preprocessing = dict(physical_scaling=scaling, feature_names=names, vehicle_quantile=.1,
                         mean=mean.tolist(), std=std.tolist(), training_rows=count)
    preprocessing['hash'] = fingerprint(preprocessing)
    targets = {}
    summaries = []
    for target in protocol['targets']:
        tid = target['target_id']
        reference = fit_reference_rate(target_values[tid]['train'], quantile=entry['reference_quantile'])
        targets[tid] = dict(parameters=target, r_ref_mbps=reference, reference_source='train_z_q95')
        for split in entry['label_data']['split_seeds']:
            arrays = [np.load(root / 'targets' / tid / split / f"seed_{r['seed']}.npy")
                      for r in entries if r['split'] == split]
            z = np.concatenate(arrays)
            summaries.append(dict(target_id=tid, split=split, samples=len(z), r_ref_mbps=reference,
                                  z_mean=float(z.mean()), saturation_fraction=float(np.mean(z >= reference)),
                                  rank_tie_row_fraction=float(np.mean(np.any(np.diff(np.sort(np.clip(z / reference, 0, 1), axis=1), axis=1) == 0, axis=1)))))
        for q in (.90, .95, .99):
            alternative = fit_reference_rate(target_values[tid]['train'], quantile=q)
            summaries.append(dict(target_id=tid, split='reference_diagnostic', reference_quantile=q,
                                  r_ref_mbps=alternative,
                                  saturation_fraction=float(np.mean(np.concatenate(target_values[tid]['train']) >= alternative)),
                                  rank_tie_row_fraction=float(np.mean(np.any(np.diff(np.sort(np.clip(
                                      np.concatenate(target_values[tid]['train']) / alternative, 0, 1), axis=1), axis=1) == 0, axis=1)))))
    np.save(root / 'source_slots.npy', source_slots)
    write_json(root / 'preprocessing.json', preprocessing)
    pd.DataFrame(summaries).to_csv(root / 'label_summary.csv', index=False)
    for path in root.rglob('*'):
        if path.is_file() and path != marker and not path.name.endswith('.partial'):
            files[str(path.relative_to(root))] = file_hash(path)
    manifest = dict(protocol_hash=protocol['protocol_hash'], entries=entries, targets=targets,
                    preprocessing=preprocessing, files=files, slots=total,
                    history_slots=entry['history_slots'], sample_stride=stride,
                    preparation_seconds=time.perf_counter() - started)
    write_json(marker, manifest)
    return manifest


def prepared_training_caches(data_root: str | Path, target_id: str, maximum_slots: int):
    # Import only when training: data preparation itself does not initialize Torch.
    from types import SimpleNamespace
    root = Path(data_root)
    manifest = read_json(root / 'prepared_manifest.json')
    _validate_prepared_files(root, manifest)
    target = manifest['targets'][target_id]
    indices = np.load(root / 'source_slots.npy')
    selected = indices < min(maximum_slots, manifest['slots'])
    if not np.any(selected):
        raise ValueError('training stage contains no prepared samples')
    caches = {'train': [], 'val': [], 'test': []}
    for row in manifest['entries']:
        features = np.load(root / row['features_path'], mmap_mode='r')[:maximum_slots]
        z = np.load(root / 'targets' / target_id / row['split'] / f"seed_{row['seed']}.npy")
        labels = np.zeros((len(features), z.shape[1]), dtype=np.float32)
        labels[indices[selected]] = np.clip(z[selected] / target['r_ref_mbps'], 0, 1)
        caches[row['split']].append(SimpleNamespace(seed=row['seed'], features=features,
                                                    labels=labels, sample_slots=indices[selected]))
    return caches, manifest, target


def rl_cache_path(root: Path, split: str, seed: int, slots: int) -> Path:
    return root / split / f'seed_{seed}_slots_{slots}'


def ensure_rl_caches(config: dict[str, Any]) -> None:
    from cvn_rl.experiments.multi_vehicle_runner import _evaluation_seed, _final_evaluation_seed
    root = Path(config['descriptors']['physical_cache_root'])
    raw = config['r5c2_physical_config']
    for seed in config['seeds']:
        requests = [('rl_train', seed, config['slots'])]
        for section, mapper, prefix in (
            ('checkpoint_evaluation', _evaluation_seed, 'rl_eval'),
            ('final_checkpoint_evaluation', _final_evaluation_seed, 'rl_final_eval'),
        ):
            budget = config[section]
            requests += [(f'{prefix}_{ep}', mapper(seed, ep), budget['eval_slots'])
                         for ep in range(budget['eval_episodes'])]
        for split, actual_seed, slots in requests:
            ensure_raw_cache(rl_cache_path(root, split, actual_seed, slots), raw, actual_seed, slots, split)


def load_rl_physical(config: dict[str, Any], seed: int, slots: int, split: str):
    root = Path(config['descriptors']['physical_cache_root'])
    path = rl_cache_path(root, split, seed, slots)
    manifest = read_json(path / 'manifest.json')
    expected = fingerprint(dict(config=config['r5c2_physical_config'], seed=seed, slots=slots, split=split))
    if manifest.get('cache_kind') != 'physical' or manifest.get('physical_config_hash') != expected:
        raise ValueError('physical cache protocol mismatch')
    trace = load_descriptor_bundle(path)
    if (trace.seed, trace.split, trace.slots, trace.num_vehicles, trace.num_channels) != (
            seed, split, slots, config['num_vehicles'], config['num_channels']):
        raise ValueError('physical cache seed/split/shape mismatch')
    if trace.physical_trace_hash() != manifest['physical_trace_hash']:
        raise ValueError('physical cache hash mismatch')
    return trace
