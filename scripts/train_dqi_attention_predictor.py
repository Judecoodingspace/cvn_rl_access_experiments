#!/usr/bin/env python3
"""Train a frozen Attention-LSTM predictor for canonical future-oracle DQI.

The target is the same canonical next-H-slot DQI used by the DQI oracle.  Inputs
are causal channel-history descriptors ending strictly before the decision slot.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from cvn_rl.experiments.config import load_config, set_global_seed
from cvn_rl.experiments.multi_vehicle_runner import _future_quantile_dqi
from cvn_rl.predictors.attention_lstm import AttentionLSTMDQIPredictor
from cvn_rl.predictors.dqi_features import DEFAULT_DQI_FEATURES, build_dqi_feature_tensor, flatten_dqi_history
from cvn_rl.simulation.multi_traces import generate_multi_vehicle_trace


@dataclass
class TraceCache:
    seed: int
    features: np.ndarray
    labels: np.ndarray
    sample_slots: np.ndarray


class DQIDataset(Dataset):
    def __init__(self, caches: list[TraceCache], history_slots: int, mean: np.ndarray, std: np.ndarray) -> None:
        self.caches = caches
        self.history_slots = int(history_slots)
        self.mean = mean.astype(np.float32)
        self.std = np.maximum(std.astype(np.float32), 1e-6)
        self.index: list[tuple[int, int]] = []
        for cache_idx, cache in enumerate(caches):
            for slot in cache.sample_slots:
                if int(slot) >= self.history_slots:
                    self.index.append((cache_idx, int(slot)))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        cache_idx, slot = self.index[idx]
        cache = self.caches[cache_idx]
        x = flatten_dqi_history(cache.features[slot - self.history_slots : slot])
        x = (x - self.mean[None, :]) / self.std[None, :]
        y = cache.labels[slot].astype(np.float32)
        return torch.from_numpy(x.astype(np.float32)), torch.from_numpy(y)


def _parse_ints(raw: str) -> list[int]:
    return [int(x) for x in raw.split(',') if x.strip()]


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind='mergesort')
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(len(values), dtype=float)
    sorted_vals = values[order]
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and sorted_vals[stop] == sorted_vals[start]:
            stop += 1
        if stop - start > 1:
            ranks[order[start:stop]] = (start + stop - 1) / 2.0
        start = stop
    return ranks


def _spearman_per_row(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    out = np.full(pred.shape[0], np.nan, dtype=float)
    for i, (p, y) in enumerate(zip(pred, target)):
        if np.allclose(p, p[0]) or np.allclose(y, y[0]):
            continue
        rp = _rankdata(p) - (len(p) - 1) / 2.0
        ry = _rankdata(y) - (len(y) - 1) / 2.0
        denom = np.sqrt(float(np.sum(rp * rp) * np.sum(ry * ry)))
        if denom > 0:
            out[i] = float(np.sum(rp * ry) / denom)
    return out


def _finite_mean(x: np.ndarray) -> float:
    return float(np.nanmean(x)) if np.isfinite(x).any() else float('nan')


def _build_caches(config: dict[str, Any], seeds: list[int], slots: int, sample_stride: int, feature_names: list[str], split: str) -> list[TraceCache]:
    caches: list[TraceCache] = []
    descriptor_cfg = dict(config.get('descriptors', {}))
    descriptor_cfg.update({
        'skip_causal_dqi': True,
        'dqi_condition': 'future_oracle',
        'dqi_update_slots': 1,
        'compute_future_dqi_for_metrics': True,
    })
    cfg = {**config, 'descriptors': descriptor_cfg}
    for seed in seeds:
        trace = generate_multi_vehicle_trace(cfg, seed=int(seed), slots=slots, split=split)
        labels = _future_quantile_dqi(trace, descriptor_cfg).astype(np.float32)
        features = build_dqi_feature_tensor(
            trace,
            feature_names,
            vehicle_quantile=float(descriptor_cfg.get('dqi_feature_vehicle_quantile', descriptor_cfg.get('dqi_quantile', 0.10))),
        )
        valid_stop = min(trace.slots, labels.shape[0])
        sample_slots = np.arange(0, valid_stop, max(int(sample_stride), 1), dtype=int)
        caches.append(TraceCache(seed=int(seed), features=features, labels=labels, sample_slots=sample_slots))
        print(f'cached seed={seed} samples={len(sample_slots)}', flush=True)
    return caches


def _normalization(caches: list[TraceCache]) -> tuple[np.ndarray, np.ndarray]:
    count = 0
    sum_x: np.ndarray | None = None
    sum_x2: np.ndarray | None = None
    for cache in caches:
        flat = cache.features.reshape(cache.features.shape[0], -1).astype(np.float64)
        if sum_x is None:
            sum_x = np.zeros(flat.shape[1], dtype=np.float64)
            sum_x2 = np.zeros(flat.shape[1], dtype=np.float64)
        sum_x += flat.sum(axis=0)
        sum_x2 += (flat * flat).sum(axis=0)
        count += flat.shape[0]
    assert sum_x is not None and sum_x2 is not None
    mean = sum_x / max(count, 1)
    var = np.maximum(sum_x2 / max(count, 1) - mean * mean, 1e-8)
    return mean.astype(np.float32), np.sqrt(var).astype(np.float32)


def _rank_hinge_loss(pred: torch.Tensor, target: torch.Tensor, margin: float = 0.02) -> torch.Tensor:
    diff_t = target.unsqueeze(2) - target.unsqueeze(1)
    sign = torch.sign(diff_t)
    mask = torch.abs(diff_t) > 1e-3
    if not torch.any(mask):
        return pred.new_tensor(0.0)
    diff_p = pred.unsqueeze(2) - pred.unsqueeze(1)
    losses = torch.relu(margin - sign * diff_p)
    return losses[mask].mean()


def _evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    preds: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    losses: list[float] = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            losses.append(float(nn.functional.mse_loss(pred, y).detach().cpu()))
            preds.append(pred.detach().cpu().numpy())
            targets.append(y.detach().cpu().numpy())
    p = np.concatenate(preds, axis=0) if preds else np.empty((0, 0), dtype=float)
    y = np.concatenate(targets, axis=0) if targets else np.empty((0, 0), dtype=float)
    rho = _spearman_per_row(p, y) if len(p) else np.asarray([], dtype=float)
    return {
        'mse': float(np.mean(losses)) if losses else float('nan'),
        'mae': float(np.mean(np.abs(p - y))) if len(p) else float('nan'),
        'spearman_mean': _finite_mean(rho),
        'spearman_p50': float(np.nanmedian(rho)) if np.isfinite(rho).any() else float('nan'),
        'top1_agreement': float((np.argmax(p, axis=1) == np.argmax(y, axis=1)).mean()) if len(p) else float('nan'),
        'predicted_mean': float(np.mean(p)) if len(p) else float('nan'),
        'future_mean': float(np.mean(y)) if len(y) else float('nan'),
        'predicted_p10': float(np.quantile(p, 0.10)) if len(p) else float('nan'),
        'predicted_p50': float(np.quantile(p, 0.50)) if len(p) else float('nan'),
        'predicted_p90': float(np.quantile(p, 0.90)) if len(p) else float('nan'),
        'future_p10': float(np.quantile(y, 0.10)) if len(y) else float('nan'),
        'future_p50': float(np.quantile(y, 0.50)) if len(y) else float('nan'),
        'future_p90': float(np.quantile(y, 0.90)) if len(y) else float('nan'),
    }


def _write_markdown(path: Path, metrics: dict[str, Any]) -> None:
    rows = metrics['offline_metrics']
    lines = [
        '# D3-A Attention DQI Offline Gate',
        '',
        f"model: `{metrics['model_path']}`",
        f"sha256: `{metrics['model_sha256']}`",
        '',
        '| split | mse | mae | spearman | top1 | pred_mean | oracle_mean |',
        '|---|---:|---:|---:|---:|---:|---:|',
    ]
    for split in ['train', 'val', 'test']:
        r = rows[split]
        lines.append(
            f"| {split} | {r['mse']:.6f} | {r['mae']:.6f} | {r['spearman_mean']:.4f} | "
            f"{r['top1_agreement']:.4f} | {r['predicted_mean']:.4f} | {r['future_mean']:.4f} |"
        )
    lines.extend([
        '',
        'Gate interpretation:',
        f"- val_spearman_ge_threshold: `{metrics['gate']['val_spearman_ge_threshold']}`",
        f"- val_top1_ge_threshold: `{metrics['gate']['val_top1_ge_threshold']}`",
        f"- val_calibration_abs_error_le_threshold: `{metrics['gate']['val_calibration_abs_error_le_threshold']}`",
        f"- passed: `{metrics['gate']['passed']}`",
    ])
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def run(args: argparse.Namespace) -> None:
    started = time.perf_counter()
    set_global_seed(int(args.seed))
    device = torch.device(args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu'))
    config = load_config(args.config)
    feature_names = [x.strip() for x in args.feature_names.split(',') if x.strip()]
    history_slots = int(args.history_slots)
    slots = int(args.slots)
    sample_stride = int(args.sample_stride)

    prepared_manifest = prepared_target = None
    if getattr(args, 'prepared_data', None):
        torch.manual_seed(int(args.seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(args.seed))
        from cvn_rl.experiments.r5c2_data import prepared_training_caches
        from cvn_rl.experiments.r5c2_protocol import read_json
        caches, prepared_manifest, prepared_target = prepared_training_caches(
            args.prepared_data, args.target_id, slots)
        if getattr(args, 'prepared_train_seed_count', None):
            caches['train'] = caches['train'][:args.prepared_train_seed_count]
        if not args.protocol_file or read_json(args.protocol_file)['protocol_hash'] != prepared_manifest['protocol_hash']:
            raise ValueError('prepared dataset and protocol file disagree')
        feature_names = prepared_manifest['preprocessing']['feature_names']
        if history_slots != prepared_manifest['history_slots']:
            raise ValueError('prepared history window mismatch')
        train_caches, val_caches, test_caches = (caches[s] for s in ('train', 'val', 'test'))
        mean = np.asarray(prepared_manifest['preprocessing']['mean'], dtype=np.float32)
        std = np.asarray(prepared_manifest['preprocessing']['std'], dtype=np.float32)
    else:
        train_caches = _build_caches(config, _parse_ints(args.train_seeds), slots, sample_stride, feature_names, 'd3_dqi_train')
        val_caches = _build_caches(config, _parse_ints(args.val_seeds), slots, sample_stride, feature_names, 'd3_dqi_val')
        test_caches = _build_caches(config, _parse_ints(args.test_seeds), slots, sample_stride, feature_names, 'd3_dqi_test')
        mean, std = _normalization(train_caches)

    train_ds = DQIDataset(train_caches, history_slots, mean, std)
    val_ds = DQIDataset(val_caches, history_slots, mean, std)
    test_ds = DQIDataset(test_caches, history_slots, mean, std)
    train_loader = DataLoader(train_ds, batch_size=int(args.batch_size), shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=int(args.batch_size), shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=int(args.batch_size), shuffle=False, num_workers=0)

    num_channels = int(config.get('num_channels', 10))
    input_size = num_channels * len(feature_names)
    model = AttentionLSTMDQIPredictor(
        input_size=input_size,
        hidden_size=int(args.hidden_size),
        num_layers=int(args.num_layers),
        attention_size=int(args.attention_size),
        dropout=float(args.dropout),
        output_size=num_channels,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(args.learning_rate), weight_decay=float(args.weight_decay))
    best_state = None
    best_val = float('inf')
    best_epoch = 0
    stale = 0
    log_rows: list[dict[str, Any]] = []
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        train_loss = []
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            loss = nn.functional.mse_loss(pred, y)
            if float(args.rank_loss_weight) > 0:
                loss = loss + float(args.rank_loss_weight) * _rank_hinge_loss(pred, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            train_loss.append(float(loss.detach().cpu()))
        val_metrics = _evaluate(model, val_loader, device)
        row = {'epoch': epoch, 'train_loss': float(np.mean(train_loss)), **{f'val_{k}': v for k, v in val_metrics.items()}}
        log_rows.append(row)
        print(f"epoch={epoch} train_loss={row['train_loss']:.6f} val_mse={val_metrics['mse']:.6f} val_spearman={val_metrics['spearman_mean']:.4f}", flush=True)
        if val_metrics['mse'] < best_val:
            best_val = float(val_metrics['mse'])
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= int(args.patience):
                break
    if best_state is not None:
        model.load_state_dict(best_state)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    model_path = out / args.model_name
    torch.save(model.state_dict(), model_path)
    sha = hashlib.sha256(model_path.read_bytes()).hexdigest()

    # Simple latency probe.
    probe = torch.zeros(1, history_slots, input_size, dtype=torch.float32, device=device)
    with torch.no_grad():
        t0 = time.perf_counter()
        for _ in range(50):
            _ = model(probe)
        latency_ms = (time.perf_counter() - t0) * 1000.0 / 50.0

    metrics = {
        'train': _evaluate(model, train_loader, device),
        'val': _evaluate(model, val_loader, device),
        'test': _evaluate(model, test_loader, device),
    }
    meta = {
        'model': 'AttentionLSTMDQIPredictor',
        'input_source': 'dqi_channel_history',
        'target_semantics': 'canonical_future_oracle_next_100_slot_dqi',
        'input_size': input_size,
        'output_size': num_channels,
        'num_channels': num_channels,
        'num_features': len(feature_names),
        'feature_names': feature_names,
        'feature_mean': mean.tolist(),
        'feature_std': std.tolist(),
        'hidden_size': int(args.hidden_size),
        'num_layers': int(args.num_layers),
        'attention_size': int(args.attention_size),
        'dropout': float(args.dropout),
        'history_window': history_slots,
        'prediction_horizon_slots': int(config.get('descriptors', {}).get('dqi_horizon_slots', 100)),
        'sample_stride': sample_stride,
        'model_sha256': sha,
        'training': {
            'config': str(Path(args.config)),
            'train_seeds': _parse_ints(args.train_seeds),
            'val_seeds': _parse_ints(args.val_seeds),
            'test_seeds': _parse_ints(args.test_seeds),
            'slots': slots,
            'epochs_requested': int(args.epochs),
            'best_epoch': best_epoch,
            'batch_size': int(args.batch_size),
            'learning_rate': float(args.learning_rate),
            'rank_loss_weight': float(args.rank_loss_weight),
            'seed': int(args.seed),
        },
        'metrics': {
            'parameter_count': int(sum(p.numel() for p in model.parameters())),
            'inference_latency_ms': float(latency_ms),
            'training_time_s': float(time.perf_counter() - started),
            'train_mse': metrics['train']['mse'],
            'val_mse': metrics['val']['mse'],
            'test_mse': metrics['test']['mse'],
            'val_spearman_mean': metrics['val']['spearman_mean'],
            'test_spearman_mean': metrics['test']['spearman_mean'],
            'val_top1_agreement': metrics['val']['top1_agreement'],
            'test_top1_agreement': metrics['test']['top1_agreement'],
        },
    }
    meta_path = model_path.with_suffix('.meta.json')
    if prepared_manifest is not None:
        meta.update(
            protocol_version='r5c2_v1', protocol_hash=prepared_manifest['protocol_hash'],
            target_id=args.target_id, target_parameters=prepared_target['parameters'],
            target_semantics='vehicle_then_time_future_service_quantile',
            prediction_horizon_slots=int(prepared_target['parameters']['horizon_ms']),
            r_ref_mbps=prepared_target['r_ref_mbps'], reference_source=prepared_target['reference_source'],
            physical_scaling=prepared_manifest['preprocessing']['physical_scaling'],
            preprocessing_hash=prepared_manifest['preprocessing']['hash'],
            dataset_files=prepared_manifest['files'], input_cutoff_rule='source_slot-1',
        )
        meta['training'].update(
            train_seeds=[c.seed for c in train_caches], val_seeds=[c.seed for c in val_caches],
            test_seeds=[c.seed for c in test_caches])
    meta_path.write_text(json.dumps(meta, indent=2), encoding='utf-8')
    pd.DataFrame(log_rows).to_csv(out / 'd3_attention_dqi_training_log.csv', index=False)
    pd.DataFrame([{'split': split, **vals} for split, vals in metrics.items()]).to_csv(out / 'd3_attention_dqi_offline_metrics.csv', index=False)
    gate = {
        'val_spearman_ge_threshold': bool(metrics['val']['spearman_mean'] >= float(args.gate_spearman)),
        'val_top1_ge_threshold': bool(metrics['val']['top1_agreement'] >= float(args.gate_top1)),
        'val_calibration_abs_error_le_threshold': bool(abs(metrics['val']['predicted_mean'] - metrics['val']['future_mean']) <= float(args.gate_calibration_abs)),
    }
    gate['passed'] = bool(all(gate.values()))
    payload = {
        'model_path': str(model_path),
        'model_sha256': sha,
        'meta_path': str(meta_path),
        'offline_metrics': metrics,
        'gate': gate,
    }
    (out / 'd3_attention_dqi_offline_gate.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
    _write_markdown(out / 'D3_ATTENTION_DQI_OFFLINE_GATE.md', payload)
    print(f'wrote {model_path}')
    print(f'sha256 {sha}')
    print(f"offline gate {'PASSED' if gate['passed'] else 'FAILED'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/e2_d3_attention_dqi_frozen.json')
    parser.add_argument('--prepared-data')
    parser.add_argument('--prepared-train-seed-count', type=int)
    parser.add_argument('--target-id')
    parser.add_argument('--protocol-file')
    parser.add_argument('--output-dir', default='models/d3_attention_dqi')
    parser.add_argument('--model-name', default='attention_lstm_dqi.pt')
    parser.add_argument('--train-seeds', default='1001,1002,1003,1004,1005,1006,1007,1008,1009,1010,1011,1012,1013,1014,1015,1016,1017,1018,1019,1020')
    parser.add_argument('--val-seeds', default='2001,2002,2003,2004,2005')
    parser.add_argument('--test-seeds', default='3001,3002,3003,3004,3005')
    parser.add_argument('--slots', type=int, default=50_000)
    parser.add_argument('--history-slots', type=int, default=200)
    parser.add_argument('--sample-stride', type=int, default=10)
    parser.add_argument('--feature-names', default=','.join(DEFAULT_DQI_FEATURES))
    parser.add_argument('--hidden-size', type=int, default=128)
    parser.add_argument('--num-layers', type=int, default=1)
    parser.add_argument('--attention-size', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--patience', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-5)
    parser.add_argument('--rank-loss-weight', type=float, default=0.1)
    parser.add_argument('--gate-spearman', type=float, default=0.45)
    parser.add_argument('--gate-top1', type=float, default=0.17)
    parser.add_argument('--gate-calibration-abs', type=float, default=0.05)
    parser.add_argument('--seed', type=int, default=20260710)
    parser.add_argument('--device', default='')
    run(parser.parse_args())


if __name__ == '__main__':
    main()
