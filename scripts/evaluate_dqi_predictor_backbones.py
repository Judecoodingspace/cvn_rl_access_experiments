"""Evaluate DQI predictor backbones on one fixed DQI split.

The DQI target is the canonical future-oracle next-H-slot DQI used by the RL
experiments.  Metrics are regression and ranking metrics rather than binary
classification metrics.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.multioutput import MultiOutputRegressor
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


class LSTMDQIPredictor(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 1, dropout: float = 0.1, output_size: int = 10) -> None:
        super().__init__()
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=lstm_dropout)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden_seq, _ = self.lstm(x)
        return torch.sigmoid(self.output(self.dropout(hidden_seq[:, -1, :])))


def parse_ints(raw: str) -> list[int]:
    return [int(x) for x in raw.split(',') if x.strip()]


def rankdata(values: np.ndarray) -> np.ndarray:
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


def spearman_per_row(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    out = np.full(pred.shape[0], np.nan, dtype=float)
    for i, (p, y) in enumerate(zip(pred, target)):
        if np.allclose(p, p[0]) or np.allclose(y, y[0]):
            continue
        rp = rankdata(p) - (len(p) - 1) / 2.0
        ry = rankdata(y) - (len(y) - 1) / 2.0
        denom = np.sqrt(float(np.sum(rp * rp) * np.sum(ry * ry)))
        if denom > 0:
            out[i] = float(np.sum(rp * ry) / denom)
    return out


def finite_mean(x: np.ndarray) -> float:
    return float(np.nanmean(x)) if np.isfinite(x).any() else float('nan')


def build_caches(config: dict[str, Any], seeds: list[int], slots: int, sample_stride: int, feature_names: list[str], split: str) -> list[TraceCache]:
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
        print(f'cached split={split} seed={seed} samples={len(sample_slots)}', flush=True)
    return caches


def normalization(caches: list[TraceCache]) -> tuple[np.ndarray, np.ndarray]:
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


def collect_arrays(ds: DQIDataset) -> tuple[np.ndarray, np.ndarray]:
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for i in range(len(ds)):
        x, y = ds[i]
        xs.append(x.numpy())
        ys.append(y.numpy())
    return np.stack(xs, axis=0), np.stack(ys, axis=0)


def metrics(model: str, pred: np.ndarray, target: np.ndarray, extra: dict[str, Any]) -> dict[str, Any]:
    pred = np.clip(pred.astype(float), 0.0, 1.0)
    target = target.astype(float)
    rho = spearman_per_row(pred, target)
    return {
        'model': model,
        'mse': float(np.mean((pred - target) ** 2)),
        'mae': float(np.mean(np.abs(pred - target))),
        'spearman': finite_mean(rho),
        'top1_agreement': float((np.argmax(pred, axis=1) == np.argmax(target, axis=1)).mean()),
        **extra,
    }


def train_torch_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, test_loader: DataLoader, device: torch.device, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(args.learning_rate), weight_decay=float(args.weight_decay))
    best_state = None
    best_val = float('inf')
    best_epoch = 0
    stale = 0
    started = time.perf_counter()
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            loss = nn.functional.mse_loss(pred, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        val_mse = eval_mse(model, val_loader, device)
        print(f'{model.__class__.__name__} epoch={epoch} val_mse={val_mse:.6f}', flush=True)
        if val_mse < best_val - 1e-7:
            best_val = val_mse
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= int(args.patience):
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    pred, target, latency = predict_torch(model, test_loader, device)
    return pred, target, {
        'params': int(sum(p.numel() for p in model.parameters())),
        'latency_ms': float(latency),
        'epochs_run': int(best_epoch),
        'training_time_s': float(time.perf_counter() - started),
        'device': str(device),
    }


def eval_mse(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    vals: list[float] = []
    with torch.no_grad():
        for x, y in loader:
            pred = model(x.to(device))
            vals.append(float(nn.functional.mse_loss(pred, y.to(device)).detach().cpu()))
    return float(np.mean(vals)) if vals else float('nan')


def predict_torch(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray, float]:
    model.eval()
    preds: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    elapsed = 0.0
    count = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            start = time.perf_counter()
            pred = model(x)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            elapsed += time.perf_counter() - start
            count += len(x)
            preds.append(pred.detach().cpu().numpy())
            targets.append(y.numpy())
    return np.concatenate(preds, axis=0), np.concatenate(targets, axis=0), elapsed * 1000.0 / max(count, 1)


def persistence_predict(x: np.ndarray) -> np.ndarray:
    channels = x.shape[2] // len(DEFAULT_DQI_FEATURES)
    features = len(DEFAULT_DQI_FEATURES)
    last = x[:, -1, :].reshape(len(x), channels, features)
    return np.clip(last[:, :, 0], 0.0, 1.0)


def xgb_features(x: np.ndarray) -> np.ndarray:
    # Compact causal history summary for tree boosting.
    last = x[:, -1, :]
    mean = x.mean(axis=1)
    std = x.std(axis=1)
    p10 = np.quantile(x, 0.10, axis=1)
    p90 = np.quantile(x, 0.90, axis=1)
    return np.concatenate([last, mean, std, p10, p90], axis=1).astype(np.float32)


def train_xgboost(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, use_gpu: bool, seed: int) -> tuple[np.ndarray, dict[str, Any]]:
    from xgboost import XGBRegressor

    started = time.perf_counter()
    base = XGBRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective='reg:squarederror',
        random_state=seed,
        n_jobs=1,
        tree_method='hist',
        device='cuda' if use_gpu else 'cpu',
    )
    model = MultiOutputRegressor(base, n_jobs=1)
    train_feat = xgb_features(train_x)
    test_feat = xgb_features(test_x)
    model.fit(train_feat, train_y)
    start = time.perf_counter()
    pred = model.predict(test_feat)
    latency = (time.perf_counter() - start) * 1000.0 / max(len(test_feat), 1)
    return pred, {
        'params': np.nan,
        'latency_ms': float(latency),
        'epochs_run': np.nan,
        'training_time_s': float(time.perf_counter() - started),
        'device': 'cuda' if use_gpu else 'cpu',
    }


def write_outputs(rows: list[dict[str, Any]], output_dir: Path, metadata: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    order = ['Attention-LSTM', 'LSTM', 'XGBoost', 'Persistence']
    df['_order'] = df['model'].map({name: idx for idx, name in enumerate(order)}).fillna(99)
    df = df.sort_values(['_order', 'model']).drop(columns=['_order'])
    df.to_csv(output_dir / 'dqi_backbone_comparison.csv', index=False)
    display = df[['model', 'mse', 'mae', 'spearman', 'top1_agreement', 'params', 'latency_ms']].copy()
    for col in ['mse', 'mae', 'spearman', 'top1_agreement', 'latency_ms']:
        display[col] = display[col].map(lambda v: f'{v:.4f}' if np.isfinite(v) else 'NA')
    display['params'] = display['params'].map(lambda v: f'{int(v):,}' if np.isfinite(v) else 'NA')
    headers = ['Model', 'MSE', 'MAE', 'Spearman', 'Top-1 Agreement', 'Params', 'Latency ms']
    md = '# DQI Backbone Comparison\n\n'
    md += '| ' + ' | '.join(headers) + ' |\n'
    md += '| ' + ' | '.join(['---'] * len(headers)) + ' |\n'
    for _, row in display.iterrows():
        md += '| ' + ' | '.join(str(row[col]) for col in display.columns) + ' |\n'
    md += '\nAll rows use the same train/validation/test split. The target is canonical future-oracle DQI.\n'
    (output_dir / 'dqi_backbone_comparison.md').write_text(md, encoding='utf-8')
    (output_dir / 'dqi_backbone_comparison.tex').write_text(display.to_latex(index=False, escape=False), encoding='utf-8')
    (output_dir / 'dqi_backbone_comparison_meta.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/e2_d3_r1_attention_dqi_frozen.json')
    parser.add_argument('--output-dir', default='paper_table/table_dqi_backbone_comparison')
    parser.add_argument('--train-seeds', default='1001,1002,1003,1004,1005,1006,1007,1008,1009,1010,1011,1012,1013,1014,1015,1016,1017,1018,1019,1020')
    parser.add_argument('--val-seeds', default='2001,2002,2003,2004,2005')
    parser.add_argument('--test-seeds', default='3001,3002,3003,3004,3005')
    parser.add_argument('--slots', type=int, default=50_000)
    parser.add_argument('--history-slots', type=int, default=200)
    parser.add_argument('--sample-stride', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--patience', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-5)
    parser.add_argument('--hidden-size', type=int, default=128)
    parser.add_argument('--attention-size', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=20260710)
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--skip-xgboost', action='store_true')
    args = parser.parse_args()

    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but not available')
    device = torch.device(args.device)
    set_global_seed(int(args.seed))
    config = load_config(args.config)
    feature_names = list(DEFAULT_DQI_FEATURES)
    train_caches = build_caches(config, parse_ints(args.train_seeds), args.slots, args.sample_stride, feature_names, 'dqi_backbone_train')
    val_caches = build_caches(config, parse_ints(args.val_seeds), args.slots, args.sample_stride, feature_names, 'dqi_backbone_val')
    test_caches = build_caches(config, parse_ints(args.test_seeds), args.slots, args.sample_stride, feature_names, 'dqi_backbone_test')
    mean, std = normalization(train_caches)
    train_ds = DQIDataset(train_caches, args.history_slots, mean, std)
    val_ds = DQIDataset(val_caches, args.history_slots, mean, std)
    test_ds = DQIDataset(test_caches, args.history_slots, mean, std)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    train_x, train_y = collect_arrays(train_ds)
    test_x, test_y = collect_arrays(test_ds)
    channels = int(config.get('num_channels', 10))
    input_size = channels * len(feature_names)

    rows: list[dict[str, Any]] = []
    pred = persistence_predict(test_x)
    rows.append(metrics('Persistence', pred, test_y, {'params': 0, 'latency_ms': 0.0, 'epochs_run': 0, 'training_time_s': 0.0, 'device': 'none'}))

    lstm = LSTMDQIPredictor(input_size=input_size, hidden_size=args.hidden_size, dropout=args.dropout, output_size=channels)
    pred, target, extra = train_torch_model(lstm, train_loader, val_loader, test_loader, device, args)
    rows.append(metrics('LSTM', pred, target, extra))

    attention = AttentionLSTMDQIPredictor(input_size=input_size, hidden_size=args.hidden_size, num_layers=1, attention_size=args.attention_size, dropout=args.dropout, output_size=channels)
    pred, target, extra = train_torch_model(attention, train_loader, val_loader, test_loader, device, args)
    rows.append(metrics('Attention-LSTM', pred, target, extra))

    if not args.skip_xgboost:
        pred, extra = train_xgboost(train_x, train_y, test_x, device.type == 'cuda', int(args.seed))
        rows.append(metrics('XGBoost', pred, test_y, extra))

    metadata = {
        'config': args.config,
        'slots': args.slots,
        'history_slots': args.history_slots,
        'sample_stride': args.sample_stride,
        'train_seeds': parse_ints(args.train_seeds),
        'val_seeds': parse_ints(args.val_seeds),
        'test_seeds': parse_ints(args.test_seeds),
        'feature_names': feature_names,
        'target': 'canonical_future_oracle_next_100_slot_dqi',
        'torch_device': str(device),
        'xgboost_device': 'skipped' if args.skip_xgboost else ('cuda' if device.type == 'cuda' else 'cpu'),
    }
    write_outputs(rows, Path(args.output_dir), metadata)
    print(pd.DataFrame(rows)[['model', 'mse', 'mae', 'spearman', 'top1_agreement']].to_string(index=False))


if __name__ == '__main__':
    main()
