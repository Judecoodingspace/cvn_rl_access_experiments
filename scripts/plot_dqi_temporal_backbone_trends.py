"""Train temporal DQI predictor backbones and plot validation trends.

Backbones: vanilla RNN, GRU, LSTM, and Attention-LSTM.  Non-epoch baselines
such as XGBoost and persistence are intentionally excluded from this figure.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
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


class RecurrentDQIPredictor(nn.Module):
    def __init__(self, cell: str, input_size: int, hidden_size: int, output_size: int, dropout: float = 0.1) -> None:
        super().__init__()
        cell = cell.lower()
        if cell == 'rnn':
            self.recurrent = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=1, batch_first=True)
        elif cell == 'gru':
            self.recurrent = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=1, batch_first=True)
        elif cell == 'lstm':
            self.recurrent = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=1, batch_first=True)
        else:
            raise ValueError(f'unknown recurrent cell: {cell}')
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden_seq, _ = self.recurrent(x)
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


def rank_hinge_loss(pred: torch.Tensor, target: torch.Tensor, margin: float = 0.02) -> torch.Tensor:
    diff_t = target.unsqueeze(2) - target.unsqueeze(1)
    sign = torch.sign(diff_t)
    mask = torch.abs(diff_t) > 1e-3
    if not torch.any(mask):
        return pred.new_tensor(0.0)
    diff_p = pred.unsqueeze(2) - pred.unsqueeze(1)
    losses = torch.relu(margin - sign * diff_p)
    return losses[mask].mean()


def listnet_loss(pred: torch.Tensor, target: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    temp = max(float(temperature), 1e-6)
    target_prob = torch.softmax(target / temp, dim=1)
    log_pred_prob = torch.log_softmax(pred / temp, dim=1)
    return -(target_prob * log_pred_prob).sum(dim=1).mean()


def build_caches(config: dict[str, Any], seeds: list[int], slots: int, sample_stride: int, feature_names: list[str], split: str) -> list[TraceCache]:
    descriptor_cfg = dict(config.get('descriptors', {}))
    descriptor_cfg.update({'skip_causal_dqi': True, 'dqi_condition': 'future_oracle', 'dqi_update_slots': 1, 'compute_future_dqi_for_metrics': True})
    cfg = {**config, 'descriptors': descriptor_cfg}
    caches: list[TraceCache] = []
    for seed in seeds:
        trace = generate_multi_vehicle_trace(cfg, seed=int(seed), slots=slots, split=split)
        labels = _future_quantile_dqi(trace, descriptor_cfg).astype(np.float32)
        features = build_dqi_feature_tensor(trace, feature_names, vehicle_quantile=float(descriptor_cfg.get('dqi_feature_vehicle_quantile', descriptor_cfg.get('dqi_quantile', 0.10))))
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


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, listnet_temperature: float) -> dict[str, float]:
    model.eval()
    preds: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    listnet_losses: list[float] = []
    with torch.no_grad():
        for x, y in loader:
            y_device = y.to(device)
            pred_device = model(x.to(device))
            listnet_losses.append(float(listnet_loss(pred_device, y_device, listnet_temperature).detach().cpu()))
            preds.append(pred_device.detach().cpu().numpy())
            targets.append(y.numpy())
    p = np.concatenate(preds, axis=0)
    y = np.concatenate(targets, axis=0)
    rho = spearman_per_row(p, y)
    return {
        'val_mse': float(np.mean((p - y) ** 2)),
        'val_mae': float(np.mean(np.abs(p - y))),
        'val_listnet_loss': float(np.mean(listnet_losses)),
        'val_spearman': finite_mean(rho),
        'val_top1_agreement': float((np.argmax(p, axis=1) == np.argmax(y, axis=1)).mean()),
    }


def train_one(name: str, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, device: torch.device, args: argparse.Namespace) -> list[dict[str, Any]]:
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(args.learning_rate), weight_decay=float(args.weight_decay))
    rows: list[dict[str, Any]] = []
    best_val = float('inf')
    stale = 0
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        losses: list[float] = []
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            mse = nn.functional.mse_loss(pred, y)
            rank = rank_hinge_loss(pred, y)
            loss = mse + float(args.rank_loss_weight) * rank
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        metrics = evaluate(model, val_loader, device, float(args.listnet_temperature))
        row = {'model': name, 'epoch': epoch, 'train_loss': float(np.mean(losses)), **metrics}
        rows.append(row)
        print(f'{name} epoch={epoch} train_loss={row["train_loss"]:.6f} val_mse={row["val_mse"]:.6f} val_listnet={row["val_listnet_loss"]:.6f}', flush=True)
        if row['val_mse'] < best_val - 1e-7:
            best_val = row['val_mse']
            stale = 0
        else:
            stale += 1
            if stale >= int(args.patience):
                break
    return rows


def plot_curves(df: pd.DataFrame, output_dir: Path) -> None:
    from cvn_rl.paper_figure_style import (
        FIG_IEEE_DIAGNOSTIC_TWO_PANEL,
        apply_ieee_diagnostic_two_panel_style,
        save_ieee_diagnostic_two_panel_figure,
        style_full_frame,
    )

    apply_ieee_diagnostic_two_panel_style()
    colors = {'RNN': '#c27ba0', 'GRU': '#e74c3c', 'LSTM': '#4c78a8', 'Attention-LSTM': '#222222'}
    markers = {'RNN': '^', 'GRU': 'v', 'LSTM': 's', 'Attention-LSTM': 'X'}
    fig, axes = plt.subplots(1, 2, figsize=FIG_IEEE_DIAGNOSTIC_TWO_PANEL)
    for model in ['RNN', 'GRU', 'LSTM', 'Attention-LSTM']:
        sub = df[df['model'] == model]
        if sub.empty:
            continue
        axes[0].plot(sub['epoch'], sub['val_mse'], marker=markers[model], markersize=5.5, linewidth=1.6, color=colors[model], label=model)
        axes[1].plot(sub['epoch'], sub['val_listnet_loss'], marker=markers[model], markersize=5.5, linewidth=1.6, color=colors[model], label=model)
    axes[0].set_ylabel('Validation MSE')
    axes[1].set_ylabel('Validation ListNet Loss')
    for ax in axes:
        ax.set_xlabel('Epoch')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', ncol=2, columnspacing=0.8, handlelength=1.6)
        style_full_frame(ax)
    fig.subplots_adjust(left=0.06, right=0.99, bottom=0.16, top=0.95, wspace=0.25)
    save_ieee_diagnostic_two_panel_figure(
        fig,
        str(output_dir / 'fig_dqi_temporal_backbone_validation_trends'),
        formats=('pdf', 'png', 'jpg'),
    )
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/e2_d3_r1_attention_dqi_frozen.json')
    parser.add_argument('--output-dir', default='paper_figures/dqi_temporal_backbone_trends')
    parser.add_argument('--train-seeds', default='1001,1002,1003,1004,1005,1006,1007,1008,1009,1010,1011,1012,1013,1014,1015,1016,1017,1018,1019,1020')
    parser.add_argument('--val-seeds', default='2001,2002,2003,2004,2005')
    parser.add_argument('--slots', type=int, default=50000)
    parser.add_argument('--history-slots', type=int, default=200)
    parser.add_argument('--sample-stride', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--patience', type=int, default=999)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-5)
    parser.add_argument('--rank-loss-weight', type=float, default=0.1)
    parser.add_argument('--listnet-temperature', type=float, default=1.0)
    parser.add_argument('--hidden-size', type=int, default=128)
    parser.add_argument('--attention-size', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=20260710)
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    args = parser.parse_args()
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but not available')
    device = torch.device(args.device)
    set_global_seed(int(args.seed))
    config = load_config(args.config)
    feature_names = list(DEFAULT_DQI_FEATURES)
    train_caches = build_caches(config, parse_ints(args.train_seeds), args.slots, args.sample_stride, feature_names, 'dqi_curve_train')
    val_caches = build_caches(config, parse_ints(args.val_seeds), args.slots, args.sample_stride, feature_names, 'dqi_curve_val')
    mean, std = normalization(train_caches)
    train_ds = DQIDataset(train_caches, args.history_slots, mean, std)
    val_ds = DQIDataset(val_caches, args.history_slots, mean, std)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    channels = int(config.get('num_channels', 10))
    input_size = channels * len(feature_names)
    specs = [('RNN', RecurrentDQIPredictor('rnn', input_size, args.hidden_size, channels, args.dropout)), ('GRU', RecurrentDQIPredictor('gru', input_size, args.hidden_size, channels, args.dropout)), ('LSTM', RecurrentDQIPredictor('lstm', input_size, args.hidden_size, channels, args.dropout)), ('Attention-LSTM', AttentionLSTMDQIPredictor(input_size=input_size, hidden_size=args.hidden_size, num_layers=1, attention_size=args.attention_size, dropout=args.dropout, output_size=channels))]
    rows: list[dict[str, Any]] = []
    for name, model in specs:
        set_global_seed(int(args.seed))
        rows.extend(train_one(name, model, train_loader, val_loader, device, args))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / 'dqi_temporal_backbone_training_log.csv', index=False)
    metadata = {'config': args.config, 'slots': args.slots, 'history_slots': args.history_slots, 'sample_stride': args.sample_stride, 'train_seeds': parse_ints(args.train_seeds), 'val_seeds': parse_ints(args.val_seeds), 'feature_names': feature_names, 'target': 'canonical_future_oracle_next_100_slot_dqi', 'device': str(device), 'models': [name for name, _ in specs], 'listnet_temperature': args.listnet_temperature, 'excluded': ['XGBoost', 'Persistence']}
    (output_dir / 'dqi_temporal_backbone_training_meta.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    plot_curves(df, output_dir)
    print(df.groupby('model').tail(1)[['model', 'epoch', 'val_mse', 'val_listnet_loss', 'val_spearman', 'val_top1_agreement']].to_string(index=False))


if __name__ == '__main__':
    main()
