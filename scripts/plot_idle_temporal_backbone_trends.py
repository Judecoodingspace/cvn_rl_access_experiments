"""Train temporal idle-probability backbones and plot validation trends.

Backbones: vanilla RNN, GRU, LSTM, and Attention-LSTM. Markov and XGBoost are
excluded because they do not provide comparable epoch-wise training curves.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from cvn_rl.predictors.attention_lstm import AttentionLSTMIdlePredictor
from cvn_rl.simulation.traffic import generate_occupancy


TRAFFIC: dict[str, Any] = {
    "family": "periodic_with_jitter",
    "period_slots": 50,
    "busy_slots": 20,
    "jitter_slots": 5,
}


@dataclass(frozen=True)
class Split:
    x: np.ndarray
    y: np.ndarray
    idle: np.ndarray


class RecurrentIdlePredictor(nn.Module):
    def __init__(self, cell: str, input_size: int, hidden_size: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        cell = cell.lower()
        recurrent_dropout = dropout if num_layers > 1 else 0.0
        if cell == "rnn":
            self.recurrent = nn.RNN(input_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=recurrent_dropout)
        elif cell == "gru":
            self.recurrent = nn.GRU(input_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=recurrent_dropout)
        elif cell == "lstm":
            self.recurrent = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=recurrent_dropout)
        else:
            raise ValueError(f"unknown recurrent cell: {cell}")
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden_seq, _ = self.recurrent(x)
        return torch.sigmoid(self.output(self.dropout(hidden_seq[:, -1, :])))


def make_split(seed: int, slots: int, history: int) -> Split:
    occupied = generate_occupancy(slots, 1, TRAFFIC, np.random.default_rng(seed))
    idle = (~occupied).astype(np.float32).reshape(-1)
    x = np.stack([idle[t - history : t] for t in range(history, slots)]).astype(np.float32)
    y = idle[history:].astype(np.float32)
    return Split(x=x[:, :, None], y=y, idle=idle)


def accuracy_from_prob(prob: torch.Tensor, target: torch.Tensor) -> float:
    pred = (prob >= 0.5).to(target.dtype)
    return float((pred == target).float().mean().detach().cpu())


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, loss_fn: nn.Module) -> dict[str, float]:
    model.eval()
    losses: list[float] = []
    accuracies: list[float] = []
    weights: list[int] = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            prob = model(xb)
            losses.append(float(loss_fn(prob, yb).detach().cpu()))
            accuracies.append(accuracy_from_prob(prob, yb))
            weights.append(int(len(xb)))
    w = np.asarray(weights, dtype=float)
    return {
        "val_loss": float(np.average(losses, weights=w)),
        "val_accuracy": float(np.average(accuracies, weights=w)),
    }


def train_one(
    name: str,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.learning_rate), weight_decay=float(args.weight_decay))
    loss_fn = nn.BCELoss()
    rows: list[dict[str, Any]] = []
    best_val = float("inf")
    stale = 0
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        train_losses: list[float] = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            prob = model(xb)
            loss = loss_fn(prob, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
        metrics = evaluate(model, val_loader, device, loss_fn)
        row = {"model": name, "epoch": epoch, "train_loss": float(np.mean(train_losses)), **metrics}
        rows.append(row)
        print(
            f'{name} epoch={epoch} train_loss={row["train_loss"]:.6f} '
            f'val_loss={row["val_loss"]:.6f} val_acc={row["val_accuracy"]:.4f}',
            flush=True,
        )
        if row["val_loss"] < best_val - 1e-7:
            best_val = row["val_loss"]
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
    colors = {"RNN": "#c27ba0", "GRU": "#e74c3c", "LSTM": "#4c78a8", "Attention-LSTM": "#222222"}
    markers = {"RNN": "^", "GRU": "v", "LSTM": "s", "Attention-LSTM": "X"}
    fig, axes = plt.subplots(1, 2, figsize=FIG_IEEE_DIAGNOSTIC_TWO_PANEL)
    for model in ["RNN", "GRU", "LSTM", "Attention-LSTM"]:
        sub = df[df["model"] == model]
        if sub.empty:
            continue
        axes[0].plot(sub["epoch"], sub["val_loss"], marker=markers[model], markersize=5.5, linewidth=1.6, color=colors[model], label=model)
        axes[1].plot(sub["epoch"], sub["val_accuracy"], marker=markers[model], markersize=5.5, linewidth=1.6, color=colors[model], label=model)
    axes[0].set_ylabel("Validation Loss")
    axes[1].set_ylabel("Validation Accuracy")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", ncol=2, columnspacing=0.8, handlelength=1.6)
        style_full_frame(ax)
    fig.subplots_adjust(left=0.06, right=0.99, bottom=0.16, top=0.95, wspace=0.25)
    save_ieee_diagnostic_two_panel_figure(
        fig,
        str(output_dir / "fig_idle_temporal_backbone_validation_trends"),
        formats=("pdf", "png", "jpg"),
    )
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="paper_figures/idle_temporal_backbone_trends_full40_acc")
    parser.add_argument("--slots", type=int, default=30_000)
    parser.add_argument("--history", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260703)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=999)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--attention-size", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    device = torch.device(args.device)

    split_seeds = {"train": args.seed + 101, "validation": args.seed + 202, "test": args.seed + 303}
    train = make_split(split_seeds["train"], args.slots, args.history)
    val = make_split(split_seeds["validation"], args.slots, args.history)
    train_ds = TensorDataset(torch.from_numpy(train.x), torch.from_numpy(train.y[:, None]))
    val_ds = TensorDataset(torch.from_numpy(val.x), torch.from_numpy(val.y[:, None]))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    specs = [
        ("RNN", RecurrentIdlePredictor("rnn", 1, args.hidden_size, args.num_layers, args.dropout)),
        ("GRU", RecurrentIdlePredictor("gru", 1, args.hidden_size, args.num_layers, args.dropout)),
        ("LSTM", RecurrentIdlePredictor("lstm", 1, args.hidden_size, args.num_layers, args.dropout)),
        ("Attention-LSTM", AttentionLSTMIdlePredictor(1, args.hidden_size, args.num_layers, args.attention_size, args.dropout, 1)),
    ]

    rows: list[dict[str, Any]] = []
    for name, model in specs:
        rows.extend(train_one(name, model, train_loader, val_loader, device, args))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "idle_temporal_backbone_training_log.csv", index=False)
    metadata = {
        "traffic": TRAFFIC,
        "slots": args.slots,
        "history": args.history,
        "horizon": 1,
        "split_seeds": split_seeds,
        "target": "next_slot_idle_probability",
        "device": str(device),
        "models": [name for name, _ in specs],
        "excluded": ["Markov", "XGBoost"],
    }
    (output_dir / "idle_temporal_backbone_training_meta.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    plot_curves(df, output_dir)
    print(df.groupby("model").tail(1)[["model", "epoch", "val_loss", "val_accuracy"]].to_string(index=False))


if __name__ == "__main__":
    main()
