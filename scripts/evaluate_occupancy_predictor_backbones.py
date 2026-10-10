"""Evaluate idle-probability predictor backbones on one fixed split.

This script produces a paper-ready comparison table for next-slot idle
probability prediction.  It deliberately keeps the split and metrics common
across all backbones so the rows are directly comparable.
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
from sklearn.metrics import accuracy_score, f1_score, log_loss
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from cvn_rl.predictors.attention_lstm import AttentionLSTMIdlePredictor
from cvn_rl.simulation.traffic import generate_occupancy


TRAFFIC = {
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


class PlainLSTMIdlePredictor(nn.Module):
    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden_seq, _ = self.lstm(x)
        logits = self.output(self.dropout(hidden_seq[:, -1, :]))
        return torch.sigmoid(logits)


def make_split(seed: int, slots: int, history: int) -> Split:
    occupied = generate_occupancy(slots, 1, TRAFFIC, np.random.default_rng(seed))
    idle = (~occupied).astype(np.float32).reshape(-1)
    x = np.stack([idle[t - history : t] for t in range(history, slots)]).astype(np.float32)
    y = idle[history:].astype(np.float32)
    return Split(x=x[:, :, None], y=y, idle=idle)


def ece_score(y_true: np.ndarray, y_prob: np.ndarray, bins: int = 10) -> float:
    total = 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    for index, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        if index == bins - 1:
            mask = (y_prob >= lo) & (y_prob <= hi)
        else:
            mask = (y_prob >= lo) & (y_prob < hi)
        if mask.any():
            total += float(mask.mean()) * abs(float(y_prob[mask].mean() - y_true[mask].mean()))
    return float(total)


def metric_row(model: str, y_true: np.ndarray, y_prob: np.ndarray, extra: dict[str, Any]) -> dict[str, Any]:
    y_prob = np.clip(y_prob.astype(float), 1e-7, 1.0 - 1e-7)
    y_pred = (y_prob >= 0.5).astype(int)
    y_int = y_true.astype(int)
    return {
        "model": model,
        "accuracy": float(accuracy_score(y_int, y_pred)),
        "macro_f1": float(f1_score(y_int, y_pred, average="macro", zero_division=0)),
        "brier": float(np.mean((y_prob - y_true) ** 2)),
        "nll": float(log_loss(y_int, y_prob, labels=[0, 1])),
        "ece": ece_score(y_true, y_prob),
        **extra,
    }


def markov_predict(train_idle: np.ndarray, test: Split, smoothing: float = 1.0) -> np.ndarray:
    prev = train_idle[:-1].astype(int)
    nxt = train_idle[1:].astype(int)
    counts = np.zeros((2, 2), dtype=float) + smoothing
    for a, b in zip(prev, nxt):
        counts[a, b] += 1.0
    probs = counts / counts.sum(axis=1, keepdims=True)
    previous_idle = test.x[:, -1, 0].astype(int)
    return probs[previous_idle, 1]


def fit_torch_model(
    model: nn.Module,
    train: Split,
    val: Split,
    test: Split,
    *,
    seed: int,
    epochs: int,
    patience: int,
    batch_size: int,
    learning_rate: float,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, Any]]:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model.to(device)
    train_ds = TensorDataset(
        torch.from_numpy(train.x),
        torch.from_numpy(train.y[:, None]),
    )
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_x = torch.from_numpy(val.x).to(device)
    val_y = torch.from_numpy(val.y[:, None]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.BCELoss()
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0
    started = time.perf_counter()
    epochs_run = 0
    for epoch in range(epochs):
        model.train()
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(val_x), val_y).item())
        epochs_run = epoch + 1
        if val_loss < best_loss - 1e-6:
            best_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    test_x = torch.from_numpy(test.x).to(device)
    with torch.no_grad():
        tick = time.perf_counter()
        pred = model(test_x).detach().cpu().numpy().reshape(-1)
        latency_ms = (time.perf_counter() - tick) * 1000.0 / max(len(test.x), 1)
    return pred, {
        "params": int(sum(p.numel() for p in model.parameters())),
        "latency_ms": float(latency_ms),
        "epochs_run": int(epochs_run),
        "val_loss": float(best_loss),
        "training_time_s": float(time.perf_counter() - started),
    }


def fit_xgboost(train: Split, test: Split, seed: int, *, use_gpu: bool) -> tuple[np.ndarray, dict[str, Any]]:
    try:
        from xgboost import XGBClassifier
    except Exception as exc:  # pragma: no cover - depends on optional package.
        raise RuntimeError("xgboost is not installed in this environment") from exc
    started = time.perf_counter()
    model = XGBClassifier(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=seed,
        n_jobs=1,
        tree_method="hist",
        device="cuda" if use_gpu else "cpu",
    )
    model.fit(train.x.reshape(len(train.x), -1), train.y.astype(int))
    tick = time.perf_counter()
    pred = model.predict_proba(test.x.reshape(len(test.x), -1))[:, 1]
    latency_ms = (time.perf_counter() - tick) * 1000.0 / max(len(test.x), 1)
    return pred, {
        "params": np.nan,
        "latency_ms": float(latency_ms),
        "epochs_run": np.nan,
        "val_loss": float(np.nan),
        "training_time_s": float(time.perf_counter() - started),
        "device": "cuda" if use_gpu else "cpu",
    }


def write_outputs(rows: list[dict[str, Any]], output_dir: Path, metadata: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    order = ["Attention-LSTM", "Markov", "XGBoost", "LSTM"]
    df["_order"] = df["model"].map({name: idx for idx, name in enumerate(order)}).fillna(99)
    df = df.sort_values(["_order", "model"]).drop(columns=["_order"])
    df.to_csv(output_dir / "prediction_backbone_comparison.csv", index=False)

    display = df[["model", "accuracy", "macro_f1", "brier", "nll", "ece"]].copy()
    for col in display.columns[1:]:
        display[col] = display[col].map(lambda value: f"{value:.4f}" if np.isfinite(value) else "NA")
    headers = list(display.columns)
    md = "# Prediction Backbone Comparison\n\n"
    md += "| " + " | ".join(headers) + " |\n"
    md += "| " + " | ".join(["---"] * len(headers)) + " |\n"
    for _, row in display.iterrows():
        md += "| " + " | ".join(str(row[col]) for col in headers) + " |\n"
    md += "\n"
    md += "All rows use the same train/validation/test split. The target is next-slot idle probability.\n"
    (output_dir / "prediction_backbone_comparison.md").write_text(md, encoding="utf-8")
    (output_dir / "prediction_backbone_comparison.tex").write_text(
        display.to_latex(index=False, escape=False),
        encoding="utf-8",
    )
    (output_dir / "prediction_backbone_comparison_meta.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="paper_table/table_prediction_backbone_comparison")
    parser.add_argument("--slots", type=int, default=30_000)
    parser.add_argument("--history", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260703)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--include-xgboost", action="store_true")
    parser.add_argument("--require-xgboost", action="store_true")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    split_seeds = {
        "train": args.seed + 101,
        "validation": args.seed + 202,
        "test": args.seed + 303,
    }
    train = make_split(split_seeds["train"], args.slots, args.history)
    val = make_split(split_seeds["validation"], args.slots, args.history)
    test = make_split(split_seeds["test"], args.slots, args.history)

    rows: list[dict[str, Any]] = []
    markov_pred = markov_predict(train.idle, test)
    rows.append(metric_row("Markov", test.y, markov_pred, {
        "params": 2,
        "latency_ms": 0.0,
        "epochs_run": 0,
        "val_loss": np.nan,
        "training_time_s": 0.0,
    }))

    lstm_pred, lstm_extra = fit_torch_model(
        PlainLSTMIdlePredictor(1, 64, 1, 0.1),
        train,
        val,
        test,
        seed=args.seed,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=device,
    )
    rows.append(metric_row("LSTM", test.y, lstm_pred, lstm_extra))

    attention_pred, attention_extra = fit_torch_model(
        AttentionLSTMIdlePredictor(1, 64, 1, 64, 0.1, 1),
        train,
        val,
        test,
        seed=args.seed,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=device,
    )
    rows.append(metric_row("Attention-LSTM", test.y, attention_pred, attention_extra))

    if args.include_xgboost or args.require_xgboost:
        try:
            xgb_pred, xgb_extra = fit_xgboost(train, test, args.seed, use_gpu=device.type == "cuda")
            rows.append(metric_row("XGBoost", test.y, xgb_pred, xgb_extra))
        except RuntimeError:
            if args.require_xgboost:
                raise
            rows.append({
                "model": "XGBoost",
                "accuracy": np.nan,
                "macro_f1": np.nan,
                "brier": np.nan,
                "nll": np.nan,
                "ece": np.nan,
                "params": np.nan,
                "latency_ms": np.nan,
                "epochs_run": np.nan,
                "val_loss": np.nan,
                "training_time_s": np.nan,
                "device": "unavailable",
            })

    metadata = {
        "traffic": TRAFFIC,
        "slots": args.slots,
        "history": args.history,
        "horizon": 1,
        "split_seeds": split_seeds,
        "target": "next_slot_idle_probability",
        "xgboost_requested": bool(args.include_xgboost or args.require_xgboost),
        "xgboost_required": bool(args.require_xgboost),
        "torch_device": str(device),
        "xgboost_device": "cuda" if device.type == "cuda" else "cpu",
    }
    write_outputs(rows, Path(args.output_dir), metadata)
    print(pd.DataFrame(rows)[["model", "accuracy", "macro_f1", "brier", "nll", "ece"]].to_string(index=False))


if __name__ == "__main__":
    main()
