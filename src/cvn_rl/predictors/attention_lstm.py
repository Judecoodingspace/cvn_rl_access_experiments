"""Attention-LSTM idle predictor architecture.

Exact copy from cvn_prediction_model_experiments for frozen-model inference.
"""

from __future__ import annotations

import torch
from torch import nn


class AttentionLSTMIdlePredictor(nn.Module):
    """Predict per-channel idle probability from an occupancy-history window."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 1,
        attention_size: int = 64,
        dropout: float = 0.1,
        output_size: int = 1,
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
        self.attn_hidden = nn.Linear(hidden_size, attention_size)
        self.attn_score = nn.Linear(attention_size, 1, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return idle probabilities.

        Args:
            x: Tensor with shape [batch, history_window, input_size].
        """
        hidden_seq, _ = self.lstm(x)
        energy = self.attn_score(torch.tanh(self.attn_hidden(hidden_seq)))
        weights = torch.softmax(energy, dim=1)
        context = torch.sum(weights * hidden_seq, dim=1)
        logits = self.output(self.dropout(context))
        return torch.sigmoid(logits)
class AttentionLSTMDQIPredictor(nn.Module):
    """Predict a per-channel DQI vector from causal channel-history features."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 1,
        attention_size: int = 128,
        dropout: float = 0.1,
        output_size: int = 10,
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
        self.attn_hidden = nn.Linear(hidden_size, attention_size)
        self.attn_score = nn.Linear(attention_size, 1, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return predicted DQI values clipped by a sigmoid to [0, 1].

        Args:
            x: Tensor with shape [batch, history_window, input_size].
        """
        hidden_seq, _ = self.lstm(x)
        energy = self.attn_score(torch.tanh(self.attn_hidden(hidden_seq)))
        weights = torch.softmax(energy, dim=1)
        context = torch.sum(weights * hidden_seq, dim=1)
        logits = self.output(self.dropout(context))
        return torch.sigmoid(logits)
