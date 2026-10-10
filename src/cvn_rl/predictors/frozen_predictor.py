"""Frozen predictor wrapper: load trained model, run inference.

Usage:
    predictor = FrozenPredictor("models/attention_lstm_periodic.pt", device="cpu")
    theta = predictor.predict(history_window)  # (window, channels) -> (channels,)
"""

from __future__ import annotations

from pathlib import Path
import hashlib

import numpy as np
import torch

from cvn_rl.predictors.attention_lstm import AttentionLSTMIdlePredictor, AttentionLSTMDQIPredictor


class FrozenPredictor:
    def __init__(self, model_path: str | Path, device: str = "cpu", *, expected_sha256: str | None = None, expected_input_source: str | None = None) -> None:
        self.device = torch.device(device)
        self.model_path = Path(model_path)

        # Load metadata to get architecture params
        meta_path = self.model_path.with_suffix(".meta.json")
        if meta_path.exists():
            import json

            with open(meta_path) as f:
                self.meta = json.load(f)
        else:
            raise FileNotFoundError(f"missing frozen predictor metadata: {meta_path}")

        self.sha256 = hashlib.sha256(self.model_path.read_bytes()).hexdigest()
        if expected_sha256 and self.sha256.lower() != expected_sha256.lower():
            raise ValueError(f"frozen predictor SHA256 mismatch: expected {expected_sha256}, got {self.sha256}")
        meta_sha = str(self.meta.get("model_sha256", ""))
        if meta_sha and meta_sha.lower() != self.sha256.lower():
            raise ValueError("frozen predictor metadata SHA256 does not match weights")
        input_source = str(self.meta.get("input_source", ""))
        if expected_input_source and input_source != expected_input_source:
            raise ValueError(f"frozen predictor input source mismatch: expected {expected_input_source}, got {input_source or '<missing>'}")

        input_size = int(self.meta.get("input_size", 1))
        self.input_size = input_size
        hidden_size = int(self.meta.get("hidden_size", 64))
        num_layers = int(self.meta.get("num_layers", 1))
        attention_size = int(self.meta.get("attention_size", hidden_size))
        dropout = float(self.meta.get("dropout", 0.1))
        output_size = int(self.meta.get("output_size", input_size))
        self.history_window = int(self.meta.get("history_window", 50))
        self.parameter_count = int(self.meta.get("metrics", {}).get("parameter_count", 0))
        self.inference_latency_ms = float(
            self.meta.get("metrics", {}).get("inference_latency_ms", 0.0)
        )

        self.model = AttentionLSTMIdlePredictor(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            attention_size=attention_size,
            dropout=dropout,
            output_size=output_size,
        ).to(self.device)

        state_dict = torch.load(self.model_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def predict(self, history: np.ndarray) -> np.ndarray:
        """Predict idle probability from history window.

        Args:
            history: (history_window, channels) 鈥?idle probabilities (0=busy, 1=idle)

        Returns:
            (channels,) 鈥?predicted idle probability for current slot
        """
        if history.ndim == 2 and self.input_size == 1:
            x = torch.as_tensor(
                history.T[:, :, None], dtype=torch.float32, device=self.device
            )  # (channels, window, 1), shared univariate model
        elif history.ndim == 2:
            x = torch.as_tensor(
                history, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
        else:
            x = torch.as_tensor(
                history, dtype=torch.float32, device=self.device
            ).unsqueeze(0).unsqueeze(-1)  # (1, window, 1)
        with torch.no_grad():
            p = self.model(x)
        return p.detach().cpu().numpy().reshape(-1)
class FrozenDQIPredictor:
    """Load a frozen Attention-LSTM DQI predictor and run causal inference."""

    def __init__(self, model_path: str | Path, device: str = "cpu", *, expected_sha256: str | None = None, expected_input_source: str | None = None) -> None:
        self.device = torch.device(device)
        self.model_path = Path(model_path)
        meta_path = self.model_path.with_suffix(".meta.json")
        if not meta_path.exists():
            raise FileNotFoundError(f"missing frozen DQI predictor metadata: {meta_path}")
        import json

        with open(meta_path, encoding="utf-8") as f:
            self.meta = json.load(f)
        self.sha256 = hashlib.sha256(self.model_path.read_bytes()).hexdigest()
        if expected_sha256 and self.sha256.lower() != expected_sha256.lower():
            raise ValueError(f"frozen DQI predictor SHA256 mismatch: expected {expected_sha256}, got {self.sha256}")
        meta_sha = str(self.meta.get("model_sha256", ""))
        if meta_sha and meta_sha.lower() != self.sha256.lower():
            raise ValueError("frozen DQI predictor metadata SHA256 does not match weights")
        input_source = str(self.meta.get("input_source", ""))
        if expected_input_source and input_source != expected_input_source:
            raise ValueError(f"frozen DQI predictor input source mismatch: expected {expected_input_source}, got {input_source or '<missing>'}")

        self.feature_names = list(self.meta.get("feature_names", []))
        self.num_channels = int(self.meta.get("num_channels", self.meta.get("output_size", 10)))
        self.num_features = int(self.meta.get("num_features", max(len(self.feature_names), 1)))
        self.input_size = int(self.meta.get("input_size", self.num_channels * self.num_features))
        self.history_window = int(self.meta.get("history_window", self.meta.get("history_slots", 200)))
        hidden_size = int(self.meta.get("hidden_size", 128))
        num_layers = int(self.meta.get("num_layers", 1))
        attention_size = int(self.meta.get("attention_size", hidden_size))
        dropout = float(self.meta.get("dropout", 0.1))
        output_size = int(self.meta.get("output_size", self.num_channels))
        self.feature_mean = np.asarray(self.meta.get("feature_mean", [0.0] * self.input_size), dtype=np.float32)
        self.feature_std = np.asarray(self.meta.get("feature_std", [1.0] * self.input_size), dtype=np.float32)
        if self.feature_mean.shape != (self.input_size,) or self.feature_std.shape != (self.input_size,):
            raise ValueError("frozen DQI predictor normalization vectors have invalid shape")
        self.feature_std = np.maximum(self.feature_std, 1e-6)
        self.parameter_count = int(self.meta.get("metrics", {}).get("parameter_count", 0))
        self.inference_latency_ms = float(self.meta.get("metrics", {}).get("inference_latency_ms", 0.0))
        self.validation_mse = float(self.meta.get("metrics", {}).get("test_mse", self.meta.get("metrics", {}).get("val_mse", np.nan)))

        self.model = AttentionLSTMDQIPredictor(
            input_size=self.input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            attention_size=attention_size,
            dropout=dropout,
            output_size=output_size,
        ).to(self.device)
        state_dict = torch.load(self.model_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def validate_target(self, parameters: dict, protocol_hash: str, r_ref_mbps: float) -> None:
        """Opt-in compatibility checks for sensitivity targets."""
        if self.meta.get('protocol_version') != 'r5c2_v1':
            raise ValueError('R5-C2 run requires a prepared R5-C2 model')
        if self.meta.get('target_parameters') != parameters or self.meta.get('protocol_hash') != protocol_hash:
            raise ValueError('R5-C2 predictor target/protocol mismatch')
        if not np.isclose(float(self.meta.get('r_ref_mbps', -1)), r_ref_mbps, rtol=1e-12):
            raise ValueError('R5-C2 model reference mismatch')
        if self.history_window != 200 or self.num_channels != 10 or not self.meta.get('physical_scaling'):
            raise ValueError('R5-C2 model schema/preprocessing mismatch')

    def predict(self, history_features: np.ndarray) -> np.ndarray:
        """Predict DQI from raw ``[history, channels, features]`` or flattened history."""
        from cvn_rl.predictors.dqi_features import flatten_dqi_history

        if history_features.ndim == 3:
            flat = flatten_dqi_history(history_features)
        elif history_features.ndim == 2:
            flat = history_features.astype(np.float32)
        else:
            raise ValueError(f"expected [history, channels, features] or [history, input_size], got {history_features.shape}")
        if flat.shape != (self.history_window, self.input_size):
            raise ValueError(f"expected history shape {(self.history_window, self.input_size)}, got {flat.shape}")
        norm = (flat - self.feature_mean[None, :]) / self.feature_std[None, :]
        x = torch.as_tensor(norm[None, :, :], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            y = self.model(x)
        return np.clip(y.detach().cpu().numpy().reshape(-1), 0.0, 1.0)
