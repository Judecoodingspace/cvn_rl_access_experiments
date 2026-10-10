from __future__ import annotations

import hashlib
import json

import numpy as np
import torch

from cvn_rl.experiments.config import load_config
from cvn_rl.experiments.multi_vehicle_runner import _apply_descriptor_perturbations
from cvn_rl.predictors.attention_lstm import AttentionLSTMDQIPredictor
from cvn_rl.predictors.dqi_features import DEFAULT_DQI_FEATURES, build_dqi_feature_tensor, flatten_dqi_history
from cvn_rl.predictors.frozen_predictor import FrozenDQIPredictor
from cvn_rl.simulation.multi_traces import generate_multi_vehicle_trace


def _write_dummy_dqi_model(path, *, channels=10, features=len(DEFAULT_DQI_FEATURES), history=20):
    model = AttentionLSTMDQIPredictor(
        input_size=channels * features,
        hidden_size=8,
        attention_size=8,
        output_size=channels,
        dropout=0.0,
    )
    for param in model.parameters():
        torch.nn.init.constant_(param, 0.0)
    torch.save(model.state_dict(), path)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    meta = {
        "model": "AttentionLSTMDQIPredictor",
        "input_source": "dqi_channel_history",
        "input_size": channels * features,
        "output_size": channels,
        "num_channels": channels,
        "num_features": features,
        "feature_names": list(DEFAULT_DQI_FEATURES),
        "feature_mean": [0.0] * (channels * features),
        "feature_std": [1.0] * (channels * features),
        "hidden_size": 8,
        "num_layers": 1,
        "attention_size": 8,
        "dropout": 0.0,
        "history_window": history,
        "model_sha256": sha,
        "metrics": {"test_mse": 0.123, "parameter_count": sum(p.numel() for p in model.parameters())},
    }
    path.with_suffix(".meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return sha


def test_dqi_feature_tensor_is_causal_shape():
    cfg = load_config("configs/e2_d3_attention_dqi_frozen.json")
    cfg = {**cfg, "slots": 64, "hidden_lookahead_slots": 100}
    trace = generate_multi_vehicle_trace(cfg, seed=123, slots=64, split="test_d3_features")
    features = build_dqi_feature_tensor(trace, DEFAULT_DQI_FEATURES)
    assert features.shape == (trace.slots, trace.num_channels, len(DEFAULT_DQI_FEATURES))
    flat = flatten_dqi_history(features[:20])
    assert flat.shape == (20, trace.num_channels * len(DEFAULT_DQI_FEATURES))
    assert not np.shares_memory(features, trace.future_dqi_oracle)


def test_frozen_dqi_predictor_hash_and_runner_cutoff(tmp_path):
    model_path = tmp_path / "attention_lstm_dqi.pt"
    sha = _write_dummy_dqi_model(model_path, history=20)
    predictor = FrozenDQIPredictor(model_path, expected_sha256=sha, expected_input_source="dqi_channel_history")
    y = predictor.predict(np.zeros((20, 10, len(DEFAULT_DQI_FEATURES)), dtype=np.float32))
    assert y.shape == (10,)
    assert np.all((0.0 <= y) & (y <= 1.0))

    cfg = load_config("configs/e2_d3_attention_dqi_frozen.json")
    cfg = {**cfg, "slots": 80, "hidden_lookahead_slots": 100}
    base_trace = generate_multi_vehicle_trace(cfg, seed=456, slots=80, split="test_d3_runner")
    trace = _apply_descriptor_perturbations(
        base_trace,
        {
            "dqi_condition": "attention_frozen",
            "dqi_frozen_model_path": str(model_path),
            "expected_dqi_model_sha256": sha,
            "dqi_prediction_input_source": "dqi_channel_history",
            "dqi_update_slots": 10,
            "dqi_neutral_value": 0.5,
        },
        seed=456,
    )
    assert trace.dqi_model_hash == sha
    assert trace.fresh_dqi.shape == (80, 10)
    valid = trace.dqi_input_cutoff_slot >= 0
    assert np.all(trace.dqi_input_cutoff_slot[valid] <= np.arange(trace.slots)[valid])
    assert np.allclose(trace.fresh_dqi[:20], 0.5)
