import hashlib
import json
import numpy as np
import pytest
import torch

from cvn_rl.predictors.attention_lstm import AttentionLSTMIdlePredictor
from cvn_rl.predictors.frozen_predictor import FrozenPredictor
from scripts.train_bs_occupancy_predictor import dataset


def test_bs_dataset_uses_past_idle_and_current_idle_target():
    features, targets = dataset(123, 200, history=50)
    assert features.shape == (150, 50, 1)
    assert targets.shape == (150, 1)
    assert set(np.unique(features.numpy())).issubset({0.0, 1.0})


def test_frozen_predictor_rejects_hash_and_input_source_mismatch(tmp_path):
    path = tmp_path / 'model.pt'
    model = AttentionLSTMIdlePredictor(1, 64, 1, 64, 0.1, 1)
    torch.save(model.state_dict(), path)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    meta = {"input_source":"bs_true_idle_history","model_sha256":sha,"history_window":50,"metrics":{}}
    path.with_suffix('.meta.json').write_text(json.dumps(meta),encoding='utf-8')
    loaded = FrozenPredictor(path, expected_sha256=sha, expected_input_source='bs_true_idle_history')
    assert loaded.sha256 == sha
    with pytest.raises(ValueError, match='SHA256 mismatch'):
        FrozenPredictor(path, expected_sha256='0'*64, expected_input_source='bs_true_idle_history')
    with pytest.raises(ValueError, match='input source mismatch'):
        FrozenPredictor(path, expected_sha256=sha, expected_input_source='vt_sensing')
