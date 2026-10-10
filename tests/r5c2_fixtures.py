import json
from pathlib import Path
from cvn_rl.experiments.config import load_config

from cvn_rl.experiments.r5c2_protocol import ROOT, resolve_protocol, write_resolution, write_json, file_hash


def small_protocol(tmp_path):
    entry = json.loads((ROOT / 'configs/r2_r05c02_dqi_sensitivity.json').read_text())
    entry.update(execution_scope='smoke', output_root=str(tmp_path / 'bundle'), rl_seeds=[1])
    entry['label_data'] = dict(slots=320, split_seeds={'train': [1001], 'val': [2001], 'test': [3001]})
    entry['predictor_recipe'].update(source_meta=None, hidden_size=8, num_layers=1,
                                     attention_size=8, dropout=0., epochs=1,
                                     batch_size=16, patience=2, learning_rate=.001, seed=42,
                                     feature_names=['quality_mbps', 'snr_db', 'rss_dbm', 'cci', 'bandwidth_mhz'])
    entry['rl_budget'] = dict(train_slots=640, checkpoint_interval=640, final_eval_episodes=1, final_scored_slots=80)
    entry['pilot'].update(predictor_slots=320, predictor_epochs=1, predictor_train_seed_count=1,
                           rl_train_slots=640, eval_episodes=1, scored_slots=80, age_slots=2000)
    # Exercise the frozen guidance loader without relying on unpublished weights.
    import torch
    from cvn_rl.predictors.attention_lstm import AttentionLSTMIdlePredictor
    model_path = tmp_path / 'occupancy.pt'
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        occupancy_model = AttentionLSTMIdlePredictor(1, 8, 1, 8, 0., 1)
    torch.save(occupancy_model.state_dict(), model_path)
    model_sha = file_hash(model_path)
    write_json(model_path.with_suffix('.meta.json'), dict(
        input_source='bs_true_idle_history', model_sha256=model_sha,
        input_size=1, hidden_size=8, num_layers=1, attention_size=8,
        dropout=0., output_size=1, history_window=50, metrics={}))
    guidance = load_config(ROOT / entry['guidance_source'])
    guidance.pop('_config_path', None)
    for condition in guidance['module_conditions']:
        descriptors = condition['descriptors']
        if 'frozen_model_path' in descriptors:
            descriptors.update(frozen_model_path=str(model_path), expected_model_sha256=model_sha)
    guidance_path = tmp_path / 'guidance.json'
    write_json(guidance_path, guidance)
    entry['guidance_source'] = str(guidance_path)
    path = tmp_path / 'entry.json'
    write_json(path, entry)
    protocol = resolve_protocol(path)
    write_resolution(protocol)
    return protocol, path


def dummy_model_index(protocol, manifest, stage='pilot', targets=('T00',)):
    import torch
    from cvn_rl.predictors.attention_lstm import AttentionLSTMDQIPredictor
    from cvn_rl.experiments.r5c2_protocol import bundle_root
    root = bundle_root(protocol)
    index = {}
    for tid in targets:
        path = root / ('models' if stage == 'formal' else 'pilot/models') / tid / 'attention_lstm_dqi.pt'
        path.parent.mkdir(parents=True, exist_ok=True)
        model = AttentionLSTMDQIPredictor(input_size=50, hidden_size=8, attention_size=8, output_size=10, dropout=0)
        for parameter in model.parameters():
            torch.nn.init.zeros_(parameter)
        torch.save(model.state_dict(), path)
        target = manifest['targets'][tid]
        meta = dict(model='AttentionLSTMDQIPredictor', input_source='dqi_channel_history', protocol_version='r5c2_v1',
                    protocol_hash=protocol['protocol_hash'], target_parameters=target['parameters'], target_id=tid,
                    r_ref_mbps=target['r_ref_mbps'], physical_scaling=manifest['preprocessing']['physical_scaling'],
                    feature_mean=manifest['preprocessing']['mean'], feature_std=manifest['preprocessing']['std'],
                    preprocessing_hash=manifest['preprocessing']['hash'], feature_names=manifest['preprocessing']['feature_names'],
                    num_channels=10, num_features=5, input_size=50, output_size=10, history_window=200,
                    hidden_size=8, attention_size=8, num_layers=1, dropout=0, model_sha256=file_hash(path))
        write_json(path.with_suffix('.meta.json'), meta)
        index[tid] = dict(model_path=str(path), model_sha256=meta['model_sha256'], target_parameters=target['parameters'],
                          protocol_hash=protocol['protocol_hash'], r_ref_mbps=target['r_ref_mbps'], stage=stage,
                          meta_sha256=file_hash(path.with_suffix('.meta.json')))
    write_json(root / ('model_index.json' if stage == 'formal' else 'pilot/model_index.json'), index)
    return index
