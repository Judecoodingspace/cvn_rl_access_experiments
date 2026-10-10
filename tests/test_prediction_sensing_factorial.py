import copy
import hashlib
import json
import numpy as np
import pandas as pd
import torch

from cvn_rl.predictors.attention_lstm import AttentionLSTMIdlePredictor
from cvn_rl.predictors.frozen_predictor import FrozenPredictor
from cvn_rl.simulation.multi_env import MultiVehicleAccessEnv
from cvn_rl.simulation.multi_traces import generate_multi_vehicle_trace
from cvn_rl.experiments.e2_r1 import _evaluate_prediction_sensing_redundancy


def _config():
    return {
        'slots': 100,
        'num_channels': 1,
        'num_vehicles': 1,
        'protection_variant': 'gate_only',
        'state_features': ['delta', 'theta', 'sensing_available', 'prediction_available'],
        'traffic': {'family': 'fixed_markov', 'p01': .2, 'p10': .2, 'sensing_pd': .9, 'sensing_pfa': .1},
        'packet_traffic': {'arrival_probability': 1.0},
    }


def test_sensing_neutral_changes_observation_not_physical_trace():
    cfg = _config(); trace = generate_multi_vehicle_trace(cfg, seed=7)
    original = trace.sensing_idle.copy()
    real = MultiVehicleAccessEnv(trace, {**cfg, 'sensing_info_source':'real', 'prediction_info_source':'neutral'})
    neutral = MultiVehicleAccessEnv(trace, {**cfg, 'sensing_info_source':'neutral', 'prediction_info_source':'neutral'})
    real_obs, _ = real.reset(); neutral_obs, _ = neutral.reset()
    assert np.array_equal(trace.sensing_idle, original)
    assert neutral_obs[0, 0] == .5
    assert real_obs.shape == neutral_obs.shape
    assert neutral_obs[0, -2:].tolist() == [0.0, 0.0]


def test_four_factor_conditions_share_gate_only_mask_and_dimension():
    cfg = _config(); trace = generate_multi_vehicle_trace(cfg, seed=8)
    masks=[]; dimensions=[]
    for sensing, prediction in [('real','real'),('real','neutral'),('neutral','real'),('neutral','neutral')]:
        env=MultiVehicleAccessEnv(trace,{**cfg,'sensing_info_source':sensing,'prediction_info_source':prediction,'sensing_mask_source':'neutral','prediction_mask_source':'neutral'})
        obs, infos=env.reset(); masks.append(infos[0]['raw_action_mask']); dimensions.append(obs.shape[1])
    assert len(set(dimensions)) == 1
    assert all(np.array_equal(masks[0], mask) for mask in masks[1:])


def test_frozen_shared_univariate_model_supports_three_channels(tmp_path):
    path = tmp_path / 'shared_occupancy.pt'
    model = AttentionLSTMIdlePredictor(1, 64, 1, 64, 0.1, 1)
    torch.save(model.state_dict(), path)
    path.with_suffix('.meta.json').write_text(json.dumps({
        'input_source': 'bs_true_idle_history', 'history_window': 50,
        'model_sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'metrics': {},
    }), encoding='utf-8')
    predictor=FrozenPredictor(path,expected_input_source='bs_true_idle_history')
    prediction=predictor.predict(np.ones((predictor.history_window,3),dtype=np.float32))
    assert prediction.shape == (3,)


def test_fixed_protection_uses_identical_sensing_mask_across_factors():
    cfg={**_config(),'protection_variant':'gate_state_mask','action_mask_thresholds':{'sensing_idle_min':.5,'overlay_theta_min':None,'underlay_power_cap_min':0.0,'underlay_theta_min':None}}
    trace=generate_multi_vehicle_trace(cfg,seed=9); masks=[]
    for sensing,prediction in [('real','real'),('real','neutral'),('neutral','real'),('neutral','neutral')]:
        env=MultiVehicleAccessEnv(trace,{**cfg,'sensing_info_source':sensing,'prediction_info_source':prediction,'sensing_mask_source':'real','prediction_mask_source':'neutral'})
        _,infos=env.reset(); masks.append(infos[0]['raw_action_mask'])
    assert all(np.array_equal(masks[0],mask) for mask in masks[1:])


def test_synthetic_redundancy_gate_passes_expected_conditional_effects():
    rows=[]
    values={'neither':(10,.40,.10),'prediction_only':(12,.30,.08),'sensing_only':(12,.30,.08),'full':(13,.25,.06)}
    sources={'neither':('neutral','neutral'),'prediction_only':('neutral','real'),'sensing_only':('real','neutral'),'full':('real','real')}
    for seed in (1,2,3):
        for variant,(goodput,plr,violation) in values.items():
            sensing,prediction=sources[variant]
            rows.append({'scenario':'s','arrival_probability':1.0,'agent':'ddqn_no_esn','seed':seed,'module_variant':variant,'served_goodput_mbps':goodput,'plr':plr,'pu_access_violation_rate':violation,'raw_action_mask_hash':f'm{seed}','train_physical_trace_hash':f't{seed}','evaluation_physical_trace_hash':f'e{seed}','sensing_info_source':sensing,'prediction_info_source':prediction,'observation_dimension':10,'evaluation_parameters_frozen':True,'diagnostic_run':False,'run_scope':'formal'})
    payload,_=_evaluate_prediction_sensing_redundancy(pd.DataFrame(rows),{'gate_conditions':{}})
    assert payload['passed']
