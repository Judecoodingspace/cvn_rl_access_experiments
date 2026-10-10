import copy
import numpy as np
import pytest
import pandas as pd

from r5c2_fixtures import small_protocol, dummy_model_index
from cvn_rl.experiments.r5c2_protocol import bundle_root
from cvn_rl.experiments.r5c2_data import prepare_data, ensure_rl_caches
from cvn_rl.experiments.r5c2_execution import pilot_freshness, case_config
from cvn_rl.experiments.multi_vehicle_runner import _resolve_trace, run_multi_vehicle_experiment
from cvn_rl.simulation.multi_env import MultiVehicleAccessEnv
from cvn_rl.simulation.state_builder import feature_slices
from cvn_rl.experiments.r5c2_analysis import paired_interval


def test_frozen_guidance_reference_and_physical_pairing(tmp_path):
    protocol, _ = small_protocol(tmp_path)
    manifest = prepare_data(protocol)
    dummy_model_index(protocol, manifest, targets=('T00', 'T06'))
    choice = pilot_freshness(protocol)
    assert choice['selected_update_slots'] in [50, 100, 500, 1000, 2000]
    traces, configs = {}, {}
    for cid in ['C00', 'C06', 'C11', 'B00']:
        cfg = case_config(protocol, cid, 1, stage='pilot')
        ensure_rl_caches(cfg)
        trace = _resolve_trace(cfg, cfg['scenario_name'] if 'scenario_name' in cfg else 'M10_N10_competitive',
                               seed=1, slots=cfg['slots'], split='rl_train')
        traces[cid], configs[cid] = trace, cfg
    assert len({t.physical_trace_hash() for t in traces.values()}) == 1
    assert np.array_equal(traces['C00'].theta, traces['B00'].theta)
    period = choice['selected_update_slots']
    t = period + 10
    if t < traces['C00'].slots:
        assert np.all(traces['C11'].fresh_dqi[t] <= traces['C00'].fresh_dqi[t])
    observations = {}
    for cid in traces:
        env = MultiVehicleAccessEnv(traces[cid], configs[cid])
        env.reset()
        env.t = min(510, traces[cid].slots-1)
        observations[cid] = env._observations()
        slices = feature_slices(10, env.state_features)
        if cid == 'B00':
            assert np.all(observations[cid][:, slices['dqi_age']] == 0)
            assert np.all(observations[cid][:, slices['fresh_dqi']] == .5)
    for name in ['current_cqi', 'theta']:
        assert np.array_equal(observations['C00'][:, slices[name]], observations['C11'][:, slices[name]])
        assert np.array_equal(observations['C00'][:, slices[name]], observations['C06'][:, slices[name]])
        assert np.array_equal(observations['C00'][:, slices[name]], observations['B00'][:, slices[name]])
    bad = copy.deepcopy(configs['C00'])
    bad['descriptors']['r5c2_target_parameters']['time_quantile'] = .05
    with pytest.raises(ValueError, match='target/protocol'):
        _resolve_trace(bad, 'M10_N10_competitive', seed=1, slots=bad['slots'], split='rl_train')


def test_evaluation_warmup_not_counted_and_no_learning(tmp_path):
    protocol, _ = small_protocol(tmp_path)
    manifest = prepare_data(protocol)
    dummy_model_index(protocol, manifest)
    pilot_freshness(protocol)
    cfg = case_config(protocol, 'C00', 1, 'pilot')
    cfg['agents'] = ['fixed_overlay']
    ensure_rl_caches(cfg)
    metrics, _ = run_multi_vehicle_experiment(cfg)
    final = metrics.iloc[0]
    assert final['evaluation_slots'] == 80
    assert final['scored_slot_records'] == 80
    assert final['total_channel_slots'] == 800
    assert final['evaluation_parameters_frozen']


def test_seed_interval_and_interaction_algebra():
    values = np.array([1., 2., 3., 4., 5.])
    interval = paired_interval(values)
    assert interval['mean'] == 3 and interval['paired_seeds'] == 5
    assert interval == paired_interval(values)
    frame = pd.DataFrame(dict(C00=values, C03=values+2, C06=values+3, C09=values+6))
    interaction = frame.C09-frame.C03-frame.C06+frame.C00
    assert np.all(interaction == 1)


def test_process_probe_does_not_interrupt_current_process():
    import os
    from cvn_rl.experiments.r5c2_execution import process_is_running
    assert process_is_running(os.getpid())


def test_analysis_keeps_condition_identity_and_rejects_missing_tasks(tmp_path):
    from cvn_rl.experiments.r5c2_protocol import write_json, file_hash, fingerprint
    from cvn_rl.experiments.r5c2_analysis import summarize, plot
    protocol, _ = small_protocol(tmp_path)
    root = bundle_root(protocol)
    write_json(root / 'model_index.json', {})
    tasks, index = [], {}
    for number, cid in enumerate([c['condition_id'] for c in protocol['conditions']] + ['B00']):
        case_path = root / 'cases' / f'{cid}.json'
        cfg = dict(r5c2_eval_warmup_slots=10, final_checkpoint_evaluation=dict(eval_slots=90, eval_episodes=1))
        write_json(case_path, cfg)
        digest = fingerprint(cfg)
        tasks.append(dict(condition_id=cid, seed=1, path=str(case_path), config_hash=digest))
        run = root / 'test_runs' / cid
        (run / 'metrics').mkdir(parents=True)
        row = dict(condition_id=cid, seed=1, is_final_checkpoint=True, evaluation_parameters_frozen=True,
                   r5c2_execution_stage='formal', r5c2_protocol_hash=protocol['protocol_hash'],
                   served_goodput_mbps=float(number+1), plr=.1, pu_access_violation_rate=.01,
                   pu_interference_violation_rate=.001, scored_physical_hash='same', theta_observation_hash='same',
                   evaluation_slots=80, scored_slot_records=80)
        pd.DataFrame([row]).to_csv(run / 'metrics/access_metrics.csv', index=False)
        index[f'{cid}_s1'] = dict(status='completed', config_hash=digest, run_dir=str(run),
                                 metrics_hash=file_hash(run / 'metrics/access_metrics.csv'))
    write_json(root / 'runs_frozen.json', dict(protocol_hash=protocol['protocol_hash'], tasks=tasks,
                                              model_index_hash=file_hash(root / 'model_index.json')))
    write_json(root / 'run_index.json', index)
    write_json(root / 'pilot/freshness_selection.json', dict(selected_update_slots=500))
    rows = summarize(protocol)
    assert len(rows) == 14
    assert rows[0]['delta_goodput'] == -14
    plot(protocol)
    assert (root / 'figures/dqi_sensitivity.pdf').is_file()
    del index['C13_s1']
    write_json(root / 'run_index.json', index)
    with pytest.raises(ValueError, match='missing or incompatible'):
        summarize(protocol)
