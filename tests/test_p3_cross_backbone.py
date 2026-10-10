from pathlib import Path
import json

import pandas as pd
import pytest

from cvn_rl.experiments.config import load_config
from cvn_rl.experiments.multi_vehicle_runner import _build_cases
from scripts.analyze_p3_cross_backbone import CONDITIONS, LABELS, MODEL_SHA, read_run, support_level, validate_and_merge


def _resolved(name: str):
    return load_config(Path('configs') / name)


def test_cross_backbone_configs_have_one_agent_and_five_conditions():
    expected = {
        'e2_p3x_esn_ddqn_frozen_predictor.json': 'proposed_esn_ddqn',
        'e2_p3x_ppo_frozen_predictor.json': 'ppo_discrete',
        'e2_p3x_lstm_ddqn_frozen_predictor.json': 'lstm_ddqn',
        'e2_p3x_sac_discrete_frozen_predictor.json': 'sac_discrete',
    }
    for name, agent in expected.items():
        config = _resolved(name)
        assert config['agents'] == [agent]
        assert len(config['module_conditions']) == 5
        assert config['seeds'] == [1, 2, 3]
        cases = _build_cases(config, config['scenarios'])
        assert len(cases) == 5
        assert {case[1]['module_variant'] for case in cases} == {
            'theta_oracle', 'theta_central_markov', 'theta_central_frozen',
            'theta_central_frozen_placebo', 'theta_neutral',
        }


def test_cross_backbone_configs_pin_same_frozen_model():
    hashes = set()
    for name in (
        'e2_p3x_esn_ddqn_frozen_predictor.json',
        'e2_p3x_ppo_frozen_predictor.json',
        'e2_p3x_lstm_ddqn_frozen_predictor.json',
        'e2_p3x_sac_discrete_frozen_predictor.json',
    ):
        config = _resolved(name)
        for condition in config['module_conditions']:
            descriptor = condition.get('descriptors', {})
            if 'expected_model_sha256' in descriptor:
                hashes.add(descriptor['expected_model_sha256'])
                assert descriptor['prediction_update_slots'] == 1
    assert hashes == {'c0d645858adf54bdcd04b9c0956629a9b11968a60013ff7d21ec5d8381cb7732'}


def _write_run(root: Path, agent: str, trace_suffix: str = '') -> Path:
    (root / 'tables').mkdir(parents=True)
    (root / 'analysis').mkdir()
    rows = []
    for condition in sorted(CONDITIONS):
        for seed in (1, 2, 3):
            rows.append({'agent': agent, 'scenario': 'P0_M1_prediction_max', 'seed': seed,
                         'module_variant': condition, 'is_final_checkpoint': True,
                         'evaluation_parameters_frozen': True,
                         'evaluation_physical_trace_hash': f'trace-{seed}{trace_suffix}',
                         'prediction_model_hash': MODEL_SHA if 'frozen' in condition else 'surrogate',
                         'served_goodput_mbps': 2.0 if condition == 'theta_central_frozen' else 1.0,
                         'pu_access_violation_rate': 0.01})
    pd.DataFrame(rows).to_csv(root / 'tables/final_checkpoint_results.csv', index=False)
    (root / 'analysis/e2_r1_gate.json').write_text(json.dumps({'passed': True}), encoding='utf-8')
    (root / 'manifest.json').write_text('{}', encoding='utf-8')
    return root


def test_support_classification():
    assert support_level({'a': True, 'b': True, 'c': True}) == 'strong_support'
    assert support_level({'a': True, 'b': True, 'c': False}) == 'partial_support'
    assert support_level({'a': True, 'b': False, 'c': False}) == 'not_supported'


def test_merger_accepts_paired_synthetic_runs(tmp_path):
    runs = {agent: _write_run(tmp_path / agent, agent) for agent in LABELS}
    merged, gates, _ = validate_and_merge(runs)
    assert len(merged) == 15 * len(LABELS)
    assert set(gates) == set(LABELS.values())


@pytest.mark.parametrize('fault', ['agent', 'condition', 'duplicate_seed', 'sha'])
def test_single_run_rejects_contract_faults(tmp_path, fault):
    agent = 'ddqn_no_esn'
    run = _write_run(tmp_path / 'run', agent)
    csv = run / 'tables/final_checkpoint_results.csv'
    df = pd.read_csv(csv)
    if fault == 'agent':
        df.loc[0, 'agent'] = 'wrong'
    elif fault == 'condition':
        df = df[df.module_variant != 'theta_neutral']
    elif fault == 'duplicate_seed':
        df.loc[(df.module_variant == 'theta_oracle') & (df.seed == 3), 'seed'] = 2
    else:
        df.loc[df.module_variant == 'theta_central_frozen', 'prediction_model_hash'] = 'wrong'
    df.to_csv(csv, index=False)
    with pytest.raises(ValueError):
        read_run(run, agent)


def test_merger_rejects_cross_agent_trace_mismatch(tmp_path):
    runs = {agent: _write_run(tmp_path / agent, agent, '-bad' if agent == 'ppo_discrete' else '') for agent in LABELS}
    with pytest.raises(ValueError, match='physical trace mismatch'):
        validate_and_merge(runs)
