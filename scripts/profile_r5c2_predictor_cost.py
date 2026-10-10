"""Separate fixed startup/validation cost from small-data training cost."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from cvn_rl.experiments.r5c2_protocol import resolve_protocol, bundle_root, read_json, write_json, fingerprint
from cvn_rl.experiments.r5c2_execution import _execute


def record_runtime_profile(root, protocol, report):
    runtime_path = root / 'pilot/runtime_summary.json'
    if runtime_path.exists():
        runtime = read_json(runtime_path)
        if runtime['protocol_hash'] != protocol['protocol_hash']:
            raise ValueError('runtime profile protocol mismatch')
        runtime.setdefault('predictor_11_naive_scaled_estimate_seconds', runtime['predictor_11_scaled_estimate_seconds'])
        runtime['predictor_11_scaled_estimate_seconds'] = report['formal_11_predictors_estimate_seconds']
        runtime['predictor_fixed_cost_profile'] = report
        runtime['predictor_estimate_method'] = 'subtract fixed startup/validation overhead before scaling epoch and sample counts'
        write_json(runtime_path, runtime)
    return report


def profile(protocol, target='T06', epochs=10):
    root = bundle_root(protocol)
    recipe = protocol['predictor_recipe']
    pilot = protocol['entry']['pilot']
    out = root / 'pilot/timing'
    signature = fingerprint(dict(protocol_hash=protocol['protocol_hash'], target=target, epochs=epochs))
    marker = out / 'predictor_cost.json'
    if marker.exists():
        previous = read_json(marker)
        if previous['signature'] != signature:
            raise ValueError('timing profile inputs changed')
        return record_runtime_profile(root, protocol, previous)
    timings = {}
    for count in [0, epochs]:
        model_dir = out / f'{target}_epochs_{count}'
        arguments = ['scripts/train_dqi_attention_predictor.py', '--config', str(root / f'training_configs/pilot_{target}.json'),
                     '--output-dir', str(model_dir), '--prepared-data', str(root / 'data'), '--target-id', target,
                     '--protocol-file', str(root / 'resolved_protocol.json'), '--slots', str(pilot['predictor_slots']),
                     '--history-slots','200','--epochs',str(count),
                     '--prepared-train-seed-count',str(pilot['predictor_train_seed_count'])]
        for key in ['hidden_size','num_layers','attention_size','dropout','batch_size','learning_rate',
                    'rank_loss_weight','patience','weight_decay','sample_stride','seed']:
            arguments += ['--'+key.replace('_','-'),str(recipe[key])]
        timings[str(count)] = _execute(arguments, model_dir / 'timing.log')
    delta = timings[str(epochs)]-timings['0']
    if delta <= 0:
        raise ValueError('timing difference not positive; no reliable extrapolation')
    completed_epochs = len((out / f'{target}_epochs_{epochs}/d3_attention_dqi_training_log.csv').read_text().splitlines())-1
    full_rows = len(protocol['entry']['label_data']['split_seeds']['train']) * (protocol['entry']['label_data']['slots']-200)
    pilot_rows = min(pilot['predictor_train_seed_count'],len(protocol['entry']['label_data']['split_seeds']['train'])) * (pilot['predictor_slots']-200)
    full_estimate = timings['0'] + delta / completed_epochs * recipe['epochs'] * full_rows / pilot_rows
    report = dict(signature=signature, protocol_hash=protocol['protocol_hash'], target=target, timings_seconds=timings,
                  completed_training_epochs=completed_epochs, fixed_cost_seconds=timings['0'],
                  added_training_seconds=delta, formal_one_predictor_estimate_seconds=full_estimate,
                  formal_11_predictors_estimate_seconds=11*full_estimate,
                  note='Conservative volume scaling of training plus validation; no guarantee, early stopping may reduce it. Timing models are not scientific checkpoints.')
    write_json(marker, report)
    return record_runtime_profile(root, protocol, report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/r2_r05c02_dqi_sensitivity.json')
    parser.add_argument('--target', default='T06')
    parser.add_argument('--epochs', type=int, default=10)
    args = parser.parse_args()
    result = profile(resolve_protocol(args.config),args.target,args.epochs)
    print(result)
