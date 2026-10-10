from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd

from cvn_rl.experiments.artifacts import (
    create_run_dir,
    freeze_config,
    write_manifest,
    write_notes,
)
from cvn_rl.experiments.config import load_config
from cvn_rl.experiments.e2_pilot import (
    E2_SCENARIOS,
    evaluate_e2_pilot_signal,
    module_seed_values_table,
    render_e2_report,
    validate_e2_pilot_metrics,
)
from cvn_rl.experiments.statistics import (
    module_ablation_bootstrap_table,
    module_interaction_bootstrap_table,
    select_training_horizon,
)
from cvn_rl.multi_metrics import summarize_checkpoint_table
from cvn_rl.reporting.e2_pilot_figures import write_e2_pilot_figures


def main() -> int:
    parser = argparse.ArgumentParser(description='Merge four completed E2 pilot shards.')
    parser.add_argument('--run-dir', action='append', required=True)
    parser.add_argument('--results-root', default='results/e2_pilot_merged')
    args = parser.parse_args()

    run_dirs = [Path(value) for value in args.run_dir]
    if len(run_dirs) != len(E2_SCENARIOS):
        parser.error(f'exactly {len(E2_SCENARIOS)} --run-dir values are required')
    frames: list[pd.DataFrame] = []
    source_scenarios: set[str] = set()
    for run_dir in run_dirs:
        manifest = pd.read_json(run_dir / 'manifest.json', typ='series')
        if bool(manifest.get('dry_run')) or bool(manifest.get('quick')):
            raise ValueError(f'diagnostic shard cannot be merged: {run_dir}')
        config = load_config(run_dir / 'config.frozen.json')
        scenarios = {str(item['name']) for item in config.get('scenarios', [])}
        if len(scenarios) != 1:
            raise ValueError(f'shard must contain exactly one scenario: {run_dir}')
        source_scenarios.update(scenarios)
        frames.append(pd.read_csv(run_dir / 'metrics' / 'access_metrics.csv'))
    if source_scenarios != set(E2_SCENARIOS):
        raise ValueError(
            f'shard scenarios mismatch: found {sorted(source_scenarios)}, '
            f'expected {list(E2_SCENARIOS)}'
        )

    metrics = pd.concat(frames, ignore_index=True)
    config = load_config('configs/e2_single_vt_module_ablation_pilot.json')
    config['source_run_dirs'] = [str(path.resolve()) for path in run_dirs]
    config['scenarios'] = [
        scenario
        for scenario in config['scenarios']
        if scenario['name'] in E2_SCENARIOS
    ]
    run_id = (
        datetime.now().strftime('%Y%m%d_%H%M%S')
        + '_e2_module_ablation_pilot_merged'
    )
    run_dir = create_run_dir(run_id, args.results_root)
    freeze_config(run_dir, config)

    outputs: list[str] = []
    metrics.to_csv(run_dir / 'metrics' / 'access_metrics.csv', index=False)
    outputs.append('metrics/access_metrics.csv')
    summary = summarize_checkpoint_table(metrics)
    summary.to_csv(run_dir / 'tables' / 'checkpoint_summary.csv', index=False)
    outputs.append('tables/checkpoint_summary.csv')
    final = metrics[metrics['is_final_checkpoint'].astype(bool)].copy()
    final.to_csv(run_dir / 'tables' / 'final_checkpoint_results.csv', index=False)
    outputs.append('tables/final_checkpoint_results.csv')

    integrity, integrity_passed = validate_e2_pilot_metrics(metrics)
    integrity.to_csv(run_dir / 'tables' / 'pilot_integrity_checks.csv', index=False)
    (run_dir / 'analysis' / 'pilot_integrity_report.md').write_text(
        render_e2_report('E2 Pilot Integrity', integrity, integrity_passed),
        encoding='utf-8',
    )
    outputs.extend(
        ['tables/pilot_integrity_checks.csv', 'analysis/pilot_integrity_report.md']
    )

    effects = module_ablation_bootstrap_table(metrics)
    effects.to_csv(
        run_dir / 'tables' / 'module_ablation_paired_bootstrap.csv',
        index=False,
    )
    interaction = module_interaction_bootstrap_table(metrics)
    interaction.to_csv(
        run_dir / 'tables' / 'module_interaction_bootstrap.csv',
        index=False,
    )
    seed_values = module_seed_values_table(metrics)
    seed_values.to_csv(run_dir / 'tables' / 'module_seed_values.csv', index=False)
    outputs.extend(
        [
            'tables/module_ablation_paired_bootstrap.csv',
            'tables/module_interaction_bootstrap.csv',
            'tables/module_seed_values.csv',
        ]
    )

    horizon_cfg = config['training_horizon_selection']
    horizon = select_training_horizon(
        metrics,
        agent=horizon_cfg['agent'],
        module_variant=horizon_cfg['module_variant'],
        min_slots=horizon_cfg['min_slots'],
        max_slots=horizon_cfg['max_slots'],
        relative_tolerance=horizon_cfg['relative_tolerance'],
        plr_absolute_tolerance=horizon_cfg['plr_absolute_tolerance'],
        consecutive_intervals=horizon_cfg['consecutive_intervals'],
    )
    horizon.to_csv(
        run_dir / 'tables' / 'training_horizon_recommendation.csv',
        index=False,
    )
    outputs.append('tables/training_horizon_recommendation.csv')

    signal, signal_passed = evaluate_e2_pilot_signal(metrics, config)
    signal.to_csv(run_dir / 'tables' / 'pilot_signal_gate.csv', index=False)
    (run_dir / 'analysis' / 'pilot_signal_report.md').write_text(
        render_e2_report('E2 Pilot Signal Gate', signal, signal_passed),
        encoding='utf-8',
    )
    outputs.extend(
        ['tables/pilot_signal_gate.csv', 'analysis/pilot_signal_report.md']
    )
    outputs.extend(write_e2_pilot_figures(run_dir, metrics))
    write_notes(run_dir, config, len(metrics))
    outputs.append('analysis/experiment_notes.md')
    write_manifest(run_dir, config, run_id, False, False, outputs)
    print(f'Merged E2 pilot: {run_dir}')
    print(f'Integrity: {"PASS" if integrity_passed else "FAIL"}')
    print(f'Signal gate: {"PASS" if signal_passed else "FAIL"}')
    return 0 if integrity_passed else 2


if __name__ == '__main__':
    raise SystemExit(main())
