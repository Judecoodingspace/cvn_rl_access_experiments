from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from cvn_rl.experiments.artifacts import (
    create_run_dir,
    freeze_config,
    write_dataframes,
    write_json,
    write_manifest,
    write_notes,
)
from cvn_rl.experiments.config import load_config, make_run_id
from cvn_rl.experiments.e2_mvp import evaluate_e2_mvp_gate, render_e2_mvp_report
from cvn_rl.experiments.e2_r1 import (
    e2_r1_paired_seed_effects,
    evaluate_e2_r1_gate,
    render_e2_r1_report,
)
from cvn_rl.experiments.claim_gate import (
    evaluate_claim_gates,
    render_claim_gate_report,
)
from cvn_rl.experiments.registry import get_runner
from cvn_rl.experiments.pilot_gate import evaluate_pilot_gate, render_pilot_gate_report
from cvn_rl.experiments.statistics import (
    module_ablation_bootstrap_table,
    module_interaction_bootstrap_table,
    paired_bootstrap_table,
    select_training_horizon,
)
from cvn_rl.reporting.figures import write_expected_figures


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser(description="Run CVN RL access experiments.")
    parser.add_argument("--config", required=True, help="Path to a JSON experiment config.")
    parser.add_argument("--dry-run", action="store_true", help="Limit slots and sweep dimensions for smoke tests.")
    parser.add_argument("--quick", action="store_true", help="Use quick_slots instead of slots.")
    parser.add_argument("--results-root", default="results", help="Directory for run artifacts.")
    parser.add_argument(
        "--run-label",
        help="Readable revision label, e.g. r2_r01c03_prediction_robustness; "
        "adds dry/quick/formal stage to the run directory name.",
    )
    parser.add_argument(
        '--gate-report',
        help='Machine-readable pilot_gate.json required by gated formal runs.',
    )
    parser.add_argument(
        '--reuse-run-dir',
        action='append',
        default=[],
        help='Prior run directory whose metrics may satisfy preregistered reuse cases.',
    )
    args = parser.parse_args(argv)

    config = load_config(Path(args.config))
    config['_reuse_run_dirs'] = [
        str(Path(path).resolve()) for path in args.reuse_run_dir
    ]
    required_r1_gate = config.get('requires_e2_r1_gate')
    if required_r1_gate and not (args.dry_run or args.quick):
        if not args.gate_report:
            parser.error('--gate-report from R1-A is required for this configuration')
        gate_path = Path(args.gate_report)
        gate_payload = json.loads(gate_path.read_text(encoding='utf-8'))
        if str(gate_payload.get('gate', '')).lower() != str(required_r1_gate).lower():
            parser.error(
                f'expected E2-R1 {required_r1_gate!r} gate report: {gate_path}'
            )
        if not bool(gate_payload.get('passed', False)):
            parser.error(f'E2-R1 prerequisite gate did not pass: {gate_path}')
        config['_verified_gate_report'] = str(gate_path.resolve())
        config['_verified_gate_field'] = 'passed'
    if bool(config.get('requires_pilot_gate', False)) and not (args.dry_run or args.quick):
        if not args.gate_report:
            parser.error('--gate-report is required for this formal configuration')
        gate_path = Path(args.gate_report)
        gate_payload = json.loads(gate_path.read_text(encoding='utf-8'))
        required_gate_field = str(config.get('required_gate_field', 'passed'))
        if not bool(gate_payload.get(required_gate_field, False)):
            parser.error(
                f'pilot gate field {required_gate_field!r} did not pass: '
                f'{gate_path}'
            )
        config['_verified_gate_report'] = str(gate_path.resolve())
        config['_verified_gate_field'] = required_gate_field
    experiment = config.get("experiment", "rl_baseline_comparison")
    try:
        run_id = make_run_id(
            experiment,
            dry_run=args.dry_run,
            quick=args.quick,
            label=args.run_label,
        )
    except ValueError as exc:
        parser.error(str(exc))
    if config.get('r5c2_protocol_version'):
        stage = 'smoke' if config.get('r5c2_execution_scope') == 'smoke' else config.get('r5c2_stage', 'formal')
        if stage != 'formal':
            run_id = run_id.removesuffix('_formal') + '_' + stage
    run_dir = create_run_dir(run_id, args.results_root)
    freeze_config(run_dir, config)
    print(
        f"Run started: {run_dir} "
        f"config={Path(args.config)} "
        f"dry_run={args.dry_run} quick={args.quick}",
        flush=True,
    )

    runner = get_runner(experiment)
    reuse_requirement = config.get('reuse_gate_requirement', {})
    if reuse_requirement and not (args.dry_run or args.quick):
        if len(args.reuse_run_dir) != 1:
            parser.error(
                'exactly one --reuse-run-dir is required by this configuration'
            )
        reuse_gate_path = (
            Path(args.reuse_run_dir[0]) / 'analysis' / 'e2_r1_gate.json'
        )
        if not reuse_gate_path.exists():
            parser.error(f'missing reuse Gate report: {reuse_gate_path}')
        reuse_gate = json.loads(
            reuse_gate_path.read_text(encoding='utf-8')
        )
        expected_gate = str(reuse_requirement.get('gate', ''))
        expected_hash = str(
            reuse_requirement.get('resolved_config_hash', '')
        )
        if not bool(reuse_gate.get('passed', False)):
            parser.error(f'reuse Gate did not pass: {reuse_gate_path}')
        if str(reuse_gate.get('gate', '')) != expected_gate:
            parser.error(
                f'reuse Gate type mismatch: expected {expected_gate!r}'
            )
        if (
            str(reuse_gate.get('resolved_config_hash', ''))
            != expected_hash
        ):
            parser.error(
                'reuse resolved config hash does not match the frozen '
                f'requirement: {expected_hash}'
            )
    reuse_frames: list[pd.DataFrame] = []
    for reuse_dir in args.reuse_run_dir:
        metrics_path = Path(reuse_dir) / 'metrics' / 'access_metrics.csv'
        if not metrics_path.exists():
            parser.error(f'missing reuse metrics: {metrics_path}')
        frame = pd.read_csv(metrics_path)
        frame['_reuse_source_run'] = str(Path(reuse_dir).resolve())
        reuse_frames.append(frame)
    runtime_config = {
        **config,
        '_run_dir': str(run_dir),
        '_reuse_metrics': (
            pd.concat(reuse_frames, ignore_index=True)
            if reuse_frames else pd.DataFrame()
        ),
    }
    metrics, summary = runner(runtime_config, dry_run=args.dry_run, quick=args.quick)
    outputs = write_dataframes(run_dir, metrics, summary, config.get("expected_outputs"))
    if "checkpoint_slot" in metrics.columns:
        paired = paired_bootstrap_table(metrics)
        paired_path = run_dir / "tables" / "paired_bootstrap_comparisons.csv"
        paired.to_csv(paired_path, index=False)
        outputs.append("tables/paired_bootstrap_comparisons.csv")

        final_metrics = (
            metrics[metrics["is_final_checkpoint"].astype(bool)].copy()
            if "is_final_checkpoint" in metrics
            else metrics.copy()
        )
        final_metrics.to_csv(
            run_dir / 'tables' / 'final_checkpoint_results.csv',
            index=False,
        )
        outputs.append('tables/final_checkpoint_results.csv')
        if 'module_variant' in metrics.columns and metrics['module_variant'].nunique() > 1:
            module_paired = module_ablation_bootstrap_table(metrics)
            module_path = run_dir / 'tables' / 'module_ablation_paired_bootstrap.csv'
            module_paired.to_csv(module_path, index=False)
            outputs.append('tables/module_ablation_paired_bootstrap.csv')
            interaction = module_interaction_bootstrap_table(metrics)
            interaction_path = run_dir / 'tables' / 'module_interaction_bootstrap.csv'
            interaction.to_csv(interaction_path, index=False)
            outputs.append('tables/module_interaction_bootstrap.csv')
            if bool(config.get('claim_gate', {}).get('enabled', False)):
                claim_table, claims_passed = evaluate_claim_gates(
                    module_paired,
                    metrics,
                    config,
                )
                claim_table.to_csv(
                    run_dir / 'tables' / 'claim_gate_checks.csv',
                    index=False,
                )
                (run_dir / 'analysis' / 'claim_gate_report.md').write_text(
                    render_claim_gate_report(claim_table, claims_passed),
                    encoding='utf-8',
                )
                outputs.extend(
                    [
                        'tables/claim_gate_checks.csv',
                        'analysis/claim_gate_report.md',
                    ]
                )

        horizon_cfg = config.get('training_horizon_selection', {})
        if bool(horizon_cfg.get('enabled', False)):
            horizon = select_training_horizon(
                metrics,
                agent=str(horizon_cfg.get('agent', 'ddqn_no_esn')),
                module_variant=str(horizon_cfg.get('module_variant', 'full')),
                min_slots=int(horizon_cfg.get('min_slots', 10_000)),
                max_slots=int(horizon_cfg.get('max_slots', 30_000)),
                relative_tolerance=float(horizon_cfg.get('relative_tolerance', 0.02)),
                plr_absolute_tolerance=float(
                    horizon_cfg.get('plr_absolute_tolerance', 0.005)
                ),
                consecutive_intervals=int(
                    horizon_cfg.get('consecutive_intervals', 3)
                ),
            )
            horizon.to_csv(
                run_dir / 'tables' / 'training_horizon_recommendation.csv',
                index=False,
            )
            outputs.append('tables/training_horizon_recommendation.csv')

        runtime_columns = [
            column
            for column in [
                'training_time_seconds', 'inference_latency_ms', 'parameter_count',
                'descriptor_inference_latency_ms', 'descriptor_parameter_count',
                'broadcast_overhead_kbps',
            ]
            if column in final_metrics.columns
        ]
        if {'parameter_count', 'descriptor_parameter_count'}.issubset(final_metrics):
            final_metrics['total_parameter_count'] = (
                final_metrics['parameter_count']
                + final_metrics['descriptor_parameter_count']
            )
            runtime_columns.append('total_parameter_count')
        if {'inference_latency_ms', 'descriptor_inference_latency_ms'}.issubset(
            final_metrics
        ):
            final_metrics['total_inference_latency_ms'] = (
                final_metrics['inference_latency_ms']
                + final_metrics['descriptor_inference_latency_ms']
            )
            runtime_columns.append('total_inference_latency_ms')
        if runtime_columns:
            runtime_group_cols = [
                column for column in ['scenario', 'agent', 'descriptor_variant']
                if column in final_metrics.columns
            ]
            runtime = final_metrics.groupby(runtime_group_cols, dropna=False)[runtime_columns].agg(["mean", "std"]).reset_index()
            runtime.columns = [
                "_".join(part for part in column if part) if isinstance(column, tuple) else column
                for column in runtime.columns
            ]
            runtime.to_csv(run_dir / "tables" / "runtime_summary.csv", index=False)
            outputs.append("tables/runtime_summary.csv")

        audit_cols = [
            column
            for column in ['scenario', 'seed', 'checkpoint_slot', 'evaluation_variant']
            if column in metrics.columns
        ]
        if audit_cols and "evaluation_trace_hash" in metrics.columns:
            audit = (
                metrics.groupby(audit_cols, dropna=False)
                .agg(
                    evaluation_trace_hash_count=("evaluation_trace_hash", "nunique"),
                    agents_present=("agent", "nunique"),
                    parameter_hash_unchanged=(
                        "parameter_hash_before_eval",
                        lambda values: bool(
                            (
                                metrics.loc[values.index, "parameter_hash_before_eval"].to_numpy()
                                == metrics.loc[values.index, "parameter_hash_after_eval"].to_numpy()
                            ).all()
                        ),
                    ),
                )
                .reset_index()
            )
            audit.to_csv(run_dir / "tables" / "evaluation_audit.csv", index=False)
            outputs.append("tables/evaluation_audit.csv")

    if not metrics.empty and not config.get('r5c2_protocol_version'):
        persisted_metrics = pd.read_csv(run_dir / 'metrics' / 'access_metrics.csv')
        outputs.extend(
            write_expected_figures(
                run_dir,
                persisted_metrics,
                config.get("expected_outputs", []),
                reporting_profile=(
                    'smoke'
                    if args.dry_run or args.quick
                    else str(config.get('reporting_profile', 'formal'))
                ),
            )
        )
    if bool(config.get('pilot_gate', {}).get('enabled', False)):
        gate_table, gate_passed = evaluate_pilot_gate(metrics, config)
        gate_table.to_csv(run_dir / 'tables' / 'pilot_gate_checks.csv', index=False)
        (run_dir / 'analysis' / 'pilot_gate_report.md').write_text(
            render_pilot_gate_report(gate_table, gate_passed),
            encoding='utf-8',
        )
        outputs.extend(['tables/pilot_gate_checks.csv', 'analysis/pilot_gate_report.md'])
        write_json(
            run_dir / 'analysis' / 'pilot_gate.json',
            {
                'passed': bool(gate_passed),
                'checks': json.loads(gate_table.to_json(orient='records')),
            },
        )
        outputs.append('analysis/pilot_gate.json')

    if config.get('e2_mvp_phase') and not (args.dry_run or args.quick):
        e2_payload, e2_checks = evaluate_e2_mvp_gate(
            metrics,
            str(config['e2_mvp_phase']),
        )
        e2_checks.to_csv(run_dir / 'tables' / 'e2_mvp_gate_checks.csv', index=False)
        write_json(run_dir / 'analysis' / 'e2_mvp_gate.json', e2_payload)
        (run_dir / 'analysis' / 'e2_mvp_gate_report.md').write_text(
            render_e2_mvp_report(e2_payload),
            encoding='utf-8',
        )
        outputs.extend(
            [
                'tables/e2_mvp_gate_checks.csv',
                'analysis/e2_mvp_gate.json',
                'analysis/e2_mvp_gate_report.md',
            ]
        )

    if config.get('e2_r1_gate'):
        r1_payload, r1_checks = evaluate_e2_r1_gate(metrics, config)
        r1_paired = e2_r1_paired_seed_effects(metrics, config)
        r1_paired.to_csv(
            run_dir / 'tables' / 'e2_r1_paired_seed_effects.csv',
            index=False,
        )
        r1_final = (
            metrics[metrics['is_final_checkpoint'].astype(bool)].copy()
            if 'is_final_checkpoint' in metrics
            else metrics.copy()
        )
        descriptor_audit_columns = [
            column
            for column in [
                'scenario',
                'agent',
                'seed',
                'module_variant',
                'theta_information_condition',
                'theta_representation',
                'train_physical_trace_hash',
                'evaluation_physical_trace_hash',
                'raw_action_mask_hash',
                'train_theta_marginal_hash',
                'evaluation_theta_marginal_hash',
                'train_theta_idle_correlation',
                'evaluation_theta_idle_correlation',
                'train_theta_unique_count',
                'evaluation_theta_unique_count',
                'train_theta_unique_fraction',
                'evaluation_theta_unique_fraction',
                'train_theta_min',
                'evaluation_theta_min',
                'train_theta_max',
                'evaluation_theta_max',
                'train_theta_busy_max',
                'evaluation_theta_busy_max',
                'train_theta_idle_min',
                'evaluation_theta_idle_min',
                'train_theta_threshold_accuracy',
                'evaluation_theta_threshold_accuracy',
                'theta_observation_hash',
                'reused_curve',
                'reuse_source_run',
                'reuse_source_resolved_config_hash',
                'reuse_source_e2_r1_gate',
                'reused_from_module_variant',
            ]
            if column in r1_final
        ]
        if descriptor_audit_columns:
            r1_final[descriptor_audit_columns].to_csv(
                run_dir / 'tables' / 'e2_r1_descriptor_audit.csv',
                index=False,
            )
        r1_checks.to_csv(
            run_dir / 'tables' / 'e2_r1_gate_checks.csv',
            index=False,
        )
        write_json(run_dir / 'analysis' / 'e2_r1_gate.json', r1_payload)
        (run_dir / 'analysis' / 'e2_r1_gate_report.md').write_text(
            render_e2_r1_report(r1_payload),
            encoding='utf-8',
        )
        r1_outputs = [
            'tables/e2_r1_power_calibration.csv',
            'tables/e2_r1_load_calibration.csv',
            'tables/e2_r1_gate_checks.csv',
            'tables/e2_r1_paired_seed_effects.csv',
            'tables/e2_r1_descriptor_audit.csv',
            'analysis/e2_r1_resolved_config.json',
            'analysis/e2_r1_calibration_failure.json',
            'analysis/e2_r1_gate.json',
            'analysis/e2_r1_gate_report.md',
        ]
        outputs.extend(
            relative for relative in r1_outputs if (run_dir / relative).exists()
        )
    write_notes(run_dir, config, len(metrics))
    outputs.append("analysis/experiment_notes.md")
    write_manifest(run_dir, config, run_id, args.dry_run, args.quick, outputs)
    print(f"Run complete: {run_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
