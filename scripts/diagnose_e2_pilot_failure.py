from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from cvn_rl.experiments.config import load_config
from cvn_rl.simulation.actions import build_action_mask
from cvn_rl.simulation.multi_traces import generate_multi_vehicle_trace


def main() -> int:
    parser = argparse.ArgumentParser(description='Diagnose a failed E2 pilot signal gate.')
    parser.add_argument('run_dir')
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    metrics = pd.read_csv(run_dir / 'metrics' / 'access_metrics.csv')
    final = metrics[metrics['is_final_checkpoint'].astype(bool)].copy()
    summary_metrics = [
        'system_vt_throughput',
        'plr',
        'overlay_collision_rate',
        'conditional_underlay_violation_rate',
        'mean_reward',
        'average_selected_dqi',
        'average_selected_theta',
        'access_selection_ratio',
        'underlay_selection_ratio',
        'mean_queue_length',
        'mean_hol_age_slots',
    ]
    summary = (
        final.groupby(['scenario', 'module_variant'])[summary_metrics]
        .agg(['mean', 'std'])
        .reset_index()
    )
    summary.columns = [
        '_'.join(part for part in column if part)
        if isinstance(column, tuple)
        else column
        for column in summary.columns
    ]
    summary.to_csv(run_dir / 'tables' / 'failure_module_summary.csv', index=False)

    pivot = final.pivot_table(
        index=['scenario', 'seed'],
        columns='module_variant',
        values=['system_vt_throughput', 'plr', 'overlay_collision_rate'],
        aggfunc='mean',
    )
    seed_effects = pd.DataFrame(index=pivot.index)
    seed_effects['dqi_throughput_gain'] = (
        pivot[('system_vt_throughput', 'full')]
        - pivot[('system_vt_throughput', 'no_dqi')]
    )
    seed_effects['prediction_throughput_gain'] = (
        pivot[('system_vt_throughput', 'full')]
        - pivot[('system_vt_throughput', 'no_prediction')]
    )
    seed_effects['prediction_plr_reduction'] = (
        pivot[('plr', 'no_prediction')] - pivot[('plr', 'full')]
    )
    seed_effects['prediction_collision_reduction'] = (
        pivot[('overlay_collision_rate', 'no_prediction')]
        - pivot[('overlay_collision_rate', 'full')]
    )
    seed_effects.reset_index().to_csv(
        run_dir / 'tables' / 'failure_seed_effects.csv',
        index=False,
    )

    config = load_config(run_dir / 'config.frozen.json')
    descriptor_rows: list[dict[str, float | int | str]] = []
    for scenario in config['scenarios']:
        scenario_name = str(scenario['name'])
        scenario_config = {
            **config,
            **{key: value for key, value in scenario.items() if key != 'name'},
        }
        for seed in (1, 2, 3):
            trace = generate_multi_vehicle_trace(
                scenario_config,
                seed=seed,
                slots=int(config.get('slots', 30_000)),
                split='diagnostic',
            )
            dqi = trace.fresh_dqi
            quality = trace.quality_mbps[:, 0, :]
            idle = (~trace.occupancy).astype(float)
            full_masks = []
            neutral_masks = []
            for slot in range(trace.slots):
                full_masks.append(
                    build_action_mask(
                        trace.sensing_idle[slot, 0],
                        trace.theta[slot],
                        trace.power_cap[slot, 0],
                        ['reward_gate', 'action_mask'],
                    )
                )
                neutral_masks.append(
                    build_action_mask(
                        trace.sensing_idle[slot, 0],
                        np.full(trace.num_channels, 0.5),
                        trace.power_cap[slot, 0],
                        ['reward_gate', 'action_mask'],
                    )
                )
            full_masks_array = np.stack(full_masks)
            neutral_masks_array = np.stack(neutral_masks)
            descriptor_rows.append(
                {
                    'scenario': scenario_name,
                    'seed': seed,
                    'dqi_quality_correlation': float(
                        np.corrcoef(dqi.ravel(), quality.ravel())[0, 1]
                    ),
                    'dqi_current_top1_accuracy': float(
                        np.mean(
                            np.argmax(dqi, axis=1)
                            == np.argmax(quality, axis=1)
                        )
                    ),
                    'dqi_neutral_fraction': float(np.mean(np.isclose(dqi, 0.5))),
                    'dqi_mean_channel_std': float(np.mean(np.std(dqi, axis=1))),
                    'theta_brier': float(np.mean((trace.theta - idle) ** 2)),
                    'theta_idle_correlation': float(
                        np.corrcoef(trace.theta.ravel(), idle.ravel())[0, 1]
                    ),
                    'full_overlay_allowed': float(full_masks_array[:, 1::2].mean()),
                    'neutral_overlay_allowed': float(
                        neutral_masks_array[:, 1::2].mean()
                    ),
                    'full_underlay_allowed': float(full_masks_array[:, 2::2].mean()),
                    'neutral_underlay_allowed': float(
                        neutral_masks_array[:, 2::2].mean()
                    ),
                    'full_actions_allowed': float(full_masks_array[:, 1:].mean()),
                    'neutral_actions_allowed': float(
                        neutral_masks_array[:, 1:].mean()
                    ),
                }
            )
    descriptor = pd.DataFrame(descriptor_rows)
    descriptor.to_csv(
        run_dir / 'tables' / 'failure_descriptor_diagnostics.csv',
        index=False,
    )
    descriptor_mean = descriptor.groupby('scenario').mean(numeric_only=True)

    high_m3 = descriptor_mean.loc['M3_high_pu']
    high_m10 = descriptor_mean.loc['M10_high_pu']
    dqi_m3 = descriptor_mean.loc[
        ['M3_low_pu', 'M3_high_pu'], 'dqi_quality_correlation'
    ].mean()
    dqi_m10 = descriptor_mean.loc[
        ['M10_low_pu', 'M10_high_pu'], 'dqi_quality_correlation'
    ].mean()
    report = f'''# E2 Pilot Failure Diagnosis

## Decision

- Formal progression: **STOP**.
- Integrity gate passed, but the pre-registered signal gate failed.
- No seeds or checkpoints were removed, and no formal experiment was started.

## DQI

- Mean DQI/current-quality correlation: {dqi_m3:.3f} for M3 and {dqi_m10:.3f} for M10.
- The descriptor is informative, but full improved throughput in only 1/4 scenarios.
- Final selected-DQI means are similar between full and no-DQI, so the controller did not convert the ranking information into a stable access gain.

## Prediction and action mask

- In M3_high_pu, prediction reduced the allowed-action fraction from {high_m3['neutral_actions_allowed']:.3f} to {high_m3['full_actions_allowed']:.3f}.
- In M3_high_pu, underlay availability fell from {high_m3['neutral_underlay_allowed']:.3f} to {high_m3['full_underlay_allowed']:.3f}.
- In M10_high_pu, prediction reduced the allowed-action fraction from {high_m10['neutral_actions_allowed']:.3f} to {high_m10['full_actions_allowed']:.3f}.
- The theta signal is informative, but the theta threshold suppresses underlay exactly in busy periods where closed-form power control is intended to permit coexistence.
- Sensing already gates overlay access, so theta adds limited overlay safety while strongly restricting underlay.

## Endpoint sensitivity

- PLR is zero for most final scenario/variant combinations, making it an insensitive pilot endpoint at arrival probability 0.2.
- The observed nonzero PLR is concentrated in isolated seed/variant failures, indicating policy instability rather than a consistent module effect.

## Required diagnosis before a new pilot

1. Separate prediction state information from the underlay action mask; test state-only prediction before changing the predictor.
2. Verify that DQI changes channel choice using action-level agreement/rank diagnostics.
3. Add a controlled higher-queue-pressure diagnostic only for endpoint sensitivity; do not replace the registered arrival-0.2 experiment silently.
4. Keep the current failed pilot as the immutable baseline and pre-register any revised pilot.
'''
    (run_dir / 'analysis' / 'failure_diagnosis.md').write_text(
        report,
        encoding='utf-8',
    )

    manifest_path = run_dir / 'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    additions = [
        'tables/failure_module_summary.csv',
        'tables/failure_seed_effects.csv',
        'tables/failure_descriptor_diagnostics.csv',
        'analysis/failure_diagnosis.md',
    ]
    manifest['outputs'] = list(dict.fromkeys([*manifest.get('outputs', []), *additions]))
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding='utf-8',
    )
    print(run_dir / 'analysis' / 'failure_diagnosis.md')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
