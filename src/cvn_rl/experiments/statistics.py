from __future__ import annotations

import numpy as np
import pandas as pd


def paired_bootstrap_table(
    metrics: pd.DataFrame,
    proposed_agent: str = "proposed_esn_ddqn",
    comparators: tuple[str, ...] = ("ddqn_no_esn", "dqn"),
    bootstrap_samples: int = 10_000,
    seed: int = 20260627,
) -> pd.DataFrame:
    if metrics.empty or "agent" not in metrics or "seed" not in metrics:
        return pd.DataFrame()
    data = metrics[metrics.get("is_final_checkpoint", True).astype(bool)] if "is_final_checkpoint" in metrics else metrics
    metrics_to_compare = [
        column
        for column in [
            "mean_reward",
            "system_vt_throughput",
            "mean_vt_throughput",
            "plr",
            "pu_active_violation_rate",
        ]
        if column in data.columns
    ]
    group_cols = [
        column
        for column in ["experiment", "scenario", "traffic_family", "traffic_label", "arrival_probability"]
        if column in data.columns
    ]
    if 'module_variant' in data.columns:
        group_cols.append('module_variant')
    if 'mask_policy_version' in data.columns:
        group_cols.append('mask_policy_version')
    if 'descriptor_variant' in data.columns:
        group_cols.append('descriptor_variant')
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for keys, group in data.groupby(group_cols, dropna=False):
        key_values = keys if isinstance(keys, tuple) else (keys,)
        context = dict(zip(group_cols, key_values))
        proposed = group[group["agent"] == proposed_agent].set_index("seed")
        for comparator in comparators:
            baseline = group[group["agent"] == comparator].set_index("seed")
            common = proposed.index.intersection(baseline.index)
            if len(common) == 0:
                continue
            for metric in metrics_to_compare:
                delta = proposed.loc[common, metric].to_numpy(float) - baseline.loc[common, metric].to_numpy(float)
                draws = rng.choice(delta, size=(bootstrap_samples, len(delta)), replace=True).mean(axis=1)
                baseline_mean = float(baseline.loc[common, metric].mean())
                rows.append(
                    {
                        **context,
                        "proposed_agent": proposed_agent,
                        "comparator": comparator,
                        "metric": metric,
                        "paired_seeds": len(common),
                        "mean_delta": float(np.mean(delta)),
                        "std_delta": float(np.std(delta, ddof=1)) if len(delta) > 1 else 0.0,
                        "ci95_low": float(np.quantile(draws, 0.025)),
                        "ci95_high": float(np.quantile(draws, 0.975)),
                        "relative_delta_percent": float(100.0 * np.mean(delta) / max(abs(baseline_mean), 1e-12)),
                    }
                )
    return pd.DataFrame(rows)


def module_ablation_bootstrap_table(
    metrics: pd.DataFrame,
    bootstrap_samples: int = 10_000,
    seed: int = 20260627,
) -> pd.DataFrame:
    required = {'agent', 'seed', 'module_variant'}
    if metrics.empty or not required.issubset(metrics.columns):
        return pd.DataFrame()
    data = (
        metrics[metrics['is_final_checkpoint'].astype(bool)]
        if 'is_final_checkpoint' in metrics
        else metrics
    )
    metric_directions = {
        'mean_reward': 'higher',
        'system_vt_throughput': 'higher',
        'mean_vt_throughput': 'higher',
        'plr': 'lower',
        'overlay_collision_rate': 'lower',
        'pu_active_violation_rate': 'lower',
        'conditional_underlay_violation_rate': 'lower',
        'contention_loss_rate': 'lower',
        'jain_fairness': 'higher',
        'served_goodput_mbps': 'higher',
        'noncompletion_rate': 'lower',
    }
    group_cols = [
        column
        for column in [
            'experiment', 'scenario', 'agent', 'traffic_family',
            'traffic_label', 'arrival_probability', 'protection_variant',
            'evaluation_variant',
            'mask_policy_version',
        ]
        if column in data.columns
    ]
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for keys, group in data.groupby(group_cols, dropna=False):
        key_values = keys if isinstance(keys, tuple) else (keys,)
        context = dict(zip(group_cols, key_values))
        full = group[group['module_variant'] == 'full'].set_index('seed')
        for ablated in (
            'no_dqi', 'no_prediction', 'no_prediction_info',
            'sensing_only', 'state_sensing_only',
        ):
            baseline = group[group['module_variant'] == ablated].set_index('seed')
            common = full.index.intersection(baseline.index)
            if len(common) == 0:
                continue
            for metric, direction in metric_directions.items():
                if metric not in group.columns:
                    continue
                raw = full.loc[common, metric].to_numpy(float) - baseline.loc[common, metric].to_numpy(float)
                oriented = raw if direction == 'higher' else -raw
                draws = rng.choice(
                    oriented,
                    size=(bootstrap_samples, len(oriented)),
                    replace=True,
                ).mean(axis=1)
                p_value = min(
                    1.0,
                    2.0 * min(float(np.mean(draws <= 0.0)), float(np.mean(draws >= 0.0))),
                )
                rows.append(
                    {
                        **context,
                        'full_variant': 'full',
                        'ablated_variant': ablated,
                        'metric': metric,
                        'preferred_direction': direction,
                        'paired_seeds': len(common),
                        'raw_full_minus_ablated': float(np.mean(raw)),
                        'oriented_module_gain': float(np.mean(oriented)),
                        'ci95_low': float(np.quantile(draws, 0.025)),
                        'ci95_high': float(np.quantile(draws, 0.975)),
                        'p_value': p_value,
                        'significant_positive_gain': bool(np.quantile(draws, 0.025) > 0.0),
                    }
                )
    table = pd.DataFrame(rows)
    if not table.empty:
        correction_groups = [*group_cols, 'metric']
        table['holm_p_value'] = table.groupby(
            correction_groups,
            dropna=False,
        )['p_value'].transform(_holm_adjust)
        table['holm_significant_0p05'] = table['holm_p_value'] < 0.05
    return table


def module_interaction_bootstrap_table(
    metrics: pd.DataFrame,
    bootstrap_samples: int = 10_000,
    seed: int = 20260628,
) -> pd.DataFrame:
    required = {'agent', 'seed', 'module_variant'}
    if metrics.empty or not required.issubset(metrics.columns):
        return pd.DataFrame()
    data = (
        metrics[metrics['is_final_checkpoint'].astype(bool)]
        if 'is_final_checkpoint' in metrics
        else metrics
    )
    metric_names = [
        name
        for name in (
            'system_vt_throughput',
            'mean_vt_throughput',
            'plr',
            'overlay_collision_rate',
            'pu_active_violation_rate',
        )
        if name in data
    ]
    group_cols = [
        name
        for name in (
            'experiment',
            'scenario',
            'agent',
            'protection_variant',
            'evaluation_variant',
            'traffic_family',
            'traffic_label',
            'arrival_probability',
            'mask_policy_version',
        )
        if name in data
    ]
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for keys, group in data.groupby(group_cols, dropna=False):
        prediction_variant = (
            'no_prediction_info'
            if 'no_prediction_info' in set(group['module_variant'])
            else 'no_prediction'
        )
        sensing_variant = (
            'state_sensing_only'
            if 'state_sensing_only' in set(group['module_variant'])
            else 'sensing_only'
        )
        variants = {'full', 'no_dqi', prediction_variant, sensing_variant}
        if not variants.issubset(set(group['module_variant'])):
            continue
        key_values = keys if isinstance(keys, tuple) else (keys,)
        context = dict(zip(group_cols, key_values))
        for metric in metric_names:
            pivot = group.pivot_table(
                index='seed',
                columns='module_variant',
                values=metric,
                aggfunc='mean',
            ).dropna(subset=sorted(variants))
            if pivot.empty:
                continue
            interaction = (
                pivot['full']
                - pivot[prediction_variant]
                - pivot['no_dqi']
                + pivot[sensing_variant]
            ).to_numpy(float)
            draws = rng.choice(
                interaction,
                size=(bootstrap_samples, len(interaction)),
                replace=True,
            ).mean(axis=1)
            rows.append(
                {
                    **context,
                    'metric': metric,
                    'paired_seeds': len(interaction),
                    'interaction_mean': float(np.mean(interaction)),
                    'interaction_std': (
                        float(np.std(interaction, ddof=1)) if len(interaction) > 1 else 0.0
                    ),
                    'ci95_low': float(np.quantile(draws, 0.025)),
                    'ci95_high': float(np.quantile(draws, 0.975)),
                }
            )
    return pd.DataFrame(rows)


def select_training_horizon(
    metrics: pd.DataFrame,
    *,
    agent: str = 'ddqn_no_esn',
    module_variant: str = 'full',
    min_slots: int = 10_000,
    max_slots: int = 30_000,
    relative_tolerance: float = 0.02,
    plr_absolute_tolerance: float = 0.005,
    consecutive_intervals: int = 3,
) -> pd.DataFrame:
    required = {
        'scenario',
        'agent',
        'checkpoint_slot',
        'mean_reward',
        'system_vt_throughput',
        'plr',
    }
    if metrics.empty or not required.issubset(metrics.columns):
        return pd.DataFrame()
    data = metrics[metrics['agent'] == agent].copy()
    if 'module_variant' in data:
        data = data[data['module_variant'] == module_variant]
    if 'evaluation_variant' in data and (data['evaluation_variant'] == 'nominal').any():
        data = data[data['evaluation_variant'] == 'nominal']
    rows: list[dict[str, object]] = []
    for scenario, group in data.groupby('scenario', dropna=False):
        curve = (
            group.groupby('checkpoint_slot', as_index=False)[
                ['mean_reward', 'system_vt_throughput', 'plr']
            ]
            .mean()
            .sort_values('checkpoint_slot')
        )
        selected = min(max_slots, int(curve['checkpoint_slot'].max()))
        converged = False
        first_index = consecutive_intervals
        for index in range(first_index, len(curve)):
            checkpoint = int(curve.iloc[index]['checkpoint_slot'])
            if checkpoint < min_slots:
                continue
            window = curve.iloc[index - consecutive_intervals : index + 1]
            reward = window['mean_reward'].to_numpy(float)
            throughput = window['system_vt_throughput'].to_numpy(float)
            plr = window['plr'].to_numpy(float)
            reward_change = np.abs(np.diff(reward)) / np.maximum(np.abs(reward[:-1]), 1e-9)
            throughput_change = np.abs(np.diff(throughput)) / np.maximum(
                np.abs(throughput[:-1]),
                1e-9,
            )
            if (
                np.all(reward_change <= relative_tolerance)
                and np.all(throughput_change <= relative_tolerance)
                and np.all(np.abs(np.diff(plr)) <= plr_absolute_tolerance)
            ):
                selected = checkpoint
                converged = True
                break
        rows.append(
            {
                'scenario': scenario,
                'agent': agent,
                'module_variant': module_variant,
                'selected_checkpoint': selected,
                'converged': converged,
                'observed_max_checkpoint': int(curve['checkpoint_slot'].max()),
                'consecutive_intervals': consecutive_intervals,
                'relative_tolerance': relative_tolerance,
                'plr_absolute_tolerance': plr_absolute_tolerance,
            }
        )
    table = pd.DataFrame(rows)
    if not table.empty:
        table = pd.concat(
            [
                table,
                pd.DataFrame(
                    [
                        {
                            'scenario': '__global__',
                            'agent': agent,
                            'module_variant': module_variant,
                            'selected_checkpoint': int(table['selected_checkpoint'].max()),
                            'converged': bool(table['converged'].all()),
                            'observed_max_checkpoint': int(
                                table['observed_max_checkpoint'].max()
                            ),
                            'consecutive_intervals': consecutive_intervals,
                            'relative_tolerance': relative_tolerance,
                            'plr_absolute_tolerance': plr_absolute_tolerance,
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
    return table


def _holm_adjust(values: pd.Series) -> pd.Series:
    raw = values.to_numpy(float)
    order = np.argsort(raw)
    adjusted = np.empty_like(raw)
    running = 0.0
    total = len(raw)
    for rank, index in enumerate(order):
        running = max(running, (total - rank) * raw[index])
        adjusted[index] = min(running, 1.0)
    return pd.Series(adjusted, index=values.index)
