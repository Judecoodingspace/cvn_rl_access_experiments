from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from cvn_rl.metrics import estimate_broadcast_overhead_kbps
from cvn_rl.simulation.dqi_policy_value import rolling_window_quantile


def summarize_multi_vehicle_records(
    records: list[dict[str, Any]],
    context: dict[str, Any],
    config: dict[str, Any],
    training_time_seconds: float = 0.0,
    inference_latency_ms: float = 0.0,
) -> dict[str, Any]:
    row = dict(context)
    if not records:
        return row

    generated = sum(int(np.sum(record["packet_generated"])) for record in records)
    dropped = sum(int(np.sum(record["packet_dropped"])) for record in records)
    completed = sum(int(np.sum(record["packet_completed"])) for record in records)
    total_served_bits = float(
        sum(float(np.sum(record.get("served_bits", 0.0))) for record in records)
    )
    unfinished = max(generated - dropped - completed, 0)
    slot_duration_ms = float(config.get("slot_duration_ms", 1.0))
    evaluation_seconds = len(records) * slot_duration_ms / 1_000.0
    attempted = sum(int(np.sum(record["attempted"])) for record in records)
    underlay = sum(int(np.sum(record["underlay_selected"] & record["attempted"])) for record in records)
    collisions = sum(int(np.sum(record["vt_collision"])) for record in records)
    interruptions = sum(int(np.sum(record["pu_interruption"])) for record in records)
    low_rate = sum(int(np.sum(record["low_rate_backlog"])) for record in records)
    busy_channel_slots = sum(int(np.sum(record["pu_busy_channels"])) for record in records)
    violations = sum(int(np.sum(record["pu_violation_per_channel"])) for record in records)
    total_channel_slots = sum(int(np.size(record["pu_violation_per_channel"])) for record in records)

    selected = np.concatenate([np.asarray(record["selected_dqi"], dtype=float) for record in records])
    selected = selected[np.isfinite(selected)]
    overlay_transmissions = sum(
        int(np.sum(record.get('overlay_transmitted', np.zeros_like(record['attempted']))))
        for record in records
    )
    underlay_transmissions = sum(
        int(np.sum(record.get('underlay_transmitted', np.zeros_like(record['attempted']))))
        for record in records
    )
    overlay_collisions = sum(
        int(np.sum(record.get('overlay_collision', np.zeros_like(record['attempted']))))
        for record in records
    )
    underlay_violations = sum(
        int(np.sum(record.get('underlay_violation', np.zeros_like(record['attempted']))))
        for record in records
    )
    busy_ratios = []
    for record in records:
        busy = np.asarray(record["pu_busy_channels"], dtype=bool)
        ratios = np.asarray(record["interference_ratio"], dtype=float)
        busy_ratios.extend(ratios[busy].tolist())
    ratio_array = np.asarray(busy_ratios, dtype=float)
    transmitted = sum(
        int(np.sum(record.get('transmitted', record['attempted'])))
        for record in records
    )
    contention_losses = sum(
        int(np.sum(record.get('contention_lost', record['vt_collision'])))
        for record in records
    )
    vehicle_slots = sum(int(np.size(record['attempted'])) for record in records)
    selected_theta = np.concatenate(
        [np.asarray(record.get('selected_theta', []), dtype=float) for record in records]
    )
    selected_theta = selected_theta[np.isfinite(selected_theta)]
    safe_regret = _finite_concat(records, 'safe_action_regret_mbps')
    selected_future_dqi = _finite_concat(records, 'selected_future_dqi')
    channel_regret = _finite_concat(records, 'channel_selection_regret')
    top1_channels = np.concatenate(
        [
            np.asarray(record.get('oracle_top1_channel', []), dtype=int).reshape(-1)
            for record in records
        ]
    )
    top1_channels = top1_channels[top1_channels >= 0]
    access_violations = sum(
        int(np.sum(record.get('pu_access_violation', False)))
        for record in records
    )
    chosen_overlay_busy = sum(
        int(record.get('chosen_overlay_busy', 0)) for record in records
    )
    chosen_overlay_idle = sum(
        int(record.get('chosen_overlay_idle', 0)) for record in records
    )
    legal_overlay_busy = sum(
        int(record.get('legal_overlay_busy_opportunities', 0))
        for record in records
    )
    legal_overlay_idle = sum(
        int(record.get('legal_overlay_idle_opportunities', 0))
        for record in records
    )
    switch_count = sum(
        int(np.sum(record.get('switch_started', False))) for record in records
    )
    dwell_violations = sum(
        int(np.sum(record.get('dwell_violation', False))) for record in records
    )
    blackout_backlog_slots = sum(
        int(np.sum(record.get('blackout_backlog', False))) for record in records
    )
    blackout_loss_bits = float(
        sum(float(np.sum(record.get('blackout_loss_bits', 0.0))) for record in records)
    )

    per_vehicle_throughput = np.mean(
        np.stack([np.asarray(record['vt_throughput_per_vehicle'], dtype=float) for record in records]),
        axis=0,
    )
    fairness_denominator = len(per_vehicle_throughput) * float(np.sum(per_vehicle_throughput**2))
    jain_fairness = (
        float(np.sum(per_vehicle_throughput) ** 2 / fairness_denominator)
        if fairness_denominator > 0.0
        else 0.0
    )

    row.update(
        {
            "plr": float(dropped / max(generated, 1)),
            "generated_packets": generated,
            "dropped_packets": dropped,
            "completed_packets": completed,
            "unfinished_packets": unfinished,
            "completion_rate": float(completed / max(generated, 1)),
            "noncompletion_rate": float(
                (dropped + unfinished) / max(generated, 1)
            ),
            "total_served_bits": total_served_bits,
            "served_goodput_mbps": float(
                total_served_bits / max(evaluation_seconds, 1e-12) / 1_000_000.0
            ),
            "system_vt_throughput": float(np.mean([record["system_vt_throughput"] for record in records])),
            "mean_vt_throughput": float(np.mean([record["mean_vt_throughput"] for record in records])),
            "system_pu_throughput": float(np.mean([record["system_pu_throughput"] for record in records])),
            "pu_interference_violation_rate": float(violations / max(total_channel_slots, 1)),
            "pu_active_violation_rate": float(violations / max(busy_channel_slots, 1)),
            "average_selected_dqi": float(np.mean(selected)) if selected.size else np.nan,
            "underlay_selection_ratio": float(underlay / max(attempted, 1)),
            "vt_collision_rate": float(collisions / max(attempted, 1)),
            "pu_interruption_rate": float(interruptions / max(attempted, 1)),
            "low_rate_backlog_rate": float(low_rate / max(attempted, 1)),
            "queue_overflow_drop_rate": float(dropped / max(generated, 1)),
            "mean_reward": float(np.mean([record["mean_reward"] for record in records])),
            "mean_queue_length": float(np.mean([np.mean(record["queue_lengths"]) for record in records])),
            "mean_queue_occupancy": float(
                np.mean([np.mean(record["queue_lengths"]) for record in records])
                / max(int(config.get("packet_traffic", {}).get("queue_capacity_packets", 10)), 1)
            ),
            "interference_ratio_p50": _quantile(ratio_array, 0.50),
            "interference_ratio_p90": _quantile(ratio_array, 0.90),
            "interference_ratio_p95": _quantile(ratio_array, 0.95),
            "interference_ratio_p99": _quantile(ratio_array, 0.99),
            "interference_ratio_max": float(np.max(ratio_array)) if ratio_array.size else 0.0,
            "training_time_seconds": float(training_time_seconds),
            "inference_latency_ms": float(inference_latency_ms),
            "broadcast_overhead_kbps": estimate_broadcast_overhead_kbps(config),
        }
    )
    row.update(
        {
            'average_selected_theta': (
                float(np.mean(selected_theta)) if selected_theta.size else np.nan
            ),
            'overlay_collision_rate': float(
                overlay_collisions / max(overlay_transmissions, 1)
            ),
            'conditional_underlay_violation_rate': float(
                underlay_violations / max(underlay_transmissions, 1)
            ),
            'overlay_transmissions': overlay_transmissions,
            'underlay_transmissions': underlay_transmissions,
            'jain_fairness': jain_fairness,
            'access_selection_ratio': float(attempted / max(vehicle_slots, 1)),
            'transmission_ratio': float(transmitted / max(vehicle_slots, 1)),
            'contention_loss_rate': float(contention_losses / max(attempted, 1)),
            'mean_hol_age_slots': float(
                np.mean([np.mean(record.get('hol_age_slots', 0.0)) for record in records])
            ),
            'pu_access_violation_rate': float(
                access_violations / max(transmitted, 1)
            ),
            'mean_safe_action_regret_mbps': (
                float(np.mean(safe_regret)) if safe_regret.size else np.nan
            ),
            'overlay_given_busy_legal_backlog': float(
                chosen_overlay_busy / max(legal_overlay_busy, 1)
            ),
            'overlay_given_idle_legal_backlog': float(
                chosen_overlay_idle / max(legal_overlay_idle, 1)
            ),
            'mean_selected_future_dqi': (
                float(np.mean(selected_future_dqi))
                if selected_future_dqi.size else np.nan
            ),
            'mean_channel_selection_regret': (
                float(np.mean(channel_regret)) if channel_regret.size else np.nan
            ),
            'oracle_top1_channel_count': int(np.unique(top1_channels).size),
            'switch_count': int(switch_count),
            'dwell_violation_count': int(dwell_violations),
            'blackout_backlog_slots': int(blackout_backlog_slots),
            'blackout_loss_bits': blackout_loss_bits,
            **_compute_system_served_rolling_tail(
                records, slot_duration_ms,
            ),
        }
    )
    for threshold_value in (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0):
        label = str(threshold_value).replace(".", "p")
        row[f"interference_ccdf_gt_{label}"] = (
            float(np.mean(ratio_array > threshold_value)) if ratio_array.size else 0.0
        )
    return row


def _compute_system_served_rolling_tail(
    records: list[dict[str, Any]],
    slot_duration_ms: float,
) -> dict[str, Any]:
    """Compute per-episode rolling 100-ms q10 of system-level net served goodput.

    Each episode is delimited by ``record['t'] == 0``.  Within each episode
    the per-slot system served goodput (Mbps) is built from ``served_bits``
    summed across all vehicles.  Rolling-window quantiles are computed
    *independently* per episode so that an episode boundary never creates a
    spurious window spanning the tail of one episode and the head of another.
    """
    if not records:
        return {
            'rolling_100ms_system_served_q10_mean': np.nan,
            'rolling_100ms_system_served_q10_p10': np.nan,
            'zero_service_fraction': np.nan,
            'backlog_conditioned_zero_service_fraction': np.nan,
        }

    # Split into episodes at t == 0 boundaries
    t_values = np.asarray([int(record.get('t', 0)) for record in records])
    boundaries = np.concatenate([
        np.flatnonzero(t_values == 0),
        [len(records)],
    ])
    boundaries = np.unique(boundaries)

    all_windows: list[float] = []
    zero_service_slots = 0
    backlog_zero_service_slots = 0
    total_slots = 0

    slot_s = slot_duration_ms / 1000.0
    for ep in range(len(boundaries) - 1):
        start = int(boundaries[ep])
        stop = int(boundaries[ep + 1])
        if start >= stop:
            continue
        ep_records = records[start:stop]
        ep_len = len(ep_records)
        # Per-slot system net served goodput
        net_mbps = np.empty(ep_len, dtype=float)
        for i, record in enumerate(ep_records):
            bits = float(np.sum(record.get('served_bits', 0.0)))
            net_mbps[i] = bits / slot_s / 1e6
            backlog = bool(np.any(record.get('decision_backlog', False)))
            if net_mbps[i] < 1e-12:
                zero_service_slots += 1
                if backlog:
                    backlog_zero_service_slots += 1
        total_slots += ep_len

        if ep_len < 100:
            continue
        # Rolling 100-ms (100-slot at 1 ms/slot) q10
        window_q10 = rolling_window_quantile(
            net_mbps,
            window_samples=100,
            quantile=0.10,
            quantile_method='linear',
        )
        all_windows.extend(window_q10.tolist())

    windows = np.asarray(all_windows, dtype=float)
    return {
        'rolling_100ms_system_served_q10_mean': (
            float(np.mean(windows)) if windows.size else np.nan
        ),
        'rolling_100ms_system_served_q10_p10': (
            float(np.quantile(windows, 0.10)) if windows.size else np.nan
        ),
        'zero_service_fraction': (
            float(zero_service_slots / max(total_slots, 1))
        ),
        'backlog_conditioned_zero_service_fraction': (
            float(backlog_zero_service_slots / max(total_slots, 1))
        ),
    }


def summarize_checkpoint_table(metrics: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        column
        for column in [
            'module_variant',
            'ablation_info_source',
            'ablation_mask_source',
            'mask_policy_version',
            'protection_variant',
            'descriptor_variant',
            'evaluation_variant',
            'descriptor_source',
            "experiment",
            "scenario",
            "resource_regime",
            "agent",
            "checkpoint_slot",
            "traffic_family",
            "traffic_label",
            "arrival_probability",
        ]
        if column in metrics.columns
    ]
    excluded = set(group_cols) | {
        "seed",
        "train_trace_hash",
        "evaluation_trace_hash",
        "parameter_hash_before_eval",
        "parameter_hash_after_eval",
    }
    metric_cols = [
        column
        for column in metrics.select_dtypes(include=[np.number]).columns
        if column not in excluded
    ]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', pd.errors.PerformanceWarning)
        grouped = (
            metrics.groupby(group_cols, dropna=False)[metric_cols]
            .agg(['mean', 'std', 'count'])
            .reset_index()
        )
    grouped.columns = [
        "_".join(part for part in column if part) if isinstance(column, tuple) else column
        for column in grouped.columns
    ]
    ci_columns: dict[str, pd.Series] = {}
    for metric in metric_cols:
        std_col = f"{metric}_std"
        count_col = f"{metric}_count"
        if std_col in grouped and count_col in grouped:
            ci_columns[f"{metric}_ci95"] = (
                grouped[count_col].apply(
                    lambda count: float(student_t.ppf(0.975, count - 1)) if count > 1 else 0.0
                )
                * grouped[std_col].fillna(0.0)
                / np.sqrt(grouped[count_col].clip(lower=1))
            )
    if ci_columns:
        grouped = pd.concat([grouped, pd.DataFrame(ci_columns, index=grouped.index)], axis=1)
    return grouped


def _quantile(values: np.ndarray, q: float) -> float:
    return float(np.quantile(values, q)) if values.size else 0.0


def _finite_concat(
    records: list[dict[str, Any]],
    key: str,
) -> np.ndarray:
    values = np.concatenate(
        [np.asarray(record.get(key, []), dtype=float).reshape(-1) for record in records]
    )
    return values[np.isfinite(values)]
