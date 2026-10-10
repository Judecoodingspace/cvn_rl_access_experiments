from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def summarize_episode(
    step_records: list[dict[str, Any]],
    context: dict[str, Any],
    config: dict[str, Any],
    training_time_seconds: float = 0.0,
    inference_latency_ms: float = 0.0,
) -> dict[str, Any]:
    df = pd.DataFrame(step_records)
    attempted = df["mode"] != "no_access"
    underlay = df["mode"] == "underlay"
    selected_dqi = df.loc[attempted, "selected_dqi"].dropna()
    row = dict(context)
    row.update(
        {
            "plr": float(df.loc[attempted, "packet_loss"].mean()) if attempted.any() else 0.0,
            "vt_throughput": float(df["vt_throughput"].mean()),
            "pu_throughput": float(df["pu_throughput"].mean()),
            "pu_interference_violation_rate": float(df["pu_violation"].mean()),
            "conditional_underlay_violation_rate": float(df.loc[underlay, "pu_violation"].mean()) if underlay.any() else 0.0,
            "average_selected_dqi": float(selected_dqi.mean()) if not selected_dqi.empty else np.nan,
            "underlay_selection_ratio": float(underlay.mean()),
            "convergence_slots": estimate_convergence_slots(df["reward"].to_numpy()),
            "mean_reward": float(df["reward"].mean()),
            "average_actual_interference_mw": float(df["actual_interference_mw"].mean()) if "actual_interference_mw" in df else 0.0,
            "mean_interference_margin_mw": float(df["interference_margin_mw"].dropna().mean()) if "interference_margin_mw" in df else np.nan,
            "training_time_seconds": float(training_time_seconds),
            "inference_latency_ms": float(inference_latency_ms),
            "broadcast_overhead_kbps": estimate_broadcast_overhead_kbps(config),
        }
    )
    return row


def estimate_convergence_slots(rewards: np.ndarray, window: int = 50) -> int:
    if len(rewards) == 0:
        return 0
    if len(rewards) < window:
        return len(rewards)
    kernel = np.ones(window) / window
    rolling = np.convolve(rewards, kernel, mode="valid")
    final = float(np.mean(rewards[-window:]))
    tolerance = max(abs(final) * 0.05, 0.02)
    hits = np.flatnonzero(np.abs(rolling - final) <= tolerance)
    return int(hits[0] + window) if len(hits) else len(rewards)


def estimate_broadcast_overhead_kbps(config: dict[str, Any]) -> float:
    payload = config.get("payload_bits_per_channel")
    if payload is None:
        payload = {"dqi": 8, "theta": 8, "power_cap": 8, "occupancy_type": 2}
    num_channels = int(config.get("num_channels", 10))
    component_updates = config.get('broadcast_component_update_slots')
    slot_ms = float(config.get("slot_duration_ms", 1))
    if component_updates:
        bits_per_ms = sum(
            float(bits) / max(float(component_updates.get(name, 1)) * slot_ms, 1e-9)
            for name, bits in payload.items()
        )
        return float(num_channels * bits_per_ms)
    bits_per_channel = float(sum(payload.values()))
    update_slots = int(
        config.get("broadcast_update_slots", config.get("underlay_power", {}).get("broadcast_update_slots", 10))
    )
    return float(num_channels * bits_per_channel / max(update_slots * slot_ms, 1e-9))


def summarize_table(metrics: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        col
        for col in [
            "experiment",
            "agent",
            "scenario",
            "resource_regime",
            "state_variant",
            "reward_variant",
            "traffic_family",
            "traffic_label",
            "sweep_name",
            "broadcast_update_slots",
            "dqi_validity_slots",
        ]
        if col in metrics.columns
    ]
    metric_cols = [
        "plr",
        "vt_throughput",
        "pu_throughput",
        "pu_interference_violation_rate",
        "conditional_underlay_violation_rate",
        "average_selected_dqi",
        "underlay_selection_ratio",
        "convergence_slots",
        "mean_reward",
        "average_actual_interference_mw",
        "mean_interference_margin_mw",
        "broadcast_overhead_kbps",
    ]
    metric_cols = [col for col in metric_cols if col in metrics.columns]
    agg = metrics.groupby(group_cols, dropna=False)[metric_cols].agg(["mean", "std"]).reset_index()
    agg.columns = ["_".join([part for part in col if part]) if isinstance(col, tuple) else col for col in agg.columns]
    return agg
