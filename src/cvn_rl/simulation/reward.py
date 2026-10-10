from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from cvn_rl.simulation.actions import AccessAction
from cvn_rl.simulation.traces import AccessTrace


@dataclass(frozen=True)
class StepOutcome:
    reward: float
    vt_throughput: float
    pu_throughput: float
    pu_violation: bool
    packet_loss: bool
    selected_dqi: float
    underlay_selected: bool
    actual_interference_mw: float
    pu_threshold_mw: float
    transmit_power_mw: float
    interference_margin_mw: float


def compute_step_outcome(
    trace: AccessTrace,
    t: int,
    action: AccessAction,
    config: dict[str, Any],
) -> StepOutcome:
    if action.mode == "no_access" or action.channel < 0:
        busy_count = float(np.mean(trace.occupancy[t]))
        return StepOutcome(0.0, 0.0, busy_count, False, False, np.nan, False, 0.0, np.nan, 0.0, np.nan)

    channel = action.channel
    busy = bool(trace.occupancy[t, channel])
    quality = float(trace.quality_mbps[t, channel])
    dqi = float(trace.fresh_dqi[t, channel])
    threshold = float(trace.pu_threshold_mw[t, channel])
    true_gain = float(trace.su_pu_gain_true[t, channel])
    tx_cap = float(trace.transmit_power_cap_mw[t, channel])
    reward_cfg = config.get("reward", {})
    underlay_cfg = config.get("underlay_power", {})
    c_max = max(float(config.get("dqi", {}).get("r_ref_mbps", 80)), 1e-6)
    gamma = float(reward_cfg.get("overlay_quality_exponent_gamma", 2.0))
    alpha = float(reward_cfg.get("underlay_gate_alpha", 5.0))
    penalty = float(reward_cfg.get("violation_penalty_kappa", 1.0))
    collision_penalty = float(reward_cfg.get("collision_penalty", 1.0))
    p_max = float(underlay_cfg.get("p_max_mw", 1.0))
    low, high = reward_cfg.get("clip_reward", [-1.0, 1.0])

    if action.mode == "overlay":
        transmit_power = p_max
        actual_interference = true_gain * transmit_power if busy else 0.0
        margin = threshold - actual_interference if busy else threshold
        collision = busy
        vt = 0.0 if collision else quality * (dqi**gamma)
        pu = 0.0 if collision else (1.0 if busy else 0.0)
        reward = vt / c_max - (collision_penalty if collision else 0.0)
        return StepOutcome(
            reward=float(np.clip(reward, low, high)),
            vt_throughput=vt,
            pu_throughput=pu,
            pu_violation=collision,
            packet_loss=collision or vt <= 0.0,
            selected_dqi=dqi,
            underlay_selected=False,
            actual_interference_mw=actual_interference,
            pu_threshold_mw=threshold,
            transmit_power_mw=transmit_power,
            interference_margin_mw=margin,
        )

    transmit_power = tx_cap
    actual_interference = true_gain * transmit_power if busy else 0.0
    margin = threshold - actual_interference if busy else threshold
    violation = busy and actual_interference > threshold
    normalized_margin = margin / max(threshold, 1e-9)
    gate = 1.0 / (1.0 + np.exp(-alpha * normalized_margin))
    vt = quality * dqi * gate * 0.65
    pu = (1.0 if busy else 0.0) * (0.2 if violation else 0.9)
    reward = vt / c_max - (penalty if violation else 0.0)
    return StepOutcome(
        reward=float(np.clip(reward, low, high)),
        vt_throughput=vt,
        pu_throughput=pu,
        pu_violation=violation,
        packet_loss=violation or vt <= 0.0,
        selected_dqi=dqi,
        underlay_selected=True,
        actual_interference_mw=actual_interference,
        pu_threshold_mw=threshold,
        transmit_power_mw=transmit_power,
        interference_margin_mw=margin,
    )
