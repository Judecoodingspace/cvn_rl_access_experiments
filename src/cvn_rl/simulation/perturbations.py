from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from cvn_rl.simulation.traces import AccessTrace, recompute_underlay_power_cap


def apply_trace_perturbations(
    trace: AccessTrace,
    params: dict[str, Any],
    seed: int,
) -> AccessTrace:
    rng = np.random.default_rng(seed)
    theta = trace.theta.copy()
    sensing_idle = trace.sensing_idle.copy()
    gain_hat = trace.su_pu_gain_hat.copy()
    local_config: dict[str, Any] = {"underlay_power": {"p_max_mw": 1.0, "fading_margin": 2.0}}

    false_alarm = float(params.get("prediction_false_alarm_rate", 0.0))
    miss_detection = float(params.get("prediction_miss_detection_rate", 0.0))
    if false_alarm or miss_detection:
        idle = ~trace.occupancy
        false_alarm_mask = idle & (rng.random(theta.shape) < false_alarm)
        miss_detection_mask = trace.occupancy & (rng.random(theta.shape) < miss_detection)
        theta[false_alarm_mask] *= 0.35
        theta[miss_detection_mask] = np.maximum(theta[miss_detection_mask], 0.75)

    if "sensing_pd" in params or "sensing_pfa" in params:
        pd = float(params.get("sensing_pd", 0.9))
        pfa = float(params.get("sensing_pfa", 0.1))
        sensed_busy = np.where(
            trace.occupancy,
            rng.random(trace.occupancy.shape) < pd,
            rng.random(trace.occupancy.shape) < pfa,
        )
        sensing_idle = (~sensed_busy).astype(float)

    if "path_loss_error_db" in params:
        error_db = float(params["path_loss_error_db"])
        gain_hat = np.clip(gain_hat * np.power(10.0, error_db / 10.0), 1e-9, None)
    if "fading_margin" in params:
        local_config["underlay_power"]["fading_margin"] = float(params["fading_margin"])

    power_cap, transmit_cap = recompute_underlay_power_cap(gain_hat, trace.pu_threshold_mw, local_config)
    return replace(
        trace,
        theta=np.clip(theta, 0.0, 1.0),
        sensing_idle=sensing_idle,
        su_pu_gain_hat=gain_hat,
        power_cap=power_cap,
        transmit_power_cap_mw=transmit_cap,
    )
