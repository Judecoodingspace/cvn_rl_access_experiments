from __future__ import annotations

import numpy as np

from cvn_rl.simulation.traces import AccessTrace


DEFAULT_FEATURES = ["delta", "fresh_dqi", "theta"]


SCALAR_FEATURES = {
    'queue_occupancy',
    'hol_age',
    'vehicle_context',
    'dqi_available',
    'prediction_available',
    'sensing_available',
    'dwell_remaining',
    'blackout_remaining',
}


def feature_width(feature: str, num_channels: int) -> int:
    return 1 if feature in SCALAR_FEATURES else num_channels


def build_state(trace: AccessTrace, t: int, features: list[str] | None = None) -> np.ndarray:
    parts = []
    for feature in features or DEFAULT_FEATURES:
        if feature == "delta":
            parts.append(trace.sensing_idle[t])
        elif feature == "fresh_dqi":
            parts.append(trace.fresh_dqi[t])
        elif feature == "theta":
            parts.append(trace.theta[t])
        elif feature == "occupancy_type":
            parts.append(trace.occupancy_type[t])
        elif feature == "power_cap":
            parts.append(trace.power_cap[t])
        elif feature == "dqi_age":
            max_age = max(float(np.max(trace.dqi_age)), 1.0)
            parts.append(trace.dqi_age[t] / max_age)
        else:
            raise ValueError(f"Unknown state feature: {feature}")
    return np.concatenate(parts).astype(float)


def feature_slices(num_channels: int, features: list[str] | None = None) -> dict[str, slice]:
    slices: dict[str, slice] = {}
    start = 0
    for feature in features or DEFAULT_FEATURES:
        width = feature_width(feature, num_channels)
        slices[feature] = slice(start, start + width)
        start += width
    return slices


def observation_feature(
    observation: np.ndarray,
    feature: str,
    num_channels: int,
    features: list[str] | None = None,
    default: float = 0.5,
) -> np.ndarray:
    slices = feature_slices(num_channels, features)
    if feature not in slices:
        return np.full(num_channels, default, dtype=float)
    return observation[slices[feature]].astype(float)
