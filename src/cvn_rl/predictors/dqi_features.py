"""Causal feature construction for DQI predictors."""

from __future__ import annotations

from typing import Sequence

import numpy as np

DEFAULT_DQI_FEATURES = [
    "quality_mbps",
    "snr_db",
    "rss_dbm",
    "cci",
    "bandwidth_mhz",
]


def _shared_vehicle_quantile(values: np.ndarray, quantile: float = 0.10) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim == 3:
        return np.quantile(values, quantile, axis=1, method="linear")
    if values.ndim == 2:
        return values
    raise ValueError(f"expected a [slots, vehicles, channels] or [slots, channels] array, got {values.shape}")


def build_dqi_feature_tensor(
    trace,
    feature_names: Sequence[str] | None = None,
    *,
    vehicle_quantile: float = 0.10,
    physical_scaling: dict[str, float] | None = None,
) -> np.ndarray:
    """Build raw causal DQI features with shape ``[slots, channels, features]``.

    The returned tensor only materializes descriptors at each source slot.  Callers
    choose windows ending at or before the current decision slot; this function
    never uses ``future_dqi_oracle`` or ``quality_lookahead_mbps``.
    """
    names = list(feature_names or DEFAULT_DQI_FEATURES)
    slots = int(trace.slots)
    channels = int(trace.num_channels)
    tensors: list[np.ndarray] = []
    for name in names:
        if name == "quality_mbps":
            arr = _shared_vehicle_quantile(trace.quality_mbps, vehicle_quantile)
            scale = max(float(physical_scaling['quality_mbps']) if physical_scaling is not None
                        else float(np.nanquantile(arr, 0.95)), 1e-6)
            arr = arr / scale
        elif name == "snr_db":
            arr = (_shared_vehicle_quantile(trace.snr_db, vehicle_quantile) + 20.0) / 60.0
        elif name == "rss_dbm":
            arr = (_shared_vehicle_quantile(trace.rss_dbm, vehicle_quantile) + 120.0) / 80.0
        elif name == "cci":
            arr = np.asarray(trace.cci, dtype=float)
        elif name == "bandwidth_mhz":
            arr = np.asarray(trace.bandwidth_mhz, dtype=float)
            scale = (float(physical_scaling['bandwidth_mhz']) if physical_scaling is not None
                     else float(np.nanmax(arr)))
            arr = arr / max(scale, 1e-6)
        elif name == "theta":
            arr = np.asarray(trace.theta, dtype=float)
        elif name == "occupancy_idle":
            arr = (~np.asarray(trace.occupancy, dtype=bool)).astype(float)
        else:
            raise ValueError(f"unknown DQI feature: {name}")
        if arr.shape != (slots, channels):
            raise ValueError(f"feature {name!r} has shape {arr.shape}, expected {(slots, channels)}")
        tensors.append(np.clip(arr, -5.0, 5.0))
    return np.stack(tensors, axis=-1).astype(np.float32)


def flatten_dqi_history(history: np.ndarray) -> np.ndarray:
    """Flatten ``[history, channels, features]`` into LSTM-ready time rows."""
    if history.ndim != 3:
        raise ValueError(f"expected [history, channels, features], got {history.shape}")
    return history.reshape(history.shape[0], history.shape[1] * history.shape[2]).astype(np.float32)
