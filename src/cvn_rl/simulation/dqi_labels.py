from __future__ import annotations

import numpy as np

from dataclasses import dataclass
from typing import Iterable

DQI_LABEL_VERSION = 'paper_dqi_v1'


@dataclass(frozen=True)
class DQILabels:
    z_mbps: np.ndarray
    y: np.ndarray
    source_slots: np.ndarray
    r_ref_mbps: float


@dataclass(frozen=True)
class FeatureScaler:
    feature_names: tuple[str, ...]
    mean: np.ndarray
    scale: np.ndarray
    sample_count: int

    def transform(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=float)
        if array.shape[-1] != len(self.feature_names):
            raise ValueError('descriptor feature dimension does not match scaler')
        return (array - self.mean) / self.scale


def transmission_delay_ms(rate_mbps: np.ndarray, packet_size_bits: float = 32_000.0) -> np.ndarray:
    rate = np.asarray(rate_mbps, dtype=float)
    return float(packet_size_bits) / (np.maximum(rate, 0.01) * 1_000.0)


def latency_constrained_goodput(
    rate_mbps: np.ndarray,
    *,
    delay_ms: np.ndarray | None = None,
    packet_size_bits: float = 32_000.0,
    delay_max_ms: float = 5.0,
    delay_beta_ms: float = 2.0,
) -> np.ndarray:
    rate = np.asarray(rate_mbps, dtype=float)
    delay = transmission_delay_ms(rate, packet_size_bits)
    if delay_ms is not None:
        delay = np.asarray(delay_ms, dtype=float)
    if delay.shape != rate.shape:
        raise ValueError('delay_ms must have the same shape as rate_mbps')
    beta = max(float(delay_beta_ms), 1e-12)
    logits = np.clip((delay - float(delay_max_ms)) / beta, -60.0, 60.0)
    soft_violation = 1.0 / (1.0 + np.exp(-logits))
    return rate * (1.0 - soft_violation)


def compute_unnormalized_dqi(
    rate_mbps: np.ndarray,
    *,
    delay_ms: np.ndarray | None = None,
    horizon_samples: int = 100,
    vehicle_quantile: float = 0.10,
    time_quantile: float = 0.10,
    quantile_method: str = 'linear',
    packet_size_bits: float = 32_000.0,
    delay_max_ms: float = 5.0,
    delay_beta_ms: float = 2.0,
    source_indices: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    rate = np.asarray(rate_mbps, dtype=float)
    if rate.ndim != 3:
        raise ValueError('rate_mbps must have shape [time, vehicle, channel]')
    horizon = int(horizon_samples)
    if horizon <= 0 or rate.shape[0] <= horizon:
        raise ValueError('rate_mbps must contain horizon_samples + 1 slots')
    for name, value in (
        ('vehicle_quantile', vehicle_quantile),
        ('time_quantile', time_quantile),
    ):
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError(f'{name} must lie in [0, 1]')
    goodput = latency_constrained_goodput(
        rate,
        delay_ms=delay_ms,
        packet_size_bits=packet_size_bits,
        delay_max_ms=delay_max_ms,
        delay_beta_ms=delay_beta_ms,
    )
    shared = np.quantile(
        goodput,
        vehicle_quantile,
        axis=1,
        method=quantile_method,
    )
    source_slots = (np.arange(rate.shape[0] - horizon, dtype=int) if source_indices is None
                    else np.asarray(source_indices, dtype=int))
    if source_slots.ndim != 1 or np.any(source_slots < 0) or np.any(source_slots >= rate.shape[0] - horizon):
        raise ValueError('source_indices must be valid one-dimensional future-window indices')
    z_mbps = np.empty((len(source_slots), rate.shape[2]), dtype=float)
    if source_indices is not None:
        # Prepared experiments select common indices; bounded chunks avoid materializing all windows.
        offsets = np.arange(1, horizon + 1)
        for start in range(0, len(source_slots), 256):
            selected = source_slots[start:start + 256]
            windows = shared[selected[:, None] + offsets[None, :]]
            z_mbps[start:start + len(selected)] = np.quantile(
                windows, time_quantile, axis=1, method=quantile_method)
        return z_mbps, source_slots
    for row, source in enumerate(source_slots):
        future = shared[source + 1 : source + horizon + 1]
        z_mbps[row] = np.quantile(
            future,
            time_quantile,
            axis=0,
            method=quantile_method,
        )
    return z_mbps, source_slots


def fit_reference_rate(
    train_z_mbps: Iterable[np.ndarray] | np.ndarray,
    *,
    quantile: float = 0.95,
    quantile_method: str = 'linear',
) -> float:
    if isinstance(train_z_mbps, np.ndarray):
        arrays = [np.asarray(train_z_mbps, dtype=float)]
    else:
        arrays = [np.asarray(x, dtype=float) for x in train_z_mbps]
    if not arrays or any(values.size == 0 for values in arrays):
        raise ValueError('training z values must be non-empty')
    flattened = np.concatenate([values.reshape(-1) for values in arrays])
    if not np.all(np.isfinite(flattened)):
        raise ValueError('training z values must be finite')
    reference = float(
        np.quantile(flattened, quantile, method=quantile_method)
    )
    if reference <= 0.0:
        raise ValueError('R_ref must be positive')
    return reference


def normalize_dqi(z_mbps: np.ndarray, r_ref_mbps: float) -> np.ndarray:
    reference = float(r_ref_mbps)
    if not np.isfinite(reference) or reference <= 0.0:
        raise ValueError('r_ref_mbps must be finite and positive')
    return np.clip(np.asarray(z_mbps, dtype=float) / reference, 0.0, 1.0)


def build_dqi_labels(
    rate_mbps: np.ndarray,
    *,
    r_ref_mbps: float,
    **kwargs: object,
) -> DQILabels:
    z_mbps, source_slots = compute_unnormalized_dqi(
        rate_mbps,
        **kwargs,
    )
    return DQILabels(
        z_mbps=z_mbps,
        y=normalize_dqi(z_mbps, r_ref_mbps),
        source_slots=source_slots,
        r_ref_mbps=float(r_ref_mbps),
    )


def channel_descriptors(
    snr_db: np.ndarray,
    rss_dbm: np.ndarray,
    bandwidth_mhz: np.ndarray,
    cci: np.ndarray,
    *,
    vehicle_quantile: float = 0.10,
    quantile_method: str = 'linear',
) -> np.ndarray:
    snr = np.asarray(snr_db, dtype=float)
    rss = np.asarray(rss_dbm, dtype=float)
    bandwidth = np.asarray(bandwidth_mhz, dtype=float)
    coherence = np.asarray(cci, dtype=float)
    if snr.ndim != 3 or rss.shape != snr.shape:
        raise ValueError('snr_db and rss_dbm require [time, vehicle, channel]')
    expected = (snr.shape[0], snr.shape[2])
    if bandwidth.shape != expected or coherence.shape != expected:
        raise ValueError('bandwidth_mhz and cci require [time, channel]')
    snr_shared = np.quantile(
        snr, vehicle_quantile, axis=1, method=quantile_method
    )
    rss_shared = np.quantile(
        rss, vehicle_quantile, axis=1, method=quantile_method
    )
    return np.stack(
        [snr_shared, rss_shared, bandwidth, coherence],
        axis=-1,
    )


def fit_feature_scaler(
    train_descriptors: Iterable[np.ndarray] | np.ndarray,
    *,
    feature_names: tuple[str, ...] = (
        'snr_q10_db',
        'rss_q10_dbm',
        'bandwidth_mhz',
        'cci',
    ),
) -> FeatureScaler:
    if isinstance(train_descriptors, np.ndarray):
        arrays = [np.asarray(train_descriptors, dtype=float)]
    else:
        arrays = [np.asarray(x, dtype=float) for x in train_descriptors]
    if not arrays:
        raise ValueError('training descriptors must be non-empty')
    flattened = np.concatenate(
        [x.reshape(-1, x.shape[-1]) for x in arrays],
        axis=0,
    )
    if flattened.shape[1] != len(feature_names):
        raise ValueError('feature_names do not match descriptor columns')
    mean = np.mean(flattened, axis=0)
    scale = np.std(flattened, axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    return FeatureScaler(
        feature_names=feature_names,
        mean=mean,
        scale=scale,
        sample_count=int(flattened.shape[0]),
    )
