from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np

from cvn_rl.simulation.traffic import generate_occupancy


@dataclass(frozen=True)
class AccessTrace:
    occupancy: np.ndarray
    sensing_idle: np.ndarray
    fresh_dqi: np.ndarray
    dqi_age: np.ndarray
    theta: np.ndarray
    quality_mbps: np.ndarray
    power_cap: np.ndarray
    occupancy_type: np.ndarray
    su_pu_gain_hat: np.ndarray
    su_pu_gain_true: np.ndarray
    pu_threshold_mw: np.ndarray
    transmit_power_cap_mw: np.ndarray

    @property
    def slots(self) -> int:
        return int(self.occupancy.shape[0])

    @property
    def num_channels(self) -> int:
        return int(self.occupancy.shape[1])

    def trace_hash(self) -> str:
        digest = hashlib.sha256()
        for array in (
            self.occupancy,
            self.sensing_idle,
            self.fresh_dqi,
            self.theta,
            self.power_cap,
            self.su_pu_gain_hat,
            self.su_pu_gain_true,
            self.pu_threshold_mw,
        ):
            digest.update(np.ascontiguousarray(array).view(np.uint8))
        return digest.hexdigest()[:16]


def generate_trace(config: dict[str, Any], seed: int, slots: int | None = None) -> AccessTrace:
    rng = np.random.default_rng(seed)
    total_slots = int(slots or config.get("slots", 5000))
    num_channels = int(config.get("num_channels", 10))
    traffic = dict(config.get("traffic", {}))
    occupancy = generate_occupancy(total_slots, num_channels, traffic, rng)
    sensing_idle = _generate_sensing_idle(
        occupancy,
        pd=float(traffic.get("sensing_pd", 0.9)),
        pfa=float(traffic.get("sensing_pfa", 0.1)),
        rng=rng,
    )
    quality = _generate_quality(total_slots, num_channels, config, rng)
    from cvn_rl.simulation.multi_traces import _latency_constrained_goodput
    dqi_cfg = config.get('dqi', {})
    goodput = _latency_constrained_goodput(
        quality,
        packet_size_bits=float(dqi_cfg.get('packet_size_bits', 32_000)),
        delay_max_ms=float(dqi_cfg.get('delay_max_ms', 5.0)),
        delay_beta_ms=float(dqi_cfg.get('delay_beta_ms', 2.0)),
    )
    raw_dqi = np.clip(goodput / max(float(dqi_cfg.get('r_ref_mbps', 80)), 1e-6), 0.0, 1.0)
    dqi_age = _generate_age(total_slots, num_channels, int(config.get("underlay_power", {}).get("broadcast_update_slots", 10)))
    validity = float(config.get("dqi", {}).get("validity", {}).get("t_max_slots", 20))
    neutral = float(config.get("state", {}).get("freshness_neutral_value", 0.5))
    discount = np.exp(-dqi_age / max(validity, 1.0))
    fresh_dqi = np.clip(raw_dqi * discount + neutral * (1.0 - discount), 0.0, 1.0)
    theta = _generate_theta(occupancy, traffic, rng)
    link = _generate_pu_link_descriptors(total_slots, num_channels, config, rng)
    return AccessTrace(
        occupancy=occupancy,
        sensing_idle=sensing_idle,
        fresh_dqi=fresh_dqi,
        dqi_age=dqi_age,
        theta=theta,
        quality_mbps=quality,
        power_cap=link["power_cap"],
        occupancy_type=occupancy.astype(float),
        su_pu_gain_hat=link["su_pu_gain_hat"],
        su_pu_gain_true=link["su_pu_gain_true"],
        pu_threshold_mw=link["pu_threshold_mw"],
        transmit_power_cap_mw=link["transmit_power_cap_mw"],
    )


def recompute_underlay_power_cap(
    su_pu_gain_hat: np.ndarray,
    pu_threshold_mw: np.ndarray,
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    p_max = float(config.get("underlay_power", {}).get("p_max_mw", 1.0))
    margin = float(config.get("underlay_power", {}).get("fading_margin", 2.0))
    denom = np.maximum(su_pu_gain_hat * max(margin, 1e-6), 1e-9)
    transmit_cap = np.minimum(p_max, pu_threshold_mw / denom)
    power_cap = np.clip(transmit_cap / max(p_max, 1e-9), 0.0, 1.0)
    return power_cap, transmit_cap


def _generate_sensing_idle(
    occupancy: np.ndarray,
    pd: float,
    pfa: float,
    rng: np.random.Generator,
) -> np.ndarray:
    sensed_busy = np.where(occupancy, rng.random(occupancy.shape) < pd, rng.random(occupancy.shape) < pfa)
    return (~sensed_busy).astype(float)


def _generate_quality(
    slots: int,
    num_channels: int,
    config: dict[str, Any],
    rng: np.random.Generator,
) -> np.ndarray:
    snr_low, snr_high = config.get("channel", {}).get("snr_db_range", [-5, 30])
    bandwidths = np.array(config.get("channel", {}).get("bandwidth_mhz", [5, 10, 20]), dtype=float)
    bw = rng.choice(bandwidths, size=num_channels)
    snr = rng.uniform(float(snr_low), float(snr_high), size=num_channels)
    quality = np.zeros((slots, num_channels), dtype=float)
    for t in range(slots):
        snr = 0.97 * snr + 0.03 * rng.uniform(float(snr_low), float(snr_high), size=num_channels)
        fading = rng.normal(0.0, 1.0, size=num_channels)
        linear = np.log2(1.0 + np.power(10.0, (snr + fading) / 10.0))
        quality[t] = np.clip(bw * linear, 1.0, 180.0)
    return quality


def _generate_age(slots: int, num_channels: int, update_slots: int) -> np.ndarray:
    update = max(int(update_slots), 1)
    age = (np.arange(slots) % update).astype(float)
    return np.repeat(age[:, None], num_channels, axis=1)


def _generate_theta(
    occupancy: np.ndarray,
    traffic: dict[str, Any],
    rng: np.random.Generator,
) -> np.ndarray:
    next_idle = np.vstack([~occupancy[1:], ~occupancy[-1:]]).astype(float)
    p01 = float(traffic.get("p01", 0.2))
    p10 = float(traffic.get("p10", 0.25))
    markov_prior = np.where(occupancy, p10, 1.0 - p01)
    theta = 0.65 * next_idle + 0.35 * markov_prior + rng.normal(0.0, 0.08, size=occupancy.shape)
    return np.clip(theta, 0.0, 1.0)


def _generate_pu_link_descriptors(
    slots: int,
    num_channels: int,
    config: dict[str, Any],
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    true_gain = rng.lognormal(mean=-1.2, sigma=0.35, size=(slots, num_channels))
    estimate_error = rng.lognormal(mean=0.0, sigma=0.18, size=(slots, num_channels))
    gain_hat = true_gain / estimate_error
    threshold = rng.uniform(0.10, 0.55, size=(slots, num_channels))
    power_cap, transmit_cap = recompute_underlay_power_cap(gain_hat, threshold, config)
    return {
        "su_pu_gain_hat": gain_hat,
        "su_pu_gain_true": true_gain,
        "pu_threshold_mw": threshold,
        "power_cap": power_cap,
        "transmit_power_cap_mw": transmit_cap,
    }
