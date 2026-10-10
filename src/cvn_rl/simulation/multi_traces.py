from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

import numpy as np

from cvn_rl.simulation.dqi_labels import (
    DQI_LABEL_VERSION,
    compute_unnormalized_dqi,
    latency_constrained_goodput,
    normalize_dqi,
)
from cvn_rl.simulation.traffic import generate_occupancy


@dataclass(frozen=True)
class MultiVehicleAccessTrace:
    occupancy: np.ndarray
    sensing_idle: np.ndarray
    dqi_predicted: np.ndarray
    fresh_dqi: np.ndarray
    dqi_age: np.ndarray
    dqi_validity_slots: np.ndarray
    dqi_source_slot: np.ndarray
    dqi_input_cutoff_slot: np.ndarray
    theta: np.ndarray
    theta_input_cutoff_slot: np.ndarray
    theta_target_slot: np.ndarray
    quality_mbps: np.ndarray
    quality_lookahead_mbps: np.ndarray
    snr_db: np.ndarray
    rss_dbm: np.ndarray
    cci: np.ndarray
    future_dqi_oracle: np.ndarray
    power_cap: np.ndarray
    occupancy_type: np.ndarray
    su_pu_gain_hat: np.ndarray
    su_pu_gain_true: np.ndarray
    pu_threshold_mw: np.ndarray
    transmit_power_cap_mw: np.ndarray
    packet_arrivals: np.ndarray
    bandwidth_mhz: np.ndarray
    pu_signal_mw: np.ndarray
    noise_mw: np.ndarray
    contention_priority: np.ndarray
    trace_id: str
    split: str
    seed: int
    dqi_model_id: str
    dqi_model_hash: str
    prediction_model_id: str
    prediction_model_hash: str
    dqi_inference_latency_ms: float
    dqi_parameter_count: int
    prediction_inference_latency_ms: float
    prediction_parameter_count: int
    dqi_validation_mse: float
    prediction_validation_brier: float
    dqi_metadata: dict[str, Any]

    @property
    def slots(self) -> int:
        return int(self.occupancy.shape[0])

    @property
    def num_channels(self) -> int:
        return int(self.occupancy.shape[1])

    @property
    def num_vehicles(self) -> int:
        return int(self.sensing_idle.shape[1])

    def trace_hash(self) -> str:
        digest = hashlib.sha256()
        for array in (
            self.occupancy,
            self.sensing_idle,
            self.dqi_predicted,
            self.fresh_dqi,
            self.dqi_source_slot,
            self.theta,
            self.theta_input_cutoff_slot,
            self.quality_mbps,
            self.quality_lookahead_mbps,
            self.snr_db,
            self.rss_dbm,
            self.cci,
            self.future_dqi_oracle,
            self.power_cap,
            self.su_pu_gain_hat,
            self.su_pu_gain_true,
            self.pu_threshold_mw,
            self.packet_arrivals,
            self.pu_signal_mw,
            self.contention_priority,
        ):
            digest.update(np.ascontiguousarray(array).view(np.uint8))
        return digest.hexdigest()[:16]

    def physical_trace_hash(self) -> str:
        """Hash environment randomness while excluding DQI/theta variants."""

        digest = hashlib.sha256()
        for array in (
            self.occupancy,
            self.sensing_idle,
            self.quality_mbps,
            self.quality_lookahead_mbps,
            self.power_cap,
            self.su_pu_gain_hat,
            self.su_pu_gain_true,
            self.pu_threshold_mw,
            self.transmit_power_cap_mw,
            self.packet_arrivals,
            self.bandwidth_mhz,
            self.snr_db,
            self.rss_dbm,
            self.cci,
            self.pu_signal_mw,
            self.noise_mw,
            self.contention_priority,
        ):
            digest.update(np.ascontiguousarray(array).view(np.uint8))
        return digest.hexdigest()[:16]


def generate_multi_vehicle_trace(
    config: dict[str, Any],
    seed: int,
    slots: int | None = None,
    split: str = 'rl_train',
) -> MultiVehicleAccessTrace:
    rng = np.random.default_rng(seed)
    requested_slots = int(slots or config.get('slots', 5000))
    descriptor_cfg = dict(config.get('descriptors', {}))
    total_slots = int(slots or config.get("slots", 5000))
    hidden_lookahead_slots = max(int(config.get('hidden_lookahead_slots', 0)), 0)
    num_channels = int(config.get("num_channels", 10))
    num_vehicles = int(config.get("num_vehicles", 1))
    traffic = dict(config.get("traffic", {}))

    occupancy = generate_occupancy(total_slots, num_channels, traffic, rng)
    sensing_idle = _generate_vehicle_sensing(occupancy, num_vehicles, traffic, rng)
    bandwidth_static, quality_with_lookahead = _generate_vehicle_quality(
        total_slots + hidden_lookahead_slots,
        num_vehicles,
        num_channels,
        config,
        rng,
    )
    quality = quality_with_lookahead[:total_slots]
    quality_lookahead = quality_with_lookahead[total_slots:]
    bandwidth_all, snr_all, rss_all, cci_all = _materialize_channel_descriptors(
        quality_with_lookahead,
        bandwidth_static,
        config,
    )
    bandwidth_mhz = bandwidth_all[:total_slots]
    snr_db = snr_all[:total_slots]
    rss_dbm = rss_all[:total_slots]
    cci = cci_all[:total_slots]
    neutral = float(config.get("state", {}).get("freshness_neutral_value", 0.5))
    if bool(descriptor_cfg.get('skip_causal_dqi', False)):
        dqi_payload = _empty_shared_dqi(total_slots, num_channels)
    else:
        dqi_payload = _generate_causal_dqi(quality, config, neutral)
    dqi_predicted = dqi_payload['predicted']
    fresh_dqi = dqi_payload['fresh']
    dqi_age = dqi_payload['age']
    dqi_validity_slots = dqi_payload['validity_slots']
    dqi_source_slot = dqi_payload['source_slot']
    dqi_input_cutoff_slot = dqi_payload['input_cutoff_slot']
    theta, theta_input_cutoff_slot, theta_target_slot = _generate_causal_theta(
        occupancy,
        traffic,
        descriptor_cfg,
        rng,
    )

    links = _generate_vehicle_pu_links(total_slots, num_vehicles, num_channels, config, rng)
    packet_cfg = config.get("packet_traffic", {})
    arrival_probability = float(packet_cfg.get("arrival_probability", 0.5))
    packet_arrivals = rng.random((total_slots, num_vehicles)) < arrival_probability
    pu_signal_mw = rng.lognormal(mean=-0.1, sigma=0.2, size=(total_slots, num_channels))
    noise_floor = float(config.get("channel", {}).get("noise_mw", 0.01))
    noise_mw = np.full((total_slots, num_channels), noise_floor, dtype=float)
    contention_priority = rng.random((total_slots, num_vehicles, num_channels))
    trace_id = _trace_id(config, split, seed, requested_slots)
    dqi_model_id = str(descriptor_cfg.get('dqi_model_id', 'causal_quantile_dqi_surrogate'))
    prediction_model_id = str(
        descriptor_cfg.get('prediction_model_id', 'causal_attention_lstm_surrogate')
    )

    return MultiVehicleAccessTrace(
        occupancy=occupancy,
        sensing_idle=sensing_idle,
        dqi_predicted=dqi_predicted,
        fresh_dqi=fresh_dqi,
        dqi_age=dqi_age,
        dqi_validity_slots=dqi_validity_slots,
        dqi_source_slot=dqi_source_slot,
        dqi_input_cutoff_slot=dqi_input_cutoff_slot,
        theta=theta,
        theta_input_cutoff_slot=theta_input_cutoff_slot,
        theta_target_slot=theta_target_slot,
        quality_mbps=quality,
        quality_lookahead_mbps=quality_lookahead,
        snr_db=snr_db,
        rss_dbm=rss_dbm,
        cci=cci,
        future_dqi_oracle=np.full(
            (total_slots, num_channels), np.nan, dtype=float
        ),
        power_cap=links["power_cap"],
        occupancy_type=occupancy.astype(float),
        su_pu_gain_hat=links["su_pu_gain_hat"],
        su_pu_gain_true=links["su_pu_gain_true"],
        pu_threshold_mw=links["pu_threshold_mw"],
        transmit_power_cap_mw=links["transmit_power_cap_mw"],
        packet_arrivals=packet_arrivals,
        bandwidth_mhz=bandwidth_mhz,
        pu_signal_mw=pu_signal_mw,
        noise_mw=noise_mw,
        contention_priority=contention_priority,
        trace_id=trace_id,
        split=split,
        seed=int(seed),
        dqi_model_id=dqi_model_id,
        dqi_model_hash=str(
            descriptor_cfg.get('dqi_model_hash', _surrogate_hash(dqi_model_id, config))
        ),
        prediction_model_id=prediction_model_id,
        prediction_model_hash=str(
            descriptor_cfg.get(
                'prediction_model_hash',
                _surrogate_hash(prediction_model_id, config),
            )
        ),
        dqi_inference_latency_ms=float(descriptor_cfg.get('dqi_inference_latency_ms', 0.0)),
        dqi_parameter_count=int(descriptor_cfg.get('dqi_parameter_count', 0)),
        prediction_inference_latency_ms=float(
            descriptor_cfg.get('prediction_inference_latency_ms', 0.0)
        ),
        prediction_parameter_count=int(descriptor_cfg.get('prediction_parameter_count', 0)),
        dqi_validation_mse=float(descriptor_cfg.get('dqi_validation_mse', np.nan)),
        prediction_validation_brier=float(
            descriptor_cfg.get('prediction_validation_brier', np.nan)
        ),
        dqi_metadata=_dqi_metadata(config, split),
    )


def _generate_vehicle_sensing(
    occupancy: np.ndarray,
    num_vehicles: int,
    traffic: dict[str, Any],
    rng: np.random.Generator,
) -> np.ndarray:
    busy = np.repeat(occupancy[:, None, :], num_vehicles, axis=1)
    pd = float(traffic.get("sensing_pd", 0.9))
    pfa = float(traffic.get("sensing_pfa", 0.1))
    draws = rng.random(busy.shape)
    sensed_busy = np.where(busy, draws < pd, draws < pfa)
    return (~sensed_busy).astype(float)


def _generate_vehicle_quality(
    slots: int,
    num_vehicles: int,
    num_channels: int,
    config: dict[str, Any],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    channel_cfg = config.get("channel", {})
    quality_family = str(channel_cfg.get("quality_family", "ar1"))

    if quality_family == "smooth_semi_markov":
        return generate_smooth_semi_markov_quality(
            slots, num_vehicles, num_channels, config, rng
        )

    if quality_family == "piecewise_latent":
        return _generate_piecewise_latent_quality(
            slots, num_vehicles, num_channels, config, rng
        )

    snr_low, snr_high = channel_cfg.get("snr_db_range", [-5, 30])
    bandwidths = np.asarray(channel_cfg.get("bandwidth_mhz", [5, 10, 20]), dtype=float)
    bandwidth_mhz = rng.choice(bandwidths, size=num_channels)
    configured_rho = channel_cfg.get('ar_coefficients')
    rho: np.ndarray | None = None
    if configured_rho is not None:
        rho = np.asarray(configured_rho, dtype=float)
        if rho.shape != (num_channels,):
            raise ValueError(
                'channel.ar_coefficients must contain one value per channel'
            )
        if np.any((rho < 0.0) | (rho >= 1.0)):
            raise ValueError('channel AR coefficients must lie in [0, 1)')
        mean_snr = (float(snr_low) + float(snr_high)) / 2.0
        stationary_std = (float(snr_high) - float(snr_low)) / np.sqrt(12.0)
        snr = rng.normal(
            mean_snr,
            stationary_std,
            size=(num_vehicles, num_channels),
        )
    else:
        snr = rng.uniform(float(snr_low), float(snr_high), size=(num_vehicles, num_channels))
    vehicle_offset = rng.normal(0.0, 2.0, size=(num_vehicles, 1))
    quality = np.empty((slots, num_vehicles, num_channels), dtype=float)
    for t in range(slots):
        if rho is None:
            innovations = rng.uniform(float(snr_low), float(snr_high), size=snr.shape)
            snr = 0.97 * snr + 0.03 * innovations
        else:
            innovations = rng.normal(0.0, stationary_std, size=snr.shape)
            snr = (
                mean_snr
                + rho[None, :] * (snr - mean_snr)
                + np.sqrt(1.0 - rho[None, :] ** 2) * innovations
            )
            snr = np.clip(snr, float(snr_low), float(snr_high))
        fading = rng.normal(0.0, 1.0, size=snr.shape)
        spectral_efficiency = np.log2(1.0 + np.power(10.0, (snr + vehicle_offset + fading) / 10.0))
        quality[t] = np.clip(bandwidth_mhz[None, :] * spectral_efficiency, 0.1, 180.0)
    return bandwidth_mhz, quality


def generate_smooth_semi_markov_quality(
    slots: int,
    num_vehicles: int,
    num_channels: int,
    config: dict[str, Any],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate equal-bandwidth links with smooth semi-Markov SNR regimes.

    All channels use the same bad-medium-good-medium cycle and duration laws,
    but have independent phases and dwell draws. Vehicle shadowing evolves per
    link, so no vehicle is permanently the weakest one. The function is public
    so the D0.5 audit can avoid constructing unrelated PU/queue arrays.
    """

    if slots <= 0 or num_vehicles <= 0 or num_channels <= 1:
        raise ValueError('smooth_semi_markov requires T>0, V>0, and M>1')
    channel_cfg = dict(config.get('channel', {}))
    state_snr = np.asarray(
        channel_cfg.get('semi_markov_state_snr_db', [-6.0, 0.0, 8.0]),
        dtype=float,
    )
    if state_snr.shape != (3,):
        raise ValueError('semi_markov_state_snr_db must contain bad/medium/good')
    bad_dwell = tuple(
        int(value)
        for value in channel_cfg.get('semi_markov_bad_dwell_slots', [60, 100])
    )
    other_dwell = tuple(
        int(value)
        for value in channel_cfg.get(
            'semi_markov_medium_good_dwell_slots', [120, 200]
        )
    )
    if (
        len(bad_dwell) != 2
        or len(other_dwell) != 2
        or bad_dwell[0] <= 0
        or other_dwell[0] <= 0
        or bad_dwell[1] < bad_dwell[0]
        or other_dwell[1] < other_dwell[0]
    ):
        raise ValueError('semi-Markov dwell ranges must be positive [low, high]')
    transition_slots = max(
        int(channel_cfg.get('semi_markov_transition_slots', 30)), 1
    )
    cycle = np.asarray([0, 1, 2, 1], dtype=int)
    latent_snr = np.empty((slots, num_channels), dtype=float)

    for channel in range(num_channels):
        warmup = 2 * (other_dwell[1] + transition_slots)
        required = slots + warmup
        values: list[np.ndarray] = []
        produced = 0
        cycle_position = int(rng.integers(0, len(cycle)))
        while produced < required:
            state = int(cycle[cycle_position])
            next_position = (cycle_position + 1) % len(cycle)
            next_state = int(cycle[next_position])
            dwell_range = bad_dwell if state == 0 else other_dwell
            dwell = int(rng.integers(dwell_range[0], dwell_range[1] + 1))
            stable = np.full(dwell, state_snr[state], dtype=float)
            fraction = np.arange(1, transition_slots + 1, dtype=float) / transition_slots
            transition = (
                state_snr[state]
                + fraction * (state_snr[next_state] - state_snr[state])
            )
            values.extend((stable, transition))
            produced += dwell + transition_slots
            cycle_position = next_position
        sequence = np.concatenate(values)
        crop = int(rng.integers(0, warmup + 1))
        latent_snr[:, channel] = sequence[crop : crop + slots]

    bandwidth_value = float(channel_cfg.get('fixed_bandwidth_mhz', 10.0))
    bandwidth_mhz = np.full(num_channels, bandwidth_value, dtype=float)
    shadow_rho = float(channel_cfg.get('shadowing_ar_coefficient', 0.995))
    shadow_std = float(channel_cfg.get('shadowing_stationary_std_db', 2.0))
    fading_std = float(channel_cfg.get('fast_fading_std_db', 1.0))
    if not 0.0 <= shadow_rho < 1.0:
        raise ValueError('shadowing_ar_coefficient must lie in [0, 1)')
    shadow = rng.normal(
        0.0, shadow_std, size=(num_vehicles, num_channels)
    )
    innovation_scale = shadow_std * np.sqrt(1.0 - shadow_rho**2)
    quality = np.empty((slots, num_vehicles, num_channels), dtype=float)
    for t in range(slots):
        shadow = shadow_rho * shadow + rng.normal(
            0.0, innovation_scale, size=shadow.shape
        )
        fading = rng.normal(0.0, fading_std, size=shadow.shape)
        link_snr = latent_snr[t][None, :] + shadow + fading
        spectral_efficiency = np.log2(
            1.0 + np.power(10.0, link_snr / 10.0)
        )
        quality[t] = np.clip(
            bandwidth_mhz[None, :] * spectral_efficiency,
            0.01,
            180.0,
        )
    return bandwidth_mhz, quality


def _generate_piecewise_latent_quality(
    slots: int,
    num_vehicles: int,
    num_channels: int,
    config: dict[str, Any],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate channel quality with piecewise-latent structure for DQI experiments.

    Creates M latent quality profiles with different baseline SNRs and high AR
    coefficients (0.98-0.995). Every ``quality_block_slots``, the mapping from
    profiles to physical channels is randomly permuted, ensuring no single
    channel is permanently the best. This forces the RL agent to use DQI
    information for channel selection rather than memorizing fixed channel
    identities.

    Design guarantees:
    - M profiles with distinct baseline SNRs spread across the configured range
    - High AR coefficients ensure slow intra-block quality variation
    - Block-boundary permutation ensures at least 3 different channels become
      oracle top-1 over a 100k-slot episode
    - Sufficient instantaneous quality overlap prevents trivial CQI-based ranking
    """
    channel_cfg = config.get("channel", {})
    snr_low, snr_high = channel_cfg.get("snr_db_range", [-5, 30])
    bandwidths = np.asarray(channel_cfg.get("bandwidth_mhz", [5, 10, 20]), dtype=float)
    bandwidth_mhz = rng.choice(bandwidths, size=num_channels)

    block_slots = max(int(channel_cfg.get("quality_block_slots", 25_000)), 1)
    ar_low, ar_high = channel_cfg.get("ar_coefficient_range", [0.98, 0.995])

    # Create M latent quality profiles with distinct baseline SNRs
    snr_baseline_low = float(channel_cfg.get("snr_db_baseline_range", [snr_low, snr_high])[0])
    snr_baseline_high = float(channel_cfg.get("snr_db_baseline_range", [snr_low, snr_high])[1])
    baseline_snr = np.linspace(snr_baseline_low, snr_baseline_high, num_channels)
    rng.shuffle(baseline_snr)

    # Each profile gets its own AR coefficient within [ar_low, ar_high]
    ar_coeffs = rng.uniform(ar_low, ar_high, size=num_channels)

    stationary_std = (float(snr_high) - float(snr_low)) / np.sqrt(12.0)

    quality = np.empty((slots, num_vehicles, num_channels), dtype=float)

    # Per-(vehicle, channel) SNR state, initialized near baseline with small jitter
    snr_state = baseline_snr[None, :] + rng.normal(
        0.0, stationary_std * 0.02, size=(num_vehicles, num_channels)
    )

    num_blocks = int(np.ceil(slots / block_slots))

    for block in range(num_blocks):
        block_start = block * block_slots
        block_end = min(block_start + block_slots, slots)

        # At block boundaries, randomly permute profile→channel mapping.
        # This rotates which physical channel hosts each quality profile,
        # preventing the agent from memorizing "channel X is always best."
        if block > 0:
            perm = rng.permutation(num_channels)
            snr_state = snr_state[:, perm]

        for t in range(block_start, block_end):
            innovations = rng.normal(0.0, stationary_std, size=snr_state.shape)
            snr_state = (
                baseline_snr[None, :]
                + ar_coeffs[None, :] * (snr_state - baseline_snr[None, :])
                + np.sqrt(1.0 - ar_coeffs[None, :] ** 2) * innovations
            )
            snr_state = np.clip(snr_state, float(snr_low), float(snr_high))

            vehicle_offset = rng.normal(0.0, 2.0, size=(num_vehicles, 1))
            fading = rng.normal(0.0, 1.0, size=snr_state.shape)
            spectral_efficiency = np.log2(
                1.0 + np.power(10.0, (snr_state + vehicle_offset + fading) / 10.0)
            )
            quality[t] = np.clip(
                bandwidth_mhz[None, :] * spectral_efficiency, 0.1, 180.0
            )

    return bandwidth_mhz, quality


def _materialize_channel_descriptors(
    quality_mbps: np.ndarray,
    bandwidth_static: np.ndarray,
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    slots, _, channels = quality_mbps.shape
    bandwidth = np.repeat(
        np.asarray(bandwidth_static, dtype=float)[None, :],
        slots,
        axis=0,
    )
    if bandwidth.shape != (slots, channels):
        raise ValueError('bandwidth descriptor has an invalid shape')
    efficiency = quality_mbps / np.maximum(bandwidth[:, None, :], 1e-9)
    snr_linear = np.maximum(np.exp2(efficiency) - 1.0, 1e-12)
    snr_db = 10.0 * np.log10(snr_linear)
    noise_dbm = float(config.get('channel', {}).get('noise_floor_dbm', -95.0))
    rss_dbm = snr_db + noise_dbm
    shared_snr = np.quantile(snr_db, 0.10, axis=1, method='linear')
    change = np.zeros_like(shared_snr)
    change[1:] = np.abs(np.diff(shared_snr, axis=0))
    cci = np.exp(-change / 10.0)
    return bandwidth, snr_db, rss_dbm, cci


def _latency_constrained_goodput(
    quality_mbps: np.ndarray,
    packet_size_bits: float = 32_000.0,
    delay_max_ms: float = 5.0,
    delay_beta_ms: float = 2.0,
) -> np.ndarray:
    """Convert raw throughput to latency-constrained goodput.

    Paper Eq.(8)-(9):
      D       = packet_size_bits / (R * 1e6)          [transmission delay, s]
      p_delay = sigma((D*1000 - D_max) / beta)        [delay-violation prob.]
      goodput = R * (1 - p_delay)                     [latency-constrained]

    Low-rate channels incur high transmission delay, high p_delay, and
    near-zero goodput — a nonlinear penalty that ``current_cqi`` (which
    only observes raw rate) cannot replicate.  This is the mechanism that
    makes DQI *structurally* different from current_cqi.
    """
    return latency_constrained_goodput(
        quality_mbps,
        packet_size_bits=packet_size_bits,
        delay_max_ms=delay_max_ms,
        delay_beta_ms=delay_beta_ms,
    )


def _dqi_validity_from_history(
    history: np.ndarray,
    num_channels: int,
    minimum_validity: float,
    maximum_validity: float,
) -> np.ndarray:
    if len(history) > 1:
        link_change = np.mean(
            np.abs(np.diff(history, axis=0)) / np.maximum(history[:-1], 1.0),
            axis=0,
        )
        relative_change = np.mean(link_change, axis=0)
        cci = np.exp(-3.0 * relative_change)
    else:
        cci = np.ones(num_channels, dtype=float)
    return minimum_validity + (maximum_validity - minimum_validity) * cci


def _canonical_dqi_labels(
    quality: np.ndarray,
    descriptor_cfg: dict[str, Any],
    dqi_cfg: dict[str, Any],
    r_ref: float,
) -> np.ndarray:
    horizon = max(int(descriptor_cfg.get('dqi_horizon_slots', 100)), 1)
    if quality.shape[0] <= horizon:
        return np.empty((0, quality.shape[2]), dtype=float)
    z_mbps, _ = compute_unnormalized_dqi(
        quality,
        horizon_samples=horizon,
        vehicle_quantile=float(descriptor_cfg.get('dqi_vehicle_quantile', descriptor_cfg.get('dqi_quantile', 0.10))),
        time_quantile=float(descriptor_cfg.get('dqi_time_quantile', descriptor_cfg.get('dqi_quantile', 0.10))),
        quantile_method=str(descriptor_cfg.get('quantile_method', 'linear')),
        packet_size_bits=float(descriptor_cfg.get('packet_size_bits', 32_000)),
        delay_max_ms=float(descriptor_cfg.get('dqi_delay_max_ms', dqi_cfg.get('delay_max_ms', 5.0))),
        delay_beta_ms=float(descriptor_cfg.get('dqi_delay_beta_ms', dqi_cfg.get('delay_beta_ms', 2.0))),
    )
    return normalize_dqi(z_mbps, r_ref)


def _generate_causal_dqi(
    quality: np.ndarray,
    config: dict[str, Any],
    neutral: float,
) -> dict[str, np.ndarray]:
    descriptor_cfg = config.get('descriptors', {})
    dqi_cfg = config.get('dqi', {})
    validity_cfg = dqi_cfg.get('validity', {})
    update_slots = max(int(descriptor_cfg.get('dqi_update_slots', 1_000)), 1)
    history_slots = max(int(descriptor_cfg.get('dqi_history_slots', 60_000)), 1)
    horizon = max(int(descriptor_cfg.get('dqi_horizon_slots', 100)), 1)
    minimum_validity = max(float(validity_cfg.get('t_min_slots', 100)), 1.0)
    maximum_validity = max(
        float(validity_cfg.get('t_max_slots', 1_000)),
        minimum_validity,
    )
    r_ref = max(float(dqi_cfg.get('r_ref_mbps', 80)), 1e-6)
    slots, _, num_channels = quality.shape
    predicted = np.empty((slots, num_channels), dtype=float)
    age = np.empty((slots, num_channels), dtype=float)
    validity = np.empty((slots, num_channels), dtype=float)
    source_slot = np.empty(slots, dtype=int)
    input_cutoff_slot = np.empty(slots, dtype=int)
    teacher = str(
        descriptor_cfg.get('dqi_causal_teacher', 'historical_tail_quantile')
    )

    canonical_labels = None
    if teacher in {'canonical_persistence', 'canonical_future_persistence'}:
        canonical_labels = _canonical_dqi_labels(
            quality, descriptor_cfg, dqi_cfg, r_ref,
        )
    apply_discount = bool(descriptor_cfg.get('dqi_apply_freshness_discount', True))

    for source in range(0, slots, update_slots):
        stop = min(source + update_slots, slots)
        needs_history = canonical_labels is None or apply_discount
        if needs_history:
            history_start = max(0, source - history_slots + 1)
            history = quality[history_start : source + 1]
        else:
            history = None
        if canonical_labels is not None:
            if source >= horizon and len(canonical_labels) > 0:
                label_idx = min(source - horizon, len(canonical_labels) - 1)
                score = canonical_labels[label_idx]
            else:
                score = np.full(num_channels, neutral, dtype=float)
        else:
            assert history is not None
            goodput = _latency_constrained_goodput(
                history,
                packet_size_bits=float(
                    dqi_cfg.get('packet_size_bits', 32_000)
                ),
                delay_max_ms=float(dqi_cfg.get('delay_max_ms', 5.0)),
                delay_beta_ms=float(dqi_cfg.get('delay_beta_ms', 2.0)),
            )
            shared = np.quantile(
                goodput, 0.10, axis=1, method='linear'
            )
            score = np.quantile(
                shared, 0.10, axis=0, method='linear'
            ) / r_ref
        if history is None:
            valid = np.full(num_channels, maximum_validity, dtype=float)
        else:
            valid = _dqi_validity_from_history(
                history, num_channels, minimum_validity, maximum_validity,
            )
        predicted[source:stop] = np.clip(score, 0.0, 1.0)
        age[source:stop] = np.arange(stop - source, dtype=float)[:, None]
        validity[source:stop] = valid
        source_slot[source:stop] = source
        input_cutoff_slot[source:stop] = source

    if apply_discount:
        discount = np.exp(-age / np.maximum(validity, 1.0))
        fresh = predicted * discount
        fresh = np.where(age <= validity, fresh, 0.0)
    else:
        fresh = predicted.copy()
    return {
        'predicted': predicted,
        'fresh': np.clip(fresh, 0.0, 1.0),
        'age': age,
        'validity_slots': validity,
        'source_slot': source_slot,
        'input_cutoff_slot': input_cutoff_slot,
    }


def _empty_shared_dqi(slots: int, channels: int) -> dict[str, np.ndarray]:
    shape = (slots, channels)
    return {
        'predicted': np.zeros(shape, dtype=float),
        'fresh': np.zeros(shape, dtype=float),
        'age': np.zeros(shape, dtype=float),
        'validity_slots': np.ones(shape, dtype=float),
        'source_slot': np.arange(slots, dtype=int),
        'input_cutoff_slot': np.arange(slots, dtype=int),
    }


def _generate_causal_theta(
    occupancy: np.ndarray,
    traffic: dict[str, Any],
    descriptor_cfg: dict[str, Any],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    slots, num_channels = occupancy.shape
    history_window = max(int(descriptor_cfg.get('prediction_history_slots', 50)), 1)
    p01 = float(traffic.get('p01', 0.2))
    p10 = float(traffic.get('p10', 0.2))
    stationary_idle = p10 / max(p01 + p10, 1e-9)
    theta = np.empty((slots, num_channels), dtype=float)
    idle = (~occupancy).astype(float)
    cumulative = np.vstack([np.zeros((1, num_channels)), np.cumsum(idle, axis=0)])
    for target in range(slots):
        if target == 0:
            markov_prior = np.full(num_channels, stationary_idle, dtype=float)
            recent_idle = markov_prior
        else:
            previous_busy = occupancy[target - 1]
            markov_prior = np.where(previous_busy, p10, 1.0 - p01)
            start = max(0, target - history_window)
            recent_idle = (cumulative[target] - cumulative[start]) / max(target - start, 1)
        theta[target] = 0.80 * markov_prior + 0.20 * recent_idle
    noise = rng.normal(0.0, float(descriptor_cfg.get('prediction_noise_std', 0.03)), theta.shape)
    input_cutoff = np.arange(slots, dtype=int) - 1
    target_slot = np.arange(slots, dtype=int)
    return np.clip(theta + noise, 0.0, 1.0), input_cutoff, target_slot


def _trace_id(config: dict[str, Any], split: str, seed: int, slots: int) -> str:
    payload = {
        'split': split,
        'seed': int(seed),
        'slots': int(slots),
        'num_channels': int(config.get('num_channels', 10)),
        'num_vehicles': int(config.get('num_vehicles', 1)),
        'traffic': config.get('traffic', {}),
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode('utf-8')).hexdigest()[:12]
    return f'{split}-s{seed}-{digest}'


def _dqi_metadata(config: dict[str, Any], split: str) -> dict[str, Any]:
    dqi = dict(config.get('dqi', {}))
    descriptors = dict(config.get('descriptors', {}))
    configured_reference = dqi.get('r_ref_mbps')
    return {
        'label_version': DQI_LABEL_VERSION,
        'slot_duration_ms': float(config.get('slot_duration_ms', 1.0)),
        'history_slots': int(descriptors.get('dqi_history_slots', 100)),
        'horizon_slots': int(descriptors.get('dqi_horizon_slots', 100)),
        'vehicle_quantile': float(dqi.get('vehicle_quantile', 0.10)),
        'time_quantile': float(dqi.get('time_quantile', 0.10)),
        'quantile_method': str(dqi.get('quantile_method', 'linear')),
        'split': split,
        'r_ref_mbps': (
            None
            if configured_reference is None
            else float(configured_reference)
        ),
        'reference_source': str(
            dqi.get('reference_source', 'legacy_config')
        ),
        'scaler_hash': str(dqi.get('scaler_hash', 'unfitted')),
    }


def _surrogate_hash(model_id: str, config: dict[str, Any]) -> str:
    payload = {
        'model_id': model_id,
        'descriptors': config.get('descriptors', {}),
        'dqi': config.get('dqi', {}),
        'traffic': config.get('traffic', {}),
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode('utf-8')).hexdigest()[:16]
    return f'surrogate-{digest}'


def _generate_vehicle_pu_links(
    slots: int,
    num_vehicles: int,
    num_channels: int,
    config: dict[str, Any],
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    shape = (slots, num_vehicles, num_channels)
    true_gain = rng.lognormal(mean=-1.2, sigma=0.35, size=shape)
    estimate_error = rng.lognormal(mean=0.0, sigma=0.18, size=shape)
    gain_hat = true_gain / estimate_error
    threshold = rng.uniform(0.10, 0.55, size=(slots, num_channels))
    p_max = float(config.get("underlay_power", {}).get("p_max_mw", 1.0))
    margin = float(config.get("underlay_power", {}).get("fading_margin", 2.0))
    denom = np.maximum(gain_hat * max(margin, 1e-6), 1e-9)
    transmit_cap = np.minimum(p_max, threshold[:, None, :] / denom)
    power_cap = np.clip(transmit_cap / max(p_max, 1e-9), 0.0, 1.0)
    return {
        "su_pu_gain_hat": gain_hat,
        "su_pu_gain_true": true_gain,
        "pu_threshold_mw": threshold,
        "power_cap": power_cap,
        "transmit_power_cap_mw": transmit_cap,
    }
