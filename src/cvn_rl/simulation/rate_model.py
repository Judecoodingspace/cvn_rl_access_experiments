from __future__ import annotations

from typing import Any

import numpy as np


def underlay_transmit_power_mw(
    transmit_power_cap_mw: float,
    config: dict[str, Any],
) -> float:
    underlay_cfg = config.get('underlay_power', {})
    p_max = max(float(underlay_cfg.get('p_max_mw', 1.0)), 1e-12)
    ceiling_fraction = float(
        underlay_cfg.get('service_power_ceiling_fraction', 1.0)
    )
    if not 0.0 <= ceiling_fraction <= 1.0:
        raise ValueError(
            'underlay service_power_ceiling_fraction must lie in [0, 1]'
        )
    return float(
        np.clip(
            min(float(transmit_power_cap_mw), ceiling_fraction * p_max),
            0.0,
            p_max,
        )
    )


def underlay_rate_mbps(
    full_power_quality_mbps: float,
    bandwidth_mhz: float,
    transmit_power_mw: float,
    busy: bool,
    config: dict[str, Any],
) -> float:
    """Return underlay service rate using the configured physical model."""

    if str(config.get('underlay_rate_model', 'legacy_v1')) != 'power_aware_v1':
        return float(full_power_quality_mbps) * (0.65 if busy else 1.0)

    bandwidth = max(float(bandwidth_mhz), 1e-12)
    p_max = max(
        float(config.get('underlay_power', {}).get('p_max_mw', 1.0)),
        1e-12,
    )
    alpha = float(np.clip(float(transmit_power_mw) / p_max, 0.0, 1.0))
    gamma_max = float(np.exp2(float(full_power_quality_mbps) / bandwidth) - 1.0)
    pressure = max(
        float(
            config.get('underlay_power', {}).get(
                'pu_to_su_interference_ratio',
                1.0,
            )
        ),
        0.0,
    )
    effective_snr = alpha * gamma_max / (1.0 + (pressure if busy else 0.0))
    rate = bandwidth * float(np.log2(1.0 + max(effective_snr, 0.0)))
    return float(np.clip(rate, 0.0, max(float(full_power_quality_mbps), 0.0)))


def potential_access_rate_mbps(
    *,
    mode: str,
    full_power_quality_mbps: float,
    bandwidth_mhz: float,
    transmit_power_cap_mw: float,
    busy: bool,
    config: dict[str, Any],
) -> float:
    if mode == 'overlay':
        return 0.0 if busy else max(float(full_power_quality_mbps), 0.0)
    if mode == 'underlay':
        transmit_power = underlay_transmit_power_mw(
            transmit_power_cap_mw,
            config,
        )
        return underlay_rate_mbps(
            full_power_quality_mbps,
            bandwidth_mhz,
            transmit_power,
            busy,
            config,
        )
    return 0.0
