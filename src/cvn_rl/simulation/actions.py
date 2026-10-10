from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np


MODES = ("overlay", "underlay")

DEFAULT_ACTION_MASK_THRESHOLDS: dict[str, float | None] = {
    "sensing_idle_min": 0.5,
    "overlay_theta_min": 0.35,
    "underlay_power_cap_min": 0.2,
    "underlay_theta_min": 0.2,
}


@dataclass(frozen=True)
class AccessAction:
    channel: int
    mode: str


def num_actions(num_channels: int) -> int:
    return 1 + 2 * num_channels


def encode_action(channel: int, mode: str, num_channels: int) -> int:
    if channel < 0 or mode == "no_access":
        return 0
    if channel >= num_channels:
        raise ValueError(f"channel {channel} outside 0..{num_channels - 1}")
    if mode not in MODES:
        raise ValueError(f"unsupported access mode: {mode}")
    return 1 + 2 * channel + MODES.index(mode)


def decode_action(index: int, num_channels: int) -> AccessAction:
    if index == 0:
        return AccessAction(channel=-1, mode="no_access")
    if index < 0 or index >= num_actions(num_channels):
        raise ValueError(f"action index {index} outside action space")
    offset = index - 1
    return AccessAction(channel=offset // 2, mode=MODES[offset % 2])


def build_action_mask(
    sensing_idle: np.ndarray,
    theta: np.ndarray,
    power_cap: np.ndarray,
    layers: list[str] | None = None,
    thresholds: Mapping[str, Any] | None = None,
    enabled_modes: list[str] | tuple[str, ...] | None = None,
) -> np.ndarray:
    mask = np.ones(num_actions(len(sensing_idle)), dtype=bool)
    enabled = set(MODES if enabled_modes is None else enabled_modes)
    unknown = enabled.difference(MODES)
    if unknown:
        raise ValueError(f'unsupported enabled access modes: {sorted(unknown)}')
    for channel in range(len(sensing_idle)):
        for mode in MODES:
            if mode not in enabled:
                mask[encode_action(channel, mode, len(sensing_idle))] = False
    if not layers or "action_mask" not in layers:
        return mask

    resolved = {**DEFAULT_ACTION_MASK_THRESHOLDS, **dict(thresholds or {})}
    underlay_theta_min = resolved["underlay_theta_min"]
    for key in ("sensing_idle_min", "underlay_power_cap_min"):
        if resolved[key] is None:
            raise ValueError(f"action-mask threshold {key} cannot be null")

    for channel in range(len(sensing_idle)):
        overlay = encode_action(channel, "overlay", len(sensing_idle))
        underlay = encode_action(channel, "underlay", len(sensing_idle))
        mask[overlay] = bool(
            sensing_idle[channel] > float(resolved["sensing_idle_min"])
            and (
                resolved["overlay_theta_min"] is None
                or theta[channel] > float(resolved["overlay_theta_min"])
            )
        )
        mask[underlay] = bool(
            power_cap[channel] > float(resolved["underlay_power_cap_min"])
            and (
                underlay_theta_min is None
                or theta[channel] > float(underlay_theta_min)
            )
        )
        if 'overlay' not in enabled:
            mask[overlay] = False
        if 'underlay' not in enabled:
            mask[underlay] = False
    mask[0] = True
    return mask
