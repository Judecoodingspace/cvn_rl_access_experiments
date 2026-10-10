from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SwitchingCost:
    blackout_slots: int
    min_dwell_slots: int

    def __post_init__(self) -> None:
        if self.blackout_slots < 0 or self.min_dwell_slots < 0:
            raise ValueError('switching costs must be non-negative')


@dataclass(frozen=True)
class PolicyEvaluation:
    selected_channel: np.ndarray
    requested_channel: np.ndarray
    blackout: np.ndarray
    switched: np.ndarray
    net_shared_goodput_mbps: np.ndarray
    gross_shared_goodput_mbps: np.ndarray
    net_raw_rate_mbps: np.ndarray
    gross_raw_rate_mbps: np.ndarray
    delay_violation: np.ndarray
    instantaneous_regret_mbps: np.ndarray
    dqi_regret_mbps: np.ndarray


def make_channel_placebo(
    values: np.ndarray,
    *,
    block_slots: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Permute channel identities per block without changing block values."""

    source = np.asarray(values, dtype=float)
    if source.ndim != 2 or source.shape[1] <= 1:
        raise ValueError('values must have shape [time, channel] with M>1')
    block = max(int(block_slots), 1)
    placebo = np.empty_like(source)
    permutations: list[np.ndarray] = []
    for start in range(0, source.shape[0], block):
        stop = min(start + block, source.shape[0])
        permutation = rng.permutation(source.shape[1])
        while np.any(permutation == np.arange(source.shape[1])):
            permutation = rng.permutation(source.shape[1])
        placebo[start:stop] = source[start:stop, permutation]
        permutations.append(permutation)
    return placebo, permutations


def make_time_block_placebo(
    values: np.ndarray,
    *,
    block_slots: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Derange complete time blocks while preserving each block internally."""

    source = np.asarray(values, dtype=float)
    if source.ndim != 2:
        raise ValueError('values must have shape [time, channel]')
    block = max(int(block_slots), 1)
    if source.shape[0] % block != 0:
        raise ValueError('time placebo requires complete equal-length blocks')
    blocks = source.reshape(source.shape[0] // block, block, source.shape[1])
    if len(blocks) <= 1:
        raise ValueError('time placebo requires at least two blocks')
    permutation = rng.permutation(len(blocks))
    while np.any(permutation == np.arange(len(blocks))):
        permutation = rng.permutation(len(blocks))
    return blocks[permutation].reshape(source.shape), permutation


def future_window_mean(values: np.ndarray, horizon_samples: int) -> np.ndarray:
    """Return the mean of t+1 through t+H for every valid source t."""

    source = np.asarray(values, dtype=float)
    if source.ndim != 2:
        raise ValueError('values must have shape [time, channel]')
    horizon = int(horizon_samples)
    if horizon <= 0 or source.shape[0] <= horizon:
        raise ValueError('values must contain horizon_samples + 1 rows')
    cumulative = np.vstack(
        (np.zeros((1, source.shape[1]), dtype=float), np.cumsum(source, axis=0))
    )
    starts = np.arange(source.shape[0] - horizon, dtype=int) + 1
    stops = starts + horizon
    return (cumulative[stops] - cumulative[starts]) / horizon


def rolling_window_quantile(
    values: np.ndarray,
    *,
    window_samples: int,
    quantile: float = 0.10,
    quantile_method: str = 'linear',
) -> np.ndarray:
    """Compute a forward rolling quantile over all complete windows.

    Uses per-window ``np.quantile`` to avoid materialising the full
    ``sliding_window_view`` array, which can cause OOM on long series.
    """
    source = np.asarray(values, dtype=float)
    if source.ndim != 1:
        raise ValueError('values must be one-dimensional')
    window = int(window_samples)
    if window <= 0 or source.size < window:
        raise ValueError('values must contain at least one complete window')
    n_windows = source.size - window + 1
    result = np.empty(n_windows, dtype=float)
    for i in range(n_windows):
        result[i] = float(
            np.quantile(source[i : i + window], quantile, method=quantile_method)
        )
    return result


def evaluate_channel_policy(
    shared_goodput_mbps: np.ndarray,
    shared_raw_rate_mbps: np.ndarray,
    link_delay_violation: np.ndarray,
    ranking_scores: np.ndarray,
    projected_values: np.ndarray,
    oracle_values_mbps: np.ndarray,
    switching_cost: SwitchingCost,
    *,
    start_slot: int,
    use_hysteresis: bool,
) -> PolicyEvaluation:
    """Evaluate a score sequence under physical blackout and minimum dwell."""

    goodput = np.asarray(shared_goodput_mbps, dtype=float)
    raw = np.asarray(shared_raw_rate_mbps, dtype=float)
    violations = np.asarray(link_delay_violation, dtype=bool)
    ranking = np.asarray(ranking_scores, dtype=float)
    projected = np.asarray(projected_values, dtype=float)
    oracle = np.asarray(oracle_values_mbps, dtype=float)
    if goodput.ndim != 2:
        raise ValueError('shared_goodput_mbps must have shape [time, channel]')
    slots, channels = goodput.shape
    if raw.shape != goodput.shape or ranking.shape != goodput.shape:
        raise ValueError('raw rates and ranking scores must match goodput shape')
    if projected.shape != goodput.shape or oracle.shape != goodput.shape:
        raise ValueError('projected and oracle values must match goodput shape')
    if (
        violations.ndim != 3
        or violations.shape[0] != slots
        or violations.shape[2] != channels
    ):
        raise ValueError(
            'link_delay_violation must have shape [time, vehicle, channel]'
        )
    start = int(start_slot)
    if not 1 <= start < slots:
        raise ValueError('start_slot must be in [1, T-1]')

    length = slots - start
    selected = np.empty(length, dtype=int)
    requested = np.empty(length, dtype=int)
    blackout = np.zeros(length, dtype=bool)
    switched = np.zeros(length, dtype=bool)
    gross_goodput = np.empty(length, dtype=float)
    gross_raw = np.empty(length, dtype=float)
    net_goodput = np.empty(length, dtype=float)
    net_raw = np.empty(length, dtype=float)
    selected_violation = np.empty(length, dtype=float)
    instant_regret = np.empty(length, dtype=float)
    dqi_regret = np.empty(length, dtype=float)

    current = int(np.argmax(ranking[start]))
    last_switch_slot = start
    blackout_remaining = 0
    for row, t in enumerate(range(start, slots)):
        candidate = int(np.argmax(ranking[t]))
        requested[row] = candidate
        dwell_ok = (t - last_switch_slot) >= switching_cost.min_dwell_slots
        permit = candidate != current and dwell_ok and blackout_remaining == 0
        if permit and use_hysteresis:
            projected_gain = projected[t, candidate] - projected[t, current]
            estimated_loss = (
                switching_cost.blackout_slots * goodput[t - 1, current]
            )
            permit = bool(projected_gain > estimated_loss)
        if permit:
            current = candidate
            last_switch_slot = t
            blackout_remaining = switching_cost.blackout_slots
            switched[row] = True

        selected[row] = current
        gross_goodput[row] = goodput[t, current]
        gross_raw[row] = raw[t, current]
        selected_violation[row] = float(np.mean(violations[t, :, current]))
        instant_regret[row] = max(
            float(np.max(goodput[t]) - goodput[t, current]), 0.0
        )
        dqi_regret[row] = max(
            float(np.max(oracle[t]) - oracle[t, current]), 0.0
        )
        if blackout_remaining > 0:
            blackout[row] = True
            net_goodput[row] = 0.0
            net_raw[row] = 0.0
            blackout_remaining -= 1
        else:
            net_goodput[row] = gross_goodput[row]
            net_raw[row] = gross_raw[row]

    return PolicyEvaluation(
        selected_channel=selected,
        requested_channel=requested,
        blackout=blackout,
        switched=switched,
        net_shared_goodput_mbps=net_goodput,
        gross_shared_goodput_mbps=gross_goodput,
        net_raw_rate_mbps=net_raw,
        gross_raw_rate_mbps=gross_raw,
        delay_violation=selected_violation,
        instantaneous_regret_mbps=instant_regret,
        dqi_regret_mbps=dqi_regret,
    )
