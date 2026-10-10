from __future__ import annotations

from typing import Any

import numpy as np


def generate_occupancy(
    slots: int,
    num_channels: int,
    traffic: dict[str, Any],
    rng: np.random.Generator,
) -> np.ndarray:
    family = traffic.get("family", "fixed_markov")
    if family == "fixed_markov":
        return _fixed_markov(slots, num_channels, traffic, rng)
    if family == "time_varying_markov":
        return _time_varying_markov(slots, num_channels, traffic, rng)
    if family == "bursty_on_off":
        return _bursty_on_off(slots, num_channels, traffic, rng)
    if family == "periodic_with_jitter":
        return _periodic_with_jitter(slots, num_channels, traffic, rng)
    if family == "mixed_by_channel":
        return _mixed_by_channel(slots, num_channels, rng)
    raise ValueError(f"Unsupported traffic family: {family}")


def _fixed_markov(
    slots: int,
    num_channels: int,
    traffic: dict[str, Any],
    rng: np.random.Generator,
) -> np.ndarray:
    p01 = float(traffic.get("p01", 0.2))
    p10 = float(traffic.get("p10", 0.25))
    busy = np.zeros((slots, num_channels), dtype=bool)
    stationary_busy = p01 / max(p01 + p10, 1e-9)
    busy[0] = rng.random(num_channels) < stationary_busy
    for t in range(1, slots):
        prev = busy[t - 1]
        busy[t] = np.where(prev, rng.random(num_channels) >= p10, rng.random(num_channels) < p01)
    return busy


def _time_varying_markov(
    slots: int,
    num_channels: int,
    traffic: dict[str, Any],
    rng: np.random.Generator,
) -> np.ndarray:
    blocks = int(traffic.get("blocks", 5))
    p01_range = traffic.get("p01_range", [0.05, 0.5])
    p10_range = traffic.get("p10_range", [0.05, 0.5])
    busy = np.zeros((slots, num_channels), dtype=bool)
    busy[0] = rng.random(num_channels) < 0.5
    edges = np.linspace(0, slots, blocks + 1, dtype=int)
    for start, end in zip(edges[:-1], edges[1:]):
        p01 = rng.uniform(float(p01_range[0]), float(p01_range[1]))
        p10 = rng.uniform(float(p10_range[0]), float(p10_range[1]))
        if start > 0:
            busy[start] = busy[start - 1]
        for t in range(max(1, start + 1), end):
            prev = busy[t - 1]
            busy[t] = np.where(prev, rng.random(num_channels) >= p10, rng.random(num_channels) < p01)
    return busy


def _bursty_on_off(
    slots: int,
    num_channels: int,
    traffic: dict[str, Any],
    rng: np.random.Generator,
) -> np.ndarray:
    idle_mean = max(float(traffic.get("idle_mean_slots", 20)), 1.0)
    busy_mean = max(float(traffic.get("busy_mean_slots", 15)), 1.0)
    busy = np.zeros((slots, num_channels), dtype=bool)
    for channel in range(num_channels):
        t = 0
        state_busy = bool(rng.random() < 0.4)
        while t < slots:
            mean = busy_mean if state_busy else idle_mean
            duration = max(1, int(rng.exponential(mean)))
            busy[t : min(slots, t + duration), channel] = state_busy
            t += duration
            state_busy = not state_busy
    return busy


def _periodic_with_jitter(
    slots: int,
    num_channels: int,
    traffic: dict[str, Any],
    rng: np.random.Generator,
) -> np.ndarray:
    period = int(traffic.get("period_slots", 50))
    busy_slots = int(traffic.get("busy_slots", 20))
    jitter_slots = int(traffic.get("jitter_slots", 5))
    busy = np.zeros((slots, num_channels), dtype=bool)
    for channel in range(num_channels):
        phase = int(rng.integers(0, max(period, 1)))
        for start in range(-phase, slots, max(period, 1)):
            jitter = int(rng.integers(-jitter_slots, jitter_slots + 1)) if jitter_slots else 0
            begin = max(0, start + jitter)
            end = min(slots, begin + busy_slots)
            if begin < slots:
                busy[begin:end, channel] = True
    return busy


def _mixed_by_channel(slots: int, num_channels: int, rng: np.random.Generator) -> np.ndarray:
    families = [
        {"family": "fixed_markov", "p01": 0.1, "p10": 0.3},
        {"family": "fixed_markov", "p01": 0.35, "p10": 0.12},
        {"family": "bursty_on_off", "idle_mean_slots": 20, "busy_mean_slots": 15},
        {"family": "periodic_with_jitter", "period_slots": 50, "busy_slots": 20, "jitter_slots": 5},
    ]
    out = np.zeros((slots, num_channels), dtype=bool)
    for channel in range(num_channels):
        channel_rng = np.random.default_rng(int(rng.integers(0, 2**32 - 1)))
        out[:, channel] = generate_occupancy(slots, 1, families[channel % len(families)], channel_rng)[:, 0]
    return out

def traffic_label(traffic: dict[str, Any]) -> str:
    family = str(traffic.get("family", "fixed_markov"))
    if family == "fixed_markov":
        return f"fixed_markov_p01_{traffic.get('p01', 0.2)}_p10_{traffic.get('p10', 0.25)}"
    if family == "time_varying_markov":
        return f"time_varying_markov_blocks_{traffic.get('blocks', 5)}"
    if family == "bursty_on_off":
        return f"bursty_on_off_idle_{traffic.get('idle_mean_slots', 20)}_busy_{traffic.get('busy_mean_slots', 15)}"
    if family == "periodic_with_jitter":
        return f"periodic_jitter_period_{traffic.get('period_slots', 50)}_busy_{traffic.get('busy_slots', 20)}"
    return family
