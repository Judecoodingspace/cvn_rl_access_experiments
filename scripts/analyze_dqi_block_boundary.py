"""Block-boundary regret analysis for DQI D0 experiment.

Analyses how oracle-vs-shuffled channel-selection regret varies with
position inside a quality block.  Hypothesis: regret should spike at
block boundaries where the channel→profile permutation rotates, because
DQI oracle immediately sees the new ordering while current_cqi still
reflects the old ordering from the previous block.

Usage:
    .venv/Scripts/python.exe scripts/analyze_dqi_block_boundary.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# Ensure the project root is on sys.path
_PROJECT = Path(__file__).resolve().parents[1]
if str(_PROJECT) not in sys.path:
    sys.path.insert(0, str(_PROJECT))

from src.cvn_rl.simulation.multi_traces import _generate_vehicle_quality
from src.cvn_rl.experiments.multi_vehicle_runner import _future_quantile_dqi


# Minimal stub trace-like object carrying only the fields that
# _future_quantile_dqi needs.  This avoids the full (and memory-heavy)
# generate_multi_vehicle_trace which allocates causal DQI arrays.
class _LightTrace:
    __slots__ = ("quality_mbps", "quality_lookahead_mbps", "slots", "num_channels")
    def __init__(self, quality: np.ndarray, lookahead: np.ndarray):
        self.quality_mbps = quality
        self.quality_lookahead_mbps = lookahead
        self.slots = quality.shape[0]
        self.num_channels = quality.shape[2]


def load_frozen_config(run_dir: Path) -> dict:
    path = run_dir / "config.frozen.json"
    if not path.exists():
        raise FileNotFoundError(f"Frozen config not found: {path}")
    with open(path) as fh:
        return json.load(fh)


def compute_regret(
    quality_mbps: np.ndarray,           # (slots, vehicles, channels)
    future_oracle_dqi: np.ndarray,      # (slots, channels)
    descriptor_cfg: dict,
    rng: np.random.Generator,
) -> dict:
    """Compute per-timestep channel-selection regret of shuffled vs oracle DQI.

    Returns a dict with:
      regret[t]        = quality at oracle-chosen channel minus quality at
                          shuffled-chosen channel (positive = oracle wins)
      oracle_ch[t]     = channel index chosen by oracle DQI
      shuffled_ch[t]   = channel index chosen by shuffled DQI
    """
    slots = quality_mbps.shape[0]
    num_channels = quality_mbps.shape[2]

    update = max(int(descriptor_cfg.get("dqi_update_slots", 1_000)), 1)
    base = np.arange(num_channels)

    # Build the shuffled DQI using the same logic as channel_shuffled
    shuffled_dqi = np.empty_like(future_oracle_dqi)
    for source in range(0, slots, update):
        stop = min(source + update, slots)
        offset = int(rng.integers(1, num_channels))
        permutation = np.roll(base, offset)
        shuffled_dqi[source:stop] = future_oracle_dqi[source:stop, permutation]

    # Oracle chooses the channel with the highest DQI value in each timestep.
    oracle_avg = future_oracle_dqi
    shuffled_avg = shuffled_dqi

    oracle_ch = np.argmax(oracle_avg, axis=1)      # (slots,)
    shuffled_ch = np.argmax(shuffled_avg, axis=1)  # (slots,)

    # Quality of the chosen channel at each timestep (avg across vehicles)
    q = quality_mbps.mean(axis=1)  # (slots, channels)
    oracle_q = q[np.arange(slots), oracle_ch]
    shuffled_q = q[np.arange(slots), shuffled_ch]

    regret = oracle_q - shuffled_q

    return {
        "regret": regret,
        "oracle_ch": oracle_ch,
        "shuffled_ch": shuffled_ch,
        "shuffled_dqi": shuffled_dqi,
        "oracle_avg": oracle_avg,
        "shuffled_avg": shuffled_avg,
    }


def analyse_block_profile(
    regret: np.ndarray,
    oracle_ch: np.ndarray,
    shuffled_ch: np.ndarray,
    block_slots: int,
) -> dict:
    """Aggregate regret by position within a quality block.

    Returns per-position mean regret and disagreement rate.
    """
    positions = np.arange(len(regret)) % block_slots
    regret_by_pos = np.zeros(block_slots)
    disagree_by_pos = np.zeros(block_slots)
    count_by_pos = np.zeros(block_slots, dtype=int)

    for pos in range(block_slots):
        mask = positions == pos
        count_by_pos[pos] = mask.sum()
        if count_by_pos[pos] > 0:
            regret_by_pos[pos] = regret[mask].mean()
            disagree_by_pos[pos] = (oracle_ch[mask] != shuffled_ch[mask]).mean()

    return {
        "regret_profile": regret_by_pos,
        "disagreement_profile": disagree_by_pos,
        "count_profile": count_by_pos,
        "block_slots": block_slots,
    }


def main() -> None:
    # Locate the D0 frozen config
    d0_run = (
        _PROJECT
        / "results"
        / "20260703_134901_e2_r1_oracle_gate_run"
    )
    config = load_frozen_config(d0_run)
    descriptor_cfg = dict(config.get("descriptors", {}))
    block_slots = int(
        config.get("channel", {}).get("quality_block_slots", 40_000)
    )
    seeds = list(config.get("seeds", [1, 2, 3]))
    slots = int(config.get("slots", 100_000))
    hidden_lookahead = max(int(config.get("hidden_lookahead_slots", 10_000)), 1)
    num_channels = int(config.get("num_channels", 10))
    num_vehicles = int(config.get("num_vehicles", 1))

    print(f"D0 config: slots={slots}, seeds={seeds}, block_slots={block_slots}")
    print(f"AR range: {config['scenarios'][0]['channel']['ar_coefficient_range']}")
    print(f"DQI horizon: {descriptor_cfg.get('dqi_horizon_slots')}, "
          f"quantile: {descriptor_cfg.get('dqi_quantile')}")

    all_regret = []
    all_disagree = []

    for seed_idx, seed in enumerate(seeds):
        print(f"\n--- Seed {seed} ---")
        rng = np.random.default_rng(seed + 73_001)
        trace_rng = np.random.default_rng(seed)  # matches generate_multi_vehicle_trace

        # Regenerate only the quality trace + lookahead (skip causal DQI).
        # full generate_multi_vehicle_trace allocates ~200 MB of causal-DQI
        # arrays that we never use.
        _bw, quality_with_lookahead = _generate_vehicle_quality(
            slots + hidden_lookahead,
            num_vehicles,
            num_channels,
            config,
            trace_rng,
        )
        quality = quality_with_lookahead[:slots]
        lookahead = quality_with_lookahead[slots:]
        trace = _LightTrace(quality, lookahead)

        # Compute oracle future DQI
        future_oracle = _future_quantile_dqi(trace, descriptor_cfg)

        # Compute channel-selection regret
        result = compute_regret(
            trace.quality_mbps, future_oracle, descriptor_cfg, rng
        )

        # Block-position profile
        profile = analyse_block_profile(
            result["regret"],
            result["oracle_ch"],
            result["shuffled_ch"],
            block_slots,
        )

        all_regret.append(profile["regret_profile"])
        all_disagree.append(profile["disagreement_profile"])

        # Per-seed stats
        print(f"  Mean regret:  {result['regret'].mean():.4f} Mbps")
        print(f"  Max regret:   {result['regret'].max():.4f} Mbps "
              f"@ pos {result['regret'].argmax() % block_slots}")
        print(f"  Oracle≠Shuffled: {(result['oracle_ch'] != result['shuffled_ch']).mean():.2%}")

        # Report regret around block boundaries
        for boundary_name, boundary_pos in [("start", 0), ("end", block_slots - 1)]:
            window = 500
            if boundary_pos == 0:
                idx = list(range(0, window)) + list(range(block_slots - window, block_slots))
            else:
                idx = list(range(boundary_pos - window, boundary_pos + window))
            idx = np.array(idx, dtype=int)
            # Collect regret across all blocks at these positions
            positions = np.arange(len(result["regret"])) % block_slots
            boundary_regret = result["regret"][
                np.isin(positions, idx)
            ]
            print(f"  Boundary ±{window} (pos≈{boundary_pos}): "
                  f"regret={boundary_regret.mean():.4f} Mbps")

    # Aggregate across seeds
    mean_regret = np.mean(all_regret, axis=0)
    std_regret = np.std(all_regret, axis=0)
    mean_disagree = np.mean(all_disagree, axis=0)

    # ----- Plot -----
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    x = np.arange(block_slots) / 1000  # k-slots

    # Panel 1: Regret profile
    ax1.fill_between(
        x,
        mean_regret - std_regret,
        mean_regret + std_regret,
        alpha=0.25,
        label="±1σ across seeds",
    )
    ax1.plot(x, mean_regret, linewidth=0.8, color="tab:blue", label="Mean regret")
    ax1.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
    ax1.axvline(x=0, color="tab:red", linestyle=":", alpha=0.5, label="Block boundary")
    ax1.axvline(x=block_slots / 1000, color="tab:red", linestyle=":", alpha=0.5)
    ax1.set_ylabel("Regret (Mbps)\noracle − shuffled quality")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.set_title("D0 Channel-Selection Regret by Block Position")
    ax1.grid(True, alpha=0.3)

    # Panel 2: Disagreement rate
    ax2.plot(x, mean_disagree, linewidth=0.8, color="tab:orange")
    ax2.axvline(x=0, color="tab:red", linestyle=":", alpha=0.5)
    ax2.axvline(x=block_slots / 1000, color="tab:red", linestyle=":", alpha=0.5)
    ax2.set_xlabel("Position within quality block (k-slots)")
    ax2.set_ylabel("Oracle ≠ Shuffled\n(channel disagreement rate)")
    ax2.grid(True, alpha=0.3)

    # Add boundary zone annotations
    ax2.axvspan(0, 2.5, alpha=0.06, color="tab:red")
    ax2.axvspan((block_slots / 1000) - 2.5, block_slots / 1000, alpha=0.06, color="tab:red")
    ax2.text(1.25, ax2.get_ylim()[1] * 0.95, "boundary\nzone",
             ha="center", va="top", fontsize=8, color="tab:red", alpha=0.7)

    # Overall stats
    overall_mean = mean_regret.mean()
    boundary_zone = 2500  # ~DQI horizon / 2
    near_start = mean_regret[:boundary_zone].mean()
    near_end = mean_regret[-boundary_zone:].mean()
    interior = mean_regret[boundary_zone:-boundary_zone].mean()

    stats_text = (
        f"Overall mean regret: {overall_mean:.4f} Mbps\n"
        f"Boundary (±{boundary_zone}): start={near_start:.4f}, end={near_end:.4f}\n"
        f"Interior (middle {block_slots - 2*boundary_zone}): {interior:.4f}"
    )
    ax1.text(
        0.98, 0.95,
        stats_text,
        transform=ax1.transAxes,
        fontsize=9,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="wheat", alpha=0.8),
    )

    plt.tight_layout()

    # Save
    out_dir = _PROJECT / "reports" / "diagnostic_findings"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_path = out_dir / "dqi_block_boundary_regret_profile.pdf"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {fig_path}")

    # Save CSV
    csv_path = out_dir / "dqi_block_boundary_regret_profile.csv"
    header = "position,position_k_slots,mean_regret,std_regret,mean_disagreement"
    rows = [
        f"{p},{p/1000:.3f},{mean_regret[p]:.6f},{std_regret[p]:.6f},{mean_disagree[p]:.6f}"
        for p in range(block_slots)
    ]
    csv_path.write_text("\n".join([header] + rows))
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
