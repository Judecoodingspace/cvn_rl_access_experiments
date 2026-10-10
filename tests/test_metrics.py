from __future__ import annotations

from cvn_rl.metrics import estimate_broadcast_overhead_kbps, summarize_episode


def test_metrics_summary_has_required_fields() -> None:
    records = [
        {
            "mode": "overlay",
            "packet_loss": False,
            "vt_throughput": 10.0,
            "pu_throughput": 1.0,
            "pu_violation": False,
            "selected_dqi": 0.8,
            "reward": 0.2,
        },
        {
            "mode": "underlay",
            "packet_loss": True,
            "vt_throughput": 3.0,
            "pu_throughput": 0.2,
            "pu_violation": True,
            "selected_dqi": 0.5,
            "reward": -0.4,
        },
    ]
    row = summarize_episode(records, {"agent": "unit", "seed": 1}, {"num_channels": 2})
    assert row["plr"] == 0.5
    assert row["pu_interference_violation_rate"] == 0.5
    assert row["underlay_selection_ratio"] == 0.5
    assert estimate_broadcast_overhead_kbps({"num_channels": 2}) > 0
