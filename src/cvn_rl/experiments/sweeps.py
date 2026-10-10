from __future__ import annotations

from itertools import product
from typing import Any, Iterable


def limited(values: Iterable[Any], dry_run: bool, limit: int = 2) -> list[Any]:
    items = list(values)
    return items[:limit] if dry_run else items


def seeds(config: dict[str, Any], dry_run: bool) -> list[int]:
    return limited(config.get("seeds", [1]), dry_run, 1)


def state_reward_cases(config: dict[str, Any], dry_run: bool) -> list[dict[str, Any]]:
    states = limited(config.get("state_variants", []), dry_run, 2)
    rewards = limited(config.get("reward_variants", []), dry_run, 2)
    return [{"state": s, "reward": r} for s, r in product(states, rewards)]


def robustness_cases(config: dict[str, Any], dry_run: bool) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    pred = config.get("prediction_error_grid", {})
    for fa, md in product(
        limited(pred.get("false_alarm_rates", []), dry_run),
        limited(pred.get("miss_detection_rates", []), dry_run),
    ):
        cases.append(
            {
                "sweep_name": "prediction_error",
                "prediction_false_alarm_rate": fa,
                "prediction_miss_detection_rate": md,
            }
        )
    sensing = config.get("sensing_error_grid", {})
    for pd, pfa in product(limited(sensing.get("pd", []), dry_run), limited(sensing.get("pfa", []), dry_run)):
        cases.append({"sweep_name": "sensing_error", "sensing_pd": pd, "sensing_pfa": pfa})
    underlay = config.get("underlay_estimation_error", {})
    for err, margin in product(
        limited(underlay.get("path_loss_error_db", []), dry_run),
        limited(underlay.get("fading_margin", []), dry_run),
    ):
        cases.append({"sweep_name": "underlay_error", "path_loss_error_db": err, "fading_margin": margin})
    return cases or [{"sweep_name": "nominal"}]


def scale_traffic_cases(config: dict[str, Any], dry_run: bool) -> list[dict[str, Any]]:
    scenarios = limited(config.get("scenarios", []), dry_run)
    families = limited(config.get("traffic_families", []), dry_run)
    return [{"scenario": s, "traffic": f} for s, f in product(scenarios, families)]


def overhead_cases(config: dict[str, Any], dry_run: bool) -> list[dict[str, Any]]:
    updates = limited(config.get("broadcast_update_slots", []), dry_run)
    validity = limited(config.get("dqi_validity_slots", []), dry_run)
    return [{"broadcast_update_slots": u, "dqi_validity_slots": v} for u, v in product(updates, validity)]
