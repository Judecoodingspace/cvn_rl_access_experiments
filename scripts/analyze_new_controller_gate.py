"""Technical integrity Gate for LSTM-DDQN and discrete SAC runs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

AGENTS = {"lstm_ddqn", "sac_discrete"}
P3_CONDITIONS = {
    "theta_oracle", "theta_central_markov", "theta_central_frozen",
    "theta_central_frozen_placebo", "theta_neutral",
}
MODEL_SHA = "c0d645858adf54bdcd04b9c0956629a9b11968a60013ff7d21ec5d8381cb7732"


def load_metrics(run_dirs: list[Path]) -> pd.DataFrame:
    frames = []
    for run_dir in run_dirs:
        path = run_dir / "metrics" / "access_metrics.csv"
        frame = pd.read_csv(path)
        frame = pd.concat([
            frame,
            pd.Series(str(run_dir.resolve()), index=frame.index, name="source_run"),
        ], axis=1)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def evaluate_gate(metrics: pd.DataFrame, require_p3: bool) -> tuple[dict, pd.DataFrame]:
    rows = []
    present = set(metrics.agent)
    rows.append(("known_agents", bool(present and present <= AGENTS), str(sorted(present))))
    numeric = [
        "served_goodput_mbps", "plr", "pu_access_violation_rate",
        "mean_reward", "training_time_seconds", "inference_latency_ms",
    ]
    rows.append(("finite_metrics", bool(np.isfinite(metrics[numeric].to_numpy()).all()), numeric))
    rows.append(("evaluation_frozen", bool(metrics.evaluation_parameters_frozen.astype(bool).all()), "all rows"))
    rows.append(("no_invalid_actions", bool((metrics.invalid_action_count == 0).all()), int(metrics.invalid_action_count.sum())))
    rows.append(("training_changed", bool(metrics.training_parameters_changed.astype(bool).all()), "all checkpoints"))
    rows.append(("gradient_updates", bool((metrics.gradient_updates > 0).all()), int(metrics.gradient_updates.min())))
    hash_equal = metrics.parameter_hash_before_eval.astype(str).eq(
        metrics.parameter_hash_after_eval.astype(str)
    )
    rows.append(("parameter_hash_frozen", bool(hash_equal.all()), int(hash_equal.sum())))
    for agent, group in metrics.groupby("agent"):
        loss_columns = ["q_loss"] if agent == "lstm_ddqn" else ["actor_loss", "critic_loss", "policy_entropy", "entropy_temperature"]
        finite = np.isfinite(group[loss_columns].to_numpy()).all()
        rows.append((f"{agent}_diagnostics_finite", bool(finite), loss_columns))
    if metrics.agent.nunique() > 1:
        trace_counts = metrics.groupby(
            ["seed", "module_variant", "checkpoint_slot"]
        ).evaluation_physical_trace_hash.nunique()
        rows.append(("paired_physical_traces", bool((trace_counts == 1).all()), int(trace_counts.max())))
    if require_p3:
        rows.append(("p3_agents_complete", present == AGENTS, str(sorted(present))))
        rows.append(("p3_conditions", set(metrics.module_variant) == P3_CONDITIONS, str(sorted(set(metrics.module_variant)))))
        expected_rows = 90 * len(AGENTS)
        rows.append(("p3_checkpoint_rows", len(metrics) == expected_rows, f"{len(metrics)}/{expected_rows}"))
        final = metrics[metrics.is_final_checkpoint.astype(bool)]
        rows.append(("p3_final_rows", len(final) == 15 * len(AGENTS), len(final)))
        frozen = final[final.module_variant.str.contains("frozen")]
        rows.append(("frozen_model_sha", set(frozen.prediction_model_hash) == {MODEL_SHA}, str(sorted(set(frozen.prediction_model_hash)))))
    checks = pd.DataFrame(rows, columns=["check", "passed", "detail"])
    payload = {
        "gate": "new_controller_technical_integrity",
        "require_p3": bool(require_p3),
        "passed": bool(checks.passed.all()),
        "checks": checks.to_dict(orient="records"),
    }
    return payload, checks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--require-p3", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload, checks = evaluate_gate(load_metrics(args.run_dirs), args.require_p3)
    checks.to_csv(args.output_dir / "new_controller_gate_checks.csv", index=False)
    (args.output_dir / "new_controller_gate.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8",
    )
    print(json.dumps(payload, indent=2))
    raise SystemExit(0 if payload["passed"] else 1)


if __name__ == "__main__":
    main()
