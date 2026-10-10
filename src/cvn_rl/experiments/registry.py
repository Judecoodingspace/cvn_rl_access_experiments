from __future__ import annotations

from cvn_rl.experiments.multi_vehicle_runner import run_multi_vehicle_experiment
from cvn_rl.experiments.e2_r1 import run_e2_r1_experiment
from cvn_rl.experiments.runners import run_experiment_matrix

RUNNERS = {
    'multi_vehicle_descriptor_sensitivity': run_multi_vehicle_experiment,
    'multi_vehicle_external_validation': run_multi_vehicle_experiment,
    'multi_vehicle_module_ablation': run_multi_vehicle_experiment,
    'multi_vehicle_protection_ablation': run_multi_vehicle_experiment,
    'multi_vehicle_evaluation_robustness': run_multi_vehicle_experiment,
    'e2_r1_oracle_gate': run_e2_r1_experiment,
    "rl_baseline_comparison": run_experiment_matrix,
    "state_reward_ablation": run_experiment_matrix,
    "prediction_sensing_robustness": run_experiment_matrix,
    "scale_and_traffic_sensitivity": run_experiment_matrix,
    "overhead_staleness_sensitivity": run_experiment_matrix,
    "multi_vehicle_baseline_comparison": run_multi_vehicle_experiment,
    "multi_vehicle_scale_traffic": run_multi_vehicle_experiment,
}


def get_runner(experiment: str):
    try:
        return RUNNERS[experiment]
    except KeyError as exc:
        raise ValueError(f"Unknown experiment: {experiment}") from exc
