from __future__ import annotations

from pathlib import Path

import pytest

from cvn_rl.experiments.artifacts import create_run_dir, freeze_config
from cvn_rl.experiments.config import load_config, make_run_id


def test_load_config_inherits_base() -> None:
    cfg = load_config("configs/baseline_comparison_gpu.json")
    assert cfg["experiment"] == "rl_baseline_comparison"
    assert cfg["slots"] == 5000
    assert "proposed_esn_ddqn" in cfg["agents"]


def test_create_run_dir_and_freeze_config(tmp_path: Path) -> None:
    run_dir = create_run_dir("unit_run", tmp_path)
    freeze_config(run_dir, {"experiment": "unit"})
    assert (run_dir / "metrics").is_dir()
    assert (run_dir / "tables").is_dir()
    assert (run_dir / "config.frozen.json").is_file()
    with pytest.raises(FileExistsError):
        create_run_dir("unit_run", tmp_path)


def test_revision_run_label_is_readable_and_cannot_escape_results_root() -> None:
    run_id = make_run_id("ignored", label="r2_r01c03_dqi_ablation", quick=True)
    assert run_id.endswith("_r2_r01c03_dqi_ablation_quick")
    with pytest.raises(ValueError):
        make_run_id("ignored", label="r2_../../outside")
