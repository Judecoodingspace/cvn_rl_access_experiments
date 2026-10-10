from __future__ import annotations

from cvn_rl.experiments.run_experiment import main


def test_cli_dry_run_creates_metrics_and_summary(tmp_path) -> None:
    rc = main(
        [
            "--config",
            "configs/baseline_comparison_gpu.json",
            "--dry-run",
            "--quick",
            "--results-root",
            str(tmp_path),
        ]
    )
    assert rc == 0
    run_dirs = list(tmp_path.iterdir())
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    assert (run_dir / "manifest.json").is_file()
    assert (run_dir / "metrics" / "access_metrics.csv").is_file()
    assert (run_dir / "tables" / "summary_by_agent.csv").is_file()
