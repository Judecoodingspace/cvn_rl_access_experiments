from __future__ import annotations

import json
import hashlib
import importlib.metadata
import os
import platform
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


def create_run_dir(run_id: str, results_root: str | Path = "results") -> Path:
    root = Path(results_root) / run_id
    # Never merge a new run into an existing directory, including two runs
    # started in the same second with the same revision label.
    root.mkdir(parents=True, exist_ok=False)
    for subdir in ["metrics", "tables", "figures", "analysis"]:
        (root / subdir).mkdir(parents=True, exist_ok=True)
    return root


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)


def freeze_config(run_dir: Path, config: dict[str, Any]) -> None:
    write_json(run_dir / "config.frozen.json", config)


def runtime_provenance() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[3]
    packages = {}
    for name in ["numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "torch"]:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    git_commit = None
    git_dirty = None
    try:
        command = ["git", "-c", f"safe.directory={root.as_posix()}"]
        head = subprocess.run(command + ["rev-parse", "HEAD"], cwd=root,
                              capture_output=True, text=True, timeout=5)
        if head.returncode == 0:
            git_commit = head.stdout.strip()
            state = subprocess.run(command + ["status", "--porcelain"], cwd=root,
                                   capture_output=True, text=True, timeout=5)
            git_dirty = bool(state.stdout.strip()) if state.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        pass
    sources = {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
               for path in sorted((root / "src" / "cvn_rl").rglob("*.py"))}
    return {"python": platform.python_version(), "platform": platform.platform(),
            "packages": packages, "device_requested": os.environ.get("CVN_RL_DEVICE", "auto"),
            "git_commit": git_commit, "git_dirty": git_dirty, "source_sha256": sources}


def write_manifest(
    run_dir: Path,
    config: dict[str, Any],
    run_id: str,
    dry_run: bool,
    quick: bool,
    outputs: list[str],
) -> None:
    write_json(
        run_dir / "manifest.json",
        {
            "run_id": run_id,
            "experiment": config.get("experiment"),
            "dry_run": dry_run,
            "quick": quick,
            **({'r5c2_stage': config.get('r5c2_stage'), 'execution_scope': config.get('r5c2_execution_scope')}
               if config.get('r5c2_protocol_version') else {}),
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "outputs": outputs,
            "provenance": runtime_provenance(),
        },
    )


def write_notes(run_dir: Path, config: dict[str, Any], row_count: int) -> None:
    multi_vehicle = str(config.get("experiment", "")).startswith("multi_vehicle")
    evaluation = "checkpoint evaluation on independent frozen traces" if multi_vehicle else "single-trace evaluation"
    text = (
        f"# Experiment notes\n\n"
        f"- Experiment: {config.get('experiment')}\n"
        f"- Rows in metrics table: {row_count}\n"
        f"- Evaluation protocol: {evaluation}.\n"
        f"- Experiment design: see docs/experiment_design.md.\n"
        f"- Dry/quick runs are diagnostic and must not be reported as formal paper results.\n"
    )
    (run_dir / "analysis" / "experiment_notes.md").write_text(text, encoding="utf-8")


def write_dataframes(
    run_dir: Path,
    metrics: pd.DataFrame,
    summary: pd.DataFrame,
    expected_outputs: list[str] | None = None,
) -> list[str]:
    outputs = []
    metrics_path = _first_output(expected_outputs, "metrics/", "metrics/access_metrics.csv")
    table_path = _first_output(expected_outputs, "tables/", "tables/summary_by_agent.csv")
    for rel_path, df in [(metrics_path, metrics), (table_path, summary)]:
        path = run_dir / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        outputs.append(rel_path)
    if metrics_path != "metrics/access_metrics.csv":
        metrics.to_csv(run_dir / "metrics" / "access_metrics.csv", index=False)
        outputs.append("metrics/access_metrics.csv")
    if table_path != "tables/summary_by_agent.csv":
        summary.to_csv(run_dir / "tables" / "summary_by_agent.csv", index=False)
        outputs.append("tables/summary_by_agent.csv")
    return outputs


def _first_output(outputs: list[str] | None, prefix: str, default: str) -> str:
    for output in outputs or []:
        if output.startswith(prefix):
            return output
    return default
