"""Cross-platform checkout validation and experiment pipeline."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def run(*arguments: str) -> None:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + environment.get("PYTHONPATH", "")
    environment.setdefault("MPLBACKEND", "Agg")
    print("+", sys.executable, *arguments, flush=True)
    subprocess.run([sys.executable, *arguments], cwd=ROOT, env=environment, check=True)


def environment_record(device: str) -> dict:
    import torch
    packages = {}
    for name in ("numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "torch", "pytest", "xgboost", "pyarrow"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {"python": platform.python_version(), "platform": platform.platform(),
            "packages": packages, "requested_device": device,
            "cuda_available": torch.cuda.is_available(), "torch_cuda": torch.version.cuda}


def check_config(config_path: Path) -> dict:
    from cvn_rl.experiments.config import load_config
    config = load_config(config_path)
    inputs = []
    def visit(value, prefix=""):
        if isinstance(value, dict):
            for key, item in value.items():
                dotted = f"{prefix}.{key}" if prefix else key
                if isinstance(item, str) and (key.endswith("model_path") or key in {
                    "source_meta", "bundle_path_template", "scenario_source", "guidance_source"
                }):
                    path = Path(item)
                    path = path if path.is_absolute() else ROOT / path
                    inputs.append({"key": dotted, "path": item,
                                   "exists": path.exists(), "template": "{" in item})
                visit(item, dotted)
        elif isinstance(value, list):
            for index, item in enumerate(value):
                visit(item, f"{prefix}[{index}]")
    visit(config)
    # Follow scenario/guidance configs as well as ordinary `inherits` parents.
    from scripts.audit_release import config_inputs
    existing_paths = {item['path'] for item in inputs}
    for name in sorted(config_inputs(config_path)):
        if name not in existing_paths:
            path = Path(name)
            path = path if path.is_absolute() else ROOT / path
            inputs.append({'key': 'transitive_input', 'path': name,
                           'exists': path.exists(), 'template': '{' in name})
    return {"config": str(config_path.relative_to(ROOT)), "experiment": config.get("experiment"),
            "descriptor_source": config.get("descriptors", {}).get("source"),
            "requires_gate": bool(config.get("requires_pilot_gate") or config.get("requires_e2_r1_gate")),
            "requires_reuse": bool(config.get("reuse_gate_requirement")), "inputs": inputs,
            "missing_inputs": [item for item in inputs if not item["exists"]]}


def smoke(results_root: Path, device: str) -> None:
    before = set(results_root.iterdir()) if results_root.exists() else set()
    run("-m", "cvn_rl.experiments.run_experiment", "--config", "configs/repro_smoke.json",
        "--quick", "--results-root", str(results_root), "--run-label", "r0_repro_smoke")
    created = set(results_root.iterdir()) - before
    if len(created) != 1:
        raise RuntimeError(f"Expected one new run directory, found {len(created)}")
    directory = created.pop()
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    for relative in ["config.frozen.json", "metrics/access_metrics.csv", "tables/summary_by_agent.csv", *manifest["outputs"]]:
        if not (directory / relative).is_file():
            raise RuntimeError(f"Missing pipeline artifact: {directory / relative}")
    import pandas as pd
    metrics = pd.read_csv(directory / "metrics/access_metrics.csv")
    if metrics.empty or set(metrics["agent"]) != {"proposed_esn_ddqn", "ddqn_no_esn", "fixed_overlay"}:
        raise RuntimeError("Smoke run did not evaluate all three requested policies")
    record = environment_record(device)
    record["artifact_sha256"] = {name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
                                for name in manifest["outputs"]}
    (directory / "environment.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"Smoke artifacts verified: {directory}", flush=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["doctor", "test", "smoke", "check", "preflight"])
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    parser.add_argument("--config", default="configs/repro_smoke.json")
    parser.add_argument("--results-root", default="results/reproducibility")
    args = parser.parse_args(argv)
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    os.environ["CVN_RL_DEVICE"] = args.device
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    if args.stage in {"doctor", "check", "smoke"}:
        record = environment_record(args.device)
        print(json.dumps(record, indent=2), flush=True)
        if args.device == "cuda" and not record["cuda_available"]:
            raise RuntimeError("CUDA requested but unavailable; use --device cpu or install a compatible GPU build")
    if args.stage == "preflight":
        record = check_config(Path(args.config).resolve())
        print(json.dumps(record, indent=2), flush=True)
        return 1 if record["missing_inputs"] else 0
    if args.stage in {"test", "check"}:
        run("-m", "pytest", "-q")
    if args.stage in {"smoke", "check"}:
        smoke(Path(args.results_root).resolve(), args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
