from __future__ import annotations

import copy
import json
import random
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


def load_config(path: str | Path, root: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = Path(root or Path.cwd()) / config_path
    config_path = config_path.resolve()
    with config_path.open("r", encoding="utf-8-sig") as fh:
        data = json.load(fh)
    parent = data.pop("inherits", None)
    if parent:
        base = load_config(config_path.parent / parent)
        data = deep_merge(base, data)
    data["_config_path"] = str(config_path)
    return data


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def make_run_id(
    experiment: str,
    dry_run: bool = False,
    *,
    quick: bool = False,
    label: str | None = None,
) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = "dry" if dry_run else "quick" if quick else "formal"
    if label is not None:
        if not re.fullmatch(r"r[0-9]+_[a-z0-9]+(?:_[a-z0-9]+)*", label):
            raise ValueError(
                "run label must look like r2_r01c03_prediction_robustness "
                "(lowercase letters, digits, underscores)"
            )
        return f"{stamp}_{label}_{suffix}"
    # Preserve historical names for existing callers and older result links.
    return f"{stamp}_{experiment}_{'dry' if dry_run else 'run'}"
