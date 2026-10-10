from __future__ import annotations

import hashlib
import json
import string
from pathlib import Path
from typing import Any, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from cvn_rl.simulation.multi_traces import MultiVehicleAccessTrace


SCHEMA_VERSION = "2.0"

PHYSICAL_ARRAYS = (
    "occupancy",
    "sensing_idle",
    "quality_mbps",
    "snr_db",
    "rss_dbm",
    "cci",
    "power_cap",
    "occupancy_type",
    "su_pu_gain_hat",
    "su_pu_gain_true",
    "pu_threshold_mw",
    "transmit_power_cap_mw",
    "packet_arrivals",
    "bandwidth_mhz",
    "pu_signal_mw",
    "noise_mw",
    "contention_priority",
)

OPTIONAL_PHYSICAL_ARRAYS = (
    "quality_lookahead_mbps",
    "future_dqi_oracle",
)

DQI_ARRAYS = (
    "dqi_predicted",
    "fresh_dqi",
    "dqi_age",
    "dqi_validity_slots",
    "dqi_source_slot",
    "dqi_input_cutoff_slot",
)

THETA_ARRAYS = (
    "theta",
    "theta_input_cutoff_slot",
    "theta_target_slot",
)


def save_descriptor_bundle(
    root: str | Path,
    trace: "MultiVehicleAccessTrace",
    *,
    extra_manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Persist a causal descriptor bundle as three NPZ files and a manifest."""

    bundle_root = Path(root)
    bundle_root.mkdir(parents=True, exist_ok=True)
    groups = {
        "physical_trace.npz": PHYSICAL_ARRAYS + OPTIONAL_PHYSICAL_ARRAYS,
        "dqi_predictions.npz": DQI_ARRAYS,
        "theta_predictions.npz": THETA_ARRAYS,
    }
    file_hashes: dict[str, str] = {}
    for filename, names in groups.items():
        path = bundle_root / filename
        np.savez_compressed(path, **{name: np.asarray(getattr(trace, name)) for name in names})
        file_hashes[filename] = _file_hash(path)

    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "trace_id": trace.trace_id,
        "split": trace.split,
        "seed": int(trace.seed),
        "slots": trace.slots,
        "num_channels": trace.num_channels,
        "num_vehicles": trace.num_vehicles,
        "trace_hash": trace.trace_hash(),
        "dqi_model_id": trace.dqi_model_id,
        "dqi_model_hash": trace.dqi_model_hash,
        "prediction_model_id": trace.prediction_model_id,
        "prediction_model_hash": trace.prediction_model_hash,
        "dqi_label_spec": dict(trace.dqi_metadata),
        "files": file_hashes,
    }
    manifest.update(
        {
            'dqi_inference_latency_ms': float(trace.dqi_inference_latency_ms),
            'dqi_parameter_count': int(trace.dqi_parameter_count),
            'prediction_inference_latency_ms': float(trace.prediction_inference_latency_ms),
            'prediction_parameter_count': int(trace.prediction_parameter_count),
            'dqi_validation_mse': float(trace.dqi_validation_mse),
            'prediction_validation_brier': float(trace.prediction_validation_brier),
        }
    )
    if extra_manifest:
        manifest.update(extra_manifest)
    (bundle_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return manifest


def load_descriptor_bundle(
    root: str | Path,
    *,
    require_frozen_models: bool = False,
) -> "MultiVehicleAccessTrace":
    """Load and validate an RL descriptor bundle without recomputing descriptors."""

    from cvn_rl.simulation.multi_traces import MultiVehicleAccessTrace

    bundle_root = Path(root)
    manifest = json.loads((bundle_root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported descriptor schema: {manifest.get('schema_version')}")
    if require_frozen_models:
        if bool(manifest.get('diagnostic_surrogate', False)) or bool(
            manifest.get('diagnostic_privileged', False)
        ):
            raise ValueError(
                'formal run rejects diagnostic surrogate or privileged descriptor bundles'
            )
        for key in ("dqi_model_hash", "prediction_model_hash"):
            value = str(manifest.get(key, ""))
            is_sha256 = len(value) == 64 and all(
                character in string.hexdigits for character in value
            )
            if not is_sha256 or 'surrogate' in value.lower():
                raise ValueError(f"formal run requires a frozen upstream model hash: {key}")
        for key in ('dqi_model_id', 'prediction_model_id'):
            value = str(manifest.get(key, ''))
            if not value or 'surrogate' in value.lower():
                raise ValueError(f'formal run requires a frozen upstream model ID: {key}')

    arrays: dict[str, np.ndarray] = {}
    for filename, names in {
        "physical_trace.npz": PHYSICAL_ARRAYS + OPTIONAL_PHYSICAL_ARRAYS,
        "dqi_predictions.npz": DQI_ARRAYS,
        "theta_predictions.npz": THETA_ARRAYS,
    }.items():
        path = bundle_root / filename
        expected_hash = manifest.get("files", {}).get(filename)
        if expected_hash and _file_hash(path) != expected_hash:
            raise ValueError(f"descriptor file hash mismatch: {filename}")
        with np.load(path, allow_pickle=False) as payload:
            required_names = (
                PHYSICAL_ARRAYS if filename == 'physical_trace.npz' else names
            )
            missing = [name for name in required_names if name not in payload]
            if missing:
                raise ValueError(f"{filename} missing arrays: {missing}")
            arrays.update({name: payload[name] for name in names if name in payload})

    quality = np.asarray(arrays['quality_mbps'])
    arrays.setdefault(
        'quality_lookahead_mbps',
        np.empty((0, *quality.shape[1:]), dtype=quality.dtype),
    )
    arrays.setdefault(
        'future_dqi_oracle',
        np.full(
            (quality.shape[0], quality.shape[2]),
            np.nan,
            dtype=float,
        ),
    )

    trace = MultiVehicleAccessTrace(
        **arrays,
        trace_id=str(manifest["trace_id"]),
        split=str(manifest["split"]),
        seed=int(manifest["seed"]),
        dqi_model_id=str(manifest["dqi_model_id"]),
        dqi_model_hash=str(manifest["dqi_model_hash"]),
        prediction_model_id=str(manifest["prediction_model_id"]),
        prediction_model_hash=str(manifest["prediction_model_hash"]),
        dqi_inference_latency_ms=float(manifest.get('dqi_inference_latency_ms', 0.0)),
        dqi_parameter_count=int(manifest.get('dqi_parameter_count', 0)),
        prediction_inference_latency_ms=float(
            manifest.get('prediction_inference_latency_ms', 0.0)
        ),
        prediction_parameter_count=int(manifest.get('prediction_parameter_count', 0)),
        dqi_validation_mse=float(manifest.get('dqi_validation_mse', np.nan)),
        prediction_validation_brier=float(
            manifest.get('prediction_validation_brier', np.nan)
        ),
        dqi_metadata=dict(manifest.get('dqi_label_spec', {})),
    )
    validate_descriptor_causality(trace)
    if trace.trace_hash() != manifest.get("trace_hash"):
        raise ValueError("descriptor trace hash does not match manifest")
    return trace


def validate_descriptor_causality(trace: "MultiVehicleAccessTrace") -> None:
    slots = np.arange(trace.slots, dtype=int)
    shared_shape = (trace.slots, trace.num_channels)
    for name in (
        'dqi_predicted',
        'fresh_dqi',
        'dqi_age',
        'dqi_validity_slots',
        'future_dqi_oracle',
        'bandwidth_mhz',
        'cci',
    ):
        if np.asarray(getattr(trace, name)).shape != shared_shape:
            raise ValueError(f'{name} must have shape [T, M]')
    link_shape = (trace.slots, trace.num_vehicles, trace.num_channels)
    for name in ('snr_db', 'rss_dbm'):
        if np.asarray(getattr(trace, name)).shape != link_shape:
            raise ValueError(f'{name} must have shape [T, V, M]')
    if trace.dqi_source_slot.shape != (trace.slots,):
        raise ValueError("dqi_source_slot must have shape [T]")
    if trace.dqi_input_cutoff_slot.shape != (trace.slots,):
        raise ValueError("dqi_input_cutoff_slot must have shape [T]")
    if np.any(trace.dqi_input_cutoff_slot > trace.dqi_source_slot):
        raise ValueError("DQI input cutoff occurs after its source slot")
    if np.any(trace.dqi_source_slot > slots):
        raise ValueError("DQI source occurs after the RL decision slot")
    if trace.theta_input_cutoff_slot.shape != (trace.slots,):
        raise ValueError("theta_input_cutoff_slot must have shape [T]")
    if trace.theta_target_slot.shape != (trace.slots,):
        raise ValueError("theta_target_slot must have shape [T]")
    if np.any(trace.theta_input_cutoff_slot >= trace.theta_target_slot):
        raise ValueError("prediction input must end before its target slot")
    if not np.array_equal(trace.theta_target_slot, slots):
        raise ValueError("one-step theta target must equal the current RL slot")


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
