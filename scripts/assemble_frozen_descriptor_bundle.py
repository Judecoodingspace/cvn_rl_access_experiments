from __future__ import annotations

import argparse
import hashlib
from dataclasses import replace
from pathlib import Path

import numpy as np

from cvn_rl.simulation.descriptor_bundle import (
    DQI_ARRAYS,
    THETA_ARRAYS,
    load_descriptor_bundle,
    save_descriptor_bundle,
    validate_descriptor_causality,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _load_arrays(path: Path, required: tuple[str, ...]) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        missing = [name for name in required if name not in payload]
        if missing:
            raise ValueError(f'{path} missing arrays: {missing}')
        return {name: payload[name] for name in required}


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Assemble a formal frozen descriptor bundle from upstream predictions.',
    )
    parser.add_argument('--base-bundle', required=True)
    parser.add_argument('--dqi-predictions', required=True)
    parser.add_argument('--theta-predictions', required=True)
    parser.add_argument('--dqi-checkpoint', required=True)
    parser.add_argument('--prediction-checkpoint', required=True)
    parser.add_argument('--dqi-model-id', required=True)
    parser.add_argument('--prediction-model-id', required=True)
    parser.add_argument('--dqi-parameter-count', required=True, type=int)
    parser.add_argument('--prediction-parameter-count', required=True, type=int)
    parser.add_argument('--dqi-latency-ms', required=True, type=float)
    parser.add_argument('--prediction-latency-ms', required=True, type=float)
    parser.add_argument('--dqi-validation-mse', type=float, default=float('nan'))
    parser.add_argument('--prediction-validation-brier', type=float, default=float('nan'))
    parser.add_argument(
        '--validation-type',
        default='hybrid_berlin_link_synthetic_pu',
    )
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    dqi_checkpoint = Path(args.dqi_checkpoint)
    prediction_checkpoint = Path(args.prediction_checkpoint)
    trace = load_descriptor_bundle(args.base_bundle)
    replacements = {
        **_load_arrays(Path(args.dqi_predictions), DQI_ARRAYS),
        **_load_arrays(Path(args.theta_predictions), THETA_ARRAYS),
        'dqi_model_id': args.dqi_model_id,
        'dqi_model_hash': _sha256(dqi_checkpoint),
        'prediction_model_id': args.prediction_model_id,
        'prediction_model_hash': _sha256(prediction_checkpoint),
        'dqi_parameter_count': args.dqi_parameter_count,
        'prediction_parameter_count': args.prediction_parameter_count,
        'dqi_inference_latency_ms': args.dqi_latency_ms,
        'prediction_inference_latency_ms': args.prediction_latency_ms,
        'dqi_validation_mse': args.dqi_validation_mse,
        'prediction_validation_brier': args.prediction_validation_brier,
    }
    formal_trace = replace(trace, **replacements)
    validate_descriptor_causality(formal_trace)
    manifest = save_descriptor_bundle(
        args.output,
        formal_trace,
        extra_manifest={
            'descriptor_source': 'frozen_upstream',
            'diagnostic_surrogate': False,
            'validation_type': args.validation_type,
            'dqi_checkpoint_sha256': formal_trace.dqi_model_hash,
            'prediction_checkpoint_sha256': formal_trace.prediction_model_hash,
        },
    )
    load_descriptor_bundle(args.output, require_frozen_models=True)
    print(
        f"Formal bundle written: {args.output} "
        f"(trace_hash={manifest['trace_hash']})"
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
