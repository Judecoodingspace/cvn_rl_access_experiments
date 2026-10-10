from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from cvn_rl.simulation.descriptor_bundle import load_descriptor_bundle


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Validate every formal descriptor bundle below a directory.',
    )
    parser.add_argument('root')
    parser.add_argument('--output')
    args = parser.parse_args()

    root = Path(args.root)
    manifests = sorted(root.rglob('manifest.json'))
    if not manifests:
        raise FileNotFoundError(f'no descriptor manifests below {root}')
    rows = []
    seen = set()
    for manifest in manifests:
        trace = load_descriptor_bundle(
            manifest.parent,
            require_frozen_models=True,
        )
        key = (trace.split, trace.seed, trace.trace_id)
        if key in seen:
            raise ValueError(f'duplicate frozen bundle identity: {key}')
        seen.add(key)
        rows.append(
            {
                'bundle': str(manifest.parent),
                'trace_id': trace.trace_id,
                'split': trace.split,
                'seed': trace.seed,
                'slots': trace.slots,
                'num_channels': trace.num_channels,
                'num_vehicles': trace.num_vehicles,
                'trace_hash': trace.trace_hash(),
                'dqi_model_id': trace.dqi_model_id,
                'dqi_model_hash': trace.dqi_model_hash,
                'prediction_model_id': trace.prediction_model_id,
                'prediction_model_hash': trace.prediction_model_hash,
            }
        )
    table = pd.DataFrame(rows)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(output, index=False)
    print(table.to_string(index=False))
    print(f'Validated {len(table)} frozen descriptor bundles.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
