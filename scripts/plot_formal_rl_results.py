from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from cvn_rl.reporting.multi_vehicle_figures import write_multi_vehicle_figures


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Regenerate paper figures only from a formal result CSV.',
    )
    parser.add_argument('run_dir')
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    manifest = json.loads(
        (run_dir / 'manifest.json').read_text(encoding='utf-8')
    )
    if manifest.get('dry_run') or manifest.get('quick'):
        raise ValueError('diagnostic dry/quick results cannot generate formal figures')
    metrics = pd.read_csv(run_dir / 'metrics' / 'access_metrics.csv')
    written = write_multi_vehicle_figures(run_dir, metrics)
    print(f'Regenerated {len(written)} formal figure files from access_metrics.csv.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
