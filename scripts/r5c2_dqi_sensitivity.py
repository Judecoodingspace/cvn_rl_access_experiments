#!/usr/bin/env python3
"""Staged R5-C2 sensitivity pipeline. Formal commands require frozen inputs."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from cvn_rl.experiments.r5c2_protocol import resolve_protocol, write_resolution
from cvn_rl.experiments.r5c2_data import prepare_data
from cvn_rl.experiments.r5c2_execution import train_predictors, pilot_freshness, pilot_runtime, freeze, run_cases


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['resolve', 'prepare-data', 'train-predictors', 'pilot-freshness',
                                          'pilot-runtime', 'freeze', 'run-rl', 'summarize', 'plot', 'verify'])
    parser.add_argument('--config', default='configs/r2_r05c02_dqi_sensitivity.json')
    parser.add_argument('--stage', choices=['pilot', 'formal'], default='formal')
    parser.add_argument('--phase', choices=['protocol', 'runs'], default='protocol')
    parser.add_argument('--targets')
    parser.add_argument('--conditions')
    parser.add_argument('--seeds')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--skip-full-probe', action='store_true', help='smoke only; cannot freeze formal protocol')
    args = parser.parse_args(argv)
    protocol = resolve_protocol(args.config)
    root = write_resolution(protocol)
    targets = args.targets.split(',') if args.targets else None
    conditions = args.conditions.split(',') if args.conditions else None
    seeds = [int(v) for v in args.seeds.split(',')] if args.seeds else None
    if args.command == 'resolve':
        result = {'protocol_hash': protocol['protocol_hash'], 'conditions': 14, 'targets': 11}
    elif args.command == 'prepare-data':
        result = prepare_data(protocol, args.resume)
    elif args.command == 'train-predictors':
        result = train_predictors(protocol, args.stage, targets, args.resume)
    elif args.command == 'pilot-freshness':
        result = pilot_freshness(protocol)
    elif args.command == 'pilot-runtime':
        result = pilot_runtime(protocol, conditions, seeds, not args.skip_full_probe, args.resume)
    elif args.command == 'freeze':
        result = freeze(protocol, args.phase)
    elif args.command == 'run-rl':
        result = run_cases(protocol, args.stage, conditions, seeds, args.resume)
    else:
        from cvn_rl.experiments.r5c2_analysis import summarize, plot, verify
        result = {'summarize': summarize, 'plot': plot, 'verify': verify}[args.command](protocol)
    print(f'{args.command} completed: {root}', flush=True)
    return result


if __name__ == '__main__':
    main()
