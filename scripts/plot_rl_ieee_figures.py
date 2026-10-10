from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from cvn_rl.reporting.figures import write_expected_figures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create IEEE-style RL figures from a metrics CSV.")
    parser.add_argument("--metrics", required=True, help="Path to metrics CSV.")
    parser.add_argument("--out-dir", default=None, help="Output directory. Defaults to metrics run figures dir.")
    args = parser.parse_args(argv)

    metrics_path = Path(args.metrics)
    run_dir = metrics_path.parents[1] if args.out_dir is None else Path(args.out_dir).parent
    out_dir = run_dir / "figures" if args.out_dir is None else Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = pd.read_csv(metrics_path)
    expected = [
        f"figures/{metrics_path.stem}_plr_ieee.pdf",
        f"figures/{metrics_path.stem}_pu_violation_ieee.pdf",
    ]
    write_expected_figures(run_dir if args.out_dir is None else out_dir.parent, metrics, expected)
    print(f"Wrote figures under {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
