"""Generate R5-C2 tables and the four-panel figure from completed formal tasks."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from cvn_rl.experiments.r5c2_protocol import resolve_protocol
from cvn_rl.experiments.r5c2_analysis import summarize, plot, verify

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/r2_r05c02_dqi_sensitivity.json')
    args = parser.parse_args()
    protocol = resolve_protocol(args.config)
    summarize(protocol)
    plot(protocol)
    print(verify(protocol))
