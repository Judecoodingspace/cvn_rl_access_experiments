"""Publication layout for validated R5-C2 aggregate data; never modifies experiments."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from cvn_rl.experiments.r5c2_protocol import resolve_protocol, bundle_root, read_json, write_json, file_hash


def render(protocol):
    root = bundle_root(protocol)
    manifest = read_json(root / 'analysis_manifest.json')
    source = root / 'tables/conditions_summary.csv'
    if manifest['protocol_hash'] != protocol['protocol_hash'] or file_hash(source) != manifest['tables']['conditions_summary.csv']:
        raise ValueError('validated summary and protocol required')
    frame = pd.read_csv(source).set_index('condition_id')
    fig, axes = plt.subplots(2, 2, figsize=(8, 6), sharey=True, layout='constrained')
    series = [
        [(['C01','C00','C02'], [.05,.10,.20], 'Vehicle quantile', 'o'),
         (['C03','C00','C04'], [.05,.10,.20], 'Time quantile', 's')],
        [(['C05','C00','C06'], [50,100,200], None, 'o')],
        [(['C07','C00','C08'], [2.5,5,10], None, 'o')],
        [(['C11','C00','C12','C13'], [0,1,2,3], None, 'o')],
    ]
    for ax, lines, ref_x, title, label in zip(
        axes.flat, series, [.10,100,5,1],
        ['(a) Lower-tail quantiles','(b) Future window','(c) Serialization threshold','(d) Freshness'],
        ['Quantile',r'$H$ (ms)',r'$D_{\max}$ (ms)','Validity scale / rule']):
        for ids, x, legend, marker in lines:
            selected = frame.loc[ids]
            y = selected.delta_goodput.to_numpy()
            error = np.maximum(np.vstack([y-selected.delta_ci95_low, selected.delta_ci95_high-y]), 0)
            ax.errorbar(x, y, yerr=error, capsize=3, marker=marker, label=legend)
        ax.scatter([ref_x], [frame.loc['C00','delta_goodput']], color='black', marker='*', s=80,
                   zorder=5, label='Reference')
        ax.axhline(0, color='grey', linestyle='--', linewidth=.7)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(label)
        ax.grid(alpha=.2)
    axes[0,0].set_xticks([.05,.10,.20])
    axes[0,1].set_xticks([50,100,200])
    axes[1,0].set_xticks([2.5,5,10])
    axes[1,1].set_xticks([0,1,2,3], ['0.5','1','2','off'])
    axes[0,0].legend(fontsize=8)
    axes[0,0].set_ylabel(r'$\Delta G$ (Mbps)')
    axes[1,0].set_ylabel(r'$\Delta G$ (Mbps)')
    if protocol['entry']['execution_scope'] == 'smoke':
        fig.suptitle('SMOKE TEST — not research evidence', fontsize=11)
    out = root / 'figures'
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / 'dqi_sensitivity.pdf')
    fig.savefig(out / 'dqi_sensitivity.png', dpi=220)
    plt.close(fig)
    if not (out / 'caption.md').exists():
        period = read_json(root / 'pilot/freshness_selection.json')['selected_update_slots']
        (out / 'caption.md').write_text(
            f'ESN-DDQN; N=10/M=10; DQI update={period} ms; synthetic causal CCI proxy. '
            f'{len(protocol["entry"]["rl_seeds"])} paired seeds; seed bootstrap 95% intervals; '
            'fixed MSE + rank-hinge training weight 0.1. Delta G is Full minus Prediction-only.\n', encoding='utf-8')
    manifest['figures'] = {p.name:file_hash(p) for p in out.iterdir() if p.is_file()}
    manifest['renderer'] = {'path':str(Path(__file__)), 'sha256':file_hash(__file__)}
    write_json(root / 'analysis_manifest.json', manifest)
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/r2_r05c02_dqi_sensitivity.json')
    args = parser.parse_args()
    print(render(resolve_protocol(args.config)))
