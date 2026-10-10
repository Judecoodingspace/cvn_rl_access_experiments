"""Paired, condition-aware R5-C2 analysis; incomplete formal datasets fail closed."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from cvn_rl.experiments.r5c2_protocol import bundle_root, file_hash, read_json, write_json, fingerprint

METRICS = ['served_goodput_mbps', 'plr', 'pu_access_violation_rate', 'pu_interference_violation_rate']


def paired_interval(values):
    values = np.asarray(values, dtype=float)
    if not len(values) or not np.all(np.isfinite(values)):
        raise ValueError('finite paired seed values required')
    draws = np.random.default_rng(20260627).choice(values, size=(10000, len(values)), replace=True).mean(axis=1)
    return dict(mean=float(values.mean()), ci95_low=float(np.quantile(draws, .025)),
                ci95_high=float(np.quantile(draws, .975)), paired_seeds=len(values))


def load_complete_metrics(protocol):
    root = bundle_root(protocol)
    frozen = read_json(root / 'runs_frozen.json')
    if frozen['protocol_hash'] != protocol['protocol_hash']:
        raise ValueError('formal analysis protocol mismatch')
    if frozen['model_index_hash'] != file_hash(root / 'model_index.json'):
        raise ValueError('frozen model index changed')
    index = read_json(root / 'run_index.json')
    expected = {(t['condition_id'], int(t['seed'])) for t in frozen['tasks']}
    required = {(c, s) for c in [v['condition_id'] for v in protocol['conditions']] + ['B00']
                for s in protocol['entry']['rl_seeds']}
    if expected != required:
        raise ValueError('frozen run matrix does not match contract')
    rows = []
    for task in frozen['tasks']:
        key = f"{task['condition_id']}_s{task['seed']}"
        record = index.get(key)
        if not record or record['status'] != 'completed' or record['config_hash'] != task['config_hash']:
            raise ValueError(f'missing or incompatible completed task: {key}')
        if fingerprint(read_json(task['path'])) != task['config_hash']:
            raise ValueError('frozen case configuration changed')
        path = Path(record['run_dir']) / 'metrics/access_metrics.csv'
        if file_hash(path) != record['metrics_hash']:
            raise ValueError(f'metrics changed after completion: {key}')
        frame = pd.read_csv(path)
        final = frame[frame['is_final_checkpoint'].astype(bool)]
        if len(final) != 1:
            raise ValueError('one final aggregate per task required')
        row = final.iloc[0].to_dict()
        if (row['condition_id'], int(row['seed'])) != (task['condition_id'], task['seed']):
            raise ValueError('task identity mismatch')
        if not bool(row['evaluation_parameters_frozen']) or row.get('r5c2_execution_stage') != 'formal':
            raise ValueError('non-frozen or pilot result in formal analysis')
        if row['r5c2_protocol_hash'] != protocol['protocol_hash']:
            raise ValueError('result protocol mismatch')
        if not all(np.isfinite(row[m]) for m in METRICS):
            raise ValueError('nonfinite primary metrics')
        cfg = read_json(task['path'])
        scored_slots = cfg['final_checkpoint_evaluation']['eval_slots'] - cfg['r5c2_eval_warmup_slots']
        if row['evaluation_slots'] != scored_slots or row['scored_slot_records'] != scored_slots * cfg['final_checkpoint_evaluation']['eval_episodes']:
            raise ValueError('scored evaluation budget mismatch')
        rows.append(row)
    frame = pd.DataFrame(rows)
    for seed, group in frame.groupby('seed'):
        for field in ('scored_physical_hash', 'theta_observation_hash'):
            if group[field].nunique() != 1:
                raise ValueError(f'nonpaired {field} for seed {seed}')
    return frame


def summarize(protocol):
    root = bundle_root(protocol)
    frame = load_complete_metrics(protocol)
    tables = root / 'tables'
    tables.mkdir(parents=True, exist_ok=True)
    frame.to_csv(tables / 'seed_metrics.csv', index=False)
    baseline = frame[frame.condition_id == 'B00'].set_index('seed')
    aggregate, pairs = [], []
    for condition in protocol['conditions']:
        cid = condition['condition_id']
        selected = frame[frame.condition_id == cid].set_index('seed').loc[baseline.index]
        row = dict(condition)
        for metric in METRICS:
            values = selected[metric].to_numpy(float)
            row[metric] = float(np.mean(values))
            interval = paired_interval(values - baseline[metric].to_numpy(float))
            pairs.append(dict(condition_id=cid, metric=metric, **interval))
            if metric == 'served_goodput_mbps':
                row.update(delta_goodput=interval['mean'], delta_ci95_low=interval['ci95_low'],
                           delta_ci95_high=interval['ci95_high'], paired_seeds=interval['paired_seeds'])
        aggregate.append(row)
    pd.DataFrame(aggregate).to_csv(tables / 'conditions_summary.csv', index=False)
    headers = ['ID', 'qveh', 'qtime', 'H(ms)', 'D(ms)', 'Rule', 'G(Mbps)', 'Gain [95% CI]', 'PLR(%)', 'PU access(%)', 'PU interference(%)']
    display = []
    for row in aggregate:
        display.append([row['condition_id'], f"{row['vehicle_quantile']:.2f}", f"{row['time_quantile']:.2f}",
                        str(row['horizon_ms']), str(row['delay_max_ms']),
                        str(row['freshness_scale']) if row['freshness_enabled'] else 'off',
                        f"{row['served_goodput_mbps']:.3f}",
                        f"{row['delta_goodput']:.3f} [{row['delta_ci95_low']:.3f}, {row['delta_ci95_high']:.3f}]",
                        f"{100*row['plr']:.2f}", f"{100*row['pu_access_violation_rate']:.3f}",
                        f"{100*row['pu_interference_violation_rate']:.3f}"])
    markdown = '| ' + ' | '.join(headers) + ' |\n| ' + ' | '.join(['---']*len(headers)) + ' |\n'
    markdown += ''.join('| ' + ' | '.join(row) + ' |\n' for row in display)
    (tables / 'appendix_table.md').write_text(markdown, encoding='utf-8')
    tex_header = [h.replace('%', r'\%') for h in headers]
    tex = '\\begin{tabular}{lrrrrlrrrrr}\n\\hline\n' + ' & '.join(tex_header) + ' \\\\\n\\hline\n'
    tex += ''.join(' & '.join(row) + ' \\\\\n' for row in display) + '\\hline\n\\end{tabular}\n'
    (tables / 'appendix_table.tex').write_text(tex, encoding='utf-8')
    pd.DataFrame(pairs).to_csv(tables / 'paired_effects.csv', index=False)
    pivot = frame.pivot(index='seed', columns='condition_id', values='served_goodput_mbps')
    interactions = []
    seed_interactions = []
    for name, joint, qonly, honly in [('I1', 'C09', 'C03', 'C06'), ('I2', 'C10', 'C04', 'C05')]:
        values = pivot[joint] - pivot[qonly] - pivot[honly] + pivot.C00
        interactions.append(dict(interaction=name, joint_condition=joint, **paired_interval(values)))
        seed_interactions.extend(dict(interaction=name, seed=int(seed), value=float(value)) for seed, value in values.items())
    pd.DataFrame(interactions).to_csv(tables / 'interactions.csv', index=False)
    pd.DataFrame(seed_interactions).to_csv(tables / 'interaction_seeds.csv', index=False)
    write_json(root / 'analysis_manifest.json', dict(protocol_hash=protocol['protocol_hash'],
               execution_scope=protocol['entry']['execution_scope'], tasks=len(frame),
               tables={p.name: file_hash(p) for p in tables.iterdir() if p.is_file()}))
    return aggregate


def plot(protocol):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root = bundle_root(protocol)
    manifest = read_json(root / 'analysis_manifest.json')
    if manifest['protocol_hash'] != protocol['protocol_hash']:
        raise ValueError('analysis protocol mismatch')
    path = root / 'tables/conditions_summary.csv'
    if file_hash(path) != manifest['tables']['conditions_summary.csv']:
        raise ValueError('summary changed after analysis')
    frame = pd.read_csv(path).set_index('condition_id')
    fig, axes = plt.subplots(2, 2, figsize=(8, 6), sharey=True)
    def curve(ax, ids, x, label=None, marker='o'):
        selected = frame.loc[ids]
        y = selected.delta_goodput.to_numpy()
        errors = np.vstack([y - selected.delta_ci95_low.to_numpy(), selected.delta_ci95_high.to_numpy() - y])
        ax.errorbar(x, y, yerr=np.maximum(errors, 0), marker=marker, capsize=3, label=label)
    curve(axes[0, 0], ['C01', 'C00', 'C02'], [.05, .1, .2], 'vehicle quantile')
    curve(axes[0, 0], ['C03', 'C00', 'C04'], [.05, .1, .2], 'time quantile', 's')
    axes[0, 0].legend(fontsize=8)
    curve(axes[0, 1], ['C05', 'C00', 'C06'], [50, 100, 200])
    curve(axes[1, 0], ['C07', 'C00', 'C08'], [2.5, 5, 10])
    curve(axes[1, 1], ['C11', 'C00', 'C12', 'C13'], [0, 1, 2, 3])
    axes[1, 1].set_xticks([0, 1, 2, 3], ['0.5', '1', '2', 'off'])
    for ax, x in zip(axes.flat, [.1, 100, 5, 1]):
        ax.scatter([x], [frame.loc['C00', 'delta_goodput']], marker='*', s=85,
                   color='black', zorder=5, label='C00 reference')
    axes[0, 0].legend(fontsize=8)
    for ax, title, xlabel in zip(axes.flat,
        ['(a) Lower-tail quantiles', '(b) Future window', '(c) Serialization threshold', '(d) Freshness scale'],
        ['Quantile', 'H (ms)', 'Dmax (ms)', 'Validity scale / rule']):
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(xlabel)
        ax.axhline(0, color='grey', linestyle='--', linewidth=.7)
        ax.grid(alpha=.2)
    axes[0, 0].set_ylabel('Goodput gain over Prediction-only (Mbps)')
    axes[1, 0].set_ylabel('Goodput gain over Prediction-only (Mbps)')
    if protocol['entry']['execution_scope'] == 'smoke':
        fig.suptitle('SMOKE TEST — not research evidence')
    fig.tight_layout()
    out = root / 'figures'
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / 'dqi_sensitivity.pdf')
    fig.savefig(out / 'dqi_sensitivity.png', dpi=220)
    plt.close(fig)
    period = read_json(root / 'pilot/freshness_selection.json')['selected_update_slots']
    (out / 'caption.md').write_text(
        f'N=10, M=10; ESN-DDQN; common DQI update={period} slots (1 ms/slot). '
        'CCI is the existing synthetic coherence proxy, available at t-1. '
        'Fixed predictor recipe: MSE plus rank-hinge weight 0.1. '
        f'{len(protocol["entry"]["rl_seeds"])} paired training seeds; 95% seed bootstrap intervals. '
        'C09/C10 and local interactions are reported in the appendix table.\n', encoding='utf-8')
    manifest['figures'] = {p.name: file_hash(p) for p in out.iterdir() if p.is_file()}
    write_json(root / 'analysis_manifest.json', manifest)
    return str(out)


def verify(protocol):
    from cvn_rl.experiments.r5c2_data import _validate_prepared_files
    root = bundle_root(protocol)
    frame = load_complete_metrics(protocol)
    prepared = read_json(root / 'data/prepared_manifest.json')
    if file_hash(root / 'data/prepared_manifest.json') != read_json(root / 'protocol_frozen.json')['data_manifest_hash']:
        raise ValueError('prepared manifest changed after protocol freeze')
    _validate_prepared_files(root / 'data', prepared)
    models = read_json(root / 'model_index.json')
    if set(models) != {t['target_id'] for t in protocol['targets']}:
        raise ValueError('incomplete predictor index')
    for model in models.values():
        path = Path(model['model_path'])
        if file_hash(path) != model['model_sha256'] or file_hash(path.with_suffix('.meta.json')) != model['meta_sha256']:
            raise ValueError('predictor artifact changed')
    analysis = read_json(root / 'analysis_manifest.json')
    if not analysis.get('figures'):
        raise ValueError('figure/caption output required')
    for directory, fields in [('tables', analysis['tables']), ('figures', analysis['figures'])]:
        for name, digest in fields.items():
            if file_hash(root / directory / name) != digest:
                raise ValueError('analysis artifact changed')
    report = dict(passed=True, execution_scope=protocol['entry']['execution_scope'],
                  research_evidence=protocol['entry']['execution_scope'] == 'formal',
                  completed_rl_tasks=len(frame), predictors=len(models), protocol_hash=protocol['protocol_hash'])
    write_json(root / 'verification.json', report)
    return report
