from __future__ import annotations

from typing import Any

import pandas as pd


FIXED_AGENTS = {'fixed_overlay', 'fixed_underlay'}


def evaluate_pilot_gate(
    metrics: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, bool]:
    if metrics.empty:
        return pd.DataFrame([{'check': 'nonempty_metrics', 'passed': False}]), False
    final = (
        metrics[metrics['is_final_checkpoint'].astype(bool)].copy()
        if 'is_final_checkpoint' in metrics
        else metrics.copy()
    )
    learned = final[~final['agent'].isin(FIXED_AGENTS)]
    gate_cfg = config.get('pilot_gate', {})
    gate_agents = set(gate_cfg.get('agents', []))
    if gate_agents:
        learned = learned[learned['agent'].isin(gate_agents)]
    gate_modules = set(gate_cfg.get('module_variants', []))
    if gate_modules and 'module_variant' in learned:
        learned = learned[learned['module_variant'].isin(gate_modules)]
    access_low, access_high = gate_cfg.get('access_ratio_range', [0.05, 0.95])
    contention_max = float(gate_cfg.get('scarce_contention_loss_max', 0.70))
    queue_fraction_max = float(gate_cfg.get('light_queue_fraction_max', 0.95))
    queue_capacity = max(float(config.get('packet_traffic', {}).get('queue_capacity_packets', 10)), 1.0)
    checks: list[dict[str, object]] = []
    for _, row in learned.iterrows():
        context = {'scenario': row['scenario'], 'agent': row['agent'], 'seed': int(row['seed'])}
        checks.append({**context, 'check': 'nonzero_throughput', 'value': row['system_vt_throughput'], 'passed': bool(row['system_vt_throughput'] > 0.0)})
        access = float(row['access_selection_ratio'])
        checks.append({**context, 'check': 'access_ratio', 'value': access, 'passed': bool(access_low <= access <= access_high)})
        if int(row['num_vehicles']) == 20:
            loss = float(row['contention_loss_rate'])
            checks.append({**context, 'check': 'scarce_contention_loss', 'value': loss, 'passed': bool(loss < contention_max)})
        if int(row['num_vehicles']) in {5, 10}:
            queue_fraction = float(row['mean_queue_length']) / queue_capacity
            checks.append({**context, 'check': 'queue_occupancy', 'value': queue_fraction, 'passed': bool(queue_fraction < queue_fraction_max)})
        unchanged = row.get('parameter_hash_before_eval') == row.get('parameter_hash_after_eval')
        checks.append({**context, 'check': 'evaluation_parameters_frozen', 'value': bool(unchanged), 'passed': bool(unchanged)})
    table = pd.DataFrame(checks)
    return table, bool(not table.empty and table['passed'].all())


def render_pilot_gate_report(table: pd.DataFrame, passed: bool) -> str:
    status = 'PASS' if passed else 'FAIL'
    failed = table[~table['passed'].astype(bool)] if not table.empty else table
    lines = ['# Multi-Module Pilot Gate Report', '', f'- Gate result: **{status}**', f'- Checks: {len(table)}', f'- Failed checks: {len(failed)}', '']
    if not failed.empty:
        lines.extend(['## Failed Checks', '', '```text', failed.to_string(index=False), '```', ''])
    lines.append('Formal runs are allowed only when every gate check passes.')
    return '\n'.join(lines) + '\n'
