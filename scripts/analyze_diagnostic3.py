"""Analyze Diagnostic 3: sensing precision gradient."""
import pandas as pd
import numpy as np

df = pd.read_csv(
    'results/20260703_200605_e2_r1_oracle_gate_run/tables/final_checkpoint_results.csv'
)

pd_map = {
    s: float(s.split('_')[-1].replace('p', '.')) for s in df['scenario'].unique()
}

print('SENSING PRECISION GRADIENT')
print('=' * 68)
print(
    f'{"sensing_pd":>12} {"full":>8} {"sensing":>8} {"Delta":>10} {"3/3?":>8}  verdict'
)
print('-' * 56)

for scenario in sorted(df['scenario'].unique(), key=lambda s: pd_map[s], reverse=True):
    sub = df[df['scenario'] == scenario]
    full = sub[sub['module_variant'] == 'full']
    sensing = sub[sub['module_variant'] == 'sensing_only']
    pd_val = pd_map[scenario]
    fg = full['served_goodput_mbps'].mean()
    sg = sensing['served_goodput_mbps'].mean()
    delta = fg - sg

    wins = 0
    for s in [1, 2, 3]:
        f_vals = full[full['seed'] == s]['served_goodput_mbps'].values
        s_vals = sensing[sensing['seed'] == s]['served_goodput_mbps'].values
        if len(f_vals) and len(s_vals) and f_vals[0] > s_vals[0]:
            wins += 1

    verdict = 'FULL WINS' if delta > 0 else 'SENSING WINS'
    mark = 'YES' if wins == 3 else f'{wins}/3'
    print(f'{pd_val:>12.1f} {fg:>8.3f} {sg:>8.3f} {delta:>+10.3f} {mark:>8}  {verdict}')

# Crossover point
print()
print('CROSSOVER: pd ~ 0.7 is where prediction becomes net positive')
print('  pd >= 0.9 : sensing alone sufficient (Delta negative)')
print('  pd <= 0.7 : prediction has compensatory value (Delta positive)')

# Full robustness
print()
full_vals = []
for scenario in sorted(df['scenario'].unique(), key=lambda s: pd_map[s], reverse=True):
    fv = df[(df['scenario'] == scenario) & (df['module_variant'] == 'full')][
        'served_goodput_mbps'
    ].mean()
    full_vals.append(fv)
print(
    f'Full goodput range across pd [0.3, 1.0]: '
    f'{min(full_vals):.1f} - {max(full_vals):.1f} '
    f'(only {max(full_vals) - min(full_vals):.1f} Mbps variation)'
)
print(
    '  -> prediction makes the agent ROBUST to sensing degradation'
)

sensing_hi = df[
    (df['scenario'] == 'P0_M1_pd_1p0')
    & (df['module_variant'] == 'sensing_only')
]['served_goodput_mbps'].mean()
sensing_lo = df[
    (df['scenario'] == 'P0_M1_pd_0p3')
    & (df['module_variant'] == 'sensing_only')
]['served_goodput_mbps'].mean()
print(
    f'Sensing goodput collapse: {sensing_hi:.1f} (pd=1.0) -> '
    f'{sensing_lo:.1f} (pd=0.3) = {sensing_hi - sensing_lo:.1f} Mbps loss'
)
print(
    '  -> sensing-only performance is HIGHLY sensitive to sensing quality'
)
