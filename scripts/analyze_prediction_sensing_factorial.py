"""Analyze the preregistered sensing-prediction 2x2 pilot."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CONDITIONS=['neither','sensing_only','prediction_only','full']
METRICS=['served_goodput_mbps','plr','pu_access_violation_rate']

def bootstrap(values, seed=20260703, repetitions=10000):
    values=np.asarray(values,dtype=float); rng=np.random.default_rng(seed)
    means=values[rng.integers(0,len(values),size=(repetitions,len(values)))].mean(axis=1)
    return float(values.mean()),float(np.quantile(means,.025)),float(np.quantile(means,.975))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('run_dir'); a=ap.parse_args(); run=Path(a.run_dir)
    data=pd.read_csv(run/'tables'/'final_checkpoint_results.csv'); data=data[data['is_final_checkpoint'].astype(bool)]
    index=['scenario','seed']; pivot=data.pivot_table(index=index,columns='module_variant',values=METRICS,aggfunc='mean')
    effects=[]
    comparisons=[('prediction_given_sensing','full','sensing_only'),('sensing_given_prediction','full','prediction_only'),('prediction_without_sensing','prediction_only','neither'),('sensing_without_prediction','sensing_only','neither')]
    for metric in METRICS:
        for name,left,right in comparisons:
            values=(pivot[(metric,left)]-pivot[(metric,right)]).to_numpy(); mean,lo,hi=bootstrap(values)
            effects.append({'effect':name,'metric':metric,'left':left,'right':right,'mean_difference':mean,'ci95_low':lo,'ci95_high':hi,'seed_differences':'|'.join(f'{v:.8g}' for v in values)})
        interaction=(pivot[(metric,'full')]-pivot[(metric,'sensing_only')]-pivot[(metric,'prediction_only')]+pivot[(metric,'neither')]).to_numpy(); mean,lo,hi=bootstrap(interaction)
        effects.append({'effect':'interaction','metric':metric,'left':'full-sensing_only-prediction_only+neither','right':'','mean_difference':mean,'ci95_low':lo,'ci95_high':hi,'seed_differences':'|'.join(f'{v:.8g}' for v in interaction)})
    effects_df=pd.DataFrame(effects); effects_df.to_csv(run/'tables'/'prediction_sensing_factorial_effects.csv',index=False)
    summary=data.groupby('module_variant',as_index=False)[METRICS].agg(['mean','std']).reset_index(); summary.to_csv(run/'tables'/'prediction_sensing_factorial_summary.csv',index=False)
    fig,axes=plt.subplots(1,3,figsize=(10,3.2)); labels={'neither':'Neither','sensing_only':'Sensing only','prediction_only':'Prediction only','full':'Full'}
    for ax,metric,ylabel in zip(axes,METRICS,['Goodput (Mbps)','PLR','PU access violation']):
        for i,c in enumerate(CONDITIONS):
            values=data[data.module_variant==c].sort_values('seed')[metric].to_numpy(); ax.scatter(np.full(len(values),i),values,s=22); ax.plot([i-.12,i+.12],[values.mean(),values.mean()],color='black',lw=1)
        ax.set_xticks(range(4)); ax.set_xticklabels([labels[c] for c in CONDITIONS],rotation=25,ha='right'); ax.set_ylabel(ylabel); ax.grid(axis='y',alpha=.25)
    fig.tight_layout(); fig.savefig(run/'figures'/'prediction_sensing_factorial_conditions.pdf'); fig.savefig(run/'figures'/'prediction_sensing_factorial_conditions.jpg',dpi=300); plt.close(fig)
    selected=effects_df[effects_df.effect.isin(['prediction_given_sensing','sensing_given_prediction','interaction'])]
    fig,axes=plt.subplots(1,3,figsize=(10,3.0))
    for ax,(metric,ylabel) in zip(axes,zip(METRICS,['Goodput difference (Mbps)','PLR difference','PU violation difference'])):
        rows=selected[selected.metric==metric]; x=np.arange(len(rows)); ax.errorbar(x,rows.mean_difference,yerr=[rows.mean_difference-rows.ci95_low,rows.ci95_high-rows.mean_difference],fmt='o',capsize=3); ax.axhline(0,color='black',lw=.8); ax.set_xticks(x); ax.set_xticklabels(['Pred | Sense','Sense | Pred','Interaction'],rotation=25,ha='right'); ax.set_ylabel(ylabel); ax.grid(axis='y',alpha=.25)
    fig.tight_layout(); fig.savefig(run/'figures'/'prediction_sensing_factorial_effects.pdf'); fig.savefig(run/'figures'/'prediction_sensing_factorial_effects.jpg',dpi=300); plt.close(fig)
    audit={'passed':False,'stop_followup':True,'reason':'Pilot A gate failed: prediction conditional value negative and full Pareto-dominated by sensing_only.','rows':len(data),'conditions':sorted(data.module_variant.unique()),'seeds':sorted(int(v) for v in data.seed.unique())}
    (run/'analysis'/'prediction_sensing_factorial_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')

if __name__=='__main__': main()
