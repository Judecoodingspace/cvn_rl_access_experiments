from pathlib import Path
import numpy as np
import pandas as pd
from .protocol import bundle_root,read_json,write_json,file_hash,check_protected

def interval(values):
    x=np.asarray(values,float); rng=np.random.default_rng(20260627)
    draws=rng.choice(x,size=(10000,len(x)),replace=True).mean(1)
    return float(x.mean()),float(np.quantile(draws,.025)),float(np.quantile(draws,.975))

def summarize(p):
    root=bundle_root(p); index=read_json(root/'run_index.json'); expected={f'{c}_s{s}' for c in [x['condition_id'] for x in p['conditions']]+['B00'] for s in p['entry']['rl_seeds']}
    if set(index)!=expected or any(x['status']!='completed' for x in index.values()): raise ValueError('complete paired task matrix required')
    rows=[]
    for key,item in index.items():
        result=read_json(root/'rl_runs'/key/'result.json')
        if file_hash(root/'rl_runs'/key/'result.json')!=item['result_hash'] or result['protocol_hash']!=p['protocol_hash']: raise ValueError('result identity/hash mismatch')
        ref=index[f"B00_s{result['seed']}"]['result']
        if result['physical_hashes']!=ref['physical_hashes'] or result['train_physical_hash']!=ref['train_physical_hash'] or result['theta_hashes']!=ref['theta_hashes']: raise ValueError('unpaired physical/theta traces')
        episodes=result['episodes']
        row={k:result[k] for k in ['condition_id','seed','group','period_slots','warmup_slots','research_evidence']}
        for name in ['goodput_mbps','per_vt_goodput_mbps','plr','pu_access_rate','pu_interference_rate','pu_active_interference_rate']:
            row[name]=float(np.mean([x[name] for x in episodes]))
        for name in ['served_bits','generated_packets','dropped_packets','completed_packets','completed_scored_arrivals','unfinished_scored_arrivals',
                     'carry_in_packets','carry_out_packets','pu_access_count','transmission_count','pu_interference_count','channel_slots','busy_channel_slots','scored_slots']:
            row[name]=float(sum(x[name] for x in episodes))
        rows.append(row)
    data=pd.DataFrame(rows); baseline=data[data.condition_id=='B00'].set_index('seed'); effects=[]
    for c in p['conditions']:
        d=data[data.condition_id==c['condition_id']].set_index('seed').reindex(p['entry']['rl_seeds'])
        reference=data[data.condition_id==('C00' if c['group']=='main' else 'C14')].set_index('seed').reindex(p['entry']['rl_seeds'])
        values=d.goodput_mbps-baseline.goodput_mbps; mean,low,high=interval(values)
        rm,rl,rh=interval(d.goodput_mbps-reference.goodput_mbps)
        effects.append({**c,'delta_g_dqi':mean,'low95':low,'high95':high,'delta_reference':rm,'reference_low95':rl,'reference_high95':rh,
                        'full_goodput':float(d.goodput_mbps.mean()),'full_plr':float(d.plr.mean()),'full_pu_interference_rate':float(d.pu_interference_rate.mean())})
    interactions=[]
    for label,ids in [('J1',['C09','C03','C06','C00']),('J2',['C10','C04','C05','C00'])]:
        vectors=[data[data.condition_id==cid].set_index('seed').reindex(p['entry']['rl_seeds']).goodput_mbps for cid in ids]
        x=vectors[0]-vectors[1]-vectors[2]+vectors[3]; mean,low,high=interval(x)
        interactions.append({'combination':label,'interaction_mbps':mean,'low95':low,'high95':high,'seed_values':x.tolist()})
    tables=root/'tables'; tables.mkdir(parents=True,exist_ok=True)
    data.to_csv(tables/'seed_metrics.csv',index=False); pd.DataFrame(effects).to_csv(tables/'paired_effects.csv',index=False); pd.DataFrame(interactions).to_csv(tables/'interactions.csv',index=False)
    # Raw episode counters make denominators and carry-in/out independently auditable.
    episode_rows=[{**{k:item['result'][k] for k in ['condition_id','seed','period_slots']},**ep} for item in index.values() for ep in item['result']['episodes']]
    pd.DataFrame(episode_rows).to_csv(tables/'episode_metrics.csv',index=False)
    table=pd.DataFrame(effects)
    columns=['condition_id','group','full_goodput','delta_g_dqi','low95','high95','full_plr','full_pu_interference_rate']
    lines=['| '+' | '.join(columns)+' |','| '+' | '.join(['---']*len(columns))+' |']
    tex=[r'\begin{tabular}{llrrrrrr}',r'Condition & Group & Goodput & Gain & CI low & CI high & PLR & PU interference \\']
    for values in table[columns].itertuples(index=False,name=None):
        text=[str(x) if isinstance(x,str) else f'{x:.5f}' for x in values]
        lines.append('| '+' | '.join(text)+' |'); tex.append(' & '.join(text)+r' \\')
    tex.append(r'\end{tabular}')
    (tables/'performance_appendix.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (tables/'performance_appendix.tex').write_text('\n'.join(tex)+'\n',encoding='utf-8')
    manifest={'protocol_hash':p['protocol_hash'],'tasks':len(expected),'research_evidence':p['entry']['execution_scope']=='formal',
              'tables':{str(f.relative_to(root)):file_hash(f) for f in tables.iterdir() if f.is_file()}}
    write_json(root/'analysis_manifest.json',manifest)
    (root/'response_evidence_map.md').write_text('# R5-C2 evidence map\n\n'
      '| Question | Conditions | Evidence |\n|---|---|---|\n'
      '| Vehicle/time quantiles | C00-C04 | panel a, paired_effects, label diagnostics |\n'
      '| Horizon | C00,C05,C06 | panel b; J1/J2 appendix |\n'
      '| Latency threshold | C00,C07,C08 | panel c; fixed actual packet task |\n'
      '| Freshness validity | C00,C11-C17 | panel d, separate main/stress references |\n'
      '| Decay shape | C14,C17,C18 | shape comparison, same validity endpoints |\n\n'
      'Report measured paired effects, PLR and actual interference violations. Restrict conclusions to tested scene, ESN-DDQN, ranges and update periods. '
      'Reference q95 and beta=2ms are not claimed end-to-end sweeps. New physical traces do not reproduce old CSV values.\n',encoding='utf-8')
    return manifest

def plot(p):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=bundle_root(p); manifest=read_json(root/'analysis_manifest.json'); path=root/'tables/paired_effects.csv'
    if file_hash(path)!=manifest['tables']['tables/paired_effects.csv']: raise ValueError('analysis table changed')
    data=pd.read_csv(path).set_index('condition_id'); fig,axes=plt.subplots(2,2,figsize=(11,7))
    def curve(ax,ids,x,label):
        d=data.loc[ids]; y=d.delta_g_dqi.to_numpy(); ax.errorbar(x,y,yerr=np.vstack([y-d.low95,d.high95-y]),marker='o',capsize=3,label=label)
    curve(axes[0,0],['C01','C00','C02'],[.05,.1,.2],'vehicle quantile')
    curve(axes[0,0],['C03','C00','C04'],[.05,.1,.2],'time quantile'); axes[0,0].set_xlabel('Lower-tail quantile')
    curve(axes[0,1],['C05','C00','C06'],[50,100,200],'main: 10 ms'); axes[0,1].set_xlabel('Future horizon (ms)')
    curve(axes[1,0],['C07','C00','C08'],[2.5,5,10],'main: 10 ms'); axes[1,0].set_xlabel('Label delay threshold (ms)')
    curve(axes[1,1],['C11','C00','C12','C13'],np.arange(4)-.06,'main: 10 ms')
    curve(axes[1,1],['C15','C14','C16','C17'],np.arange(4)+.06,'stress'); axes[1,1].set_xticks(range(4),['0.5','1','2','off']); axes[1,1].set_xlabel('Validity scale / discount disabled')
    for i,ax in enumerate(axes.flat): ax.axhline(0,color='grey',lw=.8); ax.set_ylabel('DQI goodput gain (Mbps)'); ax.grid(alpha=.2); ax.legend(fontsize=8); ax.set_title(f'({chr(97+i)})')
    if not manifest['research_evidence']: fig.suptitle('SMOKE CHECK ONLY - NOT RESEARCH EVIDENCE',color='red')
    fig.tight_layout(); out=root/'figures'; out.mkdir(parents=True,exist_ok=True)
    for ext in ['pdf','png']: fig.savefig(out/f'dqi_sensitivity_v2.{ext}',dpi=220,bbox_inches='tight')
    plt.close(fig)
    shape=data.loc[['C14','C17','C18']]; fig,ax=plt.subplots(figsize=(6,4)); y=shape.delta_g_dqi.to_numpy()
    ax.bar(['Exponential','Discount off','Endpoint-matched linear'],y,yerr=np.vstack([y-shape.low95,shape.high95-y]),capsize=4)
    ax.axhline(0,color='grey'); ax.set_ylabel('DQI goodput gain (Mbps)'); fig.tight_layout()
    for ext in ['pdf','png']: fig.savefig(out/f'decay_shape_v2.{ext}',dpi=220,bbox_inches='tight')
    plt.close(fig)
    manifest['figures']={str(f.relative_to(root)):file_hash(f) for f in out.iterdir() if f.is_file()}; write_json(root/'analysis_manifest.json',manifest)
    (out/'caption.md').write_text('N=10, M=10 competitive mixed-mode; same ESN-DDQN recipe with independently trained policies. '
     'Main DQI updates10ms, stress updates selected by age coverage; rho is received Doppler coherence. '
     'Five paired training seeds; 95% paired bootstrap intervals. Gains relative to matched Prediction-only. '
     'Freshness main/stress references are C00/C14. Curve-shape comparison uses identical validity boundaries.\n',encoding='utf-8')
    return {'figures':list(manifest['figures'])}

def verify(p):
    root=bundle_root(p); manifest=read_json(root/'analysis_manifest.json')
    if manifest['protocol_hash']!=p['protocol_hash']: raise ValueError('wrong analysis protocol')
    for rel,h in {**manifest['tables'],**manifest.get('figures',{})}.items():
        if file_hash(root/rel)!=h: raise ValueError(f'output hash mismatch {rel}')
    if not manifest.get('figures'): raise ValueError('plots missing')
    protection=check_protected(); result={'passed':True,'protocol_hash':p['protocol_hash'],'tasks':manifest['tasks'],
                                        'research_evidence':manifest['research_evidence'],'old_files':protection}
    write_json(root/'verification.json',result); return result
