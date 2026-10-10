from __future__ import annotations

import hashlib
import json
from pathlib import Path
import numpy as np
from cvn_rl.experiments.config import load_config

ROOT = Path(__file__).resolve().parents[4]
VERSION = 'r5c2_v2'
FEATURES = ['snr_q10_db', 'rss_q10_dbm', 'bandwidth_mhz', 'doppler_coherence']

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()

def file_hash(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as s:
        for c in iter(lambda:s.read(1024*1024), b''): h.update(c)
    return h.hexdigest()

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def project_path(path):
    p=Path(path)
    return p if p.is_absolute() else ROOT/p

def write_json(path, value):
    p=Path(path).resolve()
    if not p.is_relative_to(ROOT): raise ValueError('writes must stay in workspace')
    p.parent.mkdir(parents=True,exist_ok=True)
    q=p.with_suffix(p.suffix+'.partial')
    q.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    q.replace(p)

def conditions():
    params=[(.1,.1,100,5),(.05,.1,100,5),(.2,.1,100,5),(.1,.05,100,5),(.1,.2,100,5),
            (.1,.1,50,5),(.1,.1,200,5),(.1,.1,100,2.5),(.1,.1,100,10),(.1,.05,200,5),(.1,.2,50,5)]
    rows=[dict(condition_id=f'C{i:02}',target_id=f'T{i:02}',qv=v,qt=t,horizon_ms=h,delay_ms=d,
               group='main',rule='exp',scale=1.) for i,(v,t,h,d) in enumerate(params)]
    for i,group,rule,scale in [(11,'main','exp',.5),(12,'main','exp',2.),(13,'main','off',1.),
                               (14,'stress','exp',1.),(15,'stress','exp',.5),(16,'stress','exp',2.),
                               (17,'stress','off',1.),(18,'stress','linear',1.)]:
        rows.append({**rows[0],'condition_id':f'C{i:02}','group':group,'rule':rule,'scale':scale})
    return rows

def target_spec(c):
    return {k:c[k] for k in ['target_id','qv','qt','horizon_ms','delay_ms']}

def check_protected():
    p=ROOT/'revision_round2/review_comments/action_plans/r5c2/r5c2_v2_protected_sources.json'
    snapshot=read_json(p)
    changed=[n for n,h in snapshot['files'].items() if not (ROOT/n).is_file() or file_hash(ROOT/n)!=h]
    if changed: raise RuntimeError(f'protected workspace files changed: {changed}')
    return {'protected_files':len(snapshot['files']),'changed':changed,'snapshot_sha256':file_hash(p)}

def resolve(path):
    from .physics_spec import validate
    entry=load_config(project_path(path)); entry.pop('_config_path',None)
    if entry['protocol_version']!=VERSION: raise ValueError('v2 configuration required')
    scope=entry['execution_scope']
    if scope not in ['formal','smoke']: raise ValueError('invalid scope')
    out=project_path(entry['output_root']).resolve()
    required=ROOT/('results/revision_round2' if scope=='formal' else 'tmp')
    if not out.is_relative_to(required) or 'v2' not in out.name: raise ValueError('isolated v2 output root required')
    if entry['agent']!='proposed_esn_ddqn' or entry['num_channels']!=10 or entry['history_slots']!=200 or entry['slot_duration_ms']!=1:
        raise ValueError('contract requires ESN, M10, W200 and 1ms')
    if entry['label_num_vehicles']!=15 or entry['rl_num_vehicles']!=10: raise ValueError('N15 labels/N10 access required')
    if entry['reference_quantile']!=.95 or entry['input_vehicle_quantile']!=.1: raise ValueError('fixed scaling/aggregation')
    r=entry['predictor_recipe']
    if r['loss']!='mse' or r['rank_loss_weight']!=0: raise ValueError('MSE-only optimization required')
    if entry['freshness']['main_update_slots']!=10 or entry['prediction']['update_slots']!=10: raise ValueError('main guidance10ms')
    if entry['freshness']['compute_transport_delay_ms']!=0: raise ValueError('nonzero latency requires a new contract')
    splits=entry['label_data']['split_seeds']; seeds=[s for v in splits.values() for s in v]
    if len(seeds)!=len(set(seeds)) or set(splits)!= {'train','val','test'}: raise ValueError('disjoint train/val/test seeds required')
    if scope=='formal' and (entry['rl_seeds']!=[1,2,3,4,5] or [len(splits[k]) for k in ['train','val','test']]!=[20,5,5]
                             or entry['label_data']['slots']!=50000 or entry['rl_budget']!={'train_slots':30000,'eval_episodes':5,'scored_slots':5000}):
        raise ValueError('short budgets require smoke scope')
    physical=validate(entry['physics'])
    check_protected()
    protected=read_json(ROOT/'revision_round2/review_comments/action_plans/r5c2/r5c2_v2_protected_sources.json')
    code={str(p.relative_to(ROOT)).replace('\\','/'):file_hash(p) for p in (ROOT/'src/cvn_rl/experiments/r5c2_v2').glob('*.py')}
    for p in [ROOT/'scripts/r5c2_dqi_sensitivity_v2.py',ROOT/'scripts/analyze_r5c2_dqi_sensitivity_v2.py']:
        if p.exists(): code[str(p.relative_to(ROOT)).replace('\\','/')]=file_hash(p)
    code.update(protected['files'])
    for rel in ['src/cvn_rl/predictors/attention_lstm.py','src/cvn_rl/simulation/actions.py',
                'src/cvn_rl/simulation/traffic.py','src/cvn_rl/experiments/config.py',
                'src/cvn_rl/simulation/state_builder.py']:
        code[rel]=file_hash(ROOT/rel)
    c=conditions()
    value=dict(entry=entry,physical_manifest=physical,conditions=c,targets=[target_spec(x) for x in c[:11]],source_hashes=code)
    value['protocol_hash']=digest(value)
    value['data_spec_hash']=digest({k:entry[k] for k in ['physics','traffic','slot_duration_ms','num_channels','label_num_vehicles','label_data','history_slots','reference_quantile','input_vehicle_quantile']})
    return value

def bundle_root(protocol): return project_path(protocol['entry']['output_root'])

def save_resolution(p):
    import pandas as pd
    root=bundle_root(p); old=root/'resolved_protocol.json'
    if old.exists() and read_json(old)['protocol_hash']!=p['protocol_hash']:
        raise ValueError('bundle belongs to another code/config version; use a new v2 output root')
    write_json(old,p); write_json(root/'physical_manifest.json',p['physical_manifest'])
    write_json(root/'source_manifest.json',{'files':p['source_hashes'],'protection':check_protected()})
    pd.DataFrame(p['conditions']).to_csv(root/'conditions.csv',index=False)
    pd.DataFrame(p['targets']).to_csv(root/'targets.csv',index=False)
    return root

def array_hash(arrays):
    h=hashlib.sha256()
    for k,a in sorted(arrays.items()):
        a=np.asarray(a); h.update(k.encode()); h.update(str(a.shape).encode()); h.update(str(a.dtype).encode()); h.update(np.ascontiguousarray(a).view(np.uint8))
    return h.hexdigest()
