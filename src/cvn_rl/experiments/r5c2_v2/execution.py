import math
import os
import subprocess
import sys
import time
from pathlib import Path
from .protocol import ROOT,bundle_root,digest,file_hash,read_json,write_json,check_protected
from .runner import stress_period
from cvn_rl.experiments.r5c2_execution import process_is_running

def task_identity(p,cid,seed,stage,full=False):
    models=read_json(bundle_root(p)/('model_index.json' if stage=='formal' else 'pilot/model_index.json'))
    c=next(x for x in p['conditions'] if x['condition_id']==('C00' if cid=='B00' else cid))
    return {'protocol_hash':p['protocol_hash'],'condition':c,'baseline':cid=='B00','seed':seed,'stage':stage,'full_probe':full,
            'model':models[c['target_id']],'reference_model':models['T00'],'stress_period':stress_period(p)}

def freeze(p,phase):
    root=bundle_root(p); check_protected()
    if phase=='protocol':
        runtime=read_json(root/'pilot/runtime_summary.json'); age=read_json(root/'pilot/freshness_selection.json'); baseline=read_json(root/'pilot/baseline_equivalence.json')
        tests=read_json(root/'validation.json'); data=read_json(root/'data/prepared_manifest.json')
        if any(x.get('protocol_hash')!=p['protocol_hash'] for x in [runtime,age,baseline,tests]) or not baseline['passed'] or not tests['passed'] or not runtime['full_training_probes']:
            raise ValueError('matching behavior, age, baseline and full-length timing evidence required')
        if data['data_spec_hash']!=p['data_spec_hash']: raise ValueError('data spec differs')
        payload={'protocol_hash':p['protocol_hash'],'data_manifest_hash':file_hash(root/'data/prepared_manifest.json'),
                 'stress_update_slots':stress_period(p),'baseline_shared':True,'task_count':(len(p['conditions'])+1)*len(p['entry']['rl_seeds']),'runtime':runtime,
                 'validation_sha256':file_hash(root/'validation.json'),'physics_sha256':file_hash(root/'physical_manifest.json')}
        write_json(root/'protocol_frozen.json',payload); return payload
    frozen=read_json(root/'protocol_frozen.json'); models=read_json(root/'model_index.json')
    if frozen['protocol_hash']!=p['protocol_hash'] or set(models)!={f'T{i:02}' for i in range(11)}: raise ValueError('freeze and all eleven formal models required')
    tasks=[]
    for cid in [c['condition_id'] for c in p['conditions']]+['B00']:
        for seed in p['entry']['rl_seeds']:
            identity=task_identity(p,cid,seed,'formal'); signature=digest(identity)
            for model in [identity['model'],identity['reference_model']]:
                if model['stage']!='formal' or file_hash(model['model_path'])!=model['model_sha256']: raise ValueError('formal model/hash invalid')
            path=root/'resolved_cases'/f'{cid}_s{seed}.json'; write_json(path,identity)
            tasks.append({'condition_id':cid,'seed':seed,'signature':signature,'path':str(path)})
    value={'protocol_hash':p['protocol_hash'],'model_index_hash':file_hash(root/'model_index.json'),'tasks':tasks}
    write_json(root/'runs_frozen.json',value); return value

def run_cases(p,stage='formal',conditions=None,seeds=None,resume=False,full=False):
    root=bundle_root(p); known={c['condition_id'] for c in p['conditions']}|{'B00'}
    selected=conditions or sorted(known); seeds=seeds or p['entry']['rl_seeds']
    if not set(selected)<=known or not set(seeds)<=set(p['entry']['rl_seeds']): raise ValueError('unknown condition/seed')
    if stage=='formal':
        frozen=read_json(root/'runs_frozen.json')
        if frozen['protocol_hash']!=p['protocol_hash'] or frozen['model_index_hash']!=file_hash(root/'model_index.json'): raise ValueError('matching runs freeze required')
    indexpath=root/('run_index.json' if stage=='formal' else 'pilot/run_index.json'); index=read_json(indexpath) if indexpath.exists() else {}
    for cid in selected:
        for seed in seeds:
            label=f'{cid}_s{seed}'+('_fullprobe' if full else ''); identity=task_identity(p,cid,seed,stage,full); signature=digest(identity)
            previous=index.get(label); output=root/('rl_runs' if stage=='formal' else 'pilot/rl_runs')/label/'result.json'
            if previous and previous['status']=='running' and process_is_running(previous.get('pid',0)): raise RuntimeError('task already running')
            if previous and previous['status']=='completed':
                if resume and previous['signature']==signature and output.is_file() and file_hash(output)==previous['result_hash']: continue
                raise ValueError('completed task exists with incompatible inputs; do not overwrite')
            logfile=root/('logs' if stage=='formal' else 'pilot/logs')/f'{label}.log'; logfile.parent.mkdir(parents=True,exist_ok=True)
            args=[sys.executable,str(ROOT/'scripts/r5c2_dqi_sensitivity_v2.py'),'_task','--protocol',str(root/'resolved_protocol.json'),'--condition',cid,'--seed',str(seed),'--stage',stage]
            if full: args.append('--full-probe')
            print(f'v2 running {stage} {label}',flush=True); started=time.perf_counter()
            env={**os.environ,'PYTHONPATH':str(ROOT/'src')+os.pathsep+os.environ.get('PYTHONPATH','')}
            with logfile.open('w',encoding='utf-8') as log:
                proc=subprocess.Popen(args,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
                index[label]={'status':'running','pid':proc.pid,'signature':signature,'log':str(logfile)}; write_json(indexpath,index)
                code=proc.wait()
            if code:
                index[label].update(status='failed',exit_code=code); write_json(indexpath,index); raise RuntimeError(f'task failed; {logfile}')
            result=read_json(output)
            if result['protocol_hash']!=p['protocol_hash'] or not result['evaluation_parameters_frozen']: raise ValueError('task result identity/freeze')
            index[label].update(status='completed',wall_seconds=time.perf_counter()-started,result_hash=file_hash(output),result=result)
            write_json(indexpath,index)
    return index
