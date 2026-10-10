import math
import time
import numpy as np
from .protocol import bundle_root,write_json,read_json,digest
from .physics import generate
from .guidance import held_reports,freshness,fill_theta
from .environment import environment_config,PhysicalAccessEnv
from .runner import run_task

def pilot_freshness(p):
    start=time.perf_counter(); e=p['entry']; f=e['freshness']; slots=max(e['pilot']['age_slots'],5*max(f['stress_candidates_slots']))
    b=generate(e,91001,slots,10,'age_pilot'); rows=[]; chosen=None
    for period in f['stress_candidates_slots']:
        source,rho,age=held_reports(b,period); valid=source>=0
        nominal=freshness(np.ones_like(age),age,rho,f['t_min_ms'],f['t_max_ms'])
        half=freshness(np.ones_like(age),age,rho,f['t_min_ms'],f['t_max_ms'],.5)
        ratio=age/nominal['validity_ms']; inside=(ratio>0)&(ratio<=1)&valid; expired=half['expired']&valid
        inside_fraction=float(inside.sum()/max(valid.sum(),1)); expiry_fraction=float(expired.sum()/max(valid.sum(),1))
        passed=inside_fraction>=.1 and expiry_fraction>=.01
        rows.append({'update_slots':period,'valid_channel_decisions':int(valid.sum()),'inside_fraction':inside_fraction,
                     'half_expired_fraction':expiry_fraction,'passed':passed,'per_channel_inside':(inside.sum(0)/np.maximum(valid.sum(0),1)).tolist(),
                     'per_channel_half_expired':(expired.sum(0)/np.maximum(valid.sum(0),1)).tolist()})
        if passed and chosen is None: chosen=period
    result={'protocol_hash':p['protocol_hash'],'selected_update_slots':chosen,'basis':'received Doppler coherence; age coverage only',
            'candidates':rows,'physical_hash':b.physical_hash(),'slots':slots,'wall_seconds':time.perf_counter()-start}
    write_json(bundle_root(p)/'pilot/freshness_selection.json',result)
    if chosen is None: raise ValueError('no candidate meets age coverage; inspect physical/timestamp rules')
    return result

def baseline_equivalence(p,slots=1000):
    e=p['entry']; b=generate(e,91002,slots,10,'baseline_equivalence'); traces=[b.trace(),b.trace()]
    theta=[fill_theta(b,tr,e) for tr in traces]; environments=[]
    for tr,period in zip(traces,[10,read_json(bundle_root(p)/'pilot/freshness_selection.json')['selected_update_slots']]):
        tr.dqi_age[:]=(np.arange(slots)%period)[:,None]; tr.fresh_dqi[:]=.2 if period==10 else .9
        tr.dqi_source_slot[:]=np.arange(slots)[:,None]//period*period
        environments.append(PhysicalAccessEnv(tr,environment_config(e,True,20.),b))
    x,i=environments[0].reset(); y,j=environments[1].reset()
    for t in range(slots):
        np.testing.assert_array_equal(x,y)
        np.testing.assert_array_equal(np.stack([k['action_mask'] for k in i]),np.stack([k['action_mask'] for k in j]))
        actions=np.array([np.flatnonzero(k['action_mask'])[-1] for k in i])
        x,r,_,a=environments[0].step(actions); y,s,_,z=environments[1].step(actions)
        np.testing.assert_array_equal(r,s)
        for field in ['served_bits','packet_dropped','packet_completed','queue_lengths','pu_access_violation','actual_interference_mw']:
            np.testing.assert_array_equal(a[field],z[field])
        i=environments[0].agent_infos(); j=environments[1].agent_infos()
    result={'protocol_hash':p['protocol_hash'],'passed':True,'shared_baseline':True,'forced_action_slots':slots,
            'physical_hash':b.physical_hash(),'theta_hashes':[x['theta_hash'] for x in theta],
            'method':'paired forced-action replay, neutral DQI observations, same queue/reward/mask/service/interference'}
    write_json(bundle_root(p)/'pilot/baseline_equivalence.json',result); return result

def pilot_runtime(p,seeds=None,full_probe=True,resume=False):
    from .execution import run_cases
    e=p['entry']; root=bundle_root(p)
    baseline_equivalence(p)
    tasks=run_cases(p,'pilot',['C00','C14','C15','B00'],seeds or [1,2],resume)
    probes=[]
    if full_probe:
        for cid in ['C00','C14']:
            index=run_cases(p,'pilot',[cid],[1],resume,True)
            probes.append(index[f'{cid}_s1_fullprobe']['result'])
    models=read_json(root/'pilot/model_index.json')
    # Cost forecast keeps fixed setup separate from sample/epoch-dependent work.
    ratios=[]
    for v in models.values(): ratios.append(v['wall_seconds'])
    result={'protocol_hash':p['protocol_hash'],'full_training_probes':probes,'short_tasks':len(tasks),
            'pilot_predictor_seconds':{k:v['wall_seconds'] for k,v in models.items()},
            'formal_task_count':(len(p['conditions'])+1)*len(e['rl_seeds']),'hardware':{'torch_version':__import__('torch').__version__,
              'cuda':__import__('torch').cuda.is_available(),'device':__import__('torch').cuda.get_device_name(0) if __import__('torch').cuda.is_available() else 'cpu'},
            'estimate_note':'Pilot predictor totals include setup and hash validation; do not multiply fixed cost by dataset ratio. Formal estimate calibrated with first formal model.',
            'rl_matrix_training_seconds_estimate':sum(v['training_seconds'] for v in probes)/max(len(probes),1)*(len(p['conditions'])+1)*len(e['rl_seeds']) if probes else None,
            'rl_matrix_total_seconds_estimate':sum(v['wall_seconds'] for v in probes)/max(len(probes),1)*(len(p['conditions'])+1)*len(e['rl_seeds']) if probes else None}
    write_json(root/'pilot/runtime_summary.json',result); return result
