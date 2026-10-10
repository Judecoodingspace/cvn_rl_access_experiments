import math
import time
from pathlib import Path
import numpy as np
import torch
from cvn_rl.agents.shared_multi import create_shared_agent
from .protocol import bundle_root,read_json,write_json,digest,file_hash,array_hash
from .physics import generate,save_bundle,load_bundle
from .environment import PhysicalAccessEnv,environment_config
from .predictor import FrozenDQIV2
from .guidance import fill_dqi,fill_theta

def stress_period(p):
    x=read_json(bundle_root(p)/'pilot/freshness_selection.json')
    if x['protocol_hash']!=p['protocol_hash'] or x['selected_update_slots'] is None: raise ValueError('matching age pilot required')
    return x['selected_update_slots']

def physical_cache(p,seed,slots,split,stage):
    root=bundle_root(p)/'data/rl_physical'/stage/split/f'seed_{seed}_slots_{slots}'
    if (root/'manifest.json').exists():
        b=load_bundle(root)
        expected=generate_spec_hash(p['entry'])
        if (b.spec_hash,b.seed,b.slots,b.split)!=(expected,seed,slots,split): raise ValueError('RL physical cache identity')
        return b
    b=generate(p['entry'],seed,slots,10,split); save_bundle(root,b); return b

def generate_spec_hash(e):
    return digest({'physics':e['physics'],'traffic':e['traffic'],'packet':e['packet_traffic'],'vehicles':10,'channels':10,'slot':e['slot_duration_ms']})

def model_for(p,tid,stage):
    root=bundle_root(p); models=read_json(root/('pilot/model_index.json' if stage=='pilot' else 'model_index.json'))
    r=models[tid]
    if r['stage']!=stage or file_hash(r['model_path'])!=r['model_sha256'] or file_hash(Path(r['model_path']).with_suffix('.meta.json'))!=r['meta_sha256']: raise ValueError('model stage/hash mismatch')
    pre=read_json(root/'data/preprocessing.json')
    model=FrozenDQIV2(r['model_path'],r['identity'],pre,'cuda' if torch.cuda.is_available() else 'cpu')
    return model,models

def configure_trace(b,e,model,c,period,baseline=False):
    trace=b.trace(); theta=fill_theta(b,trace,e)
    if baseline:
        trace.dqi_source_slot[:]=-1; payload=None
    else: payload=fill_dqi(b,trace,model,period,c,e)
    return trace,payload,theta

def evaluate(agent,p,c,model,reference,period,warmup,scored,episodes,seed,stage,baseline):
    metrics=[]; physical=[]; theta_hashes=[]; ages=[]; before=agent.parameter_hash(); start=time.perf_counter()
    agent.set_eval_mode(True)
    for episode in range(episodes):
        eseed=600000+seed*100+episode
        b=physical_cache(p,eseed,warmup+scored,'held_out_eval',stage)
        tr,payload,theta=configure_trace(b,p['entry'],model,c,period,baseline)
        env=PhysicalAccessEnv(tr,environment_config(p['entry'],baseline,reference),b,warmup)
        obs,infos=env.reset(); agent.reset_episode(); records=[]; carry=0
        for t in range(b.slots):
            if t==warmup: carry=sum(sum(pkt[1]<warmup for pkt in q) for q in env.queues)
            actions=agent.select_actions(obs,infos); obs,_,_,record=env.step(actions); infos=env.agent_infos()
            if t>=warmup: records.append(record)
        def summed(k): return int(sum(np.sum(x[k]) for x in records))
        generated=summed('packet_generated'); dropped=summed('packet_dropped'); completed=summed('packet_completed'); window_completed=summed('window_arrival_completed')
        carryout=sum(len(q) for q in env.queues)
        if generated+carry!=dropped+completed+carryout: raise AssertionError('packet ledger does not balance')
        served=float(sum(np.sum(x['served_bits']) for x in records)); actual=summed('pu_violation_per_channel'); access=summed('pu_access_violation'); busy=summed('pu_busy_channels'); tx=summed('transmitted')
        metrics.append({'episode':episode,'served_bits':served,'goodput_mbps':served/(scored*.001)/1e6,'per_vt_goodput_mbps':served/(scored*.001)/1e7,
                        'plr':dropped/max(generated,1),'generated_packets':generated,'dropped_packets':dropped,'completed_packets':completed,
                        'completed_scored_arrivals':window_completed,'unfinished_scored_arrivals':generated-dropped-window_completed,
                        'carry_in_packets':carry,'carry_out_packets':carryout,'pu_access_count':access,'transmission_count':tx,
                        'pu_access_rate':access/max(tx,1),'pu_interference_count':actual,'channel_slots':scored*10,'busy_channel_slots':busy,
                        'pu_interference_rate':actual/(scored*10),'pu_active_interference_rate':actual/max(busy,1),'scored_slots':scored})
        physical.append(b.physical_hash()); theta_hashes.append(theta['theta_hash'])
        if payload:
            valid=payload['available'][warmup:]; a=payload['age_ms'][warmup:]; v=payload['validity_ms'][warmup:]
            ages.append({'available_channel_decisions':int(valid.sum()),'expired_fraction':float(payload['expired'][warmup:][valid].mean()),
                         'mean_age_validity_ratio':float((a/v)[valid].mean())})
    if agent.parameter_hash()!=before: raise AssertionError('evaluation changed learned parameters')
    return {'episodes':metrics,'physical_hashes':physical,'theta_hashes':theta_hashes,'freshness_audit':ages,
            'evaluation_seconds':time.perf_counter()-start,'evaluation_parameters_frozen':True}

def run_task(p,cid,seed,stage='formal',full_probe=False):
    start=time.perf_counter(); e=p['entry']; torch.set_num_threads(e['torch_threads'])
    baseline=cid=='B00'; c=next(x for x in p['conditions'] if x['condition_id']==('C00' if baseline else cid))
    model,models=model_for(p,c['target_id'],stage); reference=read_json(Path(models['T00']['model_path']).with_suffix('.meta.json'))['r_ref_mbps']
    period=10 if c['group']=='main' else stress_period(p)
    warmup=math.ceil(200/stress_period(p))*stress_period(p)
    budget=e['rl_budget']; pilot=e['pilot']; training=budget['train_slots'] if stage=='formal' or full_probe else pilot['rl_train_slots']
    scored=budget['scored_slots'] if stage=='formal' else pilot['scored_slots']; episodes=budget['eval_episodes'] if stage=='formal' else pilot['eval_episodes']
    b=physical_cache(p,seed,training,'rl_train',stage); tr,_,theta=configure_trace(b,e,model,c,period,baseline)
    env=PhysicalAccessEnv(tr,environment_config(e,baseline,reference),b); agent=create_shared_agent(e['agent'],10,10,seed,e['agent_settings'])
    obs,infos=env.reset(); agent.set_eval_mode(False); agent.reset_episode(); begin=time.perf_counter()
    for _ in range(training):
        actions=agent.select_actions(obs,infos); nxt,reward,done,_=env.step(actions); ni=env.agent_infos()
        agent.observe_batch(obs,actions,reward,nxt,done,infos,ni); obs,infos=nxt,ni
    train_seconds=time.perf_counter()-begin
    evaluation=evaluate(agent,p,c,model,reference,period,warmup,scored,episodes,seed,stage,baseline)
    label=f'{cid}_s{seed}'+('_fullprobe' if full_probe else '')
    output=bundle_root(p)/('rl_runs' if stage=='formal' else 'pilot/rl_runs')/label
    output.mkdir(parents=True,exist_ok=True)
    torch.save({'q_state':agent.q_net.state_dict(),'target_state':agent.target_net.state_dict(),'win':agent.win,'wres':agent.wres,
                'settings':e['agent_settings'],'input_dim':agent.input_dim},output/'policy.pt')
    result={'condition_id':cid,'target_id':c['target_id'],'seed':seed,'stage':stage,'group':c['group'],'period_slots':period,
            'warmup_slots':warmup,'train_slots':training,'training_seconds':train_seconds,'full_probe':full_probe,
            'train_physical_hash':b.physical_hash(),'train_theta_hash':theta['theta_hash'],'policy_hash':file_hash(output/'policy.pt'),
            'dqimodel_hash':models[c['target_id']]['model_sha256'],'protocol_hash':p['protocol_hash'],'research_evidence':stage=='formal' and e['execution_scope']=='formal',
            **evaluation,'wall_seconds':time.perf_counter()-start}
    write_json(output/'result.json',result); return result
