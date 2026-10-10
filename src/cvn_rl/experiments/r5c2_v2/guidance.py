import numpy as np
import torch
from .data import raw_features
from .protocol import project_path,array_hash
from cvn_rl.predictors.frozen_predictor import FrozenPredictor

def freshness(raw,age,rho,tmin=100.,tmax=1000.,scale=1.,rule='exp'):
    raw,age,rho=np.broadcast_arrays(np.asarray(raw,float),np.asarray(age,float),np.asarray(rho,float))
    if not 0<tmin<=tmax or scale<=0 or np.any(age<0) or not all(np.all(np.isfinite(x)) for x in [raw,age,rho]): raise ValueError('invalid freshness input')
    if rule not in ['exp','linear','off']: raise ValueError('unknown decay')
    v=scale*(tmin+(tmax-tmin)*np.clip(rho,0,1)); would=age>v
    w=np.exp(-age/v) if rule=='exp' else 1-(1-np.exp(-1))*age/v if rule=='linear' else np.ones_like(age)
    if rule!='off': w=np.where(would,0.,w)
    return {'fresh':raw*w,'validity_ms':v,'weight':w,'expired':would if rule!='off' else np.zeros_like(would),'would_expire':would}

def held_reports(b,period,history=200):
    a=b.arrays; s=b.slots; m=a['rho'].shape[1]; source=np.full((s,m),-1,int); rho=np.zeros((s,m))
    last=np.full(m,-1,int); received=np.zeros(m)
    for t in range(s):
        if t>=history and t%period==0:
            valid=a['report_mask'][t-1].any(0); last[valid]=t; received[valid]=a['rho'][t-1,valid]
        source[t]=last; rho[t]=received
    age=np.where(source>=0,np.arange(s)[:,None]-source,0.)
    return source,rho,age

def fill_dqi(b,trace,predictor,period,c,entry):
    source,rho,age=held_reports(b,period); raw=np.full(source.shape,.5)
    updates=np.unique(source[source>=0]); outputs=predictor.predict_windows(raw_features(b)[:b.slots],updates,entry['predictor_recipe']['inference_batch_size'])
    lookup={int(t):i for i,t in enumerate(updates)}
    for m in range(raw.shape[1]):
        valid=source[:,m]>=0; raw[valid,m]=np.asarray([outputs[lookup[int(t)],m] for t in source[valid,m]])
    f=entry['freshness']; payload=freshness(raw,age,rho,f['t_min_ms'],f['t_max_ms'],c['scale'],c['rule'])
    payload['fresh']=np.where(source>=0,payload['fresh'],.5)
    trace.dqi_predicted[:]=raw; trace.fresh_dqi[:]=payload['fresh']; trace.dqi_age[:]=age; trace.dqi_source_slot[:]=source
    trace.dqi_input_cutoff_slot[:]=np.where(source>=0,source-1,-1); trace.dqi_validity_slots[:]=payload['validity_ms']
    return {**payload,'source':source,'rho_received':rho,'age_ms':age,'available':source>=0}

def fill_theta(b,trace,entry):
    c=entry['prediction']; predictor=FrozenPredictor(project_path(c['model_path']),device='cuda' if torch.cuda.is_available() else 'cpu',
       expected_sha256=c['expected_sha256'],expected_input_source='bs_true_idle_history')
    if predictor.history_window!=50 or predictor.input_size!=1: raise ValueError('occupancy interface must be shared univariate50')
    updates=np.arange(50,b.slots,c['update_slots']); idle=(~b.arrays['occupancy']).astype(np.float32); values=[]
    with torch.no_grad():
        for start in range(0,len(updates),128):
            x=np.stack([idle[t-50:t].T[:,:,None] for t in updates[start:start+128]])
            y=predictor.model(torch.as_tensor(x.reshape(-1,50,1),device=predictor.device)).cpu().numpy().reshape(-1,10)
            values.extend(y)
    last=np.full(10,.5); u=-1; lookup={int(t):y for t,y in zip(updates,values)}
    for t in range(b.slots):
        if t in lookup: last=lookup[t]; u=t
        trace.theta[t]=last; trace.theta_input_cutoff_slot[t]=u-1 if u>=0 else -1; trace.theta_target_slot[t]=u
    return {'model_sha256':predictor.sha256,'theta_hash':array_hash({'theta':trace.theta})}
