from pathlib import Path
import time
import warnings
import numpy as np
from cvn_rl.simulation.dqi_labels import latency_constrained_goodput, fit_reference_rate
from .physics import generate,save_bundle,load_bundle
from .protocol import bundle_root,read_json,write_json,digest,file_hash,FEATURES

def raw_features(b):
    a=b.arrays; valid=a['report_mask']
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        snr=np.nanquantile(np.where(valid,a['snr_db'],np.nan),.1,axis=1,method='linear')
        rss=np.nanquantile(np.where(valid,a['rss_dbm'],np.nan),.1,axis=1,method='linear')
    f=np.stack([snr,rss,a['bandwidth_mhz'],a['rho']],axis=-1)
    # Causal hold for missing reports; no interpolation from future samples.
    for t in range(len(f)):
        missing=~np.isfinite(f[t])
        f[t][missing]=f[t-1][missing] if t else 0.
    return f.astype(np.float32)

def target_values(b,anchors,c):
    g=latency_constrained_goodput(b.arrays['rate_mbps'],packet_size_bits=32000,delay_max_ms=c['delay_ms'],delay_beta_ms=2.)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        shared=np.nanquantile(np.where(b.arrays['report_mask'],g,np.nan),c['qv'],axis=1,method='linear')
    z=np.empty((len(anchors),shared.shape[1]))
    for start in range(0,len(anchors),256):
        ids=anchors[start:start+256,None]+np.arange(1,int(c['horizon_ms'])+1)[None,:]
        z[start:start+256]=np.quantile(shared[ids],c['qt'],axis=1,method='linear')
    return z

def validate_files(root,manifest):
    for rel,h in manifest['files'].items():
        if not (root/rel).is_file() or file_hash(root/rel)!=h: raise ValueError(f'data hash mismatch: {rel}')

def prepare(p,resume=False):
    started=time.perf_counter(); root=bundle_root(p)/'data'; marker=root/'prepared_manifest.json'
    if marker.exists():
        old=read_json(marker)
        if old['data_spec_hash']!=p['data_spec_hash']: raise ValueError('different data specification')
        validate_files(root,old)
        if not resume: raise FileExistsError('prepared data exists; use resume')
        return old
    e=p['entry']; cfg=e['label_data']; entries=[]; root.mkdir(parents=True,exist_ok=True)
    anchors=np.arange(e['history_slots']-1,cfg['slots'],cfg['stride'])
    # Future-only tail belongs to the same split and is excluded from scaler fitting.
    for split,seeds in cfg['split_seeds'].items():
        for seed in seeds:
            raw=root/'raw_labels'/split/f'seed_{seed}'
            if (raw/'manifest.json').exists():
                b=load_bundle(raw)
                expected=digest({'physics':e['physics'],'traffic':e['traffic'],'packet':e['packet_traffic'],'vehicles':e['label_num_vehicles'],'channels':e['num_channels'],'slot':e['slot_duration_ms']})
                if (b.spec_hash,b.slots,b.future_slots,b.seed,b.split)!=(expected,cfg['slots'],cfg['future_slots'],seed,split): raise ValueError('raw cache identity mismatch')
            else:
                b=generate(e,seed,cfg['slots'],e['label_num_vehicles'],split,cfg['future_slots']); save_bundle(raw,b)
            f=raw_features(b)[:cfg['slots']]; values={c['target_id']:target_values(b,anchors,c) for c in p['targets']}
            good=np.ones(len(anchors),dtype=bool)
            for z in values.values(): good &= np.all(np.isfinite(z),axis=1)
            if not good.any(): raise ValueError('no common valid target samples')
            common=anchors[good]
            fp=root/'features'/split/f'{seed}.npy'; fp.parent.mkdir(parents=True,exist_ok=True); np.save(fp,f)
            ap=root/'anchors'/split/f'{seed}.npy'; ap.parent.mkdir(parents=True,exist_ok=True); np.save(ap,common)
            for tid,z in values.items():
                dst=root/'targets'/tid/split/f'{seed}.npy'; dst.parent.mkdir(parents=True,exist_ok=True); np.save(dst,z[good].astype(np.float32))
            entries.append({'split':split,'seed':seed,'features':str(fp.relative_to(root)),'anchors':str(ap.relative_to(root)),
                            'physical_hash':b.physical_hash(),'samples':len(common)})
            print(f'v2 data {split} seed={seed} samples={len(common)}',flush=True)
    total=np.zeros(40); total2=np.zeros(40); count=0
    for row in entries:
        if row['split']!='train': continue
        f=np.load(root/row['features']).reshape(-1,40).astype(float); total+=f.sum(0); total2+=(f*f).sum(0); count+=len(f)
    mean=total/count; std=np.sqrt(np.maximum(total2/count-mean**2,1e-8))
    preprocessing={'feature_names':FEATURES,'mean':mean.tolist(),'std':std.tolist(),'train_rows':count,'input_quantile':.1,'input_cutoff':'a=t-1'}
    preprocessing['hash']=digest(preprocessing); write_json(root/'preprocessing.json',preprocessing)
    targets={}; diagnostics=[]
    for c in p['targets']:
        tid=c['target_id']; ztrain=[np.load(root/'targets'/tid/'train'/f'{seed}.npy') for seed in cfg['split_seeds']['train']]
        ref=fit_reference_rate(ztrain,quantile=.95)
        targets[tid]={'parameters':c,'r_ref_mbps':ref,'target_hash':digest({'target':c,'reference':ref,'data':p['data_spec_hash'],'anchor':'a=t-1'})}
        for split,seeds in cfg['split_seeds'].items():
            z=np.concatenate([np.load(root/'targets'/tid/split/f'{s}.npy') for s in seeds])
            diagnostics.append({'target_id':tid,'split':split,'reference':ref,'rows':len(z),'clip_fraction':float(np.mean(z>=ref)),'z_mean':float(z.mean())})
        train=np.concatenate(ztrain)
        for q in [.9,.95,.99]:
            r=fit_reference_rate(train,quantile=q); y=np.clip(train/r,0,1)
            diagnostics.append({'target_id':tid,'split':'reference_diagnostic_train_only','reference_quantile':q,'reference':r,
                                'clip_fraction':float(np.mean(train>=r)), 'rank_tie_row_fraction':float(np.mean(np.any(np.diff(np.sort(y,axis=1),axis=1)==0,axis=1)))})
    import pandas as pd
    pd.DataFrame(diagnostics).to_csv(root/'label_diagnostics.csv',index=False)
    files={str(f.relative_to(root)):file_hash(f) for f in root.rglob('*') if f.is_file() and f.name!='prepared_manifest.json' and '.partial' not in f.name}
    manifest={'data_spec_hash':p['data_spec_hash'],'entries':entries,'targets':targets,'preprocessing_hash':preprocessing['hash'],
              'files':files,'slots':cfg['slots'],'anchor_rule':'input a-199..a; target a+1..a+H; decision a+1', 'wall_seconds':time.perf_counter()-started}
    write_json(marker,manifest); return manifest

def dataset_arrays(p,tid,max_slots=None,train_count=None):
    root=bundle_root(p)/'data'; meta=read_json(root/'prepared_manifest.json')
    if meta['data_spec_hash']!=p['data_spec_hash']: raise ValueError('wrong prepared data')
    pre=read_json(root/'preprocessing.json'); ref=meta['targets'][tid]['r_ref_mbps']; groups={k:[] for k in ['train','val','test']}
    used=0
    for row in meta['entries']:
        if row['split']=='train' and train_count is not None and used>=train_count: continue
        if row['split']=='train': used+=1
        f=np.load(root/row['features'],mmap_mode='r').reshape(-1,40); a=np.load(root/row['anchors'])
        z=np.load(root/'targets'/tid/row['split']/f"{row['seed']}.npy")
        selected=np.ones(len(a),bool) if max_slots is None else a<max_slots
        groups[row['split']].append((f,a[selected],np.clip(z[selected]/ref,0,1)))
    return groups,meta,pre
