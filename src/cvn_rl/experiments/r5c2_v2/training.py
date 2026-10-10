import copy
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
from .data import dataset_arrays,validate_files
from .predictor import build_model
from .protocol import bundle_root,digest,file_hash,read_json,write_json,FEATURES

class WindowDataset(Dataset):
    def __init__(self,rows,pre):
        self.rows=rows; self.mean=np.asarray(pre['mean'],np.float32); self.std=np.asarray(pre['std'],np.float32)
        self.ids=[(i,j) for i,(_,a,_) in enumerate(rows) for j in range(len(a))]
    def __len__(self): return len(self.ids)
    def __getitem__(self,k):
        i,j=self.ids[k]; f,a,y=self.rows[i]; end=a[j]+1
        return torch.from_numpy(((f[end-200:end]-self.mean)/self.std).astype(np.float32)),torch.from_numpy(y[j].astype(np.float32))

def evaluate(model,loader,device):
    model.eval(); total=0.; rank=0.; count=0
    with torch.no_grad():
        for x,y in loader:
            x=x.to(device); y=y.to(device); prediction=model(x)
            total+=float(((prediction-y)**2).sum().cpu()); count+=y.numel()
            # Published metric uses logits rather than sigmoid scores in predicted softmax.
            logits=torch.logit(prediction.clamp(1e-6,1-1e-6))
            rank+=float(-(torch.softmax(y,1)*torch.log_softmax(logits,1)).sum().cpu())
    return {'mse':total/count,'listnet':rank/(count/10),'samples':count//10}

def train(p,tid,stage='formal',resume=False):
    root=bundle_root(p); entry=p['entry']; recipe=copy.deepcopy(entry['predictor_recipe']); started=time.perf_counter()
    if stage=='formal':
        frozen=read_json(root/'protocol_frozen.json')
        if frozen['protocol_hash']!=p['protocol_hash']: raise ValueError('formal training needs matching freeze')
    pilot=stage=='pilot'
    if pilot: recipe['epochs']=entry['pilot']['epochs']
    groups,data,pre=dataset_arrays(p,tid,entry['pilot']['label_slots'] if pilot else None,entry['pilot']['train_seed_count'] if pilot else None)
    validate_files(root/'data',data)
    target=data['targets'][tid]; recipe_hash=digest(recipe)
    output=root/('pilot/models' if pilot else 'models')/tid; marker=output/'training_task.json'
    identity={'target_hash':target['target_hash'],'data_spec_hash':p['data_spec_hash'],'preprocessing_hash':pre['hash'],'recipe_hash':recipe_hash}
    if marker.exists():
        old=read_json(marker)
        if not resume or old['identity']!=identity or file_hash(output/'model.pt')!=old['model_sha256'] or file_hash(output/'model.meta.json')!=old['meta_sha256']: raise ValueError('existing model not reusable')
        return old
    output.mkdir(parents=True,exist_ok=True); torch.set_num_threads(entry['torch_threads'])
    torch.manual_seed(recipe['seed']); np.random.seed(recipe['seed']); device='cuda' if torch.cuda.is_available() else 'cpu'
    model=build_model(recipe).to(device); opt=torch.optim.Adam(model.parameters(),lr=recipe['learning_rate'],weight_decay=recipe['weight_decay'])
    loaders={k:DataLoader(WindowDataset(rows,pre),batch_size=recipe['batch_size'],shuffle=k=='train',num_workers=0) for k,rows in groups.items()}
    best=float('inf'); state=None; stale=0; history=[]
    for epoch in range(recipe['epochs']):
        model.train(); total=0.; samples=0
        for x,y in loaders['train']:
            x=x.to(device); y=y.to(device); opt.zero_grad(); pred=model(x); loss=torch.nn.functional.mse_loss(pred,y)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),recipe['gradient_clip']); opt.step()
            total+=float(loss.detach().cpu())*len(y); samples+=len(y)
        metrics=evaluate(model,loaders['val'],device)
        history.append({'epoch':epoch+1,'train_mse':total/samples,**metrics})
        print(f'v2 {stage} {tid} epoch={epoch+1} val_mse={metrics["mse"]:.6f}',flush=True)
        if metrics['mse']<best:
            best=metrics['mse']; state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}; stale=0
        else: stale+=1
        if stale>=recipe['patience']: break
    if state is None: raise ValueError('no finite validation checkpoint')
    model.load_state_dict(state); metrics={k:evaluate(model,loader,device) for k,loader in loaders.items()}
    torch.save(state,output/'model.pt')
    meta={**identity,'version':'r5c2_v2','target_id':tid,'target_parameters':target['parameters'],'r_ref_mbps':target['r_ref_mbps'],
          'feature_names':FEATURES,'history_slots':200,'recipe':recipe,'stage':stage,'model_sha256':file_hash(output/'model.pt'),
          'metrics':metrics,'optimization':'MSE only; ListNet evaluation only','input_anchor':'a=t-1; history a-199..a', 'wall_seconds':time.perf_counter()-started}
    write_json(output/'model.meta.json',meta); write_json(output/'history.json',history)
    record={'identity':identity,'stage':stage,'target_id':tid,'model_path':str(output/'model.pt'),'model_sha256':meta['model_sha256'],
            'meta_sha256':file_hash(output/'model.meta.json'),'wall_seconds':meta['wall_seconds'],'status':'completed'}
    write_json(marker,record); return record

def train_all(p,stage,targets=None,resume=False):
    root=bundle_root(p); index_path=root/('pilot/model_index.json' if stage=='pilot' else 'model_index.json')
    index=read_json(index_path) if index_path.exists() else {}
    known={t['target_id'] for t in p['targets']}
    for tid in targets or sorted(known):
        if tid not in known: raise ValueError('unknown target')
        index[tid]=train(p,tid,stage,resume); write_json(index_path,index)
    return index
