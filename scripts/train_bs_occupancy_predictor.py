"""Train P3 Attention-LSTM from causal BS true-idle histories."""
from __future__ import annotations
import argparse, hashlib, json, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from cvn_rl.predictors.attention_lstm import AttentionLSTMIdlePredictor
from cvn_rl.simulation.traffic import generate_occupancy

TRAFFIC={"family":"periodic_with_jitter","period_slots":50,"busy_slots":20,"jitter_slots":5}

def dataset(seed:int, slots:int, history:int=50):
    occupied=generate_occupancy(slots,1,TRAFFIC,np.random.default_rng(seed))
    idle=(~occupied).astype(np.float32)
    return torch.from_numpy(np.stack([idle[t-history:t] for t in range(history,slots)])), torch.from_numpy(idle[history:])

def ece(y,p,bins=10):
    total=0.0
    for lo,hi in zip(np.linspace(0,1,bins+1)[:-1],np.linspace(0,1,bins+1)[1:]):
        mask=(p>=lo)&(p<(hi) if hi<1 else p<=hi)
        if mask.any(): total+=float(mask.mean())*abs(float(p[mask].mean()-y[mask].mean()))
    return total

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--output',default='models/attention_lstm_bs_occupancy_h1.pt'); ap.add_argument('--seed',type=int,default=20260703); ap.add_argument('--slots',type=int,default=30000); a=ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    seeds={"train":a.seed+101,"validation":a.seed+202,"test":a.seed+303}
    data={k:dataset(v,a.slots) for k,v in seeds.items()}
    loader=DataLoader(TensorDataset(*data['train']),batch_size=128,shuffle=True)
    model=AttentionLSTMIdlePredictor(1,64,1,64,0.1,1); opt=torch.optim.Adam(model.parameters(),lr=1e-3); loss_fn=nn.BCELoss()
    best=float('inf'); state=None; stale=0; started=time.perf_counter(); epochs=0
    for epoch in range(80):
        model.train()
        for x,y in loader:
            opt.zero_grad(); loss_fn(model(x),y).backward(); opt.step()
        model.eval()
        with torch.no_grad(): val=float(loss_fn(model(data['validation'][0]),data['validation'][1]))
        epochs=epoch+1
        if val<best-1e-6: best=val; state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}; stale=0
        else:
            stale+=1
            if stale>=10: break
    assert state is not None; model.load_state_dict(state); model.eval(); tx,ty=data['test']
    with torch.no_grad():
        tick=time.perf_counter(); pred=model(tx).numpy().reshape(-1); latency=(time.perf_counter()-tick)*1000/len(tx)
    target=ty.numpy().reshape(-1); out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True); torch.save(model.state_dict(),out); sha=hashlib.sha256(out.read_bytes()).hexdigest()
    data_cfg={"traffic":TRAFFIC,"slots":a.slots,"history":50,"horizon":1,"split_seeds":seeds}
    meta={"model":"AttentionLSTMIdlePredictor","input_source":"bs_true_idle_history","target_semantics":"idle_probability","input_size":1,"hidden_size":64,"num_layers":1,"attention_size":64,"dropout":0.1,"output_size":1,"history_window":50,"prediction_horizon_slots":1,"model_sha256":sha,"data_config_hash":hashlib.sha256(json.dumps(data_cfg,sort_keys=True).encode()).hexdigest(),"training":{**data_cfg,"seed":a.seed,"epochs_requested":80,"epochs_run":epochs,"batch_size":128,"learning_rate":1e-3,"early_stopping_patience":10},"metrics":{"brier_score":float(np.mean((pred-target)**2)),"accuracy":float(np.mean((pred>=.5)==target)),"ece":ece(target,pred),"val_loss":best,"training_time_s":time.perf_counter()-started,"inference_latency_ms":latency,"parameter_count":sum(p.numel() for p in model.parameters())}}
    out.with_suffix('.meta.json').write_text(json.dumps(meta,indent=2),encoding='utf-8'); print(json.dumps(meta,indent=2))
if __name__=='__main__': main()
