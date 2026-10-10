from pathlib import Path
import numpy as np
import torch
from cvn_rl.predictors.attention_lstm import AttentionLSTMDQIPredictor
from .protocol import read_json,file_hash,FEATURES

def build_model(recipe):
    return AttentionLSTMDQIPredictor(40,recipe['hidden_size'],recipe['num_layers'],recipe['attention_size'],recipe['dropout'],10)

class FrozenDQIV2:
    def __init__(self,path,expected,preprocessing,device='cpu'):
        self.path=Path(path); self.meta=read_json(self.path.with_suffix('.meta.json')); self.device=torch.device(device)
        for key in ['target_hash','data_spec_hash','recipe_hash','preprocessing_hash']:
            if self.meta[key]!=expected[key]: raise ValueError(f'DQI identity mismatch: {key}')
        if self.meta['feature_names']!=FEATURES or self.meta['history_slots']!=200 or self.meta['version']!='r5c2_v2': raise ValueError('four-feature v2 model required')
        if preprocessing['hash']!=self.meta['preprocessing_hash'] or file_hash(self.path)!=self.meta['model_sha256']: raise ValueError('model/scaler hash')
        self.model=build_model(self.meta['recipe']).to(self.device); self.model.load_state_dict(torch.load(self.path,map_location=self.device,weights_only=True)); self.model.eval()
        self.mean=np.asarray(preprocessing['mean'],np.float32); self.std=np.asarray(preprocessing['std'],np.float32)

    def predict_windows(self,features,updates,batch_size=128):
        updates=np.asarray(updates,int)
        if np.any(updates<200) or np.any(updates>len(features)): raise ValueError('incomplete history window')
        flat=np.asarray(features).reshape(-1,40); output=[]
        with torch.no_grad():
            for i in range(0,len(updates),batch_size):
                x=np.stack([(flat[t-200:t]-self.mean)/self.std for t in updates[i:i+batch_size]])
                output.append(self.model(torch.as_tensor(x,dtype=torch.float32,device=self.device)).cpu().numpy())
        return np.concatenate(output) if output else np.empty((0,10))
