import copy
import numpy as np
import pytest
from cvn_rl.experiments.r5c2_v2.protocol import ROOT,read_json
from cvn_rl.experiments.r5c2_v2.training import WindowDataset
from cvn_rl.experiments.r5c2_v2.physics import generate,save_bundle,load_bundle
from cvn_rl.experiments.r5c2_v2.environment import PhysicalAccessEnv,environment_config
from cvn_rl.agents.shared_multi import create_shared_agent

def test_window_end_anchor_and_future_intervention():
    f=np.arange(400*40,dtype=np.float32).reshape(400,40); a=np.array([199,209]); y=np.ones((2,10),np.float32)
    pre={'mean':np.zeros(40),'std':np.ones(40)}; d=WindowDataset([(f,a,y)],pre)
    x,_=d[0]; np.testing.assert_array_equal(x.numpy(),f[:200])
    before=x.numpy().copy(); f[200:]*=2; np.testing.assert_array_equal(d[0][0].numpy(),before)

def test_physical_bundle_hash_rejects_tampering(tmp_path, monkeypatch):
    # The test's temporary workspace obeys the same write-scope guard as a checkout.
    from cvn_rl.experiments.r5c2_v2 import protocol
    monkeypatch.setattr(protocol, 'ROOT', tmp_path)
    e=read_json(ROOT/'configs/r2_r05c02_dqi_sensitivity_v2.json'); b=generate(e,6,12,10,'test')
    save_bundle(tmp_path,b); restored=load_bundle(tmp_path); assert restored.physical_hash()==b.physical_hash()
    with (tmp_path/'physical.npz').open('ab') as stream: stream.write(b'changed')
    with pytest.raises(ValueError): load_bundle(tmp_path)

def test_baseline_short_training_deterministic():
    e=read_json(ROOT/'configs/r2_r05c02_dqi_sensitivity_v2.json'); b=generate(e,7,25,10,'test'); agents=[]; outcomes=[]
    for value,period in [(.2,10),(.9,500)]:
        tr=b.trace(); tr.fresh_dqi[:]=value; tr.dqi_age[:]=np.arange(25)[:,None]%period; tr.dqi_source_slot[:]=0
        env=PhysicalAccessEnv(tr,environment_config(e,True,20),b); agent=create_shared_agent('proposed_esn_ddqn',10,10,8,e['agent_settings'])
        obs,info=env.reset(); agent.reset_episode(); acts=[]
        for _ in range(25):
            action=agent.select_actions(obs,info); nxt,reward,done,_=env.step(action); ni=env.agent_infos()
            agent.observe_batch(obs,action,reward,nxt,done,info,ni); obs,info=nxt,ni; acts.append(action.copy())
        agents.append(agent.parameter_hash()); outcomes.append(acts)
    assert agents[0]==agents[1]; np.testing.assert_array_equal(outcomes[0],outcomes[1])
