import numpy as np
import subprocess
import sys
from .protocol import ROOT,check_protected,bundle_root,write_json
from .physics import generate,umi_los,rate_mbps,dbm_to_mw,radial_speed
from .guidance import freshness,held_reports
from .data import raw_features,target_values
from .environment import PhysicalAccessEnv,environment_config

def validate(p):
    e=p['entry']; b=generate(e,91901,640,10,'behavior'); a=b.arrays; checks=[]
    expected=rate_mbps(a['gain_bs_true'],e['physics']['p_ref_mw'],a['bandwidth_mhz'][:,None,:],a['noise_mw'][:,None,:])
    np.testing.assert_allclose(expected,a['rate_mbps']); np.testing.assert_allclose(10**(a['rss_dbm']/10),e['physics']['p_ref_mw']*a['gain_bs_true'])
    np.testing.assert_allclose(10**(a['snr_db']/10),e['physics']['p_ref_mw']*a['gain_bs_true']/a['noise_mw'][:,None,:]); checks.append('rate/SNR/RSS same gain and bandwidth-dependent noise')
    np.testing.assert_allclose(a['cap_mw']*a['gain_pu_estimated']*e['physics']['protection_margin']<=a['pu_tolerance_mw'][:,None,:]+1e-25,True); checks.append('estimated cap inequality')
    f=freshness([.8,.8,.8],[0,100,101],[0,0,0],100,100); np.testing.assert_allclose(f['fresh'],[.8,.8/np.e,0]); checks.append('freshness boundaries')
    source,rho,age=held_reports(b,50)
    np.testing.assert_array_equal(rho[201:250],np.repeat(rho[200:201],49,axis=0)); checks.append('received coherence held between broadcasts')
    cfg=environment_config(e,True,20.); t1=b.trace(); t2=b.trace(); t1.theta[:]=.1; t2.theta[:]=.9
    env1=PhysicalAccessEnv(t1,cfg,b); env2=PhysicalAccessEnv(t2,cfg,b)
    _,infos1=env1.reset(); _,infos2=env2.reset(); np.testing.assert_array_equal(np.stack([x['action_mask'] for x in infos1]),np.stack([x['action_mask'] for x in infos2])); checks.append('theta-free hard mask')
    r1=env1.physical_rate(0,0,0,.01,True,'underlay'); r2=env1.physical_rate(0,0,0,1,True,'underlay')
    assert 0<=r1<r2; assert env1.physical_rate(0,0,0,0,True,'underlay')==0; checks.append('actual power controls service')
    cutoff=300; before=raw_features(b)[:cutoff+1].copy(); a['rate_mbps'][cutoff+1:]*=2
    np.testing.assert_array_equal(before,raw_features(b)[:cutoff+1]); checks.append('future rate not an input feature')
    check_protected()
    files=['tests/test_r5c2_v2_physics.py','tests/test_r5c2_v2_pipeline.py','tests/test_dqi_labels.py','tests/test_d3_attention_dqi_predictor.py',
           'tests/test_multi_vehicle.py','tests/test_statistics.py','tests/test_config.py','tests/test_cli_dry_run.py']
    args=[sys.executable,'-m','pytest','-p','no:cacheprovider',*files,'--basetemp=tmp/pytest/r5c2_v2_gate']
    run=subprocess.run(args,cwd=ROOT,capture_output=True,text=True)
    root=bundle_root(p); root.mkdir(parents=True,exist_ok=True); (root/'behavior_tests.log').write_text(run.stdout+run.stderr,encoding='utf-8')
    if run.returncode: raise RuntimeError(f'behavior/compatibility tests failed; {root / "behavior_tests.log"}')
    result={'protocol_hash':p['protocol_hash'],'passed':True,'checks':checks,'research_evidence':False,
      'pytest_exit_code':run.returncode,'pytest_files':files,'scope':'deterministic behavior tests and legacy compatibility; no scientific-performance claims'}
    write_json(bundle_root(p)/'validation.json',result); return result
