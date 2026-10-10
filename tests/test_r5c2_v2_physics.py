import copy
import numpy as np
import pytest
from cvn_rl.experiments.r5c2_v2.protocol import ROOT,read_json,check_protected,conditions
from cvn_rl.experiments.r5c2_v2.physics import generate,umi_los,radial_speed,rate_mbps
from cvn_rl.experiments.r5c2_v2.data import raw_features,target_values
from cvn_rl.experiments.r5c2_v2.guidance import freshness,held_reports
from cvn_rl.experiments.r5c2_v2.environment import PhysicalAccessEnv,environment_config

@pytest.fixture
def entry(): return read_json(ROOT/'configs/r2_r05c02_dqi_sensitivity_v2.json')

def test_old_sources_preserved(): assert not check_protected()['changed']

def test_umilos_domains_and_piecewise():
    d=np.array([100.,354.,355.,1000.]); d3=np.sqrt(d*d+8.5**2); actual=umi_los(d,d3,5.9e9)
    expected=32.4+21*np.log10(d3[0])+20*np.log10(5.9)
    assert actual[0]==pytest.approx(expected)
    with pytest.raises(ValueError): umi_los([9],[12],5.9e9)

def test_physical_prefix_and_sources(entry):
    a=generate(entry,1,205,10,'test'); b=generate(entry,1,220,10,'test')
    for key,x in a.arrays.items(): np.testing.assert_array_equal(x,b.arrays[key][:205],err_msg=key)
    x=a.arrays; e=entry['physics']
    np.testing.assert_allclose(x['rate_mbps'],rate_mbps(x['gain_bs_true'],e['p_ref_mw'],x['bandwidth_mhz'][:,None,:],x['noise_mw'][:,None,:]))
    np.testing.assert_allclose(10**(x['rss_dbm']/10),e['p_ref_mw']*x['gain_bs_true'])
    np.testing.assert_array_equal(x['reported_position_m'][1:100],np.repeat(x['reported_position_m'][0:1],99,axis=0))
    np.testing.assert_array_equal(x['cap_mw'][1:10],np.repeat(x['cap_mw'][0:1],9,axis=0))

def test_radial_velocity():
    assert radial_speed([[0,20,1.5]],[[5,0,0]],[0,0,10])[0]==0
    assert radial_speed([[30,0,0]],[[5,0,0]],[0,0,0])[0]==5

@pytest.mark.parametrize('rule',['exp','linear','off'])
def test_decay_boundaries(rule):
    value=freshness([.8]*3,[0,100,101],[0]*3,100,100,1,rule)
    np.testing.assert_allclose(value['fresh'],[.8]*3 if rule=='off' else [.8,.8/np.e,0])

def test_no_reports_and_held_coherence(entry):
    b=generate(entry,2,400,10,'test'); source,rho,age=held_reports(b,50)
    np.testing.assert_array_equal(rho[201:250],np.repeat(rho[200:201],49,0))
    b.arrays['report_mask'][249]=False
    source2,_,_=held_reports(b,50); np.testing.assert_array_equal(source2[250],source2[249])

def test_features_future_and_target_anchor(entry):
    b=generate(entry,3,410,15,'test',200); original=raw_features(b).copy(); anchors=np.array([199,209])
    c=conditions()[0]; z=target_values(b,anchors,c)
    from cvn_rl.simulation.dqi_labels import compute_unnormalized_dqi
    expected,_=compute_unnormalized_dqi(b.arrays['rate_mbps'],source_indices=anchors)
    np.testing.assert_allclose(z,expected)
    b.arrays['snr_db'][300:]+=100; np.testing.assert_array_equal(raw_features(b)[:300],original[:300])
    assert original.shape[-1]==4

def test_theta_mask_and_baseline_observation(entry):
    b=generate(entry,4,20,10,'test'); t1=b.trace(); t2=b.trace(); t1.theta[:]=.1; t2.theta[:]=.9
    cfg=environment_config(entry,True,20); x=PhysicalAccessEnv(t1,cfg,b); y=PhysicalAccessEnv(t2,cfg,b)
    _,i=x.reset(); _,j=y.reset(); np.testing.assert_array_equal(np.stack([q['action_mask'] for q in i]),np.stack([q['action_mask'] for q in j]))
    t2.theta[:]=t1.theta; t2.fresh_dqi[:]=.9; t2.dqi_age[:]=500; t2.dqi_source_slot[:]=0
    np.testing.assert_array_equal(x._observations(),y._observations())

def test_power_service_and_violation(entry):
    b=generate(entry,5,5,10,'test'); tr=b.trace(); env=PhysicalAccessEnv(tr,environment_config(entry,False,20),b)
    assert env.physical_rate(0,0,0,.01,True,'underlay')<env.physical_rate(0,0,0,1,True,'underlay')
    assert env.physical_rate(0,0,0,0,True,'underlay')==0
    assert env.physical_rate(0,0,0,1,True,'overlay')==0
    obs,_=env.reset(); assert obs.shape[0]==10 and np.all(np.isfinite(obs))
    tr.dqi_source_slot[:]=0; assert np.all(np.isfinite(env._observations()))

def test_matrix_complete():
    c=conditions(); assert len(c)==19; assert len({x['target_id'] for x in c})==11
    assert c[11]['group']=='main' and c[14]['group']=='stress' and c[18]['rule']=='linear'
