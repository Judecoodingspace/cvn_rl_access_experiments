"""Shared physical traces for revision experiments; no controller/target protocol."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
from scipy.special import j0
from cvn_rl.simulation.traffic import generate_occupancy
from cvn_rl.simulation.multi_traces import MultiVehicleAccessTrace
from cvn_rl.simulation.revision_physics_spec import array_hash, digest, file_hash, read_json, write_json, validate_physics

C_LIGHT=3e8

def dbm_to_mw(x): return np.power(10.,np.asarray(x,dtype=float)/10.)

def umi_los(distance_2d, distance_3d, fc_hz, h_bs=10., h_ut=1.5):
    d=np.asarray(distance_2d,float); d3=np.asarray(distance_3d,float)
    if np.any(~np.isfinite(d)) or np.any(~np.isfinite(d3)) or np.any(d3<d) or np.any((d<10)|(d>5000)) or h_bs!=10 or not 1.5<=h_ut<=22.5 or not .5e9<fc_hz<100e9:
        raise ValueError('outside UMi LOS applicability range; no distance clamping')
    bp=4*(h_bs-1)*(h_ut-1)*fc_hz/C_LIGHT
    f=fc_hz/1e9
    return np.where(d<=bp,32.4+21*np.log10(d3)+20*np.log10(f),
                    32.4+40*np.log10(d3)+20*np.log10(f)-9.5*np.log10(bp**2+(h_bs-h_ut)**2))

def umi_nlos(distance_2d, distance_3d, fc_hz, h_bs=10., h_ut=1.5):
    los=umi_los(distance_2d,distance_3d,fc_hz,h_bs,h_ut)
    nlos=22.4+35.3*np.log10(distance_3d)+21.3*np.log10(fc_hz/1e9)-.3*(h_ut-1.5)
    return np.maximum(los,nlos)

def pathloss(distance_2d,distance_3d,settings):
    function=umi_los if settings['pathloss']=='UMi_LOS_Table7.4.1-1' else umi_nlos
    return function(distance_2d,distance_3d,settings['carrier_hz'],10.,settings['vt_height_m'])

def radial_speed(position, velocity, endpoint):
    delta=np.asarray(position)-np.asarray(endpoint)
    return np.sum(delta*np.asarray(velocity),axis=-1)/np.linalg.norm(delta,axis=-1)

def rate_mbps(gain, power_mw, bandwidth_mhz, noise_mw, interference_mw=0.):
    return np.asarray(bandwidth_mhz)*np.log2(1+np.asarray(power_mw)*np.asarray(gain)/(np.asarray(noise_mw)+np.asarray(interference_mw)))

@dataclass
class PhysicalBundle:
    arrays: dict
    slots: int
    future_slots: int
    seed: int
    split: str
    spec_hash: str

    def physical_hash(self): return array_hash(self.arrays)

    def trace(self):
        a=self.arrays; s=self.slots; _,n,m=a['rate_mbps'].shape
        shared=np.zeros((s,m)); index=np.full((s,m),-1,dtype=int)
        return MultiVehicleAccessTrace(
            occupancy=a['occupancy'][:s],sensing_idle=a['sensing_idle'][:s],dqi_predicted=np.full_like(shared,.5),
            fresh_dqi=np.full_like(shared,.5),dqi_age=shared.copy(),dqi_validity_slots=np.full_like(shared,1000.),
            dqi_source_slot=index.copy(),dqi_input_cutoff_slot=index.copy(),theta=np.full_like(shared,.5),
            theta_input_cutoff_slot=index.copy(),theta_target_slot=index.copy(),quality_mbps=a['rate_mbps'][:s],
            quality_lookahead_mbps=a['rate_mbps'][s:],snr_db=a['snr_db'][:s],rss_dbm=a['rss_dbm'][:s],
            cci=a['rho'][:s],future_dqi_oracle=np.full_like(shared,np.nan),
            power_cap=a['power_cap'][:s],occupancy_type=a['occupancy'][:s].astype(float),
            su_pu_gain_hat=a['gain_pu_estimated'][:s],su_pu_gain_true=a['gain_pu_true'][:s],
            pu_threshold_mw=a['pu_tolerance_mw'][:s],transmit_power_cap_mw=a['cap_mw'][:s],
            packet_arrivals=a['arrivals'][:s],bandwidth_mhz=a['bandwidth_mhz'][:s],
            pu_signal_mw=a['pu_desired_signal_mw'][:s],noise_mw=a['noise_mw'][:s],
            contention_priority=a['contention_priority'][:s],trace_id=f'revision-{self.split}-s{self.seed}-{self.physical_hash()[:12]}',
            split=self.split,seed=self.seed,dqi_model_id='unfilled',dqi_model_hash='',prediction_model_id='unfilled',prediction_model_hash='',
            dqi_inference_latency_ms=0.,dqi_parameter_count=0,prediction_inference_latency_ms=0.,prediction_parameter_count=0,
            dqi_validation_mse=float('nan'),prediction_validation_brier=float('nan'),
            dqi_metadata={'version':'revision_physics_v1','coherence_source':'radial_doppler','physical_hash':self.physical_hash()})

def _gain(position,velocity,endpoints,s,shadow_rng,fading_rng,slot_seconds=.001):
    delta=position[:,:,None,:]-endpoints[None,None,:,:]
    d2=np.linalg.norm(delta[...,:2],axis=-1); d3=np.linalg.norm(delta,axis=-1)
    pl=pathloss(d2,d3,s)
    shape=d3.shape; shadow=np.empty(shape); fading=np.empty(shape,dtype=complex)
    shadow[0]=shadow_rng.normal(0,s['shadow_sigma_db'],shape[1:])
    fading[0]=(fading_rng.normal(size=shape[1:])+1j*fading_rng.normal(size=shape[1:]))/np.sqrt(2)
    vr=np.sum(delta*velocity[:,:,None,:],axis=-1)/d3
    alpha=j0(2*np.pi*np.abs(vr)*s['carrier_hz']/C_LIGHT*slot_seconds)
    # Per-step draws preserve each stream's prefix when trace length changes.
    for t in range(1,shape[0]):
        distance_step=np.linalg.norm(velocity[t-1],axis=-1)*slot_seconds
        ash=np.exp(-distance_step[:,None]/s['shadow_correlation_distance_m'])
        shadow[t]=ash*shadow[t-1]+np.sqrt(1-ash**2)*shadow_rng.normal(0,s['shadow_sigma_db'],shape[1:])
        z=(fading_rng.normal(size=shape[1:])+1j*fading_rng.normal(size=shape[1:]))/np.sqrt(2)
        fading[t]=alpha[t]*fading[t-1]+np.sqrt(np.maximum(1-alpha[t]**2,0))*z
    k=10**(s['rician_k_db']/10)
    rician=np.abs(np.sqrt(k/(k+1))+np.sqrt(1/(k+1))*fading)**2
    family=s.get('small_scale_fading','rician')
    if family=='rayleigh': rician=np.abs(fading)**2
    elif family=='none': rician=np.ones_like(rician)
    return np.power(10.,-(pl+shadow)/10.)*rician,pl

def generate(entry,seed,slots,vehicles,split,future_slots=0):
    s=entry['physics']; m=entry['num_channels']; total=slots+future_slots
    validate_physics(s)
    if slots<=0 or future_slots<0 or vehicles<=0 or m<=0 or entry['slot_duration_ms']<=0:
        raise ValueError('positive physical dimensions/time required')
    streams=[np.random.default_rng(x) for x in np.random.SeedSequence(seed).spawn(12)]
    time=np.arange(total)*entry['slot_duration_ms']/1000
    initial=streams[0].uniform(*s['initial_x_range_m'],size=vehicles)
    directions=np.where(np.arange(vehicles)%2==0,1.,-1.) if s['direction_mode']=='alternating' else np.ones(vehicles)
    v=directions*s['speed_kmh']/3.6
    position=np.zeros((total,vehicles,3)); position[:,:,0]=initial[None,:]+time[:,None]*v
    position[:,:,1]=s['road_y_m']; position[:,:,2]=s['vt_height_m']
    velocity=np.zeros_like(position); velocity[:,:,0]=v
    if s.get('speed_schedule'):
        speed=np.full(total,s['speed_kmh']/3.6)
        for row in s['speed_schedule']: speed[row['start_slot']:]=row['speed_kmh']/3.6
        velocity[:,:,0]=speed[:,None]*directions
        displacement=np.r_[0.,np.cumsum(speed[:-1]*entry['slot_duration_ms']/1000)]
        position[:,:,0]=initial[None,:]+displacement[:,None]*directions
    bs=np.repeat(np.asarray(s['bs_position_m'])[None,:],m,axis=0)
    pu=np.stack([np.linspace(*s['pu_receiver_x_range_m'],m),np.full(m,s['pu_receiver_y_m']),np.full(m,s['pu_receiver_height_m'])],axis=-1)
    settings=lambda link:{**s,**s.get('link_overrides',{}).get(link,{})}
    dt=entry['slot_duration_ms']/1000
    gain_bs,pl_bs=_gain(position,velocity,bs,settings('vt_bs'),streams[1],streams[2],dt)
    gain_pu,pl_pu=_gain(position,velocity,pu,settings('vt_pu'),streams[3],streams[4],dt)
    ptx=pu+np.asarray(s['pu_transmitter_offset_m'])
    gain_ptx,_=_gain(position,velocity,ptx,settings('pu_vt'),streams[5],streams[6],dt)
    bw_static=np.resize(np.asarray(s['bandwidth_mhz']),m)
    bw=np.broadcast_to(bw_static,(total,m)).copy()
    noise=dbm_to_mw(s['thermal_noise_dbm_hz']+s['receiver_noise_figure_db'])*bw*1e6
    rx=s['p_ref_mw']*gain_bs
    snr=10*np.log10(rx/noise[:,None,:]); rss=10*np.log10(rx)
    rates=rate_mbps(gain_bs,s['p_ref_mw'],bw[:,None,:],noise[:,None,:])
    vr=radial_speed(position,velocity,s['bs_position_m'])
    doppler=np.broadcast_to((np.abs(vr)*s['carrier_hz']/C_LIGHT)[:,:,None],(total,vehicles,m)).copy()
    reports=streams[7].random((total,vehicles,m))<s['report_probability']
    maximum=np.max(np.where(reports,doppler,-np.inf),axis=1)
    rho=np.clip(1-maximum/(s['reference_max_speed_kmh']/3.6*s['carrier_hz']/C_LIGHT),0,1)
    rho=np.where(np.any(reports,axis=1),rho,0.)
    cam=np.maximum((np.arange(total)-s['additional_report_delay_slots'])//s['cam_update_slots']*s['cam_update_slots'],0)
    # Error stream is independent of true propagation; reserved for future R2/R5-C3 experiments.
    if s.get('position_error_distribution','gaussian')=='uniform':
        bound=np.sqrt(3)*s['position_error_std_m']
        error=streams[8].uniform(-bound,bound,position.shape)
    else:
        error=streams[8].normal(0,s['position_error_std_m'],position.shape)
    reported=position[cam]+error[cam]; reported[:,:,2]=s['vt_height_m']
    dd=reported[:,:,None,:]-pu[None,None,:,:]
    estimator={**s,'pathloss':s.get('estimator_pathloss',s['pathloss'])}
    estimated=np.power(10.,-pathloss(np.linalg.norm(dd[...,:2],axis=-1),np.linalg.norm(dd,axis=-1),estimator)/10.)
    estimated*=10**(s['estimated_gain_bias_db']/10)
    if s.get('estimated_gain_error_std_db',0)>0:
        gain_error_rng=np.random.default_rng(np.random.SeedSequence([seed,103]))
        gain_error=gain_error_rng.normal(0,s['estimated_gain_error_std_db'],estimated.shape)[cam]
        estimated*=10**(gain_error/10)
    cap_source=np.arange(total)//s['cap_update_slots']*s['cap_update_slots']
    tolerance=np.full((total,m),dbm_to_mw(s['pu_tolerance_dbm']))
    cap=np.minimum(s['p_max_mw'],tolerance[:,None,:]/(s['protection_margin']*estimated))[cap_source]
    estimated_received=estimated[cap_source]
    occupancy=generate_occupancy(total,m,entry['traffic'],streams[9])
    false_rng=np.random.default_rng(np.random.SeedSequence([seed,102]))
    sensing=np.where(occupancy[:,None,:],streams[10].random((total,vehicles,m))>=entry['traffic']['sensing_pd'],
                     false_rng.random((total,vehicles,m))>=entry['traffic']['sensing_pfa']).astype(float)
    # Separate per-time draws for arrivals and priorities prevent length-dependent cross-stream shifts.
    arr_rng=np.random.default_rng(np.random.SeedSequence([seed,100])); priority_rng=np.random.default_rng(np.random.SeedSequence([seed,101]))
    arrays=dict(position_m=position,velocity_m_s=velocity,reported_position_m=reported,report_mask=reports,doppler_hz=doppler,rho=rho,
                gain_bs_true=gain_bs,gain_pu_true=gain_pu,gain_pu_estimated=estimated_received,pl_bs_db=pl_bs,pl_pu_db=pl_pu,
                cam_source_slot=cam,cap_source_slot=cap_source,cap_mw=cap,power_cap=cap/s['p_max_mw'],
                rate_mbps=rates,snr_db=snr,rss_dbm=rss,bandwidth_mhz=bw,noise_mw=noise,pu_tolerance_mw=tolerance,
                pu_to_vt_interference_mw=s['pu_tx_power_mw']*gain_ptx,
                pu_desired_signal_mw=np.full((total,m),dbm_to_mw(s['pu_desired_rx_power_dbm'])),
                occupancy=occupancy,sensing_idle=sensing,arrivals=arr_rng.random((total,vehicles))<entry['packet_traffic']['arrival_probability'],
                contention_priority=priority_rng.random((total,vehicles,m)))
    return PhysicalBundle(arrays,slots,future_slots,seed,split,digest({'physics':s,'traffic':entry['traffic'],'packet':entry['packet_traffic'],'vehicles':vehicles,'channels':m,'slot':entry['slot_duration_ms']}))

def save_bundle(path,b):
    path=Path(path); path.mkdir(parents=True,exist_ok=True)
    temp=path/'physical.partial.npz'; np.savez_compressed(temp,**b.arrays); temp.replace(path/'physical.npz')
    write_json(path/'manifest.json',{'schema':'r5c2_physical_v2','slots':b.slots,'future_slots':b.future_slots,'seed':b.seed,'split':b.split,
                                    'spec_hash':b.spec_hash,'physical_hash':b.physical_hash(),'file_sha256':file_hash(path/'physical.npz')})

def load_bundle(path):
    path=Path(path); meta=read_json(path/'manifest.json')
    if meta['schema']!='r5c2_physical_v2' or file_hash(path/'physical.npz')!=meta['file_sha256']: raise ValueError('physical bundle/schema/hash mismatch')
    with np.load(path/'physical.npz',allow_pickle=False) as z: arrays={k:z[k] for k in z.files}
    b=PhysicalBundle(arrays,meta['slots'],meta['future_slots'],meta['seed'],meta['split'],meta['spec_hash'])
    if b.physical_hash()!=meta['physical_hash']: raise ValueError('physical content mismatch')
    return b
