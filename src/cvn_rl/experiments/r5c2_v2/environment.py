"""Reuse legacy queue/mask/observation methods without editing their source."""
import numpy as np
from cvn_rl.simulation.multi_env import MultiVehicleAccessEnv
from cvn_rl.simulation.actions import decode_action
from .physics import rate_mbps

STATE=['fresh_dqi','theta','current_cqi','current_channel','dwell_remaining','blackout_remaining','queue_occupancy',
       'hol_age','previous_channel_load','vehicle_context','dqi_available','prediction_available','power_cap','dqi_age']

def environment_config(entry,baseline,reference):
    return {'num_channels':10,'num_vehicles':10,'slot_duration_ms':1.,'module_variant':'no_dqi' if baseline else 'full',
            'protection_variant':'gate_state_mask','state_features':STATE,'r5c2_protocol_version':'r5c2_v2',
            'r5c2_age_feature_scale_ms':1000.,'current_cqi_reference_mbps':reference,'state':{'freshness_neutral_value':.5},
            'action_mask_thresholds':{'sensing_idle_min':.5,'underlay_power_cap_min':.2,'overlay_theta_min':None,'underlay_theta_min':None},
            'action':{'enabled_modes':['overlay','underlay']},'packet_traffic':entry['packet_traffic'],
            'underlay_power':{'p_max_mw':entry['physics']['p_max_mw'],'service_power_ceiling_fraction':entry['physics']['underlay_ceiling_fraction']},
            'reward':{'type':'physical','service_weight':1.,'drop_weight':1.,'contention_weight':.25,'violation_weight':2.,'queue_weight':.05,'clip_reward':[-3.,2.]},
            'switching':{'min_dwell_slots':0,'blackout_slots':0}}

class PhysicalAccessEnv(MultiVehicleAccessEnv):
    def __init__(self,trace,config,bundle,score_start=0):
        super().__init__(trace,config); self.bundle=bundle; self.score_start=score_start
        self.window_completed=np.zeros(trace.num_vehicles,int)

    def _observations(self):
        t=min(self.t,self.trace.slots-1); n=self.trace.num_vehicles; m=self.trace.num_channels
        age=self.trace.dqi_age[t]/1000 if self.use_dqi else np.zeros(m)
        channel=np.zeros((n,m)); valid=np.flatnonzero(self.current_channels>=0)
        channel[valid,self.current_channels[valid]]=1
        reference=self.config['current_cqi_reference_mbps']
        cqi=np.clip(self.trace.quality_mbps[t-1]/reference,0,1) if t else np.full((n,m),.5)
        shared=lambda a:np.repeat(np.asarray(a)[None,:],n,axis=0)
        blocks={'fresh_dqi':shared(self.trace.fresh_dqi[t] if self.use_dqi else np.full(m,.5)),
                'theta':shared(self.trace.theta[t]),'current_cqi':cqi,'current_channel':channel,
                'dwell_remaining':(self.dwell_remaining/max(self.minimum_dwell_slots,1))[:,None],
                'blackout_remaining':(self.blackout_remaining/max(self.switch_blackout_slots,1))[:,None],
                'queue_occupancy':np.array([len(q)/self.queue_capacity for q in self.queues])[:,None],
                'hol_age':np.clip(self._hol_ages()/self.hol_age_scale_slots,0,1)[:,None],
                'previous_channel_load':shared(self.previous_channel_load),'vehicle_context':((np.arange(n)+.5)/n)[:,None],
                'dqi_available':np.full((n,1),float(self.use_dqi and np.any(self.trace.dqi_source_slot[t]>=0))),
                'prediction_available':np.ones((n,1)),'power_cap':self.trace.power_cap[t],'dqi_age':shared(age)}
        return np.concatenate([blocks[k] for k in self.state_features],axis=1)

    def _serve_queue(self,vehicle,service_bits,completed):
        available=max(float(service_bits),0); served=0.
        while available>0 and self.queues[vehicle]:
            packet=self.queues[vehicle][0]; amount=min(available,packet[0]); packet[0]-=amount; available-=amount; served+=amount
            if packet[0]<=1e-9:
                if packet[1]>=self.score_start: self.window_completed[vehicle]+=1
                self.queues[vehicle].popleft(); self.completed_packets[vehicle]+=1; completed[vehicle]+=1
        return served

    def physical_rate(self,t,v,m,power,busy,mode):
        a=self.bundle.arrays
        if mode=='overlay' and busy: return 0.
        interference=a['pu_to_vt_interference_mw'][t,v,m] if mode=='underlay' and busy else 0.
        return float(rate_mbps(a['gain_bs_true'][t,v,m],power,a['bandwidth_mhz'][t,m],a['noise_mw'][t,m],interference))

    def step(self,action_indices):
        raw=np.asarray(action_indices,int); t=self.t; n=self.trace.num_vehicles; m=self.trace.num_channels
        if raw.shape!=(n,): raise ValueError('one action per VT required')
        info=self.agent_infos()
        if any(not item['action_mask'][a] for item,a in zip(info,raw)): raise ValueError('infeasible action supplied')
        requested=[decode_action(int(a),m) for a in raw]
        decoded,_,_,_=self._resolve_switching_actions(requested)
        active=np.array([a.channel>=0 and a.mode!='no_access' and bool(self.queues[i]) for i,a in enumerate(decoded)])
        channels=np.array([a.channel if active[i] else -1 for i,a in enumerate(decoded)]); modes=np.array([a.mode if active[i] else 'no_access' for i,a in enumerate(decoded)])
        transmitted=np.zeros(n,bool); contenders=np.zeros(m,int)
        for channel in range(m):
            users=np.flatnonzero(active & (channels==channel)); contenders[channel]=len(users)
            if len(users): transmitted[users[np.argmin(self.trace.contention_priority[t,users,channel])]]=True
        lost=active & ~transmitted; power=np.zeros(n); served=np.zeros(n); completed=np.zeros(n,int); rates=np.zeros(n)
        interference=np.zeros(m); overlay_busy=np.zeros(n,bool); selected=np.full(n,np.nan)
        busy=self.trace.occupancy[t]; maximum=self.config['underlay_power']['p_max_mw']; ceiling=self.config['underlay_power']['service_power_ceiling_fraction']
        self.window_completed.fill(0)
        for v in np.flatnonzero(active):
            channel=channels[v]; selected[v]=self.trace.fresh_dqi[t,channel]
            self.current_channels[v]=channel
            if not transmitted[v]: continue
            power[v]=maximum if modes[v]=='overlay' else min(self.trace.transmit_power_cap_mw[t,v,channel],ceiling*maximum)
            interference[channel]+=power[v]*self.trace.su_pu_gain_true[t,v,channel]
            overlay_busy[v]=modes[v]=='overlay' and busy[channel]
            rates[v]=self.physical_rate(t,v,channel,power[v],busy[channel],modes[v])
            served[v]=self._serve_queue(v,rates[v]*1000,completed)
        channel_violation=busy & (interference>self.trace.pu_threshold_mw[t])
        underlay=(modes=='underlay') & transmitted
        underlay_violation=np.array([underlay[v] and channel_violation[channels[v]] for v in range(n)])
        access=overlay_busy|underlay_violation
        reward=self._physical_reward(served,lost,access,selected)
        record={'t':t,'served_bits':served,'packet_generated':self._arrivals_this_slot.copy(),'packet_dropped':self._drops_this_slot.copy(),
                'packet_completed':completed,'window_arrival_completed':self.window_completed.copy(),'transmitted':transmitted,'attempted':active,
                'contention_lost':lost,'pu_access_violation':access,'pu_violation_per_channel':channel_violation,'pu_busy_channels':busy.copy(),
                'actual_interference_mw':interference,'transmit_power_mw':power,'queue_lengths':np.array([len(q) for q in self.queues]),
                'mean_reward':float(np.mean(reward)),'underlay_transmitted':underlay,'overlay_transmitted':transmitted & (modes=='overlay'),
                'vt_throughput_per_vehicle':rates}
        self.previous_channel_load=contenders/max(n,1); self._advance_switch_timers(); self.t+=1
        done=self.t>=self.trace.slots
        if not done: self._apply_arrivals(self.t)
        else: self._arrivals_this_slot.fill(0); self._drops_this_slot.fill(0)
        return self._observations(),reward,done,record
