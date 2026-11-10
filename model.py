"""Event/state based synchronized uplink RA. All times seconds; energy joules."""
import numpy as np
from scipy.optimize import brentq
from channel import Loo,sample_at_elevation,thermal_noise_dbm,fspl_db

SCENES=('urban','village','rural_wooded')
CAUSES=('preamble_collision','preamble_miss','payload_collision','payload_decode')

def validate(c):
    for k in ('passes','preamble_pool_size','preamble_to_pusch_ratio','pusch_repetitions'):
        if not isinstance(c[k],int) or c[k]<1:raise ValueError(k+' must be a positive integer')
    if c['passes']<2:raise ValueError('At least 2 independent passes required')
    if c['preamble_pool_size']%c['preamble_to_pusch_ratio']:raise ValueError('Mapping must divide pool exactly')
    if c['repetitions'] not in (1,2,4,8,16,32,64,128):raise ValueError('Unsupported repetitions')
    if any(not isinstance(n,int) or n<1 for n in c['sensor_counts']):raise ValueError('Invalid sensor counts')
    for x in c['retry_intervals_s']:
        if x is None or x=='uniform_1_5':continue
        if not isinstance(x,(int,float)) or not np.isfinite(x) or x<0:raise ValueError('Invalid retry interval')
    if not set(c['scenarios']).issubset(SCENES):raise ValueError('Unknown scenario')
    for k in ('preamble_gap_ms','payload_offset_ms','receiver_processing_delay_ms','system_loss_db'):
        if not np.isfinite(c[k]) or c[k]<0:raise ValueError(k)
    for k in ('payload_duration_ms','ra_period_ms','altitude_km','frequency_hz'):
        if not np.isfinite(c[k]) or c[k]<=0:raise ValueError(k)
    for k in ('power_preamble_tx_w','power_payload_tx_w','power_rx_w','power_idle_w','power_sleep_w'):
        if not np.isfinite(c[k]) or c[k]<0:raise ValueError(k)
    if not 20<c['max_elevation_deg']<=70:raise ValueError('Peak elevation outside supported domain')
    if not 0<c['false_alarm_probability']<1:raise ValueError('Invalid P_FA')
    if not np.isfinite(c['access_trigger_snr_db']):raise ValueError('Invalid trigger')
    if not 0<=c['ra_offset_ms']<c['ra_period_ms']:raise ValueError('Invalid RA offset')
    if durations(c)[2]>=c['ra_period_ms']/1000:raise ValueError('Full transmission must fit a resource period')

def durations(c):
    # Simplified format-0 active duration. Gap is explicitly research-defined.
    pre=.0056*c['repetitions']+c['preamble_gap_ms']/1000
    payload=c['payload_duration_ms']/1000*c['pusch_repetitions']
    return pre,payload,pre+c['payload_offset_ms']/1000+payload

def psi(deg,alt):
    a=np.radians(deg)
    return np.arccos(6371/(6371+alt)*np.cos(a))-a

class Geometry:
    def __init__(self,c):
        self.radius=6371+c['altitude_km'];self.omega=np.sqrt(398600.4418/self.radius**3)
        self.beta=psi(c['max_elevation_deg'],c['altitude_km'])
        self.peak=np.arccos(np.cos(psi(10,c['altitude_km']))/np.cos(self.beta))/self.omega
        self.end=2*self.peak
    def at(self,t):
        x=self.radius*np.cos(self.omega*(np.asarray(t)-self.peak))*np.cos(self.beta)
        d=np.sqrt(self.radius**2+6371**2-2*6371*x)
        return np.degrees(np.arcsin(np.clip((x-6371)/d,-1,1))),d

def raw_snr(c,d):
    return c['uplink_tx_eirp_dbm']+c['satellite_rx_gain_dbi']-fspl_db(d,c['frequency_hz'])-c['system_loss_db']-thermal_noise_dbm(c['temperature_k'],c['bandwidth_hz'],c['noise_figure_db'])

def trigger(c,g):
    def excess(t):return float(raw_snr(c,g.at(t)[1])+10*np.log10(c.get('trigger_reference_repetitions',c['repetitions']))-c['access_trigger_snr_db'])
    if excess(0)>=0:return 0.
    if excess(g.peak)<0:return None
    return float(brentq(excess,0,g.peak))

def radio_outcomes(raw,choices,noise_i,noise_q,c):
    """Simultaneous equal-length receptions on disjoint resource bands.

    No inter-occasion overlap is permitted by simulate's interval guard.
    Thus bincount is exactly equivalent to pairwise time-frequency checking.
    """
    pool=c['preamble_pool_size'];resource=choices//c['preamble_to_pusch_ratio']
    pc=np.bincount(choices,minlength=pool)[choices]>1
    qc=np.bincount(resource,minlength=pool//c['preamble_to_pusch_ratio'])[resource]>1
    power=10**(np.clip(raw,-300,300)/10)
    total=np.bincount(resource,weights=power,minlength=pool//c['preamble_to_pusch_ratio'])
    interference=np.maximum(total[resource]-power,0)
    sinr=10*np.log10(power/(1+interference))+10*np.log10(c['pusch_repetitions'])
    # Ideal combining exactly once; noise normalized to CN(0,1).
    rho=power*c['repetitions']
    miss=(np.sqrt(rho)+noise_i)**2+noise_q**2 < -np.log(c['false_alarm_probability'])
    bad_decode=sinr<c['pusch_sinr_threshold_db']
    success=~(pc|miss|qc|bad_decode)
    cause=np.select([pc,miss,qc,bad_decode],[0,1,2,3],default=4)
    return success,cause,dict(preamble_collision=pc,preamble_miss=miss,payload_collision=qc,payload_decode=bad_decode),sinr

class EnergyLedger:
    """Close each sensor state interval exactly once; retain aggregate rate events."""
    def __init__(self,n,t0,power):
        self.n=n;self.t0=t0;self.power=power;self.state=np.full(n,'sleep',dtype='U12')
        self.current_power=np.full(n,power['sleep'])
        self.last=np.full(n,t0);self.energy=np.zeros(n)
        self.events=[(t0,n*power['sleep'])]
    def transition(self,ids,time,state):
        ids=np.asarray(ids,dtype=int)
        if not len(ids):return
        dt=time-self.last[ids]
        if np.any(dt < -1e-9):raise ValueError('Backwards energy event')
        old=self.current_power[ids]
        self.energy[ids]+=old*np.maximum(dt,0)
        self.events.append((float(time),float(np.sum(self.power[state]-old))))
        self.last[ids]=time;self.state[ids]=state;self.current_power[ids]=self.power[state]
    def close(self,time):
        self.transition(np.arange(self.n),time,'sleep')
    def cumulative(self,grid):
        events=sorted(self.events);times=np.array([x[0] for x in events]);delta=np.array([x[1] for x in events])
        rate=np.cumsum(delta);area=np.r_[0,np.cumsum(rate[:-1]*np.diff(times))]
        idx=np.searchsorted(times,grid,side='right')-1
        return np.where(idx>=0,area[np.maximum(idx,0)]+rate[np.maximum(idx,0)]*(grid-times[np.maximum(idx,0)]),0)

def distribution(values,prefix):
    return {prefix+'_'+key:float(fun(values)) if len(values) else None for key,fun in (
        ('mean',np.mean),('median',np.median),('p90',lambda x:np.percentile(x,90)),('p95',lambda x:np.percentile(x,95)))}

def retry_delays(retry,n,seed,scene_index,pass_id,occasion):
    if retry is None:return np.full(n,np.inf)
    if retry=='uniform_1_5':
        rng=np.random.default_rng(np.random.SeedSequence([seed,scene_index,pass_id,occasion,808]))
        return rng.uniform(1.,5.,size=n)
    return np.full(n,float(retry))

def simulate(c,scene,n,retry,pass_id,save_devices=False):
    validate(c);g=Geometry(c);t0=trigger(c,g)
    policy='one_attempt' if retry is None else 'retry_uniform_1_5s' if retry=='uniform_1_5' else f'retry_{retry:g}s'
    meta=dict(scenario=scene,sensors=n,policy=policy,pass_id=pass_id,trigger_snr_db=c['access_trigger_snr_db'],
        pool=c['preamble_pool_size'],mapping=c['preamble_to_pusch_ratio'],repetitions=c['repetitions'])
    if t0 is None:
        # Undefined observation window; no artificial zero-energy efficiency.
        return meta|dict(trigger_reached=0,t0_s=None,trigger_elevation_deg=None,attempts=0,successes=0,success_ratio=0.,energy_total_j=None,energy_per_success_j=None),[],[],[]
    pre,payload,total=durations(c);processing=c['receiver_processing_delay_ms']/1000
    power=dict(sleep=c['power_sleep_w'],idle=c['power_idle_w'],rx=c['power_rx_w'],preamble_tx=c['power_preamble_tx_w'],payload_tx=c['power_payload_tx_w'])
    ledger=EnergyLedger(n,t0,power);count=np.zeros(n,int);success_time=np.full(n,np.nan);energy_success=np.full(n,np.nan)
    ready=np.full(n,t0);cause_counts=np.zeros(4,int);marginal=np.zeros(4,int);attempts=0;detected=decoded=0;latency_sum=0.
    wave_rows=[];loo=Loo();last_rx_end=-np.inf
    period=c['ra_period_ms']/1000;offset=c['ra_offset_ms']/1000
    times=offset+np.arange(int(np.ceil(g.end/period))+1)*period
    for k,t in enumerate(times):
        if t+1e-9<t0:continue
        theta,d=g.at(t);prop=float(d)/299792.458;outcome=t+total+prop+processing
        if outcome>g.end+1e-9:continue
        if float(theta)<10-1e-8:continue
        ids=np.flatnonzero(np.isnan(success_time)&(ready<=t+1e-9))
        if not len(ids):continue
        # Shared geometry; complete reception intervals cannot cross resource occasions.
        if t+prop<last_rx_end-1e-9:raise ValueError('Cross-occasion overlap: use general interval engine for this configuration')
        last_rx_end=t+total+prop
        si=SCENES.index(scene)
        # Same per-occasion realization and preamble choice across retry policies.
        # Draw all n slots so survivor removal never shifts another device's stream.
        rng=np.random.default_rng(np.random.SeedSequence([c['seed'],si,pass_id,k,0]))
        gain,_=loo.sample(scene,float(np.clip(theta,20,70)),rng,size=n)
        raw=float(raw_snr(c,d))+gain[ids]
        choices=np.random.default_rng(np.random.SeedSequence([c['seed'],si,pass_id,k,1])).integers(c['preamble_pool_size'],size=n)[ids]
        prng=np.random.default_rng(np.random.SeedSequence([c['seed'],si,pass_id,k,2]));noise=prng.normal(size=(n,2))/np.sqrt(2)
        good,cause,events,sinr=radio_outcomes(raw,choices,noise[ids,0],noise[ids,1],c)
        ledger.transition(ids,t,'preamble_tx')
        ledger.transition(ids,t+pre,'idle')
        ledger.transition(ids,t+pre+c['payload_offset_ms']/1000,'payload_tx')
        ledger.transition(ids,t+total,'idle')
        ledger.transition(ids,outcome,'sleep')
        count[ids]+=1;attempts+=len(ids);latency_sum+=len(ids)*(outcome-t)
        won=ids[good];success_time[won]=outcome;energy_success[won]=ledger.energy[won]
        # Independent per-device, per-failure backoff, separate random stream.
        delays=retry_delays(retry,n,c['seed'],si,pass_id,k)[ids[~good]]
        ready[ids[~good]]=outcome+delays
        for j,key in enumerate(CAUSES):cause_counts[j]+=int(np.sum(cause==j));marginal[j]+=int(events[key].sum())
        detected+=int((~events['preamble_collision']&~events['preamble_miss']).sum())
        decoded+=int((~events['payload_collision']&~events['payload_decode']).sum())
        wave_rows.append(meta|dict(occasion=k,start_s=float(t),elapsed_s=float(t-t0),elevation_deg=float(theta),outcome_s=float(outcome),attempts=len(ids),successes=int(good.sum()),
            **{key+'_count':int(np.sum(cause==j)) for j,key in enumerate(CAUSES)}))
    ledger.close(g.end);won=np.isfinite(success_time);ns=int(won.sum());uns=~won
    assert int(cause_counts.sum())+ns==attempts
    assert int(count.sum())==attempts
    grid=np.linspace(t0,g.end,201);cumulative=ledger.cumulative(grid)
    np.testing.assert_allclose(cumulative[-1],ledger.energy.sum(),rtol=1e-9,atol=1e-7)
    completion=np.searchsorted(np.sort(success_time[won]),grid,side='right')/n
    row=meta|dict(trigger_reached=1,t0_s=t0,trigger_elevation_deg=float(g.at(t0)[0]),window_s=g.end-t0,
        attempts=attempts,successes=ns,unsuccessful=n-ns,success_ratio=ns/n,attempt_success_probability=ns/attempts if attempts else None,
        attempts_per_successful_sensor=float(count[won].mean()) if ns else None,total_attempts_per_success=attempts/ns if ns else None,
        mean_attempt_latency_ms=1000*latency_sum/attempts if attempts else None,successful_accesses_per_second=ns/(g.end-t0),
        preamble_detection_probability=detected/attempts if attempts else None,payload_decoding_probability=decoded/attempts if attempts else None,
        overall_failure_probability=1-ns/attempts if attempts else None,
        energy_total_j=float(ledger.energy.sum()),energy_per_success_j=float(ledger.energy.sum()/ns) if ns else None,
        energy_unsuccessful_total_j=float(ledger.energy[uns].sum()),energy_unsuccessful_mean_j=float(ledger.energy[uns].mean()) if uns.any() else None,
        energy_successful_at_end_j=float(ledger.energy[won].sum()),
        **distribution(success_time[won]-t0,'delay_s'),**distribution(energy_success[won],'energy_until_success_j'),
        **{key+'_exclusive_count':int(cause_counts[j]) for j,key in enumerate(CAUSES)},
        **{key+'_probability':int(marginal[j])/attempts if attempts else None for j,key in enumerate(CAUSES)})
    timeline=[meta|dict(elapsed_s=float(x-t0),energy_total_j=float(e),completion_fraction=float(f)) for x,e,f in zip(grid,cumulative,completion)]
    devices=[]
    if save_devices:
        devices=[meta|dict(sensor_id=i,attempt_count=int(count[i]),success=int(won[i]),success_elapsed_s=float(success_time[i]-t0) if won[i] else None,
            energy_until_success_j=float(energy_success[i]) if won[i] else None,energy_end_j=float(ledger.energy[i])) for i in range(n)]
    return row,timeline,devices,wave_rows
