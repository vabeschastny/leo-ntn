"""Core matrix or focused OFAT sweeps. Results include independent-pass statistics."""
import argparse,csv,json
from pathlib import Path
import numpy as np
from scipy.stats import t as student_t
from model import simulate
from units import normalize_config,external_config,external_row
ROOT=Path(__file__).parent

def write(path,rows):
    rows=[external_row(r) for r in rows]
    path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:return
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)

def cases(c,suite):
    # Focus PHY/power sweeps on a load with observable successes; the core,
    # mapping and repetition studies retain all four populations.
    if suite in ('mapping','trigger','phy','power') and not c.get('full_sweeps',False):c=c|{'sensor_counts':[c['sensors']]}
    if suite=='core':return [('baseline',c|{'sensor_counts':sorted(set(c['sensor_counts']+c['completion_sensor_counts']))})]
    if suite=='mapping':return [(f'mapping_{v}',c|{'preamble_to_pusch_ratio':v}) for v in [1,2,4]]
    if suite=='repetitions':return [(f'repetitions_{v}',c|{'sensor_counts':[c['sensors']],'repetitions':v,'trigger_reference_repetitions':c['repetitions'],'preamble_gap_ms':0}) for v in [1,16,64]]
    if suite=='trigger':
        from model import raw_snr
        from channel import slant_km
        return [(f'angle_{a}',c|{'access_trigger_snr_db':float(raw_snr(c,slant_km(a,c['altitude_km']))+10*np.log10(c['repetitions']))}) for a in range(10,61,5)]
    if suite=='phy':return [(f'payload_{v}',c|{'pusch_sinr_threshold_db':v}) for v in [-5,0,5]]+[(f'false_alarm_{v}',c|{'false_alarm_probability':v}) for v in [.0001,.001,.01]]
    if suite=='power':return [(f'tx_{v}',c|{'power_preamble_tx_w':v,'power_payload_tx_w':v}) for v in [.4,.7,1.1]]+[(f'rx_{v}',c|{'power_rx_w':v}) for v in [.08,.15,.25]]+[(f'sleep_{v}',c|{'power_sleep_w':v}) for v in [3e-6,1e-5,1e-4,1e-3]]
    raise ValueError(suite)

def summarize(rows):
    groups={}
    for r in rows:groups.setdefault((r['experiment'],r['scenario'],r['sensors'],r['policy']),[]).append(r)
    result=[]
    metadata={'scenario','sensors','policy','pass_id','trigger_snr_db','pool','mapping','repetitions','experiment'}
    for key,group in groups.items():
        row={k:group[0][k] for k in metadata if k!='pass_id'};row['passes']=len(group)
        for metric in dict.fromkeys(k for r in group for k in r if k not in metadata):
            vals=np.array([r[metric] for r in group if r.get(metric) is not None],float)
            row[metric+'_mean']=float(vals.mean()) if len(vals) else None
            row[metric+'_defined_passes']=len(vals)
            row[metric+'_ci95_halfwidth']=float(student_t.ppf(.975,len(vals)-1)*vals.std(ddof=1)/np.sqrt(len(vals))) if len(vals)>1 else None
        total_success=sum(r['successes'] for r in group)
        row['energy_per_success_pooled_j']=sum(r['energy_total_j'] for r in group)/total_success if total_success and all(r.get('energy_total_j') is not None for r in group) else None
        result.append(row)
    return result

def execute(c,suite,out):
    out.mkdir(parents=True,exist_ok=True);rows=[];traces=[];configs={}
    for name,trial in cases(c,suite):
        configs[name]=external_config(trial)
        for scene in trial['scenarios']:
            for n in trial['sensor_counts']:
                for retry in trial['retry_intervals_s']:
                    for rep in range(trial['passes']):
                        row,timeline,devices,waves=simulate(trial,scene,n,retry,rep,save_devices=(rep==0 and suite=='core'))
                        row['experiment']=name;rows.append(row)
                        for t in timeline:t['experiment']=name
                        traces.extend(timeline)
                        if rep==0:
                            stem=f'{name}_{scene}_{n}_{row["policy"]}'
                            write(out/'details'/(stem+'_waves.csv'),waves)
                            if devices:write(out/'details'/(stem+'_devices.csv'),devices)
                    print(suite,name,scene,n,'one_attempt' if retry is None else retry,flush=True)
        # Checkpoint after each experiment, to retain completed work.
        write(out/'passes.csv',rows);write(out/'summary.csv',summarize(rows));write(out/'timeline.csv',traces)
        (out/'configs_used.json').write_text(json.dumps(configs,indent=2))
    from plots import generate
    generate(rows,traces,out,suite)
    return rows

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,default=ROOT/'config.json');p.add_argument('--suite',choices=['core','mapping','repetitions','trigger','phy','power','all'],default='all');p.add_argument('--passes',type=int);p.add_argument('--output',type=Path,default=ROOT/'results')
    p.add_argument('--full-sweeps',action='store_true',help='Use all configured populations also for trigger/PHY/power studies')
    p.add_argument('--sweep-sensors',action='store_true',help='Compare configured sensor_counts instead of the default sensors value')
    p.add_argument('--sensors',type=int,help='Single population, from 20 to 200')
    a=p.parse_args();c=normalize_config(json.loads(a.config.read_text()));c['full_sweeps']=a.full_sweeps
    if a.sensors is not None:
        if not 20<=a.sensors<=200:p.error('--sensors must be in 20..200')
        c['sensors']=a.sensors
    if a.sensors is not None and not a.sweep_sensors:c['sensor_counts']=[c['sensors']]
    if a.passes is not None:c['passes']=a.passes
    suites=['core','mapping','repetitions','trigger','phy','power'] if a.suite=='all' else [a.suite]
    for suite in suites:execute(c,suite,a.output/suite)
    # Requested diagnostic map: 100 devices per environment at 60 degrees.
    # Separate from the default 50-device access population and collision metrics.
    from plot_channel_map import main as build_channel_map
    build_channel_map(['--sensors','100','--elevation','60','--seed',str(c['seed'])],config=c,output=a.output/'channel_map')
    from plot_threshold_wait import build as build_threshold_wait
    build_threshold_wait(config=c,output=a.output/'threshold_wait')
    from elevation_study import build as build_elevation_study
    build_elevation_study(c,a.output)
