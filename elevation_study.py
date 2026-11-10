"""Pages 1 and 32: fixed N=50, common launch angle, matched repetition comparison."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from model import simulate,raw_snr
from channel import slant_km
from units import normalize_config
ROOT=Path(__file__).parent

def validate_rows(rows):
    assert rows
    for r in rows:
        assert r['sensors']==50
        assert abs(r['trigger_elevation_deg']-r['start_angle_deg'])<1e-6
        assert sum(r[k+'_exclusive_count'] for k in ('preamble_collision','preamble_miss','payload_collision','payload_decode'))+r['successes']==r['attempts']

def render_repetitions(rows,root):
    from plots import axes,finish,estimate,NAMES,POLICIES
    scenes=list(dict.fromkeys(r['scenario'] for r in rows));policies=list(dict.fromkeys(r['policy'] for r in rows))
    fig,aa=axes(scenes,len(policies))
    for i,policy in enumerate(policies):
        for j,scene in enumerate(scenes):
            ax=aa[i,j];ax.set_title(NAMES[scene]+' — '+POLICIES[policy])
            for nr in (1,16,64):
                angles=sorted(set(r['start_angle_deg'] for r in rows));values=[]
                for angle in angles:values.append(estimate([r for r in rows if r['scenario']==scene and r['policy']==policy and r['repetitions']==nr and r['start_angle_deg']==angle],'success_ratio'))
                y=np.array([v[0] for v in values]);err=np.array([v[1] for v in values])
                style,marker,markevery={1:('-','o',(0,3)),16:('--','s',(1,3)),64:(':','^',(2,3))}[nr]
                line,=ax.plot(angles,y,ls=style,marker=marker,markevery=markevery,label=f'{nr} повторений')
                ax.fill_between(angles,np.maximum(0,y-err),np.minimum(1,y+err),color=line.get_color(),alpha=.08)
            ax.set_ylim(0,1.02)
    finish(fig,aa,Path(root)/'repetitions'/'figures','success_ratio','N=50; три кривые повторений; одинаковый угол общего запуска','Доля уникальных успешных устройств','Угол возвышения при общем запуске, °')

def build(c=None,root=None):
    from run import write
    from replot import read
    from plot_failure_curves import render
    c=c if c is not None else normalize_config(json.loads((ROOT/'config.json').read_text()))
    root=Path(root) if root is not None else ROOT/'results';out=root/'repetition_elevation';out.mkdir(parents=True,exist_ok=True)
    angles=list(range(10,61,5));rows=[]
    # Reuse only a fully matching completed baseline elevation series.
    cache=root/'trigger'/'passes.csv';cached=[]
    if cache.exists():
        candidate=read(cache);expected=len(angles)*len(c['scenarios'])*len(c['retry_intervals_s'])*c['passes']
        configs=json.loads((cache.parent/'configs_used.json').read_text());first=next(iter(configs.values()))
        if len(candidate)==expected and all(first[k]==c[k] for k in ('seed','repetitions','preamble_pool_size','uplink_tx_eirp_dbm','satellite_rx_gain_dbi','frequency_hz','system_loss_db','passes')):
            cached=candidate
    for nr in (1,16,64):
        if nr==c['repetitions'] and cached:
            for r in cached:
                r=dict(r);r['start_angle_deg']=int(round(r['trigger_elevation_deg']));rows.append(r)
            print('Elevation study: reused baseline trigger series',flush=True);continue
        for angle in angles:
            threshold=float(raw_snr(c,slant_km(angle,c['altitude_km']))+10*np.log10(c['repetitions']))
            trial=c|{'repetitions':nr,'preamble_gap_ms':0,'trigger_reference_repetitions':c['repetitions'],'access_trigger_snr_db':threshold,'sensor_counts':[50]}
            for scene in c['scenarios']:
                for retry in c['retry_intervals_s']:
                    for rep in range(c['passes']):
                        r,_,_,_=simulate(trial,scene,50,retry,rep);r['start_angle_deg']=angle;rows.append(r)
            print('Elevation study:',nr,'repetitions,',angle,'degrees',flush=True)
    validate_rows(rows);write(out/'passes.csv',rows)
    settings=dict(config=c,sensors=50,start_angles_deg=angles,repetition_values=[1,16,64],trigger_reference_repetitions=c['repetitions'])
    (out/'settings.json').write_text(json.dumps(settings,indent=2))
    base=[r for r in rows if r['repetitions']==c['repetitions']]
    failure=root/'failure_curves';failure.mkdir(exist_ok=True)
    write(failure/'passes.csv',base);(failure/'settings.json').write_text(json.dumps(settings,indent=2))
    render(base,root/'core'/'figures'/'01_failure_composition.png');render_repetitions(rows,root)
    return rows

if __name__=='__main__':build()
