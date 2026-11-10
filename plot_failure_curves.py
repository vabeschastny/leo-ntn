"""Failure composition versus common access-trigger elevation, fixed N=50."""
import json
from pathlib import Path
import numpy as np
from scipy.stats import t as student_t
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from channel import slant_km
from model import simulate,raw_snr,CAUSES
from units import normalize_config
ROOT=Path(__file__).parent
LABELS={'preamble_collision':'Коллизия преамбулы','preamble_miss':'Пропуск преамбулы','payload_collision':'Коллизия нагрузки','payload_decode':'Ошибка декодирования'}

def validate_rows(rows):
    assert rows
    for r in rows:
        assert int(r['sensors'])==50
        assert abs(r['trigger_elevation_deg']-r['start_angle_deg'])<1e-6
        assert sum(r[k+'_exclusive_count'] for k in CAUSES)+r['successes']==r['attempts']

def render(rows,output):
    from plots import POLICIES,NAMES
    scenes=list(dict.fromkeys(r['scenario'] for r in rows));policies=list(dict.fromkeys(r['policy'] for r in rows))
    fig,axes=plt.subplots(len(policies),len(scenes),figsize=(6*len(scenes),3.6*len(policies)),squeeze=False,layout='constrained')
    for i,policy in enumerate(policies):
        for j,scene in enumerate(scenes):
            ax=axes[i,j];angles=sorted(set(r['start_angle_deg'] for r in rows))
            for key,label,color in [(k,LABELS[k],f'C{index}') for index,k in enumerate(CAUSES)]+[('overall','Общий отказ','black')]:
                means=[];errors=[]
                for angle in angles:
                    selected=[r for r in rows if r['scenario']==scene and r['policy']==policy and r['start_angle_deg']==angle and r['attempts']]
                    v=np.array([(r['attempts']-r['successes'] if key=='overall' else r[key+'_exclusive_count'])/r['attempts'] for r in selected])
                    means.append(float(v.mean()) if len(v) else np.nan)
                    errors.append(float(student_t.ppf(.975,len(v)-1)*v.std(ddof=1)/np.sqrt(len(v))) if len(v)>1 else 0.)
                y=np.array(means);err=np.array(errors)
                ax.plot(angles,y,marker='o',markersize=3,color=color,label=label,ls='--' if key=='overall' else '-')
                ax.fill_between(angles,np.maximum(0,y-err),np.minimum(1,y+err),color=color,alpha=.07)
            ax.set(title=NAMES[scene]+' — '+POLICIES.get(policy,policy),xlabel='Угол возвышения при начале доступа, °',ylabel='Доля выполненных попыток',xlim=(10,60),ylim=(0,1.02),xticks=np.arange(10,61,10))
            ax.legend(fontsize=6);ax.grid(alpha=.2)
    fig.suptitle('Причины отказов при разных углах запуска; 50 устройств на площади 1 км²',fontsize=14)
    fig.supxlabel('Средние по независимым пролётам; 95% t-интервалы. Четыре причины взаимоисключающие; общий отказ — их сумма.',fontsize=9)
    out=Path(output);out.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(out,dpi=150);fig.savefig(out.with_suffix('.svg'));plt.close(fig)

def render_cached(results_root):
    from replot import read
    root=Path(results_root);source=root/'failure_curves'/'passes.csv'
    if source.exists():render(read(source),root/'core'/'figures'/'01_failure_composition.png')

def build(config=None,results_root=None):
    from run import write
    c=config if config is not None else normalize_config(json.loads((ROOT/'config.json').read_text()))
    root=Path(results_root) if results_root is not None else ROOT/'results';out=root/'failure_curves';out.mkdir(parents=True,exist_ok=True)
    angles=list(range(10,61,5));rows=[]
    for angle in angles:
        threshold=float(raw_snr(c,slant_km(angle,c['altitude_km']))+10*np.log10(c.get('trigger_reference_repetitions',c['repetitions'])))
        trial=c|{'access_trigger_snr_db':threshold,'sensor_counts':[50],'sensors':50}
        for scene in c['scenarios']:
            for retry in c['retry_intervals_s']:
                for rep in range(c['passes']):
                    row,_,_,_=simulate(trial,scene,50,retry,rep)
                    row['start_angle_deg']=angle;rows.append(row)
        print('Failure curves: start angle',angle,flush=True)
    validate_rows(rows);write(out/'passes.csv',rows)
    (out/'settings.json').write_text(json.dumps(dict(config=c,sensors=50,area_km2=1,start_angles_deg=angles,definition='Exclusive failure causes / performed attempts over remaining visibility window, mean across passes'),indent=2))
    render(rows,root/'core'/'figures'/'01_failure_composition.png')
    (out/'verification.json').write_text(json.dumps(dict(status='passed',rows=len(rows),checks=['population 50','actual trigger matches angle','exclusive failure balance']),indent=2))
    return rows

if __name__=='__main__':build()
