"""Individual quality-gate diagnostic, distinct from common-trigger RA."""
import csv,json
from pathlib import Path
import numpy as np
from scipy.stats import t as student_t
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from channel import Loo
from model import Geometry,raw_snr,durations
from units import normalize_config
ROOT=Path(__file__).parent

def capped_wait(snr,times,threshold):
    eligible=snr>=threshold;reached=eligible.any(axis=0);first=eligible.argmax(axis=0)
    return np.where(reached,times[first],times[-1]),reached

def build(config=None,output=None):
    c=config if config is not None else normalize_config(json.loads((ROOT/'config.json').read_text()))
    n=c['sensors'];g=Geometry(c);_,_,duration=durations(c)
    times=c['ra_offset_ms']/1000+np.arange(int(np.ceil(g.end/(c['ra_period_ms']/1000))))*c['ra_period_ms']/1000
    _,d=g.at(times)
    times=times[times+duration+d/299792.458+c['receiver_processing_delay_ms']/1000<=g.end]
    theta,d=g.at(times);thresholds=np.arange(-5,16,dtype=float);rows=[];summary=[]
    names={'urban':'Город','village':'Деревня','rural_wooded':'Сельская местность с растительностью'}
    fig,ax=plt.subplots(figsize=(9,5.4),layout='constrained');loo=Loo()
    for si,(scene,name) in enumerate(names.items()):
        means=[];errors=[]
        for rep in range(c['passes']):
            rng=np.random.default_rng(np.random.SeedSequence([c['seed'],si,rep,909]))
            gain,_=loo.sample(scene,np.repeat(np.clip(theta,20,70),n),rng)
            snr=(np.repeat(raw_snr(c,d),n)+gain+10*np.log10(c['repetitions'])).reshape(len(times),n)
            previous=np.zeros(n)
            for threshold in thresholds:
                waits,reached=capped_wait(snr,times,threshold)
                assert np.all(waits>=previous);previous=waits
                rows.append(dict(scenario=scene,pass_id=rep,sensors=n,threshold_db=float(threshold),mean_capped_wait_s=float(waits.mean()),not_reached=int((~reached).sum())))
        for threshold in thresholds:
            a=[r for r in rows if r['scenario']==scene and r['threshold_db']==threshold]
            values=np.array([r['mean_capped_wait_s'] for r in a]);mean=float(values.mean())
            err=float(student_t.ppf(.975,len(a)-1)*values.std(ddof=1)/np.sqrt(len(a))) if len(a)>1 else 0.
            means.append(mean);errors.append(err)
            summary.append(dict(scenario=scene,threshold_db=float(threshold),sensors=n,passes=c['passes'],mean_capped_wait_s=mean,ci95_halfwidth_s=err,not_reached_fraction=sum(r['not_reached'] for r in a)/(len(a)*n)))
        line,=ax.plot(thresholds,means,marker='o',markersize=3,label=name)
        ax.fill_between(thresholds,np.maximum(0,np.array(means)-errors),np.array(means)+errors,color=line.get_color(),alpha=.13)
    ax.set(xlim=(-5,15),ylim=(0,None),xticks=np.arange(-5,16,5),xlabel='Порог эффективного SNR, дБ',ylabel='Среднее min(T, горизонт), с',title=f'Ожидание индивидуального порога от восходящих 10°; N={n}')
    ax.legend();ax.grid(alpha=.25);ax.ticklabel_format(axis='y',style='plain',useOffset=False)
    fig.supxlabel(f'{c["passes"]} независимых пролётов; полосы — 95% t-интервалы; не достигшие включены',fontsize=9)
    out=Path(output) if output is not None else ROOT/'results'/'threshold_wait';out.mkdir(parents=True,exist_ok=True)
    for ext in ('png','svg'):fig.savefig(out/f'03_capped_wait.{ext}',dpi=180)
    plt.close(fig)
    for filename,data in [('passes.csv',rows),('summary.csv',summary)]:
        with (out/filename).open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    (out/'settings.json').write_text(json.dumps(dict(config=c,thresholds_db=thresholds.tolist(),censor_time_s=float(times[-1]),definition='Individual first threshold crossing; no common trigger, retries, collisions or decoding in this diagnostic'),indent=2))
    print('Threshold wait: 3 curves, -5..15 dB,',len(summary),'summary rows')
    return summary

if __name__=='__main__':build()
