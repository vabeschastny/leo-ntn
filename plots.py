"""Only plots retained by the supervisor; old PDF page numbers in catalog."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import t as student_t
NAMES={'urban':'Город','village':'Деревня','rural_wooded':'Сельская местность с растительностью'}
POLICIES={'one_attempt':'Одна попытка','retry_5s':'Повтор через 5 с','retry_uniform_1_5s':'Повтор: U[1; 5] с'}
METRICS={'delay_s_mean':'Средняя задержка до успеха, с','energy_per_success_j':'Энергия на успешный доступ, мДж/доступ','payload_collision_probability':'Коллизия нагрузки на попытку','preamble_collision_probability':'Коллизия преамбулы на попытку','success_ratio':'Доля уникальных успешных устройств'}

def estimate(rows,key):
    v=np.array([r[key] for r in rows if r.get(key) is not None],float)
    return (float(v.mean()),float(student_t.ppf(.975,len(v)-1)*v.std(ddof=1)/np.sqrt(len(v))) if len(v)>1 else 0.) if len(v) else (np.nan,0.)

def pooled_energy(rows):
    rows=[r for r in rows if r.get('energy_total_j') is not None]
    s=np.array([r['successes'] for r in rows]);e=np.array([r['energy_total_j'] for r in rows])
    if not len(s) or not s.sum():return np.nan,(0.,0.),False
    y=e.sum()/s.sum();idx=np.random.default_rng(917).integers(len(s),size=(2000,len(s)))
    den=s[idx].sum(axis=1);ratios=np.full(len(idx),np.inf)
    np.divide(e[idx].sum(axis=1),den,out=ratios,where=den>0)
    q=np.sort(ratios)[[int(.025*len(idx)),int(.975*len(idx))]];valid=np.isfinite(q).all()
    return y,(max(0,y-q[0]),max(0,q[1]-y)) if valid else (0.,0.),bool(valid)

def axes(scenes,rows=1):
    fig,aa=plt.subplots(rows,len(scenes),figsize=(6*len(scenes),4*rows),squeeze=False,layout='constrained')
    for ax,s in zip(aa[0],scenes):ax.set_title(NAMES[s])
    return fig,aa

def finish(fig,aa,out,name,title,ylabel,xlabel):
    for ax in aa.flat:
        ax.set_ylabel(ylabel);ax.set_xlabel(xlabel);ax.grid(alpha=.2)
        handles,_=ax.get_legend_handles_labels()
        if handles:ax.legend(fontsize=7)
        if not any(np.isfinite(np.asarray(line.get_ydata(),float)).any() for line in ax.lines):ax.text(.5,.5,'N/A: успешных доступов нет',transform=ax.transAxes,ha='center')
    fig.suptitle(title);out.mkdir(parents=True,exist_ok=True);fig.savefig(out/(name+'.png'),dpi=150);plt.close(fig)

def interval_line(ax,x,selected,metric,label,color=None):
    if metric=='energy_per_success_j':
        points=[pooled_energy(v) for v in selected];y=[p[0] for p in points];err=np.array([p[1] for p in points]).T
        if any(np.isfinite(p[0]) and not p[2] for p in points):ax.text(.02,.97,'Часть CI неограничена; показаны оценки',transform=ax.transAxes,va='top',fontsize=6)
    else:
        points=[estimate(v,metric) for v in selected];y=[p[0] for p in points];err=[p[1] for p in points]
    ax.errorbar(x,y,yerr=err,marker='o',capsize=3,label=label,color=color)

def generate(rows,traces,out,suite):
    rows=[{k:(v*1000 if '_j' in k and v is not None else v) for k,v in r.items()} for r in rows]
    folder=out/'figures';scenes=list(dict.fromkeys(r['scenario'] for r in rows));policies=list(dict.fromkeys(r['policy'] for r in rows))
    if suite!='core':
        allowed={'mapping':['delay_s_mean','energy_per_success_j','payload_collision_probability','success_ratio'],
            'repetitions':['delay_s_mean','energy_per_success_j'],'trigger':['delay_s_mean','energy_per_success_j'],
            'phy':['delay_s_mean','energy_per_success_j','payload_collision_probability','preamble_collision_probability'],
            'power':['energy_per_success_j']}
        for metric in allowed[suite]:
            fig,aa=axes(scenes)
            for ax,scene in zip(aa[0],scenes):
                for policy in policies:
                    groups=[]
                    for exp in dict.fromkeys(r['experiment'] for r in rows):
                        v=[r for r in rows if r['experiment']==exp and r['scenario']==scene and r['policy']==policy and r['sensors']==50]
                        if not v:continue
                        x=v[0]['mapping'] if suite=='mapping' else v[0]['repetitions'] if suite=='repetitions' else v[0].get('trigger_elevation_deg') if suite=='trigger' else exp
                        if x is not None:groups.append((x,v))
                    if suite in ('mapping','repetitions','trigger'):groups.sort(key=lambda a:a[0])
                    interval_line(ax,[g[0] for g in groups],[g[1] for g in groups],metric,POLICIES[policy])
                if suite in ('phy','power'):ax.tick_params(axis='x',rotation=35)
            xlabel={'mapping':'Преамбул на ресурс нагрузки','repetitions':'Повторения преамбулы','trigger':'Угол возвышения при общем запуске, °'}.get(suite,'Однофакторный эксперимент')
            finish(fig,aa,folder,metric,'N=50; средние по пролётам; 95% интервалы',METRICS[metric],xlabel)
        if suite=='trigger':
            fig,aa=axes(scenes,len(policies))
            for i,policy in enumerate(policies):
                for j,scene in enumerate(scenes):
                    ax=aa[i,j];ax.set_title(NAMES[scene]+' — '+POLICIES[policy])
                    for key,label in [('preamble_collision_probability','Коллизия преамбулы'),('preamble_miss_probability','Пропуск преамбулы'),('payload_collision_probability','Коллизия нагрузки'),('payload_decode_probability','Ошибка SINR'),('overall_failure_probability','Общий отказ')]:
                        groups=[]
                        for exp in dict.fromkeys(r['experiment'] for r in rows):
                            v=[r for r in rows if r['experiment']==exp and r['scenario']==scene and r['policy']==policy and r['sensors']==50]
                            if v and v[0].get('trigger_elevation_deg') is not None:groups.append((v[0]['trigger_elevation_deg'],v))
                        groups.sort(key=lambda g:g[0]);interval_line(ax,[g[0] for g in groups],[g[1] for g in groups],key,label)
                    ax.set_ylim(0,1.02)
            finish(fig,aa,folder,'01_failure_vs_trigger_elevation','Маргинальные события и их объединение; N=50; события пересекаются','Вероятность на выполненную попытку','Угол возвышения при общем запуске, °')
        if suite=='repetitions':
            source=out.parent/'repetition_elevation'/'passes.csv'
            if source.exists():
                from replot import read
                from elevation_study import render_repetitions
                render_repetitions(read(source),out.parent)
        return
    # Supervisor pages 2, 8, 12, 13, 18: exact grid 20,40,...,200; 50 only for paired completion.
    ns=sorted(n for n in set(r['sensors'] for r in rows) if n%20==0)
    if not ns:ns=sorted(set(r['sensors'] for r in rows))
    for name,metric,label in [('02_access_success','success_ratio','Доля уникальных успешных устройств'),('04_delay_mean','delay_s_mean','Средняя задержка до успеха, с'),('07_energy_per_success','energy_per_success_j','Энергия на успешный доступ, мДж/доступ'),('08_energy_until_success_mean','energy_until_success_j_mean','Средняя энергия до успеха, мДж/устройство'),('energy_unsuccessful','energy_unsuccessful_total_j','Энергия неуспешных устройств, мДж')]:
        fig,aa=axes(scenes)
        for ax,scene in zip(aa[0],scenes):
            for policy in policies:
                selected=[[r for r in rows if r['scenario']==scene and r['policy']==policy and r['sensors']==n] for n in ns]
                interval_line(ax,ns,selected,metric,POLICIES[policy])
            ax.set_xticks(ns);ax.tick_params(axis='x',labelsize=8)
        rep_text='/'.join(f'{v:g}' for v in sorted(set(r['repetitions'] for r in rows)))
        pool_text='/'.join(f'{v:g}' for v in sorted(set(r['pool'] for r in rows)))
        threshold_text='/'.join(f'{v:g}' for v in sorted(set(r['trigger_snr_db'] for r in rows)))
        finish(fig,aa,folder,name,f'Порог доступа {threshold_text} дБ; повторения {rep_text}; пул {pool_text}',label,'Число ожидающих устройств')
    # Exactly three pairs: one colour per retry policy, line style per population.
    fig,aa=axes(scenes)
    for ax,scene in zip(aa[0],scenes):
        for pi,policy in enumerate(policies):
            for n,style in [(50,'--'),(200,'-')]:
                v=[r for r in traces if r['scenario']==scene and r['policy']==policy and r['sensors']==n]
                ids=sorted(set(r['pass_id'] for r in v));groups=[[r for r in v if r['pass_id']==i] for i in ids]
                if not groups:continue
                x=[r['elapsed_s'] for r in groups[0]];y=np.array([[r['completion_fraction'] for r in g] for g in groups]);mean=y.mean(axis=0)
                err=student_t.ppf(.975,len(ids)-1)*y.std(axis=0,ddof=1)/np.sqrt(len(ids)) if len(ids)>1 else np.zeros(len(x))
                ax.plot(x,mean,color=f'C{pi}',ls=style,label=POLICIES[policy]+f'; N={n}')
                ax.fill_between(x,np.maximum(0,mean-err),np.minimum(1,mean+err),color=f'C{pi}',alpha=.06)
        ax.set_ylim(0,1.02)
    finish(fig,aa,folder,'03_completion_combined','Накопленный успех: три пары кривых для 50 и 200 устройств','Доля всех устройств с успешным доступом','Время от общего t₀, с')
    from plot_failure_curves import render_cached
    render_cached(out.parent)
