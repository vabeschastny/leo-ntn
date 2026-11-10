"""Diagnostic data-block BLER map, NOT access failure or NPRACH missed detection.

Receiver curve retained from the earlier supplied data-block CSV. Loo channel
is sampled independently of decorative buildings and sensor coordinates.
"""
import argparse,csv,json,random
from pathlib import Path
import numpy as np
from scipy.interpolate import PchipInterpolator
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from channel import Loo,slant_km
from model import raw_snr
from units import normalize_config
from map_layout import buildings,inside
ROOT=Path(__file__).parent

def main(argv=None,config=None,output=None):
    p=argparse.ArgumentParser();p.add_argument('--sensors',type=int,default=100);p.add_argument('--elevation',type=float,default=60);p.add_argument('--seed',type=int,default=42)
    a=p.parse_args(argv)
    if a.sensors<1 or not 20<=a.elevation<=70:p.error('Positive sensors and elevation in 20..70 required')
    c=config if config is not None else normalize_config(json.loads((ROOT/'config.json').read_text()))
    with (ROOT/'data/block_decoder_reference.csv').open() as f:source=list(csv.DictReader(f))
    grouped={}
    for r in source:grouped.setdefault(float(r['SNR_dB']),[]).append(float(r['BLER']))
    x=np.array(sorted(grouped));y=np.array([np.mean(grouped[v]) for v in x]);decoder=PchipInterpolator(x,y,extrapolate=False)
    fig,axes=plt.subplots(1,3,figsize=(15,5.4),layout='constrained');rows=[];summary=[]
    for si,(ax,scene) in enumerate(zip(axes,('urban','village','rural_wooded'))):
        layout=random.Random(a.seed+si);obs=buildings(layout,scene,1000);veg=[]
        if scene=='rural_wooded':
            for _ in range(12):
                px=layout.uniform(0,850);py=layout.uniform(0,850);veg.append((px,py,px+layout.uniform(50,140),py+layout.uniform(50,140)))
        points=[]
        while len(points)<a.sensors:
            px,py=layout.uniform(0,1000),layout.uniform(0,1000)
            if not any(inside(px,py,b) for b in obs):points.append((px,py))
        rng=np.random.default_rng(np.random.SeedSequence([a.seed,si,55]))
        gain,_=Loo().sample(scene,a.elevation,rng,size=a.sensors)
        snr=float(raw_snr(c,slant_km(a.elevation,c['altitude_km'])))+gain+10*np.log10(c['repetitions'])
        bler=np.clip(decoder(np.clip(snr,x[0],x[-1])),0,1)
        for px,py,x2,y2,*_ in obs:ax.add_patch(Rectangle((px,py),x2-px,y2-py,color='#64748b',alpha=.35,lw=0))
        for px,py,x2,y2 in veg:ax.add_patch(Rectangle((px,py),x2-px,y2-py,color='#86b88b',alpha=.35,lw=0))
        xy=np.array(points);scatter=ax.scatter(xy[:,0],xy[:,1],c=bler,cmap='RdYlGn_r',vmin=0,vmax=1,s=27,edgecolors='white',linewidths=.3)
        ax.set(title=f'{scene}\nN={a.sensors}; mean BLER={bler.mean():.3f}',xlabel='East (m)',ylabel='North (m)',xlim=(0,1000),ylim=(0,1000),aspect='equal')
        for i in range(a.sensors):rows.append(dict(scenario=scene,sensor_id=i+1,x_m=points[i][0],y_m=points[i][1],gain_db=float(gain[i]),effective_snr_db=float(snr[i]),block_bler=float(bler[i])))
        summary.append(dict(scenario=scene,sensors=a.sensors,mean_bler=float(bler.mean()),fraction_bler_over_0_8=float(np.mean(bler>.8))))
    fig.colorbar(scatter,ax=list(axes),label='Conditional data-block BLER (no collisions)',shrink=.72)
    fig.suptitle(f'Individual channels: {a.elevation:g}° elevation; {c["repetitions"]} repetitions; UE EIRP {c["uplink_tx_eirp_dbm"]:g} dBm',fontsize=12)
    fig.supxlabel('ITU random shadowing; buildings/vegetation illustrative only; data-block receiver curve, not NPRACH detection',fontsize=9)
    out=Path(output) if output is not None else ROOT/'results'/'channel_map';out.mkdir(parents=True,exist_ok=True)
    fig.savefig(out/'individual_channels_100.png',dpi=200);fig.savefig(out/'individual_channels_100.svg');plt.close(fig)
    with (out/'sensors.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    report=dict(elevation_deg=a.elevation,repetitions=c['repetitions'],seed=a.seed,decoder='Empirical 408-bit data-block curve from earlier supplied CSV; not used by current RA engine',settings=c,summary=summary)
    (out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
