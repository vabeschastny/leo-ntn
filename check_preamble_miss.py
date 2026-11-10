"""Independent detector audit: no collisions, fixed physical budget, angle 10 deg."""
import csv,json
from pathlib import Path
import numpy as np
from channel import Loo,slant_km
from model import raw_snr,Geometry,trigger
from detector import detect_preamble
from units import normalize_config
ROOT=Path(__file__).parent
def main():
    public=json.loads((ROOT/'config.json').read_text());c=normalize_config(public);n=100000;rows=[]
    for scene in c['scenarios']:
        gain,_=Loo().sample(scene,20.,np.random.default_rng(np.random.SeedSequence([c['seed'],('urban','village','rural_wooded').index(scene),555])),size=n)
        raw=float(raw_snr(c,slant_km(10,c['altitude_km'])))+gain
        for nr in (1,2,4,8,16,32,64):
            p,miss=detect_preamble(raw+10*np.log10(nr),np.random.default_rng(556),c['false_alarm_probability'])
            rows.append(dict(scenario=scene,repetitions=nr,elevation_deg=10,samples=n,mean_miss_probability=float(p.mean()),mc_standard_error=float(p.std(ddof=1)/np.sqrt(n)),observed_miss_fraction=float(miss.mean()),target_met=int(0<p.mean()<.2)))
    candidates=[nr for nr in (1,2,4,8,16,32,64) if all(r['target_met'] for r in rows if r['repetitions']==nr)]
    out=ROOT/'results'/'preamble_audit';out.mkdir(parents=True,exist_ok=True)
    with (out/'audit.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    report=dict(default_repetitions=c['repetitions'],access_trigger_snr_db=c['access_trigger_snr_db'],trigger_time_from_10deg_s=trigger(c,Geometry(c)),minimum_tested_repetitions_meeting_target_all_scenes=min(candidates) if candidates else None,definition='Unconditional channel-only preamble miss at the first visible elevation 10 deg; hold20 ITU parameters. No collisions or fitting.',baseline=[r for r in rows if r['repetitions']==c['repetitions']])
    (out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
    if candidates:
        alternative=public|{'repetitions':min(candidates),'preamble_gap_ms':0}
        (ROOT/'config_alternative_64.json').write_text(json.dumps(alternative,indent=2))
if __name__=='__main__':main()
