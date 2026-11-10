import csv,json
from pathlib import Path
import numpy as np
from units import internal_row
ROOT=Path(__file__).parent/'results'
def read(p):
    with p.open() as f:rows=list(csv.DictReader(f))
    return [{k.replace('_mj','_j'):str(float(v)/1000) if '_mj' in k and v else v for k,v in r.items()} for r in rows]
def main():
    report={}
    for folder in sorted(ROOT.iterdir()):
        if folder.name in ('failure_curves','repetition_elevation'):
            from elevation_study import validate_rows
            from replot import read as read_numeric
            rows=read_numeric(folder/'passes.csv');validate_rows(rows)
            report[folder.name]=dict(status='passed',pass_rows=len(rows));continue
        if folder.name=='threshold_wait':
            diag=read(folder/'summary.csv')
            assert len(diag)==63
            for scene in ('urban','village','rural_wooded'):
                group=sorted((r for r in diag if r['scenario']==scene),key=lambda r:float(r['threshold_db']))
                assert [float(r['threshold_db']) for r in group]==list(range(-5,16))
                assert np.all(np.diff([float(r['mean_capped_wait_s']) for r in group])>=-1e-12)
                assert all(0<=float(r['not_reached_fraction'])<=1 for r in group)
            report[folder.name]=dict(status='passed',summary_rows=63)
            continue
        if not (folder/'passes.csv').exists():continue
        rows=read(folder/'passes.csv')
        for r in rows:
            n=int(r['sensors']);s=int(r['successes']);a=int(r['attempts'])
            assert 0<=s<=n and s<=a
            assert abs(float(r['success_ratio'])-s/n)<1e-12
            if r['trigger_reached']=='0':assert r['energy_total_j']=='';continue
            assert sum(int(r[k+'_exclusive_count']) for k in ('preamble_collision','preamble_miss','payload_collision','payload_decode'))+s==a
            assert abs(float(r['energy_total_j'])-float(r['energy_unsuccessful_total_j'])-float(r['energy_successful_at_end_j']))<1e-6
            if s:assert abs(float(r['energy_per_success_j'])-float(r['energy_total_j'])/s)<1e-8
            else:assert r['energy_per_success_j']=='' and r['delay_s_mean']==''
        groups={}
        for r in read(folder/'timeline.csv') if (folder/'timeline.csv').exists() else []:groups.setdefault(tuple(r[k] for k in ('experiment','scenario','sensors','policy','pass_id')),[]).append(r)
        for group in groups.values():
            assert len(group)==201,'Incomplete timeline group'
            assert np.all(np.diff([float(r['energy_total_j']) for r in group])>=-1e-6)
            assert np.all(np.diff([float(r['completion_fraction']) for r in group])>=0)
            assert float(group[0]['energy_total_j'])==0
        if groups:assert len(groups)==sum(r['trigger_reached']=='1' for r in rows),'Missing timeline groups'
        for p in (folder/'details').glob('*_devices.csv'):
            d=read(p)
            for r in d:
                assert float(r['energy_end_j'])>=0
                if r['success']=='1':assert float(r['energy_until_success_j'])<=float(r['energy_end_j'])+1e-9
        report[folder.name]=dict(status='passed' if groups else 'summary_checks_passed_raw_timeline_missing',pass_rows=len(rows),timeline_groups=len(groups))
    assert report,'No results to verify'
    from figure_catalog import FIGURES,path_for
    expected={path_for(ROOT,n) for n in FIGURES}
    actual=set(ROOT.rglob('*.png'))
    assert expected==actual,dict(missing=[str(p) for p in expected-actual],unexpected=[str(p) for p in actual-expected])
    core=read(ROOT/'core'/'passes.csv')
    assert {int(r['sensors']) for r in core}==set(range(20,201,20))|{50}
    assert all(float(r['trigger_snr_db'])==0 and int(r['repetitions'])==16 and int(r['pool'])==48 for r in core)
    report['figure_catalog']=dict(status='passed',figures=len(expected),original_pages_retained=sorted(FIGURES))
    (ROOT/'verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
