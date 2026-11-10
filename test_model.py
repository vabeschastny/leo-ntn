import json,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from model import *
from units import normalize_config,external_row,internal_row
C=normalize_config(json.loads((Path(__file__).parent/'config.json').read_text()))
class Tests(unittest.TestCase):
    def test_supervisor_figure_catalog(self):
        from figure_catalog import FIGURES,REMOVED,MERGED
        self.assertEqual(len(FIGURES),24)
        self.assertEqual(set(FIGURES)|REMOVED|set(MERGED),set(range(1,52)))
        self.assertFalse(set(FIGURES)&REMOVED)
        self.assertTrue({9,10,11,14,15,16,17,22,25,28,30,31,35,41,45,46,47,49,50,51}.issubset(REMOVED))
    def test_requested_moderate_defaults(self):
        self.assertEqual(C['access_trigger_snr_db'],0);self.assertEqual(C['repetitions'],16);self.assertEqual(C['preamble_pool_size'],48)
        self.assertEqual(C['completion_sensor_counts'],[50,200]);self.assertEqual(trigger(C,Geometry(C)),0)
    def test_exact_three_completion_pairs(self):
        from plots import generate
        import tempfile
        rows=[];traces=[];capture={}
        for n in (50,200):
            for retry in (None,5,'uniform_1_5'):
                r,t,_,_=simulate(C,'urban',n,retry,0);rows.append(r);traces.extend(t)
        def record(fig,aa,out,name,*rest):
            if name=='03_completion_combined':capture['labels']=[line.get_label() for line in aa[0,0].lines]
            import matplotlib.pyplot as plt
            plt.close(fig)
        with tempfile.TemporaryDirectory() as folder,patch('plots.finish',side_effect=record):generate(rows,traces,Path(folder),'core')
        self.assertEqual(len(capture['labels']),6)
        self.assertEqual(sum('N=50' in s for s in capture['labels']),3)
        self.assertEqual(sum('N=200' in s for s in capture['labels']),3)
    def test_failure_curve_trigger_angles(self):
        from plot_failure_curves import validate_rows
        from channel import slant_km
        for angle in (10,35,60):
            threshold=float(raw_snr(C,slant_km(angle,C['altitude_km']))+10*np.log10(C['repetitions']))
            trial=C|{'access_trigger_snr_db':threshold}
            row=simulate(trial,'urban',50,None,0)[0];row['start_angle_deg']=angle
            validate_rows([row])
    def test_failure_curve_population_guard(self):
        from plot_failure_curves import validate_rows
        row=simulate(C,'urban',20,None,0)[0];row['start_angle_deg']=row['trigger_elevation_deg']
        with self.assertRaises(AssertionError):validate_rows([row])
    def test_no_cumulative_energy_plot(self):
        import tempfile
        from plots import generate
        r,trace,_,_=simulate(C,'urban',20,None,0)
        with tempfile.TemporaryDirectory() as folder,patch('matplotlib.figure.Figure.savefig') as save:
            generate([r],trace,Path(folder),'core')
        names=[str(call.args[0]) for call in save.call_args_list]
        self.assertFalse(any('09_cumulative_energy' in name for name in names))
        self.assertTrue(any('03_completion' in name for name in names))
        self.assertTrue(any('07_energy_per_success' in name for name in names))
    def test_retry_uniform_bounds_and_reproducibility(self):
        a=retry_delays('uniform_1_5',10000,42,0,1,2)
        self.assertTrue(np.all((a>=1)&(a<=5)));self.assertAlmostEqual(a.mean(),3,delta=.05)
        np.testing.assert_array_equal(a,retry_delays('uniform_1_5',10000,42,0,1,2))
        self.assertFalse(np.array_equal(a,retry_delays('uniform_1_5',10000,42,0,1,3)))
    def test_fixed_five_retry_timing(self):
        _,_,_,waves=simulate(C,'urban',50,5,0)
        for a,b in zip(waves,waves[1:]):self.assertGreaterEqual(b['start_s']+1e-9,a['outcome_s']+5)
    def test_uniform_simulation_balance(self):
        r,t,d,w=simulate(C,'urban',50,'uniform_1_5',0,True)
        self.assertEqual(r['policy'],'retry_uniform_1_5s');self.assertEqual(sum(x['success'] for x in d),r['successes'])
        self.assertEqual(sum(x['attempt_count'] for x in d),r['attempts'])
    def test_capped_threshold_wait(self):
        from plot_threshold_wait import capped_wait
        snr=np.array([[-10,0,10],[-5,5,15],[0,10,20]])
        waits,reached=capped_wait(snr,np.array([.008,2.568,5.128]),5)
        np.testing.assert_allclose(waits,[5.128,2.568,.008]);np.testing.assert_array_equal(reached,[False,True,True])
    def test_milli_units(self):
        self.assertEqual(normalize_config({'power_idle_mw':20}),{'power_idle_w':.02})
        r={'energy_total_j':1.25,'energy_per_success_j':None,'sensors':50}
        self.assertEqual(external_row(r)['energy_total_mj'],1250)
        self.assertEqual(internal_row(external_row(r)),r)
        self.assertEqual(C['sensors'],50);self.assertEqual(C['sensor_counts'],list(range(20,201,20)))
    def test_linear_interference_sum(self):
        _,_,_,sinr=radio_outcomes(np.array([10.,20.]),np.array([0,1]),np.zeros(2),np.zeros(2),C)
        np.testing.assert_allclose(sinr,10*np.log10(np.array([10/101,100/11])))
    def test_pooled_energy_includes_zero_success_pass(self):
        from plots import pooled_energy
        y,_,_=pooled_energy([{'energy_total_j':10,'successes':1},{'energy_total_j':90,'successes':0}])
        self.assertEqual(y,100)
    def test_tx_sensitivity_does_not_change_outcomes(self):
        a=simulate(C,'urban',10,None,0)[0];b=simulate(C|{'power_preamble_tx_w':1.1,'power_payload_tx_w':1.1},'urban',10,None,0)[0]
        self.assertEqual(a['successes'],b['successes']);pre,payload,_=durations(C)
        self.assertAlmostEqual(b['energy_total_j']-a['energy_total_j'],10*.4*(pre+payload))
    def test_scalar_parameters_match_vector_sampler(self):
        for scene in SCENES:
            loo=Loo();a=loo.sample(scene,np.full(100,35.),np.random.default_rng(5))[0]
            b=loo.sample(scene,35.,np.random.default_rng(5),size=100)[0]
            np.testing.assert_array_equal(a,b)
    def test_trigger(self):
        trial=C|{'access_trigger_snr_db':25};g=Geometry(trial);t=trigger(trial,g)
        self.assertGreater(t,0);self.assertLess(t,g.peak)
        self.assertAlmostEqual(float(raw_snr(trial,g.at(t)[1])+10*np.log10(trial['repetitions'])),25)
    def test_trigger_unreachable(self):self.assertIsNone(trigger(C|{'access_trigger_snr_db':1000},Geometry(C)))
    def test_trigger_at_window_start(self):self.assertEqual(trigger(C|{'access_trigger_snr_db':-5},Geometry(C)),0)
    def test_state_energy(self):
        l=EnergyLedger(2,0,dict(sleep=.001,tx=.7,idle=.02));l.transition([0],1,'tx');l.transition([0],3,'idle');l.transition([0],4,'sleep');l.close(5)
        np.testing.assert_allclose(l.energy,[.001+1.4+.02+.001,.005]);self.assertAlmostEqual(l.cumulative(np.array([5]))[0],l.energy.sum())
    def test_partial_interval_integral(self):
        l=EnergyLedger(1,0,dict(sleep=.01,tx=1));l.transition([0],2,'tx');l.transition([0],4,'sleep');l.close(5)
        np.testing.assert_allclose(l.cumulative(np.array([0,1,2,3,4,5])),[0,.01,.02,1.02,2.02,2.03])
    def test_backwards_rejected(self):
        l=EnergyLedger(1,0,dict(sleep=.01));l.transition([0],1,'sleep')
        with self.assertRaises(ValueError):l.transition([0],0,'sleep')
    def test_collision_same_preamble(self):
        ok,_,e,_=radio_outcomes(np.array([60.,60.]),np.array([0,0]),np.zeros(2),np.zeros(2),C)
        self.assertFalse(ok.any());self.assertTrue(e['preamble_collision'].all())
    def test_payload_only_collision(self):
        ok,_,e,_=radio_outcomes(np.array([60.,60.]),np.array([0,1]),np.zeros(2),np.zeros(2),C)
        self.assertFalse(e['preamble_collision'].any());self.assertTrue(e['payload_collision'].all());self.assertFalse(ok.any())
    def test_separate_resources(self):
        ok,_,_,sinr=radio_outcomes(np.array([20.,20.]),np.array([0,2]),np.zeros(2),np.zeros(2),C)
        self.assertTrue(ok.all());np.testing.assert_allclose(sinr,20)
    def test_no_capture(self):
        ok,_,_,_=radio_outcomes(np.array([80.,-30.]),np.array([0,1]),np.zeros(2),np.zeros(2),C);self.assertFalse(ok.any())
    def test_repetition_not_on_payload(self):
        args=(np.array([2.]),np.array([0]),np.zeros(1),np.zeros(1))
        np.testing.assert_allclose(radio_outcomes(*args,C)[3],radio_outcomes(*args,C|{'repetitions':1})[3])
    def test_invalid_mapping(self):
        with self.assertRaises(ValueError):validate(C|{'preamble_pool_size':54,'preamble_to_pusch_ratio':4})
    def test_no_trigger_no_energy_window(self):
        r,_,_,_=simulate(C|{'access_trigger_snr_db':1000},'urban',10,None,0)
        self.assertIsNone(r['energy_per_success_j']);self.assertIsNone(r['energy_total_j'])
    def test_single_attempt_full_energy(self):
        r,t,d,w=simulate(C,'urban',10,None,0,True)
        self.assertEqual(r['attempts'],10);self.assertEqual(len(w),1)
        pre,payload,total=durations(C);g=Geometry(C);prop=float(g.at(w[0]['start_s'])[1])/299792.458
        expected=10*(C['power_sleep_w']*r['window_s']+(C['power_preamble_tx_w']-C['power_sleep_w'])*pre+(C['power_payload_tx_w']-C['power_sleep_w'])*payload+(C['power_idle_w']-C['power_sleep_w'])*(C['payload_offset_ms']/1000+prop))
        self.assertAlmostEqual(r['energy_total_j'],expected,places=8)
        self.assertAlmostEqual(t[-1]['energy_total_j'],r['energy_total_j'])
    def test_retry_timing_and_unique_success(self):
        r,t,d,w=simulate(C,'urban',100,2,1,True)
        for left,right in zip(w,w[1:]):self.assertGreaterEqual(right['start_s']+1e-9,left['outcome_s']+2)
        self.assertEqual(sum(x['success'] for x in d),r['successes']);self.assertEqual(sum(x['attempt_count'] for x in d),r['attempts'])
        self.assertTrue(np.all(np.diff([x['energy_total_j'] for x in t])>=-1e-8));self.assertAlmostEqual(t[-1]['completion_fraction'],r['success_ratio'])
    def test_pairing(self):
        _,_,_,w1=simulate(C,'village',100,None,2);_,_,_,w2=simulate(C,'village',100,2,2)
        self.assertEqual({k:v for k,v in w1[0].items() if k!='policy'},{k:v for k,v in w2[0].items() if k!='policy'})
    def test_zero_success_na_and_all_failed_energy(self):
        c=C|{'preamble_pool_size':2}
        r,_,d,_=simulate(c,'urban',100,None,0,True)
        self.assertEqual(r['successes'],0);self.assertIsNone(r['energy_per_success_j']);self.assertIsNone(r['delay_s_mean']);self.assertAlmostEqual(r['energy_total_j'],r['energy_unsuccessful_total_j'])
    def test_rx_power_unused(self):
        a=simulate(C,'urban',10,None,0)[0];b=simulate(C|{'power_rx_w':100},'urban',10,None,0)[0]
        self.assertEqual(a,b)
if __name__=='__main__':unittest.main(verbosity=2)
