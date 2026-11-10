"""Physical link budget + ITU-R P.681 Loo mixture; no population matching."""
import csv
from pathlib import Path
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.special import ndtr,ndtri,log_ndtr
SCENARIOS = ('urban', 'village', 'rural_wooded')

def slant_km(theta, altitude=600.):
    a = np.radians(theta)
    return np.sqrt((6371.+altitude)**2-6371.**2*np.cos(a)**2)-6371.*np.sin(a)

def fspl_db(distance_km, frequency_hz=2e9):
    return 20*np.log10(4*np.pi*np.asarray(distance_km)*1000*frequency_hz/299792458.)

def thermal_noise_dbm(temperature_k, bandwidth_hz, noise_figure_db):
    if temperature_k <= 0 or bandwidth_hz <= 0 or noise_figure_db < 0:
        raise ValueError('Require positive T and B, nonnegative NF')
    return 10*np.log10(1.380649e-23*temperature_k*bandwidth_hz/1e-3)+noise_figure_db
ROOT=Path(__file__).parent

class Loo:
    def __init__(self,path=None):
        with Path(path or ROOT/'data/itu_2p2ghz.csv').open() as f:rows=list(csv.DictReader(f))
        self.curves={}
        for s in SCENARIOS:
            a=sorted((r for r in rows if r['scenario']==s),key=lambda r:float(r['theta_deg']))
            for k in a[0]:
                if k not in ('scenario','theta_deg'):
                    self.curves[s,k]=PchipInterpolator([float(r['theta_deg']) for r in a],[float(r[k]) for r in a],extrapolate=False)
    def parameters(self,s,theta):
        theta=np.asarray(theta,dtype=float)
        if np.any((theta<20)|(theta>70)) or np.any(~np.isfinite(theta)):raise ValueError('Loo parameter domain 20..70 degrees')
        p={k[1]:v(theta) for k,v in self.curves.items() if k[0]==s}
        return p
    @staticmethod
    def mean_duration(mu,sigma,minimum):
        z=(np.log(minimum)-mu)/sigma
        # Stable survival-function form of ITU (17a).
        return np.exp(mu+.5*sigma*sigma+log_ndtr(sigma-z)-log_ndtr(-z))
    def state_probability(self,p):
        dg=self.mean_duration(p['mu_G'],p['sigma_G'],p['durmin_G'])
        db=self.mean_duration(p['mu_B'],p['sigma_B'],p['durmin_B'])
        lo,hi=ndtri(p['p_B_min']),ndtri(p['p_B_max'])
        pdf=lambda z:np.exp(-.5*z*z)/np.sqrt(2*np.pi)
        # Mean of truncated BAD MA; primary ITU denominator F(max)-F(min).
        bad_mean=p['mu_MA_B']+p['sigma_MA_B']*(pdf(lo)-pdf(hi))/(p['p_B_max']-p['p_B_min'])
        transition=p['f1']*(p['mu_MA_G']-bad_mean)+p['f2']
        if np.any(transition<0):raise ValueError('Invalid negative transition length')
        return (db+transition)/(dg+db+2*transition)
    def sample(self,s,theta,rng,size=None):
        p=self.parameters(s,theta);pb=self.state_probability(p)
        if size is not None:
            p={k:np.broadcast_to(v,(size,)) for k,v in p.items()}
            pb=np.broadcast_to(pb,(size,))
        bad=rng.random(np.shape(pb))<pb
        mu=np.where(bad,p['mu_MA_B'],p['mu_MA_G'])
        sigma=np.where(bad,p['sigma_MA_B'],p['sigma_MA_G'])
        lower=np.where(bad,p['p_B_min'],ndtr(-1.645))
        upper=np.where(bad,p['p_B_max'],ndtr(1.645))
        ma=mu+sigma*ndtri(rng.uniform(lower,upper))
        sigma_a=np.where(bad,p['g1_B'],p['g1_G'])*ma+np.where(bad,p['g2_B'],p['g2_G'])
        if np.any(sigma_a<0):raise ValueError('Negative Loo amplitude standard deviation')
        mp=np.where(bad,p['h1_B'],p['h1_G'])*ma+np.where(bad,p['h2_B'],p['h2_G'])
        a=10**((ma+sigma_a*rng.normal(size=np.shape(ma)))/20)
        sigma_mp=np.sqrt(10**(mp/10)/2)
        i=a+sigma_mp*rng.normal(size=np.shape(ma));q=sigma_mp*rng.normal(size=np.shape(ma))
        gain=10*np.log10(np.maximum(i*i+q*q,np.finfo(float).tiny))
        return gain,dict(bad=bad,p_bad=pb,ma_db=ma,sigma_a_db=sigma_a,mp_db=mp)

def system_loss(cfg,theta):
    if cfg['loss_mode']=='table1':return np.zeros_like(np.asarray(theta))+cfg['system_loss_db']
    # Article section 4.1; components REPLACE fixed total, never add to it.
    return cfg['polarization_loss_db']+3-2*np.sin(np.radians(theta))+cfg['atmosphere_loss_db']

def link_snr(cfg,distance_km,theta,gain_db):
    return cfg['eirp_dbm']-fspl_db(distance_km,cfg['frequency_hz'])-system_loss(cfg,theta)+cfg['rx_gain_db']-cfg['noise_dbm']+gain_db+cfg['coding_gain_per_doubling_db']*np.log2(cfg['repetitions'])

def sample_at_elevation(loo,scenario,actual_theta,rng,policy):
    """Approved approximation applies ONLY to statistical Loo parameters.

    Keep Loo.parameters itself strict at 20..70; callers must use actual
    elevation and distance in the physical link budget.
    """
    theta=np.asarray(actual_theta,dtype=float)
    if policy!='hold20' or np.any(~np.isfinite(theta)) or np.any((theta<10)|(theta>70)):
        raise ValueError('Explicit hold20 policy and actual angles in 10..70 required')
    parameter_theta=np.maximum(theta,20.)
    gain,meta=loo.sample(scenario,parameter_theta,rng)
    return gain,meta|{'parameter_theta_deg':parameter_theta,'low_angle_approximation':theta<20.}
