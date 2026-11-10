"""Explicit idealized preamble detector, with optional legacy data-block proxy."""
import numpy as np
from scipy.stats import ncx2

def detect_preamble(snr_db,rng,false_alarm=1e-3,processing_gain=1.):
    if not 0<false_alarm<1 or processing_gain<=0:raise ValueError('Invalid detector parameters')
    rho=10**(np.clip(snr_db,-300.,300.)/10)*processing_gain
    threshold=-np.log(false_alarm)
    # Matched-filter output: sqrt(rho) + CN(0,1). Known delay/sequence, unknown phase.
    i=np.sqrt(rho)+rng.normal(size=np.shape(rho))/np.sqrt(2)
    q=rng.normal(size=np.shape(rho))/np.sqrt(2)
    missed=i*i+q*q<threshold
    # Triangle inequality: a miss implies |noise| > sqrt(rho)-sqrt(threshold).
    # CN(0,1) has P(|noise|>x)=exp(-x²). If this upper bound is below
    # floating-point range, the miss probability rounds to zero. This also
    # avoids noncentral-CDF numerical failures at enormous SNR.
    representable=(np.sqrt(rho)-np.sqrt(threshold)) <= np.sqrt(745.)
    bler=np.zeros_like(rho,dtype=float)
    bler[representable]=ncx2.cdf(2*threshold,df=2,nc=2*rho[representable])
    return bler,missed
