"""Reproducible outdoor sensor deployment; Python 3.10+, standard library only."""
import argparse, csv, json, math, random
from pathlib import Path

# Illustrative geometry, NOT calibrated propagation parameters.
PROFILES = {'rural_wooded': (0, 0, 0, 0, 0),
            'village': (65, 15, 30, 3, 9),
            'urban': (85, 25, 55, 20, 80)}
TABLES = {'rural_wooded': 'table_7_bler_vs_theta_rural_wooded.csv',
          'village': 'table_6_bler_vs_theta_village.csv',
          'urban': 'table_5_bler_vs_theta_urban.csv'}

def inside(x, y, b):
    return b[0] <= x <= b[2] and b[1] <= y <= b[3]

def buildings(rng, profile, size):
    count, lo, hi, hmin, hmax = PROFILES[profile]
    result = []
    for _ in range(20000):
        if len(result) == count: return result
        w, d = rng.uniform(lo, hi), rng.uniform(lo, hi)
        x, y = rng.uniform(0, size-w), rng.uniform(0, size-d)
        b = (x, y, x+w, y+d, rng.uniform(hmin, hmax))
        if all(b[2]+3 < a[0] or a[2]+3 < b[0] or b[3]+3 < a[1] or a[3]+3 < b[1] for a in result):
            result.append(b)
    raise ValueError('Buildings do not fit; increase area or reduce building count.')

def blocked(x, y, height, obstacles, theta, azimuth):
    # Ray-box intersection in XY; rising ray is below roof at entry.
    dx, dy = math.cos(math.radians(azimuth)), math.sin(math.radians(azimuth))
    slope = math.tan(math.radians(theta))
    for b in obstacles:
        enter, leave = 0., float('inf')
        for p, v, lower, upper in ((x, dx, b[0], b[2]), (y, dy, b[1], b[3])):
            if abs(v) < 1e-12:
                if not lower <= p <= upper: leave = -1.; break
            else:
                t1, t2 = sorted(((lower-p)/v, (upper-p)/v))
                enter, leave = max(enter, t1), min(leave, t2)
        if leave >= enter and height + enter*slope < b[4]: return True
    return False

