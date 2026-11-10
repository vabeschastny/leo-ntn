"""Public mW/mJ interface, SI watts/joules inside the physical model."""
def normalize_config(c):
    c=dict(c)
    for key in list(c):
        if key.startswith('power_') and key.endswith('_mw'):
            target=key[:-3]+'_w'
            if target in c:raise ValueError('Duplicate power units: '+key)
            c[target]=c.pop(key)/1000
    return c

def external_config(c):
    return {k[:-2]+'_mw' if k.startswith('power_') and k.endswith('_w') else k:
            v*1000 if k.startswith('power_') and k.endswith('_w') else v for k,v in c.items()}

def external_row(r):
    result={}
    for k,v in r.items():
        if '_j' in k:
            scale=not k.endswith('_defined_passes')
            k=k.replace('_j','_mj');v=v*1000 if v is not None and scale else v
        result[k]=v
    return result

def internal_row(r):
    return {k.replace('_mj','_j'):v/1000 if '_mj' in k and v is not None and not k.endswith('_defined_passes') else v for k,v in r.items()}
