"""Rebuild figures from existing CSV, without rerunning random simulation."""
import csv,sys
from pathlib import Path
from plots import generate
from units import internal_row
def read(path):
    with path.open() as f:rows=list(csv.DictReader(f))
    strings={'experiment','scenario','policy'}
    return [internal_row({k:(v if k in strings else float(v) if v else None) for k,v in row.items()}) for row in rows]
if __name__=='__main__':
    root=Path(__file__).parent/'results'
    for suite in sys.argv[1:] or ['core']:
        folder=root/suite
        if suite=='core' and not (folder/'timeline.csv').exists():raise SystemExit('Run python run.py first to regenerate the detailed timeline omitted from the compact archive.')
        generate(read(folder/'passes.csv'),read(folder/'timeline.csv') if (folder/'timeline.csv').exists() else [],folder,suite)
