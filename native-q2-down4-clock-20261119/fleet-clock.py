"""Capture time-aligned M6000 clocks from the admitted Fleetdash source and compare mirrored windows."""
import json
import math
import os
from pathlib import Path
import re
import statistics
import sys
import time
from urllib.request import urlopen


def sample(directory, deadline):
    fields = ('clock_mhz','memory_clock_mhz','util_pct','power_limit_w','temp_c')
    with (directory / 'fleet-clock.jsonl').open('x') as output:
        seen = set()
        while time.time() < deadline and not (directory / 'perf.done').exists():
            try:
                with urlopen('@PRIVATE_FLEET_URL@', timeout=2) as response:
                    data = json.load(response)
                source = data['sources']['ryzen_telemetry']
                g, = [g for g in source['data']['gpus'] if g['uuid'] == os.environ['N3_GPU']
                      and g['name'] == 'Quadro M6000 24GB']
                stamp = float(g['observed_at'])
                if source['ok'] and 0 <= data['now']-stamp <= 30 and stamp not in seen:
                    seen.add(stamp)
                    row = {k:g.get(k) for k in fields}
                    row['observed_at'] = stamp
                    print(json.dumps(row,allow_nan=False),file=output,flush=True)
            except (KeyError,ValueError,OSError,TypeError):
                pass
            time.sleep(1)


def analyze(raw, samples):
    windows = re.findall(r'^Q2_DOWN4 WINDOW (independent|partial) mode=(cohorts|down4) reps=98304 begin=([0-9.]+) end=([0-9.]+)$',raw,re.M)
    if [a[1] for a in windows] != ['cohorts','down4','down4','cohorts','down4','cohorts','cohorts','down4']:
        raise ValueError('missing or out-of-order wall clock windows')
    if [a[0] for a in windows] != ['independent']*4+['partial']*4:
        raise ValueError('window fixture ordering')
    output = []
    for name, mode, begin, end in windows:
        if float(end)<=float(begin): raise ValueError('nonpositive wall window')
        chosen = [g for g in samples if float(begin)<=g['observed_at']<=float(end)]
        row={'fixture':name,'mode':mode,'begin_epoch':float(begin),'end_epoch':float(end),'samples':len(chosen)}
        for k in ('clock_mhz','memory_clock_mhz'):
            values=[g[k] for g in chosen if isinstance(g.get(k),(int,float)) and math.isfinite(g[k]) and g[k]>0]
            row[k]={'range':[min(values),max(values)],'median':statistics.median(values)} if values else None
        output.append(row)
    paired=[]
    for a,b in zip(output[::2],output[1::2]):
        row={}
        for k in ('clock_mhz','memory_clock_mhz'):
            x,y=a[k],b[k]
            row[k+'_median_ratio']=max(x['median'],y['median'])/min(x['median'],y['median']) if x and y else None
        paired.append(row)
    ratios = [v for pair in paired for v in pair.values()]
    return {'source':'Fleetdash8090 ryzen_telemetry, admitted M6000 only','windows':output,'paired':paired,
            'all_pairs_within_two_percent':all(v is not None and 1/1.0200001<=v<=1.0200001 for v in ratios)}


if __name__ == '__main__':
    action, directory, *args = sys.argv[1:]
    d = Path(directory)
    if action=='--sample': sample(d,float(args[0]))
    elif action=='--summary':
        raw=(d/'perf.log').read_text()
        data=[json.loads(line) for line in (d/'fleet-clock.jsonl').read_text().splitlines()]
        print(json.dumps(analyze(raw,data),indent=2))
    else: raise ValueError('unknown clock command')
