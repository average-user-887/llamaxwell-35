#!/bin/bash
# g4.sh <cfg> <mode>  (lock held externally)
W=/media/avg-usr/Storage/m6000-llama-cpp/occamy35-moonshot-20261110; cfg=$1; m=$2; id=$(basename $cfg .json); id=${id#cfg-}; R=/media/avg-usr/Storage/m6000-llama-cpp/occamy35-day-20261112/rounds/B3
cd /tmp; export C04_LOCK_PREHELD=1 SWAP_URL=http://<lab-endpoint>
python3 -I $W/harness/c04/abba_run.py run --config $cfg --mode $m --execute > $R/g4-$id-$m.run.log 2>&1; echo "run rc=$?"
python3 -I $W/harness/c04/abba_run.py analyze --config $cfg --mode $m --dir $R/g4-$id --out $R/g4-$id-$m.analysis.json > $R/g4-$id-$m.analysis.txt 2>&1
echo finished $id $m; python3 -I -c "import json,sys;d=json.load(open('$R/g4-$id-$m.analysis.json'));d=d.get('$m',d);[print(c['cycle'],round(c['mean_a'],3),round(c['mean_b'],3),round(c['delta_pct'],3)) for c in d['per_cycle']];print({k:v for k,v in d.items() if k not in('per_cycle','trials')})" 2>&1 | tail -6
