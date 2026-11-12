#!/bin/bash
# g4.sh <cfg> <mode>
W=/media/avg-usr/Storage/m6000-llama-cpp/occamy35-moonshot-20261110; cfg=$1; m=$2; id=$(basename $cfg .json); id=${id#cfg-}; R=/media/avg-usr/Storage/m6000-llama-cpp/occamy35-night-20261111/rounds/GDN1
cd /tmp; export C04_LOCK_PREHELD=1 SWAP_URL=http://172.21.0.1:11445
python3 -I $W/harness/c04/abba_run.py run --config $cfg --mode $m --execute > $R/g4-$id-$m.run.log 2>&1; echo "run rc=$?"
python3 -I $W/harness/c04/abba_run.py analyze --config $cfg --mode $m --dir $R/g4-$id --out $R/g4-$id-$m.analysis.json > $R/g4-$id-$m.analysis.txt 2>&1
echo finished $id $m; tail -15 $R/g4-$id-$m.analysis.txt
