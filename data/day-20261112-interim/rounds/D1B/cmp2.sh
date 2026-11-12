#!/bin/bash
# cmp2.sh <a> <b> : per-seq logits + state compare of two dumps dirs
R=/media/avg-usr/Storage/m6000-llama-cpp/occamy35-day-20261112/rounds/D1B; a=$R/dumps/$1 b=$R/dumps/$2
for k in 0 1; do
  la=$a/s$k.logits lb=$b/s$k.logits
  [ -s $la ] || continue
  python3 -I - $la $lb $a/log.txt <<'PY'
import sys,re,numpy as np
a=np.fromfile(sys.argv[1],np.float32); b=np.fromfile(sys.argv[2],np.float32)
nv=int(re.search(r'n_vocab=(\d+)',open(sys.argv[3]).read()).group(1))
if a.size!=b.size: print('  size differ',a.size,b.size); sys.exit()
a=a.reshape(-1,nv); b=b.reshape(-1,nv)
be=(a.view(np.uint32)==b.view(np.uint32)).all()
print('  logits rows=%d bitequal=%s maxabs=%.3e top1_same=%.3f'%(a.shape[0],be,np.abs(a-b).max(),(a.argmax(1)==b.argmax(1)).mean()))
PY
  echo "  seq$k state: $(cmp -s $a/s$k.state $b/s$k.state && echo BITEQUAL || echo DIFF) $(stat -c %s $a/s$k.state)"
done
