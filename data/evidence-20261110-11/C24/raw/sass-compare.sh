#!/usr/bin/env bash
# C24: CPU-only SASS compare, pin build-clean vs merged C24 build, flags-OFF variants.
set -uo pipefail
WORK=/media/avg-usr/Storage/m6000-llama-cpp/occamy35-moonshot-20261110
taskset -c 6-11 docker run --rm --name sass-c24 --cpuset-cpus 6-11 -u "$(id -u):$(id -g)" -e TMPDIR=/tmp -v "$WORK":/w local/m6000-build-tools:20261013 bash -c '
set -u
D=ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir
cd /w/results/C24/raw/sass
for o in occamy-gdn occamy-fused; do
  cuobjdump -sass /w/build-clean/$D/$o.cu.o > clean-$o.sass
  cuobjdump -sass /w/cand/C24/build/$D/$o.cu.o > c24-$o.sass
done
ext() { awk -v f="Function : $2" "index(\$0,f){on=1;next} /Function :/{on=0} on" $1 | grep -E "/\*[0-9a-f]{4}\*/" | sed -E "s#/\*[0-9a-f]{4}\*/##; s#;.*##"; }
for p in 0 1 2 3; do
  a=_ZN10occamy_gdn9gdn_fusedILi${p}EEEvNS_6ParamsE
  b=_ZN10occamy_gdn9gdn_fusedILi${p}ELi0ELi0EEEvNS_6ParamsE
  [ $p = 1 ] && b=_ZN10occamy_gdn9gdn_fusedILi1ELi0ELi0EEEvNS_6ParamsE
  ext clean-occamy-gdn.sass $a > g-clean-$p.s; ext c24-occamy-gdn.sass $b > g-c24-$p.s
  printf "gdn<%s> pin=%s c24=%s identical=%s\n" $p $(wc -l < g-clean-$p.s) $(wc -l < g-c24-$p.s) $(cmp -s g-clean-$p.s g-c24-$p.s && echo Y || echo N)
done
for k in 0ELi0 0ELi1 0ELi2 0ELi3 1ELi0 2ELi0 3ELi0 3ELi1 3ELi2 3ELi3; do
  f=_ZN6occamy6moe_v2ILi${k}EEEvNS_6ParamsE
  ext clean-occamy-fused.sass $f > m-clean-$k.s; ext c24-occamy-fused.sass $f > m-c24-$k.s
  printf "moe_v2<%s> pin=%s c24=%s identical=%s\n" $k $(wc -l < m-clean-$k.s) $(wc -l < m-c24-$k.s) $(cmp -s m-clean-$k.s m-c24-$k.s && echo Y || echo N)
done
grep -o "Function : [^ ]*" clean-occamy-gdn.sass clean-occamy-fused.sass | sort | uniq -c | wc -l
' 2>&1 | grep -vE "^$|CUDA|Copyright|License|license|NVIDIA|GPU functionality|Toolkit|=====" | tee "$WORK/results/C24/raw/sass/summary.txt"
