#!/usr/bin/env bash
# D1BF: per-function SASS compare base (9312e7e D1B) vs D1BF objects; addresses stripped
docker run --rm --name sass-d1bf --cpuset-cpus 6-11 -u "$(id -u):$(id -g)" -e TMPDIR=/tmp -v "$1":/w local/m6000-build-tools:20261013 bash -c '
cd /w
for o in occamy-gdn occamy-fused; do cuobjdump -sass base-$o.cu.o > b-$o.sass; cuobjdump -sass $o.cu.o > n-$o.sass; done
'
