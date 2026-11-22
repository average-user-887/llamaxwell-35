#!/usr/bin/env bash
# C2 build: recompile server-context.cpp into libllama-server-impl.so (mirrors r3-truncation-a2 build-r3-all.sh step 3/4,
# server part only; libllama-common/libllama are the card1 copies placed in the work dir). usage: build-c2.sh <workdir> <cpu>
set -Eeuo pipefail
work=$(cd "$1" && pwd -P); cpu=${2:-3}
c=/media/avg-usr/Storage/m6000-llama-cpp/mtp-moonshot-20261117
n40=$c/night-one-graph-draft
source_mount=/media/avg-usr/Storage/m6000-llama-cpp/occamy35-night-20261114/9/src
image=sha256:747c8a878fc2a36aca96e10da800f6cfe80d1e813c7e4af49ded30588be16bed
docker run --rm --pull never --runtime runc --network none --cpuset-cpus "$cpu" \
  -e NVIDIA_VISIBLE_DEVICES=void -e TMPDIR=/tmp --tmpfs /tmp:rw,size=256m \
  --user "$(id -u):$(id -g)" --name "r3bench-c2-build-$(basename "$work")" \
  -v "$n40/src:$source_mount:ro" -v "$n40/build:/b:ro" -v "$work:/work" \
  -w /b/tools/server --entrypoint /bin/bash "$image" -euc '
    sflags=CMakeFiles/server-context.dir/flags.make
    read -ra sdefines <<< "$(sed -n "s/^CXX_DEFINES = //p" "$sflags")"
    read -ra sincludes <<< "$(sed -n "s/^CXX_INCLUDES = //p" "$sflags")"
    read -ra scxxflags <<< "$(sed -n "s/^CXX_FLAGS = //p" "$sflags")"
    /usr/bin/g++-14 "${sdefines[@]}" "${sincludes[@]}" "${scxxflags[@]}" -Werror -I/work \
      -o /work/server-context.cpp.o -c /work/server-context.cpp
    cp libserver-context.a /work/libserver-context.a
    /usr/bin/ar r /work/libserver-context.a /work/server-context.cpp.o
    /usr/bin/ranlib /work/libserver-context.a
    sed -e "s|-o ../../bin/libllama-server-impl.so|-o /work/libllama-server-impl.so|" \
        -e "s| libserver-context.a | /work/libserver-context.a |" \
        -e "s|../../bin/libllama-common.so.0.6.0|/work/libllama-common.so.0.6.0|" \
        -e "s|../../bin/libllama.so.0.6.0|/work/libllama.so.0.6.0|" \
        CMakeFiles/llama-server-impl.dir/link.txt > /work/server-link.sh
    bash /work/server-link.sh
  '
sha256sum "$work/libllama-server-impl.so"
