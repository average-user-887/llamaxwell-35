#!/usr/bin/env bash
set -Eeuo pipefail
cd /b/tools/server
flags=CMakeFiles/server-context.dir/flags.make
original="1138515898887c6ec6be01a210cfb59bb8e1338ccf6eeef1d032ffd4fdb22b65"
test "$(sha256sum /b/bin/libllama-server-impl.so | cut -d" " -f1)" = "$original"
sed "s|-o ../../bin/libllama-server-impl.so|-o /work/original-relink.so|" \
  CMakeFiles/llama-server-impl.dir/link.txt >/work/original-link.sh
nice -n 19 bash -ex /work/original-link.sh
test "$(sha256sum /work/original-relink.so | cut -d" " -f1)" = "$original"
read -ra defines <<< "$(sed -n "s/^CXX_DEFINES = //p" "$flags")"
read -ra includes <<< "$(sed -n "s/^CXX_INCLUDES = //p" "$flags")"
read -ra cxxflags <<< "$(sed -n "s/^CXX_FLAGS = //p" "$flags")"
set -x
nice -n 19 /usr/bin/g++-14 "${defines[@]}" "${includes[@]}" "${cxxflags[@]}" \
  -o /work/server-context.cpp.o -c /work/server-context.cpp
cp libserver-context.a /work/libserver-context.a
/usr/bin/ar r /work/libserver-context.a /work/server-context.cpp.o
/usr/bin/ranlib /work/libserver-context.a
sed "s|-o /work/original-relink.so|-o /work/libllama-server-impl.so|;s| libserver-context.a | /work/libserver-context.a |" \
  /work/original-link.sh >/work/candidate-link.sh
nice -n 19 bash -ex /work/candidate-link.sh
