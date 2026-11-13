#!/bin/bash
# Q8S shallow set: dx2seq pair mode, lib D1B, N2SPLIT=0 (= stock pair0 baseline arm), only cache types differ. Caller holds gpu lock. usage: shallow_q8.sh ARM(Q8K|Q8V|Q8KV)...
M=<LAB>; R=$M/occamy35-day-20261112/rounds/D1B; N=$M/occamy35-night-20261113; Q=$N/Q8S; O=$N/N2/shallow; D0=$M/occamy35-day-20261112/rounds/D0
GPU=${N2_GPU:?}; IMG=nvidia/cuda:12.8.1-devel-ubuntu24.04; RS=$M/run-35b-gdn4-20261112.sh; LIB=$M/occamy35-night-20261111/cand/D1B/build/bin/libggml-cuda.so
declare -A EV=( [e1]="P0:0:3000 P1:200000:203000 D40" [ea]="P1:200000:203000 D8 P0:0:3000 D32" [eb]="P0:0:3000 P1:200000:203000 D16 X0 D16" [ec]="P0:0:3000 P1:200000:206000 D12 X0 P0:100000:102000 D24" )
E=(); for a in $(grep -o -- '-e GGML_[A-Z0-9_]*=[^ ]*' $RS | sed 's/-e //'); do E+=(-e "$a"); done; E+=(-e GGML_CUDA_OCCAMY_N2SPLIT=0)
for arm in "$@"; do kt=f16; vt=f16; X=(); [ $arm = Ap ] && X=(-e GGML_CUDA_FA_PARALLEL_BLOCKS_MULT=2)
 for c in e1 ea eb ec; do d=$O/${c}_$arm; mkdir -p $d
  docker run --rm --name n2-sh --network none -u $(id -u):$(id -g) --device nvidia.com/gpu=$GPU -v $M/../uroboros-models:/models:ro -v $d:/o -v $D0/deep_text.txt:/t.txt:ro -v $M/occamy35-night-20261111/app:/app:ro \
   -v $Q/bin/dx2seq_q8:/app2/dx2seq:ro -v $LIB:/app/libggml-cuda.so:ro -e LD_LIBRARY_PATH=/app -e DX_KT=$kt -e DX_VT=$vt "${E[@]}" "${X[@]}" -w /app --entrypoint /app2/dx2seq $IMG /models/occamy-1.0-Q4_0-headq40b.gguf /t.txt /o pair 32768 ${EV[$c]} > $d/log.txt 2>&1; echo "rc=$?" >> $d/log.txt
  echo "$arm $c: $(grep -E '^rc=|DONE|failed' $d/log.txt | tr '\n' ' ')"; done; done
