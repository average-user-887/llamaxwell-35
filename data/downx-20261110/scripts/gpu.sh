#!/bin/bash
# gpu.sh <lib> : run under flock gpu-bench.lock. Gate 2 (bitwise logits, DOWNX off vs on), then ABBA x5 tg128.
LIB=$1
J=/home/avg-usr/.claude/jobs/fc533454/tmp; P=$J/prof35; O=${OUT:-$J/downx/out}; ON=${DXON:-1}; mkdir -p $O; W=/media/avg-usr/Storage/m6000-llama-cpp/native-occamy/p3/raw
GPU=GPU-684f5e72-ce85-7b4f-e4f6-2dddddd6e60e; CN=downx-run; IMG=nvidia/cuda:12.8.1-devel-ubuntu24.04
trap "docker rm -f $CN >/dev/null 2>&1" EXIT
E=(); for a in GGML_CUDA_MMVQ_SM52=1 GGML_CUDA_RMS_WARP=0 GGML_CUDA_HEAD_ROWPACK_SM52=1 GGML_CUDA_MMQ_COMPACT_SM52=1 GGML_CUDA_Q8_REUSE=1 \
 GGML_CUDA_NORM_Q8=1 GGML_CUDA_CONCAT_T=1 GGML_CUDA_OCCAMY_GDN_STATE_DIRECT=1 GGML_CUDA_GATE_COPY=1 GGML_CUDA_OCCAMY_GDN=1 GGML_CUDA_MMQ_XMAD_SM52=1 \
 GGML_CUDA_GDN_COLS=2,4 GGML_CUDA_OCCAMY_FA=1 GGML_CUDA_OCCAMY_FUSED=1 GGML_CUDA_MMQ_TAIL_SM52=1 GGML_CUDA_OCCAMY_DEBUG=1; do E+=(-e "$a"); done
run() { # run <dx 0|1> <args...>
  local dx=$1; shift
  docker run --rm --name $CN --network none --device nvidia.com/gpu=$GPU -v /media/avg-usr/Storage/uroboros-models:/models:ro -v $W:/w:ro -v $O:/o \
    -v $P/app:/app:ro -v $LIB:/app/libggml-cuda.so:ro -e LD_LIBRARY_PATH=/app "${E[@]}" -e GGML_CUDA_MOE_DOWNX=$([ $dx = 1 ] && echo $ON || echo 0) --entrypoint /app/llama $IMG "$@"
}
if [ "${SKIP_G2:-0}" != 1 ]; then
for dx in 0 1; do
  rm -f $O/logits-dx$dx.bin
  run $dx perplexity -m /models/occamy-1.0-Q4_0-headq40b.gguf -ngl 999 -fa on -c 128 -b 1 -ub 1 --chunks 1 -f /w/text.txt --seed 0 \
    --kl-divergence-base /o/logits-dx$dx.bin > $O/g2-dx$dx.txt 2>&1
  echo "g2 dx=$dx rc=$? $(grep -E 'Final estimate|\[1\]' $O/g2-dx$dx.txt | tr '\n' ' ') $(grep -h 'occamy-fused: device\|fused MoE window' $O/g2-dx$dx.txt | tr '\n' ' ')"
done
ls -l $O/logits-dx0.bin $O/logits-dx1.bin
cmp $O/logits-dx0.bin $O/logits-dx1.bin && echo "G2 BITWISE EQUAL" || { echo "G2 DIFFER"; exit 3; }
fi
[ "${G2_ONLY:-0}" = 1 ] && exit 0
: > $O/abba.txt
for c in 1 2 3 4 5; do for dx in 0 1 1 0; do
  run $dx bench -m /models/occamy-1.0-Q4_0-headq40b.gguf -ngl 999 -fa on -p 0 -n 128 -r 3 -o csv > $O/b-$c-$dx.txt 2>&1
  t=$(grep -h '^"' $O/b-$c-$dx.txt | tail -1 | awk -F'","' '{print $(NF-1)}')
  echo "cycle=$c dx=$dx tg128=$t" | tee -a $O/abba.txt
done; done
