#!/bin/bash
# C5 master (run under flock gpu-bench.lock): A raw-logit replay, B dispatch counts, C ABBA mode 2 (+mode 1), D server coverage.
# usage: c5.sh <steps e.g. "A B C D"> ; OUT dir $O
set -u
J=/home/avg-usr/.claude/jobs/fc533454/tmp; C5=$J/c5; O=${O:-$C5/out}; mkdir -p $O/dumps $O/smi $O/slots
GPU=GPU-684f5e72-ce85-7b4f-e4f6-2dddddd6e60e; IMG=nvidia/cuda:12.8.1-devel-ubuntu24.04; SIMG=sha256:3a0caa9afea12f5ca5d982621cc2d86b1fa6ac614d735b262fa475d7dc6e2a6d
PROD=/media/avg-usr/Storage/m6000-llama-cpp/prod-libs/libggml-cuda-downx35-20261110.so; DBG=$J/libs/libggml-cuda-downx-dbg.so
MODEL=/models/occamy-1.0-Q4_0-headq40b.gguf
E=(); for a in GGML_CUDA_MMVQ_SM52=1 GGML_CUDA_RMS_WARP=0 GGML_CUDA_HEAD_ROWPACK_SM52=1 GGML_CUDA_MMQ_COMPACT_SM52=1 GGML_CUDA_Q8_REUSE=1 \
 GGML_CUDA_NORM_Q8=1 GGML_CUDA_CONCAT_T=1 GGML_CUDA_OCCAMY_GDN_STATE_DIRECT=1 GGML_CUDA_GATE_COPY=1 GGML_CUDA_OCCAMY_GDN=1 GGML_CUDA_MMQ_XMAD_SM52=1 \
 GGML_CUDA_GDN_COLS=2,4 GGML_CUDA_OCCAMY_FA=1 GGML_CUDA_OCCAMY_FUSED=1 GGML_CUDA_MMQ_TAIL_SM52=1 GGML_CUDA_OCCAMY_DEBUG=1; do E+=(-e "$a"); done
trap 'docker rm -f c5-run c5-srv >/dev/null 2>&1; pkill -f "nvidia-smi.*c5smi" 2>/dev/null' EXIT
log() { echo "[$(date +%T)] $*" | tee -a $O/master.log; }
smi_on() { nvidia-smi -i $GPU --query-gpu=timestamp,clocks.sm,clocks.mem,temperature.gpu,power.draw,power.limit,clocks_throttle_reasons.active,utilization.gpu --format=csv,noheader -lms 500 > $O/smi/$1.csv 2>/dev/null & SMIPID=$!; }
smi_off() { kill $SMIPID 2>/dev/null; wait $SMIPID 2>/dev/null; }
# harness: hrun <lib> <dx> <name> [extra env...]
hrun() { local lib=$1 dx=$2 nm=$3; shift 3; local ex=(); for a in "$@"; do ex+=(-e "$a"); done
  docker run --rm --name c5-run --network none --device nvidia.com/gpu=$GPU -v /media/avg-usr/Storage/uroboros-models:/models:ro -v $O:/o \
   -v $C5/app:/app:ro -v $lib:/app/libggml-cuda.so:ro -e LD_LIBRARY_PATH=/app "${E[@]}" "${ex[@]}" -e GGML_CUDA_MOE_DOWNX=$dx --entrypoint /app/dxlogits $IMG \
   $MODEL /o/dumps/$nm.bin 64 > $O/$nm.txt 2>&1; echo "rc=$?" >> $O/$nm.txt; log "$nm rc=$(tail -1 $O/$nm.txt) $(grep DONE $O/$nm.txt) $(grep -h 'DXCOUNT exit' $O/$nm.txt)"; }
brun() { local lib=$1 dx=$2 nm=$3; shift 3
  docker run --rm --name c5-run --network none --device nvidia.com/gpu=$GPU -v /media/avg-usr/Storage/uroboros-models:/models:ro \
   -v $J/prof35/app:/app:ro -v $lib:/app/libggml-cuda.so:ro -e LD_LIBRARY_PATH=/app "${E[@]}" ${BX:-} -e GGML_CUDA_MOE_DOWNX=$dx --entrypoint /app/llama $IMG \
   bench -m $MODEL -ngl 999 -fa on -p 0 -n ${BN:-128} -r ${BR:-3} -o csv "$@" > $O/$nm.txt 2>&1; echo "rc=$?" >> $O/$nm.txt; }
tg() { grep -h '^"' $1 | tail -1 | awk -F'","' '{print $(NF-1)}'; }

for step in $1; do case $step in
A) log "A: raw FP32 logits replay, prod lib $(sha256sum $PROD | cut -c1-12)"
   for dx in 0 1 2 3; do hrun $PROD $dx A-dx$dx; done; hrun $PROD 0 A-dx0-rep
   ;;
B) log "B: dispatch counts, dbg lib $(sha256sum $DBG | cut -c1-12)"
   for dx in 0 1 2 3; do hrun $DBG $dx B-h-dx$dx GGML_CUDA_DX_COUNT=1; done
   hrun $DBG 1 B-h-3l-dx1 GGML_CUDA_DX_COUNT=1 GGML_CUDA_OCCAMY_3L=1
   for dx in 0 1 2 3; do BX="-e GGML_CUDA_DX_COUNT=1" BN=64 BR=2 brun $DBG $dx B-bench-dx$dx; log "B-bench-dx$dx $(tail -1 $O/B-bench-dx$dx.txt) $(grep -h 'DXCOUNT exit' $O/B-bench-dx$dx.txt)"; done
   SRVLIB=$DBG SRVDX=1 SRVX="-e GGML_CUDA_DX_COUNT=1" NAME=B-srv-dx1 bash $C5/scripts/srv.sh quick; log "B-srv-dx1 $(grep -h 'DXCOUNT' $O/B-srv-dx1.srv.log | tail -1)"
   ;;
C) log "C: ABBA x5 tg128 r3 dx0 vs dx${CDX:-2}"
   : > $O/C-abba-dx${CDX:-2}.txt
   for c in 1 2 3 4 5; do for dx in 0 ${CDX:-2} ${CDX:-2} 0; do nm=C${CDX:-2}-b-$c-$dx; [ -e $O/$nm.txt ] && nm=$nm-b
     smi_on $nm; brun $PROD $dx $nm; smi_off
     s=$(awk -F', ' '$8+0>50{n++;c+=$2;t=($4>t?$4:t);p+=$5}END{if(n)printf "busy_n=%d sm_mhz_mean=%.0f temp_max=%d pwr_mean=%.1f",n,c/n,t,p/n}' $O/smi/$nm.csv)
     echo "cycle=$c dx=$dx tg128=$(tg $O/$nm.txt) $s thr=$(awk -F', ' '$8+0>50{print $7}' $O/smi/$nm.csv | sort | uniq -c | tr '\n' ' ')" | tee -a $O/C-abba-dx${CDX:-2}.txt
   done; done
   ;;
D) log "D: server coverage"
   for dx in 0 1; do SRVLIB=$PROD SRVDX=$dx NAME=D-srv-dx$dx bash $C5/scripts/srv.sh full; log "D dx=$dx $(tail -3 $O/D-srv-dx$dx.client.txt | tr '\n' ' ')"; done
   ;;
esac; done
log "done $1"
