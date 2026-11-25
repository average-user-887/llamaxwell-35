#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ -x "$SCRIPT_DIR/bin/llama-server" ]]; then
    BIN_DIR="$SCRIPT_DIR/bin"
elif [[ -x "$SCRIPT_DIR/llama-server" ]]; then
    BIN_DIR="$SCRIPT_DIR"
else
    echo "Error: llama-server not found in $SCRIPT_DIR/bin or $SCRIPT_DIR" >&2
    exit 1
fi

MODE="mtp"
MODEL=""
HOST="127.0.0.1"
PORT="8080"
CTX="84224"
SLOTS="1"
CACHE_RAM="12288"
CTX_CHECKPOINTS="8"
GPU_ID=""
ALIAS="qwen36-moe"
EXTRA_ARGS=()

usage() {
    cat <<EOF
Usage: $(basename "$0") --model PATH [options] [-- extra llama-server args]

Options:
  --model PATH            Path to GGUF model (required)
  --mode MODE             mtp (default, R3-lossless MTP1), serial (no MTP), mtp-q8c (Q8C verify)
  --host IP               Bind address (default: 127.0.0.1)
  --port PORT             Listen port (default: 8080)
  --ctx N                 Total context tokens (default: 84224)
  --slots N               Parallel slots -np (default: 1)
  --cache-ram MB          Prompt cache RAM in MiB (default: 12288; 0 to disable)
  --ctx-checkpoints N     Recurrent/hybrid checkpoints per slot (default: 8)
  --gpu INDEX_OR_UUID     Set CUDA_VISIBLE_DEVICES
  --alias NAME            Model alias (default: qwen36-moe)
  -h, --help              Show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model) MODEL="$2"; shift 2 ;;
        --mode) MODE="$2"; shift 2 ;;
        --host) HOST="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        --ctx) CTX="$2"; shift 2 ;;
        --slots) SLOTS="$2"; shift 2 ;;
        --cache-ram) CACHE_RAM="$2"; shift 2 ;;
        --ctx-checkpoints) CTX_CHECKPOINTS="$2"; shift 2 ;;
        --gpu) GPU_ID="$2"; shift 2 ;;
        --alias) ALIAS="$2"; shift 2 ;;
        --) shift; EXTRA_ARGS+=("$@"); break ;;
        -h|--help) usage; exit 0 ;;
        *) EXTRA_ARGS+=("$1"); shift ;;
    esac
done

if [[ -z "$MODEL" ]]; then
    echo "Error: --model is required" >&2
    usage
    exit 2
fi

if [[ ! -f "$MODEL" ]]; then
    echo "Error: model file not found: $MODEL" >&2
    exit 1
fi

# Footgun guard: ggml-cuda.cu checks getenv("GGML_CUDA_OCCAMY_DEBUG") != nullptr,
# so even GGML_CUDA_OCCAMY_DEBUG=0 enables debug logging and slows execution.
unset GGML_CUDA_OCCAMY_DEBUG

if [[ -n "$GPU_ID" ]]; then
    export CUDA_VISIBLE_DEVICES="$GPU_ID"
fi

export LD_LIBRARY_PATH="$BIN_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# Certified R3 Maxwell sm_52 kernel stack
export GGML_CUDA_MAXWELL_GRAPHS=1
export GGML_CUDA_MOE_DOWNX=1
export GGML_CUDA_MOE_DOWNH=2
export GGML_CUDA_MOE_LOP3=1
export GGML_CUDA_MOE_TAILBAL=1
export GGML_CUDA_MOE_V3=1
export GGML_CUDA_MMVQ_SM52=1
export GGML_CUDA_RMS_WARP=0
export GGML_CUDA_HEAD_ROWPACK_SM52=1
export GGML_CUDA_MMQ_COMPACT_SM52=1
export GGML_CUDA_Q8_REUSE=1
export GGML_CUDA_NORM_Q8=1
export GGML_CUDA_CONCAT_T=1
export GGML_CUDA_GATE_COPY=1
export GGML_CUDA_MMQ_XMAD_SM52=1
export GGML_CUDA_MMQ_TAIL_SM52=1
export GGML_CUDA_OCCAMY_FUSED=1
export GGML_CUDA_OCCAMY_GDN=1
export GGML_CUDA_OCCAMY_GDN_STATE_DIRECT=1
export GGML_CUDA_OCCAMY_GDN_QKREUSE=1
export GGML_CUDA_OCCAMY_GDN_PCSHARE=1
export GGML_CUDA_OCCAMY_GDN_PREF=1
export GGML_CUDA_OCCAMY_GDN_REPACK=1
export GGML_CUDA_GDN_COLS=2,4
export GGML_CUDA_OCCAMY_HOSTGAP=1
export GGML_CUDA_OCCAMY_SMALLFUSE=2
export GGML_CUDA_OCCAMY_N2SPLIT=1
export GGML_CUDA_OCCAMY_N2GDN=1
export GGML_CUDA_OCCAMY_FA=1
export GGML_CUDA_OCCAMY_FA_PIPE=3
export GGML_CUDA_OCCAMY_FA_ADDR=1
export GGML_CUDA_FA_MAXWELL_GEMM=1
export GGML_CUDA_FA_MAXWELL_GEMM_FUSED=1
export GGML_CUDA_OCCAMY_FA_SPARSE=1
export GGML_CUDA_OCCAMY_FA_SPARSE_R2=1
export GGML_CUDA_OCCAMY_GDN_T2=1
export GGML_CUDA_OCCAMY_GDN_T2_Q8A=1
export GGML_CUDA_OCCAMY_GDN_T2_Q8A_EVENTS=0
export GGML_CUDA_OCCAMY_FA_MROW=3
export GGML_CUDA_OCCAMY_FA2_NU=4
export GGML_CUDA_OCCAMY_MOE_Q2_COHORTS=1
export GGML_CUDA_OCCAMY_MOE_Q2_COUNT=1
export GGML_CUDA_OCCAMY_MOE_Q2_READ_IDS=0
export GGML_CUDA_OCCAMY_FA_Q2_R2_TOP16=1
export GGML_CUDA_OCCAMY_FA_Q2_R2_TOP16_CHECK=0
export GGML_CUDA_OCCAMY_FA_Q2_R2_PROBE4=1
export GGML_CUDA_OCCAMY_FA_Q2_R2_STRIDE=1
export GGML_CUDA_OCCAMY_MOE_Q2_DOWN4=1
export GGML_CUDA_OCCAMY_PERSIST_SCRATCH2=1
export GGML_CUDA_OCCAMY_MOE_SH_Q8=1
export GGML_CUDA_OCCAMY_FA_Q1_R2_FAST=1
export LLAMA_MTP_MERGE_REEVAL=1
export LLAMA_MTP_CACHE_ONLY=1
export LLAMA_MTP_LAST_QUERY=1
export LLAMA_SPEC_RS_SAMPLER_SNAPSHOT_ELIDE=1
export LLAMA_OCCAMY_COMPACT_READBACK=1
export LLAMA_OCCAMY_COMPACT_DRAFT=1

SPEC_ARGS=()
case "$MODE" in
    mtp)
        SPEC_ARGS=(--spec-type draft-mtp --spec-draft-n-max 1 --spec-draft-type-k f16 --spec-draft-type-v f16)
        ;;
    mtp-q8c)
        export GGML_CUDA_OCCAMY_GDN_T2_Q8C=1
        SPEC_ARGS=(--spec-type draft-mtp --spec-draft-n-max 1 --spec-draft-type-k f16 --spec-draft-type-v f16)
        ;;
    serial)
        SPEC_ARGS=(--spec-type none)
        ;;
    *)
        echo "Error: unknown --mode '$MODE' (expected: mtp, serial, mtp-q8c)" >&2
        exit 2
        ;;
esac

exec "$BIN_DIR/llama-server" \
    -m "$MODEL" \
    --alias "$ALIAS" \
    --override-kv tokenizer.ggml.pre=str:qwen35 \
    -ngl 999 \
    -np "$SLOTS" \
    -c "$CTX" \
    --no-context-shift \
    -b 2048 \
    -ub 1024 \
    --mixed-batch-size 1024 \
    --prefill-single-stream \
    --ctx-checkpoints "$CTX_CHECKPOINTS" \
    --checkpoint-min-step 2048 \
    --cache-ram "$CACHE_RAM" \
    -fa on \
    --host "$HOST" \
    --port "$PORT" \
    --jinja \
    --metrics \
    "${SPEC_ARGS[@]}" \
    "${EXTRA_ARGS[@]}"
