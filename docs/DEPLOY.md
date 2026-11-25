# Deploying LlamaMaxwell-35 R3 on Maxwell (`sm_52`) Hardware

This guide explains how to install, configure, run, and verify the **LlamaMaxwell-35 R3** release for **Qwen3.5/Qwen3.6-35B-A3B (Occamy MoE + MTP1)** on NVIDIA Maxwell (`sm_52`) GPUs such as the Quadro M6000 24 GB.

---

## 1. Hardware and Software Requirements

### GPU
- **Release binaries (`sm_52-real`)**: Compiled specifically for NVIDIA Maxwell **Compute Capability 5.2 (`sm_52`)**, verified on **Quadro M6000 24 GB**.
- **VRAM**:
  - **24 GB VRAM** is required to fit the full 35B-A3B `Q4_0` model (~18.9 GB weights, all 42 layers, all 8 active experts) plus 84K-196K F16 KV cache on GPU (`-ngl 999`).
  - Smaller Maxwell cards (e.g. GTX 980 Ti 6 GB, Titan X Maxwell 12 GB) can run the same `sm_52` kernels with smaller `Q4_0` Qwen3.5/GDN/MoE models or partial layer offload (`-ngl`), though certified R3 benchmarks are for 24 GB full-offload.
- **Other CUDA architectures (`sm_61`, `sm_75`, `sm_86`, etc.)**:
  - The prebuilt release tarball targets `sm_52` only (to keep `libggml-cuda.so` slim at ~14.7 MB with zero register spills).
  - To test on another GPU architecture, rebuild from source with `LLAMAXWELL_CUDA_ARCH=61-real` (or `75-real`, `86-real`, etc.) and `LLAMAXWELL_OCCAMY_SLIM=OFF` via `scripts/build-r3-sm52.sh`.

### Host OS & CUDA Runtime
- **OS**: Linux `x86_64` (built on Ubuntu 24.04 / glibc 2.39, GCC 14).
- **NVIDIA Driver & CUDA 12.x libraries**:
  - Host driver supporting CUDA 12.x (`nvidia-smi` working), plus `libcudart.so.12`, `libcublas.so.12`, `libnccl.so.2`, `libssl.so.3`.
  - If your host lacks CUDA 12.8 runtime libraries, run inside a CUDA 12.8 runtime container such as `nvidia/cuda:12.8.1-runtime-ubuntu24.04` (with `libgomp1` and `libssl3` installed).

---

## 2. Model Requirements

The custom R3 Maxwell kernels are specialized for **Q4_0** weights and **F16** KV cache on Qwen3.5/Qwen3.6 MoE + Gated Delta Net (GDN) + MTP1 architectures:

- **Target model**: `occamy-1.0-Q4_0-headq40b-mtp.gguf` (or compatible Qwen3.5/3.6-35B-A3B `Q4_0` GGUF; `Q4_0` throughout with `headq40b` output/head quantization and optional MTP1 draft layer).
- **Important flags**:
  - `--override-kv tokenizer.ggml.pre=str:qwen35` (for Qwen3.6/Occamy GGUFs using the `qwen35` pre-tokenizer)
  - `-ngl 999` (full GPU offload)
  - `-fa on` (enables Maxwell FlashAttention and sparse/GEMM prefill kernels)
  - `-ub 1024 --mixed-batch-size 1024` (certified safe working memory budget; `-ub 2048` can OOM on 2-slot 24 GB setups)
  - `--ctx-checkpoints 8 --checkpoint-min-step 2048` (prevents forced full-context reprocesses on hybrid GDN/attention cache misses)

Model weights are **not** included in the repository or release archive.

---

## 3. Quickstart: Installing the Release Package

1. Download `llamaxwell-35-r3-linux-x86_64-sm52-cuda12.tar.gz` and `SHA256SUMS` from GitHub Releases.
2. Verify checksums and unpack:
   ```bash
   sha256sum -c SHA256SUMS
   tar -xzf llamaxwell-35-r3-linux-x86_64-sm52-cuda12.tar.gz
   cd llamaxwell-35-r3-linux-x86_64-sm52-cuda12
   ```
3. Verify the internal package files and install (default `/opt/llamaxwell-35`, or pass `--prefix ~/llamaxwell-35`):
   ```bash
   sudo ./install.sh
   # or without root:
   ./install.sh --prefix "$HOME/.local/llamaxwell-35"
   ```
4. Launch the OpenAI-compatible server:
   ```bash
   /opt/llamaxwell-35/llamaxwell-serve.sh \
     --model /path/to/occamy-1.0-Q4_0-headq40b-mtp.gguf \
     --port 8080
   ```
   You can also run directly from the unpacked directory without calling `install.sh`.

### Running Inside Docker (Recommended if host lacks CUDA 12.8 libs)

If your host only has the NVIDIA driver and Docker + NVIDIA Container Toolkit:

```bash
docker run --rm --gpus all \
  -p 127.0.0.1:8080:8080 \
  -v /opt/llamaxwell-35:/opt/llamaxwell-35:ro \
  -v /path/to/models:/models:ro \
  nvidia/cuda:12.8.1-runtime-ubuntu24.04 \
  /opt/llamaxwell-35/llamaxwell-serve.sh \
    --model /models/occamy-1.0-Q4_0-headq40b-mtp.gguf \
    --host 0.0.0.0 \
    --port 8080
```

---

## 4. Operating Modes (Speed vs. Quality Tradeoff)

`llamaxwell-serve.sh` supports three certified R3 profiles via `--mode`:

| Mode | Speculative Decoding | 2K Warm (`fast2k`) | 80K Warm (`warm80k`) | 80K Unscripted (`ordinary80`) | 16K Prefill (`ub=1024`) | IFEval (541) Strict Prompt | BFCL (160) Multi-Turn | Notes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `mtp` (default) | `draft-mtp` (`n=1`, `f16` KV) | 120.08 tok/s | 104.71 tok/s | 96.82 tok/s | 474.53 tok/s | 79.67% (431/541) | 66/160 (ckpt2) / 69/160 (ckpt8) | Certified R3-lossless native MTP1 + `--ctx-checkpoints 8` |
| `serial` | `none` | 87.94 tok/s | 80.88 tok/s | 80.43 tok/s | 488.40 tok/s | 81.52% (441/541) | 71/160 (44.38%) | Zero MTP verification drift; lowest variance |
| `mtp-q8c` | `draft-mtp` + `GDN_T2_Q8C=1` | 121.32 tok/s | 108.79 tok/s | 98.59 tok/s | - | 80.96% (438/541) | partial (49/160 thermal stop) | Opt-in INT8 GDN verify (`GGML_CUDA_OCCAMY_GDN_T2_Q8C=1`) |

Examples:
```bash
# Default (R3-lossless MTP1 + ctx-checkpoints 8)
./llamaxwell-serve.sh --model /models/occamy.gguf

# Zero quality-loss serial mode (no MTP)
./llamaxwell-serve.sh --model /models/occamy.gguf --mode serial

# Maximum speed Q8C mode
./llamaxwell-serve.sh --model /models/occamy.gguf --mode mtp-q8c

# Two parallel slots on a specific GPU index
./llamaxwell-serve.sh --model /models/occamy.gguf --slots 2 --ctx 168448 --gpu 0
```

---

## 5. Verifying Kernel Engagement on Your Hardware

When `llama-server` exits gracefully (e.g. `SIGINT` / `SIGTERM`), the R3 build prints per-subsystem engagement counters to stderr/logs:

- `r3_cuda_graphs: calls=... arch_off=0 incompat=0 captures=... replays=...`
  - Confirms CUDA graphs are active (`arch_off=0` means `sm_52` matched; `incompat=0` means no graph fallback).
- `r3_engaged_moe`, `r3_engaged_fa`, `r3_engaged_gdn`
  - Confirms the Maxwell `sm_52` fused MoE, FlashAttention, and GDN kernels are dispatching.
- `compact_readback` / `MTP draft compact: mode=... compact_only=... fallback=0`
  - Confirms 16-byte compact argmax readback is active on `mtp` mode with zero fallback.

### Important Pitfalls
1. **Never set `GGML_CUDA_OCCAMY_DEBUG=0`**: `ggml-cuda.cu` checks `getenv("GGML_CUDA_OCCAMY_DEBUG") != nullptr`. Even setting it to `"0"` enables verbose debug logging and hurts throughput. Always `unset GGML_CUDA_OCCAMY_DEBUG` (handled automatically by `llamaxwell-serve.sh`).
2. **Keep `--ubatch-size 1024` (`-ub 1024`) for 2x80K+**: Increasing `-ub` to `2048` doubles compute buffer + FA pool pressure. On 24 GB M6000 with `-np 2` (`draft-mtp`), `-ub 2048` passes up to `-c 155648` (`2 x 77824`, `24415 MiB` VRAM peak; `-c 147456` / `2 x 73728` uses `24207 MiB` and is safer) and OOMs at `-c 157696` (`2 x 78848`) in `ggml_cuda_flash_attn_ext`. Keep `-ub 1024` when running `2 x 84224` (`-c 168448`).
3. **Thermal management on Quadro M6000**: Passive/blower M6000 cards can hit 80 C+ under sustained 80K prefill. Monitor `nvidia-smi` temperatures during long benchmark suites.

---

## 6. Rebuilding From Source

On a machine with Docker and `local/m6000-build-tools:20261013` (or an Ubuntu 24.04 + CUDA 12.8 devel image with `gcc-14`, `g++-14`, `cmake`, `libssl-dev`):

```bash
git clone https://github.com/average-user-887/llamaxwell-35.git
cd llamaxwell-35
./scripts/build-r3-sm52.sh --out ./dist-r3-sm52
```

To build for a different CUDA architecture (non-slim):
```bash
./scripts/build-r3-sm52.sh --arch 86-real --slim OFF --out ./dist-sm86
```

To verify the slim `sm_52` output:
```bash
cuobjdump --list-elf ./dist-r3-sm52/libggml-cuda.so | grep -oE 'sm_[0-9]+' | sort -u
# Expected: sm_52
```
