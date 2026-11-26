# Deploying LlamaMaxwell-35 R3 v5 on Maxwell (`sm_52`)

The Linux x86-64 release runs **Occamy 35B-A3B** with native MTP1 on Maxwell. The matched v5 measurements use a Quadro M6000 24 GB. [Benchmark receipts](https://github.com/average-user-887/llamaxwell-35/blob/r3-v5-20261126/benches/llamaxwell-35/r3-v5/README.md)

---

## 1. Hardware and Software Requirements

### GPU
- The release targets Maxwell compute capability 5.2 (`sm_52-real`).
- The qualified full-offload profile uses a 24 GB card with the `Q4_0` model, one 84,224-cell F16 context, and `-ngl 999`. Smaller cards require a smaller model or partial offload.
- For other CUDA architectures, build from source with the target architecture and `--slim OFF`.

### Host OS & CUDA Runtime
- Use a compatible Linux `x86_64` host, with Ubuntu 24.04 (glibc 2.39) as the release build baseline.
- The runtime needs a Maxwell-capable NVIDIA driver and CUDA 12.8 libraries, including `libcuda.so.1`, `libcudart.so.12`, `libcublas.so.12`, `libnccl.so.2`, OpenSSL 3, and `libgomp.so.1`. See the Docker example for a CUDA 12.8 userspace on a host with the NVIDIA Container Toolkit.

---

## 2. Model Requirements

The custom R3 Maxwell kernels are specialized for **Q4_0** weights and **F16** KV cache on Qwen3.5/Qwen3.6 MoE with Gated Delta Net (GDN) and MTP1.

- **Target model** `occamy-1.0-Q4_0-headq40b-mtp.gguf`. The default MTP launcher requires a compatible MTP layer. The benchmark receipts identify the exact tested GGUF by SHA-256.
- **Important flags**
  - `--override-kv tokenizer.ggml.pre=str:qwen35` (for Qwen3.6/Occamy GGUFs using the `qwen35` pre-tokenizer)
  - `-ngl 999` (full GPU offload)
  - `-fa on` (enables Maxwell FlashAttention and sparse/GEMM prefill kernels)
  - `-ub 1024 --mixed-batch-size 1024` (qualified v5 microbatch and mixed-batch sizes)
  - `--ctx-checkpoints 8 --checkpoint-min-step 2048` (prevents forced full-context reprocesses on hybrid GDN/attention cache misses)

Supply your own GGUF.

---

## 3. Installing the Release Package

1. Download `llamaxwell-35-r3-v5-linux-x86_64-sm52-cuda12.tar.gz` and `SHA256SUMS`. [R3 v5 release](https://github.com/average-user-887/llamaxwell-35/releases/tag/r3-v5-20261126)
2. Verify checksums and unpack.
   ```bash
   sha256sum -c SHA256SUMS
   tar -xzf llamaxwell-35-r3-v5-linux-x86_64-sm52-cuda12.tar.gz
   cd llamaxwell-35-r3-v5-linux-x86_64-sm52-cuda12
   ```
3. Install at the user prefix. The installer verifies the internal package manifest. For a system installation at `/opt/llamaxwell-35`, run `sudo ./install.sh`.
   ```bash
   ./install.sh --prefix "$HOME/.local/llamaxwell-35"
   ```
4. Launch the OpenAI-compatible server on the default loopback address and port.
   ```bash
   "$HOME/.local/llamaxwell-35/llamaxwell-serve.sh" \
     --model /path/to/occamy-1.0-Q4_0-headq40b-mtp.gguf
   ```
   You can also run `./llamaxwell-serve.sh` from the unpacked directory. After model startup, verify readiness with `curl --fail --silent http://127.0.0.1:8080/health`.

### Running Inside Docker

Use Docker and the NVIDIA Container Toolkit with a CUDA 12.8 runtime image. Add the Ubuntu runtime packages to the base image before launching the server.

```bash
docker build -t llamaxwell-r3-v5-runtime - <<'EOF'
FROM nvidia/cuda:12.8.1-runtime-ubuntu24.04
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 libssl3t64 \
    && rm -rf /var/lib/apt/lists/*
EOF

GPU_UUID=GPU-your-device-uuid
docker run --rm --gpus "device=$GPU_UUID" \
  -p 127.0.0.1:8080:8080 \
  -v "$HOME/.local/llamaxwell-35:/opt/llamaxwell-35:ro" \
  -v /path/to/models:/models:ro \
  llamaxwell-r3-v5-runtime \
  /opt/llamaxwell-35/llamaxwell-serve.sh \
    --model /models/occamy-1.0-Q4_0-headq40b-mtp.gguf \
    --host 0.0.0.0 \
    --port 8080
```

---

## 4. Operating Modes and Rollback

`llamaxwell-serve.sh` provides `mtp` (default v5 native MTP1), `serial` (speculation disabled), and `mtp-q8c` (opt-in Q8C verification). The v5 measurements use `mtp` with one slot, 84,224 context cells, FP16 KV, `-ub 1024`, eight context checkpoints, and one maximum draft token. The launcher fixes `GGML_CUDA_MULTI_GRAPH` and `GGML_CUDA_FA_MAXWELL_GEMM_V5` at `0` in this mode.

Each v5 switch accepts `0` as an override. Setting `GGML_CUDA_OCCAMY_FA_Q2_R2_MIDSTRIDE=0` also prevents mid-context top-16 dispatch because the selector requires the fixed-stride path. That path uses padded KV widths of 16,640–19,456 and 40,192–43,264 in 256-key steps, with the route's existing model, stride, and topology guards.

| Switch | Effect |
| --- | --- |
| `LLAMA_DENSE_KQ_MASK` | Dense KQ mask fast path |
| `LLAMA_KV_TAIL_TRIM` | Indexed KV tail removal |
| `GGML_CUDA_GATE_Q8X2` | Combined gate and Q8 production |
| `GGML_CUDA_GDN_SMEM_PREFILL_TILE8` | Shared-memory GDN prefill tiles |
| `GGML_CUDA_MMQ_TAIL_J48_SM52` | 48-column work for eligible compact 64-column MoE tail tiles |
| `GGML_CUDA_OCCAMY_FA_Q2_R2_MIDSTRIDE` | Mid-context fixed-stride attention |
| `GGML_CUDA_OCCAMY_FA_Q2_R2_MID_TOP16` | Mid-context top-16 selector with fixed-stride attention |

### Earlier R3 results (2026-11-25)

These figures belong to the original R3 workloads and quality evaluations. The matched v5 benchmark receipts report output parity on the specified fixtures. [Original R3 release](https://github.com/average-user-887/llamaxwell-35/releases/tag/prod-r3-20261125) · [R3 v5 receipts](https://github.com/average-user-887/llamaxwell-35/blob/r3-v5-20261126/benches/llamaxwell-35/r3-v5/README.md)

| Mode | Speculative Decoding | 2K Warm (`fast2k`) | 80K Warm (`warm80k`) | 80K Unscripted (`ordinary80`) | 16K Prefill (`ub=1024`) | IFEval (541) Strict Prompt | BFCL (160) Multi-Turn | Notes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `mtp` | `draft-mtp` (`n=1`, `f16` KV) | 120.08 tok/s | 104.71 tok/s | 96.82 tok/s | 474.53 tok/s | 79.67% (431/541) | 66/160 (ckpt2) / 69/160 (ckpt8) | Original R3 MTP1 and eight context checkpoints |
| `serial` | `none` | 87.94 tok/s | 80.88 tok/s | 80.43 tok/s | 488.40 tok/s | 81.52% (441/541) | 71/160 (44.38%) | Speculation disabled |
| `mtp-q8c` | `draft-mtp` + `GDN_T2_Q8C=1` | 121.32 tok/s | 108.79 tok/s | 98.59 tok/s | - | 80.96% (438/541) | partial (thermal stop) | Opt-in INT8 GDN verify (`GGML_CUDA_OCCAMY_GDN_T2_Q8C=1`) |

### Launch examples

```bash
# Default v5 MTP1
./llamaxwell-serve.sh --model /models/occamy.gguf

# Selector rollback with the same v5 binaries
GGML_CUDA_OCCAMY_FA_Q2_R2_MID_TOP16=0 ./llamaxwell-serve.sh --model /models/occamy.gguf

# Legacy serial mode
./llamaxwell-serve.sh --model /models/occamy.gguf --mode serial

# Opt-in Q8C mode
./llamaxwell-serve.sh --model /models/occamy.gguf --mode mtp-q8c

# Two parallel slots on a specific GPU index
./llamaxwell-serve.sh --model /models/occamy.gguf --slots 2 --ctx 168448 --gpu 0
```

---

## 5. Verifying Kernel Engagement on Your Hardware

When `llama-server` exits gracefully (e.g. `SIGINT` or `SIGTERM`), the R3 build prints engagement counters to stderr.

- `r3_cuda_graphs: calls=... arch_off=0 incompat=0 captures=... replays=...`
  - Confirms CUDA graphs are active (`arch_off=0` means `sm_52` matched; `incompat=0` means no graph fallback).
- `r3_engaged_moe`, `r3_engaged_fa`, `r3_engaged_gdn`
  - Confirms the Maxwell `sm_52` fused MoE, FlashAttention, and GDN kernels are dispatching.
- `compact_readback` / `MTP draft compact: mode=... compact_only=... fallback=0`
  - Confirms 16-byte compact argmax readback is active on `mtp` mode with zero fallback.
- `q2_r2_stride_dispatch ... midstride=1` and `q2_r2_mid_top16_dispatch`
  - During generation, these messages identify the first eligible fixed-stride and top-16 dispatches. They include the padded width and K/V token pitch.

### Important Pitfalls
1. **Never set `GGML_CUDA_OCCAMY_DEBUG=0`**: `ggml-cuda.cu` checks `getenv("GGML_CUDA_OCCAMY_DEBUG") != nullptr`. Even setting it to `"0"` enables verbose debug logging and hurts throughput. Always `unset GGML_CUDA_OCCAMY_DEBUG` (handled automatically by `llamaxwell-serve.sh`).
2. **Keep `--ubatch-size 1024` (`-ub 1024`) for 2x80K+**: Increasing `-ub` to `2048` doubles compute buffer + FA pool pressure. On 24 GB M6000 with `-np 2` (`draft-mtp`), `-ub 2048` passes up to `-c 155648` (`2 x 77824`, `24415 MiB` VRAM peak; `-c 147456` / `2 x 73728` uses `24207 MiB` and is safer) and OOMs at `-c 157696` (`2 x 78848`) in `ggml_cuda_flash_attn_ext`. Keep `-ub 1024` when running `2 x 84224` (`-c 168448`).
3. **Thermal management on Quadro M6000.** Monitor GPU temperatures during long benchmark suites.

---

## 6. Rebuilding From Source

The release package combines the original R3 `llama-server` and companion libraries with the exact `libllama.so.0.6.0` and `libggml-cuda.so` used in the matched v5 runs. The server still reports its embedded build metadata, `51 (6694b50)`. Use the release tag and checksums to identify the complete v5 package. The current source supports a clean source build, which has its own binary identity. [Benchmark receipts](https://github.com/average-user-887/llamaxwell-35/blob/r3-v5-20261126/benches/llamaxwell-35/r3-v5/README.md)

Run the source build with Docker and an Ubuntu 24.04 CUDA 12.8 development image containing GCC and G++ 14, CMake, Make, Git, and OpenSSL headers. Replace the image name with your local build image.

```bash
git clone https://github.com/average-user-887/llamaxwell-35.git
cd llamaxwell-35
git checkout r3-v5-20261126
./scripts/build-r3-sm52.sh --image llamaxwell-cuda12.8-devel --out ./dist-r3-sm52
```

Build for a different CUDA architecture with slim mode disabled.
```bash
./scripts/build-r3-sm52.sh --image llamaxwell-cuda12.8-devel --arch 86-real --slim OFF --out ./dist-sm86
```

Verify the slim `sm_52` output with `cuobjdump` from the CUDA toolkit.
```bash
cuobjdump --list-elf ./dist-r3-sm52/libggml-cuda.so | grep -oE 'sm_[0-9]+' | sort -u
# Expected: sm_52
```
