# llamaxwell-35

Maxwell (`sm_52`) CUDA kernels and `llama.cpp` runtime for **Occamy 35B-A3B (`Qwen3.6-35B-A3B` / `Qwen3.5-MoE` + MTP1)** on the NVIDIA Quadro M6000 24 GB, based on `ggml-org/llama.cpp` `b11552`.

Bring your own compatible GGUF model.

---

## R3 v5

- The default `mtp` launcher uses native MTP1, one slot, 84,224 context cells, F16 cache, and `-ub 1024`. The Linux x86-64 package uses CUDA 12.8 and ships the exact host and CUDA libraries used in the matched v5 benchmarks.
- Seven switches enable dense KQ masks, indexed KV tail trimming, Gate Q8X2, GDN TILE8, 48-column MoE tail tiles, fixed-stride sparse attention, and top-16 selection. Each accepts `0` as a rollback. The attention changes use two guarded padded-width bands of 16,640–19,456 and 40,192–43,264.
- **Custom Maxwell `sm_52` kernels**
  - Fused Qwen3.5/3.6 MoE decode (`occamy-fused.cu`, `moe_v3`, XMAD MMQ/MMVQ)
  - Fused Gated Delta Net (`occamy-gdn.cu`, 1-row and 2-row verify, state direct, QK reuse, phase-C share)
  - Maxwell FlashAttention decode (`fattn-occamy.cu`, head-split pipe=3, sparse probe, 2-row verify) and GEMM prefill (`fattn-maxwell-gemm.cuh`)
  - Compact 16-byte argmax readback (`compact-cuda.cu`) + CUDA graphs (`GGML_CUDA_MAXWELL_GRAPHS=1`)
- **R3 fixes**
  - Rate-limited `occamy-sf2` debug logging (T1)
  - Chunked MTP draft sub-batching at `llama_n_batch(ctx_dft)` (T2, fixes 2050-token overflow)
  - Mid-stream and end-of-stream SSE UTF-8 flush fix (T3)
  - Draft last-query compact argmax + D1b 2-slot CUDA graph safety guard

On the Quadro M6000 24 GB, the v5 selector raised 16K warm decode from 108.16 / 108.12 to **110.27 / 110.23 tok/s** and 40K from 107.20 / 107.21 to **109.55 / 109.54 tok/s**. The two matched comparisons toggled the selector against the v4 configuration and preserved all 6,336 generated token IDs, streamed text, cache counts, and MTP counts per arm. A separate 16,512-token held-out prompt matched 768 output IDs per arm. [R3 v5 benchmark receipts](https://github.com/average-user-887/llamaxwell-35/blob/r3-v5-20261126/benches/llamaxwell-35/r3-v5/README.md)

## Earlier R3 Performance and Quality (2026-11-25)

The original R3 release reported the following figures on the Quadro M6000 24 GB. These historical workloads and quality evaluations use their original artifact and profile settings. [Original R3 release](https://github.com/average-user-887/llamaxwell-35/releases/tag/prod-r3-20261125)

| Profile | 2K Warm (`fast2k`) | 80K Warm (`warm80k`) | 80K Unscripted (`ordinary80`) | 16K Prefill (`ub=1024`) | IFEval (541) Strict Prompt | BFCL (160) Multi-Turn |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `mtp` (original R3 MTP1, `ctx-checkpoints 8`) | 120.08 tok/s | 104.71 tok/s | 96.82 tok/s | 474.53 tok/s | 79.67% (431/541) | 66/160 (ckpt2) / 69/160 (ckpt8) |
| `serial` (`--spec-type none`) | 87.94 tok/s | 80.88 tok/s | 80.43 tok/s | 488.40 tok/s | 81.52% (441/541) | 71/160 (44.38%) |
| `mtp-q8c` (`GGML_CUDA_OCCAMY_GDN_T2_Q8C=1`) | 121.32 tok/s | 108.79 tok/s | 98.59 tok/s | - | 80.96% (438/541) | partial (thermal stop) |

The original R3 two-slot test at `-ub 2048` passed up to `-c 155648` on 24 GB VRAM, with 24,415 MiB peak usage. At `-c 147456` it used 24,207 MiB. At `-c 157696` it ran out of memory. Use `-ub 1024` for `2 x 84224`.

## Quickstart

Download the archive and `SHA256SUMS` into the same directory. The commands below install as your current user and bind the server to `127.0.0.1:8080`. [R3 v5 release](https://github.com/average-user-887/llamaxwell-35/releases/tag/r3-v5-20261126)

```bash
sha256sum -c SHA256SUMS
tar -xzf llamaxwell-35-r3-v5-linux-x86_64-sm52-cuda12.tar.gz
cd llamaxwell-35-r3-v5-linux-x86_64-sm52-cuda12
./install.sh --prefix "$HOME/.local/llamaxwell-35"
"$HOME/.local/llamaxwell-35/llamaxwell-serve.sh" --model /path/to/occamy-1.0-Q4_0-headq40b-mtp.gguf
```

See the deployment guide for runtime dependencies, Docker, kernel verification, source builds, and rollback. [Deployment guide](https://github.com/average-user-887/llamaxwell-35/blob/r3-v5-20261126/docs/DEPLOY.md)

## Repository Branches

- `main` carries R3 v5 source, the installer and launcher, and benchmark receipts.
- `receipts/r3-20261125` carries original R3 benchmark, IFEval, BFCL, and QA prefill evidence.
- `experimental/speed-cards-20261122` carries experimental C2 and C3 scheduler patches.
