# llamaxwell-35

Maxwell (`sm_52`) CUDA kernels and `llama.cpp` runtime for **Occamy 35B-A3B (`Qwen3.6-35B-A3B` / `Qwen3.5-MoE` + MTP1)** on the NVIDIA Quadro M6000 24 GB, based on `ggml-org/llama.cpp` `b11552`.

Models and weights are **not** included.

---

## Highlights (R3 Certified Release)

- **Slim `sm_52`-only CUDA backend (`libggml-cuda.so`)**: ~14.7 MB (`-61.7%` vs unslimmed `b11552` builds), `0` register spills on critical Occamy kernels, `$ORIGIN` RPATH for portable deployment.
- **Custom Maxwell `sm_52` kernels**:
  - Fused Qwen3.5/3.6 MoE decode (`occamy-fused.cu`, `moe_v3`, XMAD MMQ/MMVQ)
  - Fused Gated Delta Net (`occamy-gdn.cu`, 1-row and 2-row verify, state direct, QK reuse, phase-C share)
  - Maxwell FlashAttention decode (`fattn-occamy.cu`, head-split pipe=3, sparse probe, 2-row verify) and GEMM prefill (`fattn-maxwell-gemm.cuh`)
  - Compact 16-byte argmax readback (`compact-cuda.cu`) + CUDA graphs (`GGML_CUDA_MAXWELL_GRAPHS=1`)
- **R3 fixes**:
  - Rate-limited `occamy-sf2` debug logging (T1)
  - Chunked MTP draft sub-batching at `llama_n_batch(ctx_dft)` (T2, fixes 2050-token overflow)
  - Mid-stream and end-of-stream SSE UTF-8 flush fix (T3)
  - Draft last-query compact argmax + D1b 2-slot CUDA graph safety guard

## Certified Performance & Quality (`Quadro M6000 24 GB`)

| Profile | 2K Warm (`fast2k`) | 80K Warm (`warm80k`) | 80K Unscripted (`ordinary80`) | 16K Prefill (`ub=1024`) | IFEval (541) Strict Prompt | BFCL (160) Multi-Turn |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `mtp` (default: R3-lossless MTP1 + `ctx-checkpoints 8`) | 120.08 tok/s | 104.71 tok/s | 96.82 tok/s | 474.53 tok/s | 79.67% (431/541) | 66/160 (ckpt2) / 69/160 (ckpt8) |
| `serial` (`--spec-type none`) | 87.94 tok/s | 80.88 tok/s | 80.43 tok/s | 488.40 tok/s | 81.52% (441/541) | 71/160 (44.38%) |
| `mtp-q8c` (`GGML_CUDA_OCCAMY_GDN_T2_Q8C=1`) | 121.32 tok/s | 108.79 tok/s | 98.59 tok/s | - | 80.96% (438/541) | partial (thermal stop) |

2-slot (`-np 2`) `-ub 2048` usable context ceiling on 24 GB VRAM: max passing `-c 155648` (`2 x 77824`, `24415 MiB` peak); `-c 147456` (`2 x 73728`, `24207 MiB` peak) recommended for margin (`-c 157696` OOMs). Keep `-ub 1024` for `2 x 84224`.

## Quickstart

Download the release tarball from [GitHub Releases](https://github.com/average-user-887/llamaxwell-35/releases), verify `SHA256SUMS`, install, and run:

```bash
sha256sum -c SHA256SUMS
tar -xzf llamaxwell-35-r3-linux-x86_64-sm52-cuda12.tar.gz
cd llamaxwell-35-r3-linux-x86_64-sm52-cuda12
sudo ./install.sh
/opt/llamaxwell-35/llamaxwell-serve.sh --model /path/to/occamy-1.0-Q4_0-headq40b-mtp.gguf
```

Full deployment, Docker usage, kernel verification, and rebuild instructions: [`docs/DEPLOY.md`](docs/DEPLOY.md).

## Repository Branches

- `main`: Clean production R3 source, build scripts (`scripts/build-r3-sm52.sh`), installer/launcher (`packaging/`), and deployment docs (`docs/DEPLOY.md`).
- `receipts/r3-20261125`: Sanitized R3 benchmark, IFEval, BFCL, and QA prefill evidence packets and checksums.
- `experimental/speed-cards-20261122`: Sanitized experimental scheduler/tuning patches (C2/C3, not adopted into `main`).
