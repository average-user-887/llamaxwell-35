# Softmax-only variants of packed-GQA GEMM attention v2

Built and frozen 2026-11-12, 15:49 CET. The root investigator subsequently ran both GPU variants; independent receipt analysis is in `GPU_RESULTS.md`. This worker used no GPU/model/service calls, and the source/binaries remain unchanged.

## Delivered variants

| Directory | Algorithm | Score-buffer traffic | Registers/thread |
|---|---|---|---:|
| `recompute` | Read maximum, read sum of exponentials without storing them, then reread scores and write normalized probabilities | Four read/write passes versus v2's five | 13 |
| `online` | One read with stable per-thread running maximum/sum; block-merge maximum and rescaled sums; reread and write normalized probabilities | Three read/write passes | 14 |

Both use the original 256-thread block and 64-byte shared reduction storage, with zero softmax spills reported by ptxas. Traffic counts distinguish a read from a write; the variants have three and two source loops, respectively. Extra exponential computation or online dependencies can offset saved memory traffic, so speed remains a measurement question.

Only `causal_softmax` and one JSON identity print differ from v2. Q/K/V layout, FP32 math, GQA packing, cuBLAS settings, allocations, masks, reference launch, ABBA order, and whole-algorithm timers are unchanged. Every causal key through `last_key` contributes to normalization and every later output position is explicitly zeroed. There is no context shortening, valid-key sampling, or extra workspace.

The prototype's original contiguous causal-prefix mask assumption is retained. Model integration still has to derive valid/padded/causal positions from the real model mask rather than assuming every physical KV slot is valid. These binaries do not claim arbitrary-mask coverage.

## Invocation for the root-owned GPU session

Mount this `softmax` directory read-only at `/w`, use workdir `/w`, and set `LD_LIBRARY_PATH=/w/lib` in the same CUDA 12.8 build image used for v2. Retain the root's ownership, GPU isolation and time/resource admission controls.

Commands inside that admitted container:

```sh
timeout 180 /w/recompute/gemm-attention-probe 1024 98304 2 64 4096
timeout 180 /w/online/gemm-attention-probe 1024 98304 2 64 4096
```

An optional sixth argument changes the nonzero seed. The original finite/max-absolute/RMSE comparator runs before timed repetitions and exits 4 if nonfinite values occur, maximum absolute error exceeds `1e-4`, or RMSE exceeds `1e-5`. Do not treat CPU validation as satisfying this GPU gate. Report both `softmax_ms` and complete candidate `milliseconds`, not only isolated kernel time. Complete timing includes KV conversion, query packing, QK, softmax, PV, split reduction and submission gaps; allocation/setup/input transfers remain excluded exactly as in v2.

## CPU verification and provenance

- Both binaries build/link and reject invalid dimensions before GPU access.
- Frozen v2 headers and `libggml-base` are included under `reference` and `lib`; no concurrent integration source is consulted by the build.
- `cuobjdump` reference-kernel SASS is byte-identical to the v2 binary in both variants. Reference SASS SHA256: `f82526f6eed48cda39dad81cd73c369bd592a133a4a365a622df4c972c67e4cd`.
- Supplemental FP32 CPU formula checks include 98K keys, partial prefixes, three valid keys, ascending logits and NaN sentinel padding. All probabilities are finite, masked outputs are zero, normalization holds, and online-versus-standard maximum error is at most `2.98e-8` in those cases. This is not a CUDA numerical result.
- `cpu-validation.json`, `cpu-link-guard.log`, `build.log`, per-variant `from-v2.patch`, and `SHA256SUMS` retain receipts.
- Build uses one sequential compiler, CPUs 0-3 with quota 2, 6 GiB memory/no swap, no network and no GPU. No build container remains active.

## Binary hashes

Recompute: `6a1edf0754516314ae21c136cc22a65f79a27c63f94a1e336397ed2dff255629`

Online: `8a3af467bdffc6b3e74ba7f75b37583657c14a5b873eac79b3028217abe9b1f8`

No existing v2, prefill integration, production or scheduler-runtime file was changed. Rollback is simply not selecting these isolated executables.
