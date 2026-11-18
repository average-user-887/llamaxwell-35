# Four-row R2 attention on Maxwell

The candidate adds the existing R2 selector and attention policy at width four. It is enabled only with `GGML_CUDA_OCCAMY_FA_Q4_R2=1`, the original R2 flags and a matching sm_52 long-context graph. The four-row path uses either one or two unit lanes per query/head group. The original c973 GPU instructions for all 37 existing device functions match exactly, and the original object relink reproduces c973 byte for byte. No model graph or server source changed. The [patch](q4-attention.patch) includes one CUDA translation unit and the operator fixture registration.

The operator harness uses the 81,920-key F16 workload with 16 query heads, two KV heads and head dimension 256. Both four-row layouts must match two independent q2 launches within the declared elementwise tolerance, `2e-5 × (1 + |reference|)`, for every finite output. The receipt counts differing FP32 words separately. Eight per-call row/KV plans and causal ends 81,657 through 81,660 must match the split reference. The original selector scores also require bitwise agreement. The distant-key test confirms the existing sparse policy's behavior for probed and deliberately omitted keys. That sparse policy can lose recall relative to full attention.

Uninstrumented timing first records actual q1 and q2 dispatch, then brackets mirrored runs of the stock full q4 path, q3 R2, and both q4 R2 layouts. A layout passes only if both measured q4 times are at most 1.35 times their respective q3 times and both improve on stock q4. With the previous q3 time near 367 µs, the threshold would be about 495 µs. A technical pass records valid evidence even when every layout misses the speed threshold.

```bash
./release.sh --check
N3_GPU="$N3_GPU" ./release.sh --run
```

The run requires the project GPU lease and fresh admission. Its 180-second cap includes owned cleanup. An optional `Q4_R2_CUTOFF_EPOCH` can shorten that window. No model is loaded or launched by this card. The run prints `Q4_R2_RECEIPT` and saves raw local logs, [compile resources](q4-compile.json), the outcome and a portable JSON receipt with normalized numeric evidence and hashes of the raw logs. It retains a portable receipt on failure.

The scope is the attention operator. Whole-model depth-three performance, physical rejection recovery, stream parity and quality would require a separately authorized matched run.
