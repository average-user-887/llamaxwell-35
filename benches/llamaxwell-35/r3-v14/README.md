# LlamaXwell-35 R3 v14 serving benchmarks

The timing comparisons use an NVIDIA Quadro M6000 24 GiB (`sm_52`) and CUDA 12.8. The model is Occamy 1.0 Q4_0 headq40b MTP with one 84,224-cell slot, batch size 2,048, microbatch size 1,024, MTP draft depth 1, FP16 draft K/V, and greedy sampling with seed 7. [Machine-readable receipts](receipts.json) include the exact source and CUDA library pins, serving metrics, clock observations and parity fields. [Released v5 baseline](../r3-v5/README.md)

## Released v5 to v12 at 65,536 tokens

A fresh prompt and two cache replays ran on both profiles. The first request emitted 192 tokens and each warm request emitted 1,024 tokens. All 2,240 corresponding output token IDs per arm matched, along with text, cache counts, prompt counts, MTP proposals and acceptances, finish reason and truncation. Median P0 graphics and memory clocks matched in the cold prefill window (1,113/3,304 MHz) and warm decode windows (1,151/3,304 MHz).

| Phase | Metric | Released v5 | v12 | Change |
| --- | --- | ---: | ---: | ---: |
| Cold | Prefill | 331.78 tok/s | 339.85 tok/s | +2.43% |
| Cold | Request wall time | 199.57 s | 194.82 s | −2.38% |
| Warm | Generation | 99.78 tok/s | 102.83 tok/s | +3.06% |
| Warm repeat | Generation | 99.95 tok/s | 102.97 tok/s | +3.02% |

## Probe4 at 56,000 and 78,000 tokens

The two probe4 comparisons toggle four-block scan on a shared v12-based CUDA library. The control runs first at 56,000 tokens and last at 78,000 tokens. Each workload matches all 2,240 output token IDs per arm, streamed text, cache counts and MTP counters. Warm windows have matching median P0 graphics and memory clocks of 1,151/3,304 MHz, with two to four samples per arm. At 78,000 tokens, cold prefill fell 0.07% and warm time to first token increased by 14–22 ms as total warm request time fell 0.35–0.52%.

| Input tokens | Warm request | Decode with probe4 off | Decode with probe4 on | Change |
| --- | --- | ---: | ---: | ---: |
| 56,000 | First | 102.89 tok/s | 103.35 tok/s | +0.45% |
| 56,000 | Repeat | 102.87 tok/s | 103.41 tok/s | +0.53% |
| 78,000 | First | 103.04 tok/s | 103.74 tok/s | +0.68% |
| 78,000 | Repeat | 103.15 tok/s | 103.75 tok/s | +0.58% |

The timed library has a wider eligibility predicate than the final v13 source inherited by v14. The final v13 guard restricts probe4 to aligned padded widths 56,064–79,872 on the eligible Q2 R2 extended top16 path. The final library preserves all 76 registered CUDA payloads and the dynamic ABI. A replay of 400,388 route cases confirmed identical dispatch throughout the measured width sets.

## Integrated v13 to v14 at 2,048 tokens

Four fresh server instances ran in the order v13, v14, v14, v13. Each emitted eight cold tokens and 1,536 warm tokens. All 1,544 IDs per arm and the associated text, prompt and cache counts, MTP counts, finish reason and truncation matched across the arms. The v14 runs witnessed the new single-column vocabulary head dispatch. CUDA graph engagement was verified in all arms. Both warm pairs had matching median P0 clocks at 1,151/3,304 MHz, with three to four samples per arm.

| Run order | v13 warm decode | v14 warm decode | Decode gain | Wall time saved | Warm TTFT increase |
| --- | ---: | ---: | ---: | ---: | ---: |
| v13 → v14 | 122.58 tok/s | 122.69 tok/s | +0.09% | 0.06% | 4.97 ms |
| v14 → v13 | 122.43 tok/s | 122.86 tok/s | +0.35% | 0.34% | 0.36 ms |

## Sampled v5 to v14 at 8,192 tokens

One cold request per arm used seed 7, temperature 0.8 and pre-sampling `n_probs=8`. All 64 generated token IDs matched, along with decoded text, cache and stop fields. Each arm proposed 34 MTP tokens and accepted 29. The first returned score row matched the order of all eight ranked tokens and all nine serialized binary32 logprobs, including the sampled token's value. Both arms used the sampler fallback with zero target compact readbacks. The result covers one prompt and its first returned score row. [Sampled parity receipt](sampled-fallback.json)

## Activate the measured profile

Keep the released v5 MTP configuration and apply the nine additions recorded in the receipt. With the pinned v14 CUDA library and pinned server and host library already selected in a separate runtime directory, the released launcher accepts these extra environment values. Run this Bash example from the receipt directory and set `RUNTIME_DIR` and `MODEL_GGUF` to your copies.

```bash
mapfile -t profile_env < <(
  jq -r '(.profile.added_v12_over_v5 + .profile.added_v13_over_v12
           + .profile.added_v14_over_v13)
         | to_entries[] | "\(.key)=\(.value)"' receipts.json
)
env "${profile_env[@]}" "$RUNTIME_DIR/llamaxwell-serve.sh" --mode mtp \
  --model "$MODEL_GGUF" --slots 1 --ctx 84224
```

The reported gains use distinct baselines and workloads, so combining their percentages would not describe a measured speedup. The serving measurements cover single-slot greedy replays. The separate sampled probe covers one fallback request per arm.
