# Two-row native MoE verifier on Quadro M6000

Two 24-block cohorts computed a full two-row, eight-expert Qwen35B MoE graph in **159.11–162.65 µs**. Running the existing two sequential V3 launches on the identical graph took **191.95–196.66 µs**. All comparisons favored the two-cohort kernel. Its smallest saving was 29.30 µs and its largest was 37.55 µs, below the predeclared 50 µs per-layer threshold.

| Synthetic expert overlap | First paired saving | Reverse paired saving | Forty-layer projection |
|---|---:|---:|---:|
| Zero of eight | 29.849 µs | 29.297 µs | 1.194 / 1.172 ms |
| Five of eight | 37.550 µs | 32.865 µs | 1.502 / 1.315 ms |

The operator passed its technical checks. On each fixture, all 4,096 FP32 outputs matched native N2SPLIT bit for bit, including finite-value checks and bit-identical candidate repeats. Both rows' ordered expert choices matched the independently predicted rankings on both candidate runs. The control's two real V3 launches independently matched the earlier V2 reference. The candidate launched once per two-row graph with 48 physically admitted 512-thread CTAs on 24 SMs. The two 24-member barriers used separate scratch. Both arms submitted the same graph and synchronized after every call. This design avoids a split-and-concatenate timing difference.

The measured native-path counters distinguish both original row launches, the combined launch and the handled second callback. Across both timing fixtures, the original path launched each row 400 times, and the candidate launched 400 times with 400 corresponding second callbacks. Each total includes warmups. The original Phase A Q8 GDN object was retained byte for byte. The candidate and control used the same 147-object link except for the MoE object, with all 16 earlier device functions and 42,664 instruction words preserved. The compiled candidate used 63 registers, no stack and no spills. The standalone operator does not execute a draft model or a recurrent layer.

The complete receipt includes the individual route rows, both full-output comparisons, CUDA occupancy, software identity, all twelve warmup and timed intervals, and successful cleanup. The trial took two seconds under a 180-second cap. The forty-layer estimates extrapolate operator time; model throughput and output quality on real prompts were outside this trial.
