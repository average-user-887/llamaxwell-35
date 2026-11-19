# Clock-matched four-row MoE operator repeat

The unchanged four-row CUDA kernel saved **8.179–10.506 µs per layer** against its original two-row MoE control. All eight timed windows had Fleetdash observations. Every sampled SM clock was 1,151 MHz and every sampled memory clock was 3,304 MHz within those windows. Numerical, ordered-route and physical dispatch checks passed. All four mirrors passed the 2.3 µs/layer screen for testing a combination, and all four remained below the 13.1 µs/layer standalone threshold. [Complete decision](result.json) · [Prespecified release](RELEASE.md)

| Experts shared | Pair order | Control µs | Four-row µs | Saved µs | Samples in chronological windows |
| --- | --- | ---: | ---: | ---: | --- |
| 0 | Control → four-row | 163.626528 | 155.447797 | 8.178731 | 3, 3 |
| 0 | Four-row → control | 163.721439 | 154.649259 | 9.072180 | 2, 4 |
| 5 | Four-row → control | 160.226476 | 149.720289 | 10.506187 | 2, 2 |
| 5 | Control → four-row | 160.142346 | 150.616094 | 9.526252 | 3, 2 |

Warmup used 4,096 calls per arm and fixture, followed by 98,304 calls per timed interval. The resulting eight intervals spanned 14.7–16.1 seconds each. Clock medians matched at ratio 1.00 in all four pairs. The clock file preserves the exact SM and memory fields with the original source timestamps. The earlier short-window card had two incomplete clock pairs. [Timing log](perf.log) · [Clock series](fleet-clock.jsonl) · [Original four-row card](../native-q2-down4-20261119/README.md)

On each fixture the original and candidate routes matched both ordered eight-expert lists. All 4,096 finite output words matched, including a second candidate run. The performance process counted 802,816 paired launches and acknowledgments, including 401,408 down4 launches. The candidate and control used the same DSO and graph. The candidate kernel, its CUDA object and the DSO are unchanged from the first card. Runner and cleanup both exited zero in 136 seconds. [Numerical log](check.log) · [Build identities](build.json) · [Terminal receipt](portable-receipt.json)

The conservative 40-layer projection is **0.327149 ms/round**. Adding the independent compact-readback operator's **0.431103 ms** gives **0.758252 ms** against the archived **0.521280 ms** gap. Transfer of either saving to the composite model requires a measured model comparison. [Companion operator](../native-q2-compact-readback-20261119/README.md)

The public delta and exact sources are included. The prior packet supplies the unchanged CUDA source and compiler recipe. Replay the physical measurement, its clock windows and all decisions with the following commands. Restoring the local Fleetdash URL in the sampler is required only for reacquisition or original source-hash verification. [Code changes](long-window.patch) · [Clock and parent pins](observation-pins.json)

```sh
sha256sum --check SHA256SUMS
python3 -B report.py --check
python3 -B report.py --report .
python3 -B verify-layout.py
```
