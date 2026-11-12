# Packed-loader experiment v2

The original `q8-attention-micro` and its sources remain unchanged. This separate artifact tests an aligned 32-bit native-Q8 loader, retaining the 34-byte quantized block layout and the same attention arithmetic. Three aligned words cover each lane's eight codes; funnel shifts handle the alternating two-byte alignment. The final block in a 256-dimensional row is odd-aligned and does not read past the row.

The driver adds at least 1,000 ms of alternating GPU work **after CPU reference validation** and alternates ABBA/BAAB block ordering. It samples NVML SM/memory clocks, temperature, power and utilization before and after each timed arm, outside CUDA events. NVML attaches by the selected CUDA device's PCI bus ID. Unavailable telemetry is null/unavailable, never inferred. Utilization fields are NVML interval samples, not attribution to an individual kernel.

The original README overstated the sampled-reference count: the implementation checks **32 output positions per stream** (four heads times eight dimensions), for each of F16 and Q8. All output elements are separately checked for finiteness and F16/Q8 deviation. The JSON count is correct in both versions.

Build and CPU-only check:

```sh
bash build.sh
./q8-attention-packed --cpu-check
```

Root-owned GPU smoke and deep cases:

```sh
CUDA_VISIBLE_DEVICES=<GPU_UUID> timeout 180 ./q8-attention-packed 4096 1 5 50 1
CUDA_VISIBLE_DEVICES=<GPU_UUID> timeout 180 ./q8-attention-packed 98304 1 5 100 1
CUDA_VISIBLE_DEVICES=<GPU_UUID> timeout 180 ./q8-attention-packed 98304 2 5 100 4
```

Do not infer an integrated model gain from this synthetic kernel result. No integrated cache-writing cost, prompt-processing cost, actual trained-model K/V distribution, agent task quality or concurrent scheduler behavior is tested here. The original identical-data, selected-shape and CPU-reference limitations still apply.
