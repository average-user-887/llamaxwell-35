# Cache-only MTP catch-up on Occamy 35B

On a Quadro M6000, enabling cache-only MTP catch-up reduced the 80,000-token cold request from **284.547 s to 269.872 s**. Time to first token improved by **14.678 s**. All 192 greedy output IDs matched across both flag settings and both cache conditions at each context length.

| Prompt and cache condition | TTFT off → on (s) | Request wall off → on (s) | Decode off → on (tok/s) | Draft accepts per arm |
|---|---:|---:|---:|---:|
| 2K cold | 4.131 → 3.970 | 6.203 → 6.031 | 92.283 → 92.684 | 88/102 |
| 2K warm | 0.084 → 0.084 | 2.129 → 2.137 | 93.373 → 93.033 | 88/102 |
| 80K cold | 281.579 → 266.901 | 284.547 → 269.872 | 64.445 → 64.588 | 91/100 |
| 80K warm | 0.256 → 0.265 | 3.179 → 3.182 | 65.463 → 65.565 | 92/99 |

Each context length used one disabled/enabled pair, each arm with a fresh server, one cold request and one warm replay on the same slot. Both arms used the same full-F16 attention stack, MTP1 model, runtime image and two-library overlay. The only environment variable that differed was `LLAMA_MTP_CACHE_ONLY`. Cold requests reused zero prompt tokens. Warm requests reused 1,996 tokens at 2K and 79,996 at 80K in both arms. Both arms offloaded 42/42 layers. The cache-only marker appeared only in the enabled arm. The 5.16% reduction is a cold-latency result for this 80K fixture. Decode throughput was close across arms and warm wall time differed by under 8 ms at each context. The exactness checks cover the two greedy fixtures. [Measurement receipt](measurements.json)

## Behavior

Set `LLAMA_MTP_CACHE_ONLY=1` to enable. Leave it unset or set it to `0` to use ordinary decode. The MTP driver requests a separate cache-only graph for a wholly outputless catch-up batch on the supported single-block Qwen35MoE draft context. The graph computes the inputs and K/V projections needed for stock cache writes. Ordinary graphs handle output requests and other shapes. A successful cache-only dispatch logs `mtp_cache_only: path=cache` with `outputs=0`. A successful live output dispatch logs `path=full`.

## Reconstruct the measured source

Use a repository checkout containing the published source reconstruction. Set `release_root` to the checkout containing the two package directories, `source_dir` to a new directory and `build_dir` to a separate build directory. [Published source reconstruction](https://github.com/average-user-887/llamaxwell-35/commit/a23f943208ccdb5c2eb6ae5e4e9cd0f0009aff7f)

```sh
git worktree add --detach "$source_dir" e4fdc23555f6a9103ea03b854ee5c70c6812517c
bash "$release_root/sparse-r2-20261117/reconstruct-source.sh" "$source_dir" base
patch="$release_root/cache-only-20261118/cache-only.patch"
git -C "$source_dir" apply --index --check "$patch"
git -C "$source_dir" apply --index "$patch"
```

The published `base` recipe reconstructs the original FA2 build-source content from `da7ba0e49dd43c69f9ee5a219ea3124f7acda0c8`. The included patch is byte-identical to the experiment patch. Forward and reverse application passed, and all six patched files matched the measured checkout byte for byte. [Source and binary provenance](provenance.json)

The measured overlay rebuilt only the `llama` and `llama-common` libraries in the existing Release configuration with GCC 14.2.0, CMake 3.28.3 and a recorded CUDA 12.8.93 toolchain for `sm_52`. The initial CMake refresh failed its CUDA host compiler check. The successful incremental retry used the existing generated Makefiles for the two C++ targets, and the resulting overlay passed runtime linkage checks. It reused the FA2 CUDA and server binaries. For a fresh build, repeat the published Maxwell CMake settings with a compatible CUDA host toolchain. This fresh build was untested during package verification.

```sh
cmake -S "$source_dir" -B "$build_dir" -G "Unix Makefiles" \
  -C "$release_root/sparse-r2-20261117/maxwell-release.cmake"
cmake --build "$build_dir" --config Release --parallel 2 --target llama-server
```
