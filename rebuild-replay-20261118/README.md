# Live replay from the published Sparse R2 and cache-only MTP build

A fresh build of the published stack reproduced all 192 output token IDs and the draft counts for each cold and warm request at 2K and 80K. Both live runs used a Quadro M6000, the real NVIDIA CUDA driver, and binaries built from the public source recipe.

| Fixture | Prior overlay wall (s) | Fresh build wall (s) | Prior server decode (tok/s) | Fresh server decode (tok/s) | Cached prompt tokens |
|---|---:|---:|---:|---:|---:|
| 2K cold | 6.0310 | 6.0491 | 92.6838 | 92.8580 | 0 |
| 2K warm | 2.1372 | 2.1318 | 93.0328 | 93.3485 | 1,996 |
| 80K cold | 268.6345 | 268.4981 | 79.2945 | 79.0293 | 0 |
| 80K warm | 2.7287 | 2.7259 | 77.4747 | 77.4175 | 79,996 |

The 2K reference is the earlier full-attention cache-only MTP1 arm. The 80K reference is the earlier Sparse R2 plus cache-only MTP1 arm. Each fixture had one fresh cold request and one same-slot warm replay. At 80K both builds accepted 90/100 drafts cold and 87/103 warm. Both columns use server `predicted_per_second`. Counting all emitted tokens over the server's decode duration gives 79.7096 tok/s for the 80K cold reference and 79.4430 tok/s for the fresh build. The replay establishes matching outputs and close timings on these fixtures. [Allowlisted live and build receipt](evidence.json)

## Live binary and dispatch checks

The receipt matches the executing server and eight mapped libraries to the fresh files by device, inode, and SHA-256. The fresh hashes differ from those of the prior measured overlay. Both runs mapped the real NVIDIA 580.178.04 driver, offloaded 42/42 GPU layers, and logged cache-only catch-up and ordinary MTP output. The 80K cold and warm requests each logged long-context Sparse R2 query-width-2 dispatch. The 2K fixture produced the expected short-context result without long sparse dispatch. Markers establish observed execution. Total dispatch coverage remains unmeasured.

## Build and source

The source came from the published stack recipe at `3eea327`, with a verified combined source index digest. The build used its Maxwell CMake preset, Release, Unix Makefiles, GCC 14.2.0, CUDA 12.8.93, CMake 3.28.3, and `52-real`. It produced `llama-server`, `test-backend-ops`, and the CUDA backend with 147 native `sm_52` images. The optional web UI assets were unavailable during the offline build, so the server has no embedded UI. [Pinned reconstruction recipe](https://github.com/average-user-887/llamaxwell-35/blob/3eea327fb4c289a54a8796899080bbbd1a3cdcfd/stacked-mtp-20261118/README.md)

The earlier validation at `b102b95` recorded a driverless ABI smoke check against the toolkit stub. These subsequent runs verify that the same fresh binaries loaded and executed with a real driver and GPU on the two replay fixtures. That earlier evidence remains the record of the offline check. [Earlier offline validation](https://github.com/average-user-887/llamaxwell-35/blob/b102b95377b6b1870fdd29436f63014baf192acb/stack-validation-20261118/README.md)
