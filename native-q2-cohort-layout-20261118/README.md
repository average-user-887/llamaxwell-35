# Row-major MoE block placement on Quadro M6000

Mapping one verification row to each grid Y coordinate changed where its tiles ran. In all four instrumented runs with the original layout, every SM received two blocks from the same row. In all four instrumented runs with the row-major layout, every SM received one block from each row and both blocks computed the same local tile. The 384 recorded block coordinates, rows, tiles and SM IDs covered every block in eight complete 48-block launches. The timed kernels ran separately, with placement recording disabled. [All observed placements and counts](result.json)

The ordinary row-major kernel saved less than the required 18 µs/layer in every comparison. The fastest pair saved 10.826 µs on five common experts. The reverse order of that fixture lost 0.142 µs. Each figure is the original two-row cohort time minus candidate time for 96 synchronized calls on the same graph and CUDA library. [All timing intervals and physical issue counters](perf.log)

| Experts shared by both rows | First paired saving | Reverse paired saving |
| --- | ---: | ---: |
| 0 of 8 | −1.072 µs | +0.250 µs |
| 5 of 8 | +10.826 µs | −0.142 µs |

On each fixture, both ordered expert lists and all 4,096 finite output words matched exactly across the ordinary kernels and candidate repeat. All eight instrumented replays also matched the original output words and route IDs. Each kernel admitted two resident blocks on each of 24 SMs. The ordinary original and candidate kernels used 63 and 64 registers, respectively, and neither spilled. The control relink reproduced 369c6b05 and the 17 original MoE functions retained all 46,032 instruction and scheduling words. The candidate CUDA library was b64efa6e. Runner and cleanup exited zero in two seconds. [Numerical and mapping observations](check.log) · [Build identities](build.json) · [Portable receipt](portable-receipt.json)

The experiment ended at the operator. Run `python3 -B report.py --report .` to reconstruct the decisions directly from the two original physical logs. Run `sha256sum --check SHA256SUMS` to verify the public files. The frozen manifest also identifies the local compiled binaries. Two build scripts use a typed placeholder for the original private build-source directory. An authorized local BUILD_SOURCE supplied to `hydrate-source.py` restores and verifies their measured bytes. [Protocol and source instructions](RELEASE.md)
