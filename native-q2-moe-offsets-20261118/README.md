# Bounded expert offsets on Quadro M6000

A two-row, eight-expert Occamy MoE operator reproduced every output and ordered expert choice on two fixtures. Switching gate/up and down weight addressing to 32-bit offsets removed 80 static XMAD instructions across the two loops. Only one of four timed comparisons became faster. The 18 µs/layer standalone gate and the conditional 13.7 µs/layer combination screen both missed. The confirmed native MTP1 runtime remains the reference. This operator does not measure model throughput.

| Experts shared between rows | First paired saving | Reverse paired saving |
| --- | ---: | ---: |
| 0 of 8 | −1.711 µs | +6.710 µs |
| 5 of 8 | −1.857 µs | −3.238 µs |

Positive savings mean the candidate finished sooner. Each number subtracts candidate time from the original cohort on the same program, CUDA library, graph and fixture. Four batches of 96 synchronized calls followed warmup per fixture. Detailed timings and physical counters are in [the full result](result.json).

Each fixture checked both sets of eight ordered router IDs, all 4,096 finite output words, and a second word-identical candidate execution. Each kernel admitted two 512-thread blocks per SM on all 24 SMs. The candidate compiled to 63 registers and zero stack or spills. Both original and modified paths physically launched in diagnostics and timings. Runner and cleanup exited zero in three seconds. [Portable receipt](portable-receipt.json) · [Exactness log](check.log) · [Timing log](perf.log)

The control reproduced CUDA library 369c6b05. The candidate was 00f9a42e. All 17 original MoE functions and 46,032 instruction/scheduling words matched. The Phase B and C loops went from 566 to 494 and 774 to 673 static instructions respectively, retaining the checked counts of floating-point operations, shuffles, unpacking and global loads. The model server with hash 4a0ff82d was an identity anchor for the parent experiment. [Build identities and loop counts](build.json)

Run `python3 -B report.py --check` for offline verifier tests and `python3 -B report.py --report .` to reconstruct the numerical and performance decision from the archived physical logs. `sha256sum --check SHA256SUMS` verifies this public package. The original frozen manifest also lists locally built binaries and disassembly-related inputs. The build and compile scripts use a typed placeholder for the original private build-source directory. `hydrate-source.py` with the authorized local BUILD_SOURCE verifies the original observed digests. [Physical test and rebuild instructions](RELEASE.md) · [Source pins](source-pins.json)
