# Four-row MoE cohorts on Quadro M6000

The four-row fused MoE operator passed its technical checks and missed the predeclared speed threshold. The standalone experiment used two synthetic four-row fixtures with eight experts selected per row. All four paired savings had to reach 190 µs against the generic four-row graph. The measured range was 140.761–168.406 µs, with gains of 107.080–117.543 µs against four sequential V3 graph rows.

| Fixture | Paired savings vs generic | Paired savings vs sequential V3 |
| --- | ---: | ---: |
| Independent routing | 168.406, 158.789 µs | 107.080, 110.997 µs |
| Partial overlap | 145.369, 140.761 µs | 111.997, 117.543 µs |

Each fixture ran six synchronized intervals in the order generic, sequential, cohorts, cohorts, sequential, generic. Each interval held 96 runs. The physical candidate counter recorded 96 launches in each of the four candidate intervals and zero in the eight controls. The Q3 counter stayed at zero. The 5.63–6.74 ms figures are the measured generic savings multiplied by forty to project repeated layer work. Whole-model throughput and quality are unmeasured by these fixtures.

The occupancy query reported two active blocks per SM on 24 SMs. The new kernel launched 48 blocks of 512 threads in four cohorts of twelve blocks, with 63 registers per thread, 96 bytes of static shared memory and 39,424 bytes of dynamic shared memory per block. Each fixture matched 32 independent CPU/GPU routing IDs and all 8,192 output words against four sequential V3 launches. V3 and V2 agreed bit for bit on these checks. The complete receipt preserves the per-row numerical differences between generic and V3.

The comparison reused the preceding Q3 MoE CUDA lineage. The Q4 TU retained all 17 prior device functions and 45,992 instruction words byte for byte. See [the complete operator receipt](q4-moe-cohorts.receipt.json), [source and runtime lineage](lineage.json) and [portable reproduction](REPRODUCE.md).
