# FK1 - Fable prefill MoE kernel (GGML_CUDA_MOE_FABLE_PREFILL)

| Check | Result | Status |
|---|---|---|
| sm_52 incremental build (CPU only) | rc=0, 3 units rebuilt | done |
| Host emulation of the kernel (hosttest/fk_host, 256 threads per CTA) | 5/5 cases OK, tile coverage exact, nonfinite 0, relRMSE ~1e-7 vs FP64 | done |
| Disassembly, NC=8 | 2080 FFMA / 128 LDS.128 (64 FFMA per 4 loads, 94% FFMA in the inner step), ~125 regs | done |
| Flag absent in prod lib | 0 strings in batch8 lib, 3 in FK1 lib | done |
| test-backend-ops MUL_MAT_ID on GPU (flag on, MIN=1) | - | PENDING (GPU) |
| Microbench 16K prefill, MoE ms/ubatch, prod vs FK1 (gate >= 1.07x; A4 got 1.025x) | - | PENDING (GPU) |
| CHECK=64 vs mmq: relRMSE, nonfinite | - | PENDING (GPU) |
| N3 ABBA n=6 (only if micro >= 1.07x) | - | PENDING (GPU) |
| Three-way prod / opus-A4 / fable-FK1, 5 rounds interleaved | - | PENDING (GPU) |

Nothing here was run on the GPU. Speedup below is an estimate, not a measurement.

## Design
- One CTA = 256 threads = 256 rows x 8*NC columns of one expert; K walked one Q4_0 block (32) per step.
- Load: thread t dequantises row t of the block to FP32 in shared memory with fma(d, q, -8d), bit-identical to (q-8)*d; activations gathered through ids_src1 as float4. No q8_1 activation quantisation.
- FMA: warp = column group, lane = row group; each thread holds an 8 x NC register microtile fed by 128-bit shared loads (64 FFMA per 4 loads at NC=8).
- Four launches per op, NC = 8/4/2/1 for tile widths 33..64 / 17..32 / 9..16 / 1..8, so a remainder of r columns costs ceil(r/NC) warps instead of a padded 64/32/16 tile. Tile lists built on the GPU by one small kernel (moe_fable_tiles).
- Eligibility: flag = 1, sm_52, Q4_0, K % 32 == 0, rows % 256 == 0, tokens >= GGML_CUDA_MOE_FABLE_PREFILL_MIN (512). Everything else stays on mmq. Numerics: FP32 weights x FP32 activations, i.e. closer to exact than mmq (no int8 activation rounding); expect relRMSE vs mmq of the same order as A4 (~2e-2), which is the mmq quantisation error.

## Expected speedup (roofline)
- Work: ~2.06e12 MAC per 2048-token ubatch (assuming ~40 MoE layers). Prod 1387 ms/ubatch = 1.49e12 MAC/s.
- Peak: 3072 cores x ~1 GHz ~ 3.07e12 FFMA/s. Prod is at ~49% of peak (integer XMAD microkernel ~58% useful x ~0.84 column padding).
- FK1: 94% FFMA density in the inner step, ~7% dequant + gather overhead at 64 columns, padding <= 7 columns in the wide class and ~0 in the narrow ones. Paper ceiling 0.75-0.8 of peak = 1.3-1.6x on the GEMM kernels.
- Honest expectation: >= 1.15-1.2x on MoE ms/ubatch. Below that means an unmodelled limiter.

## Risks
- A4 reached only 1.025x with a similar idea, so a limiter outside the microkernel may exist (250 W cap under dense FFMA, memory traffic of the Q4_0 stream, layer-count assumption in the roofline).
- ~125 registers at NC=8 limits occupancy to 2 CTAs (16 warps) per SM.
- Four launches per op: tail effect on small ops; narrow classes have lower FFMA density.
- Load phase is serial with the FMA phase (two barriers per K step).

## Tests added (tests/test-backend-ops.cpp, MUL_MAT_ID, Q4_0)
256 experts / 8 used, broadcast, 512 rows x 2048 tokens x K 2048; 2048 rows x 1024 tokens x K 512; odd token counts 1000 and 1022; 64 experts; strided expert view; 16/2 with 77 tokens (broadcast and not); 8/2 with 3 tokens; id range 64.

GPU runner (inside the FK1 libs):
`GGML_CUDA_MOE_FABLE_PREFILL=1 GGML_CUDA_MOE_FABLE_PREFILL_MIN=1 GGML_CUDA_OCCAMY_DEBUG=1 test-backend-ops -o MUL_MAT_ID -b CUDA0`
Pass = all cases OK and at least one `moe-fable: engaged launches` line. Run once with the flag unset as the control.

## Run
GPU id and unload URL come from the launch environment only (N3_GPU, N3_UNLOAD_URL); they are not stored in any file.
- Gate + N3: `N3_GPU=... N3_UNLOAD_URL=... nohup bash FK1/fk1_gpu.sh > launch.log 2>&1 &` (FK1_FORCE_N3=1 runs N3 regardless of the gate)
- Three-way: `N3_GPU=... N3_UNLOAD_URL=... nohup bash FK1/fk1_3way.sh > launch3.log 2>&1 &` -> 3way.md, 3way_rows.tsv
Both take the GPU bench lock (12 h wait), POST-unload only the 35B model, refuse unload-all, leave the 250 W cap alone, and do not restore prod.

## Files
- Source: FK1/src (branch fk1-moe-fable, commit 40ea6a6): ggml/src/ggml-cuda/moe-fable.cuh, moe-fable-api.cuh, mmid.cu, ggml-cuda.cu, tests/test-backend-ops.cpp
- Build: FK1/build/bin, FK1/build-fk1.sh, FK1/build.log, FK1/RECEIPT.txt
- Host emulation: FK1/hosttest/fk_host.cpp; disassembly: FK1/dis/mmid.sass
