# G1 advisor ruling (2026-11-12 ~19:55)
- Accepted as measurement only. GEMM (GGML_CUDA_FA_MAXWELL_GEMM=1) NOT adoptable.
- Row0 single-position KL at n~5 per bucket is noise-dominated (stock tile-order change A-vs-Ap alone = .259 at 65K); cannot pass or fail the gate.
- 16-row same-top1: B-vs-C 97.9% vs A-vs-C 99.0% at 16K (n=6 windows x 16 rows = 96 rows; -1.0 pt), 97.9% vs 99.0% at 32K (n=6 x 16 = 96 rows; -1.0 pt); 65K 100% vs 100% (n=5 x 16 = 80 rows).
- Follow-up G1c: >=64 rows x >=8 docs per bucket (32K/65K/98K), noise floor A-vs-Ap and A-vs-C on same rows, fixed owner gate, tail formula to be verified.
