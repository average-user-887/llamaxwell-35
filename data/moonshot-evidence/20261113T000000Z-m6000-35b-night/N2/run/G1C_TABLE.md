# G1c result (analyzer output)

**Verdict: FAIL in buckets [32768, 65536, 98304]; A-vs-Ap noise is of similar size or larger: gate not meaningful at this depth (owner decision)**

## bucket 32768
- B vs A (gate): rows=512 docs=8 dmean=+0.00345 CIdoc=[+0.00146,+0.00543] CIrow=[+0.00233,+0.00456] top=-1.17pt max=0.1880
- A vs Ap (noise): rows=512 docs=8 dmean=+0.00331 CIdoc=[+0.00134,+0.00528] CIrow=[+0.00245,+0.00417] top=-0.98pt max=0.1385
- A vs C (F32 floor): n/a (C did not reach this bucket or not run)
- B vs C: n/a
- C-relative dmean KL(C||B)-KL(C||A): n/a

## bucket 65536
- B vs A (gate): rows=512 docs=8 dmean=+0.00253 CIdoc=[-0.00044,+0.00549] CIrow=[+0.00137,+0.00369] top=-0.78pt max=0.2369
- A vs Ap (noise): rows=512 docs=8 dmean=+0.00137 CIdoc=[+0.00034,+0.00241] CIrow=[+0.00106,+0.00168] top=-0.39pt max=0.0328
- A vs C (F32 floor): n/a (C did not reach this bucket or not run)
- B vs C: n/a
- C-relative dmean KL(C||B)-KL(C||A): n/a

## bucket 98304
- B vs A (gate): rows=512 docs=8 dmean=+0.00332 CIdoc=[+0.00096,+0.00569] CIrow=[+0.00230,+0.00435] top=-0.98pt max=0.1882
- A vs Ap (noise): rows=512 docs=8 dmean=+0.00263 CIdoc=[+0.00129,+0.00396] CIrow=[+0.00197,+0.00329] top=-0.98pt max=0.0850
- A vs C (F32 floor): n/a (C did not reach this bucket or not run)
- B vs C: n/a
- C-relative dmean KL(C||B)-KL(C||A): n/a

