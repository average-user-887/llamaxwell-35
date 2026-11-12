# G1b table: scheduler cap 0 (A) vs --mixed-batch-size 1024 (B)

Measurement only. Timed units (identity passes excluded). Cell = mean [min, max] (n).

Cycles present: [0, 1, 2, 3, 4]; timed units: 40; invalid: 0

## (a) restored-90K append (+4096)

| metric | A cap0 | B cap1024 | paired B-A (mean, 95% CI over cycles) |
|---|---|---|---|
| TTFT s | 32.186 [32.086, 32.256] (10) | 32.378 [32.285, 32.521] (10) | +0.192 +/- 0.078 (+0.6%, n=5 cycles) |
| max gap s | 15.835 [15.745, 15.883] (10) | 8.168 [8.144, 8.204] (10) | -7.667 +/- 0.037 (-48.4%, n=5 cycles) |
| wall s | 42.039 [41.964, 42.108] (10) | 42.220 [42.078, 42.375] (10) | +0.181 +/- 0.073 (+0.4%, n=5 cycles) |
| prefill tok/s | 127.391 [127.055, 127.805] (10) | 126.628 [126.021, 126.941] (10) | -0.763 +/- 0.317 (-0.6%, n=5 cycles) |
| peak VRAM MiB | 23108.000 [23108.000, 23108.000] (10) | 23108.000 [23108.000, 23108.000] (10) | +0.000 +/- 0.000 (+0.0%, n=5 cycles) |

## (b) e3 deep stream + 16K arrival

| metric | A cap0 | B cap1024 | paired B-A (mean, 95% CI over cycles) |
|---|---|---|---|
| TTFT s | 36.839 [36.616, 37.066] (10) | 37.641 [37.415, 37.841] (10) | +0.802 +/- 0.145 (+2.2%, n=5 cycles) |
| max gap s | 5.272 [5.240, 5.302] (10) | 2.885 [2.872, 2.897] (10) | -2.387 +/- 0.014 (-45.3%, n=5 cycles) |
| wall s | 42.062 [41.860, 42.265] (10) | 42.708 [42.488, 42.870] (10) | +0.647 +/- 0.142 (+1.5%, n=5 cycles) |
| prefill tok/s | 444.977 [442.253, 447.682] (10) | 435.489 [433.190, 438.121] (10) | -9.488 +/- 1.713 (-2.1%, n=5 cycles) |
| peak VRAM MiB | 23108.000 [23108.000, 23108.000] (10) | 23108.000 [23108.000, 23108.000] (10) | +0.000 +/- 0.000 (+0.0%, n=5 cycles) |

## Output receipt per timed unit

fg/bg: authoritative tokens_predicted / streamed ID count / non-ASCII events / content sha256[:12]. id_gap must be <= nonascii events.

| cyc.pos | arm | wl | valid | fg pred/ids/nonascii/sha | bg pred/ids/nonascii/sha | receipt ok |
|---|---|---|---|---|---|---|
| 0.0 | A | a | True | 32/32/0/7dc15b0acc49 | 512/512/0/6a179b8e030b | ok |
| 0.0 | A | b | True | 32/32/0/99d9cc136fcf | 256/256/0/0dbef95b8bf1 | ok |
| 0.1 | B | a | True | 32/32/0/7dc15b0acc49 | 512/510/3/df3f46419794 | ok |
| 0.1 | B | b | True | 32/32/0/dbccc2fcb61a | 256/256/0/d6334b2d7f39 | ok |
| 0.2 | B | a | True | 32/32/0/7dc15b0acc49 | 512/512/0/7527f50b7da9 | ok |
| 0.2 | B | b | True | 32/32/0/fdaf0e6ade73 | 256/256/0/d6334b2d7f39 | ok |
| 0.3 | A | a | True | 32/32/0/7dc15b0acc49 | 512/512/0/6a179b8e030b | ok |
| 0.3 | A | b | True | 32/32/0/38ac89349a20 | 256/256/0/0dbef95b8bf1 | ok |
| 1.0 | B | a | True | 32/32/0/7dc15b0acc49 | 512/510/3/df3f46419794 | ok |
| 1.0 | B | b | True | 32/32/0/d97159cb95a4 | 256/256/0/d6334b2d7f39 | ok |
| 1.1 | A | a | True | 32/32/0/7dc15b0acc49 | 512/512/0/6a179b8e030b | ok |
| 1.1 | A | b | True | 32/32/0/0636cc6837d7 | 256/256/0/0dbef95b8bf1 | ok |
| 1.2 | A | a | True | 32/32/0/7dc15b0acc49 | 512/504/11/f7ab53100e21 | ok |
| 1.2 | A | b | True | 32/32/0/282027471e8c | 256/256/0/0dbef95b8bf1 | ok |
| 1.3 | B | a | True | 32/32/0/7dc15b0acc49 | 512/510/3/df3f46419794 | ok |
| 1.3 | B | b | True | 32/32/0/396011a6183e | 256/256/0/d6334b2d7f39 | ok |
| 2.0 | A | a | True | 32/32/0/7dc15b0acc49 | 512/512/0/6a179b8e030b | ok |
| 2.0 | A | b | True | 32/32/0/57f00de5525c | 256/256/0/0dbef95b8bf1 | ok |
| 2.1 | B | a | True | 32/32/0/7dc15b0acc49 | 512/510/3/df3f46419794 | ok |
| 2.1 | B | b | True | 32/32/0/9dea69e60651 | 256/256/0/d6334b2d7f39 | ok |
| 2.2 | B | a | True | 32/32/0/7dc15b0acc49 | 512/512/0/7527f50b7da9 | ok |
| 2.2 | B | b | True | 32/32/0/89860a4bec6b | 256/256/0/d6334b2d7f39 | ok |
| 2.3 | A | a | True | 32/32/0/7dc15b0acc49 | 512/512/0/6a179b8e030b | ok |
| 2.3 | A | b | True | 32/32/0/ef4bc19c6878 | 256/256/0/0dbef95b8bf1 | ok |
| 3.0 | B | a | True | 32/32/0/7dc15b0acc49 | 512/510/3/df3f46419794 | ok |
| 3.0 | B | b | True | 32/32/0/a47a79f986eb | 256/256/0/d6334b2d7f39 | ok |
| 3.1 | A | a | True | 32/32/0/7dc15b0acc49 | 512/512/0/6a179b8e030b | ok |
| 3.1 | A | b | True | 32/32/0/8994d95b7c9d | 256/256/0/0dbef95b8bf1 | ok |
| 3.2 | A | a | True | 32/32/0/7dc15b0acc49 | 512/504/11/f7ab53100e21 | ok |
| 3.2 | A | b | True | 32/32/0/63738a7277c9 | 256/256/0/0dbef95b8bf1 | ok |
| 3.3 | B | a | True | 32/32/0/7dc15b0acc49 | 512/510/3/df3f46419794 | ok |
| 3.3 | B | b | True | 32/32/0/dbbc93037c01 | 256/256/0/d6334b2d7f39 | ok |
| 4.0 | A | a | True | 32/32/0/7dc15b0acc49 | 512/512/0/6a179b8e030b | ok |
| 4.0 | A | b | True | 32/32/0/99d9cc136fcf | 256/256/0/0dbef95b8bf1 | ok |
| 4.1 | B | a | True | 32/32/0/7dc15b0acc49 | 512/510/3/df3f46419794 | ok |
| 4.1 | B | b | True | 32/32/0/dbccc2fcb61a | 256/256/0/d6334b2d7f39 | ok |
| 4.2 | B | a | True | 32/32/0/7dc15b0acc49 | 512/512/0/7527f50b7da9 | ok |
| 4.2 | B | b | True | 32/32/0/fdaf0e6ade73 | 256/256/0/d6334b2d7f39 | ok |
| 4.3 | A | a | True | 32/32/0/7dc15b0acc49 | 512/512/0/6a179b8e030b | ok |
| 4.3 | A | b | True | 32/32/0/38ac89349a20 | 256/256/0/0dbef95b8bf1 | ok |

## Identity pass (stream:false, exact IDs sha256[:12] of fg/bg)

- arm A wl a: fg 0c3d64af81e9 bg 469cdf18d53a (n 32/512)
- arm A wl b: fg c2cd4b6b4b84 bg 65e8e33211db (n 32/256)
- arm B wl a: fg 2f7f6d233ee5 bg 469cdf18d53a (n 32/512)
- arm B wl b: fg f880f9b3229e bg 65e8e33211db (n 32/256)
