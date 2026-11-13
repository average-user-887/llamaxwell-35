# Q8N GPU result (measurement only)

Run dir: `<LAB>/occamy35-night-20261113/Q8N/gates/gpurun-20261113T091035Z`. Steps log at the end.

## 1. Exactness (native N vs generic G; K-only Nk vs Gk)

# Q8N exactness (analyzer output)

**Verdict: FAIL: 32768 K-only dmean +0.001970 top -0.78pt; 32768 dmean +0.001534 top -0.52pt; 98304 K-only dmean +0.001373 top +0.00pt; 98304 dmean +0.001788 top -1.56pt**

## bucket 32768
- N vs G (gate): rows=192 docs=3 dmean=+0.001534 CIdoc=[-0.001436,+0.004505] top=-0.52pt klmax=0.08529 max|dlogit|=10.34
- Gp vs G (noise): n/a
- G vs F (q8 cache cost, generic): n/a
- N vs F (q8 cache cost, native): n/a
- Nk vs Gk (K-only <1,0> gate, generic q8-K/f16-V reference): rows=128 docs=2 dmean=+0.001970 CIdoc=[+nan,+nan] top=-0.78pt klmax=0.04111 max|dlogit|=3.753

## bucket 98304
- N vs G (gate): rows=64 docs=1 dmean=+0.001788 CIdoc=[+nan,+nan] top=-1.56pt klmax=0.03070 max|dlogit|=4.586
- Gp vs G (noise): n/a
- G vs F (q8 cache cost, generic): n/a
- N vs F (q8 cache cost, native): n/a
- Nk vs Gk (K-only <1,0> gate, generic q8-K/f16-V reference): rows=64 docs=1 dmean=+0.001373 CIdoc=[+nan,+nan] top=+0.00pt klmax=0.01732 max|dlogit|=6.839

- X: no occamy_fa_cmp_kvq8 lines (arm not run or path inactive)

## 2. Slot restore test

| item | value |
|---|---|
| n_restored | 98023 |
| restore decode prompt_n | 4 |
| re-prefill (prompt_n>=64) | False |
| restore tokens (16) == no-restore tokens | True (first diff idx None) |
| no-restore source | build prefill decode (fresh prefill, n=24, first 16) |
| restore tokens | [15, 15, 18, 22, 19, 11, 220, 15, 87, 15, 15, 19, 15, 2069, 198, 90] |
| no-restore tokens | [15, 15, 18, 22, 19, 11, 220, 15, 87, 15, 15, 19, 15, 2069, 198, 90] |

## 3. Two-slot deep ABBA and extra units

# Q8N server gates (analyzer output)

| unit | arm | valid/n | median | per-cycle | peak VRAM MiB |
|---|---|---|---|---|---|
| both_deep | F | 6/6 | 58.92 | 58.61 59.03 58.89 | 23047 |
| both_deep | K | 6/6 | 60.77 | 60.89 60.70 60.93 | 22235 |
| both_deep | Q | 6/6 | 58.91 | 58.76 58.85 58.95 | 21527 |
| single_98k | F | 0/0 | nan |  | 0 |
| single_98k | K | 0/0 | nan |  | 0 |
| single_98k | Q | 0/0 | nan |  | 0 |
| append_4k | F | 0/0 | nan |  | 0 |
| append_4k | K | 0/0 | nan |  | 0 |
| append_4k | Q | 0/0 | nan |  | 0 |

Paired per-cycle ratio vs F (median over cycles):
- both_deep K/F: 1.0390 1.0284 1.0347 -> median +3.47%
- both_deep Q/F: 1.0026 0.9970 1.0010 -> median +0.10%
- single_98k K/F: n/a
- single_98k Q/F: n/a
- append_4k K/F: n/a
- append_4k Q/F: n/a

Session peak VRAM MiB: F 23047, Q 21527, K 22235, Q 21527, F 23047, K 22235, Q 21527, F 23047, Q 21527, K 22235, F 23047, Q 21527, K 22235, Q 21527, F 23047

## both_deep ms/tok (98K+90K, mean of 2 slots), per cycle

| arm | n cycles | ms/tok per cycle | mean ms/tok | peak VRAM MiB |
|---|---|---|---|---|
| F | 3 | 34.168 33.925 34.007 | 34.033 | 23047 |
| K | 3 | 32.904 33.005 32.883 | 32.931 | 22235 |
| Q | 3 | 34.105 34.053 33.997 | 34.052 | 21527 |

| arm vs F | paired cycles | per-cycle delta ms/tok | mean delta ms/tok | 95% CI | mean delta % |
|---|---|---|---|---|---|
| K | 3 | -1.264 -0.919 -1.124 | -1.102 | [-1.533, -0.672] | -3.24% |
| Q | 3 | -0.063 +0.128 -0.010 | +0.018 | [-0.227, +0.264] | +0.05% |

## 4. nvprof, one decode step at 98K (slot 0)

| arm | ofa kernel | ofa ms/step | ofa ms/launch | GPU busy ms/step (window) | grid | regs |
|---|---|---|---|---|---|---|
| F | ofa_decode_hs_ar_kernel(ofa_params) | 7.253 | 0.725 | 18.173 | ['24', '2', '1'] | 118 |
| Q | void ofa_decode_hs_ar_q8_kernel<bool=1, bool=1>(ofa_params) | 6.833 | 0.683 | 18.034 | ['24', '2', '1'] | 127 |
| K | void ofa_decode_hs_ar_q8_kernel<bool=1, bool=0>(ofa_params) | 6.447 | 0.645 | 17.779 | ['24', '2', '1'] | 121 |

## 5. Single 98K / 90K+4K append

# Q8N server gates (analyzer output)

| unit | arm | valid/n | median | per-cycle | peak VRAM MiB |
|---|---|---|---|---|---|
| both_deep | F | 0/0 | nan |  | 0 |
| both_deep | Q | 0/0 | nan |  | 0 |
| single_98k | F | 4/4 | 52.66 | 52.67 52.63 | 23109 |
| single_98k | Q | 4/4 | 52.98 | 53.30 52.86 | 21589 |
| append_4k | F | 4/4 | 127.96 | 127.92 128.03 | 23109 |
| append_4k | Q | 4/4 | 127.50 | 127.60 127.48 | 21589 |

Paired per-cycle ratio vs F (median over cycles):
- both_deep Q/F: n/a
- single_98k Q/F: 1.0119 1.0043 -> median +0.81%
- append_4k Q/F: 0.9974 0.9957 -> median -0.34%

Session peak VRAM MiB: F 23109, Q 21589, F 23109, Q 21589, F 23109, Q 21589

## both_deep ms/tok (98K+90K, mean of 2 slots), per cycle

| arm | n cycles | ms/tok per cycle | mean ms/tok | peak VRAM MiB |
|---|---|---|---|---|
| F | 0 |  | n/a | 0 |
| Q | 0 |  | n/a | 0 |

| arm vs F | paired cycles | per-cycle delta ms/tok | mean delta ms/tok | 95% CI | mean delta % |
|---|---|---|---|---|---|
| Q | 0 | n/a | | | |

## Steps log

```
start lock taken lib=d2886ab998289aa7 smoke=0 09:10:35
unload rc=0 09:10:35
rows G doc=0 rc=0 09:18:55
rows N doc=0 rc=0 09:27:08
rows Gk doc=0 rc=0 09:35:25
rows Nk doc=0 rc=0 09:43:43
rows G doc=1 rc=0 09:45:27
rows N doc=1 rc=0 09:47:03
rows Gk doc=1 rc=0 09:48:39
rows Nk doc=1 rc=0 09:50:15
rows G doc=2 rc=0 09:51:53
rows N doc=2 rc=0 09:53:29
rows Gk doc=2 rc=0 09:53:30
rows Nk doc=2 rc=0 09:53:30
exact cut before doc 3 09:53:30
exact analyze rc=0 09:53:36
restore.json copied from smoke run 09:53:36
prof rc=0 09:54:44
srv rc=0 10:01:11
srv analyze rc=0 10:01:11
srv5 rc=0 10:07:29
end elapsed_min=56 10:07:29
```
