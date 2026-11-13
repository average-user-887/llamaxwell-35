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
