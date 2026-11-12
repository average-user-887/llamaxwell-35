#!/usr/bin/env python3 -I
"""Server batching / n_kv tail calculator (no GPU). Rules cited in TAIL.md.
n_kv of a ubatch = PAD256(max_pos_in_ubatch+1); tag = n_kv % 4096."""
import sys
pad = lambda x, m=256: (x + m - 1) // m * m

def batches(c, N, n_batch=2048, n_ub=1024):
    """server prompt batches for cached c -> prompt N (checkpoint split points N-1028, N-4); each batch split into n_ub ubatches"""
    cuts = sorted({x for x in (N - 4 - n_ub, N - 4) if c < x < N})
    out, p = [], c
    while p < N:
        stop = min(p + n_batch, N)
        for x in cuts:
            if p < x < stop: stop = x; break
        out.append((p, stop)); p = stop
    ub = []
    for a, b in out:
        for s in range(a, b, n_ub): ub.append((s, min(s + n_ub, b)))
    return ub

def window(N, cached):
    rows = []
    for k in range(16):  # G1: row k prompt = tokens[:N+k], 1 new token at pos N+k-1 (row0 re-decodes last token)
        p = N + k - 1
        rows.append((k, p, pad(p + 1), pad(p + 1) % 4096))
    return batches(cached, N), rows

if __name__ == '__main__':
    print('N,cached_before_stage,stage_ubatch_end_nkv%4096,last_stage_ubatches,row0_nkv,row0_tag,rows1_15_tag,N%4096')
    for N, c in [(16384, 0), (32768, 16384), (65536, 49152), (65836, 49452)]:
        ub, rows = window(N, c)
        last = ub[-3:]
        print(N, c, pad(ub[-1][1]) % 4096, last, rows[0][2], rows[0][3], sorted({r[3] for r in rows[1:]}), N % 4096, sep=',')
    print('G1c rows: pos B..B+63, ub=1024 prefill; tag per row = pad256(pos+1)%4096')
    for B in (32768, 65536, 98304):
        print(B, sorted({pad(p + 1) % 4096 for p in range(B, B + 64)}), 'prefill ubatch ends at', B, 'multiple of 1024:', B % 1024 == 0)
