# D1/GDN5 phase-A lane-oriented repack: address map

Source (unchanged ggml Q4_0): phase-A rows t = 0..12351 = qkv[0..8191] | gate[0..4095] | alpha[0..31] | beta[0..31],
each K=2048 -> 64 Q4_0 blocks x 18 B = 1152 B = 288 32-bit words. GDN4 phase A: lane loads uint4 words 4l..4l+3,
128+4l.., 256+4l.. (l<8) into smem stage, syncwarp, then lane l reads stage words 9l..9l+8 (= blocks 2l, 2l+1, 36 B)
and calls dot2w -> dot2v(w0..w8) with xs chunk l.

Packed buffer: one cudaMalloc of 12352 x 1152 B = 14,229,504 B (13.57 MiB) per layer, row t at t*1152 (16 B aligned).
Within a row, packed word q (0..287) holds source word src(q):
  q < 256 : g = q>>7, lane = (q & 127)>>2, j = 4g + (q & 3)
  q >= 256: lane = q - 256, j = 8
  src(q) = 9*lane + j
Lane l loads uint4 at packed word 4l (j 0..3), uint4 at 128+4l (j 4..7), one word at 256+l (j 8):
three fully coalesced warp loads (512 B, 512 B, 128 B), no smem stage, no syncwarp.
Inverse: src -> q is the bijection lane = s/9, j = s%9; q = j<8 ? 128*(j>>2) + 4*lane + (j&3) : 256 + lane.

Numerics: lane l's registers w0..w8 hold exactly the words GDN4 read from stg+9l; the same dot2v(w, xc, sx),
the same warp_sum butterfly and the same lane-0 store -> identical FMA and reduction order per output.

Pack-time check (always on with AREPACK): (1) hit count of src over q = 1 for all 288 words (bijection);
(2) every packed word == source word src(q), all 12352x288 words; (3) inverse-unpacked row memcmp == source row.
Any mismatch frees the copy and the layer falls back to GDN4 <0,1,1,0,1>. Log line reports words checked and mismatches.

Flags: GGML_CUDA_OCCAMY_GDN_AREPACK=1 (needs GDN_REPACK=1 path, and the layer must have its phase-C copy);
GGML_CUDA_OCCAMY_GDN_AREPACK_LAYERS=N packs the first N layers seen; GGML_CUDA_OCCAMY_GDN_RA_GRID=N caps the
A-variant grid (never raises above occupancy). Kernel gdn_fused<0,1,1,0,2>, dyn smem 18176 B (SMEM_FLOATS only).
Cache key: wqkv->data per device (same lifetime model as GDN4 rp_lookup; not released on model unload).
