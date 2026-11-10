# cmp.py <base.bin> <arm.bin>... : bitwise compare full-vocab FP32 logits per step; prints per-step sha256 of each file.
import sys, hashlib, numpy as np
NV = 248320
b = np.fromfile(sys.argv[1], dtype=np.uint32)
assert b.size % NV == 0; S = b.size // NV
for p in sys.argv[1:]:
    a = np.fromfile(p, dtype=np.uint32); f = a.view(np.float32)
    assert a.size == b.size, (p, a.size)
    mm = int((a != b).sum()); nanc = int(np.isnan(f).sum()); infc = int(np.isinf(f).sum())
    steps_eq = sum(int((a[i*NV:(i+1)*NV] == b[i*NV:(i+1)*NV]).all()) for i in range(S))
    print(f"{p.split('/')[-1]} steps={S} values={a.size} per_step={NV} mismatches={mm} steps_bitequal={steps_eq}/{S} nan={nanc} inf={infc} file_sha256={hashlib.sha256(a.tobytes()).hexdigest()}")
    with open(p + ".steps.sha256", "w") as o:
        for i in range(S): o.write(f"{i} {hashlib.sha256(a[i*NV:(i+1)*NV].tobytes()).hexdigest()}\n")
