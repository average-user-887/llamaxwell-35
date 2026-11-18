import hashlib
import struct
import subprocess
from pathlib import Path

here = Path(__file__).resolve().parent
library = here.parent / "finish33-r2-stride/libggml-cuda.so"
expected = "369c6b058813a7ca53fc62c67c2a4ea056a8d3e35ee08aca36afca4040975871"
data = library.read_bytes()
assert hashlib.sha256(data).hexdigest() == expected and data[:6] == b"\x7fELF\x02\x01"
ph, = struct.unpack_from("<Q", data, 32)
size, count = struct.unpack_from("<HH", data, 54)
loads = [struct.unpack_from("<IIQQQQQQ", data, ph+i*size) for i in range(count)]
assert [(p[2], p[3], p[5], p[6], p[7]) for p in loads if p[:2] == (1, 6)] == [
    (0x2240ee0, 0x2240ee0, 0x6898, 0x18a68, 0x1000),
]
symbols = subprocess.check_output(["nm", "-aSn", library], text=True).splitlines()
for address, size, name in [
    (0x224d190, 16, "_ZN6occamyL7q2_rowsE"),
    (0x224d1a0, 8, "_ZN6occamyL8q2_callsE"),
    (0x224d180, 8, "_ZN6occamyL8q2_skipsE"),
    (0x2248bf0, 24, "_ZL16ofa_sparse_calls"),
    (0x2248b0c, 4, "_ZZ17ofa_sparse_launchILi2EEvR25ggml_backend_cuda_contextP11ggml_tensorE11stride_seen"),
]:
    assert f"{address:016x} {size:016x} b {name}" in symbols
subprocess.run([here / "supervise", "--cpu-check", library], check=True)
print("Pinned ELF geometry and six fixed owned-child counters verified, including aggregate sparse Q2 and actual fixed-stride launches.")
