import hashlib
import struct
import subprocess
from pathlib import Path

here = Path(__file__).resolve().parent
library = here.parent / "last-ms-r2-selector/libggml-cuda.so"
expected = "9e999bcaf04397f267548abdfa3e326b65bee122fb1e24f92be55529f40a335e"
data = library.read_bytes()
assert hashlib.sha256(data).hexdigest() == expected and data[:6] == b"\x7fELF\x02\x01"
ph, = struct.unpack_from("<Q", data, 32)
size, count = struct.unpack_from("<HH", data, 54)
loads = [struct.unpack_from("<IIQQQQQQ", data, ph+i*size) for i in range(count)]
assert [(p[2], p[3], p[5], p[6], p[7]) for p in loads if p[:2] == (1, 6)] == [
    (0x2239ee0, 0x223aee0, 0x6898, 0x18a28, 0x1000),
]
symbols = subprocess.check_output(["nm", "-aSn", library], text=True).splitlines()
for address, size, name in [
    (0x2247150, 16, "_ZN6occamyL7q2_rowsE"),
    (0x2247160, 8, "_ZN6occamyL8q2_callsE"),
    (0x2247140, 8, "_ZN6occamyL8q2_skipsE"),
    (0x2242bc0, 24, "_ZL16ofa_sparse_calls"),
]:
    assert f"{address:016x} {size:016x} b {name}" in symbols
subprocess.run([here / "supervise", "--cpu-check", library], check=True)
print("Pinned ELF geometry and five fixed owned-child counters verified; the attention count is aggregate Q2.")
