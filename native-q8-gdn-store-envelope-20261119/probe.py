import ctypes
from pathlib import Path
import sys

library = Path('/candidate/libggml-cuda.so')
probe = ctypes.CDLL(str(library)).ggml_cuda_occamy_store_envelope
probe.argtypes = ()
probe.restype = ctypes.c_int
if sys.argv[1:] == ['--identity']:
    print(f'GDN_PROBE_LOAD_OK {library}')
elif not sys.argv[1:]:
    raise SystemExit(probe())
else:
    raise SystemExit(2)
