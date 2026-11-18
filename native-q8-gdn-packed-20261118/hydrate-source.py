"""Supply the original container build mount to reconstruct four release sources per attempt."""
import hashlib
import json
import sys
from pathlib import Path
root=Path(__file__).resolve().parent
mount, target=sys.argv[1:]
for attempt in ['r0','r1']:
    inp=root/attempt
    out=Path(target)/attempt
    out.mkdir(parents=True,exist_ok=True)
    pins=json.loads((inp/'actual-source-pins.json').read_text())
    for name,digest in pins.items():
        p=inp/name
        raw=(p if p.exists() else inp/(name+'.in')).read_bytes().replace(b'@CUDA_BUILD_SOURCE_MOUNT@',mount.encode())
        if hashlib.sha256(raw).hexdigest()!=digest:
            raise ValueError(f'Measured source differs: {attempt}/{name}')
        (out/name).write_bytes(raw)
print('All measured code and launch sources reproduced.')
