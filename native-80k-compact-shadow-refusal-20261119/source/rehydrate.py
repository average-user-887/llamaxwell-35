#!/usr/bin/env python3
"""Restore exact owner build-source bytes using an out-of-packet alias file."""
import argparse
import hashlib
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent


def rehydrate(aliases_path, output):
    aliases_path, output = aliases_path.resolve(), output.resolve()
    if aliases_path.is_relative_to(HERE) or output.is_relative_to(HERE):
        raise ValueError('keep private aliases and reconstructed files outside the portable packet')
    pins = json.loads((HERE/'source-pins.json').read_text())
    aliases = json.loads(aliases_path.read_text())
    required = {alias for item in pins['rehydration'] for alias in item['aliases']}
    if aliases.keys() != required or any(not isinstance(v,str) or not v or any(c in v for c in '\r\n\0') for v in aliases.values()):
        raise ValueError('exact private alias mapping required')
    payload = {}
    for item in pins['rehydration']:
        data = (HERE/item['template']).read_bytes()
        for key in item['aliases']:
            marker = key.encode()
            if marker not in data:
                raise ValueError('template alias missing')
            data = data.replace(marker,aliases[key].encode())
        if hashlib.sha256(data).hexdigest() != item['sha256']:
            raise ValueError('private alias does not reconstruct pinned original bytes')
        payload[item['original']] = data
    output.mkdir(mode=0o700,parents=True,exist_ok=False)
    for name,data in payload.items():
        path=output/name
        path.write_bytes(data)
        path.chmod(0o700 if name.endswith('.sh') else 0o600)
    return len(payload)


if __name__=='__main__':
    os.umask(0o077)
    p=argparse.ArgumentParser()
    p.add_argument('--aliases',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    print(f'reconstructed {rehydrate(a.aliases,a.output)} exact owner files')
