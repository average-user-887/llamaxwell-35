#!/usr/bin/env python3
from pathlib import Path
import argparse
import hashlib
import json
import re
import subprocess

HERE = Path(__file__).resolve().parent
C = HERE.parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(command):
    return subprocess.check_output(command, text=True)


def section(path, name):
    line, = [s for s in read(['readelf','-WS',path]).splitlines() if re.search(rf'\s{re.escape(name)}\s',s)]
    offset,size = [int(x,16) for x in line.split('PROGBITS')[1].split()[1:3]]
    return path.read_bytes()[offset:offset+size]


def symbols(path, mode):
    return {tuple(x.split()[:2]) for x in read(['nm','-D',mode,'--format=posix',path]).splitlines()}


def dynamic(path):
    return [(m.group(1),m.group(2)) for s in read(['readelf','-d',path]).splitlines()
            if (m := re.search(r'\((NEEDED|SONAME|RPATH|RUNPATH)\).*?\[(.*?)\]',s))]


def check():
    profile = json.loads((HERE/'profile.json').read_text())
    prior = json.loads((C/'break100-model-base/profile.json').read_text())
    paths = {
        'libggml-cuda.so': C/'break100-q2-down4/libggml-cuda.so',
        'libllama.so.0.6.0': C/'night-one-graph-draft/libs/libllama.so.0.6.0',
        'libllama-server-impl.so': C/'finish33-sampler-r1/libllama-server-impl.so',
    }
    assert sha(HERE/'cuda-control.so') == prior['libggml-cuda.so']
    assert sha(HERE/'core-control.so') == prior['libllama.so.0.6.0']
    abi = {}
    for name, src in paths.items():
        dst = HERE/name
        assert sha(src) == prior[name] and sha(dst) == profile[name]
        assert dynamic(src) == dynamic(dst)
        a,b = [symbols(p,'--defined-only') for p in (src,dst)]
        removed = a-b
        if name == 'libllama-server-impl.so':
            assert all(n.startswith('_Z') and 'server_context_impl' in n and ('11post_decode' in n or '6decode' in n) and t in 'VW' for n,t in removed)
        else:
            assert not removed
        abi[name] = {'original_exports':len(a),'candidate_exports':len(b),
                     'removed_server_internal_weak_symbols':len(removed),'needed_soname_search_path_equal':True}
    old,new,added = [section(p,'.nv_fatbin') for p in (paths['libggml-cuda.so'], HERE/'libggml-cuda.so', HERE/'compact-cuda.o')]
    insertion = new.index(added)
    assert new[:insertion]+new[insertion+len(added):] == old
    inputs = symbols(HERE/'libllama-server-impl.so','--undefined-only')
    exported = symbols(HERE/'libllama.so.0.6.0','--defined-only')
    entry, = [n for n,_ in inputs if n.startswith('_Z27llama_occamy_compact_decode')]
    assert (entry, 'T') in exported
    assert ('ggml_backend_cuda_occamy_compact_v1','T') in symbols(HERE/'libggml-cuda.so','--defined-only')
    return {'baseline_software':prior,'candidate_software':profile,'abi':abi,
            'unchanged_parent_device_fatbin_bytes':len(old),'added_sm52_fatbin_bytes':len(added),
            'addition_offset':insertion,'parent_fatbin_sha256':hashlib.sha256(old).hexdigest(),
            'added_fatbin_sha256':hashlib.sha256(added).hexdigest()}


if __name__ == '__main__':
    option=argparse.ArgumentParser()
    option.add_argument('--check', action='store_true')
    document=json.dumps(check(), indent=2, sort_keys=True)+'\n'
    if option.parse_args().check:
        assert (HERE/'build.json').read_text() == document
    else:
        print(document,end='')
