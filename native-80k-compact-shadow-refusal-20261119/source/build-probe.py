"""Rebind the existing owned-child counter reader to the exact repacked CUDA ELF."""
from pathlib import Path
import argparse
import hashlib
import json
import re
import struct
import subprocess

here = Path(__file__).resolve().parent
parent = here.parent / 'break100-model-base'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def elf(path):
    data = path.read_bytes()
    ph, = struct.unpack_from('<Q', data, 32)
    size, count = struct.unpack_from('<HH', data, 54)
    headers = [struct.unpack_from('<IIQQQQQQ', data, ph + i * size) for i in range(count)]
    rw, = [p for p in headers if p[:2] == (1, 6)]
    assert rw[7] == 4096
    return rw


def addresses(path):
    names = subprocess.check_output(['nm','-aSn', path], text=True).splitlines()
    expected = [(16,'_ZN6occamyL7q2_rowsE'), (8,'_ZN6occamyL8q2_callsE'),
                (8,'_ZN6occamyL8q2_skipsE'), (24,'_ZL16ofa_sparse_calls'),
                (4,'_ZZ17ofa_sparse_launchILi2EEvR25ggml_backend_cuda_contextP11ggml_tensorE11stride_seen'),
                (8,'_ZN6occamyL11down4_callsE')]
    values = []
    for n, name in expected:
        entry, = [x.split() for x in names if x.endswith(' b ' + name)]
        address, size = map(lambda x: int(x, 16), entry[:2])
        assert size == n
        values.append(address)
    rows, calls, skips, sparse, stride, down4 = values
    return rows, rows + 8, calls, skips, sparse + 16, stride, down4


def shape(header):
    offset, vaddr, filesz, memsz = header[2], header[3], header[5], header[6]
    down = lambda n: n & -4096
    up = lambda n: down(n + 4095)
    return offset, vaddr, filesz, memsz, down(offset), down(vaddr), up(vaddr+filesz), up(vaddr+memsz)


def build(check):
    current = here/'libggml-cuda.so'
    baseline = here.parent/'break100-q2-down4/libggml-cuda.so'
    profile = json.loads((parent/'profile.json').read_text())
    assert sha(baseline) == profile['libggml-cuda.so']
    old, new = elf(baseline), elf(current)
    o, v, f, m, ob, vb, file_end, mem_end = shape(old)
    no, nv, nf, nm, nob, nvb, nfile_end, nmem_end = shape(new)
    a0, a1 = addresses(baseline), addresses(current)
    assert all(nfile_end <= x < nv + nm for x in a1)
    text = (parent/'supervise.c').read_text()
    replacements = {
        ','.join(hex(x) for x in a0): ','.join(hex(x) for x in a1),
        f'ELF PT_LOAD offset {o:#x}, vaddr {v:#x}, filesz {f:#x}': f'ELF PT_LOAD offset {no:#x}, vaddr {nv:#x}, filesz {nf:#x}',
        f'memsz {m:#x}': f'memsz {nm:#x}',
        f'b==base+{file_end:#x} && a>=base+{vb:#x}': f'b==base+{nfile_end:#x} && a>=base+{nvb:#x}',
        f'a==base+{file_end:#x} && ptr>=a': f'a==base+{nfile_end:#x} && ptr>=a',
        f'ptr+word_bytes<=base+{v:#x}+{m:#x}': f'ptr+word_bytes<=base+{nv:#x}+{nm:#x}',
        f'size_t n={mem_end:#x}': f'size_t n={nmem_end:#x}',
        f'mmap(base+{vb:#x},{file_end-vb:#x},PROT_READ|PROT_WRITE,MAP_FIXED|MAP_PRIVATE,fd,{ob:#x})':
            f'mmap(base+{nvb:#x},{nfile_end-nvb:#x},PROT_READ|PROT_WRITE,MAP_FIXED|MAP_PRIVATE,fd,{nob:#x})',
        f'mmap(base+{file_end:#x},{mem_end-file_end:#x},PROT_READ|PROT_WRITE,MAP_FIXED|MAP_PRIVATE|MAP_ANONYMOUS,-1,0)':
            f'mmap(base+{nfile_end:#x},{nmem_end-nfile_end:#x},PROT_READ|PROT_WRITE,MAP_FIXED|MAP_PRIVATE|MAP_ANONYMOUS,-1,0)',
    }
    for start, end in replacements.items():
        assert text.count(start) == 1, start
        text = text.replace(start, end)
    outfile = here/'supervise.c'
    for name in ('libllama.so.0.6.0','libllama-server-impl.so','libggml-cuda.so'):
        profile[name] = sha(here/name)
    expected_profile = json.dumps(profile, indent=2, sort_keys=True)+'\n'
    if check:
        assert outfile.read_text() == text
        assert (here/'profile.json').read_text() == expected_profile
    else:
        outfile.write_text(text)
        (here/'profile.json').write_text(expected_profile)
        subprocess.run(['cc','-std=c11','-O2','-Wall','-Wextra','-Werror','-o','supervise','supervise.c'],cwd=here,check=True)
    subprocess.run([here/'supervise','--cpu-check',current],check=True)
    print(json.dumps({'sha256':profile,'down4_counter_addresses':a1,'cuda_rw_program_header':new},indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true')
    build(parser.parse_args().check)
