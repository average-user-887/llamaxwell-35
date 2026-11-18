"""Bind the prior owned-child supervisor to the exact six words of the combined CUDA DSO."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / "last-ms-r2-selector-model"


def replace(source, table):
    for before, after in table:
        if before not in source:
            raise ValueError(f"supervisor preimage changed: {before}")
        source = source.replace(before, after)
    return source


def main():
    supervisor = replace((BASE / "supervise.c").read_text(), (
        ("0x2247150,0x2247158,0x2247160,0x2247140,0x2242bd0",
         "0x224d190,0x224d198,0x224d1a0,0x224d180,0x2248c00,0x2248b0c"),
        ("index>=5", "index>=6"),
        ("    char p[96],line[2048];", "    const size_t word_bytes = index == 5 ? sizeof(uint32_t) : sizeof(uint64_t);\n    char p[96],line[2048];"),
        ("ptr+8", "ptr+word_bytes"),
        ("    ssize_t bytes=pread(fd,out,8,(off_t)(base+offsets[index]));close(fd);return bytes==8;",
         "    *out=0;\n    ssize_t bytes=pread(fd,out,word_bytes,(off_t)(base+offsets[index]));\n    close(fd);return bytes==(ssize_t)word_bytes;"),
        ("        if(write(gate[1]", "        const uint32_t count_and_guard[]={106,UINT32_MAX};\n        memcpy(base+offsets[5],count_and_guard,sizeof count_and_guard);\n        if(write(gate[1]"),
        ("    for(int i=0;i<5;i++){uint64_t v=0;", "    for(int i=0;i<6;i++){uint64_t v=0;"),
        ("word(5,real,", "word(6,real,"),
        ("five exact words", "six exact words"),
        ("uint64_t v[5];for(int i=0;i<5;i++)", "uint64_t v[6];for(int i=0;i<6;i++)"),
        ('PRIu64"}\\n",owned,v[0],v[1],v[2],v[3],v[4]);',
         'PRIu64",\\"q2_stride\\":%"PRIu64"}\\n",owned,v[0],v[1],v[2],v[3],v[4],v[5]);'),
        ("0x2239ee0", "0x2240ee0"), ("0x223aee0", "0x2240ee0"), ("0x18a28", "0x18a68"),
        ("0x2242000", "0x2248000"), ("0x223a000", "0x2240000"), ("0x2239000", "0x2240000"),
        ("0x2254000", "0x225a000"),
    ))
    (HERE / "supervise.c").write_text(supervisor)

    layout = replace((BASE / "check-layout.py").read_text(), (
        ("last-ms-r2-selector/libggml-cuda.so", "finish33-r2-stride/libggml-cuda.so"),
        ("9e999bcaf04397f267548abdfa3e326b65bee122fb1e24f92be55529f40a335e",
         "369c6b058813a7ca53fc62c67c2a4ea056a8d3e35ee08aca36afca4040975871"),
        ("0x2239ee0, 0x223aee0, 0x6898, 0x18a28", "0x2240ee0, 0x2240ee0, 0x6898, 0x18a68"),
        ("0x2247150", "0x224d190"), ("0x2247160", "0x224d1a0"), ("0x2247140", "0x224d180"),
        ('    (0x2242bc0, 24, "_ZL16ofa_sparse_calls"),',
         '    (0x2248bf0, 24, "_ZL16ofa_sparse_calls"),\n'
         '    (0x2248b0c, 4, "_ZZ17ofa_sparse_launchILi2EEvR25ggml_backend_cuda_contextP11ggml_tensorE11stride_seen"),'),
        ("five fixed owned-child counters verified; the attention count is aggregate Q2.",
         "six fixed owned-child counters verified, including aggregate sparse Q2 and actual fixed-stride launches."),
    ))
    (HERE / "check-layout.py").write_text(layout)


if __name__ == "__main__":
    main()
