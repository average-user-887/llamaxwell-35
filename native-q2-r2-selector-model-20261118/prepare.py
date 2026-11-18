"""Retarget the prior accepted supervised model runner to this one pinned library."""
from pathlib import Path

here = Path(__file__).resolve().parent
prior = here.parent / "resume-q2-moe-native-r1"

def one(text, old, new):
    if text.count(old) != 1:
        raise ValueError(old)
    return text.replace(old, new)

runner = (prior / "release.sh").read_text()
runner = one(runner, 'n82=$campaign/resume-q2-moe-cohorts', 'overlay=$campaign/last-ms-r2-selector')
runner = one(runner, '$n82/libggml-cuda.so', '$overlay/libggml-cuda.so')
runner = one(runner, '  (cd "$card" && sha256sum --check --quiet FROZEN.sha256)',
             '  (cd "$card" && sha256sum --check --quiet FROZEN.sha256 && sha256sum --check --quiet INPUTS.sha256)\n'
             '  "$campaign/resume-q2-moe-native-r1/release.sh" --check\n'
             '  "$overlay/release.sh" --check')
runner = one(runner, 'printf \'Q2_MODEL_RECEIPT=%s\\n\'', 'printf \'Q2_R2_MODEL_RECEIPT=%s\\n\'')
runner = one(runner, 'deadline=$((start + cap))\n',
             'deadline=$((start + cap))\n'
             '''cutoff=$(TZ=Europe/Warsaw date -d '2026-11-18 16:57:00' +%s)
: "${ROOT_GPU_DEADLINE_EPOCH:?supply the benchmark owner cutoff}"
(( deadline <= cutoff )) || { echo 'model arm and cleanup must fit before 16:57 CET' >&2; exit 124; }
''')
runner = one(runner, 'full serial arm exceeds the supplied benchmark lease', 'model arm exceeds the supplied benchmark lease')
runner = runner.replace('nmmoon-resume-q2-model-', 'nmmoon-r2-top16-model-')
runner = runner.replace('resume.q2.model.owner', 'r2.top16.model.owner')
(here / "release.sh").write_text(runner)

supervise = (prior / "supervise.c").read_text()
supervise = one(supervise, '0x2244110,0x2244118,0x2244120,0x2244100',
                '0x2247150,0x2247158,0x2247160,0x2247140,0x2242bd0')
supervise = supervise.replace('index>=4','index>=5').replace('for(int i=0;i<4;i++)','for(int i=0;i<5;i++)')
supervise = one(supervise, 'word(4,real,', 'word(5,real,')
supervise = one(supervise, 'uint64_t v[4]', 'uint64_t v[5]')
for old, new in {
    '0x2236f00':'0x2239ee0', '0x2237f00':'0x223aee0', '0x6858':'0x6898',
    '0x189c8':'0x18a28', '0x223f000':'0x2242000', '0x2237000':'0x223a000',
    '0x2236000':'0x2239000', '0x2252000':'0x2254000',
}.items():
    supervise = supervise.replace(old, new)
supervise = one(supervise, 'CPU_PARENT_PASS four exact words', 'CPU_PARENT_PASS five exact words')
supervise = one(supervise, 'PRIu64"}\\n",owned,v[0],v[1],v[2],v[3]);',
                'PRIu64",\\\"q2_sparse\\\":%"PRIu64"}\\n",owned,v[0],v[1],v[2],v[3],v[4]);')
(here / "supervise.c").write_text(supervise)
