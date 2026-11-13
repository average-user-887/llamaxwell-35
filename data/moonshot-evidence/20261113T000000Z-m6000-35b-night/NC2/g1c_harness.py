#!/usr/bin/env python3
"""G1c arm runner (call only from run_all.sh, which holds the GPU lock).
usage: g1c_harness.py ARM OUT DOC_LIST [ctx]   ARM in A B Ap C ; DOC_LIST e.g. 0,1,2,3 ; env G1C_GPU (required), G1C_CUT_EPOCH (optional: do not start a chunk after this unix time)
Teacher-forced rows: doc k = fixture tokens [k*98368,(k+1)*98368). Per doc: prefill 0..B, G1C_NR (default 64) rows at B..B+NR-1, continue, for B in 32768,65536,98304.
Rows are full-vocab FP32 logits (g1c_rows). Identical schedule in every arm; only env/lib/flags differ."""
import sys, os, json, time, subprocess, hashlib, pathlib
sys.dont_write_bytecode = True
M = '<LAB>'; W = M + '/codex-investigation-20261112'
G = os.environ.get('G1C_DIR', M + '/occamy35-day-20261112/evidence/G1c')   # G1C_DIR: dir holding the compiled g1c_rows
IMG = 'nvidia/cuda:12.8.1-devel-ubuntu24.04'
LIB_V4 = W + '/prefill/gemm/integration-v4/src/build/bin/libggml-cuda.so'   # Codex v4, same as G1
APP = M + '/occamy35-night-20261111/app'
MODEL = '/models/occamy-1.0-Q4_0-headq40b.gguf'
DOCLEN = 98368; BUCKETS = tuple(int(x) for x in os.environ.get('G1C_BUCKETS', '32768,65536,98304').split(',')); NR = int(os.environ.get('G1C_NR', 64))
BASE_ENV = 'GGML_CUDA_MOE_DOWNX=1 GGML_CUDA_MOE_DOWNH=2 GGML_CUDA_OCCAMY_GDN_QKREUSE=1 GGML_CUDA_OCCAMY_GDN_PCSHARE=1 GGML_CUDA_MMVQ_SM52=1 GGML_CUDA_RMS_WARP=0 GGML_CUDA_HEAD_ROWPACK_SM52=1 GGML_CUDA_MMQ_COMPACT_SM52=1 GGML_CUDA_Q8_REUSE=1 GGML_CUDA_NORM_Q8=1 GGML_CUDA_CONCAT_T=1 GGML_CUDA_OCCAMY_GDN_STATE_DIRECT=1 GGML_CUDA_GATE_COPY=1 GGML_CUDA_OCCAMY_GDN=1 GGML_CUDA_MMQ_XMAD_SM52=1 GGML_CUDA_GDN_COLS=2,4 GGML_CUDA_MOE_LOP3=1 GGML_CUDA_MOE_TAILBAL=1 GGML_CUDA_OCCAMY_GDN_PREF=1 GGML_CUDA_OCCAMY_HOSTGAP=1 GGML_CUDA_OCCAMY_SMALLFUSE=2 GGML_CUDA_MMQ_TAIL_SM52=1 GGML_CUDA_OCCAMY_GDN_REPACK=1 GGML_CUDA_OCCAMY_N2SPLIT=1 GGML_CUDA_OCCAMY_N2GDN=1'.split()
FA_ENV = ['GGML_CUDA_OCCAMY_FA=1', 'GGML_CUDA_OCCAMY_FUSED=1', 'GGML_CUDA_OCCAMY_FA_PIPE=3', 'GGML_CUDA_OCCAMY_FA_ADDR=1']
assert 'GGML_CUDA_MAXWELL_GRAPHS' not in os.environ and 'GGML_CUDA_FA_MAXWELL_GEMM' not in os.environ, 'forbidden flag in caller env'
def arm_cfg(arm):
    if arm == 'C': return list(BASE_ENV), dict(fa=0, kt='f32', ub=512)
    env = list(BASE_ENV) + FA_ENV + (['GGML_CUDA_FA_MAXWELL_GEMM=1'] if arm == 'B' else []) + (['GGML_CUDA_FA_PARALLEL_BLOCKS_MULT=2'] if arm == 'Ap' else [])
    return env, dict(fa=1, kt='f16', ub=1024)
def doc_ntok(buckets=None, nr=None):
    """Tokens a doc slice must hold: the last row reads token index maxB+NR-1. Starts stay at k*DOCLEN so A/B vectors stay comparable."""
    return max(DOCLEN, max(buckets or BUCKETS) + (NR if nr is None else nr))
def validate_plan(docs, buckets, nr, fixture_len):
    if nr < 1: raise SystemExit('G1C_NR must be >=1')
    need = doc_ntok(buckets, nr)
    for k in docs:
        if k < 0 or k * DOCLEN + need > fixture_len: raise SystemExit('doc %d needs fixture tokens [%d,%d) but fixture has %d' % (k, k * DOCLEN, k * DOCLEN + need, fixture_len))
def events(docs, buckets, d):
    ev, pos = [], 0
    for k in docs:
        ev += ['T/t/doc%d.i32' % k, 'K']; pos = 0
        for B in buckets:
            ev += ['Odoc%d_b%d' % (k, B), 'P%d:%d' % (pos, B), 'R%d' % NR]; pos = B + NR
    return ev
def sh(c, **k): return subprocess.run(c, capture_output=True, text=True, **k)
def run_chunk(arm, out, docs, ctx, buckets, tag):
    env, f = arm_cfg(arm); gpu = os.environ['G1C_GPU']; od = pathlib.Path(out) / arm; od.mkdir(parents=True, exist_ok=True)
    name = 'g1c-%s-%s' % (arm.lower(), tag)
    cmd = ['docker', 'run', '--rm', '--name', name, '--network', 'none', '-u', '%d:%d' % (os.getuid(), os.getgid()), '--device', 'nvidia.com/gpu=' + gpu,
           '-v', '<MODELS>:/models:ro', '-v', str(od / 'tok') + ':/t:ro', '-v', str(od) + ':/o', '-v', APP + ':/app:ro',
           '-v', G + '/g1c_rows:/app2/g1c_rows:ro', '-v', LIB_V4 + ':/app/libggml-cuda.so:ro', '-e', 'LD_LIBRARY_PATH=/app']
    for e in env: cmd += ['-e', e]
    cmd += ['-w', '/app', '--entrypoint', '/app2/g1c_rows', IMG, MODEL, '/o', str(ctx), str(f['fa']), f['kt'], str(f['ub'])] + events(docs, buckets, None)
    (od / ('launch_%s.json' % tag)).write_text(json.dumps({'arm': arm, 'env': env, 'cfg': f, 'ctx': ctx, 'docs': docs, 'buckets': buckets, 'lib_sha256': hashlib.sha256(open(LIB_V4, 'rb').read()).hexdigest(), 'image': IMG}, indent=1))
    t0 = time.time(); r = sh(cmd); (od / ('log_%s.txt' % tag)).write_text(r.stdout + r.stderr)
    return r.returncode, time.time() - t0
def main():
    arm, out, docs = sys.argv[1], sys.argv[2], [int(x) for x in sys.argv[3].split(',')]
    ctx = int(sys.argv[4]) if len(sys.argv) > 4 else (98560 if max(BUCKETS) > 66000 else 34000); cut = float(os.environ.get('G1C_CUT_EPOCH', 0))
    od = pathlib.Path(out) / arm; (od / 'tok').mkdir(parents=True, exist_ok=True)
    import numpy as np
    fx = json.load(open(W + '/fixture_tokens.json'))
    validate_plan(docs, BUCKETS, NR, len(fx)); nt = doc_ntok()
    for k in docs: np.asarray(fx[k * DOCLEN:k * DOCLEN + nt], dtype=np.int32).tofile(str(od / 'tok' / ('doc%d.i32' % k)))   # NR=128: 98432 tokens (was fixed 98368 -> OOB read past end)
    per = int(os.environ.get('G1C_PER', 2)); chunks = [docs[i:i + per] for i in range(0, len(docs), per)]
    buckets = BUCKETS
    for i, ch in enumerate(chunks):
        if cut and time.time() > cut: print('cut reached; skipping docs', ch, flush=True); break
        rc, dt = run_chunk(arm, out, ch, ctx, buckets, 'c%d' % i); print('arm %s docs %s rc=%d %.0fs' % (arm, ch, rc, dt), flush=True)
        if rc != 0 and arm == 'C' and ctx > 66000:   # F32/fa-off may not reach 98K: retry reachable subset (32K,65K), ctx 66000
            sh(['docker', 'rm', '-f', 'g1c-%s-c%d' % (arm.lower(), i)])
            ctx, buckets = 66000, BUCKETS[:2]; rc, dt = run_chunk(arm, out, ch, ctx, buckets, 'c%dr' % i); print('arm C retry 66K subset rc=%d %.0fs' % (rc, dt), flush=True)
        if rc != 0: print('FAILED chunk', ch, flush=True)
    sh(['docker', 'ps', '-aq', '--filter', 'name=g1c-%s-' % arm.lower()])
if __name__ == '__main__': main()
