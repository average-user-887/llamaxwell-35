#!/usr/bin/env python3
"""C3 detached chain: NODE_TIMING profile -> gate (attn >=15%) -> a0 B0 -> sweep1 (v1 lib) -> ABBA n=2 best vs B0
-> 2-slot AB(BA) -> scoring (logprob/KLD proxy from recorded top-8 probs). Each run holds the GPU lock only for itself
(c3_append.run_arm). No new runs after QA_STOP_EPOCH (08:15); driver exits by 08:50."""
import json, os, sys, time, threading, hashlib, subprocess, traceback
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import c3_append as c3  # noqa: E402
rb = c3.rb
BASE = Path(os.environ.get('QA_CHAIN', '/tmp/qa-speed-cards/c3/runs/chain'))
BASE.mkdir(parents=True, exist_ok=True)
V1 = '/tmp/qa-speed-cards/c3/build/v1/out'
EXIT_BY = time.mktime(time.strptime(time.strftime('%Y-%m-%d') + ' 08:50', '%Y-%m-%d %H:%M'))
SCORE = [sys.executable, str(Path(__file__).resolve().parent / 'c3_score.py'), str(BASE)]


def log(s):
    line = '%s %s' % (time.strftime('%H:%M:%S'), s)
    print(line, flush=True)
    with open(BASE / 'driver.log', 'a') as f: f.write(line + '\n')


def dso_for(d):
    m = rb.resolve_dso_map('card1', rb.DEFAULT_DSO_DIR)
    if d: m['libggml-cuda.so'] = Path(d) / 'libggml-cuda.so'
    return m


def late():
    return c3.STOP_AT and time.time() > c3.STOP_AT


def run_arm2(tag, env_over, dso):
    """2-slot check: -np 2 (49152/slot), each slot gets its own 40960 prefix, then appends 4096,3072 fired on
    both slots concurrently; aggregate tok/s = total prompt tokens / wall."""
    uuid = rb.resolve_gpu_uuid(); od = c3.OUT / tag; od.mkdir(parents=True, exist_ok=True)
    env = [(k, v) for k, v in c3.base_env() if k not in env_over] + sorted(env_over.items())
    rec = dict(tag=tag, env_over=env_over, np=2, dso_cuda=str(dso['libggml-cuda.so']).replace(str(rb.PROTO_DEV), '<scratch>'))
    t0 = time.time()
    with c3.lease():
        rec['lease_wait_s'] = round(time.time() - t0, 1)
        if late(): rec['skipped'] = 'stop time'; return rec
        tc = time.time()
        while c3.gpu_q(uuid)[0] > 60 and time.time() - tc < 600: time.sleep(10)
        sess = c3.Sess(dso, od, 'native', c3.CTX, 1024, 1024, env, health_timeout=420, tag=tag)
        wd = None
        try:
            with sess:
                wd = c3.Watch(uuid, sess.name, od / 'temp.log'); wd.start()
                pre = [c3.filler_tokens(sess, 40960, s) for s in (0, 11)]
                ext = [c3.filler_tokens(sess, 7168 + 64, s) for s in (37, 53)]
                for i in (0, 1): c3.complete(sess, pre[i], slot=i)
                prompts = [list(p) for p in pre]; off = 0; rows = []
                for n in (4096, 3072):
                    wd.allow_pause = False
                    tc = time.time()
                    while (wd.paused or c3.gpu_q(uuid)[0] > c3.APPEND_COOL_C) and time.time() - tc < 240: time.sleep(2)
                    for i in (0, 1): prompts[i] = prompts[i] + ext[i][off:off + n]
                    off += n; res = [None, None]
                    def go(i): res[i] = c3.complete(sess, prompts[i], slot=i)
                    ts = time.time(); th = [threading.Thread(target=go, args=(i,)) for i in (0, 1)]
                    [t.start() for t in th]; [t.join() for t in th]; wall = time.time() - ts
                    wd.allow_pause = True
                    pn = sum(r['timings']['prompt_n'] for r in res)
                    rows.append(dict(n=n, wall_s=round(wall, 3), prompt_n=pn, agg_tok_s=round(pn / wall, 2),
                                     per_slot_ms=[r['timings']['prompt_ms'] for r in res]))
                    log('%s 2slot append %d: %d tok in %.2f s = %.1f tok/s' % (tag, n, pn, wall, pn / wall))
                rec['appends2'] = rows
                rec['agg_tok_s'] = round(sum(r['prompt_n'] for r in rows) / sum(r['wall_s'] for r in rows), 2)
        except Exception as ex:
            rec['error'] = str(ex)[:400]; log('%s exception: %s' % (tag, str(ex)[:300]))
        finally:
            if wd: wd.stop.set(); wd.join(10)
        if wd: rec.update(tmax=wd.tmax, vram_peak_mib=wd.mmax, tripped=wd.tripped, invalid=bool(wd.tripped or wd.tmax >= c3.HARD_C))
        slog = od / 'server.log'
        if slog.exists():
            txt = slog.read_text(errors='replace'); rec['cuda_errors'] = len(rb.CUDA_ERROR_RE.findall(txt)); rec['oom'] = 'out of memory' in txt
    return rec


def run(stage, tag, env, dso_dir, np=1):
    c3.OUT = BASE / stage; c3.OUT.mkdir(parents=True, exist_ok=True)
    if late():
        log('skip %s/%s: past stop time' % (stage, tag)); return None
    log('queue %s/%s env=%s dso=%s np=%d' % (stage, tag, env, 'v1' if dso_dir else 'card1', np))
    c3.NP = np
    try:
        rec = c3.run_arm(tag, env, dso_for(dso_dir), False, log) if np == 1 else run_arm2(tag, env, dso_for(dso_dir))
    finally:
        c3.NP = 1
    rec.update(stage=stage, dso='v1' if dso_dir else 'card1')
    with open(c3.OUT / 'results.jsonl', 'a') as f: f.write(json.dumps(rec) + '\n')
    log('done %s/%s append=%s agg2=%s tmax=%s invalid=%s err=%s wait=%s' % (stage, tag, rec.get('append_tok_s'), rec.get('agg_tok_s'),
        rec.get('tmax'), rec.get('invalid'), rec.get('error'), rec.get('lease_wait_s')))
    subprocess.run(SCORE, capture_output=True, timeout=120)
    return rec


def ok(r):
    return r and not r.get('error') and not r.get('invalid') and not r.get('skipped') and r.get('append_tok_s')


def attn_share(path):
    nt = rb.parse_node_timing(Path(path))
    cls = dict(attn=0.0, gdn=0.0, moe=0.0, other=0.0)
    # biggest phase = prefill
    ph = max(nt['phases'], key=lambda p: nt['phases'][p]['total_node_ms'])
    allrows = []
    for line in Path(path).read_text().splitlines():
        if line.startswith('#') or not line.strip(): continue
        p, op, calls, ms, *_ = line.split('\t')
        if p == ph: allrows.append((op, float(ms)))
    for op, ms in allrows:
        u = op.upper()
        if 'FLASH_ATTN' in u: cls['attn'] += ms
        elif any(k in u for k in ('DELTA', 'SSM', 'GDN', 'RWKV')): cls['gdn'] += ms
        elif 'MUL_MAT_ID' in u or 'MOE' in u: cls['moe'] += ms
        else: cls['other'] += ms
    tot = sum(cls.values()) or 1.0
    meas = cls['attn'] / tot
    # project to the append window (FA ~ depth; rest ~ tokens): setup 81920 tok @avg 41K, appends 9216 @avg 84.5K
    ws, wa = 81920 * 41000.0, 9216 * 84500.0
    fa_a = cls['attn'] * wa / (ws + wa); o_a = (tot - cls['attn']) * 9216 / (81920 + 9216)
    est = fa_a / (fa_a + o_a)
    return dict(phase=ph, ms=cls, total_ms=tot, share_measured=meas, share_append_est=est,
                top=[(o, round(m, 1)) for o, m in sorted(allrows, key=lambda x: -x[1])[:12]])


def main():
    log('driver start pid=%d stop_epoch=%s' % (os.getpid(), c3.STOP_AT))
    # 1. profile (NODE_TIMING; per-node sync inflates totals, shares only)
    nt_env = {'GGML_CUDA_NODE_TIMING': '/out/node_timing.tsv', 'GGML_CUDA_NODE_TIMING_SPLIT': '1', 'LLAMA_MV3_PHASE': '/out/phase.tsv'}
    run('profile', 'nt', nt_env, None)
    p = BASE / 'profile' / 'nt' / 'node_timing.tsv'
    share = None
    if p.is_file():
        try:
            share = attn_share(p); (BASE / 'profile' / 'breakdown.json').write_text(json.dumps(share, indent=1))
            log('breakdown: %s measured attn=%.1f%% est-append attn=%.1f%%' % (
                {k: round(v) for k, v in share['ms'].items()}, 100 * share['share_measured'], 100 * share['share_append_est']))
        except Exception:
            log('breakdown parse failed: ' + traceback.format_exc()[-300:])
    else:
        log('node_timing.tsv missing; gate not evaluable, continuing')
    if share and max(share['share_measured'], share['share_append_est']) < 0.15:
        log('GATE: attention <15%% of append time -> stop chain (valid result)'); return
    # 2. a0 B0
    run('b0', 'a0', {}, None)
    # 3. sweep1 on v1 lib
    sw = [('d', {}), ('s2048', {'GGML_CUDA_FA_MAXWELL_GEMM_SPLIT': '2048'}), ('t128', {'GGML_CUDA_FA_MAXWELL_GEMM_TQ': '128'}),
          ('t128s2048', {'GGML_CUDA_FA_MAXWELL_GEMM_TQ': '128', 'GGML_CUDA_FA_MAXWELL_GEMM_SPLIT': '2048'}),
          ('s8192', {'GGML_CUDA_FA_MAXWELL_GEMM_SPLIT': '8192'}), ('d2', {})]
    res = {t: run('sweep1', t, e, V1) for t, e in sw}
    ctl = [res[t]['append_tok_s'] for t in ('d', 'd2') if ok(res.get(t))]
    if not ctl:
        log('sweep1: no valid control arm; stop'); return
    c = sum(ctl) / len(ctl)
    cand = sorted(((res[t]['append_tok_s'] / c - 1, t) for t, _ in sw if t not in ('d', 'd2') and ok(res.get(t))), reverse=True)
    log('sweep1 vs v1-default %.2f tok/s: %s' % (c, ['%s %+.2f%%' % (t, 100 * g) for g, t in cand]))
    if not cand or cand[0][0] < 0.01:
        log('KILL: best sweep gain <1%% -> stop chain'); return
    best = cand[0][1]; benv = dict(sw)[best]
    (BASE / 'best.json').write_text(json.dumps(dict(tag=best, env=benv, sweep_gain=cand[0][0])))
    # 4. ABBA n=2, A = v1+best env, B = B0 (card1 lib)
    for i, ab in enumerate('ABBAABBA'):
        run('abba', '%s%d' % (ab, i), benv if ab == 'A' else {}, V1 if ab == 'A' else None)
    # 5. 2-slot check (AB, then BA if time allows)
    for i, ab in enumerate('ABBA'):
        if i == 2 and time.time() > c3.STOP_AT - 1800:
            log('2slot: skipping second AB pair (time)'); break
        run('slot2', '%s%d' % (ab, i), benv if ab == 'A' else {}, V1 if ab == 'A' else None, np=2)
    log('chain complete')


if __name__ == '__main__':
    try:
        main()
    except Exception:
        log('driver crash: ' + traceback.format_exc()[-600:])
    finally:
        subprocess.run(SCORE, capture_output=True, timeout=120)
        for n in subprocess.run(['docker', 'ps', '-a', '--format', '{{.Names}}'], capture_output=True, text=True).stdout.split():
            if n.startswith('r3bench-c3-') and ('-%d' % os.getpid()) in n:
                subprocess.run(['docker', 'rm', '-f', n], capture_output=True)
        log('driver exit')
