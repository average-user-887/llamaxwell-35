#!/usr/bin/env python3
"""N3R: N3 two-slot replay with REAL generation (no ignore_eos, temperature 0, n_predict cap 800). Talks to an already running llama-server (no docker, no GPU control).
Same schedule as n3_replay: 2 slots, start depth 78000 (at ctx 98304), 6 turns alternating 1K/4K appends, slot1 offset 25 s, think 1 s, generated tokens appended to history.
--ctx C cuts depth: start = 78000-(98304-C), and n_predict is clipped so depth never exceeds C-64. Private fixtures only under B0/private/.
usage: n3r_replay.py --url URL --arm NAME --out DIR [--ctx 98304] [--turns 6] [--start N] [--n-predict 800] [--temp 0 --seed 1] [--slots NAME] [--server-log F]
       [--extra-body JSON] [--offset 25] [--think 1] [--dry-run] [--max-turns-smoke N]"""
import sys, os, json, time, hashlib, threading, argparse, re, http.client, urllib.parse
B0 = os.path.dirname(os.path.abspath(__file__)); FX = B0 + '/private/fixture'; TC = B0 + '/private/tokcache'
ap = argparse.ArgumentParser()
ap.add_argument('--url', default='http://127.0.0.1:8080'); ap.add_argument('--arm', default='arm'); ap.add_argument('--out', required=True)
ap.add_argument('--ctx', type=int, default=98304); ap.add_argument('--turns', type=int, default=6); ap.add_argument('--start', type=int, default=0)
ap.add_argument('--n-predict', type=int, default=800); ap.add_argument('--temp', type=float, default=0.0); ap.add_argument('--seed', type=int, default=-1)
ap.add_argument('--slots', default=''); ap.add_argument('--server-log', default=''); ap.add_argument('--extra-body', default='{}')
ap.add_argument('--raw', action='store_true', help='old behaviour: raw text, no ChatML wrapper (B1 showed EOS-at-once / loops)'); ap.add_argument('--old-suffix', action='store_true'); ap.add_argument('--fail-void', action='store_true')
ap.add_argument('--offset', type=float, default=25); ap.add_argument('--think', type=float, default=1.0); ap.add_argument('--dry-run', action='store_true')
A = ap.parse_args()
u = urllib.parse.urlparse(A.url); HOST, PORT = u.hostname, u.port or 80
START = A.start or (78000 - (98304 - A.ctx)); SIZES = (1024, 4096)
# B0c: every turn ends with an explicit instruction that forces a long natural-language continuation (B1/B1c showed EOS after 1-35 tokens on 'emit one JSON tool call' / 'rewrite in full' prompts and copy loops).
LONG = {'edit/copy-heavy': 'Review the code or text above in detail. Quote only short snippets, then explain in connected prose how you would rewrite the first function or paragraph so that it validates its input, and why each change is needed.',
        'tool-call': 'Describe in detail the single tool call you would make next to inspect the most relevant item above, which arguments you would pass, what result you expect, and how you would interpret that result.',
        'prose': 'Explain in plain prose what the above shows, what is wrong or unfinished, and what the next step should be.'}
TAIL = ' Continue in detail, at least 300 words, no lists, no code blocks, in English.'
def long_suffix(kind): return '\n\n---\nInstruction: ' + LONG[kind] + TAIL
def now(): return time.time()
# ChatML wrapper (Qwen3.x template, empty think block = no thinking). B1 (raw text, no template) gave slot0 EOS after 1-23 tokens and slot1 an 800-token loop.
SYS = '<|im_start|>system\nYou are a coding agent working in a repository. The earlier session history follows.<|im_end|>\n<|im_start|>user\n'
CLOSE = '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'; REOPEN = '<|im_end|>\n<|im_start|>user\n'
def void_reason(gen):
    if len(gen) < 64: return 'short(<64)'
    g4 = [tuple(gen[i:i + 4]) for i in range(len(gen) - 3)]
    if g4 and 1 - len(set(g4)) / len(g4) > 0.5: return 'loop(>50% repeated 4grams)'
    return ''
def call(path, body=None, method='POST', timeout=3600):
    c = http.client.HTTPConnection(HOST, PORT, timeout=timeout)
    c.request(method, path, json.dumps(body) if body is not None else None, {'Content-Type': 'application/json'}); r = c.getresponse(); d = r.read(); c.close()
    try: return r.status, json.loads(d)
    except Exception: return r.status, {'raw': d[:2000].decode(errors='replace')}
def sha(o): return hashlib.sha256(json.dumps(o).encode()).hexdigest()[:16]
def tok(text):
    os.makedirs(TC, exist_ok=True); p = '%s/%s.json' % (TC, hashlib.sha256(text.encode()).hexdigest()[:24])
    if os.path.exists(p): return json.load(open(p))
    st, j = call('/tokenize', {'content': text, 'add_special': False}); assert st == 200 and 'tokens' in j, ('tokenize failed', st, str(j)[:200])
    json.dump(j['tokens'], open(p, 'w')); return j['tokens']
def plan():
    T = json.load(open(FX + '/turns.json')); out = {}
    for a in (0, 1):
        out[a] = [(t['turn'], t['kind'], t['size_tok'], t['text'], t['suffix'] if A.old_suffix else long_suffix(t['kind'])) for t in T[str(a)][:A.turns]]
    return out
def tools_for(a, pl):
    ch = []
    for (k, kind, n, text, suf) in pl[a]:
        s = tok(suf) if A.raw else tok(suf + CLOSE); body = tok(text); assert len(body) >= n - len(s), ('turn text too short', a, k, len(body)); ch.append((kind, body[:n - len(s)] + s))
    return ch
def prefix_for(a):
    ids = tok(open('%s/prefix%d.txt' % (FX, a)).read() if A.raw else SYS + open('%s/prefix%d.txt' % (FX, a)).read()); assert len(ids) >= START, ('prefix too short', a, len(ids), START); return ids[:START]
class Req(threading.Thread):
    def __init__(s, ids, slot, n, tag):
        super().__init__(daemon=True); s.ids, s.slot, s.n, s.tag = ids, slot, n, tag; s.ev = []; s.final = None; s.err = None; s.status = None; s.body = None; s.t_send = s.t_end = None
    def run(s):
        try:
            c = http.client.HTTPConnection(HOST, PORT, timeout=3600)
            b = {'prompt': s.ids, 'n_predict': s.n, 'id_slot': s.slot, 'cache_prompt': True, 'temperature': A.temp, 'stream': True, 'return_tokens': True}   # NO ignore_eos
            if A.seed >= 0: b['seed'] = A.seed
            b.update(json.loads(A.extra_body)); s.t_send = now(); c.request('POST', '/completion', json.dumps(b), {'Content-Type': 'application/json'}); r = c.getresponse(); s.status = r.status
            if r.status != 200: s.body = r.read()[:600].decode(errors='replace'); return
            for line in r:
                if not line.startswith(b'data:') or line[5:].strip() == b'[DONE]': continue
                j = json.loads(line[5:]); t = now()
                if j.get('stop'): s.final = j
                else: s.ev.append((t, j.get('content', ''), j.get('tokens', [])))
        except Exception as ex: s.err = repr(ex)
        finally: s.t_end = now()
    def rec(s):
        f = s.final or {}; tm = f.get('timings', {}); ids = [i for e in s.ev for i in e[2]]; np_ = f.get('tokens_predicted', tm.get('predicted_n'))
        gaps = [s.ev[i][0] - s.ev[i - 1][0] for i in range(1, len(s.ev))]
        r = {'prompt_len': len(s.ids), 'n_cap': s.n, 't_send': s.t_send, 't_end': s.t_end, 't_first': s.ev[0][0] if s.ev else None, 'http': s.status, 'err': s.err, 'body': s.body,
             'tokens_predicted': np_, 'n_ids_visible': len(ids), 'token_sha': sha(ids), 'prompt_sha': sha(s.ids), 'stop_type': f.get('stop_type'), 'stopped_eos': f.get('stop_type') == 'eos',
             'prompt_n': tm.get('prompt_n'), 'cache_n': tm.get('cache_n'), 'prompt_ms': tm.get('prompt_ms'), 'predicted_ms': tm.get('predicted_ms'), 'predicted_tps': tm.get('predicted_per_second'),
             'draft_n': tm.get('draft_n'), 'draft_n_accepted': tm.get('draft_n_accepted'), 'max_token_gap_s': max(gaps) if gaps else None, 'truncated': f.get('truncated')}
        if r['draft_n']: r['accept_rate'] = (r['draft_n_accepted'] or 0) / r['draft_n']
        r['receipt_ok'] = bool(s.status == 200 and s.final is not None and not s.err and np_ is not None and np_ <= s.n and r['n_ids_visible'] >= 0.9 * np_)
        return r, ids
def metrics():
    try:
        c = http.client.HTTPConnection(HOST, PORT, timeout=10); c.request('GET', '/metrics'); d = c.getresponse().read().decode(errors='replace'); c.close()
        return {l.split()[0]: float(l.split()[-1]) for l in d.splitlines() if l and l[0] != '#' and re.search(r'draft|spec|accept', l) and len(l.split()) >= 2}
    except Exception as ex: return {'err': repr(ex)}
def agent(a, t_start, out, ids, tools, files):
    while now() < t_start: time.sleep(0.01)
    for k, (kind, chunk) in enumerate(tools):
        ids = ids + ([198] if k and A.raw else []) + chunk; n = min(A.n_predict, A.ctx - 64 - len(ids))
        if n < 64: out.append({'agent': a, 'turn': k, 'kind': kind, 'skipped': 'ctx', 'depth': len(ids)}); break
        q = Req(list(ids), a, n, 'a%d_t%d' % (a, k)); q.start(); q.join(3600); rec, gen = q.rec()
        vr = void_reason(gen); rec.update(agent=a, turn=k, kind=kind, append_tok=len(chunk), depth_after_append=len(ids), gen_tokens=len(gen), void=bool(vr), void_reason=vr); out.append(rec)
        json.dump(gen, open('%s/tokens/a%d_t%d.json' % (A.out, a, k), 'w'))
        if q.status != 200 or q.err: return
        if not A.raw:
            END = tok('<|im_end|>')
            if gen[-len(END):] == END: gen = gen[:-len(END)]
            ids = ids + gen + tok(REOPEN)
        else: ids = ids + gen
        time.sleep(A.think)
def spec_lines():
    if not A.server_log or not os.path.exists(A.server_log): return None
    L = [l.strip() for l in open(A.server_log, errors='replace') if 'occamy_spec' in l or 'draft acceptance' in l or 'statistics' in l and 'draft' in l]
    return {'n': len(L), 'last': L[-3:]}
def main():
    os.makedirs(A.out + '/tokens', exist_ok=True); pl = plan()
    per = sum(SIZES[k % 2] + A.n_predict for k in range(A.turns)) + A.turns - 1
    desc = {'arm': A.arm, 'url': A.url, 'ctx': A.ctx, 'start': START, 'turns': A.turns, 'n_predict': A.n_predict, 'temp': A.temp, 'seed': A.seed, 'worst_final_depth': START + per, 'kinds': {a: [p[1] for p in pl[a]] for a in (0, 1)}}
    desc['chat_wrapped'] = not A.raw; print(json.dumps(desc))
    if A.dry_run:
        for a in (0, 1):
            for p in pl[a]: print('shape slot%d turn%d %s size_tok=%d text_chars=%d suffix_chars=%d suffix_has_long_instr=%s' % (a, p[0], p[1], p[2], len(p[3]), len(p[4]), 'at least 300 words' in p[4]))
    if A.dry_run: print('DRY RUN ok: fixtures present, schedule printed, no server contact'); return
    st, _ = call('/health', method='GET', timeout=10); assert st == 200, 'server not healthy'
    files = ['n3r_s%d_%d.bin' % (a, START) for a in (0, 1)]; pre = {a: prefix_for(a) for a in (0, 1)}; tools = {a: tools_for(a, pl) for a in (0, 1)}; R = dict(desc, utc_start=now(), setup=[])
    for a in (0, 1):   # slot state: restore if --slots given and file exists, else prefill (and save when --slots)
        t = now(); ok = False
        if A.slots:
            st, j = call('/slots/%d?action=restore' % a, {'filename': A.slots + '_' + files[a]}); ok = st == 200 and j.get('n_restored', 0) >= START
        if not ok:
            call('/slots/%d?action=erase' % a, {}); q = Req(pre[a], a, 1, 'pre%d' % a); q.start(); q.join(7200); r, g = q.rec(); assert r['http'] == 200, r
            if A.slots: call('/slots/%d?action=save' % a, {'filename': A.slots + '_' + files[a]})
            pre[a] = pre[a] + g if A.slots else pre[a] + g   # the 1 sampled prefix token is part of history, same for every arm at temp 0
        R['setup'].append({'slot': a, 'restored': ok, 's': now() - t})
    m0 = metrics(); res = {0: [], 1: []}; t0 = now() + 2.0; R['t_run0'] = t0
    th = [threading.Thread(target=agent, args=(a, t0 + (0 if a == 0 else A.offset), res[a], pre[a], tools[a], files)) for a in (0, 1)]
    for t in th: t.start()
    for t in th: t.join(7200)
    R['turns_rec'] = [r for a in (0, 1) for r in res[a]]; R['makespan_s'] = max([r.get('t_end') or 0 for r in R['turns_rec']]) - t0
    R['ok'] = len(R['turns_rec']) == 2 * A.turns and all(r.get('receipt_ok') for r in R['turns_rec']); R['n_void'] = sum(1 for r in R['turns_rec'] if r.get('void')); R['chat_wrapped'] = not A.raw; R['metrics_before'] = m0; R['metrics_after'] = metrics(); R['spec_log'] = spec_lines()
    R['run_sha'] = sha([[r.get('token_sha') for r in sorted(R['turns_rec'], key=lambda r: (r['agent'], r['turn']))]]); R['utc_end'] = now()
    json.dump(R, open(A.out + '/run.json', 'w'), indent=1)
    with open(A.out + '/turns.md', 'w') as f:
        f.write('| agent | turn | kind | ctx | prompt | gen | eos | drafted | accepted | decode s | void | token sha |\n|---|---|---|---|---|---|---|---|---|---|---|---|\n')
        for r in sorted(R['turns_rec'], key=lambda r: (r['agent'], r['turn'])):
            f.write('| %s | %s | %s | %d | %s | %s | %s | %s | %s | %s | %s | %s |\n' % (r['agent'], r['turn'], r['kind'], A.ctx, r.get('prompt_len'), r.get('tokens_predicted'), r.get('stopped_eos'), r.get('draft_n'), r.get('draft_n_accepted'),
                  '%.1f' % (r['predicted_ms'] / 1000) if r.get('predicted_ms') else '-', r.get('void_reason') or '-', r.get('token_sha')))
    print('makespan_s=%.1f ok=%s run_sha=%s void_turns=%d/%d' % (R['makespan_s'], R['ok'], R['run_sha'], R['n_void'], len(R['turns_rec'])))
    sys.exit(0 if R['ok'] and not (A.fail_void and R['n_void']) else 3)
main()
