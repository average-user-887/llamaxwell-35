#!/usr/bin/env python3
"""N3R fixture builder (CPU only). Reads recent Claude Code session transcripts (~/.claude/projects/*/*.jsonl) and writes TEXT fixtures to <B0>/private/fixture/.
Tokenisation happens at run time through the server /tokenize (runner caches it under private/tokcache). Private: never publish private/.
usage: build_fixture.py [--days 14] [--prefix-chars 420000] [--seed 7]"""
import os, sys, json, glob, time, hashlib, random
B0 = os.path.dirname(os.path.abspath(__file__)); OUT = B0 + '/private/fixture'
DAYS = float(sys.argv[sys.argv.index('--days') + 1]) if '--days' in sys.argv else 14
PCH = int(sys.argv[sys.argv.index('--prefix-chars') + 1]) if '--prefix-chars' in sys.argv else 420000
SEED = int(sys.argv[sys.argv.index('--seed') + 1]) if '--seed' in sys.argv else 7
TURN_KINDS = {0: ['edit/copy-heavy', 'tool-call', 'prose', 'edit/copy-heavy', 'tool-call', 'prose'], 1: ['tool-call', 'prose', 'edit/copy-heavy', 'tool-call', 'prose', 'edit/copy-heavy']}
SUFFIX = {'edit/copy-heavy': '\n\nRewrite the code or text above in full, changing only the first function or paragraph so that it validates its input. Output the complete result.',
          'tool-call': '\n\nNow emit exactly one JSON tool call object with the fields name and arguments that would inspect the most relevant item above.',
          'prose': '\n\nExplain in plain prose what the above shows, what is wrong or unfinished, and what the next step should be.'}
EDIT_TOOLS = {'Edit', 'Write', 'MultiEdit', 'NotebookEdit'}
def blocks(c): return [{'type': 'text', 'text': c}] if isinstance(c, str) else (c or [])
def rtext(c):
    if isinstance(c, str): return c
    return '\n'.join(b.get('text', '') for b in (c or []) if isinstance(b, dict) and b.get('type') == 'text')
def render(path):
    """-> list of (kind, text) items in order; kind in edit/copy-heavy, tool-call, prose."""
    names = {}; items = []
    for ln in open(path, errors='replace'):
        try: j = json.loads(ln)
        except Exception: continue
        if j.get('type') not in ('user', 'assistant') or j.get('isSidechain'): continue
        role = j['type']
        for b in blocks((j.get('message') or {}).get('content')):
            if not isinstance(b, dict): continue
            t = b.get('type')
            if t == 'text' and len(b.get('text', '')) > 40: items.append(('prose', '[%s]\n%s\n' % (role, b['text'])))
            elif t == 'tool_use':
                names[b.get('id')] = b.get('name'); inp = b.get('input') or {}
                if b.get('name') in EDIT_TOOLS:
                    s = '[tool_use %s %s]\n' % (b['name'], inp.get('file_path', ''))
                    if 'old_string' in inp: s += '--- old\n%s\n+++ new\n%s\n' % (inp.get('old_string', ''), inp.get('new_string', ''))
                    else: s += str(inp.get('content', inp.get('new_source', ''))) + '\n'
                    items.append(('edit/copy-heavy', s))
                else: items.append(('tool-call', '[tool_use %s]\n%s\n' % (b.get('name'), json.dumps(inp, ensure_ascii=False)[:3000])))
            elif t == 'tool_result':
                nm = names.get(b.get('tool_use_id'), '?'); x = rtext(b.get('content'))[:6000]
                if len(x) < 20: continue
                items.append(('edit/copy-heavy' if nm in ('Read', 'Edit', 'Write', 'MultiEdit') else 'tool-call', '[tool_result %s]\n%s\n' % (nm, x)))
    return items
def main():
    rnd = random.Random(SEED); now = time.time()
    fs = [f for f in glob.glob(os.path.expanduser('~/.claude/projects/*/*.jsonl')) if now - os.path.getmtime(f) < DAYS * 86400 and os.path.getsize(f) > 150000]
    fs.sort(); rnd.shuffle(fs); print('candidate sessions', len(fs))
    os.makedirs(OUT, exist_ok=True); pools = {'edit/copy-heavy': [], 'tool-call': [], 'prose': []}; prefix = {0: '', 1: ''}; used = {0: [], 1: []}; pool_files = []
    for f in fs:
        try: it = render(f)
        except Exception: continue
        if not it: continue
        a = 0 if len(prefix[0]) <= len(prefix[1]) else 1
        if len(prefix[a]) < PCH: prefix[a] += ''.join(t for _, t in it); used[a].append(os.path.basename(f)); continue
        for k, t in it: pools[k].append(t)
        pool_files.append(os.path.basename(f))
        if all(sum(map(len, p)) > 400000 for p in pools.values()) and len(pool_files) >= 6: break
    assert len(prefix[0]) >= PCH and len(prefix[1]) >= PCH, 'not enough transcript text for prefixes'
    turns = {0: [], 1: []}; cursor = {k: 0 for k in pools}
    for a in (0, 1):
        for i, kind in enumerate(TURN_KINDS[a]):
            need = 24000 if i % 2 else 7000; s = ''; p = pools[kind]
            while len(s) < need and cursor[kind] < len(p): s += p[cursor[kind]]; cursor[kind] += 1
            assert len(s) >= need * 0.8, ('pool short', kind, len(s)); turns[a].append({'turn': i, 'kind': kind, 'size_tok': 4096 if i % 2 else 1024, 'text': s[:need], 'suffix': SUFFIX[kind]})
    for a in (0, 1): open('%s/prefix%d.txt' % (OUT, a), 'w').write(prefix[a][:PCH])
    meta = {'built_utc': time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()), 'seed': SEED, 'prefix_chars': PCH, 'prefix_sessions': used, 'turn_sessions': len(pool_files),
            'turns': {str(a): [{k: v for k, v in t.items() if k != 'text'} | {'chars': len(t['text']), 'sha': hashlib.sha256(t['text'].encode()).hexdigest()[:16]} for t in turns[a]] for a in (0, 1)}}
    json.dump({str(a): turns[a] for a in (0, 1)}, open(OUT + '/turns.json', 'w')); json.dump(meta, open(OUT + '/meta.json', 'w'), indent=1)
    print(json.dumps({k: v for k, v in meta.items() if k != 'prefix_sessions'}, indent=1))
main()
