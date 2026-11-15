# Insert globaltimer phase stamps TS(0..7) into kern2.inc -> kern2_ts.inc (header from kern_ts.inc lines 1-3).
s=open('kern2.inc').read().split('\n'); hdr=open('kern_ts.inc').read().split('\n')[:3]
out=[]; bars=0
for l in s:
    if l == '    grid_barrier(s0);':
        out.append('    TS(%d); grid_barrier(s0); TS(%d);' % ((1, 2) if bars == 0 else (4, 5))); bars += 1; continue
    if '// pass 2: chunks' in l: out.append('        TS(6);')
    out.append(l)
    if l.strip() == 'const Layer & L = PP.p[0].L;': out.append('    TS(0);')
j=[i for i,l in enumerate(out) if 'TS(4)' in l][0]; assert out[j-1] == '    }'; out.insert(j-1, '        if (k == 0) TS(3);')
while out[-1] == '': out.pop()
assert out[-1] == '}' and bars == 2; out.insert(len(out)-1, '    TS(7);')
open('kern2_ts.inc','w').write('\n'.join(hdr+out)+'\n')
