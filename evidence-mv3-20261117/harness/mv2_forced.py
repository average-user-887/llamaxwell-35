# MV2 forced-prefix probe: usage mv2_forced.py HOST:PORT out.json ref_B_result.json
# Teacher-forces the first-divergence prefix (B tokens) into a NO-SPEC server in different batch shapes and records top candidates.
import json,sys,glob,urllib.request as U
hp,out,refp=sys.argv[1:4]; B='http://'+hp
def post(p,b):
    with U.urlopen(U.Request(B+p,json.dumps(b).encode(),{'Content-Type':'application/json'}),timeout=7200) as f: return json.loads(f.read())
txt=''.join(open(f,errors='ignore').read() for f in sorted(glob.glob('/usr/lib/python3.12/*.py')))
P=post('/tokenize',{'content':txt[:80000*6+60000]})['tokens'][:80000]
ref=json.load(open(refp))['runs']['cold']['slots']['0']['out_tokens']; d=26
def q(prompt,cache):
    r=post('/completion',{'prompt':prompt,'n_predict':1,'temperature':0,'n_probs':5,'cache_prompt':cache,'id_slot':0})
    return {'prompt_n':r['timings']['prompt_n'],'cand':[(c['id'],c['token'],round(c['logprob'],4)) for c in r['completion_probabilities'][0]['top_logprobs']]}
R={'d':d,'ref_tok':ref[d],'ref_prefix_tail':ref[d-3:d]}
R['full_prefill_cold']=q(P+ref[:d],False)            # chunked prefill shape
q(P+ref[:d-2],True)                                    # cache P+prefix[:d-2]
R['two_row_cached']=q(P+ref[:d],True)                  # last 2 rows evaluated as one 2-row batch (verify shape)
q(P+ref[:d-1],True)
R['one_row_cached']=q(P+ref[:d],True)                  # decode shape (1 row)
R['full_prefill_cold_again']=q(P+ref[:d],False)
R['done']=True
json.dump(R,open(out,'w'),indent=1)
