# C5 D client: greedy completions against the bench server on 127.0.0.1:$SRV_PORT. Fixtures: Austen, Pride and Prejudice (public domain).
import json, sys, time, hashlib, threading, urllib.request
import os; U = "http://127.0.0.1:" + os.environ["SRV_PORT"]
AU = ("It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife. "
      "However little known the feelings or views of such a man may be on his first entering a neighbourhood, this truth is so well "
      "fixed in the minds of the surrounding families, that he is considered the rightful property of some one or other of their daughters.")
def post(path, body, t=3600):
    r = urllib.request.Request(U + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=t) as f: return json.loads(f.read())
def comp(prompt, n=128, slot=-1, cache=False):
    t0 = time.time()
    r = post("/completion", {"prompt": prompt, "n_predict": n, "temperature": 0, "top_k": 1, "seed": 0, "cache_prompt": cache,
                             "id_slot": slot, "return_tokens": True, "ignore_eos": True})
    toks = r.get("tokens") or []
    tm = r.get("timings", {})
    return {"sha": hashlib.sha256(json.dumps(toks).encode()).hexdigest()[:16], "ntok": len(toks), "prompt_n": tm.get("prompt_n"),
            "cache_n": tm.get("cache_n"), "tg": round(tm.get("predicted_per_second", 0), 3), "pp": round(tm.get("prompt_per_second", 0), 1),
            "wall": round(time.time() - t0, 2), "head": r.get("content", "")[:60].replace("\n", " ")}
def filler(nsec): return "".join(f"Section {i}. {AU}\n" for i in range(1, nsec + 1)) + "\nQuestion: In one paragraph, what is the claim made in the sections above?\nAnswer:"
def ntok(p): return len(post("/tokenize", {"content": p})["tokens"])
def out(k, v): print(k, json.dumps(v), flush=True)
mode = sys.argv[1]
if mode == "quick":
    for i in range(3): out(f"quick{i}", comp(f"Chapter {i+1}. {AU}", 64, 0))
    sys.exit(0)
out("short", comp("Chapter 1. " + AU, 128, 0))
for nm, ns in (("d16k", 190), ("d64k", 760)):
    p = filler(ns); out(nm + "_prompt_tokens", ntok(p)); out(nm, comp(p, 128, 0))
for rep in (1, 2):
    res = {}
    def w(s): res[s] = comp(f"Chapter {s+5}. {AU}", 128, s)
    t0 = time.time(); th = [threading.Thread(target=w, args=(s,)) for s in (0, 1)]
    [x.start() for x in th]; [x.join() for x in th]; wall = time.time() - t0
    out(f"two_slot_rep{rep}", {"slot0": res[0], "slot1": res[1], "agg_tok_s": round((res[0]["ntok"] + res[1]["ntok"]) / wall, 2), "wall": round(wall, 2)})
P1 = "Chapter 3. " + AU * 3
out("ss_direct", comp(P1, 64, 0))
out("ss_prime", comp(P1, 1, 0, True))
out("ss_save", post("/slots/0?action=save", {"filename": "s0.bin"}))
out("ss_overwrite", comp("Chapter 9. Unrelated text to overwrite the slot.", 8, 0, False))
out("ss_erase", post("/slots/0?action=erase", {}))
out("ss_restore", post("/slots/0?action=restore", {"filename": "s0.bin"}))
out("ss_cont", comp(P1, 64, 0, True))
