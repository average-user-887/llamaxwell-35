#!/usr/bin/env python3
"""Card-2 S3: logprob-dump parity (PROXY, not KLD). a1-style path: fresh serial server per arm, LLAMA_MV2_DUMP
top-8 logprobs for decode steps 0..96, the 5 a1 requests (miss_param_1, base_5, long_context_74, base_63 T1+T2).
Env = native BFCL run env minus GGML_CUDA_OCCAMY_DEBUG and GGML_CUDA_MAXWELL_GRAPHS (system rule).
usage: s3_lp.py LABEL:UB:CKPT[:K=V,...] ...   (arms run in the order given, one lease)"""
import sys, json, time, subprocess, threading, os
from pathlib import Path
sys.dont_write_bytecode = True
R3 = Path("~/.gemini/antigravity/brain/5dde1567-112b-432c-a28b-3e2e14940eab/scratch/r3-bench")
sys.path.insert(0, str(R3))
import r3_bench as rb
OUT = Path("/tmp/qa-prefill-card/s3"); OUT.mkdir(parents=True, exist_ok=True)
SRC = Path("/tmp/qa-mv2-replay")
NATIVE = R3 / "card2-agentic/runs/r3-bfcl-full-r3-lossless-native-20261121T013713Z-482738"
base_env = [tuple(l.split("=", 1)) for l in (NATIVE / "environment.args").read_text().splitlines()
            if l.strip() and not l.startswith(("GGML_CUDA_OCCAMY_DEBUG", "GGML_CUDA_MAXWELL_GRAPHS"))]
base_env += [("LLAMA_MV2_DUMP", "/out/mv2.log"), ("LLAMA_MV2_DUMP_LO", "0"), ("LLAMA_MV2_DUMP_HI", "96")]
a = [x for x in (NATIVE / "server.args").read_text().split("\n") if x != ""]
base_args = a[:a.index("--spec-type")] + ["--spec-type", "none"]
payloads = json.load(open(SRC / "payloads.private.json"))
order = ["multi_turn_miss_param_1", "multi_turn_base_5", "multi_turn_long_context_74", "multi_turn_base_63"]
arms = []
for spec in sys.argv[1:]:
    p = spec.split(":"); extra = [tuple(kv.split("=", 1)) for kv in p[3].split(",")] if len(p) > 3 and p[3] else []
    arms.append((p[0], int(p[1]), p[2], extra))
uuid = rb.resolve_gpu_uuid()
def temp():
    r = subprocess.run(["nvidia-smi", "-i", uuid, "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"], capture_output=True, text=True)
    try: return int(r.stdout.strip().splitlines()[0])
    except Exception: return -1
dso = rb.resolve_dso_map("card1", rb.DEFAULT_DSO_DIR)
pid = str(os.getpid()); state = dict(tmax=0, tripped=False); stop = threading.Event()
def wd():
    with open(OUT / ("temp-%s.log" % pid), "a") as f:
        while not stop.is_set():
            t = temp(); state["tmax"] = max(state["tmax"], t); f.write("%d %d\n" % (time.time(), t)); f.flush()
            if t >= 80:
                state["tripped"] = True
                for n in subprocess.run(["docker", "ps", "--format", "{{.Names}}"], capture_output=True, text=True).stdout.split():
                    if n.startswith("r3bench-serial-") and ("-%s-" % pid) in n: subprocess.run(["docker", "rm", "-f", n], capture_output=True)
                f.write("TRIP %d\n" % t); return
            stop.wait(2.0)
allres = []
with rb.gpu_lease(int(time.time()) + 600 * len(arms) + 300):
    print("admission:", rb.admit_idle_fleet().split("GPU-")[0][:200], flush=True)
    th = threading.Thread(target=wd, daemon=True); th.start()
    for label, ub, ckpt, extra in arms:
        t0 = time.time()
        while temp() > 55 and time.time() - t0 < 300: time.sleep(5)
        d = OUT / label; d.mkdir(parents=True, exist_ok=True)
        sa = list(base_args); sa[sa.index("-ub") + 1] = str(ub); sa[sa.index("--ctx-checkpoints") + 1] = ckpt
        rb.build_server_args = lambda *x, _s=sa, **k: _s
        res = []; tstart = temp()
        with rb.ServerSession(dso, d, "serial", 84224, ub, 1024, base_env + extra, health_timeout=150) as srv:
            for sid in order:
                for rq in payloads[sid]:
                    body = {"messages": rq["messages"], "tools": rq["tools"], "temperature": 0, "seed": 0, "max_tokens": 128,
                            "cache_prompt": True, "parallel_tool_calls": True}
                    r = srv.post_json("v1/chat/completions", body, timeout=300); ch = r["choices"][0]
                    res.append({"sid": sid, "turn": rq["turn"], "timings": {k: r["timings"].get(k) for k in ("cache_n", "prompt_n", "predicted_n", "prompt_ms")},
                                "finish": ch.get("finish_reason"), "content": ch["message"].get("content"), "tool_calls": ch["message"].get("tool_calls")})
            rc = srv.stop_graceful()
        allres.append({"arm": label, "ub": ub, "ckpt": ckpt, "extra": extra, "rc": rc, "temp_start": tstart, "tmax_sofar": state["tmax"], "reqs": res})
        json.dump(allres, open(OUT / "results.private.json", "w"), indent=1)
        print(label, "rc", rc, "tmax", state["tmax"], [(x["sid"][11:], x["turn"], x["timings"]["cache_n"], x["timings"]["prompt_n"], x["timings"]["predicted_n"]) for x in res], flush=True)
        if state["tripped"]: break
    stop.set(); th.join(5)
print("S3DONE tmax=%d tripped=%s" % (state["tmax"], state["tripped"]), flush=True)
