import json,sys,hashlib
W="/media/avg-usr/Storage/m6000-llama-cpp/occamy35-moonshot-20261110"
rid=sys.argv[1]; aenv=[e for e in sys.argv[2].split(",") if e]; benv=[e for e in sys.argv[3].split(",") if e]
cfg=json.load(open(W+"/harness/c04/config.c04.json"))
lib=f"{W}/cand/C24/build/bin/libggml-cuda.so"; h=hashlib.sha256(open(lib,"rb").read()).hexdigest()
cfg["run_id"]=rid; cfg["out_dir"]=f"{W}/results/C24/raw/g4-{rid}"
cfg["modes"]["d4096"]={"args":["-p","0","-n","128","-r","3","-d","4096"],"metric":"bench_tps","warmup_per_arm":1}
cfg["arms"]={"A":{"lib":lib,"env":aenv,"expected":{"lib_sha256":h}},"B":{"lib":lib,"env":benv,"expected":{"lib_sha256":h}}}
fn=f"{W}/results/C24/raw/cfg-{rid}.json"; json.dump(cfg,open(fn,"w"),indent=1); print(fn)
