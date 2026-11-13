#!/bin/bash
# Q8S VRAM: BUNDLE3 server config, -np 2 -c 196608, only cache types differ; read KV buffer from load log + nvidia-smi after load. Caller holds gpu lock. usage: vram_q8.sh VARIANT:CTK:CTV ...
M=<LAB>; Q=$M/occamy35-night-20261113/Q8S; GPU=${Q8S_GPU:?}
RS=$M/run-35b-bundle3-20261112.sh; IMG=$(grep -o 'sha256:[0-9a-f]*' $RS | head -1)
E=(); for a in $(grep -o -- '-e GGML_[A-Z0-9_]*=[^ ]*' $RS | sed 's/-e //'); do E+=(-e "$a"); done
for v in "$@"; do IFS=: read nm kt vt <<<"$v"; d=$Q/vram_$nm; mkdir -p $d
 docker rm -f nq8s-srv >/dev/null 2>&1
 docker run -d --name nq8s-srv --network none --device nvidia.com/gpu=$GPU -v <MODELS>:/models:ro -v $M/prod-libs/libggml-cuda-bundle3-20261112.so:/app/libggml-cuda.so:ro "${E[@]}" $IMG \
  -m /models/occamy-1.0-Q4_0-headq40b.gguf --alias qwen36-moe --override-kv tokenizer.ggml.pre=str:qwen35 -ngl 999 -np 2 -c 196608 -b 2048 -ub 1024 --ctx-checkpoints 0 -fa on -ctk $kt -ctv $vt --host 0.0.0.0 --port 8080 --jinja -lv 4 > $d/cid.txt 2>&1
 for i in $(seq 1 240); do docker logs nq8s-srv 2>&1 | grep -q "server is listening\|error\|failed" && break; docker ps -q -f name=nq8s-srv | grep -q . || break; sleep 1; done
 docker logs nq8s-srv > $d/load.log 2>&1
 nvidia-smi -i $GPU --query-gpu=memory.used,memory.total --format=csv,noheader > $d/smi.txt
 echo "$nm: $(grep -E 'llama_kv_cache:.*(K|V) \(|KV buffer|listening|error|failed|requires' $d/load.log | head -6 | tr '\n' '|') SMI $(cat $d/smi.txt)"
 docker rm -f nq8s-srv >/dev/null 2>&1
done
