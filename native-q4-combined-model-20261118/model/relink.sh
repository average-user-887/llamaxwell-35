#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")"
here=$(pwd -P)
c=$(dirname "$here")
image=sha256:747c8a878fc2a36aca96e10da800f6cfe80d1e813c7e4af49ded30588be16bed
python3 -B - <<'PY'
from pathlib import Path
import shlex
original=Path("../resume-q4-moe-cohorts/compile.sh").read_text().splitlines()
cmd,=[x for x in original if x.startswith("/usr/bin/g++-14 ") and "-o /work/libggml-cuda.so " in x]
base=shlex.split(cmd)
objects=[(i,t) for i,t in enumerate(base) if t.endswith(".o")]
assert len(objects)==147 and objects[23][1]=="/n98/fattn-occamy.cu.o" and objects[42][1]=="/work/occamy-fused.cu.o"
rows=["set -eu","cd /b/ggml/src/ggml-cuda"]
for name,attention,moe in [("attention-reference","/attn/fattn-occamy.cu.o","CMakeFiles/ggml-cuda.dir/occamy-fused.cu.o"),("moe-reference","/n98/fattn-occamy.cu.o","/moe/occamy-fused.cu.o"),("libggml-cuda","/attn/fattn-occamy.cu.o","/moe/occamy-fused.cu.o")]:
 parts=base.copy();parts[parts.index("-o")+1]=f"/work/{name}.so"
 parts[objects[23][0]]=attention;parts[objects[42][0]]=moe
 rows.append(shlex.join(parts))
Path("link.sh").write_text("\n".join(rows)+"\n")
PY
timeout -k 2s 70s taskset -c 2-3 nice -n 19 docker run --rm --pull never --runtime runc --network none --cpuset-cpus 2-3 --user "$(id -u):$(id -g)" -e NVIDIA_VISIBLE_DEVICES=void -v "$here:/work" -v "$c/sparse/r2/build:/b:ro" -v "$c/resume-q4-r2-sparse/fattn-occamy.cu.o:/attn/fattn-occamy.cu.o:ro" -v "$c/resume-q4-moe-cohorts/occamy-fused.cu.o:/moe/occamy-fused.cu.o:ro" -v "$c/night-q3-r2-sparse/fattn-occamy.cu.o:/n98/fattn-occamy.cu.o:ro" --entrypoint /bin/bash "$image" -euc 'nice -n 19 bash /work/link.sh'
cmp attention-reference.so "$c/resume-q4-r2-sparse/libggml-cuda.so"
cmp moe-reference.so "$c/resume-q4-moe-cohorts/libggml-cuda.so"
sha256sum attention-reference.so moe-reference.so libggml-cuda.so
