#!/usr/bin/env python3
"""Stock upstream llama.cpp (b12318) baseline benchmark on M6000.
Run ONLY via run_stock.sh (holds gpu-bench.lock, unloads prod, checks 0 apps).
"""
import http.client
import json
import os
import subprocess
import sys
import threading
import time

M = "<LAB>"
UPBIN = M + "/upstream-b12318/build/bin"
TOK = M + "/occamy35-day-20261112/evidence/fixtures/tokens"
IMG = "sha256:3a0caa9afea12f5ca5d982621cc2d86b1fa6ac614d735b262fa475d7dc6e2a6d"
PORT = 18585


def now():
    return time.time()


def sh(cmd):
    return subprocess.run(cmd, capture_output=True, text=True)


class Telemetry(threading.Thread):
    Q = "clocks.sm,clocks.mem,power.draw,memory.used,temperature.gpu"

    def __init__(self, path, gpu):
        super().__init__(daemon=True)
        self.gpu = gpu
        self.f = open(path, "w")
        self.f.write("epoch,sm_mhz,mem_mhz,power_w,mem_mib,temp_c\n")
        self.halt = False
        self.peak_mem_mib = 0

    def run(self):
        while not self.halt:
            t = now()
            o = sh([
                "nvidia-smi", "-i", self.gpu,
                "--query-gpu=" + self.Q,
                "--format=csv,noheader,nounits",
            ]).stdout.strip()
            if o:
                parts = [p.strip() for p in o.split(",")]
                try:
                    self.peak_mem_mib = max(self.peak_mem_mib, int(float(parts[3])))
                except Exception:
                    pass
                self.f.write(f"{t:.2f},{','.join(parts)}\n")
                self.f.flush()
            time.sleep(max(0.0, 1.0 - (now() - t)))

    def stop(self):
        self.halt = True
        self.join(5)
        self.f.close()


def call(path, body=None, method="POST", timeout=3600):
    c = http.client.HTTPConnection("127.0.0.1", PORT, timeout=timeout)
    c.request(
        method, path,
        json.dumps(body) if body is not None else None,
        {"Content-Type": "application/json"},
    )
    r = c.getresponse()
    d = r.read()
    c.close()
    try:
        return r.status, json.loads(d)
    except Exception:
        return r.status, {"raw": d[:400].decode(errors="replace")}


def complete(ids, slot, n_predict, tag):
    t0 = now()
    st, j = call("/completion", {
        "prompt": ids,
        "n_predict": n_predict,
        "id_slot": slot,
        "cache_prompt": True,
        "temperature": 0,
        "ignore_eos": True,
        "stream": False,
    }, timeout=3600)
    wall = now() - t0
    if st != 200:
        raise RuntimeError(f"{tag} HTTP {st}: {j}")
    tm = j.get("timings", {})
    return {
        "tag": tag,
        "slot": slot,
        "prompt_len": len(ids),
        "n_predict": n_predict,
        "wall_s": round(wall, 3),
        "prompt_n": tm.get("prompt_n"),
        "cache_n": tm.get("cache_n"),
        "prompt_ms": tm.get("prompt_ms"),
        "prompt_tps": tm.get("prompt_per_second"),
        "predicted_n": tm.get("predicted_n"),
        "predicted_ms": tm.get("predicted_ms"),
        "predicted_tps": tm.get("predicted_per_second"),
        "truncated": j.get("truncated"),
    }


def main():
    outdir = sys.argv[1]
    gpu = os.environ["STOCK_GPU"]
    os.makedirs(outdir, exist_ok=True)
    d0 = json.load(open(TOK + "/d0.json"))
    d1 = json.load(open(TOK + "/d1.json"))

    name = f"stock-b12318-{int(time.time()) % 100000}"
    cmd = [
        "docker", "run", "-d", "--name", name,
        "-p", f"127.0.0.1:{PORT}:8080",
        "--device", f"nvidia.com/gpu={gpu}",
        "-v", "<MODELS>:/models:ro",
        "-v", f"{UPBIN}:/candidate:ro",
        "-e", "LD_LIBRARY_PATH=/candidate",
        "--entrypoint", "/candidate/llama-server",
        IMG,
        "-m", "/models/occamy-1.0-Q4_0-headq40b.gguf",
        "--alias", "qwen36-moe",
        "--override-kv", "tokenizer.ggml.pre=str:qwen35",
        "-ngl", "999", "-np", "2", "-c", "196608",
        "-b", "2048", "-ub", "1024",
        "--cache-ram", "0", "--ctx-checkpoints", "16", "--checkpoint-min-step", "2048",
        "-fa", "on", "--host", "0.0.0.0", "--port", "8080",
        "--jinja", "--metrics", "-lv", "4",
    ]

    tel = Telemetry(outdir + "/telemetry.csv", gpu)
    tel.start()
    results = {"container": name, "units": {}}

    try:
        r = sh(cmd)
        if r.returncode != 0:
            raise RuntimeError("docker run failed: " + r.stderr)

        t0 = now()
        while now() - t0 < 300:
            if sh(["docker", "inspect", "-f", "{{.State.Running}}", name]).stdout.strip() != "true":
                raise RuntimeError("container exited early")
            try:
                if call("/health", method="GET", timeout=3)[0] == 200:
                    break
            except Exception:
                pass
            time.sleep(2)
        else:
            raise RuntimeError("server not ready in 300s")
        results["ready_s"] = round(now() - t0, 2)

        # 1. Warm 4K prefill + 128 decode
        results["units"]["warm_4k"] = complete(d0[:4096], 0, 128, "warm_4k")
        print("warm_4k:", results["units"]["warm_4k"], flush=True)

        # Erase slot 0 before deep prefill
        call("/slots/0?action=erase", {})

        # 2. Deep prefill: slot 0 to 90K, slot 1 to 84K
        results["units"]["prefill_90k_s0"] = complete(d0[:90000], 0, 1, "prefill_90k_s0")
        print("prefill_90k_s0:", results["units"]["prefill_90k_s0"], flush=True)

        results["units"]["prefill_84k_s1"] = complete(d1[:84000], 1, 1, "prefill_84k_s1")
        print("prefill_84k_s1:", results["units"]["prefill_84k_s1"], flush=True)

        # 3. 4K append after 90K + 64 decode (matches N1 +4K)
        results["units"]["append_4k_s0"] = complete(d0[:94096], 0, 64, "append_4k_s0")
        print("append_4k_s0:", results["units"]["append_4k_s0"], flush=True)

        # 4. Concurrent 2-slot deep decode (128 tok each)
        out2 = {}
        errs = []

        def worker(slot, ids, tag):
            try:
                out2[tag] = complete(ids, slot, 128, tag)
            except Exception as ex:
                errs.append((tag, repr(ex)))

        t_a = threading.Thread(target=worker, args=(0, d0[:94100], "deep2_s0"))
        t_b = threading.Thread(target=worker, args=(1, d1[:84004], "deep2_s1"))
        t_both0 = now()
        t_a.start()
        t_b.start()
        t_a.join()
        t_b.join()
        both_wall = now() - t_both0
        if errs:
            raise RuntimeError(f"deep2 errors: {errs}")
        s0 = out2["deep2_s0"]
        s1 = out2["deep2_s1"]
        sum_tps = (s0["predicted_tps"] or 0) + (s1["predicted_tps"] or 0)
        results["units"]["both_deep"] = {
            "wall_s": round(both_wall, 3),
            "sum_predicted_tps": round(sum_tps, 2),
            "s0": s0,
            "s1": s1,
        }
        print("both_deep:", results["units"]["both_deep"], flush=True)

    finally:
        tel.stop()
        results["peak_vram_mib"] = tel.peak_mem_mib
        logs = sh(["docker", "logs", name])
        open(outdir + "/server.log", "w").write(logs.stdout + logs.stderr)
        sh(["docker", "rm", "-f", name])
        json.dump(results, open(outdir + "/metrics.json", "w"), indent=2)

    # Write RESULT.md
    u = results["units"]
    lines = [
        "# Stock upstream llama.cpp (b12318) baseline on M6000",
        "",
        f"Peak VRAM: **{results['peak_vram_mib']} MiB**",
        "",
        "| unit | prompt_n | cache_n | prompt_s | prompt_tps | pred_n | pred_tps | wall_s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for k in ("warm_4k", "prefill_90k_s0", "prefill_84k_s1", "append_4k_s0"):
        x = u.get(k)
        if x:
            ps = (x["prompt_ms"] or 0) / 1000.0
            lines.append(
                f"| {k} | {x['prompt_n']} | {x['cache_n']} | {ps:.2f} | {x['prompt_tps']:.2f} | {x['predicted_n']} | {x['predicted_tps']:.2f} | {x['wall_s']:.2f} |"
            )
    bd = u.get("both_deep")
    if bd:
        for sk in ("s0", "s1"):
            x = bd[sk]
            ps = (x["prompt_ms"] or 0) / 1000.0
            lines.append(
                f"| both_deep_{sk} | {x['prompt_n']} | {x['cache_n']} | {ps:.2f} | {x['prompt_tps']:.2f} | {x['predicted_n']} | {x['predicted_tps']:.2f} | {x['wall_s']:.2f} |"
            )
        lines.append("")
        lines.append(f"2-slot deep concurrent wall: **{bd['wall_s']} s**, sum predicted_tps: **{bd['sum_predicted_tps']} tok/s**")
    open(outdir + "/RESULT.md", "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
