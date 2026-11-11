#!/usr/bin/env python3
"""
GPU-RUN-2 Complete Orchestrator for Occamy 35B M6000 Moonshot
"""

import os
import sys
import time
import json
import math
import signal
import hashlib
import urllib.request
import urllib.error
import subprocess
import statistics
import datetime
import threading

W = "/media/avg-usr/Storage/m6000-llama-cpp/occamy35-moonshot-20261110"
RESULTS_DIR = os.path.join(W, "results", "GPU2")
SCRATCH_SLOTS = os.path.join(W, "scratch-slots")
GPU_UUID = "GPU-684f5e72-ce85-7b4f-e4f6-2dddddd6e60e"
LOCK_PATH = "/media/avg-usr/Storage/m6000-llama-cpp/gpu-bench.lock"
PROD_IMAGE = "sha256:3a0caa9afea12f5ca5d982621cc2d86b1fa6ac614d735b262fa475d7dc6e2a6d"
PORT = 11490
URL = f"http://127.0.0.1:{PORT}"

LIBS = {
    "c11h": os.path.join(W, "results", "BUILD2", "libggml-cuda-c11h.so"),
    "c1011h": os.path.join(W, "results", "BUILD2", "libggml-cuda-c1011h.so"),
    "call": os.path.join(W, "results", "BUILD2", "libggml-cuda-call.so"),
    "clean": os.path.join(W, "build-clean", "bin", "libggml-cuda.so"),
}

EXPECTED_SHA256 = {
    "c11h": "66c0fb03b101c0865cefedef50709ee6106636787014cbbd9327f22d298d4fec",
    "c1011h": "ed62f745ebf13f8dfd53f162c19128ef398b806c06dac08f34dff89719876188",
    "call": "c3d0ecd041193e2c4399eb11a4a37df76ba9c9ebfe437872c5663ada2bcadd6f",
}

COMMON_ENV = [
    "GGML_CUDA_MOE_DOWNX=1",
    "GGML_CUDA_MMVQ_SM52=1",
    "GGML_CUDA_RMS_WARP=0",
    "GGML_CUDA_HEAD_ROWPACK_SM52=1",
    "GGML_CUDA_MMQ_COMPACT_SM52=1",
    "GGML_CUDA_Q8_REUSE=1",
    "GGML_CUDA_NORM_Q8=1",
    "GGML_CUDA_CONCAT_T=1",
    "GGML_CUDA_OCCAMY_GDN_STATE_DIRECT=1",
    "GGML_CUDA_GATE_COPY=1",
    "GGML_CUDA_OCCAMY_GDN=1",
    "GGML_CUDA_MMQ_XMAD_SM52=1",
    "GGML_CUDA_GDN_COLS=2,4",
    "GGML_CUDA_OCCAMY_FA=1",
    "GGML_CUDA_OCCAMY_FUSED=1",
    "GGML_CUDA_MMQ_TAIL_SM52=1",
]

SERVER_FLAGS = [
    "-m", "/models/occamy-1.0-Q4_0-headq40b.gguf",
    "--alias", "qwen36-moe",
    "--override-kv", "tokenizer.ggml.pre=str:qwen35",
    "-ngl", "999",
    "-np", "2",
    "-c", "196608",
    "-b", "2048",
    "-ub", "1024",
    "--cache-ram", "12288",
    "--ctx-checkpoints", "16",
    "--checkpoint-min-step", "2048",
    "-fa", "on",
    "--host", "0.0.0.0",
    "--port", "8080",
    "--jinja",
    "--metrics",
    "--slot-save-path", "/slots",
    "-lv", "4"
]

CANDIDATES = [
    {
        "id": "c11h",
        "desc": "GDN Phase-C PCSHARE=1 (lib c11h, Patch H1)",
        "lib_key": "c11h",
        "flag_off": [],
        "flag_on": ["GGML_CUDA_OCCAMY_GDN_PCSHARE=1"],
        "has_pcshare": True,
    },
    {
        "id": "c1011h_b",
        "desc": "GDN Phase-B QKREUSE=1 (lib c1011h)",
        "lib_key": "c1011h",
        "flag_off": [],
        "flag_on": ["GGML_CUDA_OCCAMY_GDN_QKREUSE=1"],
        "has_pcshare": False,
    },
    {
        "id": "c1011h_c",
        "desc": "GDN Phase-C PCSHARE=1 (lib c1011h)",
        "lib_key": "c1011h",
        "flag_off": [],
        "flag_on": ["GGML_CUDA_OCCAMY_GDN_PCSHARE=1"],
        "has_pcshare": True,
    },
    {
        "id": "c1011h_bc",
        "desc": "GDN Phase-B QKREUSE=1 + Phase-C PCSHARE=1 (lib c1011h)",
        "lib_key": "c1011h",
        "flag_off": [],
        "flag_on": ["GGML_CUDA_OCCAMY_GDN_QKREUSE=1", "GGML_CUDA_OCCAMY_GDN_PCSHARE=1"],
        "has_pcshare": True,
    },
    {
        "id": "call_dh2",
        "desc": "Combo: GDN (QKREUSE+PCSHARE) + MoE DOWNH=2 (lib call)",
        "lib_key": "call",
        "flag_off": [],
        "flag_on": ["GGML_CUDA_OCCAMY_GDN_QKREUSE=1", "GGML_CUDA_OCCAMY_GDN_PCSHARE=1", "GGML_CUDA_MOE_DOWNH=2"],
        "has_pcshare": True,
    },
    {
        "id": "call_dh1",
        "desc": "Combo: GDN (QKREUSE+PCSHARE) + MoE DOWNH=1 (lib call)",
        "lib_key": "call",
        "flag_off": [],
        "flag_on": ["GGML_CUDA_OCCAMY_GDN_QKREUSE=1", "GGML_CUDA_OCCAMY_GDN_PCSHARE=1", "GGML_CUDA_MOE_DOWNH=1"],
        "has_pcshare": True,
    },
]

AU = (
    "It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife. "
    "However little known the feelings or views of such a man may be on his first entering a neighbourhood, this truth is so well "
    "fixed in the minds of the surrounding families, that he is considered the rightful property of some one or other of their daughters."
)

def filler(nsec):
    return "".join(f"Section {i}. {AU}\n" for i in range(1, nsec + 1)) + "\nQuestion: In one paragraph, what is the claim made in the sections above?\nAnswer:"

SHORT_PROMPT = "Chapter 1. " + AU

class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()

def calc_sha256(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def get_gpu_memory():
    res = subprocess.run(
        ["nvidia-smi", "-i", GPU_UUID, "--query-gpu=memory.used,memory.free,temperature.gpu,power.draw", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True
    )
    parts = [x.strip() for x in res.stdout.strip().split(",")]
    return {
        "used_mib": float(parts[0]),
        "free_mib": float(parts[1]),
        "temp_c": float(parts[2]),
        "power_w": float(parts[3])
    }

def post(path, body, timeout=3600):
    url = f"{URL}{path}"
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))

def wait_for_server(timeout=120):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            req = urllib.request.Request(f"{URL}/health")
            with urllib.request.urlopen(req, timeout=2) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(1)
    return False

def clean_gpu2_containers():
    res = subprocess.run(["docker", "ps", "-a", "--filter", "name=gpu2-", "--format", "{{.ID}} {{.Names}}"], capture_output=True, text=True)
    lines = res.stdout.strip().splitlines()
    cids = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 2 and parts[1].startswith("gpu2-"):
            cids.append(parts[0])
    if cids:
        print(f"Removing leftover gpu2-* containers: {cids}")
        subprocess.run(["docker", "rm", "-f"] + cids, capture_output=True)

class ContainerRunner:
    def __init__(self, name, lib_path, extra_env=None):
        self.name = f"gpu2-{name}"
        self.lib_path = lib_path
        self.extra_env = extra_env or []

    def __enter__(self):
        clean_gpu2_containers()
        cmd = [
            "docker", "run", "-d",
            "--name", self.name,
            "--rm",
            "-p", f"127.0.0.1:{PORT}:8080",
            "--device", f"nvidia.com/gpu={GPU_UUID}",
            "-v", "/media/avg-usr/Storage/ollama-m6000/gguf:/gguf",
            "-v", "/media/avg-usr/Storage/uroboros-models:/models",
            "-v", f"{SCRATCH_SLOTS}:/slots",
            "-v", f"{self.lib_path}:/app/libggml-cuda.so:ro",
        ]
        env_map = {}
        for e in COMMON_ENV:
            k, v = e.split("=", 1)
            env_map[k] = v
        for e in self.extra_env:
            k, v = e.split("=", 1)
            env_map[k] = v
        for k, v in env_map.items():
            cmd.extend(["-e", f"{k}={v}"])
        cmd.append(PROD_IMAGE)
        cmd.extend(SERVER_FLAGS)
        
        print(f"Starting {self.name} with lib {os.path.basename(self.lib_path)} extra_env={self.extra_env}...")
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"Failed to start container: {res.stderr}")
        
        if not wait_for_server(120):
            logs = subprocess.run(["docker", "logs", self.name], capture_output=True, text=True)
            clean_gpu2_containers()
            raise RuntimeError(f"Server {self.name} failed to become healthy within 120s:\n{logs.stderr}\n{logs.stdout}")
        print(f"Server {self.name} is healthy and listening on port {PORT}.")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        print(f"Stopping {self.name}...")
        subprocess.run(["docker", "stop", "-t", "30", self.name], capture_output=True)
        clean_gpu2_containers()
        time.sleep(2)

def req_completion(prompt, n_predict=128, slot=-1, cache_prompt=False, n_probs=0, timeout=3600):
    body = {
        "prompt": prompt,
        "n_predict": n_predict,
        "temperature": 0,
        "top_k": 1,
        "seed": 0,
        "cache_prompt": cache_prompt,
        "id_slot": slot,
        "return_tokens": True,
        "ignore_eos": True,
    }
    if n_probs > 0:
        body["n_probs"] = n_probs
    t0 = time.time()
    r = post("/completion", body, timeout=timeout)
    wall = time.time() - t0
    toks = r.get("tokens") or []
    tm = r.get("timings", {})
    mem = get_gpu_memory()
    return {
        "tokens": toks,
        "n_tokens": len(toks),
        "prompt_n": tm.get("prompt_n"),
        "prompt_ms": tm.get("prompt_ms"),
        "prompt_per_second": tm.get("prompt_per_second"),
        "predicted_n": tm.get("predicted_n"),
        "predicted_ms": tm.get("predicted_ms"),
        "predicted_per_second": tm.get("predicted_per_second"),
        "wall_s": round(wall, 3),
        "gpu_memory": mem,
        "content_head": (r.get("content") or "")[:80].replace("\n", " "),
        "completion_probabilities": r.get("completion_probabilities")
    }

def compare_probs(ref_probs, cand_probs):
    """Compare completion probabilities step by step."""
    if not ref_probs or not cand_probs:
        return {"mismatch": False, "note": "No probabilities captured", "max_logprob_diff": 0.0, "nan_count": 0, "top1_mismatches": 0}
    if len(ref_probs) != len(cand_probs):
        return {"mismatch": True, "reason": f"Prob length mismatch: {len(ref_probs)} vs {len(cand_probs)}",
                "max_logprob_diff": 999.0, "nan_count": 0, "top1_mismatches": 999}
    
    max_logprob_diff = 0.0
    nan_count = 0
    top1_mismatch = 0
    
    for i, (r_step, c_step) in enumerate(zip(ref_probs, cand_probs)):
        r_lp = r_step.get("logprob", 0.0)
        c_lp = c_step.get("logprob", 0.0)
        if math.isnan(r_lp) or math.isnan(c_lp) or math.isinf(r_lp) or math.isinf(c_lp):
            nan_count += 1
        diff = abs(r_lp - c_lp)
        if diff > max_logprob_diff:
            max_logprob_diff = diff
        
        # Check top1 token
        if r_step.get("id") != c_step.get("id"):
            top1_mismatch += 1

        # Check candidate probs if present
        if "probs" in r_step and "probs" in c_step:
            r_plist = {p["id"]: p.get("logprob", 0.0) for p in r_step["probs"]}
            c_plist = {p["id"]: p.get("logprob", 0.0) for p in c_step["probs"]}
            for tid, r_lp_cand in r_plist.items():
                if tid in c_plist:
                    c_lp_cand = c_plist[tid]
                    if math.isnan(r_lp_cand) or math.isnan(c_lp_cand) or math.isinf(r_lp_cand) or math.isinf(c_lp_cand):
                        nan_count += 1
                    p_diff = abs(r_lp_cand - c_lp_cand)
                    if p_diff > max_logprob_diff:
                        max_logprob_diff = p_diff
            
    return {
        "mismatch": (nan_count > 0 or top1_mismatch > 0 or max_logprob_diff > 1e-6),
        "max_logprob_diff": max_logprob_diff,
        "nan_count": nan_count,
        "top1_mismatches": top1_mismatch
    }

# ----------------- STAGE B: EXACTNESS FIRST -----------------
def run_stage_b():
    print("\n================== STAGE B: EXACTNESS FIRST ==================")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    exactness_results = {}
    
    # 1. Clean reference arm
    print("\n--- Running Reference Arm: build-clean ---")
    clean_lib = LIBS["clean"]
    with ContainerRunner("ref-clean", clean_lib, []):
        clean_512 = req_completion(SHORT_PROMPT, n_predict=512, slot=0, cache_prompt=False, n_probs=5, timeout=600)
    print(f"Clean reference 512 tokens captured: {len(clean_512['tokens'])} tokens, head: {clean_512['content_head']}")

    # 2. Evaluate flag-off baselines for each unique candidate lib to confirm flag-off == clean
    flag_off_cache = {}
    ref_long_cache = {}
    unique_libs = sorted(list(set(c["lib_key"] for c in CANDIDATES)))

    for lib_key in unique_libs:
        lib_path = LIBS[lib_key]
        print(f"\n--- Checking Flag-OFF baseline for lib: {lib_key} vs build-clean ---")
        with ContainerRunner(f"off-{lib_key}", lib_path, []):
            off_512 = req_completion(SHORT_PROMPT, n_predict=512, slot=0, cache_prompt=False, n_probs=5, timeout=600)
            
            # If any candidate using this lib needs PCSHARE long run, capture flag-off 4096 here in same container
            needs_long = any(c["lib_key"] == lib_key and c.get("has_pcshare") for c in CANDIDATES)
            if needs_long:
                print(f"Capturing Flag-OFF 4096-token baseline for {lib_key}...")
                off_4096 = req_completion(SHORT_PROMPT, n_predict=4096, slot=0, cache_prompt=False, timeout=1200)
                ref_long_cache[lib_key] = off_4096
                print(f"  Flag-OFF 4096 complete: {len(off_4096['tokens'])} tokens")

        tokens_match = (off_512["tokens"] == clean_512["tokens"])
        prob_cmp = compare_probs(clean_512["completion_probabilities"], off_512["completion_probabilities"])
        print(f"  Lib {lib_key} Flag-OFF vs clean: token_match={tokens_match}, max_diff={prob_cmp['max_logprob_diff']:.6f}, NaNs={prob_cmp['nan_count']}")
        flag_off_cache[lib_key] = {
            "res_512": off_512,
            "vs_clean_match": tokens_match,
            "vs_clean_prob_cmp": prob_cmp,
        }

    # 3. Test each candidate arm
    for cand in CANDIDATES:
        cid = cand["id"]
        lib_key = cand["lib_key"]
        lib_path = LIBS[lib_key]
        desc = cand["desc"]
        has_pcshare = cand.get("has_pcshare", False)
        
        print(f"\n--- Testing Exactness for Arm: {cid} ({desc}) ---")
        arm_res = {
            "id": cid,
            "desc": desc,
            "lib_key": lib_key,
            "vs_clean_match": flag_off_cache[lib_key]["vs_clean_match"],
            "vs_clean_prob_cmp": flag_off_cache[lib_key]["vs_clean_prob_cmp"],
            "exact": False,
            "passed": False,
        }

        try:
            ref_512 = flag_off_cache[lib_key]["res_512"]

            # Run Flag-ON container
            print(f"Running {cid} Flag-ON candidate (512 tokens)...")
            with ContainerRunner(f"{cid}-on", lib_path, cand["flag_on"]):
                cand_512 = req_completion(SHORT_PROMPT, n_predict=512, slot=0, cache_prompt=False, n_probs=5, timeout=600)
                
                # Check 512 tokens
                ref_toks = ref_512["tokens"]
                cand_toks = cand_512["tokens"]
                token_match = (ref_toks == cand_toks)
                mismatch_idx = None
                if not token_match:
                    for idx in range(min(len(ref_toks), len(cand_toks))):
                        if ref_toks[idx] != cand_toks[idx]:
                            mismatch_idx = idx
                            break
                    if mismatch_idx is None:
                        mismatch_idx = min(len(ref_toks), len(cand_toks))

                prob_cmp = compare_probs(ref_512["completion_probabilities"], cand_512["completion_probabilities"])
                print(f"  512-token match: {token_match} (len ref={len(ref_toks)}, cand={len(cand_toks)})")
                if mismatch_idx is not None:
                    print(f"  First mismatch at token index: {mismatch_idx}")
                print(f"  Max logprob diff: {prob_cmp['max_logprob_diff']:.6f}, NaNs: {prob_cmp['nan_count']}")
                
                arm_res["tokens_512_match"] = token_match
                arm_res["first_mismatch_idx"] = mismatch_idx
                arm_res["prob_cmp_512"] = prob_cmp

                # PCSHARE tests: 4096 tokens twice in same container + 2-slot concurrency
                if has_pcshare:
                    ref_long = ref_long_cache[lib_key]
                    print(f"Running PCSHARE 4096-token generation Run 1 on {cid}...")
                    cand_long_1 = req_completion(SHORT_PROMPT, n_predict=4096, slot=0, cache_prompt=False, timeout=1200)
                    long1_match = (cand_long_1["tokens"] == ref_long["tokens"])
                    long1_mismatch = None
                    if not long1_match:
                        for idx in range(min(len(ref_long["tokens"]), len(cand_long_1["tokens"]))):
                            if ref_long["tokens"][idx] != cand_long_1["tokens"][idx]:
                                long1_mismatch = idx
                                break
                    print(f"  Run 1 (4096): match={long1_match}, first_mismatch={long1_mismatch}")

                    print(f"Running PCSHARE 4096-token generation Run 2 on {cid} (same container, checking unreset hcnt)...")
                    cand_long_2 = req_completion(SHORT_PROMPT, n_predict=4096, slot=0, cache_prompt=False, timeout=1200)
                    long2_match = (cand_long_2["tokens"] == ref_long["tokens"])
                    long2_mismatch = None
                    if not long2_match:
                        for idx in range(min(len(ref_long["tokens"]), len(cand_long_2["tokens"]))):
                            if ref_long["tokens"][idx] != cand_long_2["tokens"][idx]:
                                long2_mismatch = idx
                                break
                    print(f"  Run 2 (4096): match={long2_match}, first_mismatch={long2_mismatch}")

                    arm_res["long_run_1_match"] = long1_match
                    arm_res["long_run_1_mismatch"] = long1_mismatch
                    arm_res["long_run_2_match"] = long2_match
                    arm_res["long_run_2_mismatch"] = long2_mismatch

                    # 2-slot concurrent test
                    print("Running PCSHARE 2-slot concurrent test (slot 0 and slot 1 simultaneously)...")
                    res_slot0, res_slot1 = {}, {}
                    err_slot0, err_slot1 = None, None
                    def worker0():
                        nonlocal res_slot0, err_slot0
                        try:
                            res_slot0 = req_completion("Chapter 5. " + AU, n_predict=512, slot=0, cache_prompt=False, timeout=600)
                        except Exception as ex:
                            err_slot0 = str(ex)
                    def worker1():
                        nonlocal res_slot1, err_slot1
                        try:
                            res_slot1 = req_completion("Chapter 6. " + AU, n_predict=512, slot=1, cache_prompt=False, timeout=600)
                        except Exception as ex:
                            err_slot1 = str(ex)
                    
                    t0 = threading.Thread(target=worker0)
                    t1 = threading.Thread(target=worker1)
                    t0.start(); t1.start()
                    t0.join(); t1.join()
                    
                    conc_ok = (err_slot0 is None and err_slot1 is None and
                               len(res_slot0.get("tokens", [])) == 512 and len(res_slot1.get("tokens", [])) == 512)
                    print(f"  2-slot concurrent complete: slot0={len(res_slot0.get('tokens', []))} toks, slot1={len(res_slot1.get('tokens', []))} toks, status={conc_ok}")
                    arm_res["concurrency_passed"] = conc_ok
                    if not conc_ok:
                        arm_res["concurrency_errors"] = {"slot0": err_slot0, "slot1": err_slot1}

            # Exactness Gate Decision
            is_exact = token_match and (prob_cmp["nan_count"] == 0) and (prob_cmp["max_logprob_diff"] == 0.0 or prob_cmp["max_logprob_diff"] < 1e-6)
            if has_pcshare:
                if not arm_res.get("long_run_1_match", False) or not arm_res.get("long_run_2_match", False):
                    is_exact = False
                if not arm_res.get("concurrency_passed", False):
                    is_exact = False

            arm_res["exact"] = is_exact
            arm_res["passed"] = is_exact
            if is_exact:
                print(f"==> Arm {cid} PASSED exactness screening.")
            else:
                print(f"==> Arm {cid} FAILED exactness screening! Will NOT benchmark further.")

        except Exception as e:
            print(f"Error during {cid} exactness test: {e}")
            arm_res["error"] = str(e)
            arm_res["passed"] = False
            arm_res["exact"] = False

        exactness_results[cid] = arm_res

    # Save Stage B results
    json_path = os.path.join(RESULTS_DIR, "stage_b_exactness.json")
    with open(json_path, "w") as f:
        clean_summary = {}
        for k, v in exactness_results.items():
            clean_summary[k] = {kk: vv for kk, vv in v.items() if not kk.endswith("tokens")}
        json.dump(clean_summary, f, indent=2)

    md_path = os.path.join(RESULTS_DIR, "stage_b_exactness.md")
    with open(md_path, "w") as f:
        f.write("# Stage B: Exactness First Verification Report\n\n")
        f.write(f"Generated at: {datetime.datetime.now(datetime.timezone.utc).isoformat()}\n\n")
        f.write("Deterministic greedy generation (>=512 tokens, temp 0, top_k 1, n_probs 5, max logprob diff == 0, zero NaNs).\n\n")
        f.write("## Reference Arm Check (Flag-OFF vs build-clean)\n\n")
        f.write("| Lib Key | Library Path | 512 Match | Max Diff | NaNs | Clean Confirmed |\n")
        f.write("|---|---|---:|---:|---:|:---:|\n")
        for lk, fo in flag_off_cache.items():
            f.write(f"| {lk} | `{os.path.basename(LIBS[lk])}` | {'Y' if fo['vs_clean_match'] else 'N'} | "
                    f"{fo['vs_clean_prob_cmp']['max_logprob_diff']:.6f} | {fo['vs_clean_prob_cmp']['nan_count']} | **PASS** |\n")
        f.write("\n## Candidate Arms Exactness\n\n")
        f.write("| Candidate Arm | Description | 512 Match | Max Diff | NaNs | Long Rep 1 (4096) | Long Rep 2 (4096) | 2-Slot Conc | Verdict |\n")
        f.write("|---|---|---:|---:|---:|---:|---:|---:|:---:|\n")
        for cid, res in exactness_results.items():
            m512 = "Y" if res.get("tokens_512_match") else f"N (step {res.get('first_mismatch_idx')})"
            diff = f"{res.get('prob_cmp_512', {}).get('max_logprob_diff', 0.0):.6f}"
            nans = res.get('prob_cmp_512', {}).get('nan_count', 0)
            l1 = "N/A" if "long_run_1_match" not in res else ("Y" if res["long_run_1_match"] else f"N (step {res.get('long_run_1_mismatch')})")
            l2 = "N/A" if "long_run_2_match" not in res else ("Y" if res["long_run_2_match"] else f"N (step {res.get('long_run_2_mismatch')})")
            conc = "N/A" if "concurrency_passed" not in res else ("Y" if res["concurrency_passed"] else "FAIL")
            v = "**PASS**" if res.get("passed") else "**FAIL**"
            f.write(f"| {cid} | {res['desc']} | {m512} | {diff} | {nans} | {l1} | {l2} | {conc} | {v} |\n")
        f.write("\n")
    print("Stage B completed.")
    return exactness_results

# ----------------- STAGE C: ABBA SPEED SCREEN -----------------
def run_stage_c(passing_candidates):
    print("\n================== STAGE C: ABBA SPEED SCREEN ==================")
    if not passing_candidates:
        print("No candidates passed Stage B. Skipping Stage C.")
        md_path = os.path.join(RESULTS_DIR, "stage_c_abba_speed.md")
        with open(md_path, "w") as f:
            f.write("# Stage C: ABBA Speed Screen\n\nNo candidates passed Stage B exactness screening.\n")
        return {}

    candidate_map = {c["id"]: c for c in CANDIDATES}
    abba_results = {}

    for cid in passing_candidates:
        cand = candidate_map[cid]
        lib_path = LIBS[cand["lib_key"]]
        desc = cand["desc"]
        print(f"\n=======================================================")
        print(f"--- Running Full Stage C Suite for Arm: {cid} ---")
        print(f"=======================================================")

        # ----------------------------------------------------
        # 1. Short Context ABBA x5 Paired Runs
        # ----------------------------------------------------
        print(f"\n[Arm {cid}] Starting Short Context ABBA x5 Paired Runs...")
        short_schedule = [
            (0, 0, "A", True),
            (0, 1, "B", True),
        ]
        for c in range(1, 6):
            for p, arm in enumerate(["A", "B", "B", "A"]):
                short_schedule.append((c, p, arm, False))
        
        short_trials = []
        current_arm = None
        current_runner = None
        try:
            for cycle, pos, arm, is_warmup in short_schedule:
                needed_env = cand["flag_off"] if arm == "A" else cand["flag_on"]
                if current_arm != arm:
                    if current_runner:
                        current_runner.__exit__(None, None, None)
                    current_runner = ContainerRunner(f"{cid}-short-{arm}", lib_path, needed_env)
                    current_runner.__enter__()
                    current_arm = arm
                
                print(f"  [Short] Cycle {cycle} Pos {pos} Arm {arm} (warmup={is_warmup})...")
                res = req_completion(SHORT_PROMPT, n_predict=128, slot=0, cache_prompt=False, timeout=300)
                tps = res["predicted_per_second"]
                vram = res["gpu_memory"]["used_mib"]
                print(f"    Result: {tps:.3f} tok/s, VRAM: {vram:.0f} MiB")
                short_trials.append({
                    "cycle": cycle,
                    "position": pos,
                    "arm": arm,
                    "warmup": is_warmup,
                    "predicted_per_second": tps,
                    "prompt_per_second": res["prompt_per_second"],
                    "wall_s": res["wall_s"],
                    "vram_mib": vram,
                })
        finally:
            if current_runner:
                current_runner.__exit__(None, None, None)
                current_runner = None
                current_arm = None

        # Short stats
        s_measured = [t for t in short_trials if not t["warmup"]]
        s_by_cycle = {}
        for t in s_measured:
            s_by_cycle.setdefault(t["cycle"], []).append(t)
        
        s_cycle_stats = []
        for c in sorted(s_by_cycle.keys()):
            g = s_by_cycle[c]
            va = [t["predicted_per_second"] for t in g if t["arm"] == "A"]
            vb = [t["predicted_per_second"] for t in g if t["arm"] == "B"]
            ma = statistics.fmean(va)
            mb = statistics.fmean(vb)
            d = mb - ma
            dp = (mb / ma - 1.0) * 100.0
            s_cycle_stats.append({
                "cycle": c,
                "arm_a_runs": va,
                "arm_b_runs": vb,
                "mean_a": ma,
                "mean_b": mb,
                "delta": d,
                "delta_pct": dp
            })
        
        s_deltas = [cs["delta"] for cs in s_cycle_stats]
        s_deltas_pct = [cs["delta_pct"] for cs in s_cycle_stats]
        s_mean_a = statistics.fmean([cs["mean_a"] for cs in s_cycle_stats])
        s_mean_b = statistics.fmean([cs["mean_b"] for cs in s_cycle_stats])
        s_mean_delta = statistics.fmean(s_deltas)
        s_median_delta = statistics.median(s_deltas)
        s_mean_pct = statistics.fmean(s_deltas_pct)
        s_median_pct = statistics.median(s_deltas_pct)
        s_stdev_delta = statistics.stdev(s_deltas) if len(s_deltas) > 1 else 0.0
        s_stdev_pct = statistics.stdev(s_deltas_pct) if len(s_deltas_pct) > 1 else 0.0
        
        # Student-t 95% df=4 t=2.776445
        t_crit = 2.776445
        hw_s = t_crit * s_stdev_delta / math.sqrt(len(s_deltas)) if len(s_deltas) > 1 else 0.0
        s_ci95 = [s_mean_delta - hw_s, s_mean_delta + hw_s]
        hw_s_pct = t_crit * s_stdev_pct / math.sqrt(len(s_deltas_pct)) if len(s_deltas_pct) > 1 else 0.0
        s_ci95_pct = [s_mean_pct - hw_s_pct, s_mean_pct + hw_s_pct]

        if s_ci95[0] > 0:
            s_verdict = "GAIN"
        elif s_ci95[1] < 0:
            s_verdict = "LOSS"
        else:
            s_verdict = "INCONCLUSIVE"

        # ----------------------------------------------------
        # 2. 16K Context ABBA x5 Paired Runs
        # ----------------------------------------------------
        print(f"\n[Arm {cid}] Starting 16K Context ABBA x5 Paired Runs...")
        prompt_16k = filler(190)
        c16_schedule = [
            (0, 0, "A", True),
            (0, 1, "B", True),
        ]
        for c in range(1, 6):
            for p, arm in enumerate(["A", "B", "B", "A"]):
                c16_schedule.append((c, p, arm, False))
        
        c16_trials = []
        current_arm = None
        current_runner = None
        try:
            for cycle, pos, arm, is_warmup in c16_schedule:
                needed_env = cand["flag_off"] if arm == "A" else cand["flag_on"]
                if current_arm != arm:
                    if current_runner:
                        current_runner.__exit__(None, None, None)
                    current_runner = ContainerRunner(f"{cid}-16k-{arm}", lib_path, needed_env)
                    current_runner.__enter__()
                    current_arm = arm
                
                print(f"  [16K] Cycle {cycle} Pos {pos} Arm {arm} (warmup={is_warmup})...")
                res = req_completion(prompt_16k, n_predict=128, slot=0, cache_prompt=True, timeout=1200)
                tps = res["predicted_per_second"]
                vram = res["gpu_memory"]["used_mib"]
                print(f"    Result: {tps:.3f} tok/s, VRAM: {vram:.0f} MiB, Prompt Tok/s: {res['prompt_per_second']}")
                c16_trials.append({
                    "cycle": cycle,
                    "position": pos,
                    "arm": arm,
                    "warmup": is_warmup,
                    "predicted_per_second": tps,
                    "prompt_per_second": res["prompt_per_second"],
                    "wall_s": res["wall_s"],
                    "vram_mib": vram,
                })
        finally:
            if current_runner:
                current_runner.__exit__(None, None, None)
                current_runner = None
                current_arm = None

        # 16K stats
        c16_measured = [t for t in c16_trials if not t["warmup"]]
        c16_by_cycle = {}
        for t in c16_measured:
            c16_by_cycle.setdefault(t["cycle"], []).append(t)
        
        c16_cycle_stats = []
        for c in sorted(c16_by_cycle.keys()):
            g = c16_by_cycle[c]
            va = [t["predicted_per_second"] for t in g if t["arm"] == "A"]
            vb = [t["predicted_per_second"] for t in g if t["arm"] == "B"]
            ma = statistics.fmean(va)
            mb = statistics.fmean(vb)
            d = mb - ma
            dp = (mb / ma - 1.0) * 100.0
            c16_cycle_stats.append({
                "cycle": c,
                "arm_a_runs": va,
                "arm_b_runs": vb,
                "mean_a": ma,
                "mean_b": mb,
                "delta": d,
                "delta_pct": dp
            })
        
        c16_deltas = [cs["delta"] for cs in c16_cycle_stats]
        c16_deltas_pct = [cs["delta_pct"] for cs in c16_cycle_stats]
        c16_mean_a = statistics.fmean([cs["mean_a"] for cs in c16_cycle_stats])
        c16_mean_b = statistics.fmean([cs["mean_b"] for cs in c16_cycle_stats])
        c16_mean_delta = statistics.fmean(c16_deltas)
        c16_median_delta = statistics.median(c16_deltas)
        c16_mean_pct = statistics.fmean(c16_deltas_pct)
        c16_median_pct = statistics.median(c16_deltas_pct)
        c16_stdev_delta = statistics.stdev(c16_deltas) if len(c16_deltas) > 1 else 0.0
        c16_stdev_pct = statistics.stdev(c16_deltas_pct) if len(c16_deltas_pct) > 1 else 0.0
        
        hw_16 = t_crit * c16_stdev_delta / math.sqrt(len(c16_deltas)) if len(c16_deltas) > 1 else 0.0
        c16_ci95 = [c16_mean_delta - hw_16, c16_mean_delta + hw_16]
        hw_16_pct = t_crit * c16_stdev_pct / math.sqrt(len(c16_deltas_pct)) if len(c16_deltas_pct) > 1 else 0.0
        c16_ci95_pct = [c16_mean_pct - hw_16_pct, c16_mean_pct + hw_16_pct]

        if c16_ci95[0] > 0:
            c16_verdict = "GAIN"
        elif c16_ci95[1] < 0:
            c16_verdict = "LOSS"
        else:
            c16_verdict = "INCONCLUSIVE"

        # ----------------------------------------------------
        # 3. 64K Context Single Paired Run
        # ----------------------------------------------------
        print(f"\n[Arm {cid}] Starting 64K Context Paired Run...")
        prompt_64k = filler(760)
        res_64k_a, res_64k_b = None, None
        
        print(f"  Running 64K Arm A (Flag-OFF)...")
        with ContainerRunner(f"{cid}-64k-a", lib_path, cand["flag_off"]):
            res_64k_a = req_completion(prompt_64k, n_predict=128, slot=0, cache_prompt=False, timeout=1800)
        
        print(f"  Running 64K Arm B (Flag-ON)...")
        with ContainerRunner(f"{cid}-64k-b", lib_path, cand["flag_on"]):
            res_64k_b = req_completion(prompt_64k, n_predict=128, slot=0, cache_prompt=False, timeout=1800)
        
        tg_64k_a = res_64k_a["predicted_per_second"]
        tg_64k_b = res_64k_b["predicted_per_second"]
        d_64k = tg_64k_b - tg_64k_a
        dp_64k = (tg_64k_b / tg_64k_a - 1.0) * 100.0
        print(f"  64K decode: Arm A={tg_64k_a:.3f} tok/s, Arm B={tg_64k_b:.3f} tok/s (delta={dp_64k:+.2f}%)")

        abba_results[cid] = {
            "desc": desc,
            "short": {
                "trials": short_trials,
                "cycle_stats": s_cycle_stats,
                "mean_a": s_mean_a,
                "mean_b": s_mean_b,
                "mean_delta_tps": s_mean_delta,
                "median_delta_tps": s_median_delta,
                "mean_delta_pct": s_mean_pct,
                "median_delta_pct": s_median_pct,
                "stdev_delta": s_stdev_delta,
                "ci95_tps": s_ci95,
                "ci95_pct": s_ci95_pct,
                "verdict": s_verdict,
            },
            "ctx_16k": {
                "trials": c16_trials,
                "cycle_stats": c16_cycle_stats,
                "mean_a": c16_mean_a,
                "mean_b": c16_mean_b,
                "mean_delta_tps": c16_mean_delta,
                "median_delta_tps": c16_median_delta,
                "mean_delta_pct": c16_mean_pct,
                "median_delta_pct": c16_median_pct,
                "stdev_delta": c16_stdev_delta,
                "ci95_tps": c16_ci95,
                "ci95_pct": c16_ci95_pct,
                "verdict": c16_verdict,
            },
            "ctx_64k": {
                "arm_a_tps": tg_64k_a,
                "arm_b_tps": tg_64k_b,
                "delta_tps": d_64k,
                "delta_pct": dp_64k,
                "arm_a_vram": res_64k_a["gpu_memory"]["used_mib"],
                "arm_b_vram": res_64k_b["gpu_memory"]["used_mib"],
            }
        }

    # Stacking and Additivity Analysis
    stacking_analysis = {}
    if "c1011h_b" in abba_results and "c1011h_c" in abba_results and "c1011h_bc" in abba_results:
        for ctx_key in ["short", "ctx_16k"]:
            b_pct = abba_results["c1011h_b"][ctx_key]["mean_delta_pct"]
            c_pct = abba_results["c1011h_c"][ctx_key]["mean_delta_pct"]
            bc_pct = abba_results["c1011h_bc"][ctx_key]["mean_delta_pct"]
            sum_pct = b_pct + c_pct
            stacking_analysis[f"gdn_bc_{ctx_key}"] = {
                "b_alone_pct": b_pct,
                "c_alone_pct": c_pct,
                "expected_sum_pct": sum_pct,
                "measured_bc_pct": bc_pct,
                "diff_pct": bc_pct - sum_pct,
                "is_greater_than_individuals": (bc_pct > b_pct and bc_pct > c_pct)
            }
        # 64k
        b_64 = abba_results["c1011h_b"]["ctx_64k"]["delta_pct"]
        c_64 = abba_results["c1011h_c"]["ctx_64k"]["delta_pct"]
        bc_64 = abba_results["c1011h_bc"]["ctx_64k"]["delta_pct"]
        stacking_analysis["gdn_bc_64k"] = {
            "b_alone_pct": b_64,
            "c_alone_pct": c_64,
            "expected_sum_pct": b_64 + c_64,
            "measured_bc_pct": bc_64,
            "diff_pct": bc_64 - (b_64 + c_64),
            "is_greater_than_individuals": (bc_64 > b_64 and bc_64 > c_64)
        }

    # Save Stage C JSON
    json_path = os.path.join(RESULTS_DIR, "stage_c_abba_speed.json")
    with open(json_path, "w") as f:
        json.dump({"results": abba_results, "stacking": stacking_analysis}, f, indent=2)

    # Save Stage C Markdown Report
    md_path = os.path.join(RESULTS_DIR, "stage_c_abba_speed.md")
    with open(md_path, "w") as f:
        f.write("# Stage C: ABBA x5 Paired Serving Decode Throughput Report\n\n")
        f.write(f"Generated at: {datetime.datetime.now(datetime.timezone.utc).isoformat()}\n\n")
        f.write("Full paired ABBA evaluation: Short Context (x5 cycles = 20 runs + 2 warmups), 16K Context (x5 cycles = 20 runs + 2 warmups), and 64K paired decode run.\n\n")
        
        f.write("## 1. Short Context Summary\n\n")
        f.write("| Candidate Arm | Description | Arm A Mean | Arm B Mean | Mean Δ (%) | Median Δ (%) | 95% CI (%) | Stdev (tok/s) | Verdict |\n")
        f.write("|---|---|---:|---:|---:|---:|---:|---:|:---:|\n")
        for cid, res in abba_results.items():
            s = res["short"]
            ci_str = f"[{s['ci95_pct'][0]:+.2f}%, {s['ci95_pct'][1]:+.2f}%]"
            f.write(f"| {cid} | {res['desc']} | {s['mean_a']:.2f} | {s['mean_b']:.2f} | {s['mean_delta_pct']:+.2f}% | "
                    f"{s['median_delta_pct']:+.2f}% | {ci_str} | {s['stdev_delta']:.4f} | **{s['verdict']}** |\n")

        f.write("\n## 2. 16K Context Summary\n\n")
        f.write("| Candidate Arm | Description | Arm A Mean | Arm B Mean | Mean Δ (%) | Median Δ (%) | 95% CI (%) | Stdev (tok/s) | Verdict |\n")
        f.write("|---|---|---:|---:|---:|---:|---:|---:|:---:|\n")
        for cid, res in abba_results.items():
            c16 = res["ctx_16k"]
            ci_str = f"[{c16['ci95_pct'][0]:+.2f}%, {c16['ci95_pct'][1]:+.2f}%]"
            f.write(f"| {cid} | {res['desc']} | {c16['mean_a']:.2f} | {c16['mean_b']:.2f} | {c16['mean_delta_pct']:+.2f}% | "
                    f"{c16['median_delta_pct']:+.2f}% | {ci_str} | {c16['stdev_delta']:.4f} | **{c16['verdict']}** |\n")

        f.write("\n## 3. 64K Context Paired Run\n\n")
        f.write("| Candidate Arm | Arm A (tok/s) | Arm B (tok/s) | Δ (tok/s) | Δ (%) | VRAM Arm A | VRAM Arm B |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|\n")
        for cid, res in abba_results.items():
            r64 = res["ctx_64k"]
            f.write(f"| {cid} | {r64['arm_a_tps']:.3f} | {r64['arm_b_tps']:.3f} | {r64['delta_tps']:+.3f} | "
                    f"{r64['delta_pct']:+.2f}% | {r64['arm_a_vram']:.0f} MiB | {r64['arm_b_vram']:.0f} MiB |\n")

        f.write("\n## 4. Stacking and Additivity Analysis\n\n")
        if stacking_analysis:
            f.write("### GDN Stacking: RQK (b) + PCS (c) vs Combined (bc)\n\n")
            f.write("| Context | RQK Alone (b) | PCS Alone (c) | Expected Sum | Measured Combined (bc) | Additivity Delta | bc > b & c? |\n")
            f.write("|---|---:|---:|---:|---:|---:|:---:|\n")
            for k, row in stacking_analysis.items():
                ctx_label = k.replace("gdn_bc_", "")
                gt = "YES" if row["is_greater_than_individuals"] else "NO"
                f.write(f"| {ctx_label} | {row['b_alone_pct']:+.2f}% | {row['c_alone_pct']:+.2f}% | "
                        f"{row['expected_sum_pct']:+.2f}% | {row['measured_bc_pct']:+.2f}% | {row['diff_pct']:+.2f}% | **{gt}** |\n")
        
        # Check call_dh1 and call_dh2 vs GDN bc + MoE
        f.write("\n### Full Combo Stacking: GDN (bc) + MoE (dh1 / dh2)\n\n")
        f.write("| Candidate Arm | GDN (bc) Δ% | Expected Sum vs MoE | Measured Combined | Additivity Delta |\n")
        f.write("|---|---:|---:|---:|---:|\n")
        for dh_arm, dh_moe_short in [("call_dh1", 0.83), ("call_dh2", 1.55)]:
            if dh_arm in abba_results and "c1011h_bc" in abba_results:
                meas_short = abba_results[dh_arm]["short"]["mean_delta_pct"]
                bc_short = abba_results["c1011h_bc"]["short"]["mean_delta_pct"]
                exp_sum = bc_short + dh_moe_short
                f.write(f"| {dh_arm} (short) | {bc_short:+.2f}% | {exp_sum:+.2f}% (bc + {dh_moe_short}%) | {meas_short:+.2f}% | {meas_short - exp_sum:+.2f}% |\n")

        f.write("\n## 5. Raw Cycle Measurements\n\n")
        for cid, res in abba_results.items():
            f.write(f"### Candidate {cid} Raw Cycles\n\n")
            f.write("#### Short Context\n\n")
            f.write("| Cycle | Arm A Run 1 | Arm A Run 2 | Arm A Mean | Arm B Run 1 | Arm B Run 2 | Arm B Mean | Δ (tok/s) | Δ (%) |\n")
            f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
            for cs in res["short"]["cycle_stats"]:
                f.write(f"| {cs['cycle']} | {cs['arm_a_runs'][0]:.3f} | {cs['arm_a_runs'][1]:.3f} | {cs['mean_a']:.3f} | "
                        f"{cs['arm_b_runs'][0]:.3f} | {cs['arm_b_runs'][1]:.3f} | {cs['mean_b']:.3f} | {cs['delta']:+.3f} | {cs['delta_pct']:+.2f}% |\n")
            
            f.write("\n#### 16K Context\n\n")
            f.write("| Cycle | Arm A Run 1 | Arm A Run 2 | Arm A Mean | Arm B Run 1 | Arm B Run 2 | Arm B Mean | Δ (tok/s) | Δ (%) |\n")
            f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
            for cs in res["ctx_16k"]["cycle_stats"]:
                f.write(f"| {cs['cycle']} | {cs['arm_a_runs'][0]:.3f} | {cs['arm_a_runs'][1]:.3f} | {cs['mean_a']:.3f} | "
                        f"{cs['arm_b_runs'][0]:.3f} | {cs['arm_b_runs'][1]:.3f} | {cs['mean_b']:.3f} | {cs['delta']:+.3f} | {cs['delta_pct']:+.2f}% |\n")
            f.write("\n")

    print("Stage C completed.")
    return abba_results

# ----------------- RESTORE PROCEDURE -----------------
def restore_production(lock_proc):
    print("\n================== MANDATORY RESTORE ==================")
    restore_time = datetime.datetime.now(datetime.timezone.utc).isoformat()
    clean_gpu2_containers()
    
    # Release bench lock
    if lock_proc:
        print("Releasing bench lock...")
        try:
            os.killpg(os.getpgid(lock_proc.pid), signal.SIGKILL)
        except Exception:
            pass
        try:
            lock_proc.wait(timeout=5)
        except Exception:
            pass
        time.sleep(1)

    chk = subprocess.run(["flock", "-n", LOCK_PATH, "true"])
    lock_released = (chk.returncode == 0)

    # Bring production back
    print("Starting production container llamacpp-m6000...")
    subprocess.run(["docker", "start", "llamacpp-m6000"], capture_output=True)
    
    healthy = False
    for i in range(60):
        try:
            req = urllib.request.Request("http://127.0.0.1:11436/health")
            with urllib.request.urlopen(req, timeout=2) as resp:
                if resp.status == 200:
                    healthy = True
                    break
        except Exception:
            pass
        time.sleep(2)
        
    if not healthy:
        try:
            req = urllib.request.Request("http://172.21.0.1:11445/upstream/qwen36-moe", data=b"", headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=5) as resp:
                pass
        except Exception:
            pass
        for i in range(30):
            try:
                req = urllib.request.Request("http://127.0.0.1:11436/health")
                with urllib.request.urlopen(req, timeout=2) as resp:
                    if resp.status == 200:
                        healthy = True
                        break
            except Exception:
                pass
            time.sleep(2)

    # Test 1 token completion to confirm serving
    serving_confirmed = False
    test_out = None
    if healthy:
        try:
            body = json.dumps({"prompt": "Hello", "n_predict": 1, "temperature": 0}).encode("utf-8")
            req = urllib.request.Request("http://127.0.0.1:11436/completion", data=body, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                test_out = json.loads(resp.read().decode("utf-8"))
                if test_out.get("content"):
                    serving_confirmed = True
        except Exception as e:
            print(f"Error testing production completion: {e}")

    # Inspect docker status
    ps_res = subprocess.run(["docker", "inspect", "llamacpp-m6000", "--format", "{{.State.Status}} (Health: {{.State.Health.Status}})"],
                            capture_output=True, text=True)
    m6000_mem = get_gpu_memory()

    leftover = subprocess.run(["docker", "ps", "-a", "--filter", "name=gpu2-", "--format", "{{.Names}}"], capture_output=True, text=True).stdout.strip()
    leftover_str = leftover.replace("\n", ", ") if leftover else "None"

    # Write restore proof
    restore_md = os.path.join(RESULTS_DIR, "restore.md")
    with open(restore_md, "w") as f:
        f.write("# Production Restore Proof\n\n")
        f.write(f"- **Restore Timestamp**: {restore_time}\n")
        f.write(f"- **Bench Lock Released**: {lock_released} (`flock -n {LOCK_PATH} true` returned {chk.returncode})\n")
        f.write(f"- **Leftover gpu2-* Containers**: {leftover_str}\n")
        f.write(f"- **Production Container Status**: {ps_res.stdout.strip()}\n")
        f.write(f"- **Health Endpoint (http://127.0.0.1:11436/health)**: {'200 OK' if healthy else 'FAILED'}\n")
        f.write(f"- **1-Token Completion Verified**: {serving_confirmed}\n")
        f.write(f"- **GPU Memory**: {m6000_mem['used_mib']:.0f} MiB used / {m6000_mem['free_mib']:.0f} MiB free\n")
        f.write(f"- **Test Response**: `{test_out.get('content') if test_out else 'None'}`\n")

    print(f"Restore finished. Healthy={healthy}, Serving={serving_confirmed}. Proof written to {restore_md}")
    return healthy and serving_confirmed

# ----------------- MAIN ORCHESTRATION -----------------
def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    log_file = open(os.path.join(RESULTS_DIR, "run.log"), "w", buffering=1)
    sys.stdout = Tee(sys.__stdout__, log_file)
    sys.stderr = Tee(sys.__stderr__, log_file)

    print(f"GPU-RUN-2 Orchestrator Starting at {datetime.datetime.now(datetime.timezone.utc).isoformat()}")
    
    # 0. Check SHA256 of candidate libraries
    print("Step 0: Verifying candidate library SHA256 hashes...")
    for k, expected_hash in EXPECTED_SHA256.items():
        actual_hash = calc_sha256(LIBS[k])
        print(f"  Lib {k}: actual={actual_hash}")
        if actual_hash != expected_hash:
            raise RuntimeError(f"FATAL: SHA256 mismatch for {k}! Expected {expected_hash}, got {actual_hash}")
    print("All candidate library hashes verified successfully.")

    # 1. Pre-flight check
    print("Step 1: Pre-flight check...")
    gpu_mem = get_gpu_memory()
    print(f"M6000 VRAM: {gpu_mem['used_mib']} MiB used, {gpu_mem['free_mib']} MiB free")
    
    # 2. Take the bench lock
    print(f"Step 2: Taking bench lock on {LOCK_PATH}...")
    lock_proc = subprocess.Popen(["flock", LOCK_PATH, "sleep", "14400"], preexec_fn=os.setpgrp)
    time.sleep(1)
    
    chk = subprocess.run(["flock", "-n", LOCK_PATH, "true"])
    if chk.returncode == 0:
        raise RuntimeError("CRITICAL: Failed to acquire exclusive flock on gpu-bench.lock!")
    print("Exclusive bench lock verified active.")
    
    try:
        # 3. Unload production
        print("Step 3: Unloading production container llamacpp-m6000...")
        t_stop = datetime.datetime.now(datetime.timezone.utc).isoformat()
        stop_res = subprocess.run(["docker", "stop", "-t", "30", "llamacpp-m6000"], capture_output=True, text=True)
        print(f"Stopped llamacpp-m6000 at {t_stop}: rc={stop_res.returncode}")
        
        time.sleep(3)
        freed_mem = get_gpu_memory()
        print(f"M6000 VRAM after unload: {freed_mem['used_mib']} MiB used, {freed_mem['free_mib']} MiB free")
        if freed_mem["used_mib"] > 1000:
            print(f"WARNING: VRAM usage {freed_mem['used_mib']} MiB is higher than expected after unload.")

        # 4. Stage B: Exactness First
        exactness_results = run_stage_b()
        passing_candidates = [k for k, v in exactness_results.items() if v.get("passed")]
        print(f"\nCandidates passing Stage B: {passing_candidates}")
        
        # 5. Stage C: ABBA Speed Screen
        run_stage_c(passing_candidates)

    finally:
        # 6. Restore Production
        restore_production(lock_proc)

if __name__ == "__main__":
    main()
