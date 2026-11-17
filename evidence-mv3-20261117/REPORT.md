# MV3: optimizing 35B native MTP1 verification at ~80K (Quadro M6000 24G, sm_52)

Branch: mv3-mtp-optimization-20261117 (evidence only, not merged, no production change). Date: 2026-11-17.
Builds on MV2 (evidence-mv2-20261116, base 424828b). Goal was +25% over no-spec at 80K (72.5 tok/s, round <= 27.2 ms). **Not reached: best 64.6 tok/s (+11.6%), gap ~3.3 ms per round.**

## 1. Results (80K, 1 slot, greedy, tok/s; single unpaired runs)
| Config (MTP n-max 1, draft KV F16 unless noted) | tok/s | vs no-spec 57.9 |
|---|---|---|
| no-spec | 57.9 | - |
| MTP1, draft KV F16 | 46.9 | -19.0% |
| + GDN T2 (fused 2-token verify segment) | 51.3 | -11.4% |
| + multi-row FA (MROW) | 57.1-57.3 | -1.4% |
| + merged re-eval (LLAMA_MTP_MERGE_REEVAL) | 58.3-58.7 | +0.7..+1.4% |
| + FA2 NU=4 (MROW=3) | **64.6** | **+11.6%** |
| n-max 2 (merge + MROW) | 46.5 | worse; keep n-max 1 |
| n-max 3 | 39.5 | worse |

Best arm (FA2): verify 24.53 ms, head step 3.48 ms, round ~30.5 ms (budget for the goal: 27.2 ms).
2K context (1 slot): no-spec 80.1 / 80.3 (corp / rand); best MTP 93.9 corp / 96.8 rand (+17-21%).

## 2. Per-op share of the 80K verify (ms per verify graph)
| Op | ms |
|---|---|
| FlashAttn (stock, pre-FA2) | ~12 |
| MoE | ~8.0 |
| GDN (before T2) | ~5.6 |
| output projection | 1.4 |
| rest | ~3 |

FA2 floor measured at 0.89 ms/call x 10 layers (the 1-row kernel is 0.62 ms/call). Per-op numbers come from the instrumented node-timing arm (raw/c10/h_nt80), used for shares only.

## 3. Failed / null results
- MoE2 (two-row fused MoE, union of experts): 7.1-7.7 ms vs 5.5 ms goal, outputs bitwise identical, DRAM-bound; flags left off (patch 05, not adopted).
- FA2 cannot go below 0.89 ms/call with this design (head-pair dual-row split-K); needs a different attention design.
- GPU-side greedy argmax (env LLAMA_GREEDY_BACKEND, default off): net ~0 ms. Host overhead ~1.2 ms; sample_accept 0.39 -> 0.31 ms at 80K but the verify span is unchanged. Not recommended (patch 06).
- Row-projection trimming: nothing to gain.

## 4. Quality gate (raw/qgate/RESULT.md, raw/qgate/analysis.csv)
- The greedy divergence at token 18 vs no-spec is a 0.089-nat near-tie (16 vs 24) tipped by GDN_T2. FA2 and merged re-eval are neutral (merge-off is bit-identical to the full arm at that position). Not a rollback/state bug: teacher-forced fresh-prefill states reproduce the flip.
- BUT stock verify (all options off) also diverges, later (pos 26, a different near-tie, margin 0.69), and T2 shows a consistent -0.15..-0.26 nat bias on the reference token at the tie. **Exactness vs serial decode is NOT established.**
- KL uses top-8 + lumped tail (not full vocab); single fixture.

## 5. Limitations
- Unpaired single runs: run-to-run noise is not bounded.
- 1 slot only; no 2-slot or prefill test.
- Measured: the tok/s, verify/head/round ms and 0.89 ms/call figures. Estimated: the goal gap (~3.3 ms) and any projected gains.

## 6. Next actions
1. T2-vs-stock GDN state compare (GDN_CHECK) to explain the sign bias.
2. Paired, repeated runs (same session, alternating arms).
3. New attention design (shared-memory K/V convert + register-tiled GEMM) to go below 0.89 ms/call.
4. MoE phase-C both-row fusion.
5. 2-slot and prefill test before any adoption.

## 7. Setup and reproduction
- Model: occamy-1.0-Q4_0-headq40b-mtp.gguf (Qwen3.6-35B MoE, native MTP head), sha256 d5ff985eea0bcb3a7305a2033582bf63f756c9f4b3700d96b4939f81ee041eae. Hardware: Quadro M6000 24 GB, driver 580.178.04, 250 W limit, one dedicated GPU, bench lock held. Container image sha256:3a0caa9afea12f5ca5d982621cc2d86b1fa6ac614d735b262fa475d7dc6e2a6d. GGML_CUDA_MAXWELL_GRAPHS unset.
- Patches (patch/, apply in order on 424828b): 01 base (MROW kernel, test cases, LLAMA_MV3_PHASE timers, GDN_T2; up to fe984c5), 02 merged re-eval (d236648), 03 FA2 (26f4013), 04 quality-gate harness (da7ba0e), 05 MoE2 null (7bed9e7), 06 greedy-backend null (d60bf17). All new behaviour is behind env, default off.
- Library sha256: libllama-server-impl.so 1138515898887c6ec6be01a210cfb59bb8e1338ccf6eeef1d032ffd4fdb22b65 and llama-server f70e5e1f71c7ca3c9608884745d76ea2003fd6563e3a62768cd77692a554c405 (all stages); libggml-cuda.so: c5 248e65d2b3b5ac33a8a5d8f561f4115db86a46e72e969672b6c603eaca15853d, c7-c10 569e992e0597567620903d3c439616cbd6235e40365f0c28c35ae3506dad19de, c11 094cc263214e5f743d42a44a977f1b3cb5443ca50bc9e8aa90b787195afd9520.
- Stages: c5 no-spec/MTP1 phase timing (p_*), c7 +T2 (t2_*), c8 +MROW=1 (m2_*), c9 +merge (g_*), c10 node timing and n-max 2/3 (h_*), c11 FA2 NU=4 (f_*). raw/<stage>/<arm>/ holds result.json (token ids and timings only; token strings stripped), phase.tsv, mv1_host.txt, arm.txt; raw/<stage>/ holds steps.log, gpu.txt, build.sha256.
- Env per arm: T2 `GGML_CUDA_OCCAMY_GDN_T2=1`; MROW `GGML_CUDA_OCCAMY_FA_MROW=1`; FA2 `GGML_CUDA_OCCAMY_FA_MROW=3 GGML_CUDA_OCCAMY_FA2_NU=4`; merge `LLAMA_MTP_MERGE_REEVAL=1`; timers `LLAMA_MV3_PHASE=/mv1out/phase.tsv LLAMA_MV1_TRACE=1`.
- Arm pattern (from harness/mv3_gpu.sh, placeholders):

```
docker run -d --name nmv3-<TAG> --device nvidia.com/gpu=<GPU_INDEX> -p 127.0.0.1:<PORT>:8080 \
  -v <MODELS_DIR>:/models:ro -v <LIBS_DIR>:/candidate:ro -v <OUT_DIR>/<TAG>:/mv1out \
  --entrypoint /candidate/llama-server -e LD_LIBRARY_PATH=/candidate <PROD_GGML_CUDA_ENVS> \
  -e LLAMA_MV3_PHASE=/mv1out/phase.tsv -e LLAMA_MV1_TRACE=1 \
  -e GGML_CUDA_OCCAMY_GDN_T2=1 -e GGML_CUDA_OCCAMY_FA_MROW=3 -e GGML_CUDA_OCCAMY_FA2_NU=4 -e LLAMA_MTP_MERGE_REEVAL=1 \
  <IMAGE> -m /models/occamy-1.0-Q4_0-headq40b-mtp.gguf --alias <ALIAS> --override-kv tokenizer.ggml.pre=str:qwen35 \
  -ngl 999 -np 1 -c 84224 -b 2048 --mixed-batch-size 1024 --prefill-single-stream --checkpoint-min-step 2048 \
  -fa on --host 0.0.0.0 --port 8080 --jinja --metrics -ub 1024 --cache-ram 12288 --ctx-checkpoints 2 \
  --spec-type draft-mtp --spec-draft-n-max 1 --spec-draft-type-k f16 --spec-draft-type-v f16 -lv 4
python3 -I harness/mv2_client.py 127.0.0.1:<PORT> <TAG> 1 80000 256 2 <OUT_DIR>/<TAG>/result.json   # np, prompt_n, n_predict, reps
```

Whole stage: `N3_GPU=<GPU_INDEX> N3_UNLOAD_URL=<SWAP_PROXY>/unload/<model> N3_PORT=<PORT> N3_LIBS=<LIBS_DIR> bash harness/mv3_gpu.sh c11`. Fixtures: ~80000-token public stdlib-corpus text (greedy), 2000-token corpus ("corp") and seeded random hex words ("rand").
