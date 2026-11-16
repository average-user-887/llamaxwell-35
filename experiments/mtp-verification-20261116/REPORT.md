# Occamy 35B MTP slowdown: evidence audit and next diagnostic
Date: 2026-11-16. CPU/read-only investigation; no GPU use, production changes, new performance measurements, or head training.

## Decision
Prioritize **verification-path diagnosis**, not a blind replacement/training of the draft head. There is a concrete source-level explanation worth testing: the fast normal-decode paths have shape restrictions that speculative verification can violate. Acceptance is only one term of the speed equation. A more accurate head cannot make an intrinsically too-expensive verifier cheap.

This is a verified source audit, **not a measured allocation of the reported native-MTP slowdown**. Claude's MH0 timing breakdown and deployed runtime trace have not been independently inspected here.

## Evidence status
* Independently fetched 35B main: `f410f991bff88c2abdc7040ba0b01efed06764f1`.
* Independently fetched the source at the user-reported batch8 build revision: `d4c13eb607188e09d2dc4feba9202ec7b7c31db8`. Matching a source revision does not prove the loaded binary was built from it.
* Independently read the prior NG8 audit, and the current publisher model card.
* MH0 below is explicitly **user-relayed producer evidence**. Its full raw files are still needed before calling the exact native-MTP causal attribution confirmed.
* No fresh hardware experiment should duplicate Claude's existing runs or bypass the shared GPU queue.

## What MH0 reports (not remeasured)
Production model has 733 tensors and no native MTP head; the normal output projection is not MTP. The experimental GGUF already includes the released head's selected matching tensors; experts and complete trunk compatibility were not checked. Consequently “try the released head for the first time” is the wrong next experiment.

Native head previously fits only two allocated 84,224-token slots, with ubatch 1024, q8 draft KV and checkpoint setting 2. Reported peak 24,313 MiB leaves only 263 MiB. Larger configurations either fail allocation or generation. Allocation capacity must not be described as a fully populated workload.

Reported decode cycle 57–73 ms versus a stated 32 ms break-even, and 6–10% slower prefill. These numbers imply a **44–56% cycle-time reduction** is necessary to reach that reported threshold, if definitions and workload are comparable. Do not label 57–73 ms “verification alone” until the timer scope is established. Do not derive delivered tok/s without committed-token counts and concurrency.

The head targets base revision c1ce8477 whereas MH0 identifies current base 72ca9603. Compare full trunk/metadata and all head tensors before blaming a mismatch; different revision strings alone do not prove incompatible weights.

## Concrete dispatch evidence
At the reported batch8 revision:
1. [Specialized full-attention matchers](https://github.com/average-user-887/llamaxwell-35/blob/d4c13eb607188e09d2dc4feba9202ec7b7c31db8/ggml/src/ggml-cuda/fattn-occamy.cu#L811) reject query length other than one (also line 869). Multi-query verification therefore cannot use these exact one-query paths. Identify the fallback actually launched.
2. [Fused MoE matcher](https://github.com/average-user-887/llamaxwell-35/blob/d4c13eb607188e09d2dc4feba9202ec7b7c31db8/ggml/src/ggml-cuda/occamy-fused.cu#L730) permits only one activation row or a flag-enabled two-row case. Larger verification batches are outside this fusion. Two independent decode rows are not evidence of efficient three-or-more-row verification.
3. [Fused GDN matcher](https://github.com/average-user-887/llamaxwell-35/blob/d4c13eb607188e09d2dc4feba9202ec7b7c31db8/ggml/src/ggml-cuda/occamy-gdn.cu#L839) supports narrow one/two-sequence shapes and explicitly checks one time step per sequence in convolution/GDN tensor shapes. Multiple consecutive positions require recurrent state handling; they are not interchangeable with independent sequences.
4. [Native MTP driver](https://github.com/average-user-887/llamaxwell-35/blob/f410f991bff88c2abdc7040ba0b01efed06764f1/common/speculative.cpp#L1402) describes Qwen35/MoE as one trained head, and enables hidden-state extraction on target and draft contexts. Transfers, synchronization, head projection, rollback and replay need separate timing rather than assignment to “drafting.”

These are confirmed restrictions, not percentages of runtime. Verify observed tensor shapes, dispatch counters, fallback kernels and recurrent snapshot behavior in the actual binary.

## Why 27B can benefit while 35B loses
Do not compare total parameter count as if it measured work per token. The 35B is MoE; the 27B is dense, with separate optimization histories and dispatch paths.

The [27B MMVQ implementation](https://github.com/average-user-887/llamaxwell-27b/blob/5007689d2a9714faef1d873f294e2f415a8bc3a6/ggml/src/ggml-cuda/mmvq.cu#L880) has specialized Q4_0 small-column coverage up to four, with another dispatch check at line 1338. This supports a plausible small-verification-batch advantage, but does not independently measure its share of the user's MTP2 gain. It does not prove equivalent small-column kernels are absent in 35B; the relevant contrast is the actual dispatched path, not the mere existence of code.

For MoE, consecutive tokens can route to different experts. Batched verification may increase the union of experts accessed, routing/gather work and weight traffic instead of amortizing the same dense weights efficiently. This is a hypothesis to quantify, not a measured claim about this run. A highly optimized baseline also raises the hurdle for any speculative method. The user's existing 27B draft-length sweep already selected two; do not repeat a generic 1–6 sweep.

## Historical corroboration, not native-MTP attribution
The [NG8 audit](https://github.com/average-user-887/llamaxwell-35/blob/3167047d4d533b1cde3438ffad83c149e8fa6d36/experiments/codex-20261112/head9/NG8-AUDIT.txt) records 267/340 committed drafts (78.53%) yet 11.181 s wall time versus 5.672 s without speculation. Exact output tokens matched. This establishes that high acceptance alone did not ensure speed in that fixture.

It was n-gram speculation, a short prompt, and the candidate also enabled the head9 allocation workaround. It is **not a clean native-head comparison**. The audit identifies checkpoint/replay and fusion-shape hypotheses, but explicitly lacks component timing. Never transfer its timing percentages to native MTP.

## What the publisher actually establishes
The [current card](https://huggingface.co/Accio-Lab/occamy-1.0-MTP) describes an experimental, non-PEFT head adapted from Qwen3.6. Only fusion/normalization parameters were trained while the base and remaining head stayed frozen. Its earlier SGLang BF16 MTP1 heldout acceptance was 87.60% versus 87.31% with the donor head, under short-context, single-request greedy conditions. This small difference is not a speed result. The card now also reports native three-step correctness checks with a patched SGLang/H200 setup, but says final performance is unmeasured. GGUF/Maxwell deployment gains do not follow from that result. Required runtime hooks address state correctness; do not blindly transplant them into llama.cpp.

## Break-even test before spending training budget
Measure for each matched workload:
* T0: no-spec wall decode time per committed token.
* C: mean speculative-cycle wall time, including draft, verification, checkpoint/restore, replay, sampling and transfers; account for overlap without double counting.
* E: mean committed tokens per cycle, measured at the server output, not proposer acceptance.
* Speedup = E × T0 / C. Use means over the same measured interval at fixed concurrency. For aggregate throughput count committed tokens across all active slots in both arms; for per-slot latency use consistent per-slot accounting. Never mix aggregate tokens with single-slot baseline time.

If V is a lower bound on non-overlappable verifier critical-path wall time (not summed kernel durations) and Emax the maximum committed tokens for the implemented draft policy, **V >= Emax × T0 rules out a win even with free, perfectly accurate drafting**. Use this bound to decide whether head training or 1660 Ti placement can possibly help. Establish the exact bonus-token policy rather than assuming Emax.

## Bounded diagnostic for Claude
1. Publish existing MH0 and Nov15–16 native-MTP logs first: sanitized commands, build/library/model hashes, timing scopes, populated context lengths, per-slot concurrency, accepted/committed counts, output parity, peak VRAM and rejection/rollback counts.
2. Preserve GPU lock ownership and production. Reuse existing evidence where sufficient. Run no new GPU task from this audit.
3. At the next authorized diagnostic slot, use the same experimental head-bearing model for no-spec and MTP arms, same target precision, sampler, prompts, lengths and flags. Separately quantify the production-vs-head-bearing-model memory/prefill penalty. One-slot and two-slot results must remain separate.
4. Instrument draft versus target verification; target full attention, GDN, MoE/router, output head; checkpoint save/restore/replay; host gaps and transfers. Log actual sequence/position/row shapes and matcher rejection reasons. Compare an uninstrumented matched control to bound profiling overhead.
5. Validate accepted/rejected prefixes and recurrent state restoration against no-spec outputs, including forced rejection tests, not just all-accepted cases. For sampled runs, use the correct speculative acceptance procedure and distribution checks.
6. Choose the dominant measured fix: multi-query attention retaining decode economics, rollback-aware multi-position GDN, or small-row MoE verification. Do not disable snapshots/checks for speed.
7. Only test 1660 Ti placement if removing measured draft cost can clear break-even. Account for Turing-compatible build, complete runtime memory, transfer/sync cost and serial dependencies. It does not move the M6000 verifier.
8. Publish matched repetitions, raw CSV/JSON, trace excerpts, confidence/variance, exact commands and rollback/production-health receipt on an experimental branch. Keep main unchanged.

## Head replacement/training decision
**Swap:** only an architecture/tokenizer/hidden-state-compatible head is plausible. Arbitrary adapters or heads from another family are not plug-and-play. Check complete weight lineage, quantization and target compatibility first.

**Train:** potentially useful if verifier economics leave room and local acceptance is the limiter. Adapt against the exact frozen deployed target and real Hermes workload distribution, with untouched evaluation splits and long-context slices. Optimize committed-prefix length at the actual draft depth, not just teacher-forced top-1 agreement. Validate GGUF conversion and end-to-end exactness/sampling correctness after training. A smaller new head is an architecture/runtime engineering project, not merely more SFT data.

**Current priority:** quantify verification-path loss, then decide. The evidence supports diminishing returns for more blind decode-kernel tuning, not a conclusion that all MTP or all 35B optimizations are exhausted.

## Review and scope
Independently reviewed for evidence labeling, arithmetic, accounting and sanitization before publication. Source/metadata inspection only; no runtime tests or new benchmark claims. Review requested explicit concurrency accounting and critical-path bounds; incorporated above. No raw prompts, client data, credentials or private session links included.
