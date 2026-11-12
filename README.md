# llamaxwell-35

Public audit repo for the **Occamy 35B MoE** (`qwen36-35b-moe`) llama.cpp build on the Quadro M6000 (sm_52). Models, weights and saved user conversations are not included.

Start with [`audit/PROVENANCE.md`](audit/PROVENANCE.md). The upstream base is `ggml-org/llama.cpp` tag **b11552** (commit `23b0202a189c44a54625aadcb37a946dd1d6278d`); every local change is a separate commit on top, so `git log -p` is the full diff.

## Codex investigation, 12 November 2026

This separate branch contains a **default-off experimental** packed-GQA prefill path and mixed-scheduling cap. Deep-append prompt time improves approximately 26–27%, but a 65K checkpoint shows substantial probability drift: KL 0.294 and total variation 20.85%. This is not a quality-equivalent production release or a decode-speed claim.

Read the [results and limitations](experiments/codex-20261112/PUBLIC_RESULTS.md), [build notes](experiments/codex-20261112/BUILD.md), and [artifact manifest](experiments/codex-20261112/MANIFEST.json). The evidence package includes the tested runtime, sanitized receipts and rejected experiments. Extract its category archives into one directory to resolve the report's evidence links.
