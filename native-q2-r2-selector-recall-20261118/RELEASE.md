# Independent 80K midpoint recall with the exact Q2 selector

This one-request card uses the previously verified full-eight-expert Q2 MoE/Q8A DSO 9e999bcaf04397f267548abdfa3e326b65bee122fb1e24f92be55529f40a335e with the exact fast selector enabled. It reuses the independent N50/N88 scored midpoint fixture. The reference fact is at exact prompt token position 40,000. The gate requires the expected fact, all seven response token IDs matching the pinned archives, and EOS. The output payload stays private.

The typed model checks bind the exact live server and library identities, actual flags, full GPU offload, eight effective experts and native depth one. The fast selector marker must occur inside that request at 80,128 padded keys, with two-row sparse R2 diagnostics and the existing aggregate sparse attention count consistent with actual offered drafts. This is a scoped quality check, separate from the model's forced-throughput continuation.

Run `./release.sh --check` for the CPU-only archive, source, and layout verification. Under the root owner's exclusive M6000 lease, set `N3_GPU` from LAB_FACTS and `ROOT_GPU_DEADLINE_EPOCH`, then run `./release.sh --run recall`. The single 430-second cap covers startup, inference, owned cleanup, and must fit by 16:57 CET. The launcher prints `Q2_R2_RECALL_RECEIPT`.
