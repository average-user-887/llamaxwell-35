# Compact-only R1 shadow qualification at 80K

On one Quadro M6000 with eight experts, one slot and one native MTP proposal, the R1 shadow replay completed with runner and cleanup exit codes of zero. Cold and warm outputs matched all 3,264 archived native token IDs and both full texts. Per-phase cache and acceptance totals matched. The validator checked both target choices on 1,743 compact launches, totaling 3,486 row comparisons with zero mismatches and zero guard refusals. Physical down4 measurements remained zero.

The shadow path retained full logits and ran the stock CPU sampler for verification. Native and no-MTP serial timing are outside this shadow packet. The workload has no independent semantic score here. [`shadow-results.json`](shadow-results.json) is the allowlisted export from the sealed receipt. [`REPRODUCE.md`](REPRODUCE.md) covers the pinned sources and replay.
