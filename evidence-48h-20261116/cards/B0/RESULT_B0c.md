# B0c: N3R turn construction repair (no GPU)

| item | before (B1/B1c) | after (B0c) |
|---|---|---|
| turn suffix | "emit exactly one JSON tool call" / "rewrite in full" | per-kind instruction + " Continue in detail, at least 300 words, no lists, no code blocks, in English." (all 12 turns, both slots) |
| separator | none | "---\nInstruction:" before the suffix |
| chat wrapper | ChatML + empty think (B0b) | unchanged |
| old behaviour | - | `--old-suffix` |
| dry-run shapes | - | 12/12 turns carry the long instruction; slot 1 has no tool-call/rewrite-only suffix (suffix 204-313 chars) |
| mock check | - | run completes, void_turns=0 (mock cannot judge realism) |
| b1c.sh | aborts when >50% void after S_offA and E_offA | unchanged (already does this; exit 5, trap cleans containers) |

Relaunch (orchestrator, takes the GPU lock itself): `bash <W>/B1c/b1c.sh`
Fixtures and prompts stay in B0/private (never published). Only the runner changed; token shas differ from earlier runs.
Risk: real void rate is unknown until S_offA runs; if still >50% void the script aborts after that arm.
