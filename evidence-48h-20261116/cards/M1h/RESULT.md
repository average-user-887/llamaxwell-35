# M1h: trunk tensor hash comparison (CPU only)

| Item | Value |
|---|---|
| Served GGUF | <MEDIA>/Storage/uroboros-models/occamy-1.0-Q4_0-headq40b.gguf (733 tensors) |
| MTP GGUF | <MEDIA>/Storage/uroboros-models/occamy-1.0-Q4_0-headq40b-mtp.gguf (753 tensors) |
| Trunk tensors differing | 0 |
| Trunk tensors missing from MTP | 0 |
| MTP-only extra tensors | 20 (blk.40.* MTP head, expected) |
| Hashes | M1h_hashes.json (same dir) |

Verdict: equal. All 733 served trunk tensors match by name, type, shape, byte size and sha256 of data.
