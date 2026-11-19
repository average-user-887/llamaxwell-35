# Native R2 BFCL episode cache smoke

The November 19, 2026 cached smoke ran the same four public BFCL cases on Quadro M6000 in serial and native modes. The cached owner observer was revision 1. Evaluator, runner, and cleanup exit codes were zero in both arms. The evaluator's case wall time sums to 150.104 s for serial and 145.579 s for native, a 3.0% reduction on this pair.

| Cached smoke measure | Serial | Native |
| --- | ---: | ---: |
| Passes | 3/4 | 3/4 |
| Accepted model requests | 53 | 50 |
| Newly evaluated prompt tokens | 25,964 | 25,996 |
| Reused prompt tokens | 243,783 | 236,919 |
| Prompt evaluation time | 72.351 s | 76.520 s |
| Generation time | 75.829 s | 67.208 s |

The per-case score comparison gives native zero wins, zero losses, and four ties. The serial token-cap zero remains in the score and ties a scored zero in the native arm. The arms made different numbers of requests and generated different token totals. These results describe four cases. The separate 160-case cohort remains unevaluated by this package.

Both arms used one slot, disabled the RAM prompt cache, and forced the initial request of each episode cold. The worker allowed reuse only on later requests in that episode and set chunk relocation to zero. All eight episode entries reported zero reused tokens. In the first case, both arms evaluated 2,629 prompt tokens on request 1 and reused 2,795 tokens with 23 newly evaluated tokens on request 2. The request ledger records subsequent reuse for every case in both arms.

The distinct uncached evaluator receipts totaled 500.981 s serial and 455.364 s native across the same case IDs. Their pass counts were 3/4 and 2/4. The uncached native wrapper returned 1 after its evaluator completed with exit 0 because the observer rejected the server log's timestamp-only counter format. The uncached figures are descriptive evaluator results from an earlier run.

- `cached-smoke.json` carries evaluator and runner exits, case scores, wall time, and prompt, generation, and physical counters.
- `cached-requests.json` carries the request order, cache flags, and numeric timings by public case ID.
- `uncached-smoke.json` carries the earlier case scores and timing in a separate receipt.
- `cache_transport.py` is the exact transport source pinned by the worker freeze referenced by both cached arm receipts. It is an Inspect AI and httpx integration component. Its public source digest is covered by the package checksum.

Verify the package with `sha256sum -c SHA256SUMS`.
