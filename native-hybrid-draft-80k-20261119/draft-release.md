# N40 native MTP1 final query

The hybrid writes all two or three contiguous draft K/V rows in the original merged call. It runs the Q/gate projection, gated attention, residual, routed and shared experts, and output head only for the final row. N1 already computed logits for only the requested row. Native partial-cache prefill and target verification use their original graph builders.

## Local operator

With the M6000 UUID from the lab inventory set in N3_GPU, use the following commands from this directory. The second command prints its private receipt path. The owner reviews its pair.json before the optional long pair.

    ./release.sh --check
    ./release.sh 2k
    ./release.sh 80k --after-2k /absolute/path/to/receipt-2k-TIMESTAMP-PID

Each command uses the existing N6/R2 profile, signed-off model file identity from N16, single-slot manual route, same runtime image, four mapped DSO hashes and PID 1 executable, 42/42 offload, current locked Fleet admission, owned-container reclamation, and 06:50 Warsaw cutoff. The short triplet reserves up to 120 seconds per arm including cleanup, the long pair up to 390 seconds per arm. Both are serial. Short arms run merged, eager split and hybrid on identical new native DSOs. LLAMA_MTP_LAST_QUERY defaults off. Split alone uses the original LLAMA_MTP_MERGE_REEVAL=0. Long runs compare fresh split and hybrid and report the dated N27 merged control separately. Restore the original overlay by selecting night-n6-stack/libs.

## Verification

At 2K all three arms do a cold and same-slot cached greedy replay, then a separate 12-token temperature-0.3 fallback request. For each actual short graph, the diagnostic inspects the successful compute graph for the rows written by each SET_ROWS node, their order before attention, the Q projection and Flash Attention dimensions, padded key width, mask height, and post-FFN rows. Read the next proposal at the same position and check its accepted or rejected predecessor. Native one-sequence cache files after the first genuine acceptance and rejection retain full cell metadata and both 512-value F16 payloads for every occupied position. The comparator requires identical position-sorted K/V bytes for merged versus hybrid in these snapshots. It reports numerical differences versus eager split separately. Raw logs, inputs, snapshots, logits and output IDs stay inside the private receipt tree. The original native getter remains called, and a separate marker detects actual consumption of its hidden row beyond the terminal depth. Each short arm also checks the non-greedy request used full-width computation. Merged and hybrid fallback IDs must match.

The original draft builder attaches its full hidden tensor before gathering output IDs for logits. Masked extraction copies the first row from that tensor, which need not be the output row. At depth one, native draft() terminates before it reads a hidden value. At greater depths, it uses the value for another proposal. The private optimization therefore requires a real native greedy depth-one call on the sole active speculator, merged re-evaluation, one private MTP block, the admitted model path and shape, F16 KV, flash attention, native four-section text positions, a single sequence, and a sole final output. Unknown layouts, prior seed/target adapters, recurrent draft blocks, rotation, warmup, training, and unsupported batch types take the full graph. General decode and graph reservation never request this optimization. A separate graph type keys all reuse. Original CALLER sync and seed getters are preserved.

CPU smoke compiled the production position and mask view expressions, row guard, and speculative terminal branch. Native CPU GGML IMRoPE and Flash Attention ran on widths two and three with padded keys. Both matched the original last row exactly. Wrong-mask controls differed by 0.753 and 1.054. The terminal branch consumed zero hidden values at depth one and one or two values at depths two and three. This exercises isolated tensor operations without model inference. Actual GPU dispatch, numeric agreement and speed require the operator's receipts.

The long pair omits per-step tracing, fallback requests and cache dumps. A single successful hybrid call reports its actual write and attention dimensions. The shared R2 q1/q2 markers establish dispatch on the combined runtime. Compare cold and warm end-to-end wall, emitted token rate, accepted/proposed counts, and milliseconds per proposal separately. A further run needs about 0.4 ms saved per proposal versus split or a cold-wall recovery at equivalent outputs.

## Rebuild

Copy night-cache-only/src and night-cache-only/build into independent scratch, preserving timestamps. Apply last-query.patch to the private source with patch -p1. Run build.sh, package.py, then cpu-check.sh. The compiler image is pinned and offline and builds only the existing Qwen graph unity unit, llama-context.cpp and common/speculative.cpp on cores 2–3 at nice 19. N6 supplies all other shipped objects. SOURCE.sha256 covers each original and modified file. OBJECTS.json pins all final libraries and links. The private patch reverses on the packaged source. No production or original source changes are required.
