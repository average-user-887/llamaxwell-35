# Native hybrid MTP1 reaches 97.29 tok/s at 80K

One Quadro M6000 ran Qwen 35B MoE with all eight experts and one 80K slot. Cached decode reached **97.29 tok/s with native MTP1 versus 76.72 tok/s with speculation disabled**, a **26.81%** same-build gain. The hybrid remains **0.52128 ms per offered round** from 100 tok/s at its observed yield.

| 80K request | Same-build no-MTP | Native hybrid MTP1 | Native change |
| --- | ---: | ---: | ---: |
| Cached decode | 76.72 tok/s | 97.29 tok/s | +26.81% |
| Cached wall time | 40.169 s | 31.864 s | −8.305 s |
| Cold decode | 73.99 tok/s | 96.57 tok/s | +30.52% |
| Cold wall time | 265.084 s | 268.116 s | +3.032 s |

Both arms used the same five mapped object identities, executable flags and CUDA treatment. Their actual argv differed only in speculation. The serial process reported one context with zero rollback rows and took the warm Q1 path. Both offloaded 42 of 42 layers, evaluated 80,000 cold input tokens and generated 192, then reused 79,996 warm input tokens, evaluated four and generated 3,072. Both terminal validators and cleanup completed successfully. [Paired measurements](same-build-pair.json) · [Independent formula and mapping audit](pair-audit.json)

The native graph wrote three draft K/V rows and executed one final query and one FFN row in its first measured 80K call. Native accepted 1,429 of 1,642 warm proposals. Against the previously repeated split-native build, the hybrid saved **0.16792 ms per offered round** and raised cached decode from **96.45 to 97.29 tok/s**. That split comparison uses the archived baseline. The fresh serial arm supplies the same-build no-MTP denominator. [Hybrid dispatch and complete typed event index](trace-index.json) · [Archived split comparison](hybrid-vs-split.json)

Each arm reproduced all 3,264 token IDs and both full output texts from its corresponding archived mode. The native and serial outputs first differed at token 170. The cached native continuation repeated 127 of 304 normalized nonblank lines and 39 of 76 in its final quarter. The serial continuation repeated 73 of 283 and 16 of 71. Neither showed a comment-only ending or malformed characters. This was one forced, unscored continuation. [Repetition and parity](repetition.json) · [Source and replay instructions](REPRODUCE.md)
