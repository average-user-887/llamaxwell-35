# Native width-three target rollback at 2K

Changing two deliberately wrong suffix tokens changed the untrimmed state. After both independent width-three runs retained the first row and removed the wrong suffix, **all retained and corrected attention-KV bytes, recurrent-state words and compared logits were identical**. The independently re-prefilled width-one reference chose the same correction and the same next choices after correction and four teacher-forced steps. Its internal values differed.

The target-only test used the Quadro M6000 24GB, one sequence, two recurrent rollback slots and 42/42 offloaded layers. Each width-three graph entered the native gated-delta direct-cache path at all 30 recurrent layers. The recurrent serializer reported the logically active convolution and SSM rows at those layers. Dormant rollback slots and the separate learned draft context fall outside the measurement. This is one rejection event per wide arm on one 2,000-token fixture.

| Compared state | Kept, wide A vs B | Corrected, wide A vs B | Kept, wide A vs fresh serial | Corrected, wide A vs fresh serial |
|---|---:|---:|---:|---:|
| Differing attention-KV bytes | 0 / 41,110,784 | 0 / 41,131,288 | 15,676 / 41,110,784 | 30,331 / 41,131,288 |
| Differing recurrent FP32 words | 0 / 16,465,920 | 0 / 16,465,920 | 15,539,561 / 16,465,920 | 15,846,503 / 16,465,920 |
| Differing logits in the compared row | 0 / 248,320 | 0 / 248,320 | 248,320 / 248,320 | 248,320 / 248,320 |

The first run stopped after its initial prefix at position 2003, before any rejection. Its assertion assumed that the standalone partial state, including its eight-byte header, appeared at the end of the full state. Both serializers write one header at byte zero. The correction checks both headers and all 65,864,420 recurrent-body bytes, then extracts the exact attention record between them. The corrected run re-prefilled each of three independent model processes, retained row zero at position 2004 and removed positions 2005–2006 from both wide runs. It used the target's correction at position 2005 and replayed the following four inputs through position 2009.

The original attempt took 34 seconds through cleanup. The corrected three-arm run took 58 seconds through cleanup. The harness recorded graph widths and event order. Individual phase times and throughput are unavailable. Both attempts used a 190-second upper bound and completed cleanup.
