# BX result (step 1 only; step 2 NOT started)
| item | value |
|---|---|
| qualifying measured items | 1 independent (A2ROW, -0.3..-0.5%); bundle9 is the same gain packaged; cap2048 fails gap gate |
| pending items | H1, FA0 knobs, D1b (queued behind GPU lock) |
| rule ">=3 compatible items" | NOT MET |
| branch bx-cum / bx_gpu.sh | not created (nothing to cumulate; best case a ~-0.3% bundle, below 1% at 3 sd) |
| GPU | none used |
Re-open BX when H1/FA0/D1b land: if >=2 of them show gain >0 with identical tokens and VRAM <= prod+100 MiB, ladder = A2ROW(bundle9 flags) + H1 + knobs.
