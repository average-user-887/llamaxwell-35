# Q8N server gates (analyzer output)

| unit | arm | valid/n | median | per-cycle | peak VRAM MiB |
|---|---|---|---|---|---|
| both_deep | F | 6/6 | 58.92 | 58.61 59.03 58.89 | 23047 |
| both_deep | K | 6/6 | 60.77 | 60.89 60.70 60.93 | 22235 |
| both_deep | Q | 6/6 | 58.91 | 58.76 58.85 58.95 | 21527 |
| single_98k | F | 0/0 | nan |  | 0 |
| single_98k | K | 0/0 | nan |  | 0 |
| single_98k | Q | 0/0 | nan |  | 0 |
| append_4k | F | 0/0 | nan |  | 0 |
| append_4k | K | 0/0 | nan |  | 0 |
| append_4k | Q | 0/0 | nan |  | 0 |

Paired per-cycle ratio vs F (median over cycles):
- both_deep K/F: 1.0390 1.0284 1.0347 -> median +3.47%
- both_deep Q/F: 1.0026 0.9970 1.0010 -> median +0.10%
- single_98k K/F: n/a
- single_98k Q/F: n/a
- append_4k K/F: n/a
- append_4k Q/F: n/a

Session peak VRAM MiB: F 23047, Q 21527, K 22235, Q 21527, F 23047, K 22235, Q 21527, F 23047, Q 21527, K 22235, F 23047, Q 21527, K 22235, Q 21527, F 23047

## both_deep ms/tok (98K+90K, mean of 2 slots), per cycle

| arm | n cycles | ms/tok per cycle | mean ms/tok | peak VRAM MiB |
|---|---|---|---|---|
| F | 3 | 34.168 33.925 34.007 | 34.033 | 23047 |
| K | 3 | 32.904 33.005 32.883 | 32.931 | 22235 |
| Q | 3 | 34.105 34.053 33.997 | 34.052 | 21527 |

| arm vs F | paired cycles | per-cycle delta ms/tok | mean delta ms/tok | 95% CI | mean delta % |
|---|---|---|---|---|---|
| K | 3 | -1.264 -0.919 -1.124 | -1.102 | [-1.533, -0.672] | -3.24% |
| Q | 3 | -0.063 +0.128 -0.010 | +0.018 | [-0.227, +0.264] | +0.05% |
