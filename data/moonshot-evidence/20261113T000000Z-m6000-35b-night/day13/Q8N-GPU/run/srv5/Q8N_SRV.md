# Q8N server gates (analyzer output)

| unit | arm | valid/n | median | per-cycle | peak VRAM MiB |
|---|---|---|---|---|---|
| both_deep | F | 0/0 | nan |  | 0 |
| both_deep | Q | 0/0 | nan |  | 0 |
| single_98k | F | 4/4 | 52.66 | 52.67 52.63 | 23109 |
| single_98k | Q | 4/4 | 52.98 | 53.30 52.86 | 21589 |
| append_4k | F | 4/4 | 127.96 | 127.92 128.03 | 23109 |
| append_4k | Q | 4/4 | 127.50 | 127.60 127.48 | 21589 |

Paired per-cycle ratio vs F (median over cycles):
- both_deep Q/F: n/a
- single_98k Q/F: 1.0119 1.0043 -> median +0.81%
- append_4k Q/F: 0.9974 0.9957 -> median -0.34%

Session peak VRAM MiB: F 23109, Q 21589, F 23109, Q 21589, F 23109, Q 21589

## both_deep ms/tok (98K+90K, mean of 2 slots), per cycle

| arm | n cycles | ms/tok per cycle | mean ms/tok | peak VRAM MiB |
|---|---|---|---|---|
| F | 0 |  | n/a | 0 |
| Q | 0 |  | n/a | 0 |

| arm vs F | paired cycles | per-cycle delta ms/tok | mean delta ms/tok | 95% CI | mean delta % |
|---|---|---|---|---|---|
| Q | 0 | n/a | | | |
