graphs: {'decode': 748, 'prefill': 3, 'small': 1196}  timed ms by phase: {'decode': 13642, 'prefill': 30745, 'small': 111534}

| phase | class | ms | % of phase | % of N3 wall (phase share x N3 phase wall / 149 s) | achieved TFLOPS | % fp32 peak | achieved GB/s | % BW peak | roofline bound |
|---|---|---|---|---|---|---|---|---|---|
| prefill | flash-attn | 21478 | 69.9 | 18.8 | 4.22 | 62 | 0 | 0 | compute |
| prefill | dense mmq | 3734 | 12.1 | 3.3 | 4.69 | 69 | 11 | 3 | compute |
| prefill | MoE gate | 1391 | 4.5 | 1.2 | 2.89 | 42 | 17 | 5 | compute |
| prefill | MoE down | 1389 | 4.5 | 1.2 | 2.89 | 43 | 27 | 8 | compute |
| prefill | MoE up | 1373 | 4.5 | 1.2 | 2.93 | 43 | 17 | 5 | compute |
| prefill | GDN | 720 | 2.3 | 0.6 | 0.00 | 0 | 34 | 11 | memory |
| prefill | other | 346 | 1.1 | 0.3 | 0.00 | 0 | 822 | 259 | memory |
| prefill | norm/rope | 315 | 1.0 | 0.3 | 0.00 | 0 | 110 | 35 | memory |
| small | flash-attn | 66624 | 59.7 | 24.1 | 3.46 | 51 | 69 | 22 | compute |
| small | norm/rope | 17563 | 15.7 | 6.3 | 0.00 | 0 | 5 | 2 | memory |
| small | dense mmq | 12331 | 11.1 | 4.5 | 3.62 | 53 | 52 | 16 | compute |
| small | MoE gate | 3885 | 3.5 | 1.4 | 2.42 | 36 | 27 | 9 | compute |
| small | MoE down | 3884 | 3.5 | 1.4 | 2.42 | 36 | 35 | 11 | compute |
| small | MoE up | 3855 | 3.5 | 1.4 | 2.43 | 36 | 27 | 9 | compute |
| small | GDN | 1722 | 1.5 | 0.6 | 0.00 | 0 | 35 | 11 | memory |
| small | other | 1670 | 1.5 | 0.6 | 0.00 | 0 | 23008 | 7258 | memory |
| decode | norm/rope | 6402 | 46.9 | 15.4 | 0.00 | 0 | 0 | 0 | memory |
| decode | flash-attn | 5166 | 37.9 | 12.5 | 2.15 | 32 | 269 | 85 | memory |
| decode | dense mmq | 1613 | 11.8 | 3.9 | 0.72 | 11 | 205 | 65 | memory |
| decode | other | 461 | 3.4 | 1.1 | 0.00 | 0 | 41770 | 13177 | memory |

Note: phase wall assumed: prefill-only 40 s, mixed 60 s (graphs with 2..1023 rows), decode 49 s; shares are of GPU node time (sync-per-graph timing), not host wall. Roofline: fp32 6.8 TFLOPS, 317 GB/s; mmq int8 on sm_52 has no faster path, so fp32 peak is the ceiling used.
