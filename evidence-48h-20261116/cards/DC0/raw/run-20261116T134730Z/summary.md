graphs: {'dec2': 1172, 'decode': 744, 'prefill': 3, 'small': 26}  graph ms: {'dec2': 36141, 'decode': 13467, 'prefill': 30796, 'small': 75245}

| class | 1-row ms/step | 1-row nodes/step | 1-row us/node | 2-row ms/step | 2-row nodes/step | 2-row us/node | ratio 2/1 | flag |
|---|---|---|---|---|---|---|---|---|
| attention | 6.90 | 10.0 | 690.3 | 13.53 | 10.0 | 1353.3 | 1.96 |  |
| norm-led chain fused w/ MoE | 4.32 | 40.0 | 108.1 | 8.02 | 40.0 | 200.4 | 1.85 | ratio>1.3 |
| norm-led chain fused w/ GDN | 3.84 | 30.0 | 128.1 | 5.93 | 29.9 | 198.4 | 1.54 | ratio>1.3 |
| dense mmq | 2.15 | 41.0 | 52.4 | 2.42 | 41.5 | 58.2 | 1.12 |  |
| other | 0.52 | 214.0 | 2.4 | 0.56 | 208.1 | 2.7 | 1.09 |  |
| norm-led chain fused other | 0.24 | 31.0 | 7.8 | 0.25 | 31.4 | 7.8 | 1.02 |  |
| norm/rope | 0.12 | 20.0 | 6.0 | 0.12 | 20.0 | 6.1 | 1.01 |  |
| GDN | 0.00 | 0.0 | 0.0 | 0.01 | 0.2 | 29.1 | 0.00 |  |
| MoE experts | 0.00 | 0.0 | 0.0 | 0.00 | 0.0 | 1.5 | 0.00 |  |

step total ms (timed graph): 1-row 18.10, 2-row 30.84, ratio 1.70
