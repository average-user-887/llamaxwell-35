# Decision 170 from one restored 80K prefix

A live serial and native MTP1 replay chose different tokens at decision 170. The serial choice led by 0.66038 nat in the live serial arm. The native choice led by 0.53683 nat in the live native arm. Both samplers followed their target’s raw top score. The live native verifier accepted the first row’s proposal and chose output 170 on row one, zero-based. Its preceding verifier round also accepted the proposal.

From a shared, byte-verified reconstructed prefix, the accepted two-row call and two consecutive single-row calls both chose the live native choice. The two widths changed all 248,320 decision-170 logits and produced different active K/V, recurrent convolution and SSM bytes. The width change did not reproduce the live choice reversal on this prefix.

| Target branch on the reconstructed prefix | Winning choice | Lead over the live serial choice |
|---|---|---:|
| Accepted two-row verifier | Native | 1.49843216 nat |
| Two single-row target steps | Native | 1.12422752 nat |
| Single-row correction after either forced rewind | Native | 1.97565460 nat |

The two forced proposals each scored below their own verifier’s row-zero winner. Their untrimmed K/V, convolution and SSM state differed. All 248,320 row-one logits differed as well. Row zero matched bitwise between them and matched the accepted pair’s row zero. After removal of one position, active K/V, every convolution row and every SSM row were bitwise equal across the two branches. After the common single-row correction, those state components and all 248,320 output logits again matched bitwise. The retained and corrected active states differed from the serial branch.

The run used one Quadro M6000, one sequence, 42/42 GPU-offloaded layers and all eight routed experts per token. All seven measured target calls submitted one graph with the expected row counts and ten long-context Sparse R2 attention dispatches at padded key width 80,384. Each of the three q2 graphs recorded 30 successful GDN T2 dispatches with zero fallbacks and 40 paired MoE launches with 40 suppressed second callbacks. The Q8 GDN occupancy, Top16 and q2 stride markers were observed. The run and owned-container cleanup both exited 0 in 311 seconds under a 430-second cap.

The shared prefix was reconstructed with a cold prefill and 167 teacher-forced single-row target steps. The matched serial control retained one rewind slot and two permitted outputs. The live serial server had zero rewind slots and one permitted output. Those configuration and history differences remain relevant to the original divergence. The two forced rewinds establish local suffix isolation for the sampled proposals. Earlier live proposals, accepted states, rewind choices and warm-slot history remain unmeasured. Identical retained bytes across the two trims also leave open a fault that would select the same incorrect plane in both branches.
