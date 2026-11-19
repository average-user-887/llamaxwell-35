# Q8 GDN intermediate-store cost on Quadro M6000

Eliding the 2 MiB recurrent rollback snapshot saved **1.043–5.337 µs per GDN layer** against the original full kernel. The same-binary comparison saved **1.092–5.391 µs**. The 23 µs standalone and 6.4 µs companion thresholds both missed. The largest measured saving projects to 0.162 ms across 30 recurrent layers before an exact rollback repair. At the previously measured native yield, 100 tok/s still required 0.689 ms per offered round when the diagnostic was run.

| Order | Original, µs | Predicated full store, µs | Predicated elision, µs | Original saving, µs |
| --- | ---: | ---: | ---: | ---: |
| Original, full, elided | 143.508002 | 143.556744 | 142.464995 | 1.043007 |
| Elided, full, original | 144.069746 | 144.189253 | 138.967246 | 5.102500 |
| Elided, full, original | 144.130498 | 144.297495 | 138.974994 | 5.155504 |
| Original, full, elided | 144.105494 | 144.160002 | 138.768747 | 5.336747 |

Each timing interval covered 128 complete kernel launches rotating four seeded, full-size synthetic fixtures. The M6000 ran the 72-block grid with three resident CTAs per SM and 80 registers without local spill traffic. Across every fixture, both residual rows, both convolution states and the final recurrent state matched the original bit for bit. The predicated full-store path also matched the intermediate state. The elided path retained all 524,288 sentinel words in that plane. Each model rollback expects the actual intermediate state. Only the standalone synthetic entry point can access the diagnostic.

The exact original CUDA relink was `369c6b058813a7ca53fc62c67c2a4ea056a8d3e35ee08aca36afca4040975871`. All 17 earlier device functions and 37,104 machine words matched. The compiled diagnostic library was `50cdab271dc3411e4135539f27371b453314b14162e96cc9744efa8f2a994e7d`. The preliminary compile-only design specializing away the store spilled. A second selector formulation also spilled. Neither executed on the GPU. A development predication compile also remains archived. The measured version used a single predicated kernel so both compared treatments share device instructions. Its complete execution and cleanup both returned zero.
