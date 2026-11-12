# Publication review

## Pre-existing public state
The day-bundle3 source was pushed to public main (f410f991bff88c2abdc7040ba0b01efed06764f1, tag main-day-bundle3) before the advisor's evidence-only directive. It was left as-is; no history rewrite. Component commits (git rev-parse in the repo):
- 2436f7554c0c8d70afc9acbcce85b2457193a516 (BUNDLE3)
- 421f2fee718e2e46bcd0dd43a85c63b6f1c75177
- 007999bfa6d0e37223872e16319349d4e60d0fb1 (N2GDN)

## Status of claims
BUNDLE3 -8.98% is PROVISIONAL (one ABBA block, 8 reps per arm).

## Withheld
Binaries, libs, models, slot and logits dumps, nvprof prof.csv (sha256 pointers in source-manifest.json), fixture text/token arrays (hashes in fixtures/fixtures_manifest.json), E5 build tree/log, GPU device id.

## Redactions applied (legend)
- lab root path -> `<LAB>`
- model dir path -> `<MODELS>`
- IP:port -> `<lab-endpoint>`
- GPU UUID -> `<gpu>`
- hostnames -> `<host>`
- user home/account paths -> `<home>` / `<user>`

## Privacy grep
Command: grep -rniE 'coockie|gmail|token=|api.?key|password|PRIVATE|avg-usr|172\.21|192\.168|GPU-684f|/home/'
Output (all hits are inside this legend/command line; every other file is clean; a rerun also matches this pasted block itself):
```
PUBLICATION_REVIEW.md:24:Command: grep -rniE 'coockie|gmail|token=|api.?key|password|PRIVATE|avg-usr|172\.21|192\.168|GPU-684f|/home/'
```
