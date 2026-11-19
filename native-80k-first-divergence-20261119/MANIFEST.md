# Packet contents

| File | Contents |
|---|---|
| divergence.json | Full anonymous cold and cached top-eight scores at the committed first divergence, ranks of both choices and phase-level rollback/stride counts. |
| rows.jsonl | All 46 scored target rows from the four observed windows, including their commit status and rank spectra. |
| runtime.json | Mapped software and image pins, effective numeric flags, request knobs, physical context geometry, eight-expert offload, phase-bounded route counters, fallback checks, original exit/cleanup statuses and time bounds. |
| serial-revalidation.json | Original serial refusal, successful cleanup, immutable-record revalidation, phase counts and serial row alignment. |
| observer-snippets.cpp | Verbatim excerpts from the measured server implementation covering capture, native row formation, sampling and commit accounting. |
| r0-model.py.in, r0-release.sh.in | Original serial replay adapter and launcher templates. |
| r1-model.py.in, r1-release.sh.in | Native replay adapter and launcher templates with the narrow sampler-snapshot timestamp correction. |
| source-identities.json, rehydrate.py | Byte-exact source checks and private-alias rehydration. |
| export.py, replay.py | Owner revalidation/export and portable public replay. |
| README.md, REPRODUCE.md | Interpretation, scope and replay instructions. |
| privacy-audit.json | Allow-list, row schema and private-pattern screening result. |
| SHA256SUMS | Checksums for all packet payloads. |
