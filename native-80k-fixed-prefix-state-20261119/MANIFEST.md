# Packet inventory

| File | Role |
|---|---|
| README.md | Measured result and scope |
| REPRODUCE.md | Offline replay, build and physical rerun prerequisites |
| state.cpp, dispatch.inc | Exact observer and dispatch forwarding source |
| report.py, check-report.py | Exact numeric summarizer and synthetic checks |
| environment.args | Exact admitted feature profile |
| build.sh | Portable CPU-only recipe for the same compiler flags |
| replay.py | Offline equality check over the sanitized transcript and result |
| result.json | Numeric results and typed coverage checks |
| witness.txt | Selected measured graph, state, score and runtime witness records |
| runtime.json | Software identities, device, cleanup and elapsed-time evidence |
| privacy-audit.json | Packet inclusion and exclusion check |
| SHA256SUMS | File integrity manifest |
