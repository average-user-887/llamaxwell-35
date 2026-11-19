# Evidence and replay

- `shadow-observations.json` contains the allowlisted aggregate export from the single terminal shadow attempt. The owner source locates the sealed receipt via the `BREAK100_COMPACT_MODEL_RECEIPT` record. The exporter checks all receipt payload entries, the freeze binding, mapped software identity, terminal status and request-boundary witnesses, and compares actual token sequences privately with the sealed archived native scorer control.
- `source/` carries the initial frozen changes and reconstruction files. The three patches and compact implementation headers and CUDA source are exact. Build, link, ABI, loader, CPU replay and supervisor evidence comes from the device-free initial build. `source-pins.json` and the owner checksum manifests pin the original source and binary identities. Aliased templates reproduce exact owner bytes after local rehydration.
- `source/spec-type-evidence.txt` records the exact archived lines used to explain compact admission refusal. The remaining guard context is present in the full server patch.
- `export-shadow.py` replays on the owner archive. `verify-packet.py` verifies source and aggregate consistency without the archive. `SHA256SUMS` covers the complete packet payload.

## Offline verification

Run `python3 -B verify-packet.py` from this directory. This requires only Python and the packet. Private archive replay additionally requires the sealed candidate, receipt and archived native scorer control. Use `python3 -B export-shadow.py --owner-card OWNER_CARD --receipt SHADOW_RECEIPT --reference ARCHIVED_NATIVE_SCORER_CONTROL` and compare stdout with `shadow-observations.json`.

To reconstruct the exact aliased source files, create an alias JSON privately with the keys listed in `source/source-pins.json`. Supply the original source mount and archived serial reference and the original model directory name, filename, stat pair and SHA-256. Run `python3 -B source/rehydrate.py --aliases PRIVATE_JSON --output PRIVATE_EMPTY_PATH`. The helper verifies reconstructed bytes against the original file checksums. Assemble the files into the original relative layout with the pinned predecessors and build inputs before executing the archived build scripts. The package includes portable source and numeric evidence. The original build also depends on its pinned toolchain image, object files and predecessor builds.
