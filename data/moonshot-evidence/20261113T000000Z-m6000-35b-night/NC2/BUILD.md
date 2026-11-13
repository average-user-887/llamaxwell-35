# NC2 build: fixed g1c_rows (CPU only, no GPU, no docker needed)
Host g++ 13.3.0 (same GCC as the E/G1c binary; E/G1c has no stored build command, flags reconstructed from its NEEDED/RUNPATH).
Fix vs E/G1c/g1c_rows.cpp (diff): bounds checks "P end %d beyond %zu tokens" (rc 16) and "R pos %d beyond %zu tokens" (rc 17).
```
M=<LAB> ; E=<LAB>/occamy35-day-20261112/evidence ; A=$M/occamy35-night-20261111/app
cd <LAB>/occamy35-night-20261113/NC2 && g++ -O2 -std=c++17 g1c_rows.cpp -I$E/G1b/src/include -I$E/G1b/src/ggml/include -L$A -lllama -lggml -lggml-base -Wl,-rpath,$A -o bin/g1c_rows
```
Result: bin/g1c_rows. NEEDED: libllama.so.0, libggml.so.0, libstdc++, libgcc_s, libc; RUNPATH = night-20261111/app. Identical to E/G1c/g1c_rows (libggml-base.so.0 shows "not found" in ldd for both; resolved at runtime by libggml; same as original). `strings` shows both new error messages. Not run.
