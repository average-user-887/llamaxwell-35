# NC3 build (CPU only, no GPU, no docker used)
- src/: copy of E/G1b/src (rsync, no .git/build), 143 MB. E/G1b/src untouched. Added src/tools/n4-dump/{n4_dump.cpp,CMakeLists.txt} and `add_subdirectory(n4-dump)` in src/tools/CMakeLists.txt (CMake target n4-dump; not configured/built via cmake, g++ route used).
- n4_dump (host g++ 13.3, against G1b build libs, links OK, not run):
```
E=<LAB>/occamy35-day-20261112/evidence ; cd <LAB>/occamy35-night-20261113/NC3
g++ -O2 -std=c++17 n4_dump.cpp -Isrc/include -Isrc/ggml/include -L$E/G1b/build/bin -lllama -lggml -lggml-base -Wl,-rpath,$E/G1b/build/bin -o build/n4_dump
```
- n4_drive: pre-existing build/n4_drive (1.1 MB, 00:27, linked per N4_DESIGN.md section 3 nvcc command in local/m6000-build-tools). Needs at runtime LD_LIBRARY_PATH=<integration-v4/src/build/bin> (libggml-cuda.so, libggml-base) plus libcublas.so.12 (CUDA runtime). Not rebuilt.
