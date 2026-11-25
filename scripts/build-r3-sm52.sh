#!/usr/bin/env bash
set -Eeuo pipefail
umask 022

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"

IMAGE="${LLAMAXWELL_BUILD_IMAGE:-local/m6000-build-tools:20261013}"
BUILD_DIR="${LLAMAXWELL_BUILD_DIR:-$REPO_DIR/build-r3-sm52}"
OUT_DIR="${LLAMAXWELL_OUT_DIR:-$REPO_DIR/dist-r3-sm52}"
JOBS="${LLAMAXWELL_JOBS:-10}"
CUDA_ARCH="${LLAMAXWELL_CUDA_ARCH:-52-real}"
SLIM="${LLAMAXWELL_OCCAMY_SLIM:-ON}"

usage() {
    cat <<EOF
Usage: $(basename "$0") [--out DIR] [--build DIR] [--jobs N] [--arch 52-real] [--slim ON|OFF] [--image IMG]
Build the Maxwell sm_52 R3 llama-server and backend DSOs inside a clean /src container mount.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --out) OUT_DIR="$(realpath -m "$2")"; shift 2 ;;
        --build) BUILD_DIR="$(realpath -m "$2")"; shift 2 ;;
        --jobs) JOBS="$2"; shift 2 ;;
        --arch) CUDA_ARCH="$2"; shift 2 ;;
        --slim) SLIM="$2"; shift 2 ;;
        --image) IMAGE="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown arg: $1" >&2; usage; exit 2 ;;
    esac
done

mkdir -p "$BUILD_DIR" "$OUT_DIR"

INNER_SCRIPT="$(mktemp)"
trap 'rm -f "$INNER_SCRIPT"' EXIT

cat > "$INNER_SCRIPT" <<EOF
#!/usr/bin/env bash
set -Eeuo pipefail

export CC=/usr/bin/gcc-14
export CXX=/usr/bin/g++-14
export CUDAHOSTCXX=/usr/bin/g++-14

EXTRA_CMAKE=()
if [[ "${SLIM}" == "ON" ]]; then
    EXTRA_CMAKE+=(-DGGML_CUDA_OCCAMY_SLIM=ON -DGGML_CUDA_FA_QUANTS="f16-f16")
else
    EXTRA_CMAKE+=(-DGGML_CUDA_OCCAMY_SLIM=OFF)
fi

cmake -S /src -B /build \\
    -DCMAKE_BUILD_TYPE=Release \\
    -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH}" \\
    -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON \\
    -DCMAKE_INSTALL_RPATH='\$ORIGIN' \\
    -DGGML_NATIVE=OFF \\
    -DGGML_CUDA=ON \\
    -DGGML_BACKEND_DL=ON \\
    -DGGML_CPU_ALL_VARIANTS=ON \\
    -DLLAMA_BUILD_TESTS=ON \\
    -DGGML_CUDA_GRAPHS=ON \\
    "\${EXTRA_CMAKE[@]}"

cmake --build /build --config Release -j "${JOBS}" --target llama-server ggml-cuda ggml ggml-base test-arg-parser

rm -rf /out/*
cp -a /build/bin/llama-server /out/
cp -a /build/bin/lib*.so* /out/
cp -a /build/bin/test-arg-parser /out/

mkdir -p /tmp/cuda-stubs
ln -sf /usr/local/cuda/targets/x86_64-linux/lib/stubs/libcuda.so /tmp/cuda-stubs/libcuda.so.1
for bin in /out/llama-server /out/libggml-cuda.so /out/libllama.so.0* /out/libllama-common.so.0* /out/libllama-server-impl.so; do
    [[ -e "\$bin" && ! -L "\$bin" ]] || continue
    out_ldd=\$(LD_LIBRARY_PATH=/out:/tmp/cuda-stubs ldd -r "\$bin" 2>&1)
    if echo "\$out_ldd" | grep -qE 'undefined symbol|not found'; then
        echo "FAIL ldd -r \$bin:" >&2
        echo "\$out_ldd" >&2
        exit 1
    fi
done

if [[ "${CUDA_ARCH}" == "52-real" ]]; then
    archs=\$(/usr/local/cuda/bin/cuobjdump --list-elf /out/libggml-cuda.so | grep -oE 'sm_[0-9]+' | sort -u)
    if [[ "\$archs" != "sm_52" ]]; then
        echo "FAIL: expected sm_52 only, got: \$archs" >&2
        exit 1
    fi
fi

echo "Build and verification complete in /out"
EOF
chmod +x "$INNER_SCRIPT"

docker run --rm --pull never --runtime runc --network none \
    -e NVIDIA_VISIBLE_DEVICES=void -e TMPDIR=/tmp --tmpfs /tmp:rw,size=512m \
    --user "$(id -u):$(id -g)" \
    -v "$REPO_DIR:/src:ro" \
    -v "$BUILD_DIR:/build" \
    -v "$OUT_DIR:/out" \
    -v "$INNER_SCRIPT:/tmp/build-inner.sh:ro" \
    --entrypoint /bin/bash "$IMAGE" /tmp/build-inner.sh
