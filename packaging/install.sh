#!/usr/bin/env bash
set -Eeuo pipefail

PREFIX="/opt/llamaxwell-35"
VERIFY=1

usage() {
    cat <<EOF
Usage: $(basename "$0") [--prefix DIR] [--no-verify]
Install the LlamaMaxwell-35 R3 v5 (sm_52) binaries and launcher into PREFIX (default: /opt/llamaxwell-35).
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --prefix) PREFIX="$2"; shift 2 ;;
        --no-verify) VERIFY=0; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown arg: $1" >&2; usage; exit 2 ;;
    esac
done

PKG_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if [[ "$VERIFY" == "1" && -f "$PKG_DIR/SHA256SUMS" ]]; then
    echo "Verifying package checksums..."
    (cd "$PKG_DIR" && sha256sum -c SHA256SUMS)
fi

echo "Installing to $PREFIX..."
mkdir -p "$PREFIX/bin"
cp -a "$PKG_DIR"/bin/* "$PREFIX/bin/"
cp -a "$PKG_DIR/llamaxwell-serve.sh" "$PREFIX/"
chmod +x "$PREFIX/llamaxwell-serve.sh" "$PREFIX/bin/llama-server"

for doc in README.md DEPLOY.md LICENSE SHA256SUMS; do
    [[ -f "$PKG_DIR/$doc" ]] && cp -a "$PKG_DIR/$doc" "$PREFIX/"
done

echo ""
echo "Installed LlamaMaxwell-35 R3 v5 to: $PREFIX"
echo "Run with:"
echo "  $PREFIX/llamaxwell-serve.sh --model /path/to/occamy-1.0-Q4_0-headq40b-mtp.gguf"
echo ""
echo "See $PREFIX/DEPLOY.md for GPU requirements, model format notes, and mode options."
