#!/usr/bin/env bash
set -Eeuo pipefail
umask 022
export LC_ALL=C
unset TAR_OPTIONS GZIP

if [[ $# -ne 4 ]]; then
    printf 'Usage: %s R3_PACKAGE_DIR MEASURED_LIBLLAMA MEASURED_LIBGGML_CUDA OUTPUT_DIR\n' "${0##*/}" >&2
    exit 2
fi

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
base="$(realpath -e -- "$1")"
host_lib="$(realpath -e -- "$2")"
cuda_lib="$(realpath -e -- "$3")"
mkdir -p -- "$4"
out="$(realpath -e -- "$4")"
name=llamaxwell-35-r3-v5-linux-x86_64-sm52-cuda12

check_hash() {
    local result
    result="$(sha256sum < "$2")"
    if [[ ${result%% *} != "$1" ]]; then
        printf 'SHA-256 mismatch: %s\n' "$2" >&2
        return 1
    fi
}

# These pins select the shipped R3 inputs and the libraries measured on the M6000.
check_hash 7335ac40f1800dda39cfe19e7709a1445086a29828aaf91b10986f68e9df8872 "$base/SHA256SUMS"
(cd -- "$base" && sha256sum --check --strict --quiet SHA256SUMS)
check_hash a3187e5e8ce7da37cdef5b4b5d0d2fabc23c737b9f8a33e93a8e028bc3bad535 "$host_lib"
check_hash c84c5ba94a298c7aa2d8a7de233cd8c63469d0656ea3559795911a59f41ca98d "$cuda_lib"

stage="$(mktemp -d -- "$out/.pack-r3-v5.XXXXXXXX")"
trap 'rm -rf -- "$stage"' EXIT
package="$stage/$name"
mkdir -p -- "$package/bin"

while read -r _ path; do
    [[ $path == bin/* ]] || continue
    if [[ -L "$base/$path" ]]; then
        target="$(readlink -- "$base/$path")"
        if [[ $target == */* || $target == . || $target == .. ]]; then
            printf 'Unexpected symlink in base package: %s\n' "$path" >&2
            exit 1
        fi
    fi
    cp -P -- "$base/$path" "$package/$path"
done < "$base/SHA256SUMS"

for lib in libllama.so.0.6.0 libggml-cuda.so; do
    if [[ -L "$package/bin/$lib" ]]; then
        printf 'Expected a regular versioned library: %s\n' "$lib" >&2
        exit 1
    fi
done
install -m 0755 -- "$host_lib" "$package/bin/libllama.so.0.6.0"
install -m 0755 -- "$cuda_lib" "$package/bin/libggml-cuda.so"
find "$package/bin" -type f -exec chmod 0755 {} +
install -m 0755 -- "$repo_dir"/packaging/{install.sh,llamaxwell-serve.sh} "$package/"
install -m 0644 -- "$repo_dir"/{README.md,LICENSE} "$repo_dir/docs/DEPLOY.md" "$package/"

(
    cd -- "$package"
    find bin install.sh llamaxwell-serve.sh README.md DEPLOY.md LICENSE \( -type f -o -type l \) -print0 |
        sort -z | xargs -0 sha256sum > SHA256SUMS
    sha256sum --check --strict --quiet SHA256SUMS
)

tar --sort=name --format=gnu --mtime=@0 --owner=0 --group=0 --numeric-owner \
    -C "$stage" -cf - "$name" | gzip -n -6 > "$stage/$name.tar.gz"
(cd -- "$stage" && sha256sum "$name.tar.gz" > SHA256SUMS)
mv -- "$stage/$name.tar.gz" "$stage/SHA256SUMS" "$out/"
printf '%s\n' "$out/$name.tar.gz"
