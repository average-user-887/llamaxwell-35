#!/usr/bin/env bash
set -euo pipefail

package_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source_dir=$(cd -- "${1:?pass a clean detached worktree at e4fdc23}" && pwd)
revision=${2:-r2}
case "$revision" in
    base) expected=2d8daf5281b031030d43db3c2be3db0b09216811fe4edad6a27042e60aec55bf ;;
    r1) expected=8e440e1fe0bea664be7e8d61249f375f055deb3088c1bbd608e79e72c5b29711 ;;
    r2) expected=4c5fc2f721d2bcac4cd3e88c9a0b7946567cd0a65c9d8ff17844356307c00d8f ;;
    *) printf 'Use base, r1 or r2.\n' >&2; exit 2 ;;
esac
cd "$source_dir"
public_base=e4fdc23555f6a9103ea03b854ee5c70c6812517c
if [[ $(git rev-parse HEAD) != "$public_base" || -n $(git status --porcelain) ]]; then
    printf 'Use a clean worktree at %s.\n' "$public_base" >&2
    exit 2
fi
if git symbolic-ref -q HEAD >/dev/null; then
    printf 'Use a detached worktree.\n' >&2
    exit 2
fi

scratch=$(mktemp -d)
trap 'rm -rf -- "$scratch"' EXIT
mv3=evidence-mv3-20261117/patch
git mailsplit -o"$scratch" \
    "$mv3/01-base-mrow-phase-gdnT2.patch" \
    "$mv3/02-draft-merge-reeval.patch" \
    "$mv3/03-fa2.patch" > /dev/null
git apply --index "$package_dir/patch/00-public-to-mv3-base.patch"
for mail in "$scratch"/[0-9][0-9][0-9][0-9]; do
    git mailinfo "$scratch/message" "$scratch/diff" < "$mail" > /dev/null
    git apply --index "$scratch/diff"
done
if [[ "$revision" != base ]]; then
    git apply --index "$package_dir/patch/sparse-$revision.patch"
fi

# The public evidence and historical qgate scripts sit outside the compilation source.
actual=$(git ls-files --stage -- . ':(exclude)data' ':(exclude)night-20261111' \
    ':(exclude)evidence*' ':(exclude)scripts/qgate' | sha256sum)
if [[ ${actual%% *} != "$expected" ]]; then
    printf 'Unexpected %s source digest %s\n' "$revision" "${actual%% *}" >&2
    exit 1
fi
printf '%s source verified (%s).\n' "$revision" "$expected"
