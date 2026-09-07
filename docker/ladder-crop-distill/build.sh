#!/bin/sh
set -eu

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo_root=$(CDPATH= cd -- "$script_dir/../.." && pwd)
source_git_sha=${1:-$(git -C "$repo_root" rev-parse HEAD)}

case "$source_git_sha" in
    *[!0-9a-f]*)
        echo "SOURCE_GIT_SHA must be exactly 40 lowercase hexadecimal characters" >&2
        exit 2
        ;;
esac
if [ "${#source_git_sha}" -ne 40 ]; then
    echo "SOURCE_GIT_SHA must be exactly 40 lowercase hexadecimal characters" >&2
    exit 2
fi

head_git_sha=$(git -C "$repo_root" rev-parse HEAD)
if [ "$source_git_sha" != "$head_git_sha" ]; then
    echo "SOURCE_GIT_SHA must equal the checked-out HEAD ($head_git_sha)" >&2
    exit 2
fi
if [ -n "$(git -C "$repo_root" status --porcelain --untracked-files=all)" ]; then
    echo "refusing a mixed-source image: commit or remove worktree changes first" >&2
    exit 2
fi

image_repository=${CROP_DISTILL_IMAGE_REPOSITORY:-ghcr.io/tobiasedman/imint-ladder-crop-distill}
image_ref="${image_repository}:sha-${source_git_sha}"

docker buildx build \
    --platform linux/amd64 \
    --load \
    --file "$script_dir/Dockerfile" \
    --build-arg "SOURCE_GIT_SHA=$source_git_sha" \
    --tag "$image_ref" \
    "$repo_root"

printf '%s\n' "$image_ref"
