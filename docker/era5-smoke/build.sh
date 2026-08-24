#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONTEXT_SUFFIX="$(
  python3 "$SCRIPT_DIR/../../scripts/build_era5_smoke_runtime.py" |
    python3 -c 'import json, sys; print(json.load(sys.stdin)["context_sha256"][:12])'
)"
IMAGE_TAG="${IMAGE_TAG:-ghcr.io/tobiasedman/imint-era5-smoke:20260821-${CONTEXT_SUFFIX}}"
docker build --platform linux/amd64 -t "$IMAGE_TAG" \
  -f "$SCRIPT_DIR/Dockerfile" "$SCRIPT_DIR"
if [ "${PUSH:-0}" = 1 ]; then
  docker push "$IMAGE_TAG"
fi
docker buildx imagetools inspect "$IMAGE_TAG"
