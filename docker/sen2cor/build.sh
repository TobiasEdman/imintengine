#!/usr/bin/env bash
# Build the sen2cor docker image and capture the resulting digest.
#
# After the first successful build, copy the printed digest into
# Dockerfile's FROM line (if Ubuntu base) and into MANIFEST.json so
# pipeline-side k8s manifests can pin to it.
#
# Usage:
#   cd docker/sen2cor && bash build.sh                # local build
#   IMAGE_TAG=ghcr.io/.../sen2cor:2.12.04 bash build.sh
#   PUSH=1 bash build.sh                              # also push
set -euo pipefail

IMAGE_TAG="${IMAGE_TAG:-imintengine/sen2cor:2.12.04}"
PUSH="${PUSH:-0}"

echo "=== Building ${IMAGE_TAG} ==="
docker build -t "${IMAGE_TAG}" .

if [ "$PUSH" = "1" ]; then
    echo "=== Pushing ${IMAGE_TAG} ==="
    docker push "${IMAGE_TAG}"
fi

# Capture the digest so the consumer manifests can pin to it.
digest=$(docker inspect --format='{{index .RepoDigests 0}}' "${IMAGE_TAG}" \
         2>/dev/null || echo "(local image, no digest)")
echo ""
echo "=== Result ==="
echo "tag:    ${IMAGE_TAG}"
echo "digest: ${digest}"
echo ""
echo "Capture the digest into:"
echo "  • docker/sen2cor/MANIFEST.json"
echo "  • k8s manifests using this image"
