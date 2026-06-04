#!/usr/bin/env bash
# Collaborator-side downloader for the 256-px unified dataset
# (8290 .npz tiles, ~168 GB) served temporarily at ICE.
#
# The server (k8s/dataset-256-download.yaml) exposes a public nginx
# autoindex of /unified_v2/ over TLS. This mirrors the whole directory,
# resuming any partial/failed files (-c) and retrying.
#
# Usage:
#   ./download_unified_256.sh [DEST_DIR]
#
# DEST_DIR defaults to ./unified_v2. Re-run to resume — already-complete
# files are skipped (timestamping), partial files continue.
set -euo pipefail

BASE_URL="https://dataset-256.icedc.se/unified_v2/"
META_URL="https://dataset-256.icedc.se/metadata.json"
DEST="${1:-./unified_v2}"

mkdir -p "$DEST"
echo "Fetching machine-readable schema -> $DEST/metadata.json"
wget -q -O "$DEST/metadata.json" "$META_URL" || echo "  (metadata fetch skipped)"

echo "Mirroring $BASE_URL -> $DEST"
echo "(~168 GB / 8290 tiles — this will take a while; safe to re-run to resume)"

# -r recursive, -np stay under unified_v2/, -nH/--cut-dirs strip the
# leading path component, -c resume, -N skip unchanged, robots off so
# the autoindex is walked, retries for flaky transfers.
wget \
  --recursive --no-parent --no-host-directories --cut-dirs=1 \
  --reject "index.html*" \
  --continue --timestamping \
  --tries=10 --waitretry=15 --read-timeout=120 \
  -e robots=off \
  --directory-prefix="$DEST" \
  "$BASE_URL"

COUNT=$(find "$DEST" -name '*.npz' | wc -l | tr -d ' ')
echo "Done. $COUNT .npz files in $DEST (expected 8290)."
