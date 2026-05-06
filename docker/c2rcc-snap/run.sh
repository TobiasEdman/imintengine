#!/usr/bin/env bash
# Kör ESA SNAP c2rcc.msi på en Sentinel-2 L1C SAFE-arkiv via Docker.
#
# Användning:
#   ./docker/c2rcc-snap/run.sh \
#       <SAFE_PATH> <OUTPUT_DIM> \
#       --west <W> --south <S> --east <E> --north <N>
#
# Exempel:
#   ./docker/c2rcc-snap/run.sh \
#       demos/lilla_karlso_birds/cache_l1c/2025-05-12/S2A_MSIL1C_*.SAFE \
#       outputs/c2rcc_runs_lilla_karlso/2025-05-12.dim \
#       --west 17.91 --south 57.21 --east 18.21 --north 57.41
#
# Beroenden:
#   - Docker daemon
#   - Image `imint-snap-c2rcc:latest` (bygg med `docker build` på
#     ./docker/c2rcc-snap/ om den saknas)
#
# Output:
#   - <OUTPUT_DIM>      BEAM-DIMAP header
#   - <OUTPUT_DIM>.data/ ENVI-band per IOP/kd/rhow + c2rcc_flags
set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <SAFE_PATH> <OUTPUT_DIM> --west W --south S --east E --north N" >&2
    exit 1
fi

SAFE_PATH=$(realpath "$1")
OUTPUT_DIM=$(realpath "$2")
shift 2

WEST="" SOUTH="" EAST="" NORTH=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --west)  WEST="$2";  shift 2 ;;
        --south) SOUTH="$2"; shift 2 ;;
        --east)  EAST="$2";  shift 2 ;;
        --north) NORTH="$2"; shift 2 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

if [[ -z "$WEST$SOUTH$EAST$NORTH" ]]; then
    echo "Need --west --south --east --north (WGS84 decimal degrees)" >&2
    exit 1
fi

# Bygg WKT-polygon för SNAP Subset (CCW, sluten).
GEO_REGION="POLYGON((${WEST} ${SOUTH}, ${EAST} ${SOUTH}, ${EAST} ${NORTH}, ${WEST} ${NORTH}, ${WEST} ${SOUTH}))"

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
GRAPH_XML="${SCRIPT_DIR}/c2rcc_msi_graph.xml"

if [[ ! -f "$GRAPH_XML" ]]; then
    echo "Graph not found: $GRAPH_XML" >&2
    exit 1
fi

mkdir -p "$(dirname "$OUTPUT_DIM")"

# Map SAFE-katalogen + output + graph in i container.
SAFE_DIR=$(dirname "$SAFE_PATH")
SAFE_NAME=$(basename "$SAFE_PATH")
OUT_DIR=$(dirname "$OUTPUT_DIM")
OUT_NAME=$(basename "$OUTPUT_DIM")

echo "[c2rcc] SAFE:   $SAFE_PATH"
echo "[c2rcc] Output: $OUTPUT_DIM"
echo "[c2rcc] AOI:    W=$WEST S=$SOUTH E=$EAST N=$NORTH"

docker run --rm \
    -v "$SAFE_DIR:/in:ro" \
    -v "$OUT_DIR:/out" \
    -v "$GRAPH_XML:/graph.xml:ro" \
    imint-snap-c2rcc:latest \
    /usr/local/snap/bin/gpt /graph.xml \
        -PinputSafe="/in/$SAFE_NAME" \
        -PgeoRegion="$GEO_REGION" \
        -PoutputDim="/out/$OUT_NAME" \
        -e

echo "[c2rcc] Done — bands available:"
ls "${OUTPUT_DIM%.dim}.data/" 2>/dev/null | grep -E "\.(img|hdr)$" | head -10
