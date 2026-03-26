#!/bin/bash
# download_training_data.sh — Download all raw data for crop classification training
#
# Sources:
#   LUCAS Copernicus 2018: Figshare (CC BY 4.0)
#   LUCAS Eurostat 2022:   Eurostat (open data)
#   LPIS 2022-2024:        Jordbruksverket WFS (CC BY 4.0)
#   LPIS 2025:             Jordbruksverket INSPIRE Atom (CC BY 4.0)
#
# Usage:
#   bash scripts/download_training_data.sh
#
# Total download: ~5 GB
# Time: ~15-30 min depending on connection

set -e

DATA_DIR="$(cd "$(dirname "$0")/.." && pwd)/data"
LUCAS_DIR="$DATA_DIR/lucas"
LPIS_DIR="$DATA_DIR/lpis"

echo "=== ImintEngine — Download Training Data ==="
echo "Output: $DATA_DIR"
echo ""

mkdir -p "$LUCAS_DIR" "$LPIS_DIR"

# ── LUCAS Copernicus 2018 ────────────────────────────────────────────────
echo "[1/6] LUCAS Copernicus 2018 (Figshare, 12 MB)..."
if [ -f "$LUCAS_DIR/LUCAS_2018_Copernicus_attributes.csv" ]; then
    echo "  SKIP (already exists)"
else
    curl -L -o "$LUCAS_DIR/lucas_2018_copernicus.zip" \
        "https://ndownloader.figshare.com/files/26404922"
    unzip -o "$LUCAS_DIR/lucas_2018_copernicus.zip" \
        LUCAS_2018_Copernicus_attributes.csv -d "$LUCAS_DIR"
    echo "  OK: $(wc -l < "$LUCAS_DIR/LUCAS_2018_Copernicus_attributes.csv") rows"
fi

# ── LUCAS Eurostat 2022 ──────────────────────────────────────────────────
echo "[2/6] LUCAS Eurostat 2022 (Eurostat, 28 MB)..."
if [ -f "$LUCAS_DIR/EU_LUCAS_2022.csv" ]; then
    echo "  SKIP (already exists)"
else
    curl -L -o "$LUCAS_DIR/lucas_2022_eu.zip" \
        "https://ec.europa.eu/eurostat/documents/205002/17561401/EU_LUCAS_2022.zip"
    unzip -o "$LUCAS_DIR/lucas_2022_eu.zip" EU_LUCAS_2022.csv -d "$LUCAS_DIR"
    echo "  OK: $(wc -l < "$LUCAS_DIR/EU_LUCAS_2022.csv") rows"
fi

# ── LPIS 2022 (Jordbruksverket WFS) ─────────────────────────────────────
echo "[3/6] LPIS 2022 — Jordbruksskiften (SJV WFS, ~300 MB)..."
if [ -f "$LPIS_DIR/jordbruksskiften_2022.zip" ] && unzip -t "$LPIS_DIR/jordbruksskiften_2022.zip" > /dev/null 2>&1; then
    echo "  SKIP (already exists and valid)"
else
    curl -L -o "$LPIS_DIR/jordbruksskiften_2022.zip" \
        "http://epub.sjv.se/inspire/inspire/wfs?service=WFS&version=2.0.0&request=GetFeature&typeName=inspire:arslager_skifte&outputFormat=shape-zip&CQL_FILTER=arslager%3D2022"
    echo "  OK: $(ls -lh "$LPIS_DIR/jordbruksskiften_2022.zip" | awk '{print $5}')"
fi

# ── LPIS 2023 ────────────────────────────────────────────────────────────
echo "[4/6] LPIS 2023 — Jordbruksskiften (SJV WFS, ~300 MB)..."
if [ -f "$LPIS_DIR/jordbruksskiften_2023.zip" ] && unzip -t "$LPIS_DIR/jordbruksskiften_2023.zip" > /dev/null 2>&1; then
    echo "  SKIP (already exists and valid)"
else
    curl -L -o "$LPIS_DIR/jordbruksskiften_2023.zip" \
        "http://epub.sjv.se/inspire/inspire/wfs?service=WFS&version=2.0.0&request=GetFeature&typeName=inspire:arslager_skifte&outputFormat=shape-zip&CQL_FILTER=arslager%3D2023"
    echo "  OK: $(ls -lh "$LPIS_DIR/jordbruksskiften_2023.zip" | awk '{print $5}')"
fi

# ── LPIS 2024 ────────────────────────────────────────────────────────────
echo "[5/6] LPIS 2024 — Jordbruksskiften (SJV WFS, ~300 MB)..."
if [ -f "$LPIS_DIR/jordbruksskiften_2024.zip" ] && unzip -t "$LPIS_DIR/jordbruksskiften_2024.zip" > /dev/null 2>&1; then
    echo "  SKIP (already exists and valid)"
else
    curl -L -o "$LPIS_DIR/jordbruksskiften_2024.zip" \
        "http://epub.sjv.se/inspire/inspire/wfs?service=WFS&version=2.0.0&request=GetFeature&typeName=inspire:arslager_skifte&outputFormat=shape-zip&CQL_FILTER=arslager%3D2024"
    echo "  OK: $(ls -lh "$LPIS_DIR/jordbruksskiften_2024.zip" | awk '{print $5}')"
fi

# ── LPIS 2025 (INSPIRE Atom) ────────────────────────────────────────────
echo "[6/6] LPIS 2025 — Jordbruksskiften (INSPIRE Atom, 3.7 GB GML)..."
EXPECTED_SIZE=3700000000  # bytes from Atom feed length attribute
if [ -f "$LPIS_DIR/jordbruksskiften_2025.gml.zip" ]; then
    ACTUAL_SIZE=$(stat -f%z "$LPIS_DIR/jordbruksskiften_2025.gml.zip" 2>/dev/null || stat -c%s "$LPIS_DIR/jordbruksskiften_2025.gml.zip" 2>/dev/null || echo 0)
    if [ "$ACTUAL_SIZE" -ge "$EXPECTED_SIZE" ]; then
        echo "  SKIP (already exists, $ACTUAL_SIZE bytes >= expected $EXPECTED_SIZE)"
    else
        echo "  Incomplete ($ACTUAL_SIZE / $EXPECTED_SIZE bytes), re-downloading..."
        curl -L -C - -o "$LPIS_DIR/jordbruksskiften_2025.gml.zip" \
            "https://cdn.jordbruksverket.se/inspire/atom/jordbruksskiften/20250326/jordbruksskiften.gml.zip"
        echo "  OK: $(ls -lh "$LPIS_DIR/jordbruksskiften_2025.gml.zip" | awk '{print $5}')"
    fi
else
    curl -L -o "$LPIS_DIR/jordbruksskiften_2025.gml.zip" \
        "https://cdn.jordbruksverket.se/inspire/atom/jordbruksskiften/20250326/jordbruksskiften.gml.zip"
    echo "  OK: $(ls -lh "$LPIS_DIR/jordbruksskiften_2025.gml.zip" | awk '{print $5}')"
fi

# ── Verify ───────────────────────────────────────────────────────────────
echo ""
echo "=== Verification ==="
echo "LUCAS:"
for f in "$LUCAS_DIR"/*.csv; do
    echo "  $(basename "$f"): $(wc -l < "$f") rows"
done
echo "LPIS:"
for f in "$LPIS_DIR"/jordbruksskiften_*.zip "$LPIS_DIR"/jordbruksskiften_*.gml.zip; do
    [ -f "$f" ] && echo "  $(basename "$f"): $(ls -lh "$f" | awk '{print $5}')"
done

echo ""
echo "=== Next steps ==="
echo "1. Build balanced dataset:"
echo "   .venv/bin/python scripts/build_crop_dataset_v3.py \\"
echo "       --lucas-csv data/lucas/LUCAS_2018_Copernicus_attributes.csv \\"
echo "                   data/lucas/EU_LUCAS_2022.csv \\"
echo "       --lpis-dir data/lpis \\"
echo "       --output data/lucas/balanced_points_v3.json"
echo ""
echo "2. Fetch S2 tiles:"
echo "   .venv/bin/python scripts/fetch_lucas_tiles.py \\"
echo "       --balanced-json data/lucas/balanced_points_v3.json \\"
echo "       --output-dir data/crop_tiles"
echo ""
echo "3. Train on H100:"
echo "   colony submit --spec config/crop_training_h100.json"
