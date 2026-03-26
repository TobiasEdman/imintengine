"""
imint/training/crop_schema.py — Swedish crop class schema and LUCAS mapping

Maps LUCAS Copernicus 2018 land cover codes to an 8-class Swedish crop
type schema for fine-tuning Prithvi on Swedish agricultural land.

LUCAS crop classes (B-codes) are mapped to:
    0: other          Non-agricultural land
    1: wheat          B11 (common wheat), B13 (durum wheat)
    2: barley         B14 (barley)
    3: oats           B15 (oats)
    4: rapeseed       B32 (rape/turnip rape)
    5: ley_grass      B51-B55 (fodder crops, grassland, clover, lucerne, mixed)
    6: potato         B21 (potato)
    7: other_crop     B12, B16, B22-B45 (rye, triticale, sugar beet, etc.)

Data source: LUCAS Copernicus 2018 (Figshare doi:10.6084/m9.figshare.12382667)
Reference: Jordbruksverket (Swedish Board of Agriculture) crop statistics
"""
from __future__ import annotations

# ── Swedish crop class schema ─────────────────────────────────────────────

NUM_CLASSES = 8

CLASS_NAMES = {
    0: "other",
    1: "wheat",
    2: "barley",
    3: "oats",
    4: "rapeseed",
    5: "ley_grass",
    6: "potato",
    7: "other_crop",
}

CLASS_COLORS = {
    0: (0.50, 0.50, 0.50),  # grey
    1: (0.93, 0.79, 0.13),  # gold
    2: (0.85, 0.65, 0.13),  # dark gold
    3: (0.96, 0.87, 0.70),  # wheat
    4: (0.80, 0.90, 0.10),  # yellow-green
    5: (0.20, 0.80, 0.20),  # green
    6: (0.60, 0.40, 0.20),  # brown
    7: (0.70, 0.70, 0.50),  # khaki
}

# ── LUCAS land cover → Swedish crop class ─────────────────────────────────

# LUCAS level-3 crop codes (B-series) → class index
# Full reference: https://ec.europa.eu/eurostat/statistics-explained/index.php/LUCAS
LUCAS_TO_CROP = {
    # Wheat
    "B11": 1,  # Common wheat
    "B13": 1,  # Durum wheat

    # Barley
    "B14": 2,  # Barley

    # Oats
    "B15": 3,  # Oats

    # Rapeseed
    "B32": 4,  # Rape and turnip rape
    "B36": 4,  # Sunflower (rare in SE, but same oil crop class)

    # Ley / grass / fodder
    "B51": 5,  # Fodder crops (e.g., maize for silage)
    "B52": 5,  # Grassland / permanent pasture
    "B53": 5,  # Clover
    "B54": 5,  # Lucerne (alfalfa)
    "B55": 5,  # Mixed ley

    # Potato
    "B21": 6,  # Potato

    # Other crops
    "B12": 7,  # Rye
    "B16": 7,  # Triticale
    "B17": 7,  # Mixed cereals
    "B18": 7,  # Other cereals
    "B22": 7,  # Sugar beet
    "B23": 7,  # Other root crops
    "B31": 7,  # Peas
    "B33": 7,  # Soya
    "B34": 7,  # Flax/linseed
    "B35": 7,  # Other oil crops
    "B37": 7,  # Fibre crops
    "B41": 7,  # Vegetables
    "B42": 7,  # Flowers
    "B43": 7,  # Strawberry
    "B44": 7,  # Other industrial crops
    "B45": 7,  # Tobacco (unlikely in SE)
}

# All LUCAS codes that are agricultural (B-series)
AGRICULTURAL_CODES = set(LUCAS_TO_CROP.keys())


def lucas_code_to_class(lc_code: str) -> int:
    """Map a LUCAS land cover code to the Swedish crop class index.

    Args:
        lc_code: LUCAS level-3 code, e.g. "B11", "B14", "A11".

    Returns:
        Class index (0-7). Returns 0 ("other") for non-crop codes.
    """
    return LUCAS_TO_CROP.get(lc_code, 0)


def is_agricultural(lc_code: str) -> bool:
    """Check if a LUCAS land cover code is agricultural."""
    return lc_code in AGRICULTURAL_CODES


# ── LUCAS data loading ────────────────────────────────────────────────────

LUCAS_COPERNICUS_2018_URL = (
    "https://figshare.com/ndownloader/articles/12382667/versions/4"
)
LUCAS_COPERNICUS_2022_URL = (
    "https://figshare.com/ndownloader/articles/24090553/versions/1"
)
LUCAS_HARMONISED_URL = (
    "https://figshare.com/ndownloader/articles/12841202/versions/2"
)

# Supported survey years with Copernicus module (EO-ready polygons)
LUCAS_SUPPORTED_YEARS = [2018, 2022]


def load_lucas_sweden(
    csv_path: str | list[str],
    *,
    min_year: int = 2018,
    crop_only: bool = True,
) -> list[dict]:
    """Load LUCAS points filtered for Sweden.

    Reads one or more LUCAS Copernicus CSV files (2018 and/or 2022) and
    filters for Swedish points (NUTS0 == "SE"). Points appearing in both
    surveys are deduplicated (2022 takes precedence).

    Args:
        csv_path: Path to LUCAS CSV, or list of paths (e.g. 2018 + 2022).
        min_year: Minimum survey year to include (default 2018).
        crop_only: If True, only return agricultural (B-series) points.

    Returns:
        List of dicts with keys: point_id, lat, lon, lc_code, crop_class,
        crop_name, year, nuts0. Deduplicated by point_id (latest year wins).
    """
    import csv

    # Normalize to list of paths
    if isinstance(csv_path, str):
        csv_paths = [csv_path]
    else:
        csv_paths = list(csv_path)

    # Collect all points, keyed by point_id for deduplication
    seen: dict[str, dict] = {}

    for path in csv_paths:
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            # Detect column names (LUCAS uses varying headers across years)
            # 2018 Copernicus: POINT_ID, GPS_LAT, GPS_LONG, LC1, NUTS0, SURVEY_YEAR
            # 2022 Eurostat:   POINT_ID, POINT_LAT, POINT_LONG, SURVEY_LC1,
            #                  POINT_NUTS0, SURVEY_DATE
            fieldnames = reader.fieldnames or []
            fn_lower = {fn.lower(): fn for fn in fieldnames}

            col_id = fn_lower.get("point_id") or fn_lower.get("id", "POINT_ID")
            col_lat = (
                fn_lower.get("gps_lat")
                or fn_lower.get("point_lat")
                or fn_lower.get("th_lat")
                or fn_lower.get("y", "GPS_LAT")
            )
            col_lon = (
                fn_lower.get("gps_long")
                or fn_lower.get("point_long")
                or fn_lower.get("th_long")
                or fn_lower.get("x", "GPS_LONG")
            )
            col_lc = (
                fn_lower.get("lc1")
                or fn_lower.get("survey_lc1")
                or fn_lower.get("lc1_label", "LC1")
            )
            col_nuts = (
                fn_lower.get("nuts0")
                or fn_lower.get("point_nuts0", "NUTS0")
            )
            col_year = (
                fn_lower.get("survey_year")
                or fn_lower.get("survey_date")
                or fn_lower.get("year", "SURVEY_YEAR")
            )

            for row in reader:
                nuts0 = row.get(col_nuts, "")
                if nuts0.upper() != "SE":
                    continue

                try:
                    year_raw = row.get(col_year, "0").strip().strip('"')
                    # Handle multiple date formats:
                    #   "2018"              → 2018
                    #   "2022-06-15"        → 2022
                    #   "6/27/2022 11:06"   → 2022 (M/D/YYYY from Eurostat)
                    if "/" in year_raw:
                        # M/D/YYYY or D/M/YYYY — year is last segment before space
                        date_part = year_raw.split()[0]  # "6/27/2022"
                        parts = date_part.split("/")
                        year = int(parts[-1]) if len(parts) >= 3 else int(parts[0])
                    elif "-" in year_raw:
                        year = int(year_raw[:4])
                    else:
                        year = int(year_raw)
                except (ValueError, TypeError):
                    year = 2018
                if year < min_year:
                    continue

                lc_code = row.get(col_lc, "").strip().upper()

                if crop_only and not is_agricultural(lc_code):
                    continue

                try:
                    lat = float(row.get(col_lat, 0))
                    lon = float(row.get(col_lon, 0))
                except (ValueError, TypeError):
                    continue

                if lat == 0 or lon == 0:
                    continue

                crop_class = lucas_code_to_class(lc_code)
                point_id = row.get(col_id, "")

                entry = {
                    "point_id": point_id,
                    "lat": lat,
                    "lon": lon,
                    "lc_code": lc_code,
                    "crop_class": crop_class,
                    "crop_name": CLASS_NAMES[crop_class],
                    "year": year,
                    "nuts0": "SE",
                }

                # Deduplicate: latest survey year wins
                if point_id not in seen or year > seen[point_id]["year"]:
                    seen[point_id] = entry

    return list(seen.values())


def summarize_lucas_sweden(points: list[dict]) -> dict:
    """Summarize class distribution of loaded LUCAS-SE points.

    Args:
        points: Output from load_lucas_sweden().

    Returns:
        Dict with class counts and total.
    """
    from collections import Counter
    counts = Counter(p["crop_class"] for p in points)
    return {
        "total": len(points),
        "per_class": {
            CLASS_NAMES[cls]: counts.get(cls, 0)
            for cls in range(NUM_CLASSES)
        },
    }
