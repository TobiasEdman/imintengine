"""
imint/training/crop_schema.py — Swedish crop class schema and LUCAS mapping

Maps LUCAS Copernicus 2018/2022 land cover codes to Swedish crop classes
based on Jordbruksverkets (SJV) grödgrupper and SCB crop statistics.

Only crops actually grown in Sweden are included. Exotic crops not present
in Swedish agriculture (soya B33, tobacco B45) are excluded.

SJV grödgrupper (Jordbruksmarkens användning 2024):
    Spannmål (33%): vete, korn, havre, råg, rågvete
    Vall och grönfoderväxter (38%): slåttervall, betesvall, grönfoder
    Oljeväxter (3%): raps, rybs, solros, lin
    Trindsäd (2%): ärtor, åkerbönor
    Potatis (1%): matpotatis, stärkelsepotatis
    Sockerbetor (1%): sockerbeta
    Träda (5%): träda
    Övriga växter (2%): grönsaker, blommor, jordgubbar, frukt, bär

Data: LUCAS Copernicus 2018+2022, Figshare + Eurostat
Reference: Jordbruksverket grödkoder (jordbruksverket.se/stod/.../grodkoder)
           SCB Jordbruksstatistisk sammanställning 2024
"""
from __future__ import annotations

# ── Swedish crop class schema (SJV/SCB grödgrupper) ──────────────────────

NUM_CLASSES = 9

CLASS_NAMES = {
    0: "other",             # Ej jordbruksmark
    1: "vete",              # SJV grödkod 1-2 (höstvete, vårvete)
    2: "korn",              # SJV grödkod 3-4 (höstkorn, vårkorn)
    3: "havre",             # SJV grödkod 5 (havre)
    4: "oljevaxter",        # SJV grödkod 85-92 (raps, rybs, solros, lin)
    5: "vall",              # SJV grödkod 49-52, 80 (slåttervall, betesvall, grönfoder)
    6: "potatis",           # SJV grödkod 70-72 (matpotatis, stärkelsepotatis)
    7: "trindsad",          # SJV grödkod 30-43 (ärtor, åkerbönor, bruna bönor)
    8: "ovrig_akergroda",   # SJV: sockerbeta, råg, rågvete, grönsaker, frukt, bär, blommor
}

# Swedish names for display
CLASS_NAMES_SV = {
    0: "Ej jordbruk",
    1: "Vete",
    2: "Korn",
    3: "Havre",
    4: "Oljeväxter",
    5: "Vall/grönfoder",
    6: "Potatis",
    7: "Trindsäd",
    8: "Övriga åkergrödor",
}

CLASS_COLORS = {
    0: (0.50, 0.50, 0.50),  # grey
    1: (0.93, 0.79, 0.13),  # gold — wheat
    2: (0.85, 0.65, 0.13),  # dark gold — barley
    3: (0.96, 0.87, 0.70),  # light wheat — oats
    4: (0.80, 0.90, 0.10),  # yellow-green — rapeseed
    5: (0.20, 0.80, 0.20),  # green — ley/grass
    6: (0.60, 0.40, 0.20),  # brown — potato
    7: (0.40, 0.70, 0.40),  # medium green — legumes
    8: (0.70, 0.70, 0.50),  # khaki — other
}

# ── LUCAS → SJV crop class mapping ───────────────────────────────────────
#
# Only LUCAS codes for crops ACTUALLY GROWN in Sweden are mapped.
# Codes for crops NOT grown in Sweden (B33 soya, B36 sunflower,
# B45 tobacco, etc.) are EXCLUDED → mapped to 0 ("other").
#
# Source: Jordbruksverket grödkoder + SCB jordbruksstatistik 2024

LUCAS_TO_CROP = {
    # ── Spannmål (SJV grödgrupp: Spannmål, 33%) ─────────────────────
    # Vete (SJV 1-2)
    "B11": 1,  # Common wheat (höstvete/vårvete)
    "B13": 1,  # Durum wheat (durumvete — liten areal, Skåne)

    # Korn (SJV 3-4)
    "B14": 2,  # Barley (höst-/vårkorn)

    # Havre (SJV 5)
    "B15": 3,  # Oats (havre)

    # ── Oljeväxter (SJV grödgrupp: Oljeväxter, 3%) ──────────────────
    "B32": 4,  # Rape/turnip rape (höstraps, vårraps, rybs — SJV 85-88)
    "B34": 4,  # Flax/linseed (lin — SJV 91, ~3000 ha Västergötland)
    "B35": 4,  # Other oil crops (övriga oljeväxter — SJV 92)
    "B36": 4,  # Sunflower (solros — SJV 90, ~5000 ha, ökar, Skåne/Gotland)

    # ── Vall och grönfoderväxter (SJV grödgrupp: Vall, 38%) ──────────
    "B51": 5,  # Fodder crops (grönfoder, fodermajs — SJV 80)
    "B52": 5,  # Grassland / permanent pasture (betesvall — SJV 52)
    "B53": 5,  # Clover (klövervall)
    "B54": 5,  # Lucerne (lusern — södra Sverige)
    "B55": 5,  # Mixed ley (slåttervall, blandvall — SJV 49-50)

    # ── Potatis (SJV grödgrupp: Potatis, 1%) ─────────────────────────
    "B21": 6,  # Potato (matpotatis, stärkelsepotatis — SJV 70-72)

    # ── Trindsäd (SJV grödgrupp: Trindsäd, 2%) ──────────────────────
    "B31": 7,  # Peas / field beans (ärtor, åkerbönor — SJV 30-43)

    # ── Övriga åkergrödor (SJV: diverse, ~7%) ────────────────────────
    "B12": 8,  # Rye (råg — SJV 11)
    "B16": 8,  # Triticale (rågvete — SJV 13)
    "B17": 8,  # Mixed cereals (blandsäd — SJV 14)
    "B18": 8,  # Other cereals (övrig spannmål)
    "B22": 8,  # Sugar beet (sockerbeta — SJV 22, Skåne/Gotland)
    "B23": 8,  # Other root crops (övriga rotfrukter)
    "B37": 8,  # Fibre crops (hampa — SJV 93, odlas i Sverige)
    "B41": 8,  # Vegetables (grönsaker — SJV 74)
    "B42": 8,  # Flowers (blommor, frilandsodling — SJV 75)
    "B43": 8,  # Strawberry (jordgubbar — SJV 73)
    "B44": 8,  # Other industrial crops (övriga industrigrödor)
    "B71": 8,  # Apple/pear (äpple/päron — Skåne, Gotland)
    "B73": 8,  # Cherry (körsbär)
    "B74": 8,  # Plum (plommon)
    "B75": 8,  # Berry plantation (bärplantering — SJV 76)
}

# Codes EXCLUDED (not grown in Sweden):
# B33 Soya — not grown in Sweden (for och kall)
# B45 Tobacco — not grown in Sweden
# B81 Nursery — not a field crop
# B84 Christmas trees — forestry, not crop
# Bx1/Bx2 Unknown — unclear classification

# All LUCAS codes that are Swedish agricultural
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
