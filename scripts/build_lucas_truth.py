#!/usr/bin/env python
"""scripts/build_lucas_truth.py — build the LUCAS field-truth set for Sweden.

Phase L0 of docs/experiments/lucas_validation_plan.md. Parses the two raw
LUCAS surveys, filters to Sweden, maps LUCAS LC1 → the unified 28-class schema
via an explicit, reviewed table (LUCAS_LC1_TO_UNIFIED below), reprojects
lat/lon (EPSG:4326) → EPSG:3006 (the tile-grid CRS), and writes a single
parquet truth set.

Two source surveys, handled explicitly (their column names differ):
  - EU_LUCAS_2022.csv          → source='eu2022',  year=2022.
      lat/lon: POINT_LAT/POINT_LONG · code: SURVEY_LC1 · perc: SURVEY_LC1_PERC
      (empty for woodland → NaN) · use: SURVEY_LU1 · NUTS0: POINT_NUTS0.
  - LUCAS_2018_Copernicus_attributes.csv → source='cop2018', year=2018.
      lat/lon: TH_LAT/TH_LONG · code: LC1 · perc: LC1_PERC
      (POPULATED for woodland, categorical text e.g. "> 75 %") · use: LU1 ·
      NUTS0: NUTS0.

Honesty discipline (§ plan): ambiguous / non-observable / unmapped LC1 codes
are kept as rows with unified_class == EXCLUDE (-1) so every survey point is
auditable — nothing is silently dropped. Unmapped codes are printed.

Run:  .venv/bin/python scripts/build_lucas_truth.py
Out:  data/lucas/lucas_truth_sweden.parquet
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pyproj

from imint.training.unified_schema import UNIFIED_CLASSES

# ── Paths ────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[1]
LUCAS_DIR = REPO_ROOT / "data" / "lucas"
CSV_2022 = LUCAS_DIR / "EU_LUCAS_2022.csv"
CSV_2018 = LUCAS_DIR / "LUCAS_2018_Copernicus_attributes.csv"
OUT_PARQUET = LUCAS_DIR / "lucas_truth_sweden.parquet"

# Sentinel for "kept but not scorable" — never silently dropped.
EXCLUDE = -1

# CRS: WGS84 geographic → SWEREF99 TM (EPSG:3006), the unified tile grid.
_TRANSFORMER = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3006", always_xy=True)

# Sweden EPSG:3006 sanity envelope (easting, northing metres). Loose bounds
# for the spot-check assertion, not a clip.
SE_3006_E_MIN, SE_3006_E_MAX = 200_000.0, 950_000.0
SE_3006_N_MIN, SE_3006_N_MAX = 6_100_000.0, 7_700_000.0


# ── LUCAS LC1 → unified class (authoritative, reviewed row-by-row) ───────
#
# Value is (unified_class, forest_dominant, is_mixed). unified_class == EXCLUDE
# means the row is kept for audit but is NOT scorable against the tall/gran/löv
# argmax or the unified crop set. forest_dominant ∈ {'tall','gran','lov',None};
# is_mixed is only True for blandskog (class 4).
#
# Crop B-codes follow the task's explicit unified mapping (NOT the 9-class
# crop_schema.LUCAS_TO_CROP indices, which target a different, retired
# crop-only schema — e.g. crop_schema maps B13→1 'vete' whereas the unified
# truth-set maps B13→12 'korn'). The table below is the authority for this
# deliverable.
_FD = "forest_dominant"
LUCAS_LC1_TO_UNIFIED: dict[str, tuple[int, str | None, bool]] = {
    # ── Woodland (C) → forest classes 1-4 ────────────────────────────────
    "C22": (1, "tall", False),   # pine-dominated → tallskog
    "C21": (2, "gran", False),   # spruce-dominated → granskog
    "C10": (3, "lov", False),    # broadleaved → lövskog
    "C31": (4, None, True),      # mixed woodland → blandskog
    "C32": (4, None, True),      # mixed woodland → blandskog
    "C33": (4, None, True),      # mixed woodland → blandskog
    # Ambiguous conifer / generic woodland — not scorable for tall/gran/löv
    # argmax (no dominant species resolvable). Kept for audit as EXCLUDE.
    "C20": (EXCLUDE, None, False),  # coniferous (unspecified) — ambiguous
    "C23": (EXCLUDE, None, False),  # other coniferous — not tall/gran
    "C30": (EXCLUDE, None, False),  # mixed woodland (generic) — ambiguous
    "C":   (EXCLUDE, None, False),  # woodland, no level-3 detail
    # ── Cropland (B) → unified crops 11-21 ───────────────────────────────
    "B11": (11, None, False),  # common wheat → vete
    "B12": (11, None, False),  # durum wheat → vete
    "B13": (12, None, False),  # barley → korn
    "B15": (13, None, False),  # oats → havre
    "B14": (20, None, False),  # rye → råg
    "B16": (21, None, False),  # maize → majs
    "B21": (17, None, False),  # potatoes → potatis
    "B22": (18, None, False),  # sugar beet → sockerbetor
    "B32": (14, None, False),  # rape / turnip rape → oljeväxter
    "B35": (14, None, False),  # other oleaginous / fibre → oljeväxter
    "B41": (19, None, False),  # dry pulses → trindsäd
    "B43": (19, None, False),  # dry pulses → trindsäd
    "B44": (19, None, False),  # dry pulses → trindsäd
    "B45": (19, None, False),  # dry pulses → trindsäd
    "B51": (15, None, False),  # clovers / ley → slåttervall
    "B52": (15, None, False),  # lucerne / ley → slåttervall
    "B53": (15, None, False),  # other leguminous ley → slåttervall
    "B54": (15, None, False),  # mixes leguminous ley → slåttervall
    "B55": (15, None, False),  # temporary grasslands (ley) → slåttervall
    # Cropland codes outside the SE unified crop set / fallow → EXCLUDE.
    "B18": (EXCLUDE, None, False),  # other non-permanent industrial crops
    "B23": (EXCLUDE, None, False),  # other root crops
    "B71": (EXCLUDE, None, False),  # apple fruit (permanent) — not in set
    "B73": (EXCLUDE, None, False),  # nuts trees — not in set
    "B74": (EXCLUDE, None, False),  # other fruit trees / berries — not in set
    "B75": (EXCLUDE, None, False),  # oranges — not in set
    "B81": (EXCLUDE, None, False),  # nurseries — not a field crop
    "B84": (EXCLUDE, None, False),  # permanent industrial (e.g. xmas trees)
    "BX1": (EXCLUDE, None, False),  # unclassified arable (LUCAS Bx1) — fallow
    "BX2": (EXCLUDE, None, False),  # unclassified arable (LUCAS Bx2) — fallow
    # ── Artificial (A) → bebyggelse 9 ────────────────────────────────────
    "A11": (9, None, False),  # buildings with 1-3 floors
    "A12": (9, None, False),  # buildings with >3 floors
    "A13": (9, None, False),  # greenhouses
    "A21": (9, None, False),  # non-built artificial (roads/rail)
    "A22": (9, None, False),  # other artificial (yards, car parks)
    "A30": (9, None, False),  # other artificial areas
    # ── Shrubland (D) → buskdominerad 24 ─────────────────────────────────
    # LUCAS cannot split busk (24) vs ris (25) — both fall under D shrubland.
    # We map to 24; documented one-way collapse (plan RL2).
    "D10": (24, None, False),  # shrubland with sparse tree cover
    "D20": (24, None, False),  # shrubland without tree cover
    # ── Grassland (E) → öppen mark 8, LU-split → bete 16 ─────────────────
    # Base map is öppen mark (8); overridden to bete (16) at build time when
    # LU1 flags agriculture (LU code prefix in _AGRI_LU_PREFIXES). Encoded as
    # class 8 here; the LU override is applied in _map_row().
    "E10": (8, None, False),  # grassland with sparse tree/shrub cover
    "E20": (8, None, False),  # grassland without tree/shrub cover
    "E30": (8, None, False),  # spontaneously re-vegetated surfaces
    # ── Bare / lichen-moss (F) → öppen mark utan vegetation 27 ───────────
    "F10": (27, None, False),  # rocks and stones
    "F20": (27, None, False),  # sand
    "F30": (27, None, False),  # lichens and moss
    "F40": (27, None, False),  # other bare soil
    # ── Water (G) → vatten 10 ────────────────────────────────────────────
    "G11": (10, None, False),  # inland running water
    "G12": (10, None, False),  # inland standing water
    "G21": (10, None, False),  # sea/ocean coastal water
    "G40": (EXCLUDE, None, False),  # glaciers / permanent snow — retired class
    # ── Wetland (H) → våtmark 7 ──────────────────────────────────────────
    "H11": (7, None, False),  # inland marshes
    "H12": (7, None, False),  # peatbogs
}

# LUCAS LU1 (land-use) prefixes that mark managed agriculture. Used only to
# split grassland (E) into bete (16, agriculture/grazing) vs öppen mark (8).
# LUCAS nomenclature: U11x = Agriculture (U111 livestock/grazing, U112 fodder,
# U113 other agriculture). Everything else (U12x forestry, U3xx recreation/
# nature, U4xx unused/abandoned) leaves grassland as öppen mark.
_AGRI_LU_PREFIXES = ("U11",)
_GRASSLAND_CLASS = 8   # öppen mark
_BETE_CLASS = 16       # bete

# 2018 Copernicus LC1_PERC is categorical text; map each band to its midpoint
# fraction (%) so lc1_perc is a usable float for magnitude validation. 2022
# leaves this NaN (survey did not record it).
_PERC_BAND_MIDPOINT = {
    "> 75 %": 87.5,
    "50 - 75 %": 62.5,
    "25 - 50 %": 37.5,
    "10 - 25 %": 17.5,
    "< 10 %": 5.0,
}


# ── Helpers ──────────────────────────────────────────────────────────────
def _unified_name(cls: int) -> str:
    """Human-readable unified class name; 'EXCLUDE' for the -1 sentinel."""
    return "EXCLUDE" if cls == EXCLUDE else UNIFIED_CLASSES[cls]


def _parse_lc1_perc(raw: object, source: str) -> float:
    """Parse LC1_PERC to a float percentage.

    2018 Copernicus stores categorical bands ("> 75 %", "50 - 75 %", …) →
    mapped to the band midpoint. 2022 leaves it empty → NaN. Numeric strings
    (defensive) are parsed directly.
    """
    if raw is None:
        return np.nan
    s = str(raw).strip()
    if s == "" or s.lower() == "nan":
        return np.nan
    if s in _PERC_BAND_MIDPOINT:
        return _PERC_BAND_MIDPOINT[s]
    try:
        return float(s)
    except ValueError:
        return np.nan


def _map_row(lc1: str, lu1: str) -> tuple[int, str | None, bool]:
    """Map one LUCAS LC1 (+ LU1 for the grassland split) to unified.

    Returns (unified_class, forest_dominant, is_mixed). Codes absent from
    LUCAS_LC1_TO_UNIFIED return EXCLUDE (counted, never dropped).
    """
    if lc1 not in LUCAS_LC1_TO_UNIFIED:
        return EXCLUDE, None, False
    cls, dominant, is_mixed = LUCAS_LC1_TO_UNIFIED[lc1]
    # Grassland → bete when land-use is agriculture.
    if cls == _GRASSLAND_CLASS and lu1.startswith(_AGRI_LU_PREFIXES):
        return _BETE_CLASS, None, False
    return cls, dominant, is_mixed


def _load_survey(
    csv_path: Path,
    *,
    source: str,
    year: int,
    col_id: str,
    col_nuts: str,
    col_lat: str,
    col_lon: str,
    col_lc1: str,
    col_perc: str,
    col_lu1: str,
) -> pd.DataFrame:
    """Load one LUCAS survey CSV, filter to Sweden, map to unified truth rows.

    Column names are passed explicitly per survey — the two files disagree on
    every column name. All rows (including EXCLUDE) are kept.
    """
    df = pd.read_csv(
        csv_path,
        usecols=[col_id, col_nuts, col_lat, col_lon, col_lc1, col_perc, col_lu1],
        dtype=str,
        low_memory=False,
    )
    se = df[df[col_nuts].str.startswith("SE", na=False)].copy()

    lat = pd.to_numeric(se[col_lat], errors="coerce")
    lon = pd.to_numeric(se[col_lon], errors="coerce")
    lc1 = se[col_lc1].fillna("").str.strip().str.upper()
    lu1 = se[col_lu1].fillna("").str.strip().str.upper()

    mapped = [_map_row(c, u) for c, u in zip(lc1, lu1)]
    unified_class = np.array([m[0] for m in mapped], dtype=np.int16)
    forest_dominant = [m[1] for m in mapped]
    is_mixed = np.array([m[2] for m in mapped], dtype=bool)

    lc1_perc = np.array(
        [_parse_lc1_perc(v, source) for v in se[col_perc].tolist()],
        dtype=np.float64,
    )

    # Reproject EPSG:4326 → EPSG:3006. always_xy → pass (lon, lat).
    easting, northing = _TRANSFORMER.transform(lon.to_numpy(), lat.to_numpy())

    out = pd.DataFrame(
        {
            "point_id": se[col_id].to_numpy(),
            "source": source,
            "year": np.int16(year),
            "lat": lat.to_numpy(),
            "lon": lon.to_numpy(),
            "easting": easting,
            "northing": northing,
            "lc1": lc1.to_numpy(),
            "unified_class": unified_class,
            "unified_name": [_unified_name(int(c)) for c in unified_class],
            "forest_dominant": forest_dominant,
            "is_mixed": is_mixed,
            "lc1_perc": lc1_perc,
            "lu1": lu1.to_numpy(),
        }
    )
    return out


# ── Verification ─────────────────────────────────────────────────────────
def _print_counts(df: pd.DataFrame, title: str) -> None:
    """Per-unified-class point counts for a (sub)frame."""
    print(f"\n── {title} (n={len(df)}) ──")
    counts = df["unified_class"].value_counts().sort_index()
    for cls, n in counts.items():
        print(f"  {int(cls):>3}  {_unified_name(int(cls)):<28} {int(n):>7}")


def _print_exclude_table(df: pd.DataFrame) -> None:
    """LC1 codes that mapped to EXCLUDE, with counts (audit trail)."""
    excl = df[df["unified_class"] == EXCLUDE]
    print(f"\n── EXCLUDE codes (n={len(excl)}) ──")
    for code, n in Counter(excl["lc1"]).most_common():
        print(f"  {code:<5} {n:>7}")


def _verify(df: pd.DataFrame) -> list[str]:
    """Return unmapped LC1 codes (not in the table at all) and run asserts."""
    known = set(LUCAS_LC1_TO_UNIFIED)
    present = set(df["lc1"].unique()) - {""}
    unmapped = sorted(present - known)

    # forest_dominant domain.
    fd = set(df["forest_dominant"].dropna().unique())
    assert fd <= {"tall", "gran", "lov"}, f"bad forest_dominant: {fd}"

    # is_mixed only True for class 4.
    mixed_classes = set(df.loc[df["is_mixed"], "unified_class"].unique())
    assert mixed_classes <= {4}, f"is_mixed set on non-blandskog: {mixed_classes}"
    assert df.loc[df["unified_class"] == 4, "is_mixed"].all(), (
        "class 4 rows must all be is_mixed=True"
    )

    # forest_dominant present iff dominated forest (1/2/3).
    dom_rows = df["forest_dominant"].notna()
    assert set(df.loc[dom_rows, "unified_class"].unique()) <= {1, 2, 3}, (
        "forest_dominant set outside classes 1-3"
    )

    # lc1_perc: populated for 2018 woodland, NaN for 2022 woodland.
    wood_2018 = df[(df["source"] == "cop2018") & df["lc1"].str.startswith("C")]
    wood_2022 = df[(df["source"] == "eu2022") & df["lc1"].str.startswith("C")]
    assert wood_2018["lc1_perc"].notna().all(), (
        "2018 woodland must have lc1_perc"
    )
    assert wood_2022["lc1_perc"].isna().all(), (
        "2022 woodland must have NaN lc1_perc"
    )
    print(
        f"\n2018 woodland lc1_perc populated: {wood_2018['lc1_perc'].notna().sum()}"
        f"/{len(wood_2018)} · 2022 woodland NaN: "
        f"{wood_2022['lc1_perc'].isna().sum()}/{len(wood_2022)}"
    )
    return unmapped


def _spot_check(df: pd.DataFrame, n: int = 3) -> None:
    """Print n lat/lon → E/N conversions and assert Sweden 3006 bounds."""
    print(f"\n── Spot-check {n} points (3006 bounds) ──")
    sample = df.sample(n, random_state=42)
    for _, r in sample.iterrows():
        in_bounds = (
            SE_3006_E_MIN <= r["easting"] <= SE_3006_E_MAX
            and SE_3006_N_MIN <= r["northing"] <= SE_3006_N_MAX
        )
        print(
            f"  {r['point_id']:<12} lat={r['lat']:.5f} lon={r['lon']:.5f}"
            f" → E={r['easting']:.1f} N={r['northing']:.1f}"
            f"  in_SE={in_bounds}"
        )
        assert in_bounds, f"point {r['point_id']} outside Sweden 3006 bounds"


# ── Main ─────────────────────────────────────────────────────────────────
def main() -> None:
    df_2022 = _load_survey(
        CSV_2022,
        source="eu2022",
        year=2022,
        col_id="POINT_ID",
        col_nuts="POINT_NUTS0",
        col_lat="POINT_LAT",
        col_lon="POINT_LONG",
        col_lc1="SURVEY_LC1",
        col_perc="SURVEY_LC1_PERC",
        col_lu1="SURVEY_LU1",
    )
    df_2018 = _load_survey(
        CSV_2018,
        source="cop2018",
        year=2018,
        col_id="POINT_ID",
        col_nuts="NUTS0",
        col_lat="TH_LAT",
        col_lon="TH_LONG",
        col_lc1="LC1",
        col_perc="LC1_PERC",
        col_lu1="LU1",
    )
    df = pd.concat([df_2022, df_2018], ignore_index=True)

    n_in = len(df)
    n_excl = int((df["unified_class"] == EXCLUDE).sum())
    n_mapped = n_in - n_excl
    print(f"Total SE points: {n_in}  mapped: {n_mapped}  excluded: {n_excl}")

    # Verification (mandatory).
    _print_counts(df, "ALL (both years)")
    _print_counts(df[df["source"] == "cop2018"], "2018 Copernicus subset")
    _print_exclude_table(df)
    unmapped = _verify(df)
    if unmapped:
        print(f"\n!! UNMAPPED LC1 codes (kept as EXCLUDE, counted): {unmapped}")
    else:
        print("\nNo unmapped LC1 codes — every present code is in the table.")
    _spot_check(df)

    # Write.
    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PARQUET, index=False)
    print(f"\nWrote {OUT_PARQUET}  ({len(df)} rows)")

    # Round-trip verification.
    rt = pd.read_parquet(OUT_PARQUET)
    assert len(rt) == len(df), f"round-trip row mismatch: {len(rt)} != {len(df)}"
    assert rt["easting"].notna().all(), "NaN in easting after round-trip"
    assert rt["northing"].notna().all(), "NaN in northing after round-trip"
    print(
        f"Round-trip OK: {len(rt)} rows, no NaN in easting/northing "
        f"(E/N NaN = {rt['easting'].isna().sum()}/{rt['northing'].isna().sum()})"
    )


if __name__ == "__main__":
    main()
