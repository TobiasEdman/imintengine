"""SLU Riksskogstaxeringen (Swedish National Forest Inventory) plot data.

Reads the NFI field-plot table — ≈44 000 circular r=7 m plots, inventory
years 2007–2025, coordinates in SWEREF99 TM (EPSG:3006) — into a pandas
DataFrame. The source ships as a ~14 MB ``.xlsx``; the first call converts
the data sheet to a parquet cache (``data/nfi/nfi_plots.parquet``) and every
later call reads that, so the slow openpyxl parse happens once.

Provides year / bbox / land-use filtering so plots can be co-located to S2
tiles (bbox in the same ``(west, south, east, north)`` EPSG:3006 convention
as :class:`imint.training.spatial_parquet.SpatialParquet`).

Deliberately a *reference-data reader*, not a label builder: it can decode
the categorical code columns to text but does NOT derive model targets
(dominant-species → forest class, maturity → harvest-ready). Those mappings
live in ``docs/data/nfi_plotdata_DATA_CARD.md`` and are left to whatever
consumes the plots — keeping ingest and labelling decoupled, the same
separation the fetch and label pipelines keep elsewhere in the repo.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

# ── Paths / source layout ─────────────────────────────────────────────────

_REPO_ROOT = Path(__file__).resolve().parents[2]
NFI_DATA_DIR = _REPO_ROOT / "data" / "nfi"
RAW_XLSX_NAME = "swe_nfi_plotdata2.xlsx"
PARQUET_NAME = "nfi_plots.parquet"
DATA_SHEET = "2007-2025"
CRS_EPSG = 3006  # SWEREF99 TM — Easting/Northing columns

# ── Code lookups ──────────────────────────────────────────────────────────
# Verified against the workbook's ReadMe sheet and the observed value counts
# (each code present in the data has exactly one ReadMe label). Soil-moisture
# maps the five ordered ReadMe labels onto the five codes 1..5 (driest→wettest).

LANDUSE_CLASS = {
    1: "Productive forest",
    4: "Mire",
    5: "Rockland",
    6: "Sub-alpine spruce forest",
    7: "Alpine",
    8: "Other forest impediment",
}
SI_SPECIES = {10: "Pine", 20: "Spruce"}
SOILMOISTURE = {1: "Dry", 2: "Fresh", 3: "Fresh-moist", 4: "Moist", 5: "Wet"}

_DECODE = {
    "LandUseClass": ("LandUseClass_text", LANDUSE_CLASS),
    "SISpecies": ("SISpecies_text", SI_SPECIES),
    "Soilmoisture_code": ("Soilmoisture_text", SOILMOISTURE),
}


# ── Cache construction ────────────────────────────────────────────────────

def build_parquet_cache(raw_xlsx: Path, parquet_path: Path) -> Path:
    """Convert the NFI data sheet of ``raw_xlsx`` to a parquet cache.

    Reads only the ``2007-2025`` sheet (the ReadMe sheet is documentation),
    strips whitespace from column names, and writes parquet atomically
    (``.tmp`` then rename) so a crashed write never leaves a half-file.

    Args:
        raw_xlsx: Path to the source ``swe_nfi_plotdata2.xlsx``.
        parquet_path: Destination parquet path.

    Returns:
        ``parquet_path``.
    """
    raw_xlsx = Path(raw_xlsx)
    parquet_path = Path(parquet_path)
    if not raw_xlsx.exists():
        raise FileNotFoundError(
            f"NFI source xlsx not found: {raw_xlsx}\n"
            f"Stage it under {NFI_DATA_DIR}/ — see docs/data/nfi_plotdata_DATA_CARD.md."
        )

    print(f"  Building NFI parquet cache from {raw_xlsx.name} …")
    df = pd.read_excel(raw_xlsx, sheet_name=DATA_SHEET)
    df.columns = [str(c).strip() for c in df.columns]

    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = parquet_path.with_suffix(".parquet.tmp")
    df.to_parquet(tmp_path, index=False)
    tmp_path.rename(parquet_path)
    print(f"  Saved {len(df):,} plots → {parquet_path}")
    return parquet_path


# ── Filters ───────────────────────────────────────────────────────────────

def filter_years(df: pd.DataFrame, years: int | Iterable[int]) -> pd.DataFrame:
    """Return rows whose ``Year`` is in ``years`` (a single int or an iterable)."""
    wanted = {years} if isinstance(years, int) else set(years)
    return df[df["Year"].isin(wanted)]


def plots_in_bbox(
    df: pd.DataFrame, bbox: tuple[float, float, float, float],
) -> pd.DataFrame:
    """Return plots whose (Easting, Northing) fall inside an EPSG:3006 bbox.

    Args:
        df: NFI plot frame (must carry ``Easting`` / ``Northing``).
        bbox: ``(west, south, east, north)`` in EPSG:3006 metres — same
            convention as :meth:`SpatialParquet.query`.

    Returns:
        Filtered copy (inclusive bounds).
    """
    west, south, east, north = bbox
    e, n = df["Easting"], df["Northing"]
    return df[(e >= west) & (e <= east) & (n >= south) & (n <= north)]


def decode_codes(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``*_text`` columns for the categorical code columns we can decode.

    Adds ``LandUseClass_text``, ``SISpecies_text`` and ``Soilmoisture_text``
    next to their code columns (unmapped codes / NaN become NaN). Returns a
    copy; the original frame is untouched.
    """
    out = df.copy()
    for code_col, (text_col, mapping) in _DECODE.items():
        if code_col in out.columns:
            out[text_col] = out[code_col].map(mapping)
    return out


# ── Public entry point ────────────────────────────────────────────────────

def load_nfi_plots(
    *,
    data_dir: Path = NFI_DATA_DIR,
    raw_xlsx: Path | None = None,
    rebuild: bool = False,
    years: int | Iterable[int] | None = None,
    bbox: tuple[float, float, float, float] | None = None,
    land_use: int | Iterable[int] | None = None,
    decode: bool = False,
) -> pd.DataFrame:
    """Load the NFI plot table, building the parquet cache on first use.

    Args:
        data_dir: Directory holding the raw xlsx and the parquet cache.
        raw_xlsx: Override path to the source xlsx (defaults to
            ``data_dir/swe_nfi_plotdata2.xlsx``).
        rebuild: Force a rebuild of the parquet cache from the xlsx.
        years: Keep only these inventory year(s). For S2 co-location use
            ``range(2018, 2026)`` (Sentinel-2 era).
        bbox: ``(west, south, east, north)`` EPSG:3006 — keep only plots
            inside it.
        land_use: Keep only these ``LandUseClass`` code(s) (e.g. ``1`` for
            productive forest).
        decode: If True, append the ``*_text`` decode columns.

    Returns:
        A pandas DataFrame, one row per plot-visit.
    """
    data_dir = Path(data_dir)
    raw = Path(raw_xlsx) if raw_xlsx else data_dir / RAW_XLSX_NAME
    parquet_path = data_dir / PARQUET_NAME

    if rebuild or not parquet_path.exists():
        build_parquet_cache(raw, parquet_path)

    df = pd.read_parquet(parquet_path)

    if years is not None:
        df = filter_years(df, years)
    if land_use is not None:
        wanted = {land_use} if isinstance(land_use, int) else set(land_use)
        df = df[df["LandUseClass"].isin(wanted)]
    if bbox is not None:
        df = plots_in_bbox(df, bbox)
    if decode:
        df = decode_codes(df)

    return df.reset_index(drop=True)


if __name__ == "__main__":  # pragma: no cover — quick characterisation
    full = load_nfi_plots()
    s2 = filter_years(full, range(2018, 2026))
    print(f"NFI plots: {len(full):,} total, {len(s2):,} in S2 era (≥2018)")
    print(f"  years   : {int(full['Year'].min())}–{int(full['Year'].max())}")
    print(f"  Easting : {int(full['Easting'].min()):,}..{int(full['Easting'].max()):,}")
    print(f"  Northing: {int(full['Northing'].min()):,}..{int(full['Northing'].max()):,}")
    print(f"  columns : {len(full.columns)}")
