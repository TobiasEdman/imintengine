"""Year-stratified target-group negatives.

The single-shot random-offset fetch returned negatives from 2024 only,
because GBIF's default ordering is not random across years. Year then
predicted the class label almost perfectly (year-only control AUC 0.896),
so the embedding result measured year, not lupine. Fetching per year with
an explicit year filter removes the confound at the source.
"""
import csv, sys
sys.path.insert(0, ".")
from fetch_gbif import BASE, OUT, TRACHEOPHYTA_KEY, fetch_random_offsets

rows = []
for year in (2018, 2019, 2020, 2021, 2022, 2023, 2024):
    p = OUT / f"neg_{year}.csv"
    fetch_random_offsets(
        {**BASE, "taxonKey": TRACHEOPHYTA_KEY, "month": "6,8",
         "coordinateUncertaintyInMeters": "0,25", "year": str(year)},
        p, n_pages=14, seed=1000 + year)
    with p.open() as f:
        rows.extend(list(csv.DictReader(f)))

with (OUT / "negatives_raw.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["lat", "lon", "year", "month",
                                      "uncert_m", "gbifID", "speciesKey"])
    w.writeheader()
    w.writerows(rows)
print(f"negatives_raw.csv rebuilt: {len(rows)} rows across 7 years")
