# Annual service source checks — 2026-10-29

This is implementation evidence, not classification or accuracy evidence.
No private field identifiers, locations, held-out scores or weights are included.

The official [GeoTessera 0.10.2 release metadata](https://pypi.org/pypi/geotessera/0.10.2/json)
and its referenced wheel were inspected. Wheel SHA256:
`6f0bc52d61b59a534ea0209d2e09e3a3960f19eab608cc2cb6e8d4c9e925800d`.
It requires Python >=3.12. Explicit dataset `v1`, variant `vultr`, normalizes
to version `1.0`. The provided project environment is Python 3.11 with no
GeoTessera. Preparation therefore accepts a separate explicit interpreter.

The public Uppsala centre (17.64 E, 59.86 N) snaps to EPSG:3006
(647850, 6638770), with input bounds (645290, 6636210, 650410, 6641330).
The pinned producer's registry contains both overlapping blocks (17.55,59.85)
and (17.65,59.85) for every candidate year 2018–2024 (14 rows; asserted).
Only 3,285,465 bytes of range-read metadata were read; no embedding was fetched.

Registry: [v1 manifest](https://data.source.coop/tessera/tessera/npy/v1/manifest.parquet),
212,614,861 bytes, ETag `60a7264717b3d025860cd699644d0a53-26`,
modified 2026-08-05T15:44:18Z, object version
`MH1jJYjZ3rsfCyFqRZYJwQVzduTmgW10`. Footer identifies `1.0` / `vultr`.
This verifies registry coverage only, not source bytes, per-pixel validity,
auxiliary availability or the trained head's compatibility with new inputs.

`fetch_embedding` returns dequantized floating H×W×128 arrays and raises on
failure. The iterator `fetch_embeddings` catches and skips failures, so serving
uses individual fetches. Landmask TIFF supplies georeferencing; its pixel mask
is not applied by the library. The preparer verifies shape/affine/CRS and
independently propagates embedding validity through bilinear reprojection.

Initial access was blocked by absent local model/mirror paths and no active
pod mounting the source volume. After explicit user approval, a bounded,
read-only ICE job opened the actual checkpoint and sources and completed a real
Uppsala 2024 test. See [real-case evidence](real-browser.json) and the final
section of [verification](verification.md). The registry-only checks above
remain the earlier coverage investigation; they are distinct from the later
actual embedding downloads and per-pixel validation.
