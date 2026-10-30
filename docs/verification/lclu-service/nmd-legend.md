# Annual-map legend compared with NMD

Verified 2026-10-29 against Naturvårdsverket's official
[NMD 2023 basskikt v2.1 archive](https://geodata.naturvardsverket.se/nedladdning/marktacke/NMD2023/Basskikt_v2_x/NMD2023_basskikt_v2_1.zip).
Only the QML member was retrieved, using HTTP byte ranges; the 2.7 GB
raster archive was not downloaded. The pinned reference is
`NMD2023bas_v2_1.qml`, SHA256
`0154ee34d2ccc150246ea4901adaac3765b9cf02b1ea5c2d35ff07fc800a4322`.

The presentation palette is isolated in `imint/inference/lclu_legend.py`.
Model weights, numerical output IDs and training schema are unchanged.
GeoTIFF colour table, nearest-neighbour PNG, legend and pixel popup share
that presentation. The module participates in the result cache identity.

| Model ID | Display name | Official colour source | Match |
|---|---|---|---|
| 1 | Tallskog på fastmark | 111 `#6d8b05` | Model class; see lineage below |
| 2 | Granskog på fastmark | 112 `#2c5f00` | Model class; see lineage below |
| 3 | Lövskog | 115 `#4be600` | Representative; see below |
| 4 | Blandskog | 114 `#38a800` | Representative; model class |
| 5 | Skog på våtmark | 121 `#588b54` | Representative of merged forest types |
| 6 | Temporärt ej skog på fastmark | 118 `#cdcd66` | Direct |
| 7 | Öppen våtmark | 200 `#c29cd5` | General class; subtypes merged |
| 8 | Öppen mark (sammanslagen) | Own `#d2b48c` | No unique NMD counterpart |
| 9 | Exploaterad mark | 52 `#e24549` | Representative; includes buildings and roads |
| 10 | Sjö, vattendrag och hav | 61 `#6699cc` | Representative; inland and sea merged |
| 11–21 | LPIS agriculture classes | Existing own colours | Extensions beyond NMD |
| 22 | Hygge | Existing own colour | SKS extension beyond NMD |
| 23 | Exploaterad mark, torvtäkt | 54 `#8c1964` | Direct |
| 24 | Buskdominerad mark | 4212 `#abc8a6` | Fresh subtype represents merged moisture types |
| 25 | Risdominerad mark | 4222 `#cdaa66` | Fresh subtype represents merged moisture types |
| 26 | Gräsdominerad mark | 4232 `#ffd27e` | Fresh subtype represents merged moisture types |
| 27 | Öppen mark utan vegetation | 411 `#e0e0e0` | Direct, excluding glacier/permanent snow |

Class 0 remains opaque black; nodata 255 remains transparent and distinct.

The base lookup maps NMD 113 (mixed conifer), 115–117 (deciduous types)
and 43 (low mountain forest on dry land) to class 3. That lookup alone does
not describe the pinned R4 training targets. The 2026-10-29 lineage audit
traced the forest-label refinement through `scripts/distill_forest_labels.py`
and `scripts/validate_against_nfi.py::derive_nfi_forest_class`: field-volume
predictions refine classes 1–4 into pine, spruce, deciduous and mixed forest.
The producer has no separate mixed-conifer output. Unknown head predictions
and the four-pixel border retain the original labels, so the legacy mapping
cannot be declared absent everywhere.

The map therefore uses **Lövskog** and **Blandskog**, with a note that these
field-refined model classes do not exactly match NMD and mixed conifer is not
reported separately. Colour references remain representative NMD swatches;
this correction changes no class ID, prediction or checkpoint. The previous
“Lövskog / barrblandskog” display inferred too much from the base lookup.

In the bounded development probe (24 historical training tiles, 2022
features against fixed NMD2023), none of the 481 sampled interior NMD113
pixels had distilled class 3 or R4 output class 3. This supports the lineage
correction, not a national claim about every pixel or model accuracy.
Private aggregate evidence is held under shared continuity evidence,
`ImintEngine/class-separability-0127/`; no plot locations or raw features are
included in the public demo. Schema changes and independent validation
remain IM-023 work for the next model iteration.

Class 5 also includes temporarily non-forested wetland
and low mountain forest. Class 8 spans legacy open land, fallow and
NMD 2023 glacier/permanent snow, so it retains a distinct own colour.

The UI exposes aggregation notes in the legend details and click popups.
Tests compare every selected official swatch against the pinned QML,
verify all 28 native class IDs remain unchanged, and compare GeoTIFF,
PNG and exported legend colours. The click endpoint samples native 10 m
GeoTIFF cells and reads the legend saved with that result, without inference.

Full source receipt and reproducible 28-class comparison are saved locally
under `outputs/lclu-nmd-legend-20261029/`.
