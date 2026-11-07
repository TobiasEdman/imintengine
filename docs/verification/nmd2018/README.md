# NMD 2018 basskikt legend

`nmd2018bas_ogeneraliserad_v1_1.qml` is the QGIS layer style that
Naturvårdsverket ships with the NMD 2018 basskikt raster, retrieved
2026-11-07 from the official archive
[NMD2018_basskikt_ogeneraliserad_Sverige_v1_1.zip](https://geodata.naturvardsverket.se/nedladdning/marktacke/NMD2018/NMD2018_basskikt_ogeneraliserad_Sverige_v1_1.zip)
(1,406,790,147 bytes, Last-Modified 2023-06-09 10:54:46 GMT, ETag
`"53d9ea03-5fdb0327cf9a4"`). Only this member was read, using HTTP byte
ranges (member CRC-32 `e5df0777`, 17,547 bytes, dated 2020-07-01); the
archive itself was not downloaded. The member is byte-identical to the copy
in the locally held archive of the same size.

SHA256 `8faa67351c71a641a7ee2f4e4d868c076e44e1810b9dfeec99241e3769752dfd`
(`NMD2018_QML_SHA256` in `scripts/showcase_fire_years.py`).

The style lists all 256 raster values. The 25 basskikt classes are the
entries whose label carries a name after the code (2, 3, 41, 42, 51–53, 61,
62, 111–118, 121–128); the other entries are unnamed, and the showcase
refuses a raster code without a name. The fire tab draws NMD 2018 from raw
codes with these names and colours. The engine's NMD 2018 analyzer tables
(`imint/analyzers/nmd.py`) are separate and unchanged.
