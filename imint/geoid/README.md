# EGM96 geoid grid

| File | Source | Bytes | sha256 |
|---|---|---|---|
| `us_nga_egm96_15.tif` | https://cdn.proj.org/us_nga_egm96_15.tif | 2710815 | `db493027562c9b004d7220fa881f5603adada4e1c5029b933fa7de4547b0e78d` |

EGM96 geoid undulation above WGS84 on a 15' global lattice (1440 x 721 float32,
`AREA_OR_POINT=Point`, nodes on pixel centres). Derived by PROJ from NGA's
`egm96_15.gtx`; NGA work, public domain.

Used by `imint.geoid.egm96_undulation` (bilinear, matches PROJ `vgridshift` on
this file to < 1e-9 m). Over the Skagen–Marstrand AOI it agrees with EGM2008
2.5' within 0.22 m.
