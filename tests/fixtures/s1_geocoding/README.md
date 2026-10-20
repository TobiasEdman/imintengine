# S1 geocoding fixtures (tests/test_s1_geocoding.py)

Built by `build_fixtures.py` from the 2026-10-20 S1 offset diagnosis; ~140 KB.

| File | Content | Source |
|---|---|---|
| `headers/<scene>_vv_header.json.gz` | `_s1_scene_header` output: raw GCPs + incidence per GCP, 4 scenes | Copernicus Sentinel-1 data 2026 (MPC `sentinel-1-grd`) |
| `annotation/<scene>_vv_trimmed.xml.gz` | product annotation cut to adsHeader, productInformation, orbitList, imageInformation, geolocationGrid (06-28 S1D, 06-27 S1C asc) | same |
| `anchored_ships.json` | per layer, slow ships (SOG < 1 kn, true heading and A/B/C/D known): frozen radar (row, col) of the detected hull, AIS antenna and hull centre at scene time | detections: chain-1 layers; AIS: Danish Maritime Authority raw AIS (aisdata, public) |

The radar coordinates are GDAL pixel coordinates (corner lattice) in each
scene's full raster, from the chain-1 TPS inverse of the detection's box
centre; they locate the hull in the raster and do not depend on the GCPs.

## Build inputs (sha256)

`build_fixtures.py --diagnosis <marine-awareness research>/2026-10-20_s1_offset_diagnosis
--headers <that dir>/fixture_inputs/headers` reproduces the fixtures from:

| Input | sha256 |
|---|---|
| `rows_step3.json` | `7bc75c1ee71bafe4cf5d249e50ccdc6e08ba78d1f7a482748e5dad86bf6c1ac4` |
| `rows_step7.json` | `e1c3ecddd71e5d52f8387be794d826c1d0b688c7f7847b470770e30668a2f2d6` |
| `annotation/S1D_IW_GRDH_1SDV_20260628T053135_20260628T053200_003432_0060CF_vv_annotation.xml` | `b284fd32a11ad60769c00c17075cefcdad8bcb53433181364aebbdb304062c15` |
| `annotation/S1C_IW_GRDH_1SDV_20260627T165257_20260627T165322_008294_010682_vv_annotation.xml` | `051d0ca6655e5425f39cb56b80cfde2040f85b3004f4af606ab5def7fb4029fa` |
| `fixture_inputs/headers/S1C_IW_GRDH_1SDV_20260627T053911_20260627T053936_008287_01063E/vv_header.json` | `ce3b27da6edf7da980cca8215cbecfb945105108bab18115d15609b112f3c79c` |
| `fixture_inputs/headers/S1C_IW_GRDH_1SDV_20260627T053936_20260627T054001_008287_01063E/vv_header.json` | `3b1978e1eb2a0ae667461daad20f9204d3ba3e6d891f6dbcba822e08c2c5ee6a` |
| `fixture_inputs/headers/S1C_IW_GRDH_1SDV_20260627T165257_20260627T165322_008294_010682/vv_header.json` | `dbcc3732872284b006480ba8c90dde7b91a1d340b21a418d5ad7b182f5ff4b14` |
| `fixture_inputs/headers/S1D_IW_GRDH_1SDV_20260628T053135_20260628T053200_003432_0060CF/vv_header.json` | `eb6fa5ed6e96f30fac3fdef08f48dfff5980a75e4f3cd76e3367ae2f64c7f0cb` |

The headers were written by `_s1_scene_header` from the MPC products on 2026-10-20; the
gzipped copies here decode to the same JSON.
