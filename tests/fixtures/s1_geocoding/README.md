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
