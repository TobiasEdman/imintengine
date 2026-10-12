# scripts/marine/s1_port/

Re-runnable checks behind `docs/marine/s1_vessels_reference.md`: the AI2
Sentinel-1 vessel reference run and the comparison of the ImintEngine port
(`imint/analyzers/ai2_vessels_s1.py`, `imint.fetch.fetch_s1_vessel_scenes`)
with it. Their small outputs are committed in `docs/marine/s1_port_results/`,
whose README maps every result file to the command that produced it.

Run from the repository root. Every script takes `--help`.

## The reference record

Rasters and full run outputs stay outside git. The scripts take the record
root as `--record`; its layout is in the docstring of `_record.py`. On the
machine that produced it the record is `outputs/s1_reference_20261010/` of the
main checkout (gitignored). `docs/marine/s1_port_results/large_files.sha256`
lists the sha256 of every large file, relative to the record root, so a copy
can be verified with `cd <record> && shasum -a 256 -c <repo>/docs/marine/s1_port_results/large_files.sha256`.

## Environments

- Repo `.venv` plus `pystac-client` and `planetary-computer` (pinned in
  `requirements.txt`) for everything that touches MPC.
- `port_feasibility.py` also needs `rslearn`: AI2's environment, built as in
  the dossier, section a.
- `ma_load_check.py` needs a marine-awareness checkout (`--ma-repo`); it reads
  `load_detections` from git without modifying that repo.

## Scripts

| Script | Checks |
|---|---|
| `pick_historical.py` | AI2's historical-scene rule, independently on MPC |
| `crop_grd.py` | builds the reference run's radar-geometry crops (`inputs/`) |
| `run_ai2_reference.sh` | AI2's own pipeline, unmodified, on those crops (needs AI2's env) |
| `compare_reference_runs.py` | two runs of AI2's pipeline give the same output |
| `mpc_scenes.py` | catalogue facts: scenes over the bbox, product properties, MPC vs CDSE OData footprint |
| `dump_ckpt.py` | checkpoint keys, shapes, parameter counts |
| `reproduce_inputs.py` | independent bit-exact reproduction of AI2's model input |
| `tps_subset_check.py` | geometric effect of the crop's GCP subset |
| `port_feasibility.py` | plain-torchvision detector vs rslearn's model (AI2 env) |
| `port_feasibility_attr.py` | plain-torchvision attribute model vs AI2's attributes |
| `analyze_reference.py` | counts, scores, attributes, infra distances of AI2's run |
| `warp_window_dependence.py` | GDAL warp output depends on the request window |
| `sparse_vrt_on_reference_crop.py` | the fetch's sparse VRT on real data, bit-exact |
| `full_scene_check.py` | the fetch against the whole downloaded scene, bit-exact |
| `compare_inputs.py` | port model input vs AI2's (table a) |
| `run_analyzer_on.py` | runs A (AI2 inputs) and B (port inputs, AI2 tiling) |
| `compare_to_reference.py` | detections of a run vs AI2's reference (tables b-e) |
| `diag_unmatched.py` | best score of every unmatched detection in both tilings |
| `attr_window_check.py` | analyzer attribute crops vs separately warped windows |
| `ma_load_check.py` | the GeoJSON read by marine-awareness's own reader |
| `parity_check.py` | two runs (and model inputs) identical, for refactors |
| `mutation_check.py` | the S1 tests catch the two riskiest one-token bugs |
