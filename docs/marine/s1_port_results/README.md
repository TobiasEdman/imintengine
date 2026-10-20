# docs/marine/s1_port_results/

Results behind `docs/marine/s1_vessels_reference.md`, each produced by a script in
`scripts/marine/s1_port/` (run from the repository root). Rasters and full run outputs
stay outside git in the reference record; `large_files.sha256` lists them with their
sha256, relative to the record root (`cd <record> && shasum -a 256 -c <this file>`).

Variables in the commands:

```bash
S=scripts/marine/s1_port
REC=<record root>                 # outputs/s1_reference_20261010/ of the main checkout
PORT=$REC/s1port/cache/s1_vessel_inputs/2026-06-28_76b7613d2936b018.tif     # bbox fetch
REFWIN=$REC/s1port/cache/s1_vessel_inputs/2026-06-28_b6c843d64568538a.tif   # AI2-window fetch
PORT2=<chain-2 cache>/s1_vessel_inputs/2026-06-28_<key>.tif   # bbox fetch, chain 2, --pass-time 05:31
MA=<marine-awareness checkout>
```

`PORT` is `fetch_s1_vessel_scenes` for the bbox 10.50 57.55 11.70 58.05 on 2026-06-28;
`REFWIN` the same for 10.12 57.28 12.0 58.21 (covers AI2's window), written by the
pre-review cache format (same pixel chain). MPC scripts need `pystac-client` and
`planetary-computer`.

`PORT` and `REFWIN` are pixel chain 1 (GCPs on the ellipsoid), as are AI2's rasters. The
scripts that compare with AI2 refuse another chain; `sparse_vrt_on_reference_crop.py`
runs only on a chain-1 checkout. `PORT2` below is a chain-2 fetch of the same request.

## Reference run (dossier sections a–h)

| File | Command |
|---|---|
| `historical_selection.json` | `$S/pick_historical.py --out historical_selection.json` |
| `mpc_search_2026-06-27_28.json` | `$S/mpc_scenes.py search 2026-06-27T00:00:00Z/2026-06-29T00:00:00Z --out …` |
| `mpc_inspect.json` | `$S/mpc_scenes.py inspect <target id> <historical id> --out …` |
| `mpc_footprint_vs_odata.json` | `$S/mpc_scenes.py footprint <target id> <historical id> --out …` |
| `ckpt_detect.json`, `ckpt_attribute.json` | `$S/dump_ckpt.py imint/fm/ai2_vessels_s1/<ckpt> $REC/<detect\|attribute>_keys.txt --out …` |
| `reproduce_inputs.json` | `$S/reproduce_inputs.py --record $REC --out …` |
| `tps_subset_check.json` | `$S/tps_subset_check.py --record $REC --out …` |
| `port_feasibility.json` | `<AI2 env>/.venv/bin/python $S/port_feasibility.py --record $REC --out …` |
| `port_feasibility_attr.json` | `$S/port_feasibility_attr.py --record $REC --out …` |
| `reference_summary.json` | `$S/analyze_reference.py --record $REC --out …` |
| `ai2_rerun_check.json` | `$S/run_ai2_reference.sh <AI2 env> <new record with inputs/>`, then `$S/compare_reference_runs.py --record $REC --other <new record> --out …` |

## Port comparison

| File | Command |
|---|---|
| `warp_window_dependence.json` | `$S/warp_window_dependence.py --record $REC --out …` |
| `sparse_vrt_target_vv.json`, `sparse_vrt_historical1_vh.json` | `$S/sparse_vrt_on_reference_crop.py --record $REC --scene <target\|historical1> --pol <vv\|vh> --work-dir <tmp> --out …` |
| `full_scene_check.json` | `$S/full_scene_check.py --port-input $PORT --full-tif $REC/s1port/full_target_vv.tif --out …` (chain 1, before `--work-dir` existed) |
| `full_scene_check_chain2.json` | `$S/full_scene_check.py --port-input $PORT2 --full-tif $REC/s1port/full_target_vv.tif --work-dir <tmp> --out …` |
| `cmp_inputs.json` | `$S/compare_inputs.py --record $REC --port-input $PORT --part1 --out …` |
| `cmp_run_a.json` | `$S/run_analyzer_on.py --record $REC --mode ai2 --out run_a.json`, then `$S/compare_to_reference.py run_a.json "A: AI2 inputs" --record $REC --out …` |
| `cmp_run_b.json` | `$S/run_analyzer_on.py --record $REC --mode port --port-input $REFWIN --out run_b.json`, then `compare_to_reference.py run_b.json "B: …" --record $REC --out …` |
| `cmp_run_c.json` | `scripts/detect_vessels_s1.py --date 2026-06-28 --bbox 10.50 57.55 11.70 58.05 --cache-dir $REC/s1port/cache --infra-path $REC/inputs/marine_infrastructure.geojson --out surv.geojson --run-json run_c.json`, then `compare_to_reference.py run_c.json "C: …" --record $REC --bbox 10.50 57.55 11.70 58.05 --out …` |
| `diag_run_b.json` | `$S/diag_unmatched.py cmp_run_b.json --record $REC --port-input $REFWIN --run-b --out …` |
| `diag_run_c.json` | `$S/diag_unmatched.py cmp_run_c.json --record $REC --port-input $PORT --out …` |
| `attr_window_check.json` | `$S/attr_window_check.py --record $REC --port-input $PORT --cache-dir $REC/s1port/cache --out …` |
| `ma_load_check.json` | `$S/ma_load_check.py $REC/surv_aoi_2026-06-28_s1.geojson $REC/s1port/run_c_after_review.json --ma-repo $MA --out …` |

## Review fixes (2026-10-10)

| File | Command |
|---|---|
| `parity_run_c_before_after.json` | `$S/parity_check.py $REC/s1port/run_c.json $REC/s1port/run_c_after_review.json --inputs $REC/s1port/cache/s1_vessel_inputs/2026-06-28_a6d730b7d5be7f7a.tif $PORT --out …` (`run_c.json` and `…a6d730b7d5be7f7a.tif` are the pre-review run and model input) |
| `parity_run_a.json` | `$S/parity_check.py $REC/s1port/run_a.json $REC/s1port/run_a_after_review.json --out …` |
| `memory_run_c.json` | `/usr/bin/time -l scripts/detect_vessels_s1.py …` from the cache, before and after; logs in `$REC/s1port/logs/` |
| `vessel_model_parity.txt` | `scripts/marine/vessel_model_parity.py` (dump per code tree, then `--compare`) |
| `mutation_check.json` | `$S/mutation_check.py --out …` |
