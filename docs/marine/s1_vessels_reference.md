# AI2 Sentinel-1 vessel detector: reference run and port dossier

Reference run of AI2's own `rslearn_projects` Sentinel-1 vessel pipeline on the
Skagen–Marstrand test scene (2026-06-28). It pins the exact input contract, the model
architecture and the checkpoint layout, so the detector can be ported to plain
torch/torchvision as `imint/analyzers/ai2_vessels.py` was for Sentinel-2.

Written 2026-10-10; review fixes the same day (section "Review fixes").

**Where the evidence is.**

- **Scripts:** `scripts/marine/s1_port/` (README there). Every check is re-runnable from
  the repository root with arguments.
- **Results:** `docs/marine/s1_port_results/`, small JSON files. Its README maps each file
  to the command that produced it. Every number in this document comes from one of them,
  except run times and RSS, which come from the logs in the record (`record:run1/predict.log`,
  `record:s1port/logs/`).
- **The record** (rasters, full run outputs; outside git): the directory the scripts take
  as `--record`. On the machine that produced it, it is `outputs/s1_reference_20261010/`
  of the main checkout (gitignored). `docs/marine/s1_port_results/large_files.sha256`
  lists the sha256 of every large file in it, relative to its root. Paths written as
  `record:<path>` below are relative to that root.

**Bottom line**

- **Port feasibility (verified, not assumed).** Both checkpoints load with `strict=True`
  into modules built only from `torchvision` (`swin_v2_b`, `FeaturePyramidNetwork`,
  `RegionProposalNetwork`, `RoIHeads`), plus a 10-line pooling head.
- **Detector match.** On a real 512 px patch, the plain detector matches rslearn's own
  model to 1.2e-7 in score and 3e-5 px in box coordinates.
- **Attribute match.** The plain attribute model reproduces all 107 reference attribute
  sets: max |Δlength| 2.4e-4 m, and 107/107 type matches.
- **Input contract (verified bit-exact).** The input is reproduced exactly (max diff 0)
  by a two-step rasterio/GDAL warp of the raw GRD DN (§c).

---

## a. Reproduction

| Item | Value |
|---|---|
| rslearn_projects | `9cfaf6e9a13fcb993c687b591eb3f71df7c144c5` (2026-10-08) |
| rslearn | `5780896aa5927e23b21f3a3be5be941b16411d1e` (git master, version 0.1.17). AI2's Dockerfile also installs rslearn from git master. |
| Python / torch / torchvision | 3.12.13 / 2.14.1 / 0.29.1 (CPU; AI2's Docker image is torch 2.7.0 + CUDA) |
| lightning / rasterio / GDAL | 2.6.6 / 1.5.2 / 3.12.2 |
| Machine | Apple M1, 8 cores, 16 GB, CPU only (`DISABLE_MPS=1`) |
| Detector ckpt | `imint/fm/ai2_vessels_s1/detect_best.ckpt`, sha256 `85f86043…33c5` |
| Attribute ckpt | `imint/fm/ai2_vessels_s1/attribute_best.ckpt`, sha256 `24b1e759…5913` |
| Infra GeoJSON | `record:inputs/marine_infrastructure.geojson`, sha256 `a40bebd5…6f63` (Satlas `latest.geojson`, last-modified 2026-07-27) |

Full checkpoint hashes and source URLs are in `imint/fm/ai2_vessels_s1/LICENSE.md`. Both
files match the GCS crc32c, and the detector also matches its md5.

AI2's environment (`$AI2ENV`, any directory) and the run, from the repository root:

```bash
cd $AI2ENV
git clone https://github.com/allenai/rslearn_projects.git
git -C rslearn_projects checkout 9cfaf6e9a13fcb993c687b591eb3f71df7c144c5
git clone https://github.com/allenai/rslearn.git
git -C rslearn checkout 5780896aa5927e23b21f3a3be5be941b16411d1e
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python "./rslearn[extra]" ./rslearn_projects
cd -

python scripts/marine/s1_port/pick_historical.py --out historical_selection.json  # AI2 rule, on MPC
python scripts/marine/s1_port/crop_grd.py $REC/inputs                            # radar-geometry crops
scripts/marine/s1_port/run_ai2_reference.sh $AI2ENV $REC > $REC/run1/predict.log 2>&1
```

`run_ai2_reference.sh` links the checkpoints into the `RSLP_PREFIX` layout the configs
expect and runs the unmodified CLI from the `rslearn_projects` root (the config paths are
relative):

```bash
export RSLP_PREFIX=$AI2ENV/project_data
export MARINE_INFRA_PATH=$REC/inputs/marine_infrastructure.geojson   # as AI2's Dockerfile does
export DISABLE_MPS=1                                                # Lightning auto -> CPU
python -m rslp.main sentinel1_vessels predict \
  --tasks '[{"image": {"vv": ".../inputs/target_vv.tif", "vh": ".../inputs/target_vh.tif"},
             "historical1": {"vv": ".../inputs/historical1_vv.tif", "vh": ".../inputs/historical1_vh.tif"},
             "geojson_path": ".../run1/reference.geojson", "json_path": ".../run1/reference.json",
             "crop_path": ".../run1/crops/"}]' \
  --score_threshold 0.7 --scratch_path $REC/run1/scratch_dir/
```

`--score_threshold` is a required argument of `predict_pipeline()`, even though the
docs example omits it. The value 0.7 is the API default (`SENTINEL1_SCORE_THRESHOLD`).

Re-executed with `run_ai2_reference.sh` in the same environment (2026-10-10): the
materialised model input is identical (0 pixels differ in all four rasters) and so are
the 107 detections, every field except the crop-file paths (`ai2_rerun_check.json`).

Verification scripts for this run, all in `scripts/marine/s1_port/`:

| Script | What it does | Result |
|---|---|---|
| `reproduce_inputs.py` | Bit-exact input contract check | `reproduce_inputs.json` |
| `port_feasibility.py` | Plain-torchvision detector vs rslearn's model (AI2 env) | `port_feasibility.json` |
| `port_feasibility_attr.py` | Plain-torchvision attribute model vs the reference | `port_feasibility_attr.json` |
| `tps_subset_check.py` | Geometric effect of cropping | `tps_subset_check.json` |
| `analyze_reference.py` | Output statistics | `reference_summary.json` |
| `mpc_scenes.py` | Catalogue facts behind the scene choice | `mpc_search_2026-06-27_28.json`, `mpc_inspect.json`, `mpc_footprint_vs_odata.json` |
| `dump_ckpt.py` | Checkpoint structure | `ckpt_detect.json`, `ckpt_attribute.json` |

## b. Scenes

All data comes from Microsoft Planetary Computer `sentinel-1-grd` (anonymous, SAS-signed).
Both scenes come from the same satellite, orbit direction and relative orbit.

| | Target | Historical (`historical1`) |
|---|---|---|
| Product | `S1D_IW_GRDH_1SDV_20260628T053135_20260628T053200_003432_0060CF_71E0.SAFE` | `S1D_IW_GRDH_1SDV_20260523T053133_20260523T053158_002907_004F80_AE87.SAFE` |
| Acquisition (UTC) | 2026-06-28 05:31:35.5 – 05:32:00.5 | 2026-05-23 05:31:33.1 – 05:31:58.1 |
| Orbit | descending, relative 66, absolute 3432 | descending, relative 66, absolute 2907 |
| Timeliness / orbit file | NRT-3h, PREORB | NRT-3h, PREORB |
| IPF | 004.03 | 004.03 |
| Full raster (px × lines) | 26483 × 16666 | 26731 × 16678 |
| Crop (radar geometry) | col 13250, row 0, 9276 × 8001 px, 40 GCPs | col 12033, row 0, 13371 × 12013 px, 77 GCPs |
| Crop footprint | 10.15–11.92 °E, 57.32–58.17 °N | 9.61–12.16 °E, 56.94–58.21 °N |

**Target choice.** This is the only 2026-06-28 scene covering the bbox (99.8 % cover).
The other two that touch it are an S1A scene with 6.1 % cover and the preceding S1D slice
with 0.2 % (`mpc_search_2026-06-27_28.json`). marine-awareness records no scene ID for its
σ⁰ file.

**Historical choice.** I applied AI2's own rule from `setup_dataset_with_scene_ids`:

- Search the 30-day window `[t0 − 60 d, t0 − 30 d]`, i.e. 2026-04-29 05:31:35 to
  2026-05-29 05:31:35 UTC.
- Keep products that intersect the target footprint.
- Take the one with the largest intersection area. **There is no orbit-direction filter
  in that query.** Direction is only enforced by the overlap criterion.

Result: S1D 2026-05-23, with overlap 0.99990 of the target footprint. That is **36 days**
earlier, not 60, because the rule's window only covers days 30–60.

- A near-tie: S1C 2026-05-22, same relative orbit, overlap 0.99986. AI2 looks up
  footprints through Copernicus OData, so it could pick either one.
- S1C 2026-04-28 (61 days) falls just outside the window.
- Training used a different window: `config.json` has `time_offset −90d`, `duration 60d`.

Candidates are listed in `historical_selection.json`.

**Crop rather than full scene (stated per the brief).** Each crop is snapped to the ESA
geolocation grid, at GCP rows/cols, and covers the whole bbox 10.50–11.70 °E,
57.55–58.05 °N.

- Only the GCPs inside the crop are kept, because the local-file pipeline derives the
  window from the GCP extent of the VV file.
- The TPS through 40 GCPs differs from the TPS through all 210 GCPs by mean 1.2 m and max
  7.1 m inside the bbox. Over the whole crop it is mean 2.6 m and max 16.5 m.
- DN values are copied bit-for-bit.

## c. Input contract (exact)

**Source.** The ESA GRD measurement TIFFs `measurement/iw-vv.tiff` and `iw-vh.tiff`:

- uint16 DN, i.e. **detected amplitude, uncalibrated**: no σ⁰, no thermal-noise removal,
  no dB.
- Radar geometry, with no CRS: 210 GCPs (a 21 × 10 grid) in EPSG:4326.

**Same DN as `aws_sentinel1`.** Evidence:

1. `rslearn/data_sources/aws_sentinel1.py` downloads
   `s3://sentinel-s1-l1c/GRD/{y}/{m}/{d}/IW/DV/{product}/measurement/iw-{band}.tiff`. It
   passes the file unchanged to `tile_store.write_raster_file`, which only warps it; there
   is no calibration or scaling anywhere on the path. The local-file path
   (`LocalFiles` → `RasterImporter.ingest_item`) calls the same `write_raster_file`.
2. The MPC asset is `…/GRD/2026/6/28/IW/DV/<same SAFE incl. CRC 71E0>/measurement/iw-vv.tiff`
   (`mpc_inspect.json`):
   - the same product and file;
   - uint16, with the ESA TIFF tags intact (`TIFFTAG_SOFTWARE=Sentinel-1 IPF 004.03`,
     original `TIFFTAG_DATETIME`);
   - plus `GRD_ORIGINAL_HEADER_SIZE` / `GRD_ORIGINAL_FOOTER_SIZE`, which indicate a
     lossless zstd-COG repack;
   - the same 210 GCPs, and the raster shape equals `s1:shape`.
3. The values are DN-like: median DN in the window is VV 40 / VH 22, p99 280 / 138.

**Not done.** A pixel-level comparison against the AWS object (requester-pays, excluded)
or against the CDSE original SAFE. That needs a CDSE user account or S3 keys; only the
PU-billed Sentinel Hub OAuth client exists here, and I did not use it.

**One known difference.** The MPC COG declares `nodata=0`; ESA's original declares none.
That changes step 1 only for pixels next to zero-DN pixels, i.e. at the scene borders.
In step 2 rslearn uses 0 as nodata either way.

**Geometry and resampling (two cubic warps).** This was reproduced **bit-exactly** for all
4 rasters by `scripts/marine/s1_port/reproduce_inputs.py` (`reproduce_inputs.json`).

1. **Ingest** (`DefaultTileStore.write_raster_file`, `gcp_transform_method: tps`):
   - `rasterio.vrt.WarpedVRT(src, crs=UTM(zone of gcps[0]), resampling=cubic, MAX_GCP_ORDER=-1)`.
     `-1` selects thin-plate spline.
   - Output resolution and extent are GDAL-suggested: 10.059 m for the target, 10.062 m
     for the historical.
   - Output stays **uint16**, so cubic over/undershoot is rounded and clamped, with
     nodata 0.
   - The UTM zone comes from the **first GCP** (pixel 0, line 0). In this crop that is
     32 for the target and 33 for the historical.
2. **Materialize** onto the window grid (`GeotiffRasterFormat.decode_raster`):
   - `WarpedVRT(tile, crs=window CRS, transform=window, resampling=cubic, src_nodata=0)`,
     still uint16, then cast to float32.
   - The window CRS is the UTM zone of the **centroid** of the scene's GCP bbox: EPSG:32632.
   - Resolution is exactly 10 m, north-up. The window bounds are
     `(floor, floor, ceil, ceil)` of the GCP bbox in 10 m pixel coordinates `(x/10, y/−10)`
     (local-file mode), so the pixel origin sits on integer multiples of 10 m.
   - Reference window: pixel bounds `[56781, −645167, 67589, −635319]`, i.e.
     10808 × 9848 px, origin (567810 E, 6451670 N).

**Full-scene production differs.** For the full scenes the first GCP lies at 14.1 °E,
which is zone 33, while the centroid is at 11.7 °E, zone 32 (`tps_subset_check.json`). So
AI2's production path warps both images GRD → EPSG:32633 → EPSG:32632, a double cubic
resample. In this crop the target was warped only once. A port should reproduce the
production chain: first-GCP zone, then centroid zone.

**Historical image.** It goes through its own TPS warp and is then materialized onto the
**target's** window grid. It must have the same orbit direction, so that the radar
geometry roughly aligns without terrain correction.

**Stacking.** `Concatenate` is applied along the **time** axis. The model input `image` is
CTHW = (2 [vv, vh], 2 [target, historical], H, W). Band order is **vv, vh**, and the
target always comes first.

**Normalisation.** Detector: `x / 250`, then clamp to `[0, 4]`, so DN ≥ 1000 saturates.
Only 0.02 % (VV) and 0.001 % (VH) of valid target pixels exceed 1000. Cubic undershoot
below 0 is clamped. The attribute model uses `x / 250` **without** the clamp.

**Tiling.** `AllCropsDataset` cuts 512 × 512 patches with overlap `round(512·0.1) = 51` px,
i.e. stride 461:

- The first patch is at the window origin. Interior patches follow at the stride.
- The last row and column are snapped to `bound − 512`.
- That gives 24 × 22 = **528 patches** here, with batch size 8.

**Model input dtype.** float32 in rslearn. Pixels outside the swath are 0; 70 % of the
target window and 91 % of the historical window is valid. The port keeps the integer DN
(uint16) and converts each 512 px patch and 128 px crop to float32 as it normalises it,
which gives the same values.

**Implications for the ImintEngine fetch.**

- `mpc_s1.py` / `cdse_s1_stac.py` resample straight into EPSG:3006 with bilinear, and
  `s1_shared` calibrates to σ⁰. Both are wrong for this model.
- The fetch needs a radar-geometry window plus GCPs, and then the two-step warp above.
  `scripts/marine/s1_port/reproduce_inputs.py` is a short, independent reference for it.
- The 128 px attribute windows are byte-identical to crops of the detector-window raster
  (214/214 band windows), so no second warp is needed.

## d. Detector architecture (port spec)

Config: `data/sentinel1_vessels/config.yaml`. Modules are listed with their checkpoint
prefixes.

1. **Backbone** (`model.encoder.0.encoder.model.`):
   - `torchvision.models.swin_v2_b()`, window 8, with
     `features[0][0] = Conv2d(2, 128, kernel_size=4, stride=4)`.
   - It is initialised from SatlasPretrain `sentinel1_swinb_mi.pth` (prefix remap
     `backbone.backbone.backbone.` → `encoder.0.encoder.model.`). The backbone was
     unfrozen at epoch 4, so the checkpoint contains all fine-tuned backbone weights; the
     SatlasPretrain file is not needed.
   - **One backbone with shared weights** is applied separately to target and historical.
   - Iterate `model.features[0..7]` and take the outputs of indices **1, 3, 5, 7**, permuted
     NHWC→NCHW. That gives 128/256/512/1024 channels at strides 4/8/16/32.
   - The final `norm` and the classification `head` are **not** used. Their weights are
     in the checkpoint but unused.
2. **Temporal combination** (`SimpleTimeSeries`, `groups=[[0],[1]]`, op max): each group
   holds one image, so max is the identity. Concatenate per scale on channels as
   `[target, historical]`, giving **256 / 512 / 1024 / 2048**.
3. **FPN** (`model.encoder.1.fpn.`): `torchvision.ops.FeaturePyramidNetwork([256, 512,
   1024, 2048], 256)` with **no extra block**, so 4 levels named `feat0..feat3`.
4. **RPN** (`model.decoders.detect.0.rpn.`):
   - `AnchorGenerator(((32,), (64,), (128,), (256,)), ((0.5, 1.0, 2.0),) * 4)` and
     `RPNHead(256, 3)`.
   - fg/bg IoU 0.7/0.3, batch 256, positive fraction 0.5, pre/post-NMS top-n 2000/2000
     (test), NMS 0.7.
5. **ROI heads** (`model.decoders.detect.0.roi_heads.`):
   - `MultiScaleRoIAlign(['feat0'..'feat3'], 7, sampling_ratio=2)`,
     `TwoMLPHead(256·49, 1024)`, `FastRCNNPredictor(1024, 2)`.
   - fg/bg 0.5/0.5, batch 512, positive fraction 0.25, `bbox_reg_weights=None`
     (→ 10, 10, 5, 5), `box_score_thresh=0.05`, box NMS 0.5, max 100 detections per patch.
6. **Transform**: none. There is no `GeneralizedRCNNTransform` normalisation or resize;
   images are only padded to a multiple of 32. `image_sizes` is the patch size.
7. **Classes**: `["unknown", "vessel"]`, so `num_classes=2` and label 1 means vessel.
8. **Score threshold**: `score > 0.7`, applied per patch in `DetectionTask.process_output`.
9. **Cross-patch merge**: `rslp.utils.nms.NMSDistanceMerger`, per class:
   - grid 64, `distance_threshold = 10` px (100 m) between **box centres**;
   - the higher score wins, and on equal score the lower index wins.
   - Here it kept 107 of 136 detections.
10. **Point**: `col, row = int(box centroid)` in window pixel coordinates, then
    `(col·10, row·−10)` → WGS84. The pixel corner is used, with no +0.5. `int()` truncates
    the **negative** row toward zero, so the row ends up rounded up by up to 1 px (≤10 m).
    I saw exactly this off-by-one when comparing per patch.

**Box size.** Predicted boxes are 17–30 px. Training converts points to boxes with
`box_size: 15`, so box extent is **not** a length estimate. Length comes from the
attribute model.

**Verified** (`port_feasibility.json`):

- A `torch.nn.Module` built from items 1–6, using only torchvision classes, loads the
  checkpoint with `strict=True` ("All keys matched").
- On the window patch at offset (2766, 5532) it gives 6 raw and 5 above-0.7 detections,
  the same as rslearn's `MultiTaskModel`.
- max |Δscore| = 1.2e-7 and max |Δbox| = 3.1e-5 px.

## e. Checkpoint structure

Both files are Lightning 2.6.1 checkpoints. Top-level keys: `epoch`, `global_step`,
`pytorch-lightning_version`, `state_dict`, `loops`, `callbacks`, `optimizer_states`,
`lr_schedulers`. There is **no `hyper_parameters`**, so the architecture must come from
the config (`ckpt_detect.json`, `ckpt_attribute.json`).

~~Loading needs `weights_only=False`~~ (corrected by the port, see "Port comparison"):
both files load with `torch.load(..., weights_only=True, mmap=True)` under torch 2.12,
in 0.2 s. The pickled optimizer and loop state contain only allow-listed types.

| | Detector | Attribute |
|---|---|---|
| epoch / global_step | 36 / 166130 | 29 / 510120 |
| Tensors / params | 479 / 105.9 M (455 float32, 24 int64 `relative_position_index`) | 497 / 97.1 M |
| Best metric (`ManagedBestLastCheckpoint.best_value`) | val_detect/mAP 0.503 | val_ship_type/accuracy 0.709 |

| Detector key (prefix → torchvision module) | Shape |
|---|---|
| `model.encoder.0.encoder.model.features.0.0.weight` (patch-embed conv, 2 ch) | (128, 2, 4, 4) |
| `model.encoder.0.encoder.model.features.1.0.attn.relative_coords_table` | (1, 15, 15, 2) |
| `model.encoder.0.encoder.model.features.1.0.attn.relative_position_index` (int64) | (4096,) |
| `model.encoder.0.encoder.model.features.7.1.attn.qkv.weight` | (3072, 1024) |
| `model.encoder.0.encoder.model.norm.weight` / `head.weight` (unused) | (1024,) / (1000, 1024) |
| `model.encoder.1.fpn.inner_blocks.3.0.weight` | (256, 2048, 1, 1) |
| `model.encoder.1.fpn.layer_blocks.0.0.weight` | (256, 256, 3, 3) |
| `model.decoders.detect.0.rpn.head.conv.0.0.weight` | (256, 256, 3, 3) |
| `model.decoders.detect.0.rpn.head.cls_logits.weight` / `bbox_pred.weight` | (3, 256, 1, 1) / (12, 256, 1, 1) |
| `model.decoders.detect.0.roi_heads.box_head.fc6.weight` | (1024, 12544) |
| `model.decoders.detect.0.roi_heads.box_predictor.cls_score.weight` / `bbox_pred.weight` | (2, 1024) / (8, 1024) |

Prefix remap for the plain port: `model.encoder.0.encoder.model.` → swin,
`model.encoder.1.fpn.` → FPN, `model.decoders.detect.0.rpn.` → RPN,
`model.decoders.detect.0.roi_heads.` → RoIHeads. After stripping the prefixes, the names
match torchvision exactly; no key renaming is needed.

| Attribute key | Shape |
|---|---|
| `model.encoder.0.model.features.0.0.weight` | (128, 2, 4, 4) |
| `model.decoders.<head>.0.conv_layers.0.0.weight` | (128, 1024, 3, 3) |
| `model.decoders.<head>.0.fc_layers.{0,1}.0.weight` | (512, 128), (512, 512) |
| `model.decoders.<head>.0.output_layer.weight` | (1, 512); `ship_type`: (9, 512) |

Here `<head>` is one of `length`, `width`, `speed`, `heading_x`, `heading_y`, `ship_type`.

Full key and shape lists: `record:detect_keys.txt`, `record:attribute_keys.txt`
(`scripts/marine/s1_port/dump_ckpt.py`).

## f. Attribute model

Config: `data/sentinel1_vessel_attribute/config.yaml`.

**Input.** A 128 × 128 px window centred on the detection, `[col−64, row−64, col+64, row+64]`,
on the same 10 m UTM grid:

- **target image only** (VV, VH), with no historical image;
- normalised with `x/250`, no clamp.

**Architecture.**

- **Backbone**: `swin_v2_b()` with the 2-channel patch-embed. Output layer `[7]` is
  `features[7]` permuted to NCHW: 1024 ch, 4 × 4.
- **Heads**: six independent `PoolingDecoder`s, each
  `Conv2d(1024, 128, 3, padding=1)` + ReLU → global **max** over H, W → `Linear(128, 512)`
  + ReLU → `Linear(512, 512)` + ReLU → `Linear(512, out)`.
- Regression heads output 1 value, with identity output (no sigmoid). `ship_type` outputs
  9 values, followed by softmax.

**Decoding.**

| Output | Formula |
|---|---|
| `length_m` | raw × 100 (`scale_factor 0.01`) |
| `width_m` | raw × 100 |
| `speed_kn` | raw × 100; trained on AIS SOG in knots |
| `heading_deg` | `90 − degrees(atan2(heading_y, heading_x))`, where training encoded `angle = 90 − cog`, `x = cos`, `y = sin` |
| `type` | argmax over `["cargo", "tanker", "passenger", "service", "tug", "pleasure", "fishing", "enforcement", "sar"]` |

**Heading is not wrapped.** It comes out in (−90, 270]; the reference range is −88.6 to
268.5. A port should apply `% 360`, or deliberately keep AI2's raw value.

**Verified** (`port_feasibility_attr.json`): a plain-torchvision module loads
`strict=True`. On the 107 reference crops it reproduces every attribute: max |Δ| of
length 2.4e-4 m, width 4e-5 m, speed 1.4e-5 kn and heading 6e-4°, and 107/107 types.

The pipeline drops detections whose 128 px crop cannot be materialized (scene edge); none
were dropped here.

## g. Reference result

`record:run1/reference.geojson` (sha256 `e181b156…78b7`). Its summary is
`reference_summary.json`.

**Counts.**

| Stage | Detections |
|---|---|
| Raw, score > 0.7, all patches | 136 |
| After distance-NMS | 107 |
| After edge drop and infra filter | **107** |
| Inside the bbox 10.50–11.70 °E, 57.55–58.05 °N | **76** |

**Scores.** min 0.705, p25 0.970, median 0.993, p75 0.994, max 0.998. In 0.05 bins from
0.70 the histogram is [4, 6, 2, 5, 5, 85], so the distribution is strongly bimodal toward
1.

**Attributes.**

| Attribute | Range (median) |
|---|---|
| length | 15–344 m (106) |
| width | 6–52 m (17) |
| speed | 0.2–19.4 kn (3.2) |

Types: cargo 35, tanker 35, fishing 24, passenger 11, service 2.

**Runtime.** M1 CPU, wall time 1296 s (21.6 min), peak RSS 2.6 GB (`record:run1/predict.log`).
By phase:

| Phase | Time |
|---|---|
| CLI and imports | ~45 s |
| prepare | 24 s |
| ingest (TPS warps) | 62 s |
| materialize | 55 s |
| detector inference, 528 patches | **973 s** (~1.84 s per 512² patch, both Swin passes included) |
| attribute materialize and inference, 107 crops | ~130 s |

A full scene would take roughly 1.5 h on this CPU. That is an estimate from about 2800
patches × 1.84 s, not measured.

**GeoJSON.** Each feature is a `Point` in lon/lat. Its properties are:

- `source` (`"sentinel1"`), `col`, `row`, `projection` (`{crs, x_resolution, y_resolution}`);
- `score`, `ts`, `scene_id`;
- `crop_fname`, `crop_fnames` (`{vv, vh}` PNG paths; DN clipped to 0–255, uint8);
- `longitude`, `latitude`;
- `attributes {length, width, speed, heading, vessel_type}`.

There is no `length_m`, `heading_deg` or `speed_kn` naming; the ImintEngine output must
map them. In local-file mode **`scene_id` and `ts` are `null`**, because they are only
filled when the pipeline is given a scene ID.

**Not measured.** Accuracy. This run is the port reference, not an evaluation. The AIS
hit-rate check against marine-awareness's 28–30 June AIS is a separate step. A detection
without an AIS match is a candidate, not a confirmed dark vessel.

## h. Infrastructure filter

- **Dataset**: Satlas marine infrastructure (Allen AI). It contains 39,582 points
  (26,738 `offshore_platform`, 12,844 `offshore_wind_turbine`), detected from Sentinel-2.
  - It is published under **ODC-BY** as monthly snapshots at
    `storage.googleapis.com/satlas-explorer-public/outputs/marine/latest.geojson`
    (also KML/SHP), with public, anonymous access.
  - It is fetched at runtime by default, or from `MARINE_INFRA_PATH`. AI2's Docker image
    bakes in a copy.
  - So it is reproducible from open data: it *is* open data.
  - Independent open alternatives, if wanted: EMODnet Human Activities (offshore
    installations, wind farms) and OSM `man_made=offshore_platform` /
    `generator:source=wind`.
- **Rule**: drop a detection if the haversine distance to *any* infra point is below the
  threshold. It is applied after the attribute model.
- **Threshold**: the docs say 50 m, but the **code default is 200 m**
  (`SENTINEL1_INFRA_DISTANCE_KM` defaults to `0.2` in `rslp/sentinel1_vessels/config.py`).
  The `NearInfraFilter` class itself defaults to 100 m. I ran with the pipeline default,
  **200 m**.
- **Effect here: none.** The nearest infra point to any detection is 1.52 km away, so all
  107 survive at 50, 100 or 200 m. Four infra points lie inside the crop's lon/lat bounding box.

## Surprises

1. **Local-file mode needs radar-geometry GRD with GCPs, not projected GeoTIFFs.**
   `setup_dataset_with_image_files` reads `raster.gcps` from the VV file. By code
   reading (not executed), it fails on a GeoTIFF that has a CRS but no GCPs, because
   `min()` is called on an empty GCP list. "Local VV/VH GeoTIFFs" therefore means the raw
   measurement TIFFs.
2. **Infra distance mismatch**: 50 m in the docs, 200 m in the code default, 100 m in the
   class default.
3. **The historical image is 30–60 days old at inference, but was 30–90 days old in
   training**, and there is no orbit-direction filter in the inference query. The "~60 d"
   in the brief is really "best overlap in days 30–60". Here that gave 36 days.
4. **The UTM zone comes from the first GCP at ingest**, but from the scene centroid in the
   window. Full scenes straddling 12 °E are therefore double-resampled
   (32633 → 32632), and both steps round to uint16.
5. **Heading is not wrapped to [0, 360)**, and `scene_id`/`ts` are null in local-file
   output.
6. **Box size is not ship size**: boxes are about 17–30 px by construction of the
   point→box training labels.
7. **The MPC `sentinel-1-grd` item_assets text says "radiometric terrain correction
   applied"**. That is wrong, copied from the RTC collection. The assets are raw uint16
   GRD DN in radar geometry.
8. **Two CLI details on a Mac.** `--score_threshold` is mandatory on the CLI. Lightning
   auto-selects MPS unless `DISABLE_MPS=1` is set. Monkey-patching
   `torch.backends.mps.is_available` breaks `torch._dynamo` imports in torch 2.14.

---

## Port comparison

The ImintEngine port (worktree branch `agent/te/claude/marine-s1-vessels`, 2026-10-10)
against the reference run above. Scripts: `scripts/marine/s1_port/`; results:
`docs/marine/s1_port_results/`; full run outputs and the fetch cache: `record:s1port/`.
Environment: the repo's `.venv` (Python 3.11.15, torch 2.12.0, torchvision 0.27.0,
rasterio 1.4.4 / GDAL 3.10.3) plus `pystac-client` 0.9.0 and `planetary-computer` 1.0.0
(pinned in `requirements.txt`), Apple M1 CPU.

**Bottom line**

- **Analyzer = AI2 on AI2's inputs.** Fed AI2's own materialised model input (run A),
  the port returns the same 107 detections: identical `(col, row)` and lon/lat,
  |Δscore| ≤ 2.0e-6, boxes within 3.1e-5 px, |Δlength| ≤ 2.4e-4 m, |Δheading| ≤ 3.5e-4°,
  107/107 types. Counts per stage match too: 136 raw, 107 after NMS, 107 final.
- **Fetch = rslearn's warp chain.** Run on AI2's own crop inputs and window, the
  production warp code reproduces all four of AI2's rasters bit for bit (0 of 106.4 M
  pixels differ each).
- **The reference inputs are not the full-scene chain.** The crop shortcut warped the
  target once (zone 32) and fitted the TPS through a GCP subset. The port warps
  33 → 32 through all 210 GCPs, as AI2 production does for this scene. Over the bbox the
  two inputs differ at speckle level: median |Δ| 1 DN, no bias, log-correlation
  0.989–0.996.
- **Production run on the bbox (run C).**
  - Counts: 77 detections inside the bbox, against AI2's 76; 73 match within 100 m.
  - Positions: median offset 10 m (one pixel), p90 22 m, max 51 m.
  - Attributes: |Δlength| median 1.0 m (1.1 %), p90 8 m; types agree on 68 of 73.
  - Coverage: all 64 AI2 detections with score ≥ 0.9 are found.
  - Misses, every one explained:
    - Six are small targets (19–36 m) that score 0.74–0.87 in one run and at best
      0.64–0.70 in the other, so the 0.7 cut decides.
    - The seventh is a second AI2 box at a patch edge, 111 m from a 227 m ship that both
      runs report.
- **Cost.** Bbox fetch 346 s cold (1.9 GB RSS), or ~2 s from the cache. Detection 337 s for
  208 patches (1.5 s per patch), against 1296 s for the reference run on its larger crop
  window. Peak memory footprint of a cached run: 3.17 GB (see "Review fixes").
- **GeoJSON for marine-awareness:** `record:surv_aoi_2026-06-28_s1.geojson`
  (`outputs/s1_reference_20261010/surv_aoi_2026-06-28_s1.geojson` in the main checkout).
  - 77 features, all inside the bbox; the 4 detections in the grid's UTM overshoot are
    dropped and counted (`n_outside_bbox: 4`).
  - marine-awareness's own `load_detections` (origin/main `2658af4`) reads all 77, with
    `score` and `pixel_count` (`ma_load_check.json`).

### What was built

| File | Role |
|---|---|
| `imint/fetch.py` `fetch_s1_vessel_scenes` | Target + historical GRD DN (uint16) on AI2's grid for a WGS84 bbox, with the AOI's acquisition time; cached. Chooses the pass (`orbit_state` / `pass_time_utc`) and mosaics its slices; `s1_vessel_passes` lists a date's passes. Puts the tie points on the sea surface (§ Geocoding). Owns the band keys and source string |
| `imint/geoid/` | EGM96 15' grid (package data) and `egm96_undulation` |
| `imint/data/mpc_s1.py` | `search_grd_items` / `sign_item` factored out of `fetch_s1_scene` (σ⁰ path unchanged, pinned by a test) |
| `imint/data/s1_shared.py` | `footprint_intersection_area` (footprint, not bbox, overlap) |
| `imint/analyzers/ai2_rslearn_blocks.py` | torchvision blocks shared by the S2 and S1 analyzers |
| `imint/analyzers/ai2_vessels_s1.py` | `AI2S1VesselAnalyzer` (registered as `ai2_vessels_s1`) + pure helpers |
| `imint/exporters/export.py` | `s1_vessel_feature_collection` / `save_s1_vessel_geojson` (marine-awareness seam) |
| `scripts/detect_vessels_s1.py` | CLI: fetch → analyzer → GeoJSON (`--orbit-state`, `--pass-time`) |
| `scripts/generate_marine_radar_showcase.py` | Radar sub-tab of the Öresund showcase |
| `tests/test_ai2_vessels_s1.py` + `tests/fixtures/ai2_s1_vessel_crop.npz` | Tests (§ Tests) |
| `tests/test_s1_geocoding.py` + `tests/fixtures/s1_geocoding/` | Geocoding against independent truth (§ Geocoding) |
| `scripts/marine/s1_port/`, `docs/marine/s1_port_results/` | This dossier's checks and their results |

The detector and attribute model are built from `ai2_rslearn_blocks.py`, the same blocks
the Sentinel-2 analyzer uses, and load both checkpoints with `strict=True`,
`weights_only=True`.

### Fetch design

**Scenes.** MPC `sentinel-1-grd`, anonymous and SAS-signed; no CDSE, no requester-pays.

- Pass: the IW GRDH VV+VH slices of the date whose footprints intersect the AOI, grouped
  by satellite and absolute orbit (one datatake, cut into ~25 s slices; MPC's duplicate
  listings of one product are dropped). One pass on the date is used as is. With several,
  `orbit_state` and/or `pass_time_utc` (within 2 min of the pass) must leave exactly one;
  otherwise the fetch raises and lists the passes. Taking one silently would date the
  detections to a pass the caller did not ask for. The Skagen bbox on 2026-06-28 has two:
  S1A rel. 168 at 05:23:51 clips 6 % of it, so the reference run now needs
  `--pass-time 05:31`.
- Target: the pass's slices ranked by AOI footprint cover, earliest start on a tie (the
  first is the former one-scene choice). Skagen: S1D 2026-06-28 (`…_0060CF`, SAFE
  `…_71E0`), AOI cover 0.998; the pass's other slice (`…T053110…`, cover 0.002) is not
  fetched because the first leaves no AOI pixel without data (Pass mosaic).
- Historical: AI2's rule, per slice, best footprint overlap among scenes starting in
  `[t0 − 60 d, t0 − 30 d]`, restricted to the slice's orbit direction (AI2 says the
  scenes must share it, but its query does not filter). Skagen: S1D 2026-05-23,
  descending, relative orbit 66, the same scene as the reference run.
- The MPC item geometry equals the CDSE OData `GeoFootprint` that AI2 reads (checked on
  the free OData catalogue; symmetric difference 1.7e-14 deg², `mpc_footprint_vs_odata.json`),
  and so does the start time.

**Grid.** AI2's scene-ID window: the UTM zone of the footprint centroid (EPSG:32632), and
`int()` of the footprint bounds in `(x/10, y/−10)` pixel units, giving
`[51737, −645423, 80357, −624992]`. This was cross-checked against rslearn's own
`get_utm_ups_projection` + `STGeometry.to_projection` in the reference venv. The AOI is
that window's lattice clipped to the bbox, `[58853, −643747, 66157, −637960]`, i.e.
7304 × 5787 px. Its pixels coincide with the reference window's.

**Pass mosaic.** AI2 processes one scene per window and never joins slices. When the AOI
spans a cut between slices of the chosen pass (2026-06-27 descending: 0.409 + 0.591), the
fetch:

- warps every slice onto its own AI2 scene window, in the first slice's UTM zone (its own
  zone in the common case, so the window is AI2's), so all lie on one 10 m lattice;
- builds the AOI grid on the union of those windows;
- gives each pixel the first-ranked slice whose target VV is valid, with the target and
  the historical both from that slice: a pixel's pair is the one AI2 would see for it;
- pastes the first slice whole, nodata included, so a one-slice pass gives exactly the
  one-scene input;
- skips (does not search or download) a later slice once no AOI pixel is left without
  data, and drops one that adds no pixel.

`provenance["pass"]` records the pass, how it was chosen, the other passes of the date,
each slice used (historical, window, AOI part, pixels supplied) and each slice skipped.
The GeoJSON carries it as `s1_pass`, with `scene_id` / `historical_scene_id` the slices
joined by `" + "`, `scene_start_utc` the first slice's start and `scene_end_utc` the last
slice's end.

**Acquisition time.** A slice spans ~25 s, so its start time can be up to ~155 m of travel
at 12 kn away from when the AOI was imaged, more than marine-awareness's 150 m match
radius. GRD lines are in azimuth-time order, so the fetch dates the AOI by the radar line
of its centre (inverse TPS): `scene_time_utc = start + (row_c / lines) · (end − start)`.
Skagen: line 3335 of 16666, 05:31:40.548 Z, 5.0 s after the slice start. The start and end
stay available as separate fields. In a mosaic the slice that supplied the AOI's centre
pixel dates it (the first slice if the centre has no data).

**Why a small warp is not enough.** GDAL's warp output depends on the request window,
not only on the geometry. A read is split recursively until each chunk fits the 64 MB
warp memory, and every chunk gets its own kernel scale and approximate-transformer
segments. Measured on the reference target crop (step 1, identical in GDAL 3.10.3 and
3.12.2; `warp_window_dependence.json`):

| Read of the same `WarpedVRT` | Pixels differing from the full read | Max \|Δ\| DN |
|---|---|---|
| window 700 × 500 | 46 % | 5 |
| window 1500 × 900 | 50 % | 40 |
| window 2048 × 1024 | 57 % | 9 |
| block by block (512 × 128) | 56 % | — |
| same window, exact transformer (`tolerance=0`) | 46 % | — |

So a warp of a crop, as in the reference shortcut, is not the full-scene result, even
with identical GCPs.

**Sparse full-size source.** The fetch downloads only a radar-geometry window around the
AOI, with a 32 px margin; 8121 × 6710 px per band for Skagen, out of 26483 × 16666 (px ×
lines). It then wraps that window in a VRT (built with ElementTree, so the GCP CRS is
XML-escaped) that has the full scene's size, all 210 GCPs and nodata 0. Both of rslearn's
warps run as whole-dataset reads over that VRT:

1. TPS over the sea-surface GCPs (§ Geocoding), cubic, into the first-GCP zone (33).
2. Cubic onto the full scene window (32).

GDAL therefore chunks exactly as in the full-scene run. Pixels outside the radar window
are nodata, and they reach no AOI pixel through either cubic kernel.

- A guard checks the inverse TPS against the forward TPS on the AOI outline, and raises
  if the round trip exceeds 80 m. Skagen: 0.000 m.
- Verified bit-exact three ways:
  1. On the reference crop as a stand-in scene, with the production functions: 0 of
     1.54 M AOI pixels differ from the non-sparse chain and from AI2's materialised
     raster, for a single warp (target VV) and a 33 → 32 double warp (historical VH)
     (`sparse_vrt_target_vv.json`, `sparse_vrt_historical1_vh.json`).
  2. On the real scene, which is the ground truth: the whole target VV GRD was
     downloaded (26483 × 16666 px, `record:s1port/full_target_vv.tif`) and run through
     the same warps without any sparsity. Inside the Skagen AOI, 0 of 42.3 M pixels
     differ from the fetch output (`full_scene_check.json`).
  3. A synthetic GCP raster in the unit tests. The test fails, with 194 pixels off, if
     the radar window is cut 8 px short.

**Cache.** The model input is cached as one uint16 GeoTIFF plus a JSON record (scenes,
slices, grid, AOI time, provenance); the key hashes the pixel-chain version (2), the
geocoding model (tie points, geoid grid sha256), the
record-format version (3), the date, the bbox and the chosen pass's slice ids, so another
pass on the same date and bbox never meets the entry. Format-2 entries (keyed by date and
bbox only, so either pass of a two-pass date) are not reused. The radar-window files are
named by the geocoding id (`vv_c…_r…_W×H_g<id>.tif`) because they embed their GCPs, and
scene headers without incidence angles are rebuilt, so a chain-1 cache downloads its
windows again (one header read and one 1.8 MB annotation per scene and polarisation). Every cache file is written to a temp file and moved into
place (`os.replace`), the record last, so a cache hit never reads a partial file.

**Cost (Skagen bbox, cold cache).** 346 s wall clock and 1.9 GB peak RSS for both scenes
and both polarisations. The time is dominated by the four full-request warps (~1 min per
raster, independent of AOI size); the download is 4 radar windows of 39–47 MB each (zstd),
against ~420 MB per band for full scenes. Cached requests return in a few seconds: they
run the target day's catalogue search (the pass is part of the key), no historical
search, download or warp.

### (a) Model inputs

| Raster | Port code on AI2's crop inputs: px ≠ AI2 | Port (full-scene chain) vs AI2 over the bbox: exact | median / p99 / max \|Δ\| DN | median rel. | log-corr. |
|---|---|---|---|---|---|
| target VV | 0 / 106.4 M | 27.6 % | 1 / 28 / 5192 | 2.6 % | 0.993 |
| target VH | 0 / 106.4 M | 39.3 % | 1 / 13 / 1156 | 3.7 % | 0.989 |
| historical VV | 0 / 106.4 M | 32.5 % | 1 / 20 / 3527 | 2.3 % | 0.996 |
| historical VH | 0 / 106.4 M | 45.2 % | 1 / 9 / 896 | 3.0 % | 0.993 |

Source: `cmp_inputs.json` (`scripts/marine/s1_port/compare_inputs.py --part1`).

- The valid footprint is identical (0 mismatching nodata pixels), and the mean signed
  difference is ≤ 0.002 DN.
- The differences are the crop shortcut's, not the port's:
  - **Target:** the crop's first GCP lies in zone 32, so it was warped once. The full
    scene's first GCP lies at 14.1 °E, so the full scene goes 33 → 32.
  - **Historical:** the TPS goes through 77 crop GCPs instead of 210.
  - **Both:** the chunk geometry differs between the crop window and the full window.
- Sub-pixel resampling changes on speckle give a few percent per pixel. The largest
  absolute differences sit on bright point targets.

### (b)–(e) Detections

Three runs isolate three causes. All of them use score > 0.7, distance NMS at 10 px, and
the reference run's Satlas file (sha256 `a40bebd5…`) at 200 m.

| Run | Input | Tiling | Isolates |
|---|---|---|---|
| A | AI2's materialised rasters, reference window 10808 × 9848 | AI2's, 528 patches | the port itself |
| B | port fetch over the reference window, masked to AI2's valid pixels | AI2's, 528 patches | the input chain |
| C | port fetch over the bbox, `scripts/detect_vessels_s1.py` | the AOI's own, 208 patches | production |

Runs A and B: `scripts/marine/s1_port/run_analyzer_on.py`; run C:
`scripts/detect_vessels_s1.py --run-json`. Full outputs: `record:s1port/run_{a,b,c}.json`
and, with the reviewed code, `record:s1port/run_{a,c}_after_review.json`; summaries
`cmp_run_{a,b,c}.json` (`compare_to_reference.py`).

| | A | B | C (inside the bbox) |
|---|---|---|---|
| raw / after NMS / final | 136 / 107 / 107 | 137 / 107 / 107 | 103 / 81 / 81 (77 in the bbox) |
| AI2 reference | 107 | 107 | 76 |
| matched (mutual nearest, ≤ 100 m) | 107 | 105 | 73 |
| offset: identical / median / p90 / max | 107 / 0 / 0 / 1e-9 m | 68 / 0 / 10 / 41 m | 26 / 10 / 22 / 51 m |
| \|Δscore\| median / max | 0 / 2.0e-6 | 3e-4 / 0.18 | 0.002 / 0.16 |
| \|Δlength\| median / p90 / max | 0 / 2e-5 / 2.4e-4 m | 0.8 / 3.9 / 27 m | 1.0 / 8.0 / 43 m |
| \|Δlength\| / length, median / p90 | 0 / 0 | 0.8 % / 3.9 % | 1.1 % / 4.9 % |
| \|Δwidth\| median / max | 0 / 1.5e-5 m | 0.10 / 2.8 m | 0.18 / 2.1 m |
| \|Δspeed\| median / max | 0 / 6.7e-6 kn | 0.03 / 1.3 kn | 0.03 / 0.65 kn |
| \|Δheading\| median / p90 | 1e-6 / 2e-5° | 0.8 / 17° | 1.5 / 42° |
| heading flips > 150° | 0 | 1 | 5 |
| type agreement | 107 / 107 | 99 / 105 | 68 / 73 |
| AI2-only / port-only | 0 / 0 | 2 / 2 | 3 / 4 |

Notes on the table:

- **Offsets** are measured between points in AI2's convention, `int()` of the box centre.
  A box centre that moves by a fraction of a pixel across an integer moves the point by
  10 m.
- **Heading flips** are bow/stern ambiguity. In C, two of the five are near-stationary
  (≤ 0.3 kn); three are moving at 3.4–6.9 kn.
- **Length outliers** are small boats. The largest is a fishing boat at the same position
  and score, 30.5 m (AI2) against 57.9 m (B) and 73.8 m (C).
- **Type disagreements** are cargo ↔ tanker on 84–281 m ships (4 cases) and passenger ↔
  fishing on a 19–22 m boat (1 case). These are the classes the attribute model confuses
  (validation accuracy 0.709).

**Every unmatched detection.** `scripts/marine/s1_port/diag_unmatched.py` re-runs the
detector without the 0.7 cut on every patch that contains the point, in both tilings, and
reports the best score within 10 px (`diag_run_b.json`, `diag_run_c.json`). The
patch-edge distance is given in parentheses.

Run B (same tiling, so only the inputs differ):

| lon, lat | Seen by | Score | Other run's best | Cause |
|---|---|---|---|---|
| 10.6688, 57.8234 | AI2 | 0.713 | 0.693 | input chain; 0.7 crossing (matched again in C) |
| 11.6559, 57.7465 | AI2 | 0.741 | 0.578 | input chain |
| 11.6977, 57.6692 | port | 0.799 | 0.685 | input chain |
| 10.4795, 58.0641 | port | 0.707 | 0.674 | input chain; 0.7 crossing (outside the bbox) |

Run C (production, bbox):

| lon, lat | Seen by | Length | Score | Other run's best | Cause |
|---|---|---|---|---|---|
| 11.6559, 57.7465 | AI2 | 36 m | 0.741 | 0.698 (161 px), 0.642 (12 px) | input chain, as in B |
| 11.6221, 57.6356 | AI2 | 19 m | 0.778 | 0.696 (31 px), 0.655, 0.489, 0.463 (≤ 20 px) | tiling: matched in B, but here every covering patch has it within 31 px of its edge |
| 10.6790, 57.6665 | AI2 | 214 m | 0.874 | none ≥ 0.05 (196 px) | AI2 tiling artefact: a second box 111 m from AI2's 0.994 detection of a 227 m ship. It comes from the AI2 patch where the point is 8 px from the edge; AI2's other covering patch has no box there either. The port reports the ship once, at 0.995 |
| 11.5875, 57.6084 | port | 21 m | 0.869 | 0.637 (132 px) | small target; input and tiling |
| 11.6977, 57.6693 | port | 26 m | 0.777 | 0.685 (89 px) | input chain (port-only in B too); 67 px from the AOI edge |
| 10.6435, 57.7224 | port | 21 m | 0.768 | 0.652 (197 px) | tiling: 14 px from a patch edge in the port tiling (0.565 in its other patch) |
| 11.6950, 57.6646 | port | 23 m | 0.740 | 0.648 (75 px) | small target at the threshold; 81 px from the AOI edge |

None of these is a confident detection that the other run lacks. Each one has a best
counterpart score of 0.58–0.70 in the other run (B and C together), except the extra box
at AI2's patch edge. The S1 model is simply this sensitive at 0.7 on 20–40 m targets, and the
historical-image chain and the tiling are enough to move them across it.

### Attribute windows

- **What AI2 does.** It materialises every 128 px attribute window with its own cubic
  warp from the step-1 raster.
- **What the analyzer does.** It crops the detector-window raster instead, which keeps the
  step-1 raster out of the analyzer.
- **Why it matters here.** In the reference run the two were byte-identical (§c,
  214/214), because the target was warped once. With the full-scene 33 → 32 chain they
  are not.
- **Measured** on the 77 AI2 positions inside the AOI (`attr_window_check.json`):
  - Pixels: 84 % equal, mean |Δ| 0.21 DN.
  - Attributes: |Δlength| median 0.11 m (0.1 %), max 4.5 m (7 %); |Δwidth| ≤ 0.37 m;
    |Δspeed| ≤ 0.25 kn; |Δheading| median 0.13°, max 8.9°; 77/77 types.
- **Decision.** The crop stays. Its effect is an order of magnitude below the input-chain
  effect in run B.

### Deliberate differences from AI2, and known residuals

**Deliberate differences:**

- **Heading** is wrapped to [0, 360), never 360: AI2 reports 90 − atan2, in (−90, 270];
  `np.mod` of a tiny negative angle and rounding to 0.1° (359.96 → 360.0) are both mapped
  to 0.
- **Historical scene:** candidates are restricted to the target's orbit direction. AI2
  states that requirement but does not filter on it; the pick for Skagen is unchanged.
- **Pass mosaic:** slices of one pass are joined on one lattice (Fetch design); AI2 would
  run each slice on its own window. A date with several passes is refused unless the
  caller names one.
- **Geocoding (chain 2):** the GCPs are moved onto the sea surface before the warp
  (§ Geocoding). AI2/rslearn warp with the ellipsoid-height GCPs and place every ship
  41–56 m toward the satellite; the port's pixels are therefore no longer bit-identical
  to AI2's, and AI2 parity holds in radar coordinates.
- **Infrastructure filter:** 200 m, AI2's code default.
  - The Satlas `latest.geojson` is downloaded once, cached in
    `~/.cache/imint/satlas_marine_infrastructure/` and recorded by sha256. It is only
    loaded when there is at least one detection.
  - If it cannot be loaded, `metadata.infra_filter` says `applied: false` with the
    reason, the analyzer logs a WARNING, and the GeoJSON carries the same block.
- **No water mask:** Sentinel-1 has no SCL, and AI2 applies none. Land clutter is left to
  the historical image and the infrastructure filter, as in AI2.
- **GeoJSON:**
  - Each detection is the detector box as a closed WGS84 polygon with a
    counter-clockwise exterior ring (RFC 7946), as for Sentinel-2.
  - marine-awareness's `load_detections` takes its centre (checked: within 6 mm).
  - AI2's reported point is the truncated pixel corner, median 7.5 m and max 12.8 m from
    that centre.
  - The box is a detection footprint (~170–300 m), not a hull: `pred_length_m` is the
    length. `pixel_count` is the box area in pixels, the name marine-awareness reads.
  - Only detections whose AI2 point lies inside the requested bbox are written;
    `n_outside_bbox` counts the rest (the UTM grid overshoots a WGS84 bbox at its corners).
  - `scene_time_utc` is the AOI's acquisition time (Fetch design); `scene_start_utc` and
    `scene_end_utc` are the slice's (first start and last end for a mosaic, which also
    adds `s1_pass`).

**Known residuals:**

- **Attribute crops at the AOI edge.** Within 64 px of the edge they are zero-padded,
  where AI2 reads real pixels. `metadata.attr_crops_padded` counts them (3 in run C).
  Fetch with a margin if edge attributes matter.
- **MPC versus AWS.**
  - MPC declares nodata 0; ESA's original on AWS declares none. That changes swath
    borders, and possibly GDAL's chunking, since a validity mask costs warp memory.
  - The pipeline here is exactly AI2's chain on MPC data. It was not verified against
    the requester-pays AWS object.
- **Slice seam.** A mosaicked AOI has a thin line of nodata along the cut, where the
  cubic kernels of neither slice reach, and its historical changes date across the cut
  (each slice has its own). Patches across the seam therefore see two (target,
  historical) pairs; AI2 would see one slice per run. `valid_frac_target` and
  `valid_frac_historical` (analyzer metadata and GeoJSON) show the coverage, and the
  analyzer logs a WARNING when the historical scene covers less of the AOI than the
  target. Skagen: 1.0 / 1.0.
- **GDAL version.** Bit-exactness was shown for GDAL 3.10.3 (port) and 3.12.2 (reference
  environment). Another GDAL may chunk differently, at the speckle level of table (a);
  the fetch records its GDAL version in the provenance.
- **Wording.** A detection without an AIS match is a candidate, not a confirmed dark
  vessel.

### Pass choice check (2026-10-20)

The pass selection and slice mosaic were run on real data from cached radar windows
(GDAL 3.10.3, CPU). The 2026-06-27 references are the manual run of the same day: each
slice pinned and fetched alone, detected alone, and the detections joined by hand. Its
caches and layers stayed outside git.

- **2026-06-28 Skagen bbox, `--pass-time 05:31`:** target and historical are
  bit-identical to the format-2 one-scene entry (42 268 248 px), as are the grid, the
  AOI time (05:31:40.547748 Z) and the radar windows. The pass's sliver slice
  (`…T053110…`) is skipped without a search or download. 392 s, peak footprint 2.8 GB.
- **2026-06-27 descending (`--orbit-state descending`):** the mosaic equals the two
  manual per-slice inputs, combined by the rule, on all 42 268 248 px:
  - 25 824 691 px come from `…T053936…` (rank 0, historical S1D 05-16) and 16 443 115 px
    from `…T053911…` (historical S1C 05-15), on the same EPSG:32632 lattice (origin
    588530, 6437470);
  - 442 px of nodata lie along the seam, and 2 748 target pixels have no historical;
  - AOI time 05:39:37.623515 Z, unchanged.

  One detector run over 208 patches gives 90 raw, 66 after NMS, 1 removed by the
  infrastructure filter, 62 in the bbox and 3 outside. All 62 match the hand-joined layer
  (mutual nearest, max offset 0.09 m, max |Δscore| 0.0006). Fetch 638 s, detection 441 s.
- **2026-06-27 ascending:** the input is bit-identical to the manual one, and the 71
  features in the bbox are identical to its raw output. The marine-awareness layer holds
  70 because one patch-seam duplicate (score 0.9202) was removed by hand.
- marine-awareness `load_detections` reads 62 and 71.

### Geocoding (chain 2, 2026-10-20)

**The error.** Chain 1, like AI2/rslearn, placed every ship 41–56 m toward the satellite.
Measured against AIS hull centres (antenna position + true heading + A/B/C/D) on slow
anchored ships, the range residual was +55.1 m (2026-06-27 05:39 S1C descending), +56.7 m
(06-27 16:53 S1C ascending) and +44.9 m (06-28 05:31 S1D descending). The cause: ESA's GRD
GCPs (the annotation's geolocation grid) are zero-Doppler solutions at the GCP's own
ellipsoidal height, z ≈ 0 over the sea, but the sea lies on the geoid, N = 36–38 m higher
off Skagen. A target that is h higher appears h / tan θ closer to the satellite, and GDAL's
GCP thin-plate spline is 2-D (it drops Z), so no height reaches the warp. Diagnosis:
`2026-10-20_s1_offset_diagnosis/offset_diagnosis.md` (marine-awareness research).

**The fix: tie points on the sea surface.** One code path, before every GCP consumer
(radar window, AOI dating, the window TIFF, the sparse VRT and so the warp), target and
historical scenes alike (`_s1_sea_surface_gcps`):

- N is the EGM96 undulation at the GCP (`imint/geoid/us_nga_egm96_15.tif`, PROJ's grid,
  2.7 MB package data, bilinear; it matches PROJ `vgridshift` to < 1e-9 m and EGM2008 to
  0.22 m over the AOI). A missing grid raises; N is never silently 0.
- The GCP moves (N − z) / tan θ along the ground direction of increasing pixel on its GCP
  row, which is away from the right-looking satellite. The direction is the central
  difference of the row's neighbouring GCPs in the GCP's tangent plane (geodesic). Then
  z = N. θ is the annotation's `incidenceAngle` of that grid point.

Land GCPs above the geoid move toward the satellite (up to ~800 m in the Norwegian
mountains), so the TPS describes sea-level targets everywhere; land in the backdrop is
not terrain-corrected (it never was). The sign of the shift flips with the pass, so no
constant per-layer correction can replace this.

**Incidence angles.** `_s1_scene_header` reads the product annotation and stores
`gcp_incidence_deg` in the cached header; headers without it are rebuilt. The annotation
grid must equal the TIFF GCPs in count, order, pixel/line and lon/lat/height (to 1e-9° /
1e-6 m), or the fetch raises; the four 2026-06-27/28 products pass (measured on two:
1e-14°). MPC's `schema-product-{pol}` asset is not the product annotation for these products: it
points at `annotation/rfi/rfi-iw-{pol}.xml` (the RFI report, 41 KB, no geolocation grid).
The fetch takes the measurement's sibling `annotation/iw-{pol}.xml` (1.8 MB) with the
same SAS token and validates it.

**Accuracy of the first-order shift.** Against an independent range-Doppler solver
(orbit state vectors + slant-range time of each grid point, re-solved at h = N): sea GCPs
≤ 0.14 m, land GCPs ≤ 1.5 m, on the four products.

**Provenance and cache.** `S1_VESSEL_CHAIN_VERSION` = 2. `provenance["geocoding"]`
records the tie-point model and the geoid grid and its sha256;
each `radar_windows` entry records `gcp_shift_m` (min / median / max over the sea GCPs,
|z| < 1 m, and over all GCPs; + = away from the satellite). The cache key and the
radar-window file names include the geocoding model.

**AI2 parity is now in radar coordinates.** The port's model input is no longer
bit-identical to rslearn's: the raster itself moves, by the sea-surface shift. The
detector sees the same
radar data, so AI2 parity is checked by mapping our detections through the sea-surface
TPS⁻¹ and AI2's through the TPS⁻¹ of its own 40-GCP crop, and comparing radar
coordinates. The earlier parity checks in this dossier (tables (a)–(e), run C) are chain 1
and remain valid as pixel-chain checks; they never were geolocation truth, because the
reference carries the same error.

**The lesson.** No test compared S1 positions with independent truth. The checks were TPS
round trips, AI2 parity on identical (equally wrong) pixels and self-consistency, so a
41–56 m error that flips with the pass passed all of them. `tests/test_s1_geocoding.py`
now does:

1. sea-surface GCPs vs the range-Doppler oracle at h = N (≤ 0.5 m), each shift = (N − z) /
   tan θ pointing away from the satellite (E < 0 descending, > 0 ascending), and a guard
   that the raw sea GCPs are at |z| < 1 m (catches a change of ESA's convention);
2. frozen radar positions of 29/30/30 slow anchored ships, geocoded through the
   pipeline's own TPS, vs their AIS hull centres: |median| ≤ 10 m in range and along track
   per layer, asc − dsc within 15 m; the same fixture gives +45 to +57 m on chain 1's GCPs
   (asserted, so the check has teeth);
3. `@pytest.mark.slow`, with `IMINT_S1_E2E_CACHE` set to a local S1 cache (the MPC
   catalogue search still needs network): fetch + detector for 2026-06-28 against the same
   ships, and with `IMINT_S1_AI2_REFERENCE_DIR` the AI2 parity in radar coordinates.

**On real data (2026-10-20).** The three Skagen–Marstrand layers re-run through chain 2
(cold cache, `scripts/detect_vessels_s1.py`, bbox 10.5–11.7 E, 57.55–58.05 N), against the
AIS hull centres of the same slow anchored ships (mutual nearest ≤ 150 m; range + = toward
the satellite):

| layer | n | range median | along-track median | chain 1 range | layer moved (median dE, dN) |
|---|---|---|---|---|---|
| 06-27 05:39 S1C dsc (2 slices) | 29 | +3.0 m | −4.1 m | +55.1 m | (−47.5, +15.0) m |
| 06-27 16:53 S1C asc | 30 | +6.2 m | −4.3 m | +56.7 m | (+47.7, +4.8) m |
| 06-28 05:31 S1D dsc | 30 | +5.1 m | +0.6 m | +44.9 m | (−35.5, +10.0) m |

- All pass (|median| ≤ 10 m, n ≥ 20). Same-day asc − dsc range medians: +3.3 m; the same
  ship's E-W residual, asc − dsc: −10.6 m (n 21; chain 1 −111.8 m).
- The layers move west on descending and east on ascending passes: the sea-surface shift
  (41–57 m at the ships).
- AI2 parity in radar coordinates, 06-28: 76 pairs, median |Δ| 0.76 px, mean vector
  0.13 m. In map coordinates the same pairs are 40.6 m apart, by design.
- Detection sets change only at the margins: 62 → 64, 70 → 65 and 77 → 77 detections in
  the bbox; every changed one scores 0.70–0.83 or lies at the bbox's east edge, matched
  pairs differ by a median 0.001–0.002 in score, and all AIS-matched slow ships are still
  detected.
- Details, per-ship residuals and the layers: marine-awareness research
  `2026-10-20_s1_offset_diagnosis/fix_verification.md` and `fix_runs/`.

### Review fixes

A review of the port (verdict approve-with-fixes) led to these changes, each verified:

- **Results unchanged.** Run C re-run with the reviewed code from the cached radar
  windows (new cache record format, so the model input was recomputed): all 81 regions
  are identical to the pre-review run (boxes, scores, points, lon/lat, attributes), and
  the recomputed model input is bit-identical to the earlier one (pixels, dtype, grid)
  (`parity_run_c_before_after.json`; `cmp_run_c.json` is unchanged). Run A (AI2's
  inputs, float32) re-run likewise (`parity_run_a.json`); run B was not re-run, as run C
  already covers the uint16 path.
- **Memory.** The fetch keeps uint16 and the analyzer reads the bands in place,
  normalising per patch and per attribute crop. Before, the fetch returned float32 and
  the analyzer added float32 stacks and normalised copies. By array arithmetic that is
  48 bytes per pixel at the detector's peak against 8 now, i.e. ~28 GB against ~4.7 GB
  for the full 28620 × 20431 scene window. Measured on cached run C: peak memory
  footprint 4.86 → 3.17 GB (`memory_run_c.json`).
- **Shared blocks.** `ai2_rslearn_blocks.py` holds the torchvision blocks both analyzers
  use. Both S2 checkpoints and both S1 checkpoints, loaded `strict=True`, give
  bit-identical outputs on fixed random inputs before and after the move (40 arrays,
  `vessel_model_parity.txt`, `scripts/marine/vessel_model_parity.py`).
- **Tests reach the risky paths.** Moving the historical window forwards
  (`lo = t0 + 60 d`) or offsetting patch boxes by the row origin in x both fail the suite
  now (`mutation_check.json`); each passed all 40 tests before.

### Tests

`tests/test_ai2_vessels_s1.py`, `tests/test_s1_geocoding.py`, `tests/test_geoid.py` and
`tests/test_mpc_s1.py`, with the synthetic GRD writer shared in `tests/s1_synthetic.py`
and the range-Doppler oracle in `tests/s1_rd_oracle.py`; run with the repo `.venv`:

```bash
.venv/bin/python -m pytest tests/test_ai2_vessels_s1.py tests/test_s1_geocoding.py \
    tests/test_geoid.py tests/test_mpc_s1.py -q
```

**Geocoding** (`tests/test_s1_geocoding.py`, `tests/test_geoid.py`; § Geocoding): the
sea-surface GCPs against the range-Doppler oracle, anchored ships against AIS hull
centres (and failing on chain 1's GCPs), the annotation checks and header rebuild, the
EGM96 helper against PROJ `vgridshift`, and the slow opt-in end-to-end run. The synthetic
chain tests compare the sparse VRT with the full raster carrying the same sea-surface
GCPs (`sea_surface_copy`).

**Pure logic:**

- normalisation (uint16 = float32 input), heading wrap (never 360) and attribute decoding;
- AI2 tiling (528 patches on the reference window; full coverage; column-major order);
- the detector loop with a fake detector on a 1000 × 1500 image: global box coordinates
  of every patch, including the snapped last row and column, duplicates merged by NMS,
  uint16 and float32 input giving the same detections;
- distance NMS, including inclusive 10 px, ties and non-chaining;
- the AI2 point, the grid guard, and the infrastructure filter (applied, failing, and not
  loaded without detections);
- coverage fractions and the WARNING when the historical scene covers less;
- the analyzer skip, grid check and registry;
- the GeoJSON schema: closed counter-clockwise rings, required properties, no `crs`,
  `pixel_count`, UTC times, bbox filtering with `n_outside_bbox`, headings after rounding,
  and the `load_detections` centroid.

**Fetch:**

- scene selection with the catalogue faked: the pass (the only one; two passes refused
  with a listing; `orbit_state` and `pass_time_utc` forms, a selector matching none, two
  passes in one direction, MPC duplicates, invalid selectors before any search), slice
  ranking (cover, earliest start on a tie) and historical (AI2's window, orbit
  direction, overlap over relative orbit, tie), including the exact search requests;
  the 2026-06-27 items are the real MPC ids, times and footprints;
- the UTM-zone and window rules (literal values checked against rslearn), AOI clipping,
  a pinned literal cache key that differs per pass, and the AOI acquisition time;
- the sparse VRT against a full synthetic 33 → 32 scene, the VRT XML with a quoted WKT
  CRS and an unsupported dtype, and atomic writes;
- `fetch_s1_vessel_scenes` end to end, offline: uint16 output equal to the full-scene
  chain, the AOI time against an independent inverse TPS, the pass record, the cache
  written without temp files and answered without historical search, read or warp;
- a pass of two synthetic slices cut from one scene (cut on either side of the AOI
  centre): every mosaic pixel equal to its slice's own full-raster chain, each
  historical from its target's slice, the AOI time equal to the whole scene's within
  2 ms, the slices and joined ids recorded; and a second slice that is never searched or
  downloaded once the first covers the AOI;
- the CLI passing `--orbit-state` / `--pass-time` to the fetch and reporting a refusal;
- the cache write → read round trip (arrays, dtype, Affine, scene records, provenance);
- chain 2: the provenance's geocoding model and per-window GCP shifts, and a cache key and
  window file names that change with the geocoding model.

**σ⁰ path** (`tests/test_mpc_s1.py`): `mpc_s1.fetch_s1_scene` offline after the
STAC-helper refactor: the search
request, signed `/vsicurl/` reads, σ⁰ = DN² / LUT² (and dB), nodata rejection, and which
failures return `None` and which raise.

**Model:** the 192 px fixture (103 KB, Copernicus Sentinel-1 data) around the 241 m
tanker at `(60505, −641058)`, as uint16 DN:

- position within 1 px of AI2's;
- length within 3 % (243.0 against 241.4 m), width within 5 %, speed within 0.5 kn and
  heading within 3°;
- type matches.

It skips if the checkpoints are absent.

**Other test runs:**

- The 24 existing test files that touch the changed modules (fetch, engine, export,
  `mpc_s1`, `s1_shared`, utils, repo hygiene, integration): 459 passed, 5 skipped.
- The full suite: 2597 passed, 12 skipped, and 4 failed, all in `test_slu_markfukt.py`.
  Those are live calls to the SLU ImageServer, which answers with an HTML page today; the
  same 4 failed before this change.
- Chain 2 (2026-10-20): the full suite 3743 passed, 15 skipped (environment: no DES login,
  gdal CLI, pydensecrf, local NFI data; the slow geocoding run without its env var), 0
  failed. The slow `test_s1_pipeline_end_to_end_06_28` passed separately on a real cache
  (449 s).
