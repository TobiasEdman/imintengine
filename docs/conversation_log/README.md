# ImintEngine — Conversation Log

This directory contains Claude Code conversation logs split into topic-based markdown files.

## Topics

| # | File | Topic |
|---|------|-------|
| 1 | [01_demo_report.md](01_demo_report.md) | Demo Report — Multi-scenario HTML report (Brand/Marin tabs, Leaflet, COT fixes) |
| 2 | [02_pipeline_fundamentals.md](02_pipeline_fundamentals.md) | Pipeline Fundamentals — Grid, NMD, DES, quality gates, data format, resumability, dashboard, MPS |
| 3 | [03_yolo_vessel_detection.md](03_yolo_vessel_detection.md) | YOLO Vessel Detection — Fine-tuning, L2A vs TCI, on-board simulation |
| 4 | [04_slu_sjokort.md](04_slu_sjokort.md) | SLU GET Sjokort — Nautical charts via SLU API, S-57 data, rendering |
| 5 | [05_vessel_heatmap.md](05_vessel_heatmap.md) | Vessel Heatmap — Multi-temporal pipeline, cloud thresholds, full scene |
| 6 | [06_ai2_vessel_detection.md](06_ai2_vessel_detection.md) | AI2 Vessel Detection — Google/Nature 2023 paper, rslearn repo |
| 7 | [07_change_detection_alignment.md](07_change_detection_alignment.md) | Change Detection — Grid alignment, sub-pixel shift, Affine transforms |
| 8 | [08_grazing_lpis_analysis.md](08_grazing_lpis_analysis.md) | Grazing & LPIS — Betesmark analysis, NMD stats, prediction-colored overlays |
| 9 | [09_wordpress_deployment.md](09_wordpress_deployment.md) | WordPress Deployment — Integration, CSS theming, DES site, licensing |
| 10 | [10_shoreline_kustlinje.md](10_shoreline_kustlinje.md) | Shoreline / Kustlinje — CoastSeg models, ShorelineAnalyzer, NDWI extraction |
| 11 | [11_densification_class_balance.md](11_densification_class_balance.md) | Densification & Class Balance — SCB tatort, sea cells, sumpskog, class distribution |
| 12 | [12_training_results_improvements.md](12_training_results_improvements.md) | Training Results — 19-class baseline (27% mIoU), 10-class schema, focal loss, backbone unfreezing |
| 13 | [13_auxiliary_channels_fusion.md](13_auxiliary_channels_fusion.md) | Auxiliary Channels — Height, volume, basal area, diameter, DEM, AuxEncoder late fusion |
| 14 | [14_cdse_dual_source_fetch.md](14_cdse_dual_source_fetch.md) | CDSE & Dual-Source — Copernicus backend, DN offset, batch benchmarks, seasonal fetch |
| 15 | [15_colonyos_deployment.md](15_colonyos_deployment.md) | ColonyOS Deployment — Docker, M1 Max setup, job spec debugging, parallel execution |
| 16 | [16_s2_process_api_fetch.md](16_s2_process_api_fetch.md) | S2 Process API — Migration from openEO to HTTP POST, 1-stage cloud filter, ColonyOS fixes, 4,381 jobs |
| 17 | [17_training_runs_comparison.md](17_training_runs_comparison.md) | Training Runs — Base vs AuxEncoder comparison, 43.27% mIoU best result, checkpoint inventory |
| 18 | [18_vpp_phenology_pipeline.md](18_vpp_phenology_pipeline.md) | HR-VPP Phenology — PPI, SOSD/EOSD/LENGTH, Sentinel Hub BYOC collection, integration with AuxEncoder |
| 19 | [19_marine_commercial_showcase.md](19_marine_commercial_showcase.md) | Marine Commercial Showcase — Kalmarsund shipping tab, AI2 vessel detection, coordinate convention fix, grazing statistics |

## Sources

Files 01-10: Original conversation log (`Data preperation and training pipeline.txt`, ~1 MB)
Files 02, 11-15: Second pipeline conversation (`Data preperation and training pipeline_2.txt`, ~585 KB)
Files 16-18: Third pipeline conversation (S2 Process API, training results, VPP phenology)

The conversations cover satellite imagery analysis with ImintEngine, including Sentinel-2 data processing, LULC classification with Prithvi-EO-2.0, auxiliary data fusion, distributed computing with ColonyOS, vessel detection, nautical chart integration, change detection, grazing analysis, shoreline mapping, and web deployment.
