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
| 20 | [20_lulc_inference_dashboard.md](20_lulc_inference_dashboard.md) | LULC Inference Dashboard — Per-tile prediction pipeline, NIR false-color, tile gallery, modular dashboard refactoring |
| 21 | [21_nmd_accuracy_and_label_noise.md](21_nmd_accuracy_and_label_noise.md) | NMD Label Accuracy — Label noise analysis, 70-80% NMD accuracy, per-class explanation, 5 label cleaning approaches |
| 22 | [22_phenology_seasonal_strategy.md](22_phenology_seasonal_strategy.md) | Phenology Strategy — VPP/seasonal impact on forest classes, SOSD/EOSD, 3 architectural options, literature review |
| 23 | [23_final_model_evaluation.md](23_final_model_evaluation.md) | Final Model Evaluation — 44.14% mIoU, 0.9334 AUC-ROC, user/producer accuracy, ROC curves, dashboard deployment |
| 24 | [24_colonyos_operations_runbook.md](24_colonyos_operations_runbook.md) | ColonyOS Runbook — CFS networking fix, OOM diagnosis, container regulation, watchdog/backup/safe-down, SSH config |
| 25 | [25_seasonal_s2_fetch_completion.md](25_seasonal_s2_fetch_completion.md) | Seasonal S2 Fetch — 4,305/4,381 multitemporal tiles via CDSE, tile format, VPP status, H100 training plan |

## Sources

Files 01-10: Original conversation log (`Data preperation and training pipeline.txt`, ~1 MB)
Files 02, 11-15: Second pipeline conversation (`Data preperation and training pipeline_2.txt`, ~585 KB)
Files 16-18: Third pipeline conversation (S2 Process API, training results, VPP phenology)
Files 19-20: Fourth pipeline conversation (marine commercial showcase, LULC inference dashboard)
Files 21-24: Fifth pipeline conversation (`Train_fetch_prompt.txt` — NMD accuracy research, phenology strategy, final evaluation, ColonyOS runbook)
File 25: Sixth pipeline conversation (seasonal S2 fetch completion, data inventory, VPP status, H100 training preparation)

The conversations cover satellite imagery analysis with ImintEngine, including Sentinel-2 data processing, LULC classification with Prithvi-EO-2.0, auxiliary data fusion, distributed computing with ColonyOS, vessel detection, nautical chart integration, change detection, grazing analysis, shoreline mapping, and web deployment.
