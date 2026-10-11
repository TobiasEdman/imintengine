# AI2 Sentinel-1 vessel detection weights

Model weights and Code: Apache License 2.0 (allenai/rslearn_projects),
annotations CC-BY 4.0.

Source repository: https://github.com/allenai/rslearn_projects
(`docs/sentinel1_vessels.md`; licence statement in the repository README / LICENSE).
Repository commit inspected at download time: `9cfaf6e9a13fcb993c687b591eb3f71df7c144c5`.

| File | Source URL | Bytes | sha256 |
|---|---|---|---|
| `detect_best.ckpt` | https://storage.googleapis.com/ai2-rslearn-projects-data/projects/20260626_num_historical/data_20250521_model_20250530_satlaspretrain_unfreeze4_13_1hist/best.ckpt | 1262487287 | `85f86043db84f7e899de5c6642c3a0afc591621d277080a298d023b81c3633c5` |
| `attribute_best.ckpt` | https://storage.googleapis.com/ai2-rslearn-projects-data/projects/2026_09_17_vessel_attribute_l1/sentinel1_l1/best.ckpt | 1157039279 | `24b1e759ba4f250c84cdbc5913fec24e140195f59bc7b13f50c9654e7b435913` |

Downloaded 2026-10-10. Integrity checked against the GCS object headers
(crc32c match for both files; md5 match for `detect_best.ckpt`).
GCS last-modified: detector 2026-06-30, attribute model 2026-09-21.

The `*.ckpt` files are gitignored; only this note is meant to be tracked.
