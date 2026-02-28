# Third-Party Licenses

IMINT Engine incorporates and depends on third-party components with their
own licenses. This file documents those components and their licensing terms.

---

## Models

| Component | License | Copyright | Notes |
|-----------|---------|-----------|-------|
| **YOLO11s** (vessel detection) | AGPL-3.0 | Ultralytics | Copyleft — commercial closed-source use requires [Enterprise license](https://www.ultralytics.com/license) |
| **AI2 rslearn** (detection + attributes) | Apache 2.0 | 2024 Allen Institute for AI | Weights: Apache 2.0, annotations: CC-BY 4.0 |
| **SatlasPretrain** (Swin V2 B backbone) | Apache 2.0 | 2024 Allen Institute for AI | Pre-trained on Sentinel-2 via Satlas |
| **Prithvi-EO 2.0** (burn segmentation) | Apache 2.0 | IBM, NASA, Juelich Supercomputing Centre | Geospatial foundation model (ViT-MAE, 600M params). [HuggingFace](https://huggingface.co/ibm-nasa-geospatial/Prithvi-100M) |
| **COT MLP5 ensemble** (cloud optical thickness) | **TBD** | Aleksis Pirinen / RISE | Pirinen et al., 2024. "Creating and Leveraging a Synthetic Dataset of Cloud Optical Thickness Measures for Cloud Detection in MSI." Remote Sensing. [GitHub](https://github.com/DigitalEarthSweden/ml-cloud-opt-thick). Commercial license not yet clarified — contact aleksis.pirinen@ri.se |
| **PyTorch / Torchvision** | BSD 3-Clause | PyTorch Foundation / Meta Platforms | |

## Data Sources

| Source | License | Copyright | Notes |
|--------|---------|-----------|-------|
| **Sentinel-2 L2A** | Open and free | ESA / Copernicus | Free use, attribution recommended |
| **Digital Earth Sweden** (openEO) | Apache 2.0 / CC0 | Rymdstyrelsen / RISE | Code: Apache 2.0, data: CC0 |
| **NMD** (Nationellt Marktackedata) | CC0 | Naturvardsverket | Public domain, attribution recommended |
| **Sjokort S-57** (nautical charts) | Academic-restricted | Sjofartsverket | Available via [SLU GET](https://maps.slu.se/get/) for SLU staff/students. Publication in scientific works permitted with attribution. |

## JavaScript Libraries (embedded in HTML reports)

| Library | License | Copyright |
|---------|---------|-----------|
| **Leaflet.js** | BSD 2-Clause | 2010-2026 Volodymyr Agafonkin |
| **Leaflet.Sync** | MIT | Bjorn Sandvik |
| **Chart.js** | MIT | 2014-2024 Chart.js Contributors |

## Python Dependencies

Runtime dependencies installed via `requirements.txt` carry their own
licenses. Key ones include:

| Package | License |
|---------|---------|
| numpy | BSD 3-Clause |
| scipy | BSD 3-Clause |
| Pillow | MIT-CMU (HPND) |
| matplotlib | PSF / BSD |
| rasterio | BSD 3-Clause |
| openeo | Apache 2.0 |
| ultralytics | AGPL-3.0 |

---

## AGPL-3.0 Notice (Ultralytics YOLO)

The YOLO11s model and `ultralytics` Python package are distributed under
the GNU Affero General Public License v3.0 (AGPL-3.0). This means:

- If you modify and distribute this software (or provide it as a network
  service), you must make the complete source code available under AGPL-3.0.
- For commercial closed-source use, an Ultralytics Enterprise License is
  required. See <https://www.ultralytics.com/license>.

## Apache 2.0 Notice (Prithvi, AI2, SatlasPretrain)

Licensed under the Apache License, Version 2.0. You may obtain a copy at
<https://www.apache.org/licenses/LICENSE-2.0>. Files under Apache 2.0 are
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND.

The file `imint/fm/prithvi_mae/prithvi_mae.py` contains the original
Apache 2.0 copyright header from IBM Corp.
