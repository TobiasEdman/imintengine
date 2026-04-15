# NMD Production Methodology

Source: NMD2018 Product Description v2.2, Naturvardsverket/Metria

## Overview

NMD (Nationella Marktackedata) is Sweden's national land cover dataset at 10m resolution.
NMD2018 uses a multi-stage pipeline combining satellite data, LiDAR, and vector databases.

## Input Data

### Satellite
- Sentinel-2 summer images (2 dates) + spring image
- Sentinel-2 time series (up to 113 images per tile for arable land)
- Lantmateriet Saccess mosaics (historical reference)

### LiDAR
- National airborne laser scanning (Lantmateriet, 2009-2019)
- Products: P95 height, point density, canopy density layers
- Object height at 2m resolution, aggregated to 10m
- Critical for forest boundary delineation (crown coverage + height)

### Vector / Map Data
- LM Fastighetskartan: building footprints (rasterized as hard constraint)
- LM Terrangkartan: terrain features
- LM Hydrografi: water network boundaries
- SCB: roads, railways, infrastructure, urban areas (Tatorter)
- SJV Blockdatabasen: LPIS agricultural parcels
- SKS: actual clearcuts
- SGU: soil types and soil depth (for wetland)
- DEM 2m (Lantmateriet)

## Classification Pipeline

### Pre-classification (Forklassning)
- Segmentation of satellite imagery
- Broad class assignment: forest/water/open land/built-up
- SCB urban overlay

### Main Classification (Grundklassning)
Four parallel tracks:

**Forest:**
- Maximum Likelihood classifier on Sentinel-2 (spring + summer)
- LiDAR analysis for forest/non-forest boundary (crown cover >10%, height >5m)
- Tree species classification: pine, spruce, deciduous, mixed

**Wetland:**
- DEM-derived Soil Topographic Index (STI)
- Combined Moisture/Flow Index (MFI)
- SGU soil type + soil depth
- Sampling and categorization

**Arable land:**
- LPIS (Blockdatabasen) parcel boundaries as primary source
- Sentinel-2 time series confirmation
- GIS analysis for parcel-level crop identification

**Water:**
- Lantmateriet Hydrografi network product
- Orthophoto interpretation for transitional waters

### Weighted Combination (Sammanvagning)
- Conditional merging with priority rules
- Vector data (buildings, roads, water, parcels) overrides satellite classification
- Produces hard edges from authoritative boundaries

### Post-Processing (Efterbehandling)
- Mosaicking of regional tiles
- Generalization (minimum mapping unit)
- Metadata generation

## Why Boundaries Are Crisp

1. **Object-based approach (NMD2018):** Pixel classification followed by segmentation
   + majority vote per segment. Eliminates pixel-level noise at boundaries.

2. **Vector data as hard constraints:** Buildings from Fastighetskartan, roads from SCB,
   water from Hydrografi, parcels from LPIS — these have vector-precision boundaries
   that are simply rasterized onto the output.

3. **LiDAR forest delineation:** Sub-meter accuracy for forest edge detection via
   crown coverage analysis. Far exceeds 10m satellite-only capability.

4. **Contextual cleanup:** Isolated pixels near boundaries removed via contextual analysis.

## NMD2018 vs NMD2023

- NMD2018: Object-based (segment + majority vote)
- NMD2023: Pixel-based (no segmentation step), AI-based tools
- NMD2023 v0.1: 16 simplified classes; v2.0 restores full class set

## Relevance to ImintEngine

Our pipeline already mirrors NMD's approach in several ways:
- LPIS overlay for crop parcels (like NMD's Blockdatabasen integration)
- SKS clearcut data overlay (like NMD's SKS integration)
- NMD base layer for forest/water/wetland classes

What we're missing:
- LiDAR data (would require Lantmateriet data access)
- Object-based segmentation post-processing (implementing via superpixels)
- Fastighetskartan building footprints (could add as vector overlay)
- Hydrografi water network (could add as vector overlay)

## Potential Improvements

1. Add superpixel segmentation + majority vote (replicates NMD2018 approach)
2. Add Fastighetskartan building footprints as hard constraint
3. Add Hydrografi water boundaries as hard constraint
4. Investigate Lantmateriet LiDAR access for forest edge precision
