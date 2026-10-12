#!/bin/bash
# AI2's own Sentinel-1 vessel predict pipeline, unmodified, on the reference crops
# (docs/marine/s1_vessels_reference.md, section a). CPU, ~22 min on an M1.
#
# usage: scripts/marine/s1_port/run_ai2_reference.sh <ai2_env> <record>
#   <ai2_env>  holds .venv/ (rslearn[extra] + rslearn_projects installed) and the
#              rslearn_projects checkout; RSLP_PREFIX (project_data/) is created here,
#              linking the checkpoints in imint/fm/ai2_vessels_s1/
#   <record>   holds inputs/ (crop_grd.py output); run1/ is written there
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../../.." && pwd)
ENV=$(cd "$1" && pwd)
REC=$(cd "$2" && pwd)
OUT=$REC/run1
CKPT=$REPO/imint/fm/ai2_vessels_s1
DET=$ENV/project_data/projects/20260626_num_historical/data_20250521_model_20250530_satlaspretrain_unfreeze4_13_1hist
ATT=$ENV/project_data/projects/2026_09_17_vessel_attribute_l1/sentinel1_l1
mkdir -p "$OUT" "$DET" "$ATT"
ln -sf "$CKPT/detect_best.ckpt" "$DET/best.ckpt"
ln -sf "$CKPT/attribute_best.ckpt" "$ATT/best.ckpt"

export RSLP_PREFIX=$ENV/project_data
export MARINE_INFRA_PATH=$REC/inputs/marine_infrastructure.geojson   # as AI2's Dockerfile does
export PYTHONUNBUFFERED=1
export DISABLE_MPS=1                                                  # Lightning auto -> CPU
IN=$REC/inputs
TASKS=$(printf '[{"image": {"vv": "%s", "vh": "%s"}, "historical1": {"vv": "%s", "vh": "%s"}, "geojson_path": "%s", "json_path": "%s", "crop_path": "%s"}]' \
    "$IN/target_vv.tif" "$IN/target_vh.tif" "$IN/historical1_vv.tif" "$IN/historical1_vh.tif" \
    "$OUT/reference.geojson" "$OUT/reference.json" "$OUT/crops/")

cd "$ENV/rslearn_projects"            # the configs use paths relative to it
date -u +%Y-%m-%dT%H:%M:%SZ > "$OUT/started_utc.txt"
/usr/bin/time -l "$ENV/.venv/bin/python" -m rslp.main sentinel1_vessels predict \
    --tasks "$TASKS" --score_threshold 0.7 --scratch_path "$OUT/scratch_dir/"
date -u +%Y-%m-%dT%H:%M:%SZ > "$OUT/finished_utc.txt"
