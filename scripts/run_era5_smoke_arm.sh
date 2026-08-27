#!/usr/bin/env bash
# Run one sealed arm of the paired Prithvi-600M ERA5 smoke experiment.
set -Eeuo pipefail

: "${ARM:?Set ARM to control or treatment}"
: "${RUN_ID:?Set the content-addressed RUN_ID}"
: "${BASE_GIT_SHA:?Set the immutable source commit}"
: "${POD_UID:?POD_UID must come from the Kubernetes downward API}"

case "$ARM" in
  control|treatment) ;;
  *) echo "Unknown ARM=$ARM" >&2; exit 2 ;;
esac

CONTAINER_IMAGE="${CONTAINER_IMAGE:?Set the pinned ERA5 smoke runtime image}"
ROOT="/checkpoints/era5_smoke_prithvi600m/$RUN_ID"
COHORT="$ROOT/cohort"
ENVIRONMENT="$ROOT/environment"
ATTEMPT="$ROOT/attempts/$ARM/$POD_UID"
BASE_MODEL="$ROOT/base_model"
BASE_CHECKPOINT="$BASE_MODEL/epoch_010.pt"
TILE_DIR="/cephfs/unified_v2_512"
TILE_MANIFEST="/cephfs/tile_locations_full.json"
FOUNDATION_CHECKPOINT="/checkpoints/model_cache/Prithvi_EO_V2_600M_TL.pt"
FOUNDATION_CACHE="/checkpoints/cache/huggingface/prithvi600m"
ERA5_DIR="$ROOT/era5"
ERA5_CACHE="/checkpoints/cache/era5_smoke_v5"
SEEDS="41,42,43"

export CUBLAS_WORKSPACE_CONFIG=:4096:8
mkdir -p "$ROOT" "$ATTEMPT"
python /patches/era5_smoke_provenance.py verify-code-bundle \
  --code-manifest /patches/bundle_manifest.json
if [ "$ARM" = control ]; then
  python /patches/fetch_foundation.py \
    --target "$FOUNDATION_CHECKPOINT" --cache-dir "$FOUNDATION_CACHE"
fi
python /patches/era5_smoke_provenance.py verify-foundation \
  --foundation-checkpoint "$FOUNDATION_CHECKPOINT"
python /patches/runtime_smoke.py --cuda
test "$(cat /opt/imintengine/.base_git_sha)" = "$BASE_GIT_SHA"
cd /opt/imintengine

cp /patches/config.py imint/training/config.py
cp /patches/era5_aux.py imint/training/era5_aux.py
cp /patches/tile_time.py imint/training/tile_time.py
cp /patches/trainer.py imint/training/trainer.py
cp /patches/unified_dataset.py imint/training/unified_dataset.py
cp /patches/upernet.py imint/fm/upernet.py
cp /patches/train_unified.py scripts/train_unified.py
cp /patches/build_cohort.py scripts/build_era5_smoke_cohort.py
cp /patches/fetch_era5.py scripts/fetch_era5_aux.py
cp /patches/analyze_smoke.py scripts/analyze_era5_smoke.py
cp /patches/era5_smoke_provenance.py scripts/era5_smoke_provenance.py
cp /patches/data_preflight.py scripts/preflight_era5_smoke_data.py

python --version > "$ATTEMPT/python_version.txt"
pip freeze | LC_ALL=C sort > "$ATTEMPT/pip_freeze.txt"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader \
  > "$ATTEMPT/gpu_identity.txt"
nvidia-smi -q > "$ATTEMPT/nvidia_smi.txt"
cp /opt/imintengine/.base_git_sha "$ATTEMPT/base_git_sha.txt"
cp /patches/bundle_manifest.json "$ATTEMPT/bundle_manifest.json"

LOCK_MODE=()
if [ "$ARM" = treatment ]; then
  LOCK_MODE=(--require-existing)
fi
for NAME in python_version.txt pip_freeze.txt gpu_identity.txt; do
  python scripts/era5_smoke_provenance.py lock-file \
    --source "$ATTEMPT/$NAME" --target "$ENVIRONMENT/$NAME" \
    "${LOCK_MODE[@]}"
done

if [ "$ARM" = control ]; then
  if [ -d "$COHORT" ]; then
    python scripts/build_era5_smoke_cohort.py \
      --tile-dir "$TILE_DIR" --output-dir "$COHORT" \
      --manifest-path "$TILE_MANIFEST" --validate-existing
  else
    COHORT_STAGING="$ROOT/.cohort-$POD_UID"
    test ! -e "$COHORT_STAGING"
    # val stays at 128: the builder now inflates the val POOL by --oversample
    #   when probing, so the ~9% ERA5-Land loss is absorbed without shrinking
    #   the validation set. Lowering --val-tiles was the wrong lever — the pool
    #   is sized to the request, so it shrank in step and the loss recurred.
    # --prefer-cereal-tiles: uniform sampling gave vete 13 of 128 val tiles
    #   (cereals 0.62% of val pixels) — too thin to resolve the crop effect
    #   ERA5 exists to test.
    # --era5-land-probe-cache: drop cells ERA5-Land does not cover, so the
    #   cohort never needs the 0.25 deg fallback that six grid validators
    #   reject. Shares $ERA5_CACHE, so probing pre-warms the real fetch.
    python scripts/build_era5_smoke_cohort.py \
      --tile-dir "$TILE_DIR" --output-dir "$COHORT_STAGING" \
      --train-tiles 256 --val-tiles 128 --seed 20260821 \
      --max-label 22 --min-val-crop-classes 5 \
      --prefer-cereal-tiles \
      --era5-land-probe-cache "$ERA5_CACHE" \
      --manifest-path "$TILE_MANIFEST"
    python scripts/build_era5_smoke_cohort.py \
      --tile-dir "$TILE_DIR" --output-dir "$COHORT_STAGING" \
      --manifest-path "$TILE_MANIFEST" --validate-existing
    mv -T "$COHORT_STAGING" "$COHORT"
  fi
else
  python scripts/build_era5_smoke_cohort.py \
    --tile-dir "$TILE_DIR" --output-dir "$COHORT" \
    --manifest-path "$TILE_MANIFEST" --validate-existing
fi

if [ "$ARM" = control ]; then
  python scripts/preflight_era5_smoke_data.py \
    --tile-dir "$TILE_DIR" --cohort-dir "$COHORT" \
    --era5-mode control --output "$ATTEMPT/data_preflight.json"
fi

ln -sf "$FOUNDATION_CHECKPOINT" \
  imint/fm/prithvi_mae/Prithvi_EO_V2_600M_TL.pt

if [ "$ARM" = control ]; then
  PREFLIGHT_DIR="$ATTEMPT/strict_cuda_preflight"
  test ! -e "$PREFLIGHT_DIR"
  python scripts/train_unified.py \
    --backbone-name prithvi_600m \
    --data-dirs "$TILE_DIR" --split-dir "$COHORT" \
    --checkpoint-dir "$PREFLIGHT_DIR" --num-classes 23 \
    --era5-mode control --enable-markfukt \
    --enable-multitemporal --num-temporal-frames 4 \
    --unfreeze-layers 0 --epochs 1 --save-every 1 \
    --batch-size 2 --lr 1e-4 --weight-decay 0.35 \
    --loss-type focal_dice --focal-gamma 2.0 \
    --patch-size 504 --img-size 504 --weighting-method sqrt \
    --label-smoothing 0.05 --lovasz-weight 0.3 \
    --patience 2 --train-loss-patience 2 \
    --log-confusion-every-epoch --num-workers 0 \
    --seed 20260821 --deterministic --preflight-one-batch \
    --device cuda
fi

BASE_PROVENANCE_ARGS=(
  --base-dir "$BASE_MODEL"
  --base-git-sha "$BASE_GIT_SHA"
  --code-manifest /patches/bundle_manifest.json
  --cohort-dir "$COHORT"
  --environment-dir "$ENVIRONMENT"
  --foundation-checkpoint "$FOUNDATION_CHECKPOINT"
  --seed 20260821 --expected-epochs 10 --num-classes 23
)

if [ "$ARM" = control ]; then
  if [ -d "$BASE_MODEL" ]; then
    python scripts/era5_smoke_provenance.py validate-base \
      "${BASE_PROVENANCE_ARGS[@]}"
  else
    BASE_STAGING="$ATTEMPT/base_model"
    test ! -e "$BASE_STAGING"
    python scripts/train_unified.py \
      --backbone-name prithvi_600m \
      --data-dirs "$TILE_DIR" --split-dir "$COHORT" \
      --checkpoint-dir "$BASE_STAGING" --num-classes 23 \
      --era5-mode control --enable-markfukt \
      --enable-multitemporal --num-temporal-frames 4 \
      --unfreeze-layers 0 --epochs 10 --save-every 10 \
      --fixed-checkpoint-only \
      --batch-size 2 --lr 1e-4 --weight-decay 0.35 \
      --loss-type focal_dice --focal-gamma 2.0 \
      --patch-size 504 --img-size 504 --weighting-method sqrt \
      --label-smoothing 0.05 --lovasz-weight 0.3 \
      --patience 11 --train-loss-patience 100 \
      --log-confusion-every-epoch --log-training-exposure --num-workers 4 \
      --seed 20260821 --deterministic \
      --device cuda
    python scripts/build_era5_smoke_cohort.py \
      --tile-dir "$TILE_DIR" --output-dir "$COHORT" \
      --manifest-path "$TILE_MANIFEST" --validate-existing
    BASE_PROVENANCE_ARGS[1]="$BASE_STAGING"
    python scripts/era5_smoke_provenance.py finalize-base \
      "${BASE_PROVENANCE_ARGS[@]}"
    mv -T "$BASE_STAGING" "$BASE_MODEL"
    BASE_PROVENANCE_ARGS[1]="$BASE_MODEL"
    python scripts/era5_smoke_provenance.py validate-base \
      "${BASE_PROVENANCE_ARGS[@]}"
  fi
else
  python scripts/era5_smoke_provenance.py validate-base \
    "${BASE_PROVENANCE_ARGS[@]}"
fi

INIT_MODE=()
if [ "$ARM" = treatment ]; then
  INIT_MODE=(--require-existing)
fi
python scripts/era5_smoke_provenance.py init-run \
  --root "$ROOT" --run-id "$RUN_ID" --base-git-sha "$BASE_GIT_SHA" \
  --code-manifest /patches/bundle_manifest.json \
  --cohort-manifest "$COHORT/manifest.json" \
  --environment-dir "$ENVIRONMENT" \
  --initial-checkpoint "$BASE_CHECKPOINT" \
  --base-completion "$BASE_MODEL/completion.json" \
  --foundation-checkpoint "$FOUNDATION_CHECKPOINT" \
  --container-image "$CONTAINER_IMAGE" --seeds "$SEEDS" \
  --expected-epochs 5 --num-classes 23 "${INIT_MODE[@]}"

COVERAGE_ARGS=()
if [ "$ARM" = treatment ]; then
  for SEED in 41 42 43; do
    python scripts/era5_smoke_provenance.py validate-seed \
      --seed-dir "$ROOT/control_seed${SEED}" \
      --run-manifest "$ROOT/run_manifest.json" \
      --arm control --seed "$SEED"
  done
  if [ "${ALLOW_OPEN_METEO_TILE_METADATA:-no}" != yes ]; then
    echo "Treatment blocked: explicit metadata-disclosure approval is absent" >&2
    exit 3
  fi
  mkdir -p "$ERA5_DIR" "$ERA5_CACHE"
  if [ -f "$ERA5_DIR/coverage.json" ]; then
    python scripts/fetch_era5_aux.py \
      --tile-dir "$TILE_DIR" --cohort-dir "$COHORT" \
      --output-dir "$ERA5_DIR" --cache-dir "$ERA5_CACHE" \
      --manifest-path "$TILE_MANIFEST" --validate-existing
  else
    python scripts/fetch_era5_aux.py \
      --tile-dir "$TILE_DIR" --cohort-dir "$COHORT" \
      --output-dir "$ERA5_DIR" --cache-dir "$ERA5_CACHE" \
      --manifest-path "$TILE_MANIFEST"
  fi
  COVERAGE_ARGS=(--era5-coverage "$ERA5_DIR/coverage.json")
fi

if [ "$ARM" = treatment ]; then
  python scripts/preflight_era5_smoke_data.py \
    --tile-dir "$TILE_DIR" --cohort-dir "$COHORT" \
    --era5-mode treatment --era5-dir "$ERA5_DIR" \
    --output "$ATTEMPT/data_preflight.json"
fi

for SEED in 41 42 43; do
  FINAL="$ROOT/${ARM}_seed${SEED}"
  if [ -d "$FINAL" ]; then
    python scripts/era5_smoke_provenance.py validate-seed \
      --seed-dir "$FINAL" --run-manifest "$ROOT/run_manifest.json" \
      --arm "$ARM" --seed "$SEED" "${COVERAGE_ARGS[@]}"
    continue
  fi

  python scripts/build_era5_smoke_cohort.py \
    --tile-dir "$TILE_DIR" --output-dir "$COHORT" \
    --manifest-path "$TILE_MANIFEST" --validate-existing
  if [ "$ARM" = treatment ]; then
    python scripts/fetch_era5_aux.py \
      --tile-dir "$TILE_DIR" --cohort-dir "$COHORT" \
      --output-dir "$ERA5_DIR" --cache-dir "$ERA5_CACHE" \
      --manifest-path "$TILE_MANIFEST" --validate-existing
  fi

  STAGING="$ATTEMPT/${ARM}_seed${SEED}"
  test ! -e "$STAGING"
  ERA5_ARGS=(--era5-mode "$ARM")
  if [ "$ARM" = treatment ]; then
    ERA5_ARGS+=(--era5-dir "$ERA5_DIR")
  fi
  python scripts/train_unified.py \
    --backbone-name prithvi_600m \
    --data-dirs "$TILE_DIR" --split-dir "$COHORT" \
    --checkpoint-dir "$STAGING" --num-classes 23 \
    --resume-from "$BASE_CHECKPOINT" --freeze-spectral \
    --strict-checkpoint-loading --enable-markfukt "${ERA5_ARGS[@]}" \
    --enable-multitemporal --num-temporal-frames 4 \
    --unfreeze-layers 0 --epochs 5 --save-every 5 \
    --fixed-checkpoint-only \
    --batch-size 2 --lr 1e-4 --weight-decay 0.35 \
    --loss-type focal_dice --focal-gamma 2.0 \
    --patch-size 504 --img-size 504 --weighting-method sqrt \
    --label-smoothing 0.05 --lovasz-weight 0.3 \
    --patience 6 --train-loss-patience 100 \
    --log-confusion-every-epoch --log-training-exposure --num-workers 4 \
    --seed "$SEED" --deterministic \
    --device cuda
  python scripts/build_era5_smoke_cohort.py \
    --tile-dir "$TILE_DIR" --output-dir "$COHORT" \
    --manifest-path "$TILE_MANIFEST" --validate-existing
  if [ "$ARM" = treatment ]; then
    python scripts/fetch_era5_aux.py \
      --tile-dir "$TILE_DIR" --cohort-dir "$COHORT" \
      --output-dir "$ERA5_DIR" --cache-dir "$ERA5_CACHE" \
      --manifest-path "$TILE_MANIFEST" --validate-existing
  fi
  python scripts/era5_smoke_provenance.py finalize-seed \
    --seed-dir "$STAGING" --run-manifest "$ROOT/run_manifest.json" \
    --arm "$ARM" --seed "$SEED" "${COVERAGE_ARGS[@]}"
  mv -T "$STAGING" "$FINAL"
  python scripts/era5_smoke_provenance.py validate-seed \
    --seed-dir "$FINAL" --run-manifest "$ROOT/run_manifest.json" \
    --arm "$ARM" --seed "$SEED" "${COVERAGE_ARGS[@]}"
done

if [ "$ARM" = treatment ]; then
  python scripts/analyze_era5_smoke.py \
    --root "$ROOT" --run-id "$RUN_ID" --seeds "$SEEDS" \
    --output "$ROOT/verdict.json"
fi
