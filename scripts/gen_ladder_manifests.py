#!/usr/bin/env python3
"""Generate the label-source ladder manifests (6 backbones × 4 rungs),
plus the distill stage (NFI) and the LUCAS crop-distill stage per column.

See docs/experiments/label_source_ladder.md. Every rung is the same trainer on
the same tiles with the same per-backbone regime; ONLY the label flags change,
so a rung-to-rung delta attributes to exactly one thing:

    rung 1 nmd2018    in-tile 23-class label       (no --label-dir)
    rung 2 nmd2023    28-class NMD2023 sidecar     (--label-dir nmd2023_labels)
    rung 3 nfi        + NFI-distilled forest type  (--label-dir …_distill_…)
    rung 4 tradslag   + Trädslag fraction head     (+ --frac-dir)

Hand-writing 24 near-identical manifests invites exactly the drift this
experiment cannot tolerate (one stray --epochs and a rung delta becomes an
early-stopping artefact). So they are generated from the six existing
per-backbone yamls, which stay the single source of each backbone's regime —
crop size, aux fusion, ΔSAR, model-specific preflights and all.

    python scripts/gen_ladder_manifests.py           # write k8s/ladder/
    python scripts/gen_ladder_manifests.py --check   # verify on disk == generated

--check exits non-zero if any file is missing or stale, so CI can pin the
manifests to this generator.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "k8s" / "ladder"

# Backbone → the yaml that defines its regime. Prithvi-600M uses the v8b
# NMD2023 long job (30 epochs, 504 crop, markfukt off) rather than
# train-prithvi-600m-512-job.yaml (10 epochs); its --warm-start-from is
# stripped below so the ladder cold-starts like every other column.
BASES = {
    "prithvi300m": "k8s/train-prithvi300m-job.yaml",
    "prithvi600m": "k8s/train-v8b-nmd2023-long-job.yaml",
    "croma": "k8s/train-croma-job.yaml",
    "terramind": "k8s/train-terramind-job.yaml",
    "tessera": "k8s/train-tessera-gated-job.yaml",
    "clay": "k8s/train-clay-job.yaml",
}

# The cohort gate. Every rung trains on exactly the tiles that have an NMD2023
# sidecar (~94.5% coverage), so rung 1 — which reads the in-tile 23-class label
# and would otherwise see a superset — cannot mix a cohort change into its
# delta. Rungs 2-4 land on this set anyway via --label-dir; passing it as
# --cohort-dir on rung 1 makes all four identical by construction.
COHORT_DIR = "/cephfs/nmd2023_labels"

# rung → (slug, label-dir or None, num-classes, frac head?)
# Rung 3/4's label dir is per-backbone: each model distils ITS OWN rung-2
# features (user, 2026-08-29 — "that distinguish them more"), so the sidecars
# live in a model-scoped directory rather than one shared pool.
RUNGS = {
    1: ("nmd2018", None, 23, False),
    2: ("nmd2023", COHORT_DIR, 28, False),
    3: ("nfi", "/cephfs/distill/{model}_r2", 28, False),
    4: ("tradslag", "/cephfs/distill/{model}_r2", 28, True),
}

# Per-column regime for the distillation stage (between rungs 2 and 3).
# img_size follows the column's training crop — it is part of the backbone's
# regime, and after the load_model img_size fix (tests/
# test_distill_stage_wiring.py) it also decides the grid the backbone is
# BUILT at, which for clay/croma is the difference between features and
# garbage. sar_cohort marks the two columns whose trainable tiles are the
# s1_vv_vh subset; their dense pass pre-filters to exactly that cohort.
DISTILL = {
    "prithvi300m": {"img_size": 496, "backbone": "prithvi_300m"},
    "prithvi600m": {"img_size": 504, "backbone": "prithvi_600m"},
    "croma": {"img_size": 504, "backbone": "croma_base",
              "require_keys": ("s1_vv_vh",),
              "extra_setup": (
                  "git clone --depth 1 https://github.com/antofuller/CROMA "
                  "/workspace/CROMA && pip install --quiet --no-cache-dir "
                  "-e /workspace/CROMA || true")},
    "terramind": {"img_size": 496, "backbone": "terramind_v1_base",
                  "require_keys": ("s1_vv_vh",),
                  "extra_setup": "pip install --quiet --no-cache-dir terratorch"},
    "tessera": {"img_size": 504, "backbone": "tessera_v1",
                # The dense pass must mirror the TRAINING cohort: the
                # dataset drops tessera-embedding-less tiles at index
                # construction (unified_dataset._MODEL_REQUIRED_TILE_KEYS
                # + the runtime required_key skip), so training ran on
                # 7874 tiles while the unfiltered dense pass walked 7882
                # and its cohort gate refused on the 8 stragglers —
                # correctly, but for a cohort nobody trains on.
                "require_keys": ("tessera",)},
    "clay": {"img_size": 504, "backbone": "clay_v1_5",
             "extra_setup": ("pip install --quiet --no-cache-dir "
                             "\"git+https://github.com/Clay-foundation/model.git\"")},
}

# LUCAS crop-distill stage — the R5 evidence pass. Per column: extract
# features at the frozen LUCAS crop distill points, score the pinned-
# protocol crop OOF. The numbers decide WHETHER a rung 5 exists
# [user-stated 2026-08-31: distillability before any retraining], so this
# stage deliberately writes NO _GATE_OK and no queue rung consumes it —
# rung 5 must not auto-train off these files before the numbers are read.
CROP_TRUTH_COL = "unified_class"
CROP_INDEX = "/cephfs/distill/lucas_crop_distill_index.parquet"
CROP_SPLIT = "/cephfs/distill/lucas_crop_split.json"

# Digest-pinned, never a mutable tag (repo zero-tolerance rule: a tag can
# be repointed under the pipeline's feet). Manifest-list digest for
# python:3.11-slim, resolved 2026-09-01 from registry-1.docker.io.
PYTHON_IMAGE = ("python:3.11-slim@sha256:"
                "d1e9ca7c4e78d1e8ecadb5d44bfc8e956e7a65b659a9950f569f243d72b326d0")

# One Job per column, three script steps + the distillability OOF, all
# reading/writing the shared cephfs PVC. Runs on the 2080ti pool so the
# distill stage NEVER competes with the H100 memory quota the ladder
# trainings are packed against. The protocol constants (folds, head, seed,
# test-frac) are pinned HERE, once, for all six columns — that uniformity
# IS the distillability experiment; see docs/experiments/
# ladder_distill_stage.md.
DISTILL_TEMPLATE = """apiVersion: batch/v1
kind: Job
metadata:
  name: ladder-distill-{model}
  namespace: prithvi-training-default
  labels: {{ app: unified-training, purpose: ladder-distill, model: {model} }}
spec:
  backoffLimit: 0
  # Budget: dense pass = 7882 tiles (optical) x 1-5 s/tile on a 2080ti
  # (forward + cephfs np.load + 256-ch upsample + per-pixel matmul)
  # = 2.2-11 h, plus extract (~0.5 h) and CPU steps. 24 h leaves ~2x
  # headroom over the pessimistic end; the deadline only caps runaway.
  activeDeadlineSeconds: 86400
  ttlSecondsAfterFinished: 172800
  template:
    metadata:
      labels: {{ app: unified-training, purpose: ladder-distill, model: {model} }}
    spec:
      restartPolicy: Never
      nodeSelector:
        accelerator: nvidia-gtx-2080ti
      containers:
        - name: distill
          image: {python_image}
          command:
            - bash
            - -c
            - |
              set -e
              export PYTHONUNBUFFERED=1
              echo "=== ladder distill stage — {model} ==="
              apt-get update -qq && apt-get install -y -qq git > /dev/null 2>&1
              pip install --quiet --no-cache-dir torch torchvision \\
                --index-url https://download.pytorch.org/whl/cu121
              pip install --quiet --no-cache-dir \\
                timm einops numpy Pillow scipy scikit-learn huggingface_hub \\
                pandas pyarrow rasterio pyproj

              mkdir -p /workspace && cd /workspace
              BRANCH=main
              git clone --depth 1 --branch "$BRANCH" \\
                https://github.com/TobiasEdman/ImintEngine.git imintengine
              cd /workspace/imintengine
              pip install --no-cache-dir -e . --no-deps 2>/dev/null || true
              echo "CLONED $BRANCH HEAD: $(git rev-parse --short HEAD)"
              # Per-column backbone deps — the r2 checkpoints cannot even
              # LOAD without them (terramind: terratorch ImportError killed
              # the first submission; croma/clay: their loader packages).
              {extra_setup}

              CKPT=/cephfs/checkpoints/ladder/{model}_r2/best_model.pt
              PIN=/cephfs/distill/pinned_plots.json
              OUT=/cephfs/distill/{model}_r2
              test -f "$CKPT" || {{ echo "FATAL: no rung-2 checkpoint at $CKPT"; exit 1; }}
              test -f "$PIN"  || {{ echo "FATAL: no pinned plot set at $PIN — run k8s/ladder/distill-pinned-plots-job.yaml first"; exit 1; }}
              mkdir -p "$OUT" /cephfs/distill/heads

              echo "=== 1/4 extract plot features (rung-2 checkpoint, 11 aux) ==="
              python3 scripts/extract_plot_features.py \\
                --checkpoint "$CKPT" \\
                --plot-index /cephfs/nfi/nfi_index_unified_v2_512.parquet \\
                --img-size {img_size} \\
                --backbone-name {backbone} \\
                --enable-markfukt \\
                --out /cephfs/distill/heads/{model}_r2_plot_features.parquet \\
                --device cuda

              echo "=== 2/4 distillability — pinned-protocol OOF (the cross-backbone number) ==="
              python3 scripts/nfi_head_cv.py \\
                --features /cephfs/distill/heads/{model}_r2_plot_features.parquet \\
                --folds 5 \\
                --heads mlp \\
                --pinned-plots "$PIN" \\
                --out /cephfs/distill/heads/{model}_r2_distillability.json

              echo "=== 3/4 deployable head (grouped-by-tile split, full plot set) ==="
              python3 scripts/train_distill_head.py \\
                --features /cephfs/distill/heads/{model}_r2_plot_features.parquet \\
                --test-frac 0.2 --seed 42 \\
                --out-head /cephfs/distill/heads/{model}_r2_head.npz \\
                --out-split /cephfs/distill/heads/{model}_r2_split.json

              echo "=== 4/4 dense sidecars + cohort gate ==="
              python3 scripts/distill_forest_labels.py \\
                --checkpoint "$CKPT" \\
                --head /cephfs/distill/heads/{model}_r2_head.npz \\
                --data-dir /cephfs/unified_v2_512 \\
                --label-dir /cephfs/nmd2023_labels \\
                --out-dir "$OUT" \\
                --img-size {img_size} \\
                --backbone-name {backbone} \\
                --enable-markfukt \\{sar_filter}
                --device cuda

              echo ""
              # The queue's signal: state on DISK, never job existence (the
              # reaper deletes finished Jobs — the exact interaction that
              # bit rung-1 resubmission). set -e means reaching this line
              # proves all four steps INCLUDING the cohort gate passed.
              touch "$OUT/_GATE_OK"
              echo "=== distill stage complete for {model} — rungs 3/4 unlocked ==="
          env:
            - name: PYTHONUNBUFFERED
              value: "1"
            - name: HUGGING_FACE_HUB_TOKEN
              valueFrom:
                secretKeyRef: {{ name: hf-token, key: token, optional: true }}
          resources:
            requests: {{ cpu: "4", memory: "24Gi", nvidia.com/gpu: "1" }}
            limits: {{ cpu: "4", memory: "24Gi", nvidia.com/gpu: "1" }}
          volumeMounts:
            - {{ name: cephfs, mountPath: /cephfs }}
            # Same PVC, second mount point. The NFI plot index stores
            # ABSOLUTE tile paths under /data/ (written by the nfi-* jobs,
            # which mount there); with only /cephfs mounted the extract
            # step drops all 982 plots as "no longer in the dataset" and
            # dies — the first distill submission failed exactly so
            # (reaper archive 20260831T0920Z).
            - {{ name: cephfs, mountPath: /data }}
      volumes:
        - name: cephfs
          persistentVolumeClaim: {{ claimName: training-data-cephfs }}
"""

CROP_DISTILL_TEMPLATE = """apiVersion: batch/v1
kind: Job
metadata:
  name: ladder-crop-distill-{model}
  namespace: prithvi-training-default
  labels: {{ app: unified-training, purpose: ladder-crop-distill, model: {model} }}
spec:
  backoffLimit: 0
  # Budget: extract forwards ~1.5k tiles (the LUCAS distill side) once
  # each at 1-5 s/tile on a 2080ti = 0.5-2 h; the OOF is CPU-minutes.
  # 12 h caps runaway only.
  activeDeadlineSeconds: 43200
  ttlSecondsAfterFinished: 172800
  template:
    metadata:
      labels: {{ app: unified-training, purpose: ladder-crop-distill, model: {model} }}
    spec:
      restartPolicy: Never
      nodeSelector:
        accelerator: nvidia-gtx-2080ti
      containers:
        - name: crop-distill
          image: {python_image}
          command:
            - bash
            - -c
            - |
              set -euo pipefail
              export PYTHONUNBUFFERED=1
              echo "=== LUCAS crop-distill stage — {model} ==="
              # One terminal record per attempt, installed BEFORE the
              # first thing that can fail. EXIT (not ERR): explicit
              # `exit 1` paths bypass ERR, and the trap must observe
              # early failures too — HEAD/artifacts default to unknown.
              RUN_ID=$(date -u +%Y%m%dT%H%M%SZ)-$$
              mkdir -p /cephfs/ops
              trap 'rc=$?; echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) run=$RUN_ID job=ladder-crop-distill-{model} HEAD=${{HEAD_SHA:-unknown}} ${{ARTIFACTS:-artifacts=none}} rc=$rc status=$([ "$rc" -eq 0 ] && echo OK || echo FAIL)" >> /cephfs/ops/crop_distill.log' EXIT
              apt-get update -qq && apt-get install -y -qq git > /dev/null 2>&1
              pip install --quiet --no-cache-dir torch torchvision \\
                --index-url https://download.pytorch.org/whl/cu121
              pip install --quiet --no-cache-dir \\
                timm einops numpy Pillow scipy scikit-learn huggingface_hub \\
                pandas pyarrow rasterio pyproj
              mkdir -p /workspace && cd /workspace
              BRANCH=main
              git clone --depth 1 --branch "$BRANCH" \\
                https://github.com/TobiasEdman/ImintEngine.git imintengine
              cd /workspace/imintengine
              pip install --no-cache-dir -e . --no-deps 2>/dev/null || true
              # Own line, not inside an echo: under set -e a failed
              # substitution here kills the run instead of being masked.
              HEAD_SHA=$(git rev-parse HEAD)
              echo "CLONED $BRANCH HEAD: $HEAD_SHA"
              # Per-column backbone deps — the r2 checkpoints cannot even
              # LOAD without them.
              {extra_setup}

              CKPT=/cephfs/checkpoints/ladder/{model}_r2/best_model.pt
              INDEX={crop_index}
              SPLIT={crop_split}
              test -f "$CKPT"  || {{ echo "FATAL: no rung-2 checkpoint at $CKPT"; exit 1; }}
              echo "=== 0/2 verify the frozen split (marker + cross-artifact consistency) ==="
              # Existence tests are not enough: a consumer must never
              # read a publish window or a crash-left pair. --verify
              # validates the MANIFEST hashes AND JSON<->parquet key
              # equality, and exits non-zero on anything less.
              python3 scripts/build_lucas_crop_split.py --verify --out-dir /cephfs/distill
              mkdir -p /cephfs/distill/heads

              echo "=== 1/2 extract crop-point features (rung-2 checkpoint) ==="
              python3 scripts/extract_plot_features.py \\
                --checkpoint "$CKPT" \\
                --plot-index "$INDEX" \\
                --truth-col {truth_col} \\
                --img-size {img_size} \\
                --backbone-name {backbone} \\
                --enable-markfukt \\
                --out /cephfs/distill/heads/{model}_r2_crop_features.parquet \\
                --device cuda

              echo "=== 2/2 crop distillability — pinned-protocol OOF (the R5 evidence) ==="
              python3 scripts/nfi_head_cv.py \\
                --features /cephfs/distill/heads/{model}_r2_crop_features.parquet \\
                --folds 5 \\
                --heads mlp \\
                --truth-col {truth_col} \\
                --pinned-plots "$SPLIT" \\
                --out /cephfs/distill/heads/{model}_r2_crop_distillability.json

              # NO _GATE_OK here, on purpose: rung 5 does not exist until a
              # human reads these numbers [user-stated 2026-08-31 —
              # distillability before retraining]. A gate marker would let
              # the queue auto-train a rung the decision has not approved.
              # Evidence: hashes bind THIS run's outputs into the EXIT
              # record, so a later overwrite of the fixed paths cannot
              # ride an old OK line. pipefail makes a failed sha256sum
              # fatal; the greps reject anything but a full 64-hex digest.
              OOF_SHA=$(sha256sum /cephfs/distill/heads/{model}_r2_crop_distillability.json | cut -d" " -f1)
              FEAT_SHA=$(sha256sum /cephfs/distill/heads/{model}_r2_crop_features.parquet | cut -d" " -f1)
              echo "$OOF_SHA"  | grep -qE "^[0-9a-f]{{64}}$"
              echo "$FEAT_SHA" | grep -qE "^[0-9a-f]{{64}}$"
              ARTIFACTS="oof_sha256=$OOF_SHA features_sha256=$FEAT_SHA"
              echo "=== crop-distill complete for {model} — numbers ready for the R5 decision ==="
          env:
            - name: PYTHONUNBUFFERED
              value: "1"
            - name: HUGGING_FACE_HUB_TOKEN
              valueFrom:
                secretKeyRef: {{ name: hf-token, key: token, optional: true }}
          resources:
            requests: {{ cpu: "4", memory: "24Gi", nvidia.com/gpu: "1" }}
            limits: {{ cpu: "4", memory: "24Gi", nvidia.com/gpu: "1" }}
          volumeMounts:
            - {{ name: cephfs, mountPath: /cephfs }}
            # Same PVC, second mount point: the LUCAS index inherits the L1
            # index's ABSOLUTE /data/… tile paths (same trap as the NFI
            # index — reaper archive 20260831T0920Z).
            - {{ name: cephfs, mountPath: /data }}
      volumes:
        - name: cephfs
          persistentVolumeClaim: {{ claimName: training-data-cephfs }}
"""

LUCAS_SPLIT_TEMPLATE = """apiVersion: batch/v1
kind: Job
metadata:
  name: ladder-lucas-crop-split
  namespace: prithvi-training-default
  labels: {{ app: unified-training, purpose: ladder-crop-distill, model: shared }}
spec:
  backoffLimit: 0
  activeDeadlineSeconds: 3600
  ttlSecondsAfterFinished: 172800
  template:
    metadata:
      labels: {{ app: unified-training, purpose: ladder-crop-distill, model: shared }}
    spec:
      restartPolicy: Never
      containers:
        - name: split
          image: {python_image}
          command:
            - bash
            - -c
            - |
              set -euo pipefail
              export PYTHONUNBUFFERED=1
              # One terminal record per attempt, installed BEFORE the
              # first thing that can fail (EXIT, not ERR — see the
              # crop-distill template).
              RUN_ID=$(date -u +%Y%m%dT%H%M%SZ)-$$
              mkdir -p /cephfs/ops
              trap 'rc=$?; echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) run=$RUN_ID job=ladder-lucas-crop-split HEAD=${{HEAD_SHA:-unknown}} ${{ARTIFACTS:-artifacts=none}} rc=$rc status=$([ "$rc" -eq 0 ] && echo OK || echo FAIL)" >> /cephfs/ops/crop_distill.log' EXIT
              apt-get update -qq && apt-get install -y -qq git > /dev/null 2>&1
              pip install --quiet --no-cache-dir numpy pandas pyarrow
              mkdir -p /workspace && cd /workspace
              BRANCH=main
              git clone --depth 1 --branch "$BRANCH" \\
                https://github.com/TobiasEdman/ImintEngine.git imintengine
              cd /workspace/imintengine
              # Own line, not inside an echo: under set -e a failed
              # substitution here kills the run instead of being masked.
              HEAD_SHA=$(git rev-parse HEAD)
              echo "CLONED $BRANCH HEAD: $HEAD_SHA"
              python3 scripts/build_lucas_crop_split.py \\
                --lucas-index /cephfs/lucas/lucas_tile_index.parquet \\
                --data-dir /cephfs/unified_v2_512 \\
                --out-dir /cephfs/distill \\
                --git-sha "$HEAD_SHA"
              # Evidence: the MANIFEST (published LAST by the builder)
              # binds both artifacts by content hash; its own hash goes
              # into the EXIT record. pipefail makes a failed sha256sum
              # fatal; the grep rejects a partial digest.
              MANIFEST_SHA=$(sha256sum /cephfs/distill/lucas_crop_split.MANIFEST.json | cut -d" " -f1)
              echo "$MANIFEST_SHA" | grep -qE "^[0-9a-f]{{64}}$"
              ARTIFACTS="manifest_sha256=$MANIFEST_SHA"
          resources:
            requests: {{ cpu: "2", memory: "8Gi" }}
            limits: {{ cpu: "2", memory: "8Gi" }}
          volumeMounts:
            - {{ name: cephfs, mountPath: /cephfs }}
      volumes:
        - name: cephfs
          persistentVolumeClaim: {{ claimName: training-data-cephfs }}
"""

PINNED_PLOTS_TEMPLATE = """apiVersion: batch/v1
kind: Job
metadata:
  name: ladder-distill-pinned-plots
  namespace: prithvi-training-default
  labels: {{ app: unified-training, purpose: ladder-distill, model: shared }}
spec:
  backoffLimit: 0
  activeDeadlineSeconds: 3600
  ttlSecondsAfterFinished: 172800
  template:
    metadata:
      labels: {{ app: unified-training, purpose: ladder-distill, model: shared }}
    spec:
      restartPolicy: Never
      containers:
        - name: pin
          image: {python_image}
          command:
            - bash
            - -c
            - |
              set -e
              export PYTHONUNBUFFERED=1
              apt-get update -qq && apt-get install -y -qq git > /dev/null 2>&1
              pip install --quiet --no-cache-dir numpy pandas pyarrow
              mkdir -p /workspace && cd /workspace
              BRANCH=main
              git clone --depth 1 --branch "$BRANCH" \\
                https://github.com/TobiasEdman/ImintEngine.git imintengine
              cd /workspace/imintengine
              echo "CLONED $BRANCH HEAD: $(git rev-parse --short HEAD)"
              python3 scripts/build_pinned_plot_set.py \\
                --plot-index /cephfs/nfi/nfi_index_unified_v2_512.parquet \\
                --data-dir /cephfs/unified_v2_512 \\
                --out /cephfs/distill/pinned_plots.json
          resources:
            requests: {{ cpu: "2", memory: "8Gi" }}
            limits: {{ cpu: "2", memory: "8Gi" }}
          volumeMounts:
            - {{ name: cephfs, mountPath: /cephfs }}
      volumes:
        - name: cephfs
          persistentVolumeClaim: {{ claimName: training-data-cephfs }}
"""


def render_distill(model: str) -> str:
    cfg = DISTILL[model]
    keys = cfg.get("require_keys", ())
    sar_filter = "".join(
        f"\n                --require-npz-key {k} \\" for k in keys)
    header = (
        f"# GENERATED by scripts/gen_ladder_manifests.py — do not edit.\n"
        f"# Distill stage for {model}: r2 checkpoint → plot features →\n"
        f"# pinned-protocol distillability OOF → deployable head → dense\n"
        f"# sidecars in /cephfs/distill/{model}_r2 (consumed by rungs 3/4).\n"
        f"# Prereq: ladder-distill-pinned-plots (once, shared).\n"
        f"# Plan: docs/experiments/ladder_distill_stage.md\n"
    )
    return header + DISTILL_TEMPLATE.format(
        model=model, img_size=cfg["img_size"], backbone=cfg["backbone"],
        sar_filter=sar_filter,
        extra_setup=cfg.get("extra_setup", "true  # no extra deps"),
        python_image=PYTHON_IMAGE)


def render_crop_distill(model: str) -> str:
    cfg = DISTILL[model]
    header = (
        f"# GENERATED by scripts/gen_ladder_manifests.py — do not edit.\n"
        f"# LUCAS crop-distill stage for {model}: r2 checkpoint → features at\n"
        f"# the frozen LUCAS crop distill points → pinned-protocol crop OOF\n"
        f"# ({model}_r2_crop_distillability.json). Evidence for the R5\n"
        f"# decision — writes NO gate, feeds NO queue rung.\n"
        f"# Prereq: ladder-lucas-crop-split (once, shared).\n"
        f"# Plan: docs/experiments/ladder_distill_stage.md\n"
    )
    return header + CROP_DISTILL_TEMPLATE.format(
        model=model, img_size=cfg["img_size"], backbone=cfg["backbone"],
        truth_col=CROP_TRUTH_COL, crop_index=CROP_INDEX, crop_split=CROP_SPLIT,
        extra_setup=cfg.get("extra_setup", "true  # no extra deps"),
        python_image=PYTHON_IMAGE)

EPOCHS = 30  # fixed across the ladder: see the doc's "Controls"

# The SLU Markfuktighetskarta soil-moisture aux (11th channel) is opt-in in
# the trainer, so every ladder run must ask for it explicitly. It is on
# because the only matched pre-ladder pair measured it as worth ~+0.018 mIoU
# — v8b_markfukt (on, 15 epochs) 0.5527 vs unified_v8b_full7882_e20 (off, 20
# epochs) 0.5352, i.e. better on five FEWER epochs — roughly four times the
# VPP-repair effect. A ladder run at the handicapped 10-aux config would
# still isolate label source, but its winning cell would not be a model
# worth shipping. [user-stated 2026-08-29]
ENABLE_MARKFUKT = True

_FLAG_LINE = r"^(?P<indent>\s*)--{flag}[ =][^\n]*?(?P<cont>\s*\\)?$"

# The base manifests use both YAML label styles — flow (`labels: {a: b}`) and
# block (`labels:\n  a: b`). Both must be rewritten or a job keeps its old
# purpose label and drops out of the ladder's selectors.
_LABELS_FLOW = re.compile(r"^(?P<indent>[ ]*)labels: \{[^}]*\}$", re.M)
_LABELS_BLOCK = re.compile(
    r"^(?P<indent>[ ]*)labels:\n(?:(?P=indent)[ ]{2}\S[^\n]*\n)+", re.M)


def _set_labels(text: str, rung: int, model: str) -> str:
    def flow(m: re.Match) -> str:
        return (f'{m.group("indent")}labels: {{ app: unified-training, '
                f'purpose: ladder, rung: "r{rung}", model: {model} }}')
    text = _LABELS_FLOW.sub(flow, text)
    return _LABELS_BLOCK.sub(lambda m: flow(m) + "\n", text)


def _drop_flag(text: str, flag: str) -> str:
    return re.sub(_FLAG_LINE.format(flag=re.escape(flag)), "", text,
                  flags=re.M).replace("\n\n", "\n")


def _set_flag(text: str, flag: str, value: str) -> str:
    """Rewrite `--flag <value>`, preserving indent and any trailing backslash.

    The replacement is a callable: these lines end in YAML's `\\` continuation,
    which re.sub would otherwise read as a dangling escape.
    """
    pattern = re.compile(_FLAG_LINE.format(flag=re.escape(flag)), re.M)
    m = pattern.search(text)
    if not m:
        return text
    repl = f"{m.group('indent')}--{flag} {value}{m.group('cont') or ''}"
    return pattern.sub(lambda _: repl, text, count=1)


def _ensure_bool_flag(text: str, anchor: str, flag: str) -> str:
    """Insert a valueless store_true flag after the anchor if absent."""
    if re.search(rf"^\s*--{re.escape(flag)}\b", text, re.M):
        return text
    pattern = re.compile(_FLAG_LINE.format(flag=re.escape(anchor)), re.M)
    m = pattern.search(text)
    if not m:
        raise ValueError(f"anchor --{anchor} not found; cannot insert --{flag}")
    line = f"{m.group('indent')}--{flag}{m.group('cont') or ''}"
    return text[:m.end()] + "\n" + line + text[m.end():]


def _ensure_flag_after(text: str, anchor: str, flag: str, value: str) -> str:
    """Insert `--flag value` right after the anchor flag if absent."""
    if re.search(_FLAG_LINE.format(flag=re.escape(flag)), text, re.M):
        return _set_flag(text, flag, value)
    pattern = re.compile(_FLAG_LINE.format(flag=re.escape(anchor)), re.M)
    m = pattern.search(text)
    if not m:
        raise ValueError(f"anchor --{anchor} not found; cannot insert --{flag}")
    cont = m.group("cont") or ""
    line = f"{m.group('indent')}--{flag} {value}{cont}"
    return text[:m.end()] + "\n" + line + text[m.end():]


def render(model: str, rung: int, base_text: str) -> str:
    slug, label_dir, num_classes, with_frac = RUNGS[rung]
    label_dir = label_dir.format(model=model) if label_dir else None
    job = f"ladder-r{rung}-{model}"
    out = base_text

    # Label source — the one axis this experiment varies.
    if label_dir is None:
        out = _drop_flag(out, "label-dir")
    else:
        out = _ensure_flag_after(out, "data-dirs", "label-dir", label_dir)
    # Cohort held constant across all rungs. Rungs 2-4 already land on the
    # sidecar set via --label-dir; rung 1 needs the explicit gate.
    if label_dir is None:
        out = _ensure_flag_after(out, "data-dirs", "cohort-dir", COHORT_DIR)
    if with_frac:
        out = _ensure_flag_after(out, "label-dir", "frac-dir",
                                 "/cephfs/tradslag_fracs")
    else:
        out = _drop_flag(out, "frac-dir")
    out = _set_flag(out, "num-classes", str(num_classes))
    if ENABLE_MARKFUKT:
        out = _ensure_bool_flag(out, "data-dirs", "enable-markfukt")

    # Controls: cold start, fixed epochs. The base manifests carry prose
    # describing the warm-start we just removed ("Continue from the 0.478
    # checkpoint…"); left in place it would document a control the run does
    # not apply, which is worse than no comment at all.
    out = _drop_flag(out, "warm-start-from")
    out = re.sub(r"^\s*#.*warm-start.*\n", "", out, flags=re.M | re.I)
    out = _set_flag(out, "epochs", str(EPOCHS))

    # Isolate outputs so no ladder run can overwrite a historical checkpoint.
    out = _set_flag(out, "checkpoint-dir",
                    f"/cephfs/checkpoints/ladder/{model}_r{rung}")
    out = re.sub(r"^(\s*)mkdir -p /cephfs/checkpoints/\S+",
                 rf"\1mkdir -p /cephfs/checkpoints/ladder/{model}_r{rung}",
                 out, flags=re.M)
    out = re.sub(r"^(\s*)rm -f /cephfs/checkpoints/\S+/\*\.pt\.tmp",
                 rf"\1rm -f /cephfs/checkpoints/ladder/{model}_r{rung}/*.pt.tmp",
                 out, flags=re.M)

    # Strip vestigial RWO PVC mounts (training-data, training-checkpoints).
    # Ladder checkpoints live on cephfs (RWX); the RWO volumes only supplied
    # the /data weight cache, and an RWO volume held by the dashboard on
    # p02r08srv01 Multi-Attach-blocks any ladder pod scheduled elsewhere
    # (ladder-r1-prithvi600m sat in ContainerCreating exactly this way).
    # Weights fall back to the HF download path the manifests already handle.
    for vol in ("training-data", "training-checkpoints"):
        out = re.sub(
            rf"^\s*- name: {vol}\n(?:\s+(?:mountPath|readOnly|persistentVolumeClaim|claimName):[^\n]*\n)+",
            "", out, flags=re.M)

    # Identity: name + selectable labels (by rung, by model, or both).
    out = re.sub(r"^  name: \S+$", f"  name: {job}", out, count=1, flags=re.M)
    out = _set_labels(out, rung, model)

    header = (
        f"# GENERATED by scripts/gen_ladder_manifests.py — do not edit.\n"
        f"# Ladder rung {rung} ({slug}) for {model}. Edit the base manifest\n"
        f"# ({BASES[model]}) or the generator, then regenerate.\n"
        f"# Plan: docs/experiments/label_source_ladder.md\n"
    )
    return header + out.lstrip("\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true",
                    help="verify the committed manifests match this generator")
    args = ap.parse_args()

    outputs: dict[Path, str] = {}
    for model, base_rel in BASES.items():
        base_text = (REPO / base_rel).read_text()
        for rung in RUNGS:
            dest = OUT_DIR / f"ladder-r{rung}-{model}-job.yaml"
            outputs[dest] = render(model, rung, base_text)
        outputs[OUT_DIR / f"distill-{model}-job.yaml"] = render_distill(model)
        outputs[OUT_DIR / f"crop-distill-{model}-job.yaml"] = (
            render_crop_distill(model))
    outputs[OUT_DIR / "lucas-crop-split-job.yaml"] = (
        "# GENERATED by scripts/gen_ladder_manifests.py — do not edit.\n"
        "# Freezes the LUCAS crop 70/30 distill/holdout split ONCE, before\n"
        "# any crop training touches LUCAS (grouped by tile; prior 71-point\n"
        "# freeze forced into holdout). Run before any crop-distill-<model>.\n"
        "# Plan: docs/experiments/ladder_distill_stage.md\n"
        + LUCAS_SPLIT_TEMPLATE.format(python_image=PYTHON_IMAGE))
    outputs[OUT_DIR / "distill-pinned-plots-job.yaml"] = (
        "# GENERATED by scripts/gen_ladder_manifests.py — do not edit.\n"
        "# Pins the shared NFI plot set every column's distillability OOF\n"
        "# scores. Run ONCE before any distill-<model> job.\n"
        "# Plan: docs/experiments/ladder_distill_stage.md\n"
        + PINNED_PLOTS_TEMPLATE.format(python_image=PYTHON_IMAGE))

    stale: list[str] = []
    for dest, text in outputs.items():
        if args.check:
            if not dest.exists() or dest.read_text() != text:
                stale.append(str(dest.relative_to(REPO)))
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(text)
        print(f"  wrote {dest.relative_to(REPO)}")

    if args.check:
        if stale:
            print(f"STALE ({len(stale)}): " + ", ".join(stale))
            print("Re-run: python scripts/gen_ladder_manifests.py")
            return 1
        print(f"all {len(outputs)} ladder manifests up to date")
        return 0

    print(f"\n{len(outputs)} manifests in {OUT_DIR.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
