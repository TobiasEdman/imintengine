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
    python scripts/gen_ladder_manifests.py --non-crop-only --check

--check exits non-zero if any file is missing or stale, so CI can pin the
manifests to this generator.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

if __package__:
    from . import crop_distill_protocol as crop_protocol
else:
    import crop_distill_protocol as crop_protocol

CROP_MODELS = crop_protocol.CROP_MODELS
CROP_MODEL_UIDS = crop_protocol.CROP_MODEL_UIDS
PROTOCOL_CROP_INDEX = crop_protocol.CROP_INDEX
PROTOCOL_CROP_SPLIT = crop_protocol.CROP_SPLIT
PROTOCOL_CROP_SPLIT_MANIFEST = crop_protocol.CROP_SPLIT_MANIFEST
TRUTH_COLUMN = crop_protocol.TRUTH_COLUMN

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
_LEGACY_DISTILL_SETUP = {
    "croma": (
        "git clone --depth 1 https://github.com/antofuller/CROMA "
        "/workspace/CROMA && pip install --quiet --no-cache-dir "
        "-e /workspace/CROMA || true"
    ),
    "terramind": "pip install --quiet --no-cache-dir terratorch",
    "clay": (
        "pip install --quiet --no-cache-dir "
        "\"git+https://github.com/Clay-foundation/model.git\""
    ),
}
DISTILL = {
    model: {
        "img_size": protocol.img_size,
        "backbone": protocol.backbone,
        **(
            {"require_keys": protocol.required_npz_keys}
            if protocol.required_npz_keys
            else {}
        ),
        **(
            {"extra_setup": _LEGACY_DISTILL_SETUP[model]}
            if model in _LEGACY_DISTILL_SETUP
            else {}
        ),
    }
    for model, protocol in CROP_MODELS.items()
}

# LUCAS crop-distill stage — the R5 evidence pass. Per column: extract
# features at the frozen LUCAS crop distill points, score the pinned-
# protocol crop OOF. The numbers decide WHETHER a rung 5 exists
# [user-stated 2026-08-31: distillability before any retraining], so this
# stage deliberately writes NO _GATE_OK and no queue rung consumes it —
# rung 5 must not auto-train off these files before the numbers are read.
CROP_TRUTH_COL = TRUTH_COLUMN
CROP_INDEX = str(PROTOCOL_CROP_INDEX)
CROP_SPLIT = str(PROTOCOL_CROP_SPLIT)
CROP_SPLIT_MANIFEST = str(PROTOCOL_CROP_SPLIT_MANIFEST)

# Multi-phase bootstrap for immutable Pod code. Payload commit A builds the
# image. Later reviewed commits pin A/image, then the read-only source-access
# plan, the metadata-only apply completion, and finally the frozen split. Each
# downstream renderer refuses until its immediate upstream evidence is pinned.
#
# Replaced with the real Commit-A identities before generator output is
# committed. The impossible all-zero sentinels make an accidental partial
# bootstrap fail tests and deployment review loudly.
CROP_DISTILL_SOURCE_GIT_SHA = "c6ad69242e7239662461bf7ff0b6bcd4d072509a"
CROP_DISTILL_IMAGE = (
    "ghcr.io/tobiasedman/imint-ladder-crop-distill@sha256:"
    "7c4fe00f9df2a28caaf05f006f2c567ffb912d514bff13289ef6ec3dd039ba7c"
)
CROP_SOURCE_ACCESS_INDEX_SHA256 = crop_protocol.SOURCE_ACCESS_INDEX_SHA256
CROP_SOURCE_ACCESS_PLAN_SHA256 = (
    "d8cdd10b8e9e2668aadafec1aef1a87f5067c2f414e9453e4ce33c196bc04e98"
)
CROP_SOURCE_ACCESS_PLAN_POD_UID = "fe949aaa-8191-4c03-8989-2b3516a2e2a7"
CROP_SOURCE_ACCESS_COMPLETION_SHA256 = (
    "4cbce71f3224c4e6170b2113c774104ab96a6ef14b4e67d51ab0ddf03354aa97"
)
CROP_SOURCE_ACCESS_COMPLETION_POD_UID = "2d193bf7-2217-4878-b2e5-f1367e7137b3"
CROP_DISTILL_SPLIT_MANIFEST_SHA256 = "0" * 64
_CROP_DISTILL_IMAGE_RE = re.compile(
    r"^ghcr\.io/tobiasedman/imint-ladder-crop-distill@sha256:"
    r"(?P<digest>[0-9a-f]{64})$"
)


def _validate_crop_distill_runtime_identity() -> None:
    """Refuse to render a Job against an unpinned source or image."""
    if (
        re.fullmatch(r"[0-9a-f]{40}", CROP_DISTILL_SOURCE_GIT_SHA) is None
        or CROP_DISTILL_SOURCE_GIT_SHA == "0" * 40
    ):
        raise ValueError(
            "CROP_DISTILL_SOURCE_GIT_SHA must be one nonzero, lowercase, "
            "full 40-hex commit"
        )
    image_match = _CROP_DISTILL_IMAGE_RE.fullmatch(CROP_DISTILL_IMAGE)
    if image_match is None or image_match.group("digest") == "0" * 64:
        raise ValueError(
            "CROP_DISTILL_IMAGE must be "
            "ghcr.io/tobiasedman/imint-ladder-crop-distill@sha256:"
            "<64 lowercase nonzero hex>"
        )


def _validate_crop_source_index_identity() -> None:
    if CROP_SOURCE_ACCESS_INDEX_SHA256 != crop_protocol.SOURCE_ACCESS_INDEX_SHA256:
        raise ValueError(
            "CROP_SOURCE_ACCESS_INDEX_SHA256 must equal the baked producer-index "
            "authority"
        )


def _validate_source_access_plan_authority() -> None:
    _validate_crop_distill_runtime_identity()
    _validate_crop_source_index_identity()
    crop_protocol.require_source_access_sha256(
        CROP_SOURCE_ACCESS_PLAN_SHA256,
        "CROP_SOURCE_ACCESS_PLAN_SHA256",
    )
    crop_protocol.require_source_access_run_id(
        CROP_SOURCE_ACCESS_PLAN_POD_UID,
        "CROP_SOURCE_ACCESS_PLAN_POD_UID",
    )


def _validate_source_access_completion_authority() -> None:
    _validate_source_access_plan_authority()
    crop_protocol.require_source_access_sha256(
        CROP_SOURCE_ACCESS_COMPLETION_SHA256,
        "CROP_SOURCE_ACCESS_COMPLETION_SHA256",
    )
    crop_protocol.require_source_access_run_id(
        CROP_SOURCE_ACCESS_COMPLETION_POD_UID,
        "CROP_SOURCE_ACCESS_COMPLETION_POD_UID",
    )


def _validate_crop_distill_identity() -> None:
    """Refuse crop consumers until Git pins the frozen split manifest."""
    _validate_source_access_completion_authority()
    if (
        re.fullmatch(
            r"[0-9a-f]{64}", CROP_DISTILL_SPLIT_MANIFEST_SHA256
        )
        is None
        or CROP_DISTILL_SPLIT_MANIFEST_SHA256 == "0" * 64
    ):
        raise ValueError(
            "CROP_DISTILL_SPLIT_MANIFEST_SHA256 must be one nonzero, "
            "lowercase, full 64-hex digest from the reviewed split freeze"
        )
    try:
        crop_protocol.validate_model_uid_map(CROP_MODEL_UIDS)
    except ValueError as exc:
        raise ValueError(f"CROP_MODEL_UIDS invalid: {exc}") from exc

# Exact byte identities measured on the ladder PVC.  The extractor hashes and
# deserializes each checkpoint through one descriptor; terminal provenance
# binds that authenticated expected identity without reopening 2.7 GB files.
CROP_CHECKPOINTS = {
    model: {
        "size": protocol.checkpoint_size,
        "sha256": protocol.checkpoint_sha256,
    }
    for model, protocol in CROP_MODELS.items()
}

# Digest-pinned, never a mutable tag (repo zero-tolerance rule: a tag can
# be repointed under the pipeline's feet). Manifest-list digest for
# python:3.11-slim, resolved 2026-09-01 from registry-1.docker.io.
PYTHON_IMAGE = ("python:3.11-slim@sha256:"
                "d1e9ca7c4e78d1e8ecadb5d44bfc8e956e7a65b659a9950f569f243d72b326d0")

# Deps for the legacy NFI-distill/pinned-plots Jobs. The LUCAS crop stage uses
# CROP_DISTILL_IMAGE instead: its complete model-loading stack is hash-locked
# and baked once, so staggered columns cannot drift at all.
SCORING_PINS = "numpy==1.26.4 pandas==2.2.2 pyarrow==17.0.0"
SKLEARN_PIN = "scikit-learn==1.5.1"

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
                --enable-markfukt \\{extract_filter}
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
            requests:
              cpu: "4"
              memory: "24Gi"
              nvidia.com/gpu: "1"
            limits:
              cpu: "4"
              memory: "24Gi"
              nvidia.com/gpu: "1"
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
  ttlSecondsAfterFinished: 172800
  template:
    metadata:
      labels: {{ app: unified-training, purpose: ladder-crop-distill, model: {model} }}
    spec:
      activeDeadlineSeconds: 43200
      automountServiceAccountToken: false
      restartPolicy: Never
      imagePullSecrets:
        - name: ghcr-push
      securityContext:
        runAsNonRoot: true
        runAsUser: {run_uid}
        runAsGroup: 2000
        seccompProfile: {{ type: RuntimeDefault }}
      nodeSelector:
        accelerator: nvidia-gtx-2080ti
      containers:
        - name: crop-distill
          image: {crop_image}
          imagePullPolicy: IfNotPresent
          command:
            - /usr/local/bin/python
          args:
            - /opt/imintengine/scripts/run_crop_distill_job.py
            - --model
            - {model}
          env:
            - name: CROP_DISTILL_SOURCE_GIT_SHA
              value: "{source_git_sha}"
            - name: CROP_DISTILL_IMAGE
              value: "{crop_image}"
            - name: CROP_DISTILL_SPLIT_MANIFEST_SHA256
              value: "{split_manifest_sha256}"
            - name: HOME
              value: /work/home
            - name: TMPDIR
              value: /work/tmp
            - name: POD_UID
              valueFrom:
                fieldRef:
                  fieldPath: metadata.uid
          securityContext:
            allowPrivilegeEscalation: false
            capabilities: {{ drop: ["ALL"] }}
            readOnlyRootFilesystem: true
            runAsNonRoot: true
            runAsUser: {run_uid}
            runAsGroup: 2000
          resources:
            requests: {{ cpu: "4", memory: "24Gi", ephemeral-storage: "8Gi", nvidia.com/gpu: "1" }}
            limits: {{ cpu: "4", memory: "24Gi", ephemeral-storage: "8Gi", nvidia.com/gpu: "1" }}
          volumeMounts:
            - name: training-data-cephfs
              mountPath: /cephfs/unified_v2_512
              subPath: unified_v2_512
              readOnly: true
            - name: training-data-cephfs
              mountPath: /cephfs/checkpoints/ladder/{model}_r2
              subPath: checkpoints/ladder/{model}_r2
              readOnly: true
            - name: training-data-cephfs
              mountPath: /cephfs/distill/crop_split
              subPath: distill/crop_split/crop_consumer
              readOnly: true
            - name: training-data-cephfs
              mountPath: /cephfs/crop-heads
              subPath: {head_subpath}
            - name: training-data-cephfs
              mountPath: /cephfs/crop-records
              subPath: {record_subpath}
            - name: work
              mountPath: /work
      volumes:
        - name: training-data-cephfs
          persistentVolumeClaim: {{ claimName: training-data-cephfs }}
        - name: work
          emptyDir:
            sizeLimit: 8Gi
"""

CROP_STORAGE_PREP_TEMPLATE = """apiVersion: batch/v1
kind: Job
metadata:
  name: ladder-crop-distill-storage-prep
  namespace: prithvi-training-default
  labels: {{ app: unified-training, purpose: ladder-crop-distill-storage }}
spec:
  backoffLimit: 0
  ttlSecondsAfterFinished: 172800
  template:
    metadata:
      labels: {{ app: unified-training, purpose: ladder-crop-distill-storage }}
    spec:
      activeDeadlineSeconds: 600
      automountServiceAccountToken: false
      restartPolicy: Never
      imagePullSecrets:
        - name: ghcr-push
      securityContext:
        runAsUser: 0
        runAsGroup: 2000
        seccompProfile: {{ type: RuntimeDefault }}
      containers:
        - name: storage-prep
          image: {crop_image}
          imagePullPolicy: IfNotPresent
          command:
            - /usr/local/bin/python
          args:
            - /opt/imintengine/scripts/prepare_crop_distill_storage.py
          env:
            - name: CROP_DISTILL_SOURCE_GIT_SHA
              value: "{source_git_sha}"
            - name: CROP_DISTILL_IMAGE
              value: "{crop_image}"
            - name: POD_UID
              valueFrom:
                fieldRef:
                  fieldPath: metadata.uid
          securityContext:
            allowPrivilegeEscalation: false
            capabilities:
              drop: ["ALL"]
              add: ["CHOWN", "FOWNER"]
            readOnlyRootFilesystem: true
            runAsUser: 0
            runAsGroup: 2000
          resources:
            requests: {{ cpu: "500m", memory: "256Mi" }}
            limits: {{ cpu: "500m", memory: "256Mi" }}
          volumeMounts:
            - name: training-data-cephfs
              mountPath: /cephfs/distill
              subPath: distill
            - name: training-data-cephfs
              mountPath: /cephfs/ops
              subPath: ops
      volumes:
        - name: training-data-cephfs
          persistentVolumeClaim: {{ claimName: training-data-cephfs }}
"""

CROP_SOURCE_ACCESS_PLAN_TEMPLATE = """apiVersion: batch/v1
kind: Job
metadata:
  name: ladder-crop-source-access-plan
  namespace: prithvi-training-default
  labels: {{ app: unified-training, purpose: ladder-crop-source-access-plan }}
spec:
  backoffLimit: 0
  ttlSecondsAfterFinished: 172800
  template:
    metadata:
      labels: {{ app: unified-training, purpose: ladder-crop-source-access-plan }}
    spec:
      activeDeadlineSeconds: 7200
      automountServiceAccountToken: false
      restartPolicy: Never
      imagePullSecrets:
        - name: ghcr-push
      securityContext:
        runAsUser: 0
        runAsGroup: 2000
        seccompProfile: {{ type: RuntimeDefault }}
      containers:
        - name: source-access-plan
          image: {crop_image}
          imagePullPolicy: IfNotPresent
          command:
            - /opt/venvs/scoring/bin/python
          args:
            - /opt/imintengine/scripts/crop_source_access.py
            - plan
          env:
            - name: CROP_DISTILL_SOURCE_GIT_SHA
              value: "{source_git_sha}"
            - name: CROP_DISTILL_IMAGE
              value: "{crop_image}"
            - name: CROP_SOURCE_ACCESS_INDEX_SHA256
              value: "{source_index_sha256}"
            - name: CROP_SOURCE_FREEZE_LEASE_PATH
              value: /var/run/crop-source-freeze/lease.json
            - name: POD_UID
              valueFrom:
                fieldRef:
                  fieldPath: metadata.uid
          securityContext:
            allowPrivilegeEscalation: false
            capabilities: {{ drop: ["ALL"] }}
            readOnlyRootFilesystem: true
            runAsUser: 0
            runAsGroup: 2000
          resources:
            requests: {{ cpu: "2", memory: "4Gi" }}
            limits: {{ cpu: "2", memory: "4Gi" }}
          volumeMounts:
            - name: training-data-cephfs
              mountPath: /cephfs/unified_v2_512
              subPath: unified_v2_512
              readOnly: true
            - name: training-data-cephfs
              mountPath: /cephfs/lucas/lucas_tile_index.parquet
              subPath: lucas/lucas_tile_index.parquet
              readOnly: true
            - name: training-data-cephfs
              mountPath: /cephfs/source-access-plan-records
              subPath: ops/crop-distill/source-access/plan
            - name: training-data-cephfs
              mountPath: /cephfs/source-access-lock
              subPath: ops/crop-distill/source-access/locks
            - name: crop-source-freeze-lease
              mountPath: /var/run/crop-source-freeze
              readOnly: true
      volumes:
        - name: training-data-cephfs
          persistentVolumeClaim: {{ claimName: training-data-cephfs }}
        - name: crop-source-freeze-lease
          configMap:
            name: crop-source-freeze-lease
            optional: false
            items:
              - key: lease.json
                path: lease.json
"""

CROP_SOURCE_ACCESS_APPLY_TEMPLATE = """apiVersion: batch/v1
kind: Job
metadata:
  name: ladder-crop-source-access-apply
  namespace: prithvi-training-default
  labels: {{ app: unified-training, purpose: ladder-crop-source-access-apply }}
spec:
  backoffLimit: 0
  ttlSecondsAfterFinished: 172800
  template:
    metadata:
      labels: {{ app: unified-training, purpose: ladder-crop-source-access-apply }}
    spec:
      activeDeadlineSeconds: 7200
      automountServiceAccountToken: false
      restartPolicy: Never
      imagePullSecrets:
        - name: ghcr-push
      securityContext:
        runAsUser: 0
        runAsGroup: 2000
        seccompProfile: {{ type: RuntimeDefault }}
      containers:
        - name: source-access-apply
          image: {crop_image}
          imagePullPolicy: IfNotPresent
          command:
            - /opt/venvs/scoring/bin/python
          args:
            - /opt/imintengine/scripts/crop_source_access.py
            - apply
          env:
            - name: CROP_DISTILL_SOURCE_GIT_SHA
              value: "{source_git_sha}"
            - name: CROP_DISTILL_IMAGE
              value: "{crop_image}"
            - name: CROP_SOURCE_ACCESS_INDEX_SHA256
              value: "{source_index_sha256}"
            - name: CROP_SOURCE_ACCESS_PLAN_SHA256
              value: "{plan_sha256}"
            - name: CROP_SOURCE_ACCESS_PLAN_POD_UID
              value: "{plan_pod_uid}"
            - name: CROP_SOURCE_FREEZE_LEASE_PATH
              value: /var/run/crop-source-freeze/lease.json
            - name: POD_UID
              valueFrom:
                fieldRef:
                  fieldPath: metadata.uid
          securityContext:
            allowPrivilegeEscalation: false
            capabilities:
              drop: ["ALL"]
              add: ["CHOWN", "FOWNER"]
            readOnlyRootFilesystem: true
            runAsUser: 0
            runAsGroup: 2000
          resources:
            requests: {{ cpu: "2", memory: "4Gi" }}
            limits: {{ cpu: "2", memory: "4Gi" }}
          volumeMounts:
            - name: training-data-cephfs
              mountPath: /cephfs/unified_v2_512
              subPath: unified_v2_512
            - name: training-data-cephfs
              mountPath: /cephfs/lucas/lucas_tile_index.parquet
              subPath: lucas/lucas_tile_index.parquet
              readOnly: true
            - name: training-data-cephfs
              mountPath: /cephfs/source-access-plan/plan.json
              subPath: ops/crop-distill/source-access/plan/{plan_pod_uid}/plan.json
              readOnly: true
            - name: training-data-cephfs
              mountPath: /cephfs/source-access-apply-records
              subPath: ops/crop-distill/source-access/apply
            - name: training-data-cephfs
              mountPath: /cephfs/source-access-lock
              subPath: ops/crop-distill/source-access/locks
            - name: crop-source-freeze-lease
              mountPath: /var/run/crop-source-freeze
              readOnly: true
      volumes:
        - name: training-data-cephfs
          persistentVolumeClaim: {{ claimName: training-data-cephfs }}
        - name: crop-source-freeze-lease
          configMap:
            name: crop-source-freeze-lease
            optional: false
            items:
              - key: lease.json
                path: lease.json
"""

CROP_DENY_EGRESS_TEMPLATE = """apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: ladder-crop-distill-deny-egress
  namespace: prithvi-training-default
  labels: { app: unified-training, purpose: ladder-crop-distill-security }
spec:
  podSelector:
    matchExpressions:
      - key: purpose
        operator: In
        values:
          - ladder-crop-distill
          - ladder-crop-distill-storage
          - ladder-crop-source-access-plan
          - ladder-crop-source-access-apply
  policyTypes:
    - Egress
  egress: []
"""

LUCAS_SPLIT_TEMPLATE = """apiVersion: batch/v1
kind: Job
metadata:
  name: ladder-lucas-crop-split
  namespace: prithvi-training-default
  labels: {{ app: unified-training, purpose: ladder-crop-distill, model: shared }}
spec:
  backoffLimit: 0
  ttlSecondsAfterFinished: 172800
  template:
    metadata:
      labels: {{ app: unified-training, purpose: ladder-crop-distill, model: shared }}
    spec:
      activeDeadlineSeconds: 21600
      automountServiceAccountToken: false
      restartPolicy: Never
      imagePullSecrets:
        - name: ghcr-push
      securityContext:
        runAsNonRoot: true
        runAsUser: 2000
        runAsGroup: 2000
        seccompProfile: {{ type: RuntimeDefault }}
      containers:
        - name: split
          image: {crop_image}
          imagePullPolicy: IfNotPresent
          command:
            - /opt/venvs/scoring/bin/python
          args:
            - /opt/imintengine/scripts/run_lucas_crop_split_job.py
          env:
            - name: CROP_DISTILL_SOURCE_GIT_SHA
              value: "{source_git_sha}"
            - name: CROP_DISTILL_IMAGE
              value: "{crop_image}"
            - name: CROP_SOURCE_ACCESS_PLAN_SHA256
              value: "{source_access_plan_sha256}"
            - name: CROP_SOURCE_ACCESS_PLAN_POD_UID
              value: "{source_access_plan_pod_uid}"
            - name: CROP_SOURCE_ACCESS_COMPLETION_SHA256
              value: "{source_access_completion_sha256}"
            - name: CROP_SOURCE_ACCESS_COMPLETION_POD_UID
              value: "{source_access_completion_pod_uid}"
            - name: CROP_SOURCE_FREEZE_LEASE_PATH
              value: /var/run/crop-source-freeze/lease.json
            - name: HOME
              value: /work/home
            - name: TMPDIR
              value: /work/tmp
            - name: POD_UID
              valueFrom:
                fieldRef:
                  fieldPath: metadata.uid
          securityContext:
            allowPrivilegeEscalation: false
            capabilities: {{ drop: ["ALL"] }}
            readOnlyRootFilesystem: true
            runAsNonRoot: true
            runAsUser: 2000
            runAsGroup: 2000
          resources:
            requests: {{ cpu: "2", memory: "8Gi" }}
            limits: {{ cpu: "2", memory: "8Gi" }}
          volumeMounts:
            - name: training-data-cephfs
              mountPath: /cephfs/unified_v2_512
              subPath: unified_v2_512
              readOnly: true
            - name: training-data-cephfs
              mountPath: /cephfs/lucas
              subPath: lucas
              readOnly: true
            - name: training-data-cephfs
              mountPath: /cephfs/distill/crop_split
              subPath: distill/crop_split
            - name: training-data-cephfs
              mountPath: /cephfs/ops/crop-distill
              subPath: ops/crop-distill/split
            - name: training-data-cephfs
              mountPath: /cephfs/source-access-completion/completion.json
              subPath: ops/crop-distill/source-access/apply/{source_access_completion_pod_uid}/completion.json
              readOnly: true
            - name: training-data-cephfs
              mountPath: /cephfs/source-access-lock
              subPath: ops/crop-distill/source-access/locks
            - name: work
              mountPath: /work
            - name: crop-source-freeze-lease
              mountPath: /var/run/crop-source-freeze
              readOnly: true
      volumes:
        - name: training-data-cephfs
          persistentVolumeClaim: {{ claimName: training-data-cephfs }}
        - name: work
          emptyDir: {{}}
        - name: crop-source-freeze-lease
          configMap:
            name: crop-source-freeze-lease
            optional: false
            items:
              - key: lease.json
                path: lease.json
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
        extract_filter=sar_filter, sar_filter=sar_filter,
        extra_setup=cfg.get("extra_setup", "true  # no extra deps"),
        python_image=PYTHON_IMAGE)


def render_crop_distill(model: str) -> str:
    _validate_crop_distill_identity()
    if model not in CROP_MODELS:
        raise ValueError(f"unknown crop-distill model: {model}")
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
        model=model,
        source_git_sha=CROP_DISTILL_SOURCE_GIT_SHA,
        crop_image=CROP_DISTILL_IMAGE,
        split_manifest_sha256=CROP_DISTILL_SPLIT_MANIFEST_SHA256,
        run_uid=CROP_MODEL_UIDS[model],
        head_subpath=crop_protocol.crop_head_backing_dir(model).relative_to(
            crop_protocol.PVC_ROOT
        ),
        record_subpath=crop_protocol.crop_record_backing_dir(model).relative_to(
            crop_protocol.PVC_ROOT
        ),
    )


def render_crop_storage_prep() -> str:
    _validate_crop_distill_runtime_identity()
    header = (
        "# GENERATED by scripts/gen_ladder_manifests.py — do not edit.\n"
        "# One-shot least-privilege migration: prepares crop_split plus\n"
        "# isolated per-model head/record directories and the split record\n"
        "# directory plus isolated source-access evidence/lock roots. Run\n"
        "# before PLAN; broad PVC roots stay unchanged.\n"
        "# Plan: docs/experiments/ladder_distill_stage.md\n"
    )
    return header + CROP_STORAGE_PREP_TEMPLATE.format(
        source_git_sha=CROP_DISTILL_SOURCE_GIT_SHA,
        crop_image=CROP_DISTILL_IMAGE,
    )


def render_crop_source_access_plan() -> str:
    _validate_crop_distill_runtime_identity()
    _validate_crop_source_index_identity()
    header = (
        "# GENERATED by scripts/gen_ladder_manifests.py — do not edit.\n"
        "# Read-only PLAN: derives the exact post-window 2,074 LUCAS tile\n"
        "# candidates and publishes a canonical, immutable inventory.\n"
        "# This Job cannot change source-tile bytes or metadata.\n"
        "# Requires a fresh plan-phase crop-source-freeze watchdog lease.\n"
        "# Plan: docs/experiments/ladder_distill_stage.md\n"
    )
    return header + CROP_SOURCE_ACCESS_PLAN_TEMPLATE.format(
        source_git_sha=CROP_DISTILL_SOURCE_GIT_SHA,
        crop_image=CROP_DISTILL_IMAGE,
        source_index_sha256=CROP_SOURCE_ACCESS_INDEX_SHA256,
    )


def render_crop_source_access_apply() -> str:
    _validate_source_access_plan_authority()
    header = (
        "# GENERATED by scripts/gen_ladder_manifests.py — do not edit.\n"
        "# Metadata-only APPLY: consumes the reviewed PLAN hash and changes\n"
        "# only its root:root 0600 candidates to root:2000 0640. The RW\n"
        "# source mount is the unified_v2_512 dataset subPath, not PVC root.\n"
        "# Requires a fresh apply-phase crop-source-freeze watchdog lease.\n"
        "# Plan: docs/experiments/ladder_distill_stage.md\n"
    )
    return header + CROP_SOURCE_ACCESS_APPLY_TEMPLATE.format(
        source_git_sha=CROP_DISTILL_SOURCE_GIT_SHA,
        crop_image=CROP_DISTILL_IMAGE,
        source_index_sha256=CROP_SOURCE_ACCESS_INDEX_SHA256,
        plan_sha256=CROP_SOURCE_ACCESS_PLAN_SHA256,
        plan_pod_uid=CROP_SOURCE_ACCESS_PLAN_POD_UID,
    )


def render_crop_deny_egress() -> str:
    header = (
        "# GENERATED by scripts/gen_ladder_manifests.py — do not edit.\n"
        "# Crop-distill runtime Pods require no network after image pull.\n"
        "# This policy selects source-access, producer, consumer, and prep Pods.\n"
        "# Plan: docs/experiments/ladder_distill_stage.md\n"
    )
    return header + CROP_DENY_EGRESS_TEMPLATE


def render_lucas_crop_split() -> str:
    _validate_source_access_completion_authority()
    header = (
        "# GENERATED by scripts/gen_ladder_manifests.py — do not edit.\n"
        "# Freezes the LUCAS crop 70/30 distill/holdout split ONCE, before\n"
        "# any crop training touches LUCAS (grouped by tile; prior 71-point\n"
        "# freeze forced into holdout). Run before any crop-distill-<model>.\n"
        "# Requires a fresh split-phase crop-source-freeze watchdog lease.\n"
        "# Plan: docs/experiments/ladder_distill_stage.md\n"
    )
    return header + LUCAS_SPLIT_TEMPLATE.format(
        crop_image=CROP_DISTILL_IMAGE,
        source_git_sha=CROP_DISTILL_SOURCE_GIT_SHA,
        source_access_plan_sha256=CROP_SOURCE_ACCESS_PLAN_SHA256,
        source_access_plan_pod_uid=CROP_SOURCE_ACCESS_PLAN_POD_UID,
        source_access_completion_sha256=CROP_SOURCE_ACCESS_COMPLETION_SHA256,
        source_access_completion_pod_uid=CROP_SOURCE_ACCESS_COMPLETION_POD_UID,
    )


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


def render_non_crop_outputs() -> dict[Path, str]:
    """Render every legacy ladder/distill manifest without crop anchors.

    Payload commit A and producer commit B intentionally cannot render crop
    consumers yet.  Keeping the anchor-independent outputs available lets CI
    prove that those committed manifests remain regenerable during both
    bootstrap phases without weakening the crop generator's fail-closed
    identity checks.
    """
    outputs: dict[Path, str] = {}
    for model, base_rel in BASES.items():
        base_text = (REPO / base_rel).read_text()
        for rung in RUNGS:
            dest = OUT_DIR / f"ladder-r{rung}-{model}-job.yaml"
            outputs[dest] = render(model, rung, base_text)
        outputs[OUT_DIR / f"distill-{model}-job.yaml"] = render_distill(model)
    outputs[OUT_DIR / "distill-pinned-plots-job.yaml"] = (
        "# GENERATED by scripts/gen_ladder_manifests.py — do not edit.\n"
        "# Pins the shared NFI plot set every column's distillability OOF\n"
        "# scores. Run ONCE before any distill-<model> job.\n"
        "# Plan: docs/experiments/ladder_distill_stage.md\n"
        + PINNED_PLOTS_TEMPLATE.format(python_image=PYTHON_IMAGE)
    )
    return outputs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true",
                    help="verify the committed manifests match this generator")
    phase = ap.add_mutually_exclusive_group()
    phase.add_argument(
        "--crop-bootstrap-only",
        action="store_true",
        help=(
            "write/check only storage-prep and read-only source-access PLAN "
            "manifests (legacy name for the first crop bootstrap phase)"
        ),
    )
    phase.add_argument(
        "--crop-apply-only",
        action="store_true",
        help="write/check only the hash-bound source-access APPLY manifest",
    )
    phase.add_argument(
        "--crop-split-only",
        action="store_true",
        help="write/check only the completion-gated LUCAS split manifest",
    )
    phase.add_argument(
        "--non-crop-only",
        action="store_true",
        help=(
            "write/check only the anchor-independent ladder and legacy "
            "distill manifests"
        ),
    )
    args = ap.parse_args()

    if not args.non_crop_only:
        try:
            if args.crop_bootstrap_only:
                _validate_crop_distill_runtime_identity()
                _validate_crop_source_index_identity()
            elif args.crop_apply_only:
                _validate_source_access_plan_authority()
            elif args.crop_split_only:
                _validate_source_access_completion_authority()
            else:
                _validate_crop_distill_identity()
        except ValueError as exc:
            print(
                f"REFUSING crop-distill manifest generation: {exc}",
                file=sys.stderr,
            )
            return 2

    outputs: dict[Path, str] = {}
    if args.non_crop_only:
        outputs.update(render_non_crop_outputs())
    elif args.crop_bootstrap_only:
        outputs[OUT_DIR / "crop-distill-deny-egress.yaml"] = (
            render_crop_deny_egress()
        )
        outputs[OUT_DIR / "crop-distill-storage-prep-job.yaml"] = (
            render_crop_storage_prep()
        )
        outputs[OUT_DIR / "crop-source-access-plan-job.yaml"] = (
            render_crop_source_access_plan()
        )
    elif args.crop_apply_only:
        outputs[OUT_DIR / "crop-distill-deny-egress.yaml"] = (
            render_crop_deny_egress()
        )
        outputs[OUT_DIR / "crop-source-access-apply-job.yaml"] = (
            render_crop_source_access_apply()
        )
    elif args.crop_split_only:
        outputs[OUT_DIR / "crop-distill-deny-egress.yaml"] = (
            render_crop_deny_egress()
        )
        outputs[OUT_DIR / "lucas-crop-split-job.yaml"] = render_lucas_crop_split()
    else:
        outputs[OUT_DIR / "crop-distill-deny-egress.yaml"] = (
            render_crop_deny_egress()
        )
        outputs[OUT_DIR / "crop-source-access-plan-job.yaml"] = (
            render_crop_source_access_plan()
        )
        outputs[OUT_DIR / "crop-source-access-apply-job.yaml"] = (
            render_crop_source_access_apply()
        )
        outputs[OUT_DIR / "lucas-crop-split-job.yaml"] = render_lucas_crop_split()
        outputs[OUT_DIR / "crop-distill-storage-prep-job.yaml"] = (
            render_crop_storage_prep()
        )
    if not (
        args.crop_bootstrap_only
        or args.crop_apply_only
        or args.crop_split_only
        or args.non_crop_only
    ):
        outputs.update(render_non_crop_outputs())
        for model in BASES:
            outputs[OUT_DIR / f"crop-distill-{model}-job.yaml"] = (
                render_crop_distill(model)
            )

    if args.crop_bootstrap_only:
        downstream_paths = [
            OUT_DIR / "crop-source-access-apply-job.yaml",
            OUT_DIR / "lucas-crop-split-job.yaml",
            *[
                OUT_DIR / f"crop-distill-{model}-job.yaml"
                for model in CROP_MODELS
            ],
        ]
        present_downstream = [
            str(path.relative_to(REPO))
            for path in downstream_paths
            if path.exists()
        ]
        if present_downstream:
            print(
                "REFUSING crop PLAN bootstrap while stale downstream manifests "
                "exist: " + ", ".join(present_downstream),
                file=sys.stderr,
            )
            return 2
    elif args.crop_split_only:
        consumer_paths = [
            OUT_DIR / f"crop-distill-{model}-job.yaml"
            for model in CROP_MODELS
        ]
        present_consumers = [
            str(path.relative_to(REPO))
            for path in consumer_paths
            if path.exists()
        ]
        if present_consumers:
            print(
                "REFUSING crop split while stale consumer manifests exist: "
                + ", ".join(present_consumers),
                file=sys.stderr,
            )
            return 2
    elif args.crop_apply_only:
        downstream_paths = [
            OUT_DIR / "lucas-crop-split-job.yaml",
            *[
                OUT_DIR / f"crop-distill-{model}-job.yaml"
                for model in CROP_MODELS
            ],
        ]
        present_downstream = [
            str(path.relative_to(REPO))
            for path in downstream_paths
            if path.exists()
        ]
        if present_downstream:
            print(
                "REFUSING crop APPLY while stale downstream manifests exist: "
                + ", ".join(present_downstream),
                file=sys.stderr,
            )
            return 2

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
