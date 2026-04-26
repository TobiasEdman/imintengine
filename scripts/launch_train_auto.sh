#!/bin/bash
# scripts/launch_train_auto.sh — Launch a unified-train-* job on the
# best available accelerator.
#
# Strategy:
#   1. Try H100 first by submitting via launch_train.sh with
#      ACCELERATOR=nvidia-h100. Wait up to PROBE_TIMEOUT_S (default 90s)
#      for the pod to reach Pending → Running, OR for the job-controller
#      to report a webhook denial (project H100 quota 8/8 used).
#   2. If H100 schedules: keep it running, exit 0.
#   3. If H100 is denied AND --fallback-2080 is set (default true):
#      delete the H100 job, re-submit with ACCELERATOR=nvidia-gtx-2080ti.
#      88 GPUs in the cluster, never contended.
#
# Usage:
#   RUN_ID=v7d-prithvi300 BACKBONE_NAME=prithvi_300m \
#     ./scripts/launch_train_auto.sh
#
#   # Force a specific accelerator (skip auto-probe):
#   ACCELERATOR=nvidia-h100 RUN_ID=... BACKBONE_NAME=... \
#     ./scripts/launch_train.sh
#
# Exit codes:
#   0 — pod scheduled (on H100 or 2080 Ti)
#   1 — both classes failed (no H100 quota, no 2080 Ti either; user error
#       or systemic cluster outage)

set -euo pipefail

: "${RUN_ID:?RUN_ID is required}"
: "${BACKBONE_NAME:?BACKBONE_NAME is required}"

NS="${NAMESPACE:-prithvi-training-default}"
PROBE_TIMEOUT_S="${PROBE_TIMEOUT_S:-90}"
FALLBACK_2080="${FALLBACK_2080:-1}"
JOB_NAME="unified-train-${RUN_ID}"
HERE="$(dirname "$0")"

probe_pod_state() {
    # Returns: "running" | "pending" | "denied" | "unknown"
    local active failed pending
    active=$(kubectl get job "$JOB_NAME" -n "$NS" \
                -o jsonpath='{.status.active}' 2>/dev/null || echo "")
    failed=$(kubectl get job "$JOB_NAME" -n "$NS" \
                -o jsonpath='{.status.failed}' 2>/dev/null || echo "")
    # Look for the H100 quota webhook denial
    local denied
    denied=$(kubectl get events -n "$NS" \
        --field-selector "involvedObject.name=$JOB_NAME,reason=FailedCreate" \
        -o jsonpath='{.items[*].message}' 2>/dev/null \
        | grep -c "Not enough GPU quota" || true)

    if [ "${active:-0}" = "1" ]; then
        # Pod scheduled — but Pending vs Running matters less here than
        # the fact that admission accepted it.
        echo "running"
        return
    fi
    if [ "${denied:-0}" -gt 0 ]; then
        echo "denied"
        return
    fi
    if [ "${failed:-0}" = "1" ]; then
        echo "failed"
        return
    fi
    echo "pending"
}

attempt_h100() {
    echo "=== Trying H100 first ==="
    ACCELERATOR=nvidia-h100 RUN_ID="$RUN_ID" BACKBONE_NAME="$BACKBONE_NAME" \
        IMG_SIZE="${IMG_SIZE:-}" BATCH_SIZE="${BATCH_SIZE:-}" LR="${LR:-}" \
        EPOCHS="${EPOCHS:-}" DISABLE_BF16="${DISABLE_BF16:-}" \
        COLLAPSE_REWIND="${COLLAPSE_REWIND:-}" \
        "$HERE/launch_train.sh"

    local end=$(( $(date +%s) + PROBE_TIMEOUT_S ))
    local state="unknown"
    while [ "$(date +%s)" -lt "$end" ]; do
        sleep 6
        state=$(probe_pod_state)
        case "$state" in
            running)
                echo "✓ H100 pod scheduled within ${PROBE_TIMEOUT_S}s probe window."
                return 0
                ;;
            denied)
                echo "✗ H100 quota denial after $((end - $(date +%s) + PROBE_TIMEOUT_S))s of probing."
                return 1
                ;;
            failed)
                echo "✗ H100 job failed during probe."
                return 1
                ;;
        esac
    done
    # Timed out without a definitive verdict.
    state=$(probe_pod_state)
    if [ "$state" = "denied" ]; then
        echo "✗ H100 quota denial detected at end of probe window."
        return 1
    fi
    if [ "$state" = "running" ]; then
        echo "✓ H100 pod scheduled at end of probe window."
        return 0
    fi
    echo "? H100 probe timed out after ${PROBE_TIMEOUT_S}s with state=$state."
    echo "  Treating as denied — H100 not available."
    return 1
}

fallback_2080() {
    echo ""
    echo "=== Falling back to 2080 Ti ==="
    # Clean up the H100 job before re-submitting under the same RUN_ID.
    kubectl delete job "$JOB_NAME" -n "$NS" --ignore-not-found 2>&1 | sed 's/^/  /'
    sleep 3
    ACCELERATOR=nvidia-gtx-2080ti RUN_ID="$RUN_ID" BACKBONE_NAME="$BACKBONE_NAME" \
        IMG_SIZE="${IMG_SIZE:-}" EPOCHS="${EPOCHS:-}" \
        COLLAPSE_REWIND="${COLLAPSE_REWIND:-}" \
        "$HERE/launch_train.sh"
    sleep 6
    local state
    state=$(probe_pod_state)
    if [ "$state" = "running" ] || [ "$state" = "pending" ]; then
        echo "✓ 2080 Ti pod scheduled."
        return 0
    fi
    echo "✗ 2080 Ti also failed (state=$state)."
    return 1
}

if attempt_h100; then
    exit 0
fi

if [ "$FALLBACK_2080" = "1" ]; then
    if fallback_2080; then
        exit 0
    fi
fi

exit 1
