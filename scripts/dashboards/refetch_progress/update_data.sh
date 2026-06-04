#!/usr/bin/env bash
# Polls kubectl for the refetch-late-autumn-512 pod and writes data.json
# in the same directory. Designed to run in a background loop.
set -u

NS="prithvi-training-default"
LABEL="job-name=refetch-late-autumn-512"
OUT="$(dirname "$0")/data.json"
TOTAL_TILES=6786

while true; do
  POD=$(kubectl get pod -l "$LABEL" -n "$NS" \
        -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)
  STATUS=$(kubectl get pod -l "$LABEL" -n "$NS" \
        -o jsonpath='{.items[0].status.phase}' 2>/dev/null)
  AGE=$(kubectl get pod -l "$LABEL" -n "$NS" --no-headers 2>/dev/null \
        | awk '{print $5}')

  # Pull the most recent progress line + the last status counter
  L=$(kubectl logs "$POD" -n "$NS" --tail=200 2>/dev/null)
  LAST=$(echo "$L" | grep -oE "\[[0-9]+/[0-9]+\] status=\{[^}]*\}.*ETA=[0-9.]+min" | tail -1)

  if [ -n "$LAST" ]; then
    SCANNED=$(echo "$LAST" | grep -oE "\[[0-9]+" | grep -oE "[0-9]+")
    OK=$(echo "$LAST"      | grep -oE "'ok': [0-9]+"      | grep -oE "[0-9]+")
    FAIL=$(echo "$LAST"    | grep -oE "'failed': [0-9]+"  | grep -oE "[0-9]+")
    SKIP=$(echo "$LAST"    | grep -oE "'skipped': [0-9]+" | grep -oE "[0-9]+")
    ERR=$(echo "$LAST"     | grep -oE "'error': [0-9]+"   | grep -oE "[0-9]+")
    RATE=$(echo "$LAST"    | grep -oE "rate=[0-9]+/h"     | grep -oE "[0-9]+")
    ETA=$(echo "$LAST"     | grep -oE "ETA=[0-9.]+min"    | grep -oE "[0-9.]+")
  else
    SCANNED=0; OK=0; FAIL=0; SKIP=0; ERR=0; RATE=""; ETA=""
  fi

  cat > "$OUT" <<JSON
{
  "timestamp": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
  "pod": {
    "name":   "${POD:-unknown}",
    "status": "${STATUS:-unknown}",
    "age":    "${AGE:-—}"
  },
  "progress": {
    "scanned": ${SCANNED:-0},
    "total":   $TOTAL_TILES,
    "ok":      ${OK:-0},
    "failed":  ${FAIL:-0},
    "skipped": ${SKIP:-0},
    "error":   ${ERR:-0},
    "rate_per_hour": ${RATE:-null},
    "eta_min":       ${ETA:-null}
  }
}
JSON
  sleep 30
done
