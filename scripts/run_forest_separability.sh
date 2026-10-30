#!/usr/bin/env bash
# Run the pre-registered forest separability probe (IM-023) on ICE, one year at a
# time, from the operator's machine:
#
#   bash scripts/run_forest_separability.sh sumpskog5 2022
#   bash scripts/run_forest_separability.sh forest5 2023
#
# For each year: render the bounded CPU Job (2 CPU, 4 GiB, no GPU, <= 30 min),
# dry-run it on the server, create it, wait for OUTPUT_READY, copy the
# location-free result (probe.json + bundle hashes) to the private evidence
# directory, acknowledge so the Job ends, and print a summary. If extraction
# fails, the tile-selection counts (availability.json) and the end of the
# extraction log are copied first, before the acknowledgement discards them.
set -uo pipefail

[ $# -ge 2 ] || { echo "usage: $0 <scheme> <year> [year ...]" >&2; exit 2; }
SCHEME=$1; shift
CTX=icekube
NS=prithvi-training-default
ROOT=$(cd "$(dirname "$0")/.." && pwd)
EVIDENCE="$HOME/.agents/continuity/evidence/ImintEngine/class-separability-0127"
PY=$(command -v python3.13 || command -v python3.12 || command -v python3.11 || command -v python3) \
  || { echo "python3 saknas" >&2; exit 1; }
kc() { kubectl --context "$CTX" -n "$NS" "$@"; }
log() { printf '[%s] %s\n' "$(date '+%H:%M')" "$*"; }

for YEAR in "$@"; do
  out="$ROOT/outputs/forest-separability/${SCHEME}-${YEAR}-$(date +%Y%m%dT%H%M%S)"
  ev="$EVIDENCE/${SCHEME}-${YEAR}"
  ( cd "$ROOT" && "$PY" scripts/render_forest_separability_job.py --output "$out" \
      --scheme "$SCHEME" --feature-year "$YEAR" > /dev/null ) || { log "rendering misslyckades"; exit 1; }
  name=$("$PY" -c "import json,sys; print(json.load(open(sys.argv[1]))['job'])" "$out/identity.json")
  kc create --dry-run=server -f "$out/job.json" > /dev/null || { log "dry-run avvisades"; exit 1; }
  kc create -f "$out/job.json" > /dev/null || { log "kunde inte skapa $name"; exit 1; }
  log "$SCHEME $YEAR: $name skapad, väntar (max ~30 min)"
  pod=""
  until [ -n "$pod" ]; do sleep 3; pod=$(kc get pod -l job-name="$name" -o name 2>/dev/null | head -1); done
  status=""
  for _ in $(seq 1 220); do
    status=$(kc logs "$pod" 2>/dev/null | sed -n 's/^OUTPUT_READY \([01]\)$/\1/p' | tail -1)
    [ -n "$status" ] && break
    [ "$(kc get job "$name" -o jsonpath='{.status.failed}' 2>/dev/null)" = 1 ] && break
    sleep 10
  done
  mkdir -p "$ev"
  if [ "$status" = 0 ]; then
    kc cp "${pod#pod/}:/scratch/results/probe.json" "$ev/probe.json" > /dev/null
    kc exec "$pod" -- sha256sum /scratch/results/features.npz /scratch/results/features.json > "$ev/hashes.txt"
  else
    log "$SCHEME $YEAR: misslyckades (OUTPUT_READY ${status:-saknas}); sparar urval och logg"
    kc exec "$pod" -- cat /scratch/results/availability.json > "$ev/availability.json" 2>/dev/null
    kc exec "$pod" -- tail -n 8 /scratch/extract.log > "$ev/extract-tail.log" 2>/dev/null
  fi
  kc exec "$pod" -- touch /scratch/ACK 2>/dev/null
  "$PY" - "$ev" "$SCHEME" "$YEAR" <<'PY'
import json, pathlib, sys
ev, scheme, year = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3]
probe = ev / 'probe.json'
if not probe.is_file():
    for name in ('availability.json', 'extract-tail.log'):
        f = ev / name
        print(f'  {name}:', f.read_text().strip() if f.is_file() and f.stat().st_size else '(saknas)')
    sys.exit(0)
r = json.loads(probe.read_text())
names = list(r['models']['r4']['per_class'])
print(f"{scheme} {year}: {r['n_samples']} pixlar, {r['n_tiles']} tiles")
for model in ('r4', 'tessera'):
    m = r['models'][model]
    print(f"  {model}: balanserad träffsäkerhet {m['balanced_accuracy']:.3f}, 95% "
          f"[{m['balanced_accuracy_95pct_block_bootstrap'][0]:.3f}, {m['balanced_accuracy_95pct_block_bootstrap'][1]:.3f}]")
    for name, row in zip(names, m['confusion_matrix']):
        total = sum(row) or 1
        print(f"    {name:32s} -> " + ', '.join(f"{n.split(' (')[0]} {100 * v / total:.0f}%" for n, v in zip(names, row) if v))
for key in ('distilled_label', 'r4_prediction'):
    print(f"  {key} (unified-klass per NMD-klass):")
    for name, counts in r['assigned_unified_class'][key].items():
        print(f"    {name:32s} -> {counts}")
PY
done
log "Klart. Resultat i $EVIDENCE"
