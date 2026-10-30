#!/usr/bin/env bash
# Bring up the full IM-024 demo (ten municipalities + Kiruna tätort, 2017-2025)
# from the operator's machine, unattended, in one run:
#
#   1. wait for the WEkEO HR-VPP gap-fill Job (k8s/lclu-vpp-gapfill-job.yaml);
#   2. store its campaign preflight in git (docs/verification/lclu-service/);
#   3. size the demo volume for every source-ready area-year, expand the PVC,
#      render the campaign Job r8 and create it beside the serving demo r7;
#   4. wait until r8 has staged, qualified and serves;
#   5. move the 127.0.0.1:8788 port-forward to r8 and stop r7.
#
# Every step checks whether it is already done, so after an interruption
# (sleep, network loss, Ctrl+C) run the same command again. Decisions by Tobias
# 2026-10-30: all source-ready cells after the gap-fill, parallel switch without
# downtime, r8 ends with r7's approved week. One confirmation at start.
#
#   bash docker/lclu/run_full_demo.sh [--yes]
#
# On macOS it keeps the machine awake (caffeinate; the lid must stay open).
# Log: ~/lclu-full-demo.log
set -uo pipefail

CTX=icekube
NS=prithvi-training-default
VPP_JOB=lclu-vpp-gapfill-0128
OLD=lclu-stockholm-0124-r7
NEW=lclu-stockholm-0124-r8
PVC=lclu-stockholm-0124
BRANCH=claude/upbeat-goldberg-bnbbux
BASE=python@sha256:05cda9777409a9c3ffddd94a4c476b79f0769a0b4857f0c7ed9226b6800b0d6f
YEARS="2017 2018 2019 2020 2021 2022 2023 2024 2025"
POLL=${LCLU_POLL_SECONDS:-300}
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
PREFLIGHT=docs/verification/lclu-service/campaign-preflight.json
PLAN=docs/verification/lclu-service/vpp-gapfill-plan.json
OUT="$ROOT/outputs/lclu-campaign-r8"

if [ -z "${LCLU_AWAKE:-}" ] && command -v caffeinate >/dev/null 2>&1; then
  export LCLU_AWAKE=1
  exec caffeinate -i -s bash "$0" "$@"
fi
exec > >(tee -a "${LCLU_LOG:-$HOME/lclu-full-demo.log}") 2>&1

log() { printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M')" "$*"; }
die() { log "STOPP: $*"; log "r7 serverar oförändrat. Kör skriptet igen när orsaken är åtgärdad."; exit 1; }
kc() { kubectl --context "$CTX" -n "$NS" "$@"; }
PY=$(command -v python3.13 || command -v python3.12 || command -v python3.11 || command -v python3) \
  || die "python3 saknas"

job_state() {  # succeeded | failed | running | absent | unknown
  local out
  if ! out=$(kc get job "$1" -o jsonpath='{.status.succeeded}/{.status.failed}' 2>&1); then
    case "$out" in *NotFound*|*"not found"*) echo absent ;; *) echo unknown ;; esac
    return
  fi
  case "$out" in 1/*) echo succeeded ;; */1) echo failed ;; *) echo running ;; esac
}
pod_ready() {
  kc get pod -l job-name="$1" -o jsonpath='{.items[0].status.conditions[?(@.type=="Ready")].status}' 2>/dev/null
}
preflight_is_new() {  # the committed preflight is no longer the pre-gap-fill one
  "$PY" - "$ROOT/$PLAN" "$ROOT/$PREFLIGHT" <<'PY'
import hashlib, json, sys
old = json.load(open(sys.argv[1]))['preflight_sha256']
now = hashlib.sha256(open(sys.argv[2], 'rb').read()).hexdigest()
sys.exit(0 if now != old else 1)
PY
}

log "== Hela demon: 10 kommuner + Kiruna tätort, $YEARS"
log "Källa: $(git -C "$ROOT" log -1 --format='%h %s')"
log "Läge: VPP=$(job_state "$VPP_JOB") r7=$(job_state "$OLD") r8=$(job_state "$NEW")"
if [ "${1:-}" != "--yes" ]; then
  printf '\nSkriptet väntar in HR-VPP, sparar preflighten i git, expanderar demovolymen,\n'
  printf 'startar r8 bredvid r7, väntar tills r8 svarar (kan ta över ett dygn), flyttar\n'
  printf 'port 8788 till r8 och stoppar r7. Inga fler frågor efter detta. Kör? [y/N] '
  read -r ans </dev/tty
  [ "$ans" = y ] || { log "Avbrutet, inget ändrat."; exit 0; }
fi

# 1-2. HR-VPP gap-fill and its preflight -------------------------------------
if preflight_is_new; then
  log "1-2. Preflighten efter luckfyllnaden finns redan i git."
else
  n=0
  while :; do
    case "$(job_state "$VPP_JOB")" in
      succeeded) log "1. HR-VPP-luckfyllnaden är klar."; break ;;
      failed) die "HR-VPP-jobbet misslyckades: kubectl --context $CTX -n $NS logs job/$VPP_JOB | tail -50" ;;
      absent) die "Hittar inte jobbet $VPP_JOB." ;;
    esac
    [ $((n % 6)) -eq 0 ] && log "1. Väntar på HR-VPP-luckfyllnaden ..."
    n=$((n + 1)); sleep "$POLL"
  done
  joblog=$(mktemp)
  kc logs "job/$VPP_JOB" > "$joblog" || die "kunde inte läsa loggen från $VPP_JOB"
  grep 'PREFETCH_STATUS' "$joblog" | sed 's/^/    /'
  "$PY" - "$joblog" "$ROOT/$PREFLIGHT" <<'PY' || die "preflighten saknas eller är ogiltig i VPP-loggen"
import collections, json, sys
lines = open(sys.argv[1], encoding='utf-8').read().splitlines()
body = '\n'.join(lines[lines.index('PREFLIGHT_JSON_BEGIN') + 1:lines.index('PREFLIGHT_JSON_END')]) + '\n'
report = json.loads(body)
if not report['cells']:
    raise SystemExit('empty preflight')
open(sys.argv[2], 'w', encoding='utf-8').write(body)
print('    celler:', dict(collections.Counter(c['status'] for c in report['cells'])))
PY
  if ! { git -C "$ROOT" fetch -q origin "$BRANCH" && git -C "$ROOT" merge -q --ff-only FETCH_HEAD; }; then
    log "VARNING: kunde inte uppdatera från GitHub; committar lokalt."
  fi
  git -C "$ROOT" add "$PREFLIGHT"
  git -C "$ROOT" commit -q -m "Record campaign preflight after the HR-VPP gap-fill (task 0128)" \
    -m "Extracted unchanged from the $VPP_JOB log by docker/lclu/run_full_demo.sh." \
    || die "kunde inte committa preflighten"
  pushed=no
  for i in 1 2 3 4; do
    if git -C "$ROOT" push -q origin "HEAD:$BRANCH"; then pushed=yes; break; fi
    sleep $((2 ** i))
  done
  if [ "$pushed" = yes ]; then
    log "2. Preflighten committad och pushad."
  else
    log "VARNING: push misslyckades; committen finns lokalt. Fortsätter."
  fi
fi

# 3. Volume and campaign Job r8 ----------------------------------------------
if [ "$(job_state "$NEW")" = absent ]; then
  [ "$(job_state "$OLD")" != absent ] || die "r7 saknas, så veckans slut går inte att räkna ut."
  read -r budget cells <<<"$("$PY" -c "import json,sys; p=json.load(open(sys.argv[1])); print(p['source_ready_original_bytes'], sum(c['status']=='source_ready' for c in p['cells']))" "$ROOT/$PREFLIGHT")"
  [ "${cells:-0}" -gt 0 ] || die "inga källklara celler i preflighten"
  gib=$("$PY" -c "import math,sys; b,c=int(sys.argv[1]),int(sys.argv[2]); print(math.ceil(b*2.3/2**30) + math.ceil(c*0.1) + 30)" "$budget" "$cells")
  deadline=$(kc get job "$OLD" -o jsonpath='{.status.startTime}' | "$PY" -c "import sys,datetime as d; s=d.datetime.fromisoformat(sys.stdin.read().strip().replace('Z','+00:00')); print(int((s+d.timedelta(days=7)-d.datetime.now(d.timezone.utc)).total_seconds()))")
  [ "${deadline:-0}" -gt 3600 ] || die "under en timme kvar av den godkända veckan"
  [ "$(kubectl --context "$CTX" get sc rook-ceph-fs -o jsonpath='{.allowVolumeExpansion}')" = true ] \
    || die "lagringsklassen rook-ceph-fs tillåter inte volymexpansion"
  log "3. $cells källklara celler, $((budget / 1000000000)) GB original -> volym ${gib} GiB (~$((gib * 2)) kr/mån), r8 i $((deadline / 3600)) h."
  ( cd "$ROOT" && "$PY" docker/lclu/build.py --tag imint-lclu-demo:campaign-r8 \
      --output "$OUT/recipe" --bundle-only ) || die "kunde inte bygga källarkivet"
  # shellcheck disable=SC2086  # YEARS is a deliberate word list
  ( cd "$ROOT" && "$PY" docker/lclu/render_job.py --image "$BASE" --recipe "$OUT/recipe/source.tar.gz" \
      --year 2024 --job-name "$NEW" --reuse-volume --deadline-seconds "$deadline" \
      --campaign-years $YEARS --storage-gib "$gib" --max-original-bytes "$budget" \
      --output "$OUT/demo-job-r8.json" ) || die "kunde inte rendera r8"
  kc create --dry-run=server -f "$OUT/demo-job-r8.json" > /dev/null || die "klustret avvisade r8 i dry-run"
  current=$(kc get pvc "$PVC" -o jsonpath='{.spec.resources.requests.storage}')
  if [ "${current%Gi}" -lt "$gib" ] 2>/dev/null; then
    kc patch pvc "$PVC" -p "{\"spec\":{\"resources\":{\"requests\":{\"storage\":\"${gib}Gi\"}}}}" > /dev/null \
      || die "kunde inte expandera volymen"
  fi
  i=0
  until [ "$(kc get pvc "$PVC" -o jsonpath='{.status.capacity.storage}')" = "${gib}Gi" ]; do
    i=$((i + 1)); [ "$i" -le 60 ] || die "volymen har inte vuxit till ${gib}Gi efter 10 min"
    sleep 10
  done
  kc create -f "$OUT/demo-job-r8.json" > /dev/null || die "kunde inte skapa r8"
  log "3. Volymen är ${gib}Gi och r8 är skapad."
else
  log "3. r8 finns redan ($(job_state "$NEW"))."
fi

# 4. Wait until r8 serves ------------------------------------------------------
n=0
until [ "$(pod_ready "$NEW")" = True ]; do
  [ "$(job_state "$NEW")" = failed ] && die "r8 misslyckades: kubectl --context $CTX -n $NS logs job/$NEW | tail -50"
  if [ $((n % 6)) -eq 0 ]; then
    pod=$(kc get pod -l job-name="$NEW" -o name 2>/dev/null | head -1)
    done_cells=$(kc exec "$pod" -- cat /work/campaign/progress.json 2>/dev/null | "$PY" -c "import json,sys; c=json.load(sys.stdin)['cells']; print(sum(x['status'] in ('prepared','qualified') for x in c), 'av', sum(x['status'] in ('source_ready','prepared','qualified') for x in c))" 2>/dev/null)
    log "4. r8 förbereder: ${done_cells:-ingen cell klar än} celler."
  fi
  n=$((n + 1)); sleep "$POLL"
done
log "4. r8 svarar."

# 5. Switch 8788 to r8 and stop r7 -------------------------------------------
if [ "$(job_state "$OLD")" != absent ]; then
  pkill -f "port-forward.*8789:8000" 2>/dev/null
  kc port-forward --address 127.0.0.1 "job/$NEW" 8789:8000 > /dev/null 2>&1 &
  probe=$!
  areas=""
  for i in 1 2 3 4 5 6; do
    sleep 5
    areas=$(curl -s --max-time 10 http://127.0.0.1:8789/v1/lclu | "$PY" -c "import json,sys; print(len(json.load(sys.stdin)['areas']))" 2>/dev/null) && break
  done
  kill "$probe" 2>/dev/null
  [ -n "$areas" ] && [ "$areas" -gt 0 ] || die "r8 svarar inte med några områden på /v1/lclu"
  log "5. r8 annonserar $areas områden. Flyttar 8788 till r8."
  pkill -f "port-forward.*8788:8000" 2>/dev/null
  nohup kubectl --context "$CTX" -n "$NS" port-forward --address 127.0.0.1 "job/$NEW" 8788:8000 \
    > "$HOME/lclu-portforward.log" 2>&1 &
  code=""
  for i in 1 2 3 4 5 6; do
    sleep 5
    code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 10 http://127.0.0.1:8788/lclu)
    [ "$code" = 200 ] && break
  done
  [ "$code" = 200 ] || die "8788 svarar inte från r8 (HTTP $code); r7 är inte stoppad"
  kc delete job "$OLD" > /dev/null || log "VARNING: kunde inte stoppa r7"
fi
log "KLART: hela demon på http://127.0.0.1:8788/lclu (r8). Logg: ${LCLU_LOG:-$HOME/lclu-full-demo.log}"
