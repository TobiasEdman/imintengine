#!/usr/bin/env bash
# Launch LUCAS L2 validation jobs (one per member) by substituting the
# placeholders in k8s/lucas-validate-job.template.yaml. Idempotent-ish:
# re-applying replaces the Job spec (delete the old Job first if it ran).
#
#   scripts/launch_lucas_validate.sh [member ...]
#   (no args → the three 28-class campaign members)
set -euo pipefail

CTX=${CTX:-icekube}
NS=${NS:-prithvi-training-default}
TEMPLATE="$(dirname "$0")/../k8s/lucas-validate-job.template.yaml"

# member | checkpoint dir | backbone | extra flags
declare -A CKPT=(
  [distill]=v8b_nmd2023_distill
  [tradslag]=v8b_nmd2023_tradslag
  [tessera]=tessera_distill
  [v8b_markfukt]=v8b_markfukt
  [v8b_nmd2023_long]=v8b_nmd2023_long
)
declare -A BACKBONE=(
  [distill]=prithvi_600m [tradslag]=prithvi_600m [tessera]=tessera_v1
  [v8b_markfukt]=prithvi_600m [v8b_nmd2023_long]=prithvi_600m
)
declare -A EXTRA=( [v8b_markfukt]="--enable-markfukt" )

members=("$@")
[ ${#members[@]} -eq 0 ] && members=(distill tradslag tessera)

for m in "${members[@]}"; do
  sed -e "s/__MEMBER__/$m/g" \
      -e "s#__CKPT__#${CKPT[$m]}#g" \
      -e "s/__BACKBONE__/${BACKBONE[$m]}/g" \
      -e "s/__EXTRA__/${EXTRA[$m]:-}/g" \
      "$TEMPLATE" | kubectl --context "$CTX" -n "$NS" apply -f -
done
