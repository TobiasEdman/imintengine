#!/usr/bin/env bash
# Migrate every tracked reference to the old GitHub location after the
# repo transfer to the DES organisation (prepared 2026-07-07, before the
# transfer — the target org name is passed as an argument on transfer day).
#
# GitHub sets up permanent redirects on transfer, so nothing breaks
# immediately — this migration exists so the ~160 hardcoded clone/raw
# URLs (k8s Job yamls, dashboard deployment, docs) stop depending on a
# redirect that dies the day a same-named repo is recreated under the
# old account.
#
# Usage:
#   scripts/migrate_repo_urls.sh <new-org> [new-repo-name]            # DRY-RUN
#   scripts/migrate_repo_urls.sh <new-org> [new-repo-name] --apply    # rewrite
#
# After --apply:
#   1. Review:  git diff --stat && git diff | head -100
#   2. Commit + push (redirect still carries the push if remote not yet updated)
#   3. Update local remote:
#        git remote set-url origin https://github.com/<new-org>/<name>.git
#   4. Rollout-restart long-lived deployments that clone at pod start
#      (campaign-dashboard) at a convenient time; one-shot Jobs pick the
#      new URL up automatically at next submit.
set -euo pipefail

OLD_ORG="TobiasEdman"
OLD_NAMES=("ImintEngine" "imintengine")   # both casings exist in the tree

NEW_ORG="${1:?usage: migrate_repo_urls.sh <new-org> [new-repo-name] [--apply]}"
NEW_NAME="ImintEngine"
APPLY=0
for arg in "${@:2}"; do
  case "$arg" in
    --apply) APPLY=1 ;;
    *)       NEW_NAME="$arg" ;;
  esac
done

cd "$(git rev-parse --show-toplevel)"

# URL prefixes under which the repo path appears. Plain github.com covers
# clone/push/web; raw.githubusercontent.com covers the in-pod
# verify-remote fetches; mybinder.org's /gh/ covers the notebook badges
# (Binder builds FROM the repo, so a stale badge builds the old fork).
PREFIXES=("github\\.com/" "raw\\.githubusercontent\\.com/" "mybinder\\.org/v2/gh/")

# Tracked text files that mention the old location under any prefix.
# (while-read, not mapfile — macOS ships bash 3.2 which lacks mapfile.)
FILES=()
while IFS= read -r f; do
  FILES+=("$f")
done < <(git grep -Il -e "github.com/${OLD_ORG}/" \
                     -e "raw.githubusercontent.com/${OLD_ORG}/" \
                     -e "mybinder.org/v2/gh/${OLD_ORG}/" -- . | sort -u)

echo "Old : github.com/${OLD_ORG}/{${OLD_NAMES[*]}}"
echo "New : github.com/${NEW_ORG}/${NEW_NAME}"
echo "Hits: ${#FILES[@]} tracked files"
[ "${#FILES[@]}" -eq 0 ] && { echo "Nothing to do."; exit 0; }

for f in "${FILES[@]}"; do
  n=$(grep -cE "(github\.com|raw\.githubusercontent\.com|mybinder\.org/v2/gh)/${OLD_ORG}/" "$f" || true)
  if [ "$APPLY" -eq 1 ]; then
    for prefix in "${PREFIXES[@]}"; do
      for old in "${OLD_NAMES[@]}"; do
        # Bounded match: name must end at a word boundary / URL delimiter
        # (incl. markdown backtick) so sibling repos under the same account
        # (des-contracts, space-data-lab, ...) are never rewritten. The
        # replacement keeps each prefix (raw stays raw, binder stays binder).
        perl -pi -e "s{(${prefix})${OLD_ORG}/${old}(?=[/\\.\"'\\s)\\\`,;:?\\]]|\$)}{\${1}${NEW_ORG}/${NEW_NAME}}g" "$f"
      done
    done
    echo "  rewrote $n ref(s): $f"
  else
    echo "  would rewrite $n ref(s): $f"
  fi
done

if [ "$APPLY" -eq 1 ]; then
  # Count what remains for THIS repo's names only — other repos under the
  # old account (des-contracts, space-data-lab, ...) legitimately stay.
  left=0
  for old in "${OLD_NAMES[@]}"; do
    n=$(git grep -IE "(github\.com|raw\.githubusercontent\.com|mybinder\.org/v2/gh)/${OLD_ORG}/${old}([/.\"'\s)\`,;:?\]]|\$)" -- . | wc -l | tr -d ' ')
    left=$((left + n))
  done
  echo ""
  echo "Remaining ${OLD_NAMES[*]} refs after rewrite: $left (expect 0;"
  echo "other repos under ${OLD_ORG} are intentionally untouched)"
  echo "Next: review diff -> commit -> push -> git remote set-url origin" \
       "https://github.com/${NEW_ORG}/${NEW_NAME}.git"
else
  echo ""
  echo "DRY-RUN ONLY — re-run with --apply to rewrite."
fi
