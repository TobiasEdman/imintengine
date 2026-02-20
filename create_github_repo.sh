#!/usr/bin/env bash
# Run these commands once to create and push the repo to GitHub.
# Requires: git, gh (GitHub CLI) — install gh with: brew install gh

set -e

# ── 1. Init git ───────────────────────────────────────────────────────────────
git init
git add .
git commit -m "feat: initial IMINT engine

- Modular analyzer architecture (change detection, spectral, object detection)
- Executor abstraction: LocalExecutor and ColonyOSExecutor
- Engine is fully decoupled from job scheduler
- ColonyOS is one optional executor, not a hard dependency
- CLI via executors/local.py
- Config-driven via config/analyzers.yaml"

# ── 2. Create GitHub repo and push ───────────────────────────────────────────
gh repo create imint-engine \
  --public \
  --description "Modular satellite image intelligence engine for SDL 3.0 / DES" \
  --source=. \
  --remote=origin \
  --push

echo ""
echo "✓ Repo created and pushed."
echo "  Visit: https://github.com/$(gh api user --jq .login)/imint-engine"
