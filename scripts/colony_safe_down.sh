#!/usr/bin/env bash
# scripts/colony_safe_down.sh — Safety wrapper for docker compose down
#
# Prevents accidental volume destruction.
# Usage:
#   ./scripts/colony_safe_down.sh          # Safe stop (no volumes)
#   ./scripts/colony_safe_down.sh --reset  # Interactive reset with confirmation

set -euo pipefail

export PATH="/usr/local/bin:/opt/homebrew/bin:$PATH"

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
COMPOSE_FILE="$PROJECT_DIR/docker-compose.colonyos.yml"
ENV_FILE="$PROJECT_DIR/.env.colonyos"

# Source env for docker compose
set -a
# shellcheck disable=SC1091
[ -f "$ENV_FILE" ] && source "$ENV_FILE"
set +a

COMPOSE_CMD="docker compose -f $COMPOSE_FILE"

if [ "${1:-}" = "--reset" ]; then
    echo ""
    echo "  WARNING: This will DESTROY all ColonyOS data:"
    echo "    - TimescaleDB (all job history, colony state)"
    echo "    - MinIO (all CFS-stored tiles)"
    echo "    - Colonies etcd (server state)"
    echo "    - Docker registry (cached images)"
    echo ""

    # Show current tile count
    CFS_DIR="$HOME/cfs"
    if [ -d "$CFS_DIR" ]; then
        tile_count=$(find "$CFS_DIR" -name "*.npz" 2>/dev/null | wc -l | tr -d ' ')
        cfs_size=$(du -sh "$CFS_DIR" 2>/dev/null | cut -f1)
        echo "  CFS contains: $tile_count tiles ($cfs_size)"
        echo "  NOTE: CFS bind mount ($CFS_DIR) is NOT deleted by this command."
        echo ""
    fi

    read -p "  Type 'DESTROY' to confirm volume deletion: " confirm
    if [ "$confirm" = "DESTROY" ]; then
        echo "  Stopping and removing volumes..."
        $COMPOSE_CMD down -v 2>&1 | grep -v "level=warning"
        echo "  Done. All ColonyOS docker volumes have been removed."
        echo "  CFS bind mount ($CFS_DIR) is still intact."
    else
        echo "  Aborted. No data was deleted."
        exit 1
    fi
else
    echo "  Stopping ColonyOS services (data preserved)..."
    $COMPOSE_CMD down 2>&1 | grep -v "level=warning"
    echo "  Done. Volumes preserved. Use 'make colony-up' to restart."
fi
