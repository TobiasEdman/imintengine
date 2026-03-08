#!/usr/bin/env bash
# scripts/colony_backup.sh — Backup critical ColonyOS state
#
# Creates timestamped backups of:
#   1. Config/credentials (.env.colonyos, .cdse_credentials)
#   2. CFS tile data (from host bind mount ~/cfs/)
#   3. TimescaleDB dump (job history)
#
# Usage:
#   ./scripts/colony_backup.sh                    # Full backup
#   ./scripts/colony_backup.sh --config-only      # Just config/creds
#
# Backups go to ~/imint_backups/<timestamp>/

set -euo pipefail

export PATH="/usr/local/bin:/opt/homebrew/bin:$PATH"

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
COMPOSE_FILE="$PROJECT_DIR/docker-compose.colonyos.yml"
ENV_FILE="$PROJECT_DIR/.env.colonyos"
CFS_DIR="$HOME/cfs"
BACKUP_ROOT="$HOME/imint_backups"
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
BACKUP_DIR="$BACKUP_ROOT/$TIMESTAMP"

mkdir -p "$BACKUP_DIR"

echo "ColonyOS Backup — $TIMESTAMP"
echo "  Backup dir: $BACKUP_DIR"

# ── 1. Config and credentials ─────────────────────────────────────────
echo ""
echo "  [1/3] Backing up config..."
mkdir -p "$BACKUP_DIR/config"
for f in .env.colonyos .cdse_credentials .env .skg_endpoints; do
    if [ -f "$PROJECT_DIR/$f" ]; then
        cp "$PROJECT_DIR/$f" "$BACKUP_DIR/config/"
        echo "    Saved: $f"
    fi
done

# ── 2. CFS tile data (rsync for incremental) ──────────────────────────
if [ "${1:-}" != "--config-only" ] && [ -d "$CFS_DIR" ]; then
    echo ""
    echo "  [2/3] Backing up CFS tiles..."
    cfs_size=$(du -sh "$CFS_DIR" 2>/dev/null | cut -f1)
    tile_count=$(find "$CFS_DIR" -name "*.npz" 2>/dev/null | wc -l | tr -d ' ')
    echo "    CFS size: $cfs_size ($tile_count tiles)"

    # Use rsync with --link-dest for space-efficient incremental backups
    latest_link="$BACKUP_ROOT/latest"
    if [ -L "$latest_link" ] && [ -d "$latest_link/cfs" ]; then
        rsync -a --link-dest="$latest_link/cfs" "$CFS_DIR/" "$BACKUP_DIR/cfs/"
        echo "    Incremental rsync complete (hardlinked unchanged files)"
    else
        rsync -a "$CFS_DIR/" "$BACKUP_DIR/cfs/"
        echo "    Full rsync complete"
    fi
else
    echo ""
    echo "  [2/3] Skipping CFS backup"
fi

# ── 3. TimescaleDB dump (job history) ─────────────────────────────────
if [ "${1:-}" != "--config-only" ]; then
    echo ""
    echo "  [3/3] Backing up TimescaleDB..."

    # Source env for docker compose
    set -a
    # shellcheck disable=SC1091
    [ -f "$ENV_FILE" ] && source "$ENV_FILE"
    set +a

    local_compose="docker compose -f $COMPOSE_FILE"

    # Check if TimescaleDB is running
    if $local_compose ps timescaledb 2>/dev/null | grep -q "Up"; then
        $local_compose exec -T timescaledb \
            pg_dump -U postgres colonies \
            > "$BACKUP_DIR/colonies_db.sql" 2>/dev/null

        if [ -s "$BACKUP_DIR/colonies_db.sql" ]; then
            gzip "$BACKUP_DIR/colonies_db.sql"
            echo "    Database dump: $(du -sh "$BACKUP_DIR/colonies_db.sql.gz" | cut -f1)"
        else
            rm -f "$BACKUP_DIR/colonies_db.sql"
            echo "    Warning: DB dump was empty"
        fi
    else
        echo "    Skipped: TimescaleDB not running"
    fi
else
    echo ""
    echo "  [3/3] Skipping DB backup"
fi

# ── Update 'latest' symlink ──────────────────────────────────────────
ln -sfn "$BACKUP_DIR" "$BACKUP_ROOT/latest"

# ── Prune old backups (keep last 5) ──────────────────────────────────
echo ""
backup_count=$(ls -d "$BACKUP_ROOT"/20* 2>/dev/null | wc -l | tr -d ' ')
if [ "$backup_count" -gt 5 ]; then
    old_count=$((backup_count - 5))
    echo "  Pruning $old_count old backup(s)..."
    ls -d "$BACKUP_ROOT"/20* | head -n "$old_count" | xargs rm -rf
fi

echo ""
echo "  Backup complete: $BACKUP_DIR"
du -sh "$BACKUP_DIR" | awk '{print "  Total size: " $1}'
