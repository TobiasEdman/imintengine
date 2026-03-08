#!/usr/bin/env bash
# scripts/colony_watchdog.sh — Self-healing watchdog for ColonyOS stack
#
# Checks service health, cleans orphaned .tmp.npz files, resubmits
# failed jobs, and logs all activity.
#
# Usage:
#   ./scripts/colony_watchdog.sh              # Run once
#   ./scripts/colony_watchdog.sh --loop 300   # Run every 5 minutes
#
# Add to crontab for unattended operation:
#   */5 * * * * cd ~/developer/imintengine && ./scripts/colony_watchdog.sh >> logs/watchdog.log 2>&1

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
COMPOSE_FILE="$PROJECT_DIR/docker-compose.colonyos.yml"
ENV_FILE="$PROJECT_DIR/.env.colonyos"
LOG_DIR="$PROJECT_DIR/logs"
CFS_DIR="$HOME/cfs"
TMP_MAX_AGE_MIN=30    # Orphaned .tmp.npz older than this get deleted
FAILED_RESUBMIT=true  # Set to false to disable auto-resubmission
MEM_WARN_PCT=85       # Warn if memory usage exceeds this
MEM_PAUSE_PCT=92      # Pause executor if memory exceeds this

export PATH="/usr/local/bin:/opt/homebrew/bin:$PATH"

mkdir -p "$LOG_DIR"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

# ── 1. Service health check ──────────────────────────────────────────
check_services() {
    log "=== Service Health Check ==="

    # Source env for docker compose
    set -a
    # shellcheck disable=SC1091
    [ -f "$ENV_FILE" ] && source "$ENV_FILE"
    set +a

    local compose_cmd="docker compose -f $COMPOSE_FILE"
    local unhealthy=0

    for svc in timescaledb minio colonies-server docker-executor registry; do
        local status
        status=$($compose_cmd ps --format '{{.Status}}' "$svc" 2>/dev/null || echo "not found")

        if echo "$status" | grep -qi "up"; then
            if echo "$status" | grep -qi "unhealthy"; then
                log "  UNHEALTHY: $svc ($status) — restarting"
                $compose_cmd restart "$svc" 2>/dev/null
                unhealthy=$((unhealthy + 1))
            else
                log "  OK: $svc"
            fi
        else
            log "  DOWN: $svc ($status) — starting"
            $compose_cmd up -d "$svc" 2>/dev/null
            unhealthy=$((unhealthy + 1))
        fi
    done

    if [ $unhealthy -gt 0 ]; then
        log "  Restarted $unhealthy service(s), waiting 15s for stabilization..."
        sleep 15
    fi
}

# ── 2. Orphaned .tmp.npz cleanup ─────────────────────────────────────
cleanup_tmp_files() {
    log "=== Temp File Cleanup ==="

    # Clean CFS bind mount
    if [ -d "$CFS_DIR" ]; then
        local count
        count=$(find "$CFS_DIR" \( -name "*.tmp.npz" -o -name "*.vpp_tmp.npz" \) \
            -mmin +"$TMP_MAX_AGE_MIN" 2>/dev/null | wc -l | tr -d ' ')

        if [ "$count" -gt 0 ]; then
            log "  Removing $count orphaned temp file(s) older than ${TMP_MAX_AGE_MIN}min"
            find "$CFS_DIR" \( -name "*.tmp.npz" -o -name "*.vpp_tmp.npz" \) \
                -mmin +"$TMP_MAX_AGE_MIN" -delete 2>/dev/null || true
        else
            log "  No orphaned temp files"
        fi
    else
        log "  CFS dir not found: $CFS_DIR"
    fi

    # Also clean local data directory
    local data_dir="$PROJECT_DIR/data"
    if [ -d "$data_dir" ]; then
        local local_count
        local_count=$(find "$data_dir" -name "*.tmp.npz" \
            -mmin +"$TMP_MAX_AGE_MIN" 2>/dev/null | wc -l | tr -d ' ')
        if [ "$local_count" -gt 0 ]; then
            log "  Removing $local_count local temp file(s)"
            find "$data_dir" -name "*.tmp.npz" \
                -mmin +"$TMP_MAX_AGE_MIN" -delete 2>/dev/null || true
        fi
    fi
}

# ── 3. Failed job resubmission ────────────────────────────────────────
resubmit_failed() {
    if [ "$FAILED_RESUBMIT" != "true" ]; then
        log "=== Resubmission disabled ==="
        return
    fi

    log "=== Failed Job Check ==="

    # Source ColonyOS env for CLI
    set -a
    # shellcheck disable=SC1091
    [ -f "$ENV_FILE" ] && source "$ENV_FILE"
    set +a

    # Check if colonies CLI is available
    local colonies_bin
    colonies_bin=$(command -v colonies 2>/dev/null || echo "$HOME/bin/colonies")
    if [ ! -x "$colonies_bin" ]; then
        log "  colonies CLI not found, skipping resubmission"
        return
    fi

    # Get failed processes as JSON
    local failed_json
    failed_json=$("$colonies_bin" process psf --insecure --count 100 --json 2>/dev/null || echo "null")

    if [ "$failed_json" = "null" ] || [ -z "$failed_json" ]; then
        log "  No failed jobs (or CLI unavailable)"
        return
    fi

    # Count and resubmit failed seasonal jobs
    python3 -c "
import sys, json, subprocess, tempfile, os

try:
    procs = json.loads('''$failed_json''')
except:
    procs = None

if not procs:
    print('  No failed jobs to resubmit')
    sys.exit(0)

# Filter for seasonal fetch jobs
fetch_jobs = [p for p in procs
              if 'seasonal_fetch' in p.get('spec',{}).get('kwargs',{}).get('cmd','')]

if not fetch_jobs:
    print('  No failed seasonal jobs')
    sys.exit(0)

print(f'  Found {len(fetch_jobs)} failed seasonal job(s)')

resubmitted = 0
for proc in fetch_jobs[:20]:  # Rate-limit: max 20 per cycle
    spec = proc.get('spec', {})
    env = spec.get('env', {})
    key = f\"{env.get('EASTING','?')}_{env.get('NORTHING','?')}\"

    new_spec = {
        'conditions': spec.get('conditions', {}),
        'env': env,
        'funcname': spec.get('funcname', 'execute'),
        'kwargs': spec.get('kwargs', {}),
        'maxexectime': spec.get('maxexectime', 600),
        'maxretries': spec.get('maxretries', 3),
        'maxwaittime': spec.get('maxwaittime', -1),
    }
    if 'fs' in spec:
        new_spec['fs'] = spec['fs']

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(new_spec, f)
        spec_path = f.name

    try:
        result = subprocess.run(
            ['$colonies_bin', 'function', 'submit', '--spec', spec_path, '--insecure'],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0:
            resubmitted += 1
        else:
            print(f'  Failed to resubmit {key}: {result.stderr.strip()[:80]}')
    except Exception as e:
        print(f'  Error resubmitting {key}: {e}')
    finally:
        os.unlink(spec_path)

print(f'  Resubmitted: {resubmitted}/{len(fetch_jobs[:20])}')
" 2>/dev/null || log "  Resubmission script failed"
}

# ── 4. Memory & resource check ────────────────────────────────────────
check_resources() {
    log "=== Memory & Disk ==="

    # ── Memory (macOS uses vm_stat, Linux uses /proc/meminfo) ──
    if command -v vm_stat &>/dev/null; then
        # macOS: parse vm_stat for memory pressure
        local page_size mem_free mem_inactive mem_speculative mem_total mem_used_pct
        page_size=$(sysctl -n hw.pagesize 2>/dev/null || echo 4096)
        mem_total=$(sysctl -n hw.memsize 2>/dev/null || echo 0)

        local vm_out
        vm_out=$(vm_stat 2>/dev/null)
        mem_free=$(echo "$vm_out" | awk '/Pages free/ {gsub(/\./,"",$3); print $3}')
        mem_inactive=$(echo "$vm_out" | awk '/Pages inactive/ {gsub(/\./,"",$3); print $3}')
        mem_speculative=$(echo "$vm_out" | awk '/Pages speculative/ {gsub(/\./,"",$3); print $3}')

        # Available = free + inactive + speculative (in bytes)
        local mem_available_bytes
        mem_available_bytes=$(( (${mem_free:-0} + ${mem_inactive:-0} + ${mem_speculative:-0}) * page_size ))
        local mem_total_gb mem_avail_gb
        mem_total_gb=$(( mem_total / 1073741824 ))
        mem_avail_gb=$(echo "scale=1; $mem_available_bytes / 1073741824" | bc 2>/dev/null || echo "?")

        if [ "$mem_total" -gt 0 ]; then
            mem_used_pct=$(( 100 - (mem_available_bytes * 100 / mem_total) ))
        else
            mem_used_pct=0
        fi

        log "  Memory: ${mem_used_pct}% used (${mem_avail_gb}GB available / ${mem_total_gb}GB total)"

        # Container count
        local container_count
        container_count=$(docker ps --filter "ancestor=localhost:5000/imint-engine:latest" -q 2>/dev/null | wc -l | tr -d ' ')
        log "  Active fetch containers: $container_count"

        if [ "$mem_used_pct" -ge "$MEM_PAUSE_PCT" ]; then
            log "  CRITICAL: Memory at ${mem_used_pct}% — pausing docker-executor"
            # Source env for docker compose
            set -a
            [ -f "$ENV_FILE" ] && source "$ENV_FILE"
            set +a
            docker compose -f "$COMPOSE_FILE" pause docker-executor 2>/dev/null && \
                log "  Executor paused. Run 'docker compose -f docker-compose.colonyos.yml unpause docker-executor' to resume." || \
                log "  Failed to pause executor"
        elif [ "$mem_used_pct" -ge "$MEM_WARN_PCT" ]; then
            log "  WARNING: Memory usage high (${mem_used_pct}%)"
        fi

    elif [ -f /proc/meminfo ]; then
        # Linux: parse /proc/meminfo
        local mem_total_kb mem_avail_kb mem_used_pct
        mem_total_kb=$(awk '/MemTotal/ {print $2}' /proc/meminfo)
        mem_avail_kb=$(awk '/MemAvailable/ {print $2}' /proc/meminfo)
        mem_used_pct=$(( 100 - (mem_avail_kb * 100 / mem_total_kb) ))
        local mem_avail_gb
        mem_avail_gb=$(echo "scale=1; $mem_avail_kb / 1048576" | bc 2>/dev/null || echo "?")
        log "  Memory: ${mem_used_pct}% used (${mem_avail_gb}GB available)"
    else
        log "  Memory: unable to determine"
    fi

    # ── Disk ──
    local disk_usage
    disk_usage=$(df -h / | tail -1 | awk '{print $5}' | tr -d '%')
    log "  Disk: ${disk_usage}% used"
    if [ "$disk_usage" -gt 90 ]; then
        log "  WARNING: Disk usage above 90%!"
    fi

    if [ -d "$CFS_DIR" ]; then
        local cfs_size tile_count
        cfs_size=$(du -sh "$CFS_DIR" 2>/dev/null | cut -f1)
        tile_count=$(find "$CFS_DIR" -name "*.npz" 2>/dev/null | wc -l | tr -d ' ')
        log "  CFS: $cfs_size ($tile_count tiles)"
    fi
}

# ── Main ──────────────────────────────────────────────────────────────
main() {
    log "========================================="
    log "ColonyOS Watchdog"
    log "========================================="

    check_services
    cleanup_tmp_files
    resubmit_failed
    check_resources

    log "Watchdog cycle complete"
    log ""
}

# Loop mode or single run
if [ "${1:-}" = "--loop" ]; then
    interval="${2:-300}"
    log "Starting watchdog loop (interval: ${interval}s)"
    while true; do
        main
        sleep "$interval"
    done
else
    main
fi
