# ColonyOS Operations Runbook

> Operational knowledge from running the ColonyOS distributed fetch pipeline on M1 Max. Covers CFS networking fix, OOM diagnosis, container regulation, rate limiting, watchdog/backup/safe-down scripts, and stack management procedures.

---

## Common Issues and Fixes

### CFS Networking: MinIO DNS Resolution

**Problem**: Spawned child containers can't resolve `minio` because they're on a different Docker network than the ColonyOS stack.

**Fix**: Change `AWS_S3_ENDPOINT` from `minio:9000` to `host.docker.internal:9000` in `docker-compose.colonyos.yml`. MinIO already exposes port 9000 on the host, so both the executor and spawned containers can reach it.

**File**: `docker-compose.colonyos.yml`

### OOM Kills (Exit Code 137)

**Symptom**: MinIO and docker-executor exit with code 137 (SIGKILL = 128 + 9). TimescaleDB, registry, and colonies-server stay up.

**Cause**: 8 parallel fetch containers + MinIO + TimescaleDB + colonies-server on 32GB. Docker Desktop has a memory limit; when exceeded, it kills the heaviest containers. MinIO (object cache) and docker-executor (spawns child containers) are the heaviest.

**Fix**:
1. `restart: unless-stopped` policy on all services (auto-recovery)
2. Watchdog script monitors memory and pauses executor at 92% usage
3. Reduce `EXECUTOR_HW_CPU` from 10 to 6-8 to limit parallel containers

### Container Count Explosion

**Problem**: 48 running containers from 10 submitted jobs. The executor spawns containers without a hard limit -- each job can spawn multiple containers.

**Context**: S2 fetch is mostly network-bound (HTTP calls to Sentinel Hub), so many containers CAN coexist. But excessive parallelism triggers CDSE rate limiting (HTTP 429).

**Recommendations**:
- Limit `EXECUTOR_HW_CPU` to 6-8 for balanced throughput vs stability
- The code handles 429 responses: reads `Retry-After` header and sleeps accordingly (exponential backoff for other errors)
- At ~12 seconds per tile with 10 parallel containers: ~500 tiles/hour, full 4,381 tiles in ~9 hours

### Docker Image Out of Date

**Symptom**: Containers run old executor (e.g., `SeasonalFetch` instead of `S2Fetch`).

**Fix**: Rebuild and push to local registry:
```bash
make build && docker tag imint-engine:latest localhost:5000/imint-engine:latest && docker push localhost:5000/imint-engine:latest
```

**Warning**: `docker compose down -v` wipes the registry data too. Use `make colony-down` (safe-down wrapper) instead of raw `docker compose down -v`.

---

## Infrastructure Scripts

### Watchdog (`scripts/colony_watchdog.sh`)

Health check, cleanup, failed job resubmission, memory monitoring:

```bash
make colony-watchdog
```

Reports:
- Service health (up/down for all 5 services)
- Memory usage (auto-pauses executor at 92%)
- Disk usage
- Tile count on CFS
- Failed job count

### Safe-Down (`scripts/colony_safe_down.sh`)

Prevents accidental `docker compose down -v` which destroys volumes (including MinIO data, TimescaleDB state, and local registry images):

```bash
make colony-down     # Safe: stops services, preserves volumes
make colony-reset    # Destructive: stops + destroys volumes (requires confirmation)
```

### Backup (`scripts/colony_backup.sh`)

Incremental rsync to `~/imint_backups/`:

```bash
make colony-backup
```

---

## Stack Management Procedures

### Full Restart (preserve data)

```bash
make colony-down && make colony-up
```

### Clean Restart (wipe state)

Only when you need to clear the job queue and start fresh:

```bash
make colony-reset && make colony-up
```

**Warning**: This destroys MinIO data, TimescaleDB job history, and local Docker registry images. Re-push the image after reset.

### Deploy Updated Code

```bash
# On dev machine
git push

# On M1 Max
cd ~/Developer/ImintEngine && git pull
make build
docker tag imint-engine:latest localhost:5000/imint-engine:latest
docker push localhost:5000/imint-engine:latest
make colony-down && make colony-up
```

### Resume After OOM/Crash

The pipeline is designed to be resumable:
- S2 fetch: `skip_existing=True` skips already-fetched tiles
- Training: `--checkpoint last_checkpoint.pt` resumes from last epoch
- Dashboard: restart with `python -c "from imint.training.dashboard import start_dashboard_server; ..."`

With `restart: unless-stopped`, crashed services auto-recover without intervention.

---

## Sentinel Hub Rate Limiting

CDSE Sentinel Hub limits:
- Requests/min: ~100 for standard accounts
- Processing Units: monthly quota based on account tier
- Each tile = ~8 API calls (4 windows x up to 2 years)
- 10 parallel containers = up to ~80 concurrent requests

The code handles this in `imint/training/cdse_s2.py`:
- HTTP 429 -> reads `Retry-After` header and sleeps
- Other errors -> exponential backoff
- Rate limiting is transparent -- containers back off individually

---

## Docker Compose Services

| Service | Port | Memory | Healthcheck |
|---------|------|--------|-------------|
| timescaledb | 5432 | ~500 MB | pg_isready |
| minio | 9000, 9001 | ~1-2 GB | HTTP /minio/health/live |
| colonies-server | 50080 | ~200 MB | -- |
| registry | 5000 | ~100 MB | -- |
| docker-executor | -- | variable | pgrep |

All services have `restart: unless-stopped` and healthchecks where applicable.

---

## SSH to M1 Max

```
Host m1max
  HostName 192.168.50.100
  User tobiasedm
  IdentityFile ~/.ssh/id_ed25519
  IdentitiesOnly yes
```

**Note**: Without `IdentitiesOnly yes`, the SSH agent tries multiple keys and exhausts the `MaxAuthTries` limit (default 6), causing "Too many authentication failures" errors.

---

## Key Files

| File | Purpose |
|------|---------|
| `docker-compose.colonyos.yml` | Full ColonyOS stack with restart policies |
| `scripts/colony_watchdog.sh` | Health check + memory monitor |
| `scripts/colony_safe_down.sh` | Safe shutdown wrapper |
| `scripts/colony_backup.sh` | Incremental backup |
| `Makefile` | colony-up, colony-down, colony-watchdog, colony-backup targets |
| `scripts/submit_s2_jobs.py` | Job submission (skips already-fetched) |
| `scripts/monitor_seasonal_jobs.py` | Job progress monitoring |
