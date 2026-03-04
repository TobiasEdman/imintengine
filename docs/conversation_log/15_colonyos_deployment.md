# ColonyOS Deployment & M1 Max Setup

> ColonyOS distributed job scheduler, Docker setup, M1 Max deployment, job spec debugging, parallel execution.

---

## ColonyOS Overview

Distributed job scheduler for containerized Sentinel-2 fetch + analysis jobs. Not beneficial for single ML training jobs, but designed for batch inference and distributed data fetching with DES/CDSE.

---

## Infrastructure Stack

```yaml
# docker-compose.colonyos.yml
Services:
  - TimescaleDB (pg17)        # Job state persistence
  - MinIO (S3)                # CFS (Colony File System) storage
  - Colonies server (v1.9.12) # Job scheduler
  - Docker executor           # Runs container jobs
  - Local registry            # ARM64 image registry
```

All services `platform: linux/arm64`.

### Key Environment Variables

| Var | Purpose |
|-----|---------|
| `COLONIES_TLS` | false (dev) |
| `COLONIES_SERVER_PORT` | 50080 |
| `COLONIES_COLONY_NAME` | imint |
| `EXECUTOR_TYPE` | container-executor |
| `EXECUTOR_PARALLEL_CONTAINERS` | true |
| `EXECUTOR_FS_DIR` | /Users/tobiasedm/cfs |

Credentials in `.env.colonyos` (gitignored).

---

## M1 Max Setup (192.168.50.100)

### Machine
- Apple M1 Max, 32GB RAM, 10 CPU cores
- macOS Sequoia
- User: `tobiasedm`
- SSH: `ssh tobiasedm@192.168.50.100` (ed25519 key auth)

### Installation Steps

1. **Xcode CLI Tools:** `xcode-select --install` (or download .dmg from developer.apple.com)
2. **Docker Desktop:** Already at `/Applications/Docker.app`. Settings: 8GB+ memory, Rosetta disabled.
3. **Colonies CLI:** Downloaded ARM64 binary to `~/bin` (v1.9.12) — no Homebrew (needs admin)
4. **pycolonies:** `pip install pycolonies` in venv
5. **Repo:** Cloned to `~/Developer/ImintEngine`
6. **Data:** rsync'd ~12.5 GB from dev machine

### SSH Key Setup
```bash
ssh-keygen -t ed25519 -N ""
cat ~/.ssh/id_ed25519.pub | ssh tobiasedm@192.168.50.100 \
  "mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys"
```

### Data Transfer
```bash
rsync -avz --progress --exclude='.venv' \
  /Users/tobiasedman/Developer/ImintEngine/ \
  tobiasedm@192.168.50.100:~/Developer/ImintEngine/
```

---

## Docker Build Issues

| Issue | Fix |
|-------|-----|
| Keychain error over SSH | `DOCKER_BUILDKIT=0` for image build |
| `libgdal32` not found (Debian Trixie) | Changed to `libgdal36` |
| `COPY .cdse_credentials` fails when missing | Bracket trick: `COPY .cdse_credential[s] /app/` |
| ENTRYPOINT conflicts with executor `sh -c` | Removed ENTRYPOINT entirely |

### Dockerfile Key Sections
```dockerfile
# No ENTRYPOINT — ColonyOS Docker executor wraps cmd in sh -c
CMD ["python", "executors/colonyos.py"]
```

### Image Push via Docker API Socket (bypass keychain)
```bash
curl --unix-socket /var/run/docker.sock \
  '.../tag?repo=localhost:5000/imint-engine&tag=latest' -X POST
curl --unix-socket /var/run/docker.sock \
  '.../push' -X POST -H 'X-Registry-Auth: e30='
```

---

## ColonyOS Stack Startup Issues

| Issue | Fix |
|-------|-----|
| DB not initialized | Added `--initdb` flag to colonies server |
| YAML multiline entrypoint broken | Single-line commands in colonies-setup |
| Wrong executor env var | `EXECUTOR_TYPE` not `COLONIES_EXECUTOR_TYPE` |
| Missing executor auth | Added `COLONIES_PRVKEY` for executor |
| Empty HW vars cause failure | All `EXECUTOR_HW_*` and `EXECUTOR_LOCATION_*` need non-empty values |
| MinIO missing credentials | Added environment block to minio-setup |
| CFS bind mount path | Changed from Docker volume to host-path `/Users/tobiasedm/cfs` |

### colonies-setup Entrypoint (final)
```yaml
entrypoint:
  - /bin/sh
  - -c
  - |
    sleep 5
    colonies colony add --name ${COLONIES_COLONY_NAME} --colonyid ${COLONIES_COLONY_ID} \
      --serverprvkey ${COLONIES_SERVER_PRVKEY} --host colonies-server \
      --port ${COLONIES_SERVER_PORT} --insecure 2>/dev/null || echo 'Colony already exists'
    colonies user add --name imint-user --email dev@imint.local --phone '' \
      --userid ${COLONIES_ID} --colonyprvkey ${COLONIES_COLONY_PRVKEY} \
      --host colonies-server --port ${COLONIES_SERVER_PORT} --insecure 2>/dev/null || echo 'User already exists'
```

### Running State
| Service | Status |
|---------|--------|
| TimescaleDB | healthy |
| MinIO | :9000 API, :9001 console |
| Colonies server | :50080 |
| Colony "imint" | created |
| Docker executor | dev-docker, registered, listening |
| imint-engine:latest | built (ARM64) |

---

## Job Spec (Final Working Version)

```python
{
    "conditions": {
        "colonyname": "imint",
        "executortype": "container-executor",
        "nodes": 1,
        "processes": 1,
        "processespernode": 1,
        "walltime": 900,
        "cpu": "1000m",
        "mem": "2Gi",
    },
    "env": {
        "EASTING": str(cell.easting),
        "NORTHING": str(cell.northing),
        "WEST_WGS84": str(cell.west_wgs84),
        # ... all WGS84 coords
        "FETCH_SOURCE": "auto",
        "YEARS": ",".join(years),
        "SEASONAL_WINDOWS": windows,
        "DES_USER": des_user,
        "DES_PASSWORD": des_password,
        "CDSE_CLIENT_ID": cdse_client_id,
        "CDSE_CLIENT_SECRET": cdse_client_secret,
    },
    "funcname": "execute",  # docker executor only supports "execute"
    "kwargs": {
        "docker-image": "localhost:5000/imint-engine:latest",
        "rebuild-image": False,
        "cmd": "python executors/seasonal_fetch.py",
    },
    "maxexectime": 900,
    "maxretries": 3,
    "fs": {
        "mount": "/cfs/tiles",
        "dirs": [{
            "label": cfs_dir,
            "dir": "/cfs/tiles",
            "keepfiles": False,
            "onconflicts": {
                "onstart": {"keeplocal": False},
                "onclose": {"keeplocal": False},
            },
        }],
    },
}
```

### Job Spec Debugging History

1. `fs` field: `[{...}]` array -> `{...}` object
2. `funcname`: `"seasonal-tile-fetch"` -> `"execute"` (docker executor constraint)
3. `processespernode cannot be 0`: moved into `conditions` block
4. `walltime cannot be 0`: added to `conditions`
5. `CPU format`: `"2000m"` (millicores)
6. Missing Python packages in Docker: added geopandas, fiona, pyproj, requests, etc.

---

## Parallel Execution

### Configuration
- `EXECUTOR_PARALLEL_CONTAINERS: "true"` in docker-compose
- Per-job resources: `cpu: "1000m"`, `mem: "2Gi"` (allows ~10 concurrent)
- M1 Max: 10 CPU cores, 32GB RAM

### Throughput Estimates

| Backend | Workers | Time/tile | Rate |
|---------|---------|-----------|------|
| DES | 3 | ~4 min | ~45/h |
| CDSE | ~10 | ~30 min | ~20/h |
| Combined | 13 | — | ~65/h |
| Optimistic | 13 | — | ~120/h |

Total: ~36-67 hours for 4,381 tiles.

### Submit Strategy
```bash
# Test batch
python scripts/submit_seasonal_jobs.py --max-jobs 10
# Full run
python scripts/submit_seasonal_jobs.py
```

---

## Monitoring

### `scripts/monitor_seasonal_jobs.py` (~240 lines)

Polls ColonyOS via `colonies process ps/psw/pss/psf --json --insecure`:
- Filters for seasonal fetch jobs by cmd or env vars
- Computes per-source stats: completed, failed, running, avg_time, success_rate
- Writes `seasonal_fetch_log.json` for dashboard

### Dashboard Panel

Added "Seasonal Fetch (ColonyOS)" section to `dashboard.py`:
- Progress bar, cards (completed, running, failed, rate, ETA)
- CDSE vs DES comparison chart (Chart.js grouped bar)
- Source performance boxes

### Makefile Targets
```makefile
colony-up:      # Start ColonyOS stack
colony-down:    # Stop stack
colony-reset:   # Full reset
colony-logs:    # View logs
colony-status:  # Check status
colony-submit:  # Submit jobs
colony-monitor: # Run monitor + dashboard JSON
```

---

## run_training.sh (M1 Max)

One-command training launcher. Key fixes for M1 Max Python 3.9.6:

```bash
# Create python -> python3 symlink in venv
if [ ! -e "$VENV_DIR/bin/python" ] && [ -e "$VENV_DIR/bin/python3" ]; then
    ln -s python3 "$VENV_DIR/bin/python"
fi
# Bootstrap pip (Xcode CLT Python may not include it)
python3 -m ensurepip --upgrade -q 2>/dev/null || true
# Use python3 -m pip (bare pip not on PATH)
python3 -m pip install -r requirements.txt -q
```

---

## Local Batch Fetch

### `scripts/batch_local_fetch.py`
Alternative to ColonyOS — runs `SeasonalFetchExecutor` per tile locally:
- Grid generation with SWEREF99 -> WGS84 conversion
- Skips completed tiles (resume support)
- `FETCH_SOURCE=auto` for dynamic source selection

---

## Deployment Steps (Reconnect to Network)

1. `git pull` on M1 Max
2. `docker build --platform linux/arm64 -t imint-engine:latest .`
3. Push to local registry via Docker API socket
4. Restart ColonyOS: `make colony-down && make colony-up`
5. Test batch: `python scripts/submit_seasonal_jobs.py --max-jobs 10`
6. Monitor: `docker ps` for parallel containers
7. Full run: submit remaining ~4,300 tiles

---

## Key Files

| File | Purpose |
|------|---------|
| `executors/seasonal_fetch.py` | ColonyOS executor (311 lines) |
| `executors/colonyos.py` | Analysis executor |
| `scripts/submit_seasonal_jobs.py` | Job coordinator (326 lines) |
| `scripts/monitor_seasonal_jobs.py` | Progress monitor (240 lines) |
| `scripts/batch_local_fetch.py` | Local alternative to ColonyOS |
| `docker-compose.colonyos.yml` | Full ColonyOS stack |
| `Dockerfile` | ARM64, no ENTRYPOINT |
| `Makefile` | Colony-* targets |
| `config/seasonal_fetch_job.json` | Job template |
| `run_training.sh` | One-command training launcher |

---

## Key Commits

| Commit | Description |
|--------|-------------|
| `a2a4192` | ColonyOS local dev stack + Docker Trixie fixes |
| `1be4cb6` | ColonyOS seasonal fetch executor + Docker + benchmarks |
| `15fa854` | Seasonal fetch monitoring dashboard + CLI fixes |
| — | Dynamic CDSE/DES fetch with parallel execution |
| — | Adaptive source selection + SCL batch GeoTIFF fix |

---

## Synced Locations

| Location | Status |
|----------|--------|
| Local machine (Tobias Dator) | Up to date |
| GitHub (origin/main) | Up to date |
| M1 Max (192.168.50.100) | Up to date (only `.env.colonyos` untracked) |
