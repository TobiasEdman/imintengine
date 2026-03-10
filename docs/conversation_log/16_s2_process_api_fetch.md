# Sentinel Hub Process API Fetch & ColonyOS Pipeline

> Migration from openEO to Sentinel Hub Process API for S2 L2A data fetch, 1-stage cloud filtering, HTTP 429 rate limiting, ColonyOS networking fixes, 4,381 job submission.

---

## Why Process API (Not openEO)

The openEO batch endpoint was slow and unreliable for per-tile fetching. The Sentinel Hub Process API provides:
- **Single HTTP POST** per tile (no batch job polling)
- **Evalscript** for server-side band selection and format control
- **Direct OAuth2 token** authentication (client_credentials flow)
- **Faster throughput** for individual tile requests

### Authentication

```python
# OAuth2 client_credentials flow
TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
token = requests.post(TOKEN_URL, data={
    "grant_type": "client_credentials",
    "client_id": CDSE_CLIENT_ID,
    "client_secret": CDSE_CLIENT_SECRET,
}).json()["access_token"]
```

Credentials in `.env` (gitignored): `CDSE_CLIENT_ID=sh-e7f28390-...`, `CDSE_CLIENT_SECRET=...`

---

## 1-Stage Cloud Filtering

Previous (openEO): 2-stage — SCL batch pre-screen, then spectral download.
New (Process API): 1-stage — download all bands + SCL in one POST, quality-check locally.

### Quality Gates

1. **Cloud fraction from SCL** — count SCL pixels in cloud classes (8, 9, 10, 11), reject if > threshold
2. **Nodata** — reject if > 10% zero pixels in B02
3. **Haze** — reject if B02 mean reflectance > 0.06

```python
# All 7 bands in one HTTP POST
_PRITHVI_BANDS = ["B02", "B03", "B04", "B8A", "B11", "B12"]
_ALL_BANDS = _PRITHVI_BANDS + ["SCL"]
```

### Evalscript

Server-side evalscript selects exact bands and returns multi-band GeoTIFF:

```javascript
//VERSION=3
function setup() {
  return {
    input: [{bands: ["B02","B03","B04","B8A","B11","B12","SCL"], units: "DN"}],
    output: {bands: 7, sampleType: "UINT16"}
  };
}
function evaluatePixel(sample) {
  return [sample.B02, sample.B03, sample.B04,
          sample.B8A, sample.B11, sample.B12, sample.SCL];
}
```

---

## Rate Limiting (HTTP 429)

Sentinel Hub enforces API rate limits. The fetch module handles `429 Too Many Requests`:

```python
_MAX_RETRIES = 3
_RETRY_DELAY_S = 2.0
_REQUEST_TIMEOUT_S = 60

if response.status_code == 429:
    retry_after = int(response.headers.get("Retry-After", _RETRY_DELAY_S))
    time.sleep(retry_after)
    # retry...
```

---

## Prithvi Band Mapping

Prithvi EO v2 uses 6 spectral bands. The config.json uses HLS naming, the fetch uses S2 naming -- same physical bands:

| Prithvi config (HLS) | S2 name | Wavelength | Purpose |
|----------------------|---------|------------|---------|
| B02 | B02 | 490 nm (Blue) | Visible blue |
| B03 | B03 | 560 nm (Green) | Visible green |
| B04 | B04 | 665 nm (Red) | Visible red |
| B05 | B8A | 865 nm (NIR) | Narrow NIR |
| B06 | B11 | 1610 nm (SWIR1) | Short-wave IR |
| B07 | B12 | 2190 nm (SWIR2) | Short-wave IR |

---

## ColonyOS Pipeline Fixes

### CFS Networking (commit `ef0a188`)

Spawned child containers could not resolve `minio` hostname (Docker internal DNS). Fix: use `host.docker.internal` which routes through the host network.

```yaml
# docker-compose.colonyos.yml
AWS_S3_ENDPOINT: host.docker.internal:9000  # was: minio:9000
```

### Parallel Container Limit (commit `2f0ab8a`)

`EXECUTOR_PARALLEL_CONTAINERS` is a boolean (`"true"`/`"false"`), not a number. Setting `"8"` was treated as non-"true" (= sequential). Fix: keep it `"true"` and limit parallelism via CPU allocation:

```yaml
EXECUTOR_PARALLEL_CONTAINERS: "true"
EXECUTOR_HW_CPU: "8"    # was: "10"
```

### Docker Image Rebuild

After merge, the Docker image was outdated (missing `executors/s2_seasonal_fetch.py`). Containers ran the old openEO executor (`[SeasonalFetch]` log prefix). Fix:

```bash
docker build --platform linux/arm64 -t imint-engine:latest .
docker tag imint-engine:latest localhost:5000/imint-engine:latest
docker push localhost:5000/imint-engine:latest
```

**Lesson:** `docker compose down -v` wipes the local registry volume. Must re-push image after any `-v` reset.

### Executor "Access Denied" Fix

After `--force-recreate` on individual services, the executor key didn't match. Fix: always restart the full stack (`docker compose down && up -d`) so the dependency chain (`colonies-setup` -> executor registration) runs correctly.

---

## SSH Config Fix

Multiple SSH keys caused "Too many authentication failures" when connecting to M1 Max. Fix:

```
# ~/.ssh/config
Host m1max
    HostName 192.168.50.100
    User tobiasedm
    IdentityFile ~/.ssh/id_ed25519
    IdentitiesOnly yes
```

---

## Job Submission

### Full Pipeline

```bash
# Source credentials
set -a; source .env; set +a

# Dry run (count cells)
python scripts/submit_s2_jobs.py --dry-run --tiles-dir seasonal-tiles

# Submit all 4,381 jobs
python scripts/submit_s2_jobs.py --tiles-dir seasonal-tiles

# Monitor progress
python scripts/submit_s2_jobs.py --status --tiles-dir seasonal-tiles
```

### Results

```
Submitted: 4381
Failed:    0
Backend:   Sentinel Hub Process API (CDSE)
```

8 parallel containers on M1 Max, processing tiles autonomously.

**Final fetch result**: 4,305/4,381 tiles (98.3%) successfully fetched via `--local` mode. 76 far-north failures due to persistent cloud cover. See [25_seasonal_s2_fetch_completion.md](25_seasonal_s2_fetch_completion.md) for full data inventory and tile format details.

---

## File Reference

| File | Purpose |
|------|---------|
| `imint/training/cdse_s2.py` | S2 Process API fetch module (513 lines) |
| `executors/s2_seasonal_fetch.py` | ColonyOS executor for S2 fetch (167 lines) |
| `scripts/submit_s2_jobs.py` | Job submission script (548 lines) |
| `config/s2_seasonal_fetch_job.json` | ColonyOS job spec template |
| `docker-compose.colonyos.yml` | Full ColonyOS stack with fixes |

---

## Key Commits

| Commit | Description |
|--------|-------------|
| `ac7d509` | Add S2 L2A fetch via Sentinel Hub Process API |
| `ef0a188` | Fix Docker executor CFS networking (host.docker.internal) |
| `2f0ab8a` | Fix parallel containers: revert to bool, limit via CPU cores |
