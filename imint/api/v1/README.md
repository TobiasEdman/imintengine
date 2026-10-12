# API v1 service contract

Run from the repository root with production fetch dependencies and operator
credentials configured separately:

```sh
uvicorn imint.api.v1:app --host 127.0.0.1 --workers 1 --lifespan on
```

Use **one worker, one replica, one host**, no reload, and lifespan enabled.
`IMINT_API_OUTPUT_DIR` (default `outputs/api`, relative to the working directory)
must be a stable local directory. All processes for this service must use that
same directory. Startup takes a POSIX flock there; a second local worker fails
startup. Disabling lifespan leaves analysis submission unavailable (503).
Do not remove the lock file while running. Different directories or hosts evade
this local guard; horizontal replication is unsupported, even behind a sticky
load balancer. This is not a distributed store. Windows is unsupported.

The supplied Docker command fixes one uvicorn worker and enables lifespan.
The Kubernetes example uses one replica and Recreate updates; do not add an HPA
or override those defaults. Its container-local outputs disappear when the pod
is replaced unless an operator separately configures persistent storage.

Jobs live only in memory. Restart loses job IDs/status, so clients must treat a
404 after restart as unknown, not as evidence that no output was written.
Artifacts remain under `<output-root>/<job-id>/`. Accepted work runs one job at
a time, including fetching. Shutdown stops admission and drains queued/running
jobs before releasing the lock. Give the process sufficient shutdown time;
forced termination can leave partial output and loses in-memory status. There
is no durable queue, retry, job cancellation, retention policy, or resource quota.

POST `/v1/analyze` accepts only `date` and `bbox`. Date must be exact YYYY-MM-DD;
bbox must be ordered, nonempty WGS84 (EPSG:4326). Antimeridian spans and other
CRSs are unsupported. Unknown top-level fields, including `callback_url`,
`analyzers`, `disable_analyzers`, and `config_overrides`, are rejected with 422,
even if null. There are no outbound completion callbacks; poll the job URL.
Callbacks require a future server-registered destination policy.

The worker calls production `imint.fetch.fetch_des_data` for the exact day
(`date_window=0`, `include_scl=True`, `cloud_threshold=0.1`), and forwards RGB,
bands, SCL and georeferencing to the engine. Fetch/cloud failures fail the job;
there is no synthetic-data fallback. `config/analyzers.yaml` in the working
directory is required; it alone selects/configures analyzers. No per-request
configuration merge semantics are offered.

This API has no application authentication. Keep it behind a trusted operator
access boundary; the loopback invocation above is the default supported setup.
Do not publish it as a public or multi-tenant service. Output isolation and
request validation are not quotas or authorization.
