# ERA5 Prithvi-600M smoke runtime

This image freezes the Python/CUDA dependencies and the immutable repository
base used by the paired ERA5 smoke test. The experiment-specific files remain
content-addressed in its Kubernetes ConfigMap and overwrite the same paths in
a copy of `/opt/imintengine` at runtime.

Build locally with `PUSH=1 ./build.sh`, or use
`python scripts/build_era5_smoke_runtime.py --dry-run` followed by `--apply`
on ICE. Both paths publish the content-addressed
`:20260821-<build-context-sha12>` tag. After the push, resolve the registry
digest, place the exact tag and digest in `MANIFEST.json`, and pin both smoke
Jobs to that digest before server-side dry-run or execution.

The Docker build runs the CPU import/autograd smoke inline. Every GPU Job runs
the same versioned `runtime_smoke.py` from its immutable code ConfigMap before
touching data; the subsequent one-batch preflight exercises the complete
Prithvi-600M forward, loss, backward, and optimizer path.
