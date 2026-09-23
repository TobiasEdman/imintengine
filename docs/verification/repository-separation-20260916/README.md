# Repository separation verification — 2026-09-16

Implementation is recorded as local commits in isolated worktrees after explicit
user approval. Remote publication, hosting changes, cluster submissions and
scientific data fetches remain outside this commit step. Task 0009 remains claimed
pending integration. Website removals have their own commit for the hosting cutover.

## Locations
- Engine: `/Users/tobiasedman/Developer/ImintEngine-wt-modularize`, branch `agent/te/codex/architecture-modularize`.
- Studies: `/Users/tobiasedman/Developer/studies-wt-imint-split`, branch `agent/te/codex/studies-engine-consumer`.
- Showcase: `/Users/tobiasedman/Developer/imint-showcase`, new local Git repository on `agent/te/codex/showcase-extract`, without a remote.
- Verified wheel: `dist/imint_engine-0.2.0-py3-none-any.whl` (local artifact, ignored by Git).
- Wheel SHA-256: `bff4d03a39756e1d8136944c947d9ebfa4ace8c0ae1048bf2da414d3eefa8f46`.

## Completed checks
- Initial targeted schema/provider/fetch regression selection: 116 passed.
- Final boundary and artifact-export tests: 10 passed; see `boundaries.log`.
- Clean base-library wheel installation and `python -I tests/smoke_installed_package.py`: pass; no training/PyTorch import, correct packaged COT models, default config and notices. Added as an independent CI job.
- Study probe: all result fields exactly match saved `probe_s2_road10.json` using 3,046 observations and the 10 m road filter. See `lupin-probe.json` and `study-check.json`.
- All eight study engine-import statements resolve from the installed wheel outside the source checkout.
- Hash parity: 289 extracted site files; 1,020 study source records resolve to 1,019 destinations. All 1,005 non-Python study files retain their recorded hashes. Fourteen runners have only import/path adaptation; source hashes and provenance are preserved in the study MIGRATION.json.
- All 76 modified Kubernetes YAML manifests and the CI workflow parse successfully.
- Browser inspection: the standalone site serves HTTP 200, renders the initial dashboard, and switches to the marine panels correctly.
- Fresh-context review: LGTM after fixing package notices, bundled model files, Docker dependency installation, explicit training extras, artifact symlink rejection and clean-directory CI installation.
- `git diff --check` passes in engine and studies worktrees.

## Final regression results
- Broad regression: **2,540 passed, 8 skipped, 1 warning** in 989.23 seconds. Command: `python -m pytest tests/ --ignore=tests/test_des_connection.py -q --disable-warnings --maxfail=10`. See `regression.log`. The final focused run covers three exporter cases added after the broad run collected its tests.
- Final installed-wheel spectral smoke passes: valid summary JSON, PNG exports, exact band-array round trip, and all ten packaged COT models loaded. See `installed-smoke.log`.
- All 137 packaged Python modules match the final source files byte for byte. The default packaged analyzer configuration matches the repository default; CI checks this. Final commit preparation removed only two blank EOF lines; AST equivalence against the previously tested wheel was checked for all 137 modules.

## Practical limits
Docker images were not built; native Linux/CUDA runtime validation remains for image CI. Kubernetes manifests were parsed but not submitted or server-dry-run validated; repository policy requires server dry-run before actual submission. The live DES authentication module is excluded from the broad regression run because it starts interactive OIDC polling during test collection.

## Integration order
1. Publish the approved engine packaging/module commit and immutable artifact. Keep the separate showcase-removal commit out of main until the hosting cutover.
2. Publish the studies commit, which records the exact engine source commit and wheel hash. The copied pre-existing study narrative edit remains uncommitted; original study worktrees are untouched.
3. Publish the imint-showcase commit, configure its static host with `docs/` as document root, and verify the hosted preview.
4. Switch hosting, then merge the engine's public-site removals. Complete task 0009 after integration.

The original engine checkout remains at `abe3ea6` with its pre-existing untracked
`docs/ops/`; original studies remains at `247b105` with its pre-existing modified
`lupin-vagkant/02-s2-juni-jamforelse.md`. Existing lupin worktrees and cached inputs
remain intact. Training and evaluation stay together in the engine for this phase;
future extraction requires a versioned checkpoint/model contract.
