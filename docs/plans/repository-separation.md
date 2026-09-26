# Repository separation

Approved 2026-09-16. Implementation branch: agent/te/codex/architecture-modularize.

## Scope and order
1. Installable engine with fetch, inference, training, api, atmosphere, sr and viz extras.
2. Study runners in studies/lupin-vagkant; public static site in imint-showcase.
3. Shared provider/schema/metric code independent of training; checked by import-boundary tests.
4. Keep training and evaluation together in ImintEngine for this phase.

## Ownership
- ImintEngine: reusable EO capabilities, models, training/evaluation, pipeline Dockerfiles and job manifests.
- studies: question-specific runners, controls, results and provenance. Install the engine; never extend sys.path into another checkout.
- imint-showcase: public HTML/CSS/JS and published artifacts. No Python engine dependency.
- Cluster-wide scheduling/cleanup remains a separately owned operational responsibility within ImintEngine. Its source and deployment must be pinned together before a future repository extraction.

## Training decision
Do not create a separate training repository yet. Training and inference share model loaders, normalization, schema and checkpoint reconstruction. Removing reverse imports makes future extraction possible, but changing their versions independently now would introduce compatibility coordination without an established release need. A later split requires an independently versioned model/checkpoint contract and an installed-artifact integration test.

## Compatibility and rollout
Provider/schema imports moved from imint.training to imint.data/imint.schema. All in-repository executable callers are migrated in this branch. External callers must update imports when upgrading to 0.2.0. Existing pinned runtimes and other worktrees keep their existing source paths.

Source history is retained by the original Git repositories; extraction manifests record original paths, commits and SHA-256 hashes. Dirty study code is copied and verified, never removed from its owning worktree. Cached data is preserved locally and excluded from Git.

Public-site extraction is local until reviewed and published. Existing hosting continues to use the old main revision until a deliberate hosting cutover. Do not merge the source removals before the new repository and hosting are ready.

Study dependencies use the locally built wheel during verification. Before publishing the studies changes, publish the reviewed engine revision and replace the local wheel requirement with its immutable release/commit reference.
