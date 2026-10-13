# Prediction cache identity and migration

Stage A (`infer_tiles.py`) is the only inference producer. Stage B
(`score_against_truth.py`) reads recorded predictions and never loads a model
or regenerates a cache. Cache schema 2 requires an explicitly supplied producer
contract SHA-256, printed by Stage A. Record this digest separately when
approving a produced cache; copying an untrusted manifest's digest into the
command is not an independent provenance check.

## Producer scope

Ordinary `import imint` retains its lazy API and does not activate observation,
scan sources, discover encoders or query package metadata. Stage A explicitly activates capture; Stage B leaves it inactive.
Stage A snapshots candidate source bytes
before loading preprocessing or models; only observed producer code and model
resources enter its contract. A source must agree with the actual executed
module code, including aliases and timestamp-based bytecode. A relevant module
loaded before capture is refused: reading disk afterward cannot authenticate
that earlier execution. Observation failures are retained as cache errors.

Shared preprocessing is imported before the producer contract is fixed.
Cached imports are observed too, including imports local to a producer
function. Unexpected lazy dependency growth after pinning is refused before
publishing a prediction; it is not silently added to an existing contract.
Workers validate their preprocessing subset against the complete parent source
contract. Restart a producer after changing relevant source. Unused encoder
installations, unrelated files and unrelated installed distributions do not
belong to the produced identity.

The contract records relevant numerical package versions and Python, device
and numerical backend settings. This is provenance and a conservative reuse
policy, not a claim of full reproducibility from source alone: hardware,
numerical libraries and nondeterministic kernels can still matter. Stage B
validates the recorded contract, entry metadata, source identity and array
shape/type/range. It does not require the scoring machine to recreate the
GPU producer's import graph or installed environment.

## Explicit Stage B cost/trust choice

The scorer prints the number of distinct candidate source tiles and the sum
of their file sizes using stat calls before reading tile payloads. Without
`--input-validation`, it then stops. The count is a pre-crop candidate budget,
not a timing estimate: filtering can reduce the set; repeated predictor calls
verify again. Prediction arrays, receipt reads and coordinate/crop reads are
additional I/O. There is no promised runtime or throughput.

Choose exactly one mode, always with `--producer-pin <approved-contract-sha>`:

* `--input-validation content`: explicitly authorize SHA-256 validation of
  source bytes on every predictor call. This is also the Python API default.
  Same-size changes, restored mtimes and changes after an earlier hit are not
  trusted. Do not blindly select this in a large scoring job.
* `--input-validation immutable --input-manifest <receipt.json>
  --input-manifest-sha256 <independently-pinned-sha>
  --immutable-input-root <actual-immutable-root> --assume-immutable-inputs`:
  skip tile payload hashing only under the operator's explicit guarantee that
  the entire source snapshot remains immutable for the cache's lifetime/use.
  Stage A prints a content-addressed input receipt. Pin its SHA separately and
  retain the exact input paths. The receipt, matching sizes and mtimes, or a
  self-declared snapshot name **do not prove immutability**. Reject symlinked
  roots/paths, paths outside the root, absent entries and mismatched receipts.
  If the guarantee cannot be established, use content validation.

No real immutable dataset snapshot is supplied or asserted by this change.
The shipped scoring template requires explicit producer and validation
placeholders, with no automatic content scan or assumed immutable snapshot.
For sharded Stage A, each printed receipt covers that shard's input set; select
an independently pinned receipt covering the scoring set. Do not infer that
one shard's receipt covers the others.

## Existing cache decision

Schema-1 and metadata-free caches are refused. A changed source/checkpoint,
inference configuration, relevant runtime or corrupt entry is reported with
field names and an affected-entry count before model construction. Existing
entries are preserved by default. Missing entries are different from invalid
existing entries. After inspecting the cache and cost, explicitly authorize
`infer_tiles.py --regenerate-invalid` or choose a new cache directory. This
flag is a local invocation decision, not authorization for live GPU work.

The external PR76 review reports a legacy cache of approximately 75 GB and
approximately 2.5 minutes for NFI / 26 minutes for LUCAS per checkpoint to
regenerate. These are **unverified external-review cost estimates**, not local
measurements, performance guarantees or permission to regenerate. The review
also estimates roughly half a TB of source I/O for a LUCAS content-validation
pass. Use the scorer's actual stat-only preflight for the selected input set.

Manifest decisions and per-entry publications are serialized under a cache
publication lock. Arrays and their metadata are published together by atomic
replacement. Same-contract shards may cooperate; a conflicting contract
refuses publication. Explicit regeneration must not run concurrently with
another producer using a different contract. Stage B refuses manifest/entry
mismatches rather than repairing them.
