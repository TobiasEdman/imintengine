# Historical NFI2024 campaign jobs (PR #67)

The 34 Jobs listed in `tests/test_historical_campaign_jobs.py::HISTORICAL`
are historical records. Their complete image, source and dependency
identities cannot be reconstructed from these recipes. Existing recorded
pins are preserved; missing pins have not been invented.

Every Job is suspended and its shell exits 64 before any install, clone,
fetch or data operation. Do not unsuspend or strip the guard for reuse.
Create a separate reviewed replacement with immutable image/source pins,
locked dependencies, observed producer identities and a bounded input scope.
The corrected Python bodies are integration candidates, not evidence that
the historical runs used the corrected logic. Historical measurements and
reports keep their original identities.

The promotion candidate requires `TESSERA_PROMOTION_EXPECTED` to point to a
reviewed JSON file with this structure (names and source below are illustrative):

```json
{
  "source": "geotessera-observed-version",
  "cohort": {"example.npz": 2024},
  "campaign": {"campaign.npz": 2024},
  "exclusions": {"cohort/example.npz": "documented quality exclusion"}
}
```

Both mappings list the complete expected directory membership, including
excluded files, with each file's intended embedding year. At least one cohort
member and one admitted member are required. Source stamps must already be
supported by producer evidence. Unstamped campaign arrays are rejected, not
retroactively assigned a guessed source. Empty or invalid arrays require
explicit exclusions. Original cohort embeddings remain under `tessera_v1`.

V2 comparison records are atomic, fsynced per-tile JSON files under
`<REPORT>.d/`, keyed by absolute path to distinguish equal basenames in
cohort and staging. A restart reconstructs records from stored old/new arrays
before accepting an existing v2 tile. The JSONL report is a replaceable
summary; per-tile records survive an interrupted scan. Historical missing
source stamps are reported as `unknown`, not inferred from today's client.

The reusable fallback driver preserves tried-state. Repair of the historical
poisoned record needs its own authenticated scope and backup; this patch does
not authorize or implement a reset.

Offline validation: `python -m pytest tests/test_historical_campaign_jobs.py`.
The tests execute extracted Python with temporary synthetic storage and stub
runners. They do not run manifests, remote jobs, package installs, real scene
fetches or private-data promotion. Live storage/runtime validation and the
final changes belong to the author of the replacement.
