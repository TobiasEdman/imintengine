# scripts/dashboards/refetch_progress/

Live HTML dashboard for the `refetch-late-autumn-512` K8s job. Built
during the 2026-06 audit-tile refetch arc to give a single glance at
whether the job is making progress, what the rate looks like, and when
it should land — without re-tailing `kubectl logs` every few minutes.

Browser-side only; no server runtime, no backend. A bash poller writes
`data.json` next to `index.html` every 30 s; the page polls that file
and ticks an HH:MM:SS countdown locally every second between polls.

## Files

| File | Role |
|---|---|
| `index.html` | Static dashboard. Polls `data.json` every 30 s; ticks countdown every 1 s. |
| `update_data.sh` | Bash loop, calls `kubectl logs` + `kubectl get pod` every 30 s, writes `data.json`. |
| `data.json` | Generated on every poll. **Gitignored — do not commit.** |

## Run

```bash
cd scripts/dashboards/refetch_progress/

# Background poller — writes data.json every 30s
./update_data.sh &

# Serve the static files (any port)
python3 -m http.server 8765
```

Open `http://localhost:8765/` in a browser. The countdown starts as
`—:—:—` until the first `[N/M] status={...} ... ETA=Xmin` line surfaces
in the pod's stdout, then locks onto the pod-reported ETA and ticks
down in real time. Each subsequent poll re-anchors the baseline against
the live rate (so a stall is visible within ~30 s as the countdown
diverging from clock time).

## data.json shape

Written by `update_data.sh`, consumed by `index.html`:

```json
{
  "timestamp": "2026-06-04T12:34:56Z",
  "pod": {
    "name":   "refetch-late-autumn-512-q2jhg",
    "status": "Running",
    "age":    "3h17m"
  },
  "progress": {
    "scanned":        1234,
    "total":          6786,
    "ok":             523,
    "failed":         611,
    "skipped":        100,
    "error":          0,
    "rate_per_hour":  410,
    "eta_min":        812.4
  }
}
```

All `progress.*` fields are scraped from the pod's last `[N/M] status=...`
line via grep/awk in `update_data.sh`. If the pod hasn't emitted that
line yet (still in pip-install / SkipIndex warmup), the counters stay
at 0 and the dashboard shows the connecting state.

## Strategy panel (static)

The `per-slot brokenness` and `total slot-fetches needed` figures are
hardcoded from `audit_strategy.py`'s output against the 2026-06-03
audit JSON:

- slot 0 — 6 109 broken (90.0 %) — autumn y-1, mostly year=2018 cohort
- slot 1 — 2 322 broken (34.2 %) — spring growing season
- slot 2 — 1 620 broken (23.9 %) — summer
- slot 3 — 5 520 broken (81.3 %) — late-summer, hit by PR #15 cap_doy=244 bug
- 352 tiles already complete → skipped via SkipIndex
- ~15 571 total slot-fetches needed

If the audit list is regenerated, re-run `audit_strategy.py` and update
the constants in `index.html` (search for `slot-bar` and the
`tiles needing ≥1 slot refetch` row).

## Pod label

The poller targets `job-name=refetch-late-autumn-512` in the
`prithvi-training-default` namespace. To repoint at a different job,
edit the `LABEL` and `NS` constants at the top of `update_data.sh`.

## Why not just kubectl logs -f

Three reasons it justified a dashboard instead of a terminal tail:

1. **Live ETA countdown.** `kubectl logs` shows the pod's
   self-reported ETA only when it next emits a status line (every
   ~250 tiles). The dashboard interpolates between polls so the timer
   reads as a real clock.
2. **Multiple parallel watchers.** Three people watching different
   panes is cheaper than three `kubectl logs -f` sessions all
   round-tripping to the cluster.
3. **Survives session close.** Closes the laptop → poller keeps
   writing → reopen browser → state is still current.
