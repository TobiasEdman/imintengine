# Dashboards

Operational dashboards for ImintEngine.

## Files

- `../unified_training_dashboard.html` — main training-progress dashboard. Polls `training_log.json` and `tile_previews.json` every 30s. Now includes the W4.4 operational overlay (see below).
- `../crop_dashboard.html`, `../training_dashboard.html`, `../fetch_dashboard.html` — older / experimental dashboards. Add the overlay here too if needed (one `<script>` tag).
- `health_overlay.js` — drop-in operational badge overlay (W4.4).

## health_overlay.js

A single JS file that adds three top-right badges to any dashboard:

| Badge | Source | States |
|-------|--------|--------|
| **API** | `GET /v1/health` (FastAPI W4.1) | green = ok / red = unreachable / grey = unknown |
| **Data** | Prometheus `imint_fetch_run_completed_timestamp` | green ≤ 8 days / yellow > 8d / red > 14d |
| **Loss** | `training_log.json` last 8 epochs | green = decreasing / yellow = plateau detected |

Each badge has a hover tooltip with detail.

### Adding to an existing dashboard

```html
<script>
  window.HEALTH_OVERLAY_CONFIG = {
    apiBase: "https://imint-api.example.com",
    prometheusBase: "https://prometheus.observability",
  };
</script>
<script src="dashboards/health_overlay.js" defer></script>
```

If both URLs are left empty (default), the overlay still works for any same-origin API and skips Prometheus polling silently. Loss-plateau detection requires `training_log.json` reachable from the page.

### Configuration knobs

```js
window.HEALTH_OVERLAY_CONFIG = {
  apiBase: "",                      // default same-origin
  prometheusBase: "",               // default empty -> data freshness disabled
  refreshMs: 30000,                 // poll cadence
  lossPlateauEpochs: 8,             // window length for plateau detection
  lossPlateauTolerance: 0.001,      // relative-range threshold below which loss counts as plateau
  stalenessWarnHours: 192,          // 8 days
  stalenessAlertHours: 336,         // 14 days
};
```

### Why JS instead of server-side rendering

Per rollout plan W4.4: customer-facing dashboards. A pure-JS overlay means anyone with the dashboard URL gets the operational signals without us deploying a server-side rendering layer. Read-only by design — the overlay never mutates the underlying dashboard.
