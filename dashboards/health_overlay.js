/* Customer-facing dashboard overlay — W4.4.
 *
 * Adds three operational signals on top of any existing ImintEngine
 * dashboard:
 *   1. API health badge (green/red) polling /v1/health
 *   2. Data freshness badge (green/yellow/red) based on
 *      imint_fetch_run_completed_timestamp from Prometheus
 *   3. Training-loss-plateau alert (yellow/red) based on the last
 *      8 epochs of training_log.json
 *
 * Drop into any dashboard with:
 *   <script src="health_overlay.js"></script>
 *   <div id="health-overlay"></div>
 *
 * Configuration via window.HEALTH_OVERLAY_CONFIG before script load:
 *   apiBase:        URL prefix for /v1/health (default: same origin)
 *   prometheusBase: URL prefix for Prometheus query API (default: empty -> data freshness disabled)
 *   refreshMs:      poll interval (default 30000)
 */
(function () {
  "use strict";

  const cfg = Object.assign(
    {
      apiBase: "",
      prometheusBase: "",
      refreshMs: 30000,
      lossPlateauEpochs: 8,
      lossPlateauTolerance: 0.001,
      stalenessWarnHours: 24 * 8, // 8 days = expected weekly cadence + slack
      stalenessAlertHours: 24 * 14,
    },
    window.HEALTH_OVERLAY_CONFIG || {}
  );

  const styles = `
    #health-overlay {
      position: fixed; top: 12px; right: 12px;
      display: flex; gap: 8px; z-index: 9999;
      font-family: -apple-system, BlinkMacSystemFont, sans-serif;
      font-size: 12px;
    }
    .ho-badge {
      display: inline-flex; align-items: center; gap: 6px;
      padding: 6px 10px; border-radius: 6px;
      background: rgba(15, 23, 42, 0.85);
      border: 1px solid rgba(148, 163, 184, 0.3);
      color: #e2e8f0; backdrop-filter: blur(6px);
    }
    .ho-dot {
      width: 8px; height: 8px; border-radius: 50%;
      box-shadow: 0 0 6px currentColor;
    }
    .ho-ok { color: #10b981; }
    .ho-warn { color: #f59e0b; }
    .ho-error { color: #ef4444; }
    .ho-unknown { color: #6b7280; }
    .ho-tt {
      display: none; position: absolute; top: 32px; right: 0;
      background: #0f172a; border: 1px solid #1e293b; border-radius: 6px;
      padding: 8px 10px; min-width: 220px; max-width: 320px;
      color: #cbd5e1; font-size: 11px; line-height: 1.4;
    }
    .ho-badge:hover .ho-tt { display: block; }
  `;

  function injectStyles() {
    const s = document.createElement("style");
    s.textContent = styles;
    document.head.appendChild(s);
  }

  function ensureContainer() {
    let el = document.getElementById("health-overlay");
    if (el) return el;
    el = document.createElement("div");
    el.id = "health-overlay";
    document.body.appendChild(el);
    return el;
  }

  function setBadge(slot, level, label, tooltip) {
    const container = ensureContainer();
    let badge = container.querySelector(`[data-slot="${slot}"]`);
    if (!badge) {
      badge = document.createElement("div");
      badge.className = "ho-badge";
      badge.dataset.slot = slot;
      badge.style.position = "relative";
      container.appendChild(badge);
    }
    badge.className = `ho-badge ho-${level}`;
    badge.innerHTML = `
      <span class="ho-dot"></span>
      <span>${label}</span>
      <span class="ho-tt">${tooltip || ""}</span>
    `;
  }

  // ── 1. API health ────────────────────────────────────────────────────────
  async function pollApiHealth() {
    try {
      const r = await fetch(`${cfg.apiBase}/v1/health`, { cache: "no-store" });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const body = await r.json();
      setBadge(
        "api",
        "ok",
        "API",
        `${body.api_version} · schema ${body.schema_version}<br>${body.embedding_identity}`
      );
    } catch (e) {
      setBadge("api", "error", "API", `Unreachable: ${e.message}`);
    }
  }

  // ── 2. Data freshness ────────────────────────────────────────────────────
  async function pollDataFreshness() {
    if (!cfg.prometheusBase) {
      setBadge("data", "unknown", "Data", "Prometheus URL not configured");
      return;
    }
    try {
      const url = `${cfg.prometheusBase}/api/v1/query?query=imint_fetch_run_completed_timestamp`;
      const r = await fetch(url, { cache: "no-store" });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const data = await r.json();
      const result = data?.data?.result?.[0];
      if (!result) {
        setBadge(
          "data",
          "warn",
          "Data",
          "No fetch metric yet — has the CronJob run since deploy?"
        );
        return;
      }
      const ts = parseFloat(result.value[1]);
      const ageHours = (Date.now() / 1000 - ts) / 3600;
      const lastIso = new Date(ts * 1000).toISOString();
      let level = "ok";
      if (ageHours > cfg.stalenessAlertHours) level = "error";
      else if (ageHours > cfg.stalenessWarnHours) level = "warn";
      setBadge(
        "data",
        level,
        `Data ${ageHours < 24 ? Math.round(ageHours) + "h" : Math.round(ageHours / 24) + "d"}`,
        `Last fetch: ${lastIso}<br>Age: ${ageHours.toFixed(1)}h`
      );
    } catch (e) {
      setBadge("data", "error", "Data", `Prometheus query failed: ${e.message}`);
    }
  }

  // ── 3. Loss plateau ──────────────────────────────────────────────────────
  async function pollLossPlateau() {
    try {
      const r = await fetch("training_log.json?t=" + Date.now(), { cache: "no-store" });
      if (!r.ok) {
        setBadge("loss", "unknown", "Loss", "training_log.json not reachable");
        return;
      }
      const log = await r.json();
      const series = (log?.epochs || log?.history || log)?.slice?.(-cfg.lossPlateauEpochs);
      if (!series || series.length < cfg.lossPlateauEpochs) {
        setBadge(
          "loss",
          "ok",
          "Loss",
          `Need ≥${cfg.lossPlateauEpochs} epochs to detect plateau (have ${series?.length || 0})`
        );
        return;
      }
      const losses = series
        .map((e) => e?.val_loss ?? e?.loss ?? e?.train_loss)
        .filter((v) => typeof v === "number");
      if (losses.length < cfg.lossPlateauEpochs) {
        setBadge("loss", "ok", "Loss", "Insufficient loss data in epochs");
        return;
      }
      const min = Math.min(...losses);
      const max = Math.max(...losses);
      const range = max - min;
      const mean = losses.reduce((a, b) => a + b, 0) / losses.length;
      const relRange = mean > 0 ? range / mean : 0;
      let level = "ok";
      let label = `Loss ${mean.toFixed(3)}`;
      let tt = `Mean of last ${cfg.lossPlateauEpochs} epochs: ${mean.toFixed(4)}<br>Range: ${range.toFixed(4)}`;
      if (relRange < cfg.lossPlateauTolerance) {
        level = "warn";
        label = "Plateau";
        tt += `<br><b>Plateau detected</b> — relative range ${(relRange * 100).toFixed(2)}% < ${(cfg.lossPlateauTolerance * 100).toFixed(2)}%`;
      }
      setBadge("loss", level, label, tt);
    } catch (e) {
      setBadge("loss", "unknown", "Loss", `Could not parse training_log: ${e.message}`);
    }
  }

  function pollAll() {
    pollApiHealth();
    pollDataFreshness();
    pollLossPlateau();
  }

  function init() {
    injectStyles();
    ensureContainer();
    pollAll();
    setInterval(pollAll, cfg.refreshMs);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
