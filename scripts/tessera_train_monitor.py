#!/usr/bin/env python3
"""Live tessera training monitor — training progress + ensemble state.

Renamed from campaign_dashboard.py 2026-08-31: that name belonged to the
retired fetch-era campaign monitor (1118 lines, --recoreg-dir/--watch
interface) whose k8s Deployment cloned main and read the OLD interface —
reusing its filename was what made commit 33d44dd's replacement
invisible. One name, one tool.

A dependency-free (stdlib only) HTTP server that:
  * polls ``kubectl logs`` for the tessera training job and parses the
    per-epoch metric lines the trainer prints (trainer.py:639),
  * reads the local ``ensemble_results.json`` for the P2 combiner /
    G1-gate state,
  * serves an auto-refreshing single-page dashboard in the DES visual
    identity (forest green / mint / Space Grotesk).

kubectl is shelled out with a short result cache so a browser refreshing
every few seconds never hammers the API server. Every kubectl failure
degrades to a labelled "unavailable" tile rather than crashing the page.

Usage:
    python3 scripts/tessera_train_monitor.py --port 8097
    # then open http://localhost:8097
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RESULTS_JSON = REPO / "data/distill/ensemble_results.json"
BASELINE_JSON = REPO / "data/distill/ensemble_baselineA_vs_gate.json"

# member name -> (accuracy_suite JSON, verified all-944 fallback OA). Every
# nfi_validation_*.json stores the all-944 accuracy_suite, so these are the
# whole-set numbers — NOT the same-209 held-out comparison (that lives in the
# G1 card: the 0.579 gate). The fallback is the verified published number used
# when the JSON isn't present locally (v8b base dump lives only on the PVC).
MEMBER_VALIDATION = {
    "v8b":              ("data/nfi/nfi_validation_v8b.json", 0.465),
    "v8b_markfukt":     ("data/nfi/nfi_validation_v8b_markfukt.json", 0.4661),
    "v8b_nmd2023_long": ("data/nfi/nfi_validation_v8b_nmd2023_long.json", 0.4597),
    "distill":          ("data/nfi/nfi_validation_distill.json", 0.5265),
    "tradslag":         ("data/nfi/nfi_validation_tradslag.json", 0.5201),
}

# One epoch line, e.g.:
#   Epoch   3/30 | loss=0.1421 | L_frac=0.5220 | val_mIoU=0.4123 |
#   worst=0.0012 (bakgrund) | lr=1.00e-04 | 47s
EPOCH_RE = re.compile(
    r"Epoch\s+(\d+)/(\d+)\s+\|\s+loss=([\d.]+)"
    r"(?:\s+\|\s+L_frac=([\d.]+))?"
    r"\s+\|\s+val_mIoU=([\d.]+)"
    r"\s+\|\s+worst=([\d.]+)\s+\(([^)]+)\)"
    r"\s+\|\s+lr=([\d.eE+-]+)\s+\|\s+(\d+)s"
)


class Cache:
    """Tiny time-boxed memoizer so kubectl runs at most once per ``ttl``."""

    def __init__(self, ttl: float) -> None:
        self.ttl = ttl
        self._val: dict | None = None
        self._t = 0.0

    def get(self, producer) -> dict:
        now = time.monotonic()
        if self._val is None or now - self._t > self.ttl:
            self._val = producer()
            self._t = now
        return self._val


def _kubectl(args: list[str], ctx: str, ns: str, timeout: float = 15.0) -> str:
    """Run kubectl; return stdout or '' on any failure (never raises)."""
    try:
        out = subprocess.run(
            ["kubectl", "--context", ctx, "-n", ns, *args],
            capture_output=True, text=True, timeout=timeout,
        )
        return out.stdout if out.returncode == 0 else ""
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return ""


def _training_status(ctx: str, ns: str, job: str) -> dict:
    """Pod phase + parsed epoch history for the training job."""
    sel = f"purpose={job.replace('train-', '').replace('-', '-')}"
    # Resolve the pod by job-name label (robust to pod-hash suffix).
    pods = _kubectl(
        ["get", "pods", "-l", f"job-name={job}",
         "-o", "jsonpath={.items[0].metadata.name}"
         "|{.items[0].status.phase}"
         "|{.items[0].spec.nodeName}"
         "|{.items[0].status.containerStatuses[0].state.running.startedAt}"],
        ctx, ns,
    )
    pod = phase = node = started = ""
    if pods and "|" in pods:
        pod, phase, node, started = (pods.split("|") + ["", "", "", ""])[:4]

    runtime_min = None
    if started:
        try:
            import datetime as _dt
            s = _dt.datetime.fromisoformat(started.replace("Z", "+00:00"))
            n = _dt.datetime.now(_dt.timezone.utc)
            runtime_min = round((n - s).total_seconds() / 60, 1)
        except ValueError:
            pass

    job_cond = _kubectl(
        ["get", "job", job, "-o",
         "jsonpath={.status.conditions[0].type}={.status.conditions[0].reason}"],
        ctx, ns,
    )

    epochs: list[dict] = []
    total = None
    if pod:
        logs = _kubectl(["logs", pod, "--tail", "200"], ctx, ns, timeout=25.0)
        for m in EPOCH_RE.finditer(logs):
            (ep, tot, loss, frac, miou, worst, wclass, lr, secs) = m.groups()
            total = int(tot)
            epochs.append({
                "epoch": int(ep), "loss": float(loss),
                "frac": float(frac) if frac else None,
                "val_miou": float(miou), "worst": float(worst),
                "worst_class": wclass, "secs": int(secs),
            })

    best = max((e["val_miou"] for e in epochs), default=None)
    return {
        "pod": pod, "phase": phase or "unknown", "node": node,
        "runtime_min": runtime_min, "job_condition": job_cond,
        "epochs": epochs[-12:], "n_epochs": len(epochs),
        "total_epochs": total, "best_val_miou": best,
    }


def _json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _ensemble_status() -> dict:
    res = _json(RESULTS_JSON) or {}
    base = _json(BASELINE_JSON) or {}
    members = []
    for name, (rel, fallback) in MEMBER_VALIDATION.items():
        j = _json(REPO / rel)
        oa = None
        if j:
            suite = j.get("accuracy_suite", j)
            oa = suite.get("overall_accuracy_5class")
        members.append({"name": name, "oa": oa if oa is not None else fallback,
                        "denom": "944"})
    rep = res.get("reported", {})
    return {
        "members": members,
        "gate": res.get("_meta", {}).get("g1_gate")
        or res.get("g1_reference_tradslag", {}).get("overall_accuracy_5class"),
        "reported": {
            "config": f"{rep.get('member_set','?')}/{rep.get('variant','?')}"
                      f"/{rep.get('head','?')}",
            "holdout_oa": (rep.get("holdout_suite") or {}).get(
                "overall_accuracy_5class"),
            "verdict": rep.get("g1_verdict"),
            "mcnemar_p": (rep.get("mcnemar_vs_tradslag") or {}).get("p_value"),
            "ci": rep.get("bootstrap_delta_vs_tradslag"),
        },
        "baseline_a": {
            "oa": base.get("baseline_A_209"),
            "verdict": base.get("verdict"),
        },
    }


PHASES = [
    ("P0 wiring", "done", "D1 tessera-wiring shipped (65 tests)"),
    ("P0 train", "live", "tessera member training on H100"),
    ("P1 dumps", "done", "v8b / markfukt / nmd2023 reproduced exactly"),
    ("CROMA prep", "done", "CROMA_base.pt cached; train yaml gated on G2"),
    ("P2 combiner", "done", "logreg/MLP, honest OOF selection"),
    ("G1/G2 verdict", "pending", "awaits tessera dump → P2 rerun"),
]


def build_status(ctx: str, ns: str, job: str) -> dict:
    canary = _kubectl(
        ["get", "pod", "srv02-gpu-canary", "-o",
         "jsonpath={.status.phase}"], ctx, ns)
    return {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "training": _training_status(ctx, ns, job),
        "ensemble": _ensemble_status(),
        "phases": [{"name": n, "state": s, "detail": d} for n, s, d in PHASES],
        "canary_phase": canary or "gone",
    }


PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Ensemble Campaign — Live</title>
<style>
 :root{--fg:#171717;--green:#1A4338;--green2:#245045;--mint:#cff8e4;
   --grey:#6b7280;--line:#e5e7eb;--live:#1A4338;--warn:#b45309;--bg:#fff}
 *{box-sizing:border-box}
 body{margin:0;font-family:'Space Grotesk',-apple-system,Segoe UI,sans-serif;
   background:var(--bg);color:var(--fg);line-height:1.45}
 .header{background:#fff;border-bottom:1px solid var(--line);padding:18px 28px;
   display:flex;align-items:center;gap:14px;position:sticky;top:0;z-index:5}
 .header .brand{font-weight:700;font-size:20px;letter-spacing:-.02em}
 .header .brand span{color:var(--green)}
 .header .sub{color:var(--grey);font-size:13px}
 .dot{width:9px;height:9px;border-radius:50%;background:var(--live);
   display:inline-block;animation:pulse 1.6s infinite}
 @keyframes pulse{0%,100%{opacity:1}50%{opacity:.25}}
 .wrap{max-width:1080px;margin:0 auto;padding:24px 28px 60px}
 .grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}
 @media(max-width:820px){.grid{grid-template-columns:1fr}}
 .card{border:1px solid var(--line);border-radius:12px;padding:18px 20px;
   background:#fff}
 .card h2{margin:0 0 12px;font-size:13px;text-transform:uppercase;
   letter-spacing:.08em;color:var(--green2)}
 .big{font-size:34px;font-weight:700;letter-spacing:-.02em}
 .muted{color:var(--grey);font-size:13px}
 .bar{height:10px;background:var(--line);border-radius:6px;overflow:hidden;
   margin:10px 0}
 .bar > i{display:block;height:100%;background:var(--green);border-radius:6px;
   transition:width .5s}
 table{width:100%;border-collapse:collapse;font-size:13px}
 th,td{text-align:left;padding:6px 8px;border-bottom:1px solid var(--line)}
 th{color:var(--grey);font-weight:600}
 td.num{text-align:right;font-variant-numeric:tabular-nums}
 .pill{display:inline-block;padding:2px 9px;border-radius:20px;font-size:11px;
   font-weight:600}
 .pill.done{background:var(--mint);color:var(--green)}
 .pill.live{background:var(--green);color:#fff}
 .pill.pending{background:#f3f4f6;color:var(--grey)}
 .pill.noise{background:#fef3c7;color:var(--warn)}
 .phaselist{display:flex;flex-direction:column;gap:8px}
 .phase{display:flex;align-items:center;gap:10px;font-size:13px}
 .phase .nm{font-weight:600;min-width:120px}
 .phase .dt{color:var(--grey)}
 footer{color:var(--grey);font-size:12px;margin-top:22px;text-align:center}
</style></head><body>
<div class="header">
  <span class="dot" id="dot"></span>
  <span class="brand"><span>IMINT</span> · Ensemble Campaign</span>
  <span class="sub" id="gen">loading…</span>
</div>
<div class="wrap">
  <div class="grid">
    <div class="card" id="train"></div>
    <div class="card" id="phases"></div>
  </div>
  <div class="grid" style="margin-top:18px">
    <div class="card" id="members"></div>
    <div class="card" id="gate"></div>
  </div>
  <footer>Auto-refreshes every 15 s · kubectl-backed · read-only</footer>
</div>
<script>
const f3=x=>x==null?'—':Number(x).toFixed(3);
const f1=x=>x==null?'—':Number(x).toFixed(1);
function pill(s){return `<span class="pill ${s==='within_noise'?'noise':s}">${s.replace('_',' ')}</span>`}
async function tick(){
  let s; try{s=await (await fetch('/api/status')).json()}catch(e){return}
  document.getElementById('gen').textContent='updated '+s.generated;
  const t=s.training, done=t.total_epochs?Math.round(100*t.n_epochs/t.total_epochs):0;
  const last=t.epochs[t.epochs.length-1];
  const cond=t.job_condition&&t.job_condition!=='='?` · <b>${t.job_condition}</b>`:'';
  document.getElementById('train').innerHTML=`<h2>P0 · Tessera training</h2>
    <div><span class="pill ${t.phase==='Running'?'live':(t.phase==='Succeeded'?'done':'pending')}">${t.phase}</span>
      <span class="muted"> ${t.node||''} · run ${f1(t.runtime_min)} min${cond}</span></div>
    <div class="big">${t.best_val_miou==null?'—':f3(t.best_val_miou)}<span class="muted" style="font-size:14px"> best val mIoU</span></div>
    <div class="bar"><i style="width:${done}%"></i></div>
    <div class="muted">${t.n_epochs}/${t.total_epochs||'?'} epochs${last?` · last loss ${f3(last.loss)}${last.frac!=null?` · L_frac ${f3(last.frac)}`:''}`:' · epoch 1 in progress…'}</div>
    ${t.epochs.length?`<table style="margin-top:12px"><tr><th>ep</th><th class="num">loss</th><th class="num">L_frac</th><th class="num">val mIoU</th><th class="num">s</th></tr>
      ${t.epochs.map(e=>`<tr><td>${e.epoch}</td><td class="num">${f3(e.loss)}</td><td class="num">${f3(e.frac)}</td><td class="num">${f3(e.val_miou)}</td><td class="num">${e.secs}</td></tr>`).join('')}</table>`:''}`;
  document.getElementById('phases').innerHTML=`<h2>Campaign phases</h2>
    <div class="phaselist">${s.phases.map(p=>`<div class="phase"><span class="pill ${p.state}">${p.state}</span><span class="nm">${p.name}</span><span class="dt">${p.detail}</span></div>`).join('')}</div>
    <div class="muted" style="margin-top:12px">srv02 canary: ${s.canary_phase}</div>`;
  const e=s.ensemble;
  document.getElementById('members').innerHTML=`<h2>Members (NFI 5-class OA, all 944)</h2>
    <table><tr><th>member</th><th class="num">OA</th><th>n</th></tr>
    ${e.members.map(m=>`<tr><td>${m.name}${m.name==='tradslag'?' · gate':''}</td><td class="num">${f3(m.oa)}</td><td>${m.denom}</td></tr>`).join('')}
    <tr><td>tessera</td><td class="num">—</td><td>training</td></tr></table>`;
  const r=e.reported, ci=r.ci;
  document.getElementById('gate').innerHTML=`<h2>G1 gate · ${f3(e.gate)}</h2>
    <div class="big">${f3(r.holdout_oa)} ${r.verdict?pill(r.verdict):''}</div>
    <div class="muted">reported combiner ${r.config} on the 209 · McNemar p=${f3(r.mcnemar_p)}${ci?` · CIΔ [${f3(ci.ci95_low)}, ${f3(ci.ci95_high)}]`:''}</div>
    <div class="muted" style="margin-top:8px">baseline A (softmax-mean): ${f3(e.baseline_a.oa)} · ${e.baseline_a.verdict||''}</div>
    <div class="muted" style="margin-top:8px">G1/G2 verdict pending the tessera member (variant iii = G2 ablation).</div>`;
}
tick(); setInterval(tick, 15000);
</script></body></html>"""


def make_handler(ctx: str, ns: str, job: str):
    cache = Cache(ttl=10.0)

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):  # quiet
            pass

        def do_GET(self):
            if self.path.startswith("/api/status"):
                body = json.dumps(
                    cache.get(lambda: build_status(ctx, ns, job))
                ).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
            else:
                body = PAGE.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return H


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8097)
    ap.add_argument("--context", default="icekube")
    ap.add_argument("--namespace", default="prithvi-training-default")
    ap.add_argument("--job", default="train-tessera-distill")
    args = ap.parse_args()

    handler = make_handler(args.context, args.namespace, args.job)
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), handler)
    print(f"campaign dashboard → http://localhost:{args.port}  "
          f"(job={args.job}, ctx={args.context})")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        srv.shutdown()


if __name__ == "__main__":
    main()
