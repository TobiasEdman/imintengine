#!/usr/bin/env python3
"""Lightweight HTTP server for training results.

Serves training logs, checkpoint metadata, and run listings from both
the CephFS data PVC (/data/results/) and the checkpoints PVC (/checkpoints/).

Endpoints:
  GET /api/runs                     → list all runs with latest metrics
  GET /api/runs/{id}/log            → training_log.json
  GET /api/runs/{id}/meta           → checkpoint_meta.json
  GET /api/runs/{id}/checkpoint     → download best_model.pt (streaming)
  GET /health                       → {"ok": true}

Usage:
  python scripts/results_server.py [--port 8095]

K8s: deployed as a persistent pod mounting both PVCs. Scale to 0 during
training (checkpoints PVC is RWO), scale back to 1 after.
"""
from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from socketserver import ThreadingMixIn
from urllib.parse import urlparse

RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", "/data/results"))
CHECKPOINTS_DIR = Path(os.environ.get("CHECKPOINTS_DIR", "/checkpoints"))
PORT = int(os.environ.get("PORT", 8095))


class ResultsHandler(BaseHTTPRequestHandler):

    def do_GET(self):
        path = urlparse(self.path).path.rstrip("/")

        if path == "/health":
            self._json({"ok": True})
        elif path == "/api/runs":
            self._list_runs()
        elif path.startswith("/api/runs/") and path.endswith("/log"):
            run_id = path.split("/")[3]
            self._serve_file(RESULTS_DIR / run_id / "training_log.json")
        elif path.startswith("/api/runs/") and path.endswith("/meta"):
            run_id = path.split("/")[3]
            self._serve_file(RESULTS_DIR / run_id / "checkpoint_meta.json")
        elif path.startswith("/api/runs/") and path.endswith("/checkpoint"):
            run_id = path.split("/")[3]
            self._serve_binary(CHECKPOINTS_DIR / run_id / "best_model.pt")
        else:
            self.send_error(404)

    def _list_runs(self):
        runs = []
        for d in sorted(RESULTS_DIR.iterdir()) if RESULTS_DIR.exists() else []:
            if not d.is_dir():
                continue
            run = {"id": d.name}
            log_path = d / "training_log.json"
            if log_path.exists():
                try:
                    log = json.loads(log_path.read_text())
                    run["status"] = log.get("status", "unknown")
                    run["best_metric"] = log.get("best_metric")
                    run["best_epoch"] = log.get("best_epoch")
                    run["n_epochs"] = len(log.get("epochs", []))
                except Exception:
                    pass
            meta_path = d / "checkpoint_meta.json"
            if meta_path.exists():
                try:
                    run["checkpoint"] = json.loads(meta_path.read_text())
                except Exception:
                    pass
            # Check if checkpoint file exists
            ckpt = CHECKPOINTS_DIR / d.name / "best_model.pt"
            run["checkpoint_available"] = ckpt.exists()
            runs.append(run)
        self._json({"runs": runs})

    def _json(self, data: dict):
        body = json.dumps(data, indent=2, default=str).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_file(self, path: Path):
        if not path.exists():
            self.send_error(404, f"Not found: {path.name}")
            return
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_binary(self, path: Path):
        if not path.exists():
            self.send_error(404, f"Checkpoint not available (PVC may be mounted by training job)")
            return
        size = path.stat().st_size
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Disposition", f'attachment; filename="{path.name}"')
        self.send_header("Content-Length", str(size))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        with open(path, "rb") as f:
            while chunk := f.read(1024 * 1024):
                self.wfile.write(chunk)

    def log_message(self, fmt, *args):
        pass  # suppress per-request logging


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=PORT)
    p.add_argument("--results-dir", default=str(RESULTS_DIR))
    p.add_argument("--checkpoints-dir", default=str(CHECKPOINTS_DIR))
    args = p.parse_args()

    global RESULTS_DIR, CHECKPOINTS_DIR
    RESULTS_DIR = Path(args.results_dir)
    CHECKPOINTS_DIR = Path(args.checkpoints_dir)

    server = ThreadedHTTPServer(("0.0.0.0", args.port), ResultsHandler)
    print(f"Results server on port {args.port}")
    print(f"  Results: {RESULTS_DIR}")
    print(f"  Checkpoints: {CHECKPOINTS_DIR}")
    server.serve_forever()


if __name__ == "__main__":
    main()
