"""Tiny HTTP server for the aircraft-parallax apex picker.

Serves static files from docs/ and adds two endpoints:

  GET  /api/picks  -> returns current saved picks JSON (empty object if none)
  POST /api/picks  -> overwrites user_picks.json with the request body

Picks live in outputs/showcase/aircraft_parallax/user_picks.json so they
survive sessions and end up in the repo on commit.

Run from repo root:
    python demos/aircraft_parallax/picker_server.py [--port 8096]

Then open http://localhost:8096/aircraft_parallax_picker.html

Writes require application/json, a local Host matching the listening port,
and a matching http Origin. Missing/null/foreign origins are rejected. The
bundled client uses same-origin fetch with application/json and needs no change.
Only loopback binding is supported; remote/custom-host/proxy use is unsupported.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCS_DIR = REPO_ROOT / "docs"
PICKS_FILE = REPO_ROOT / "outputs" / "showcase" / "aircraft_parallax" / "user_picks.json"


class PickerHandler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(DOCS_DIR), **kw)

    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.split("?", 1)[0] == "/api/picks":
            if PICKS_FILE.exists():
                payload = json.loads(PICKS_FILE.read_text())
            else:
                payload = {"picks": {}, "updated_at": None}
            self._send_json(200, payload)
            return
        return super().do_GET()

    def _valid_write_origin(self) -> bool:
        """Only same-origin browser writes to a local hostname are accepted.

        Fixed hostname allowlist prevents a forged Host/Origin pair (DNS
        rebinding). Proxies/custom hostnames are intentionally unsupported.
        """
        hosts = self.headers.get_all("Host", [])
        origins = self.headers.get_all("Origin", [])
        if len(hosts) != 1 or len(origins) != 1:
            return False
        try:
            host = urlsplit("http://" + hosts[0])
            origin = urlsplit(origins[0])
            port = self.server.server_address[1]
            allowed_hosts = {"127.0.0.1", "localhost", "::1"}
            return (
                host.hostname in allowed_hosts and host.path == ""
                and not host.query and not host.fragment
                and host.username is None and host.password is None
                and (host.port or 80) == port
                and origin.scheme == "http" and origin.path == ""
                and not origin.query and not origin.fragment
                and origin.username is None and origin.password is None
                and origin.hostname == host.hostname
                and (origin.port or 80) == port
            )
        except ValueError:
            return False

    def do_POST(self):
        if self.path.split("?", 1)[0] != "/api/picks":
            self._send_json(404, {"error": "unknown endpoint"})
            return
        if not self._valid_write_origin():
            self._send_json(403, {"error": "same-origin local Host and Origin required"})
            return
        content_types = self.headers.get_all("Content-Type", [])
        if (len(content_types) != 1 or
                content_types[0].split(";", 1)[0].strip().lower() != "application/json"):
            self._send_json(415, {"error": "Content-Type must be application/json"})
            return
        lengths = self.headers.get_all("Content-Length", [])
        if len(lengths) != 1 or self.headers.get("Transfer-Encoding"):
            self._send_json(400, {"error": "one Content-Length is required"})
            return
        try:
            length = int(lengths[0])
        except ValueError:
            self._send_json(400, {"error": "invalid Content-Length"})
            return
        if length <= 0 or length > 64 * 1024:
            self._send_json(400, {"error": "missing or oversized body"})
            return
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            self._send_json(400, {"error": f"bad JSON: {exc}"})
            return

        # Allow either {picks: {...}} or {<band>: {...}, ...}
        picks = payload.get("picks", payload) if isinstance(payload, dict) else None
        if not isinstance(picks, dict):
            self._send_json(400, {"error": "picks must be an object"})
            return

        record = {
            "picks": picks,
            "updated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "scene": "S2B_MSIL1C_20260111T104319_N0511_R008_T32VPK",
            "aoi_centre_wgs84": {"lat": 57.71809, "lon": 11.66456},
            "crop_size_native_px": 120,
            "metres_per_native_px": 10.0,
            "delta_t_seconds_from_b02": {
                "B02": 0.0, "B03": 0.5214, "B04": 0.9990, "B08": 0.2599,
            },
            "delta_t_source": "MTD_DS.xml detector 7 (parse_band_times_v2.py)",
        }
        PICKS_FILE.parent.mkdir(parents=True, exist_ok=True)
        PICKS_FILE.write_text(json.dumps(record, indent=2))
        self._send_json(200, {"ok": True, "saved_to": str(PICKS_FILE.relative_to(REPO_ROOT))})

    def log_message(self, fmt, *args):
        sys.stderr.write(f"[picker-server] {self.address_string()} {fmt % args}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8096)
    ap.add_argument("--host", choices=("127.0.0.1", "localhost"), default="127.0.0.1")
    args = ap.parse_args()

    httpd = ThreadingHTTPServer((args.host, args.port), PickerHandler)
    print(f"Picker server: http://{args.host}:{args.port}/aircraft_parallax_picker.html")
    print(f"Picks persist to: {PICKS_FILE.relative_to(REPO_ROOT)}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped.")


if __name__ == "__main__":
    main()
