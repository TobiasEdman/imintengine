"""The public annual-map surface exposes only what the showcase page needs (task 0131).

The API behind it is unauthenticated, so the manifest is the boundary: one host,
only the /v1/lclu prefix, CORS for the exact showcase origin, a rate limit, and
a Service pinned to the serving Job rather than to a label r7 and r8 share.
Only the Job carrying the runtime storage budget (r10) may be exposed: r8 has
none, and a stream of distinct requests could fill the demo volume.
"""
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "k8s/lclu-public.yaml"
HOST = "lclu.icedc.se"
SHOWCASE_ORIGIN = "https://tobiasedman.github.io"
NGINX = "nginx.ingress.kubernetes.io/"


def documents() -> dict:
    return {doc["kind"]: doc for doc in yaml.safe_load_all(MANIFEST.read_text())}


def test_ingress_exposes_only_the_lclu_prefix_on_one_tls_host():
    ingress = documents()["Ingress"]["spec"]
    (rule,) = ingress["rules"]
    assert rule["host"] == HOST
    assert [(p["path"], p["pathType"]) for p in rule["http"]["paths"]] == [("/v1/lclu", "Prefix")]
    assert ingress["tls"] == [{"hosts": [HOST], "secretName": "lclu-icedc-tls"}]


def test_cors_allows_exactly_the_showcase_origin_and_requests_are_bounded():
    notes = documents()["Ingress"]["metadata"]["annotations"]
    assert notes[NGINX + "enable-cors"] == "true"
    assert notes[NGINX + "cors-allow-origin"] == SHOWCASE_ORIGIN
    assert notes[NGINX + "cors-allow-credentials"] == "false"
    assert int(notes[NGINX + "limit-rps"]) > 0
    assert notes[NGINX + "proxy-body-size"] == "16k"
    assert notes["cert-manager.io/issuer"] == "letsencrypt-prod"


def test_service_is_pinned_to_the_serving_job_and_matches_the_backend():
    docs = documents()
    service, backend = docs["Service"], docs["Ingress"]["spec"]["rules"][0]["http"]["paths"][0]["backend"]
    assert service["spec"]["selector"] == {"job-name": "lclu-stockholm-0124-r10"}
    assert backend["service"]["name"] == service["metadata"]["name"]
    assert backend["service"]["port"]["number"] == service["spec"]["ports"][0]["port"] == 8000


def test_showcase_page_calls_the_public_host():
    config = (ROOT / "docs/js/lclu-config.js").read_text()
    assert re.search(r"apiBase:\s*'([^']*)'", config)[1] == f"https://{HOST}"
    index = (ROOT / "docs/index.html").read_text()
    assert 'id="tab-lclu" data-src="lclu.html"' in index
    assert '<script defer src="js/lclu-config.js"></script>' in (ROOT / "docs/lclu.html").read_text()
