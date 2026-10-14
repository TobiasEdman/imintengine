"""The top-up TESSERA re-probe Job: execute its exact heredoc offline over synthetic staging roots.

Same harness as tests/test_holdout_eval_tessera_retry.py (Codex's review evidence):
only ``/data/`` path literals are rewritten, installed-version metadata is mocked,
and the helper or the GeoTessera handle is replaced at the module seam.
"""
from __future__ import annotations

import ast
import hashlib
import json
import subprocess
import sys
import types
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import yaml

from scripts import enrich_tiles_tessera as REAL
from tests.test_holdout_eval_tessera_retry import BLOCK, SwallowingGT

REPO = Path(__file__).resolve().parents[1]
DOC = yaml.safe_load((REPO / "k8s/nfi-topup-tessera-retry-job.yaml").read_text())
CONTAINER = DOC["spec"]["template"]["spec"]["containers"][0]
SHELL = CONTAINER["args"][0]
BODY = SHELL.split("python - <<'PYEOF'\n", 1)[1].split("\nPYEOF", 1)[0]
ENV = {e["name"]: e["value"] for e in CONTAINER["env"]}
SRC = "geotessera-0.10.2"
UNCONFIRMED = "no_tessera_coverage_or_fetch_failed__verified_2026-10-14"


def _tile(path: Path, year: int, **fields) -> None:
    arrays = {"spectral": np.ones((6, 4, 4), np.float32), "year": np.array(year),
              "easting": np.array(500000), "northing": np.array(6500000), "marker": np.array(987),
              "tessera": np.ones((128, 4, 4), np.float16), "tessera_year": np.array(year),
              "has_tessera": np.array(1), "tessera_source": np.array(SRC)}
    arrays.update(fields)
    np.savez_compressed(path, **arrays)


def _partial():
    a = np.ones((128, 4, 4), np.float16)
    a[:, :, 0] = 0
    return a


def execute(tmp_path, monkeypatch, layout: dict, outcomes: dict | None = None, *, years="2018 2022",
            version="0.10.2", registry_blocks=None):
    """layout: {year: {name: fields}}; outcomes: {name: ok|zero|nan|partial|<failure reason>}."""
    outcomes = outcomes or {}
    (tmp_path / "manifests").mkdir(exist_ok=True)
    for year, tiles in layout.items():
        root = tmp_path / f"unified_v2_512_nfi{year}_staging"
        root.mkdir(exist_ok=True)
        for name, fields in tiles.items():
            if not (root / f"{name}.npz").exists():
                _tile(root / f"{name}.npz", year, **fields)
    calls = []

    def enrich(path, gt, skip_existing):
        assert skip_existing is False
        name = Path(path).stem; calls.append(name); outcome = outcomes[name]
        if outcome in ("ok", "zero", "nan", "partial"):
            with np.load(path, allow_pickle=True) as z:
                arrays = {k: z[k] for k in z.files}
            year = int(arrays["year"])
            arrays["tessera"] = {"ok": np.ones((128, 4, 4), np.float16), "zero": np.zeros((128, 4, 4), np.float16),
                                 "nan": np.full((128, 4, 4), np.nan, np.float16), "partial": _partial()}[outcome]
            arrays["has_tessera"] = np.array(1); arrays["tessera_year"] = np.array(year)
            np.savez_compressed(path, **arrays)
            return {"status": "ok"}
        return {"status": "failed", "reason": outcome}

    fake = types.ModuleType("scripts.enrich_tiles_tessera")
    for attr in ("TESSERA_YEARS", "_clamp_year_for_tessera", "_infer_tile_year"):
        setattr(fake, attr, getattr(REAL, attr))
    fake.enrich_one_tile = enrich
    geo = types.ModuleType("geotessera")
    geo.GeoTessera = lambda **kwargs: types.SimpleNamespace(registry=None, fetch_embeddings=None)
    if registry_blocks is not None:
        fake.enrich_one_tile = REAL.enrich_one_tile
        geo.GeoTessera = lambda **kwargs: SwallowingGT(registry_blocks, calls)
    monkeypatch.setenv("YEARS", years)
    monkeypatch.setenv("GEOTESSERA_VERSION", "0.10.2")
    monkeypatch.setenv("SCRIPTS_COMMIT", ENV["SCRIPTS_COMMIT"])
    body = BODY.replace('"/data/', json.dumps(str(tmp_path) + "/")[:-1])   # also rewrites f"/data/...
    ast.parse(body)
    code = 0
    with patch.dict(sys.modules, {"scripts.enrich_tiles_tessera": fake, "geotessera": geo}), \
            patch("importlib.metadata.version", return_value=version):
        try:
            exec(compile(body, "topup-reprobe-heredoc", "exec"), {})
        except SystemExit as exc:
            code = exc.code
    reports = {int(p.stem[3:7]): json.loads(p.read_text())
               for p in (tmp_path / "manifests").glob("nfi*_topup_tessera_retry.json")}
    return calls, code, reports


def _npz(tmp_path, year, name):
    with np.load(tmp_path / f"unified_v2_512_nfi{year}_staging/{name}.npz", allow_pickle=True) as z:
        return {k: z[k] for k in z.files}


def _sha(tmp_path, year, name):
    return hashlib.sha256((tmp_path / f"unified_v2_512_nfi{year}_staging/{name}.npz").read_bytes()).hexdigest()


def test_shell_and_python_parse():
    subprocess.run(["bash", "-n"], input=SHELL, text=True, check=True)
    ast.parse(BODY)


def test_a_valid_tile_is_not_a_target_and_stays_byte_identical(tmp_path, monkeypatch):
    calls, code, reports = execute(tmp_path, monkeypatch, {2018: {"good": {}}})
    before = _sha(tmp_path, 2018, "good")
    calls, code, reports = execute(tmp_path, monkeypatch, {2018: {"good": {}}})
    assert code == 0 and calls == [] and reports[2018]["counts"]["targets"] == 0
    assert _sha(tmp_path, 2018, "good") == before


@pytest.mark.parametrize("fields", [
    {"tessera": _partial()},                                     # an undelivered block
    {"tessera_year": np.array(2017)},                            # wrong year
    {"tessera": np.full((128, 4, 4), np.inf, np.float16)},       # non-finite
    {"tessera": np.ones((64, 4, 4), np.float16)},                # wrong shape
])
def test_a_usable_flag_on_an_invalid_embedding_is_reprobed_and_recovered(tmp_path, monkeypatch, fields):
    calls, code, reports = execute(tmp_path, monkeypatch, {2018: {"bad": fields}}, {"bad": "ok"})
    r = reports[2018]
    assert code == 0 and calls == ["bad"] and r["prescan_invalid"] == ["bad"] and r["recovered"] == ["bad"]
    t = _npz(tmp_path, 2018, "bad")
    assert int(t["has_tessera"]) == 1 and str(t["tessera_source"]) == SRC and int(t["marker"]) == 987


def test_unconfirmed_holes_are_reprobed_and_classified(tmp_path, monkeypatch):
    hole = {"has_tessera": np.array(0), "tessera_gap_reason": np.array(UNCONFIRMED)}
    layout = {2018: {"back": hole, "empty": hole}, 2022: {"down": hole}}
    calls, code, reports = execute(tmp_path, monkeypatch, layout,
                                   {"back": "ok", "empty": "zero", "down": "fetch: TimeoutError: upstream"})
    assert sorted(calls) == ["back", "down", "empty"]
    assert reports[2018]["recovered"] == ["back"] and reports[2018]["coverage_hole"] == ["empty"]
    assert reports[2022]["unresolved"] == ["down"] and "TimeoutError" in reports[2022]["unresolved_reasons"]["down"]
    assert isinstance(code, str) and "[2022]" in code                 # non-zero, names the root
    assert "confirmed_by_single_worker_reprobe" in str(_npz(tmp_path, 2018, "empty")["tessera_gap_reason"])
    assert str(_npz(tmp_path, 2022, "down")["tessera_gap_reason"]) == UNCONFIRMED   # left as it was


def test_a_swallowed_download_is_never_a_confirmed_hole(tmp_path, monkeypatch):
    """Real pinned helper over a 0.10.2-style handle that logs and skips a failed download."""
    hole = {"has_tessera": np.array(0), "tessera_gap_reason": np.array(UNCONFIRMED)}
    calls, code, reports = execute(tmp_path, monkeypatch, {2022: {"t": hole}}, years="2022", registry_blocks=[BLOCK])
    r = reports[2022]
    assert r["unresolved"] == ["t"] and r["coverage_hole"] == [] and isinstance(code, str)
    assert "download_incomplete: 0/1" in r["unresolved_reasons"]["t"]
    assert str(_npz(tmp_path, 2022, "t")["tessera_gap_reason"]) == UNCONFIRMED


def test_an_empty_registry_is_a_confirmed_hole(tmp_path, monkeypatch):
    hole = {"has_tessera": np.array(0), "tessera_gap_reason": np.array(UNCONFIRMED)}
    calls, code, reports = execute(tmp_path, monkeypatch, {2022: {"t": hole}}, years="2022", registry_blocks=[])
    assert code == 0 and reports[2022]["coverage_hole"] == ["t"]
    reason = str(_npz(tmp_path, 2022, "t")["tessera_gap_reason"])
    assert reason.startswith("no_tessera_coverage__no_tessera_tiles_for_year=2022") and "confirmed_by" in reason


def test_a_nan_result_is_never_promoted(tmp_path, monkeypatch):
    hole = {"has_tessera": np.array(0), "tessera_gap_reason": np.array(UNCONFIRMED)}
    calls, code, reports = execute(tmp_path, monkeypatch, {2018: {"t": hole}}, {"t": "nan"}, years="2018")
    assert isinstance(code, str) and reports[2018]["unresolved"] == ["t"]
    assert "invalid_embedding" in reports[2018]["unresolved_reasons"]["t"]
    assert int(_npz(tmp_path, 2018, "t")["has_tessera"]) == 0


def test_partial_after_a_complete_probe_is_recorded(tmp_path, monkeypatch):
    calls, code, reports = execute(tmp_path, monkeypatch, {2018: {"t": {"tessera": _partial()}}}, {"t": "partial"},
                                   years="2018")
    assert code == 0 and reports[2018]["recovered"] == ["t"] and reports[2018]["partial_coverage"] == {"t": 0.25}


def test_a_rerun_skips_confirmed_holes_and_recovered_tiles(tmp_path, monkeypatch):
    hole = {"has_tessera": np.array(0), "tessera_gap_reason": np.array(UNCONFIRMED)}
    layout = {2018: {"back": hole, "empty": hole}}
    execute(tmp_path, monkeypatch, layout, {"back": "ok", "empty": "zero"}, years="2018")
    calls, code, reports = execute(tmp_path, monkeypatch, layout, {"back": "nan", "empty": "nan"}, years="2018")
    assert code == 0 and calls == [] and reports[2018]["counts"]["targets"] == 0


def test_refuses_before_any_probe_if_the_tessera_job_has_not_run(tmp_path, monkeypatch):
    root = tmp_path / "unified_v2_512_nfi2018_staging"; root.mkdir()
    np.savez_compressed(root / "raw.npz", spectral=np.ones((6, 4, 4), np.float32), year=np.array(2018))
    hole = {"has_tessera": np.array(0), "tessera_gap_reason": np.array(UNCONFIRMED)}
    calls, code, reports = execute(tmp_path, monkeypatch, {2022: {"t": hole}}, {"t": "ok"})
    assert isinstance(code, str) and "run nfi-topup-tessera first" in code and calls == [] and reports == {}


def test_a_year_without_a_staging_root_is_skipped(tmp_path, monkeypatch):
    calls, code, reports = execute(tmp_path, monkeypatch, {2018: {"good": {}}}, years="2018 2019")
    assert code == 0 and set(reports) == {2018}


def test_version_mismatch_refuses_before_any_write(tmp_path, monkeypatch):
    hole = {"has_tessera": np.array(0), "tessera_gap_reason": np.array(UNCONFIRMED)}
    calls, code, reports = execute(tmp_path, monkeypatch, {2018: {"t": hole}}, {"t": "ok"}, version="0.10.3")
    assert isinstance(code, str) and "REFUSING" in code and calls == [] and reports == {}


def test_source_is_pinned_and_guarded_before_any_python_runs():
    assert ENV["SCRIPTS_COMMIT"] == "d4d24f9ab1581ecdc8db739d4271905806401b6a"
    i_checkout = SHELL.index('git checkout -q "$SCRIPTS_COMMIT"')
    i_head = SHELL.index('test "$(git rev-parse HEAD)" = "$SCRIPTS_COMMIT"')
    i_sha = SHELL.index("sha256sum -c -")
    i_py = SHELL.index("python scripts/enrich_tiles_tessera.py --help")
    assert i_checkout < i_head < i_sha < i_py
    blob = subprocess.run(["git", "-C", str(REPO), "show", f"{ENV['SCRIPTS_COMMIT']}:scripts/enrich_tiles_tessera.py"],
                          capture_output=True)
    if blob.returncode != 0:
        pytest.skip("pinned commit not in this clone")
    assert hashlib.sha256(blob.stdout).hexdigest() == ENV["HELPER_SHA256"]
