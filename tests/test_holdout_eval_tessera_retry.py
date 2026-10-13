"""The tessera re-probe Job: execute its exact heredoc offline.

Harness adapted from Codex's review evidence (reviews 5496125869 and 5496217165):
only the ``/data/`` path literals are rewritten, installed-version metadata is
mocked, and the helper or the GeoTessera handle is replaced at the module seam.
"""
from __future__ import annotations

import ast
import hashlib
import json
import logging
import subprocess
import sys
import types
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import yaml

from scripts import enrich_tiles_tessera as REAL

REPO = Path(__file__).resolve().parents[1]
DOC = yaml.safe_load((REPO / "k8s/holdout-eval-tessera-retry-job.yaml").read_text())
SHELL = DOC["spec"]["template"]["spec"]["containers"][0]["args"][0]
BODY = SHELL.split("python - <<'PYEOF'\n", 1)[1].split("\nPYEOF", 1)[0]
SRC = "geotessera-0.10.2"
BLOCK = (2022, 15.0, 58.0)


class SwallowingGT:
    """geotessera 0.10.2's fetch_embeddings: a failed download is logged and skipped (core.py:533-543)."""

    def __init__(self, registry_blocks, calls):
        self.calls = calls
        self.registry = types.SimpleNamespace(load_blocks_for_region=lambda **kwargs: list(registry_blocks))
        self.logger = logging.getLogger("offline-geotessera")

    def fetch_embeddings(self, tiles_to_fetch, progress_callback=None):
        for year, lon, lat in tiles_to_fetch:
            try:
                self.calls.append((lon, lat, year))
                raise TimeoutError("offline transient-download witness")
            except Exception as e:  # noqa: BLE001 — mirrors the release
                self.logger.warning("Failed to download tile (%s, %s): %s", lon, lat, e)
                continue
        yield from ()  # a generator like the release's, which yields only on a successful download


def execute(tmp_path, monkeypatch, outcomes, *, existing=None, report=None, version="0.10.2",
            extra_tiles=None, registry_blocks=None):
    data = tmp_path / "holdout_eval_512"; data.mkdir(exist_ok=True)
    reports = tmp_path / "manifests"; reports.mkdir(exist_ok=True)
    for name in list(outcomes) + list(extra_tiles or {}):
        path = data / f"{name}.npz"
        if not path.exists() and outcomes.get(name) != "missing":
            np.savez_compressed(path, spectral=np.ones((6, 4, 4), np.float32),
                                tessera=np.zeros((128, 4, 4), np.float16), has_tessera=0, year=2022,
                                easting=500000, northing=6500000, tessera_source="old", marker=987)
    for name, fields in {**(existing or {}), **(extra_tiles or {})}.items():
        p = data / f"{name}.npz"
        with np.load(p) as z:
            arrays = {k: z[k] for k in z.files}
        arrays.update(fields)
        np.savez_compressed(p, **arrays)
    original = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in data.glob("*.npz")}
    report = report or {"dir": str(data), "source": SRC, "failed_or_hole": list(outcomes)}
    report["dir"] = report["dir"].replace("/data/", str(tmp_path) + "/")  # same rewrite the heredoc gets
    (reports / "holdout_eval_512_tessera.json").write_text(json.dumps(report))
    calls = []

    def enrich(path, gt, skip_existing):
        assert skip_existing is False
        name = Path(path).stem; calls.append(name); outcome = outcomes[name]
        if outcome in ("ok", "zero", "nan", "inf", "badshape", "partial"):
            with np.load(path) as z:
                arrays = {k: z[k] for k in z.files}
            if outcome == "badshape":
                arrays["tessera"] = np.ones((64, 4, 4), np.float16)
            elif outcome == "partial":
                arrays["tessera"] = partial_embedding()
            else:
                value = {"ok": 1.0, "zero": 0.0, "nan": float("nan"), "inf": float("inf")}[outcome]
                arrays["tessera"] = np.full((128, 4, 4), value, np.float16)
            arrays["has_tessera"] = np.array(1); arrays["tessera_year"] = np.array(2022)
            np.savez_compressed(path, **arrays)
            return {"status": "ok"}
        return {"status": "failed", "reason": outcome}

    fake = types.ModuleType("scripts.enrich_tiles_tessera")
    for attr in ("TESSERA_YEARS", "_clamp_year_for_tessera", "_infer_tile_year"):
        setattr(fake, attr, getattr(REAL, attr))
    fake.enrich_one_tile = enrich
    geo = types.ModuleType("geotessera")
    geo.GeoTessera = lambda **kwargs: types.SimpleNamespace(registry=None, fetch_embeddings=None)  # never reached by the fake helper
    if registry_blocks is not None:   # composed: the real pinned helper over a swallowing handle
        fake.enrich_one_tile = REAL.enrich_one_tile
        geo.GeoTessera = lambda **kwargs: SwallowingGT(registry_blocks, calls)
    monkeypatch.setenv("GEOTESSERA_VERSION", "0.10.2")
    monkeypatch.setenv("SCRIPTS_COMMIT", "d4d24f9ab1581ecdc8db739d4271905806401b6a")
    body = BODY.replace('"/data/', json.dumps(str(tmp_path) + "/")[:-1])
    ast.parse(body)  # only absolute /data literals changed, not control flow
    code = 0
    with patch.dict(sys.modules, {"scripts.enrich_tiles_tessera": fake, "geotessera": geo}), \
            patch("importlib.metadata.version", return_value=version):
        try:
            exec(compile(body, "retry-manifest-heredoc", "exec"), {})
        except SystemExit as exc:
            code = exc.code
    out = reports / "holdout_eval_512_tessera_retry.json"
    result = json.loads(out.read_text()) if out.exists() else None
    return data, original, calls, code, result


def flag_and_reason(data, name):
    with np.load(data / f"{name}.npz") as z:
        return int(z["has_tessera"]), str(z["tessera_gap_reason"]) if "tessera_gap_reason" in z.files else None


def sha(data, name):
    return hashlib.sha256((data / f"{name}.npz").read_bytes()).hexdigest()


def test_shell_and_python_parse():
    subprocess.run(["bash", "-n"], input=SHELL, text=True, check=True)
    ast.parse(BODY)


def test_recovery_and_zero_block_controls(tmp_path, monkeypatch):
    data, _, calls, code, result = execute(tmp_path, monkeypatch, {"recover": "ok", "hole": "zero"})
    assert code == 0 and sorted(calls) == ["hole", "recover"]
    assert result["recovered"] == ["recover"] and result["coverage_hole"] == ["hole"]
    with np.load(data / "recover.npz") as z:
        assert z["has_tessera"] == 1 and str(z["tessera_source"]) == SRC and z["marker"] == 987
        assert z["tessera_year"] == 2022 and "tessera_gap_reason" not in z.files
    flag, reason = flag_and_reason(data, "hole")
    assert flag == 0 and "confirmed_by_single_worker_reprobe" in reason


def test_version_mismatch_refuses_before_tile_writes(tmp_path, monkeypatch):
    data, before, calls, code, result = execute(tmp_path, monkeypatch, {"tile": "ok"}, version="0.10.3")
    assert code != 0 and not calls and result is None
    assert before["tile.npz"] == sha(data, "tile")


def test_explicit_fetch_failure_stays_unresolved_and_untouched(tmp_path, monkeypatch):
    data, before, _, code, result = execute(tmp_path, monkeypatch, {"tile": "fetch: TimeoutError: temporary upstream outage"})
    assert code != 0 and result["unresolved"] == ["tile"] and result["coverage_hole"] == []
    assert "TimeoutError" in result["unresolved_reasons"]["tile"]
    assert before["tile.npz"] == sha(data, "tile")


def test_swallowed_download_failure_is_not_a_confirmed_hole(tmp_path, monkeypatch):
    """Real pinned helper + a 0.10.2-style handle that logs and skips a failed download."""
    data, before, calls, code, result = execute(tmp_path, monkeypatch, {"tile": "unused"}, registry_blocks=[BLOCK])
    assert calls == [(15.0, 58.0, 2022)]
    assert code != 0 and result["unresolved"] == ["tile"] and result["coverage_hole"] == []
    assert "download_incomplete: 0/1" in result["unresolved_reasons"]["tile"]
    assert before["tile.npz"] == sha(data, "tile")


def test_empty_registry_is_a_confirmed_hole(tmp_path, monkeypatch):
    """Real pinned helper: the registry lists no block for the bbox and year."""
    data, _, calls, code, result = execute(tmp_path, monkeypatch, {"tile": "unused"}, registry_blocks=[])
    assert calls == [] and code == 0 and result["coverage_hole"] == ["tile"]
    flag, reason = flag_and_reason(data, "tile")
    assert flag == 0 and reason.startswith("no_tessera_coverage__no_tessera_tiles_for_year=2022")


def test_rerun_skips_a_previously_recovered_tile(tmp_path, monkeypatch):
    execute(tmp_path, monkeypatch, {"tile": "ok"})
    data, _, calls, code, result = execute(tmp_path, monkeypatch, {"tile": "fetch: OSError: transient failure"})
    assert code == 0 and calls == [] and result["already_recovered"] == ["tile"]
    with np.load(data / "tile.npz") as z:
        assert np.all(z["tessera"] == 1) and z["has_tessera"] == 1 and str(z["tessera_source"]) == SRC


@pytest.mark.parametrize("have", [2021, None])
def test_wrong_or_missing_year_is_re_probed_not_kept(tmp_path, monkeypatch, have):
    fields = {"tessera": np.ones((128, 4, 4), np.float16), "has_tessera": np.array(1), "tessera_source": np.array(SRC)}
    if have is not None:
        fields["tessera_year"] = np.array(have)
    data, _, calls, code, result = execute(tmp_path, monkeypatch, {"tile": "ok"}, existing={"tile": fields})
    assert calls == ["tile"] and result["already_recovered"] == []
    assert code == 0 and result["recovered"] == ["tile"]
    with np.load(data / "tile.npz") as z:
        assert z["tessera_year"] == 2022 and z["has_tessera"] == 1


def test_missing_target_is_listed_and_fails_the_job(tmp_path, monkeypatch):
    _, _, calls, code, result = execute(tmp_path, monkeypatch, {"gone": "missing"})
    assert code != 0 and calls == [] and result["missing_in_copy"] == ["gone"]


@pytest.mark.parametrize("outcome", ["nan", "inf", "badshape"])
def test_invalid_embedding_is_never_promoted(tmp_path, monkeypatch, outcome):
    data, _, _, code, result = execute(tmp_path, monkeypatch, {"tile": outcome})
    assert code != 0 and result["unresolved"] == ["tile"] and result["recovered"] == []
    assert "invalid_embedding" in result["unresolved_reasons"]["tile"]
    assert flag_and_reason(data, "tile")[0] == 0


def test_prescan_targets_usable_flags_on_nan_or_wrong_year_embeddings(tmp_path, monkeypatch):
    """Tiles the first pass flagged usable are re-checked over the whole copy, not only the listed ones."""
    stamped = {"has_tessera": np.array(1), "tessera_source": np.array(SRC)}
    extra = {"nan_tile": {**stamped, "tessera": np.full((128, 4, 4), float("nan"), np.float16), "tessera_year": np.array(2022)},
             "year_tile": {**stamped, "tessera": np.ones((128, 4, 4), np.float16), "tessera_year": np.array(2021)},
             "good_tile": {**stamped, "tessera": np.ones((128, 4, 4), np.float16), "tessera_year": np.array(2022)},
             "part_tile": {**stamped, "tessera": partial_embedding(), "tessera_year": np.array(2022)}}
    data, before, calls, code, result = execute(
        tmp_path, monkeypatch, {"listed": "ok", "nan_tile": "nan", "year_tile": "ok", "part_tile": "ok", "good_tile": "unused"},
        extra_tiles=extra, report={"dir": "/data/holdout_eval_512", "source": SRC, "failed_or_hole": ["listed"]})
    assert result["prescan_invalid"] == ["nan_tile", "part_tile", "year_tile"]
    assert sorted(calls) == ["listed", "nan_tile", "part_tile", "year_tile"]
    assert result["recovered"] == ["listed", "part_tile", "year_tile"] and result["unresolved"] == ["nan_tile"] and code != 0
    assert result["counts"]["targets"] == 4 and flag_and_reason(data, "nan_tile")[0] == 0
    assert result["partial_coverage"] == {}   # the re-probe delivered the whole tile
    assert before["good_tile.npz"] == sha(data, "good_tile")


def partial_embedding():
    """One block of a two-block tile missing: a column of pixels whose 128 channels are all zero."""
    a = np.ones((128, 4, 4), np.float16)
    a[:, :, 0] = 0
    return a


def test_partial_delivery_after_complete_probe_is_recorded_not_hidden(tmp_path, monkeypatch):
    data, _, calls, code, result = execute(tmp_path, monkeypatch, {"tile": "partial"})
    assert calls == ["tile"] and code == 0 and result["recovered"] == ["tile"]
    assert result["partial_coverage"] == {"tile": 0.25}
    with np.load(data / "tile.npz") as z:
        assert z["has_tessera"] == 1 and str(z["tessera_source"]) == SRC


def test_wrong_report_root_or_source_is_refused(tmp_path, monkeypatch):
    _, _, calls, code, result = execute(tmp_path, monkeypatch, {"tile": "ok"}, report={
        "dir": "/data/unified_v2_512", "source": "wrong-version", "failed_or_hole": ["tile"]})
    assert code != 0 and calls == [] and result is None


def test_helper_pin_matches_the_named_main_commit():
    env = {e["name"]: e["value"] for e in DOC["spec"]["template"]["spec"]["containers"][0]["env"]}
    blob = subprocess.run(["git", "-C", str(REPO), "show", f"{env['SCRIPTS_COMMIT']}:scripts/enrich_tiles_tessera.py"],
                          capture_output=True)
    if blob.returncode != 0:
        pytest.skip("pinned main commit not in this clone")
    assert hashlib.sha256(blob.stdout).hexdigest() == env["HELPER_SHA256"]
    assert hashlib.sha256((REPO / "scripts/enrich_tiles_tessera.py").read_bytes()).hexdigest() == env["HELPER_SHA256"]
