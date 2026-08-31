"""The renamed live training monitor (ex campaign_dashboard.py, 2026-08-31).

Commit 33d44dd landed the tool with zero tests while deleting 1063 lines
under the same filename; these cover the pure logic — log parsing, cache
TTL, ensemble-state degradation — without kubectl or a cluster. Written
against the REAL signatures (Cache.get takes only a producer; EPOCH_RE
requires the full trainer line incl. worst/lr/seconds).
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location(
        "_ttm", str(ROOT / "scripts" / "tessera_train_monitor.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_epoch_re_parses_trainer_line() -> None:
    ttm = _load()
    line = ("  Epoch  12/30 | loss=0.6152 | val_mIoU=0.4752 "
            "| worst=0.2515 (bete) | lr=7.02e-05 | 3144s")
    m = ttm.EPOCH_RE.search(line)
    assert m, "trainer epoch line must parse"
    assert (int(m.group(1)), int(m.group(2))) == (12, 30)
    assert float(m.group(5)) == 0.4752
    assert m.group(7) == "bete"


def test_epoch_re_parses_frac_variant() -> None:
    """The optional L_frac segment (fraction-head runs) must not break the
    match — and its absence above must not either (group(4) is None)."""
    ttm = _load()
    line = ("Epoch 3/15 | loss=0.9 | L_frac=0.1234 | val_mIoU=0.4000 "
            "| worst=0.2000 (gran) | lr=1e-04 | 100s")
    m = ttm.EPOCH_RE.search(line)
    assert m and m.group(4) == "0.1234"


def test_cache_expires_by_ttl(monkeypatch) -> None:
    ttm = _load()
    clock = {"t": 1000.0}
    monkeypatch.setattr(ttm.time, "monotonic", lambda: clock["t"])
    c = ttm.Cache(ttl=5.0)
    calls: list[int] = []

    def produce() -> dict:
        calls.append(1)
        return {"v": len(calls)}

    assert c.get(produce) == {"v": 1}
    assert c.get(produce) == {"v": 1}      # within TTL → cached
    clock["t"] += 6.0
    assert c.get(produce) == {"v": 2}      # past TTL → re-produced
    assert len(calls) == 2


def test_ensemble_status_degrades_without_files(monkeypatch, tmp_path) -> None:
    """No results/baseline/member JSONs → labelled degraded values (member
    fallbacks, '?/?/?' config, None metrics), never an exception. The
    monitor's whole contract is degrade-don't-crash."""
    ttm = _load()
    monkeypatch.setattr(ttm, "REPO", tmp_path)
    monkeypatch.setattr(ttm, "RESULTS_JSON", tmp_path / "absent.json")
    monkeypatch.setattr(ttm, "BASELINE_JSON", tmp_path / "absent2.json")
    s = ttm._ensemble_status()
    assert set(s) == {"members", "gate", "reported", "baseline_a"}
    assert s["gate"] is None
    assert s["reported"]["config"] == "?/?/?"
    assert s["reported"]["holdout_oa"] is None
    fallbacks = {name: fb for name, (_, fb) in ttm.MEMBER_VALIDATION.items()}
    assert {m["name"]: m["oa"] for m in s["members"]} == fallbacks
