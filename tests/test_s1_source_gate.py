"""The SAR dataset gate requires the current version AND the RTC source (task 0113).

A tile's S1 composite is used only when ``s1_enrich_v == S1_ENRICH_VERSION``
and ``s1_source == S1_SOURCE`` ("pc-rtc-gamma0"). The CDSE GRD sigma0 backend
(retired: ellipsoid-height tie points) and tiles without a source stamp are
refused, at index time (dropped, logged) and in ``__getitem__`` (raised).
"""
from __future__ import annotations

import logging
import sys

import numpy as np
import pytest

from imint.training.errors import TilePrerequisiteError
from imint.training.s1_enrichment import (
    S1_ENRICH_VERSION,
    S1_SOURCE,
    s1_source_of,
    s1_unusable_reason,
    s1_usable,
)
from imint.training.unified_dataset import UnifiedDataset
from tests.test_multimodel_wiring import H, _write_tile

_MISSING = object()
# name -> (s1_enrich_v, s1_source); _MISSING drops the key.
_CASES = {
    "accept_v4_rtc": (S1_ENRICH_VERSION, "pc-rtc-gamma0"),
    "reject_v4_cdse": (S1_ENRICH_VERSION, "cdse-grd-sigma0"),
    "reject_v4_no_source": (S1_ENRICH_VERSION, _MISSING),
    "reject_v3_rtc": (3, "pc-rtc-gamma0"),
    "reject_unstamped": (_MISSING, _MISSING),
}


def _tile(path, version, source):
    _write_tile(path, with_s1=True)
    data = dict(np.load(path, allow_pickle=True))
    for key, value in (("s1_enrich_v", version), ("s1_source", source)):
        if value is _MISSING:
            data.pop(key, None)
        elif key == "s1_enrich_v":
            data[key] = np.int32(value)
        else:
            data[key] = np.bytes_(value)
    np.savez_compressed(path, **data)
    return path


def test_shared_source_constant_is_pinned():
    """7854/7854 v4 tiles on the PVC carry this stamp (2026-10-22): changing
    the value would silently drop every SAR tile."""
    assert S1_SOURCE == "pc-rtc-gamma0"


def test_producer_and_consumer_share_the_source_constant():
    import imint.training.unified_dataset as ud
    import scripts.enrich_tiles_s1 as enricher

    assert ud.S1_SOURCE is S1_SOURCE
    assert enricher.S1_SOURCE is S1_SOURCE
    assert enricher._BACKENDS["pc-rtc"]["source"] is S1_SOURCE
    assert set(enricher._BACKENDS) == {"pc-rtc"}


@pytest.mark.parametrize("value, expected", [
    (np.bytes_("pc-rtc-gamma0"), "pc-rtc-gamma0"),
    (np.array(b"cdse-grd-sigma0"), "cdse-grd-sigma0"),
    (np.array("pc-rtc-gamma0"), "pc-rtc-gamma0"),
    ("pc-rtc-gamma0", "pc-rtc-gamma0"),
    (None, None),
])
def test_s1_source_of_decodes_stored_forms(value, expected):
    assert s1_source_of(value) == expected


# The shared rule every SAR gate applies (dataset, pinned set, plot features,
# LUCAS split, k8s preflights).
@pytest.mark.parametrize("case, reason", [
    ("accept_v4_rtc", None),
    ("reject_v4_cdse", "s1_source='cdse-grd-sigma0'"),
    ("reject_v4_no_source", "s1_source=None"),
    ("reject_v3_rtc", "s1_enrich_v=3"),
    ("reject_unstamped", "s1_enrich_v=0"),
])
def test_s1_usable_rule(tmp_path, case, reason):
    version, source = _CASES[case]
    path = _tile(tmp_path / "t.npz", version, source)
    with np.load(path, allow_pickle=False) as z:     # lazy NpzFile, as the gates read it
        assert s1_unusable_reason(z) == reason
        assert s1_usable(z) is (reason is None)
    assert s1_usable(dict(np.load(path))) is (reason is None)


@pytest.mark.parametrize("data, reason", [
    ({"s1_enrich_v": 4, "s1_source": "pc-rtc-gamma0"}, None),
    ({"s1_enrich_v": np.array([4, 4]), "s1_source": "pc-rtc-gamma0"},
     "s1_enrich_v=<not a scalar>"),
    ({"s1_enrich_v": np.int32(5), "s1_source": "pc-rtc-gamma0"}, "s1_enrich_v=5"),
])
def test_s1_usable_rule_edge_values(data, reason):
    assert s1_unusable_reason(data) == reason


def test_pinned_set_and_lucas_split_use_the_rule(tmp_path):
    from scripts import build_lucas_crop_split as lucas
    from scripts import build_pinned_plot_set as bps

    assert bps.VERSION_REQUIREMENTS["s1_vv_vh"] is s1_usable
    for case, (version, source) in _CASES.items():
        path = _tile(tmp_path / f"{case}.npz", version, source)
        want = case.startswith("accept")
        assert bps.npz_version_ok(path, ("s1_vv_vh",)) is want, case
        assert lucas._qualify_npz_bytes(path.read_bytes(), ("s1_vv_vh",)) is want, case


def test_k8s_sar_preflights_call_the_rule():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "k8s"
    jobs = [root / "train-croma-job.yaml", root / "train-terramind-job.yaml",
            *sorted((root / "ladder").glob("ladder-r*-croma-job.yaml")),
            *sorted((root / "ladder").glob("ladder-r*-terramind-job.yaml"))]
    assert len(jobs) == 10
    for job in jobs:
        text = job.read_text()
        assert "from imint.training.s1_enrichment import s1_usable" in text, job.name
        assert "if s1_usable(z):" in text, job.name
        assert 'int(z["s1_enrich_v"].item()) == 4' not in text, job.name


def test_index_gate_keeps_only_v4_rtc(tmp_path, caplog):
    d = tmp_path / "tiles"
    d.mkdir()
    names = [f"{name}.npz" for name in _CASES]
    for name, (version, source) in _CASES.items():
        _tile(d / f"{name}.npz", version, source)
    (d / "split_train.txt").write_text("\n".join(names) + "\n")

    with caplog.at_level(logging.INFO, logger="imint.training.unified_dataset"):
        ds = UnifiedDataset(lulc_dir=d, split="train", patch_size=H, augment_override=False,
                            model_keys=("croma_base",), backbone_family="croma")
    assert [e["name"] for e in ds._entries] == ["accept_v4_rtc.npz"]
    log = caplog.text
    reasons = {"s1_enrich_v=0": 1, "s1_enrich_v=3": 1,
               "s1_source='cdse-grd-sigma0'": 1, "s1_source=None": 1}
    assert f"dropped 4 with unusable S1 {reasons}" in log


def test_index_gate_ignores_source_for_non_sar_models(tmp_path):
    d = tmp_path / "tiles"
    d.mkdir()
    _tile(d / "cdse.npz", S1_ENRICH_VERSION, "cdse-grd-sigma0")
    (d / "split_train.txt").write_text("cdse.npz\n")
    ds = UnifiedDataset(lulc_dir=d, split="train", patch_size=H, augment_override=False,
                        model_keys=("clay_v1_5",), backbone_family="clay")
    assert len(ds) == 1


def _read(tmp_path, version, source):
    data = dict(np.load(_tile(tmp_path / "t.npz", version, source), allow_pickle=True))
    ds = UnifiedDataset.__new__(UnifiedDataset)
    ds.model_keys = ("terramind_v1_base",)
    return ds._build_model_specific_tensors(data, source="lulc")


def test_read_accepts_v4_rtc(tmp_path):
    out = _read(tmp_path, *_CASES["accept_v4_rtc"])
    assert out["s1_vv_vh"].shape == (2, H, H)


@pytest.mark.parametrize("case, match", [
    ("reject_v4_cdse", "found s1_source='cdse-grd-sigma0'"),
    ("reject_v4_no_source", "found s1_source=None"),
    ("reject_v3_rtc", "found s1_enrich_v=3"),
    ("reject_unstamped", "found s1_enrich_v=0"),
])
def test_read_refuses_other_tiles(tmp_path, case, match):
    with pytest.raises(TilePrerequisiteError, match=match):
        _read(tmp_path, *_CASES[case])


@pytest.mark.parametrize("backend", ["cdse-stac", "cascade"])
def test_enricher_refuses_retired_backends(tmp_path, monkeypatch, capsys, backend):
    import scripts.enrich_tiles_s1 as enricher

    path = _tile(tmp_path / "t.npz", 3, "pc-rtc-gamma0")
    with pytest.raises(ValueError, match="task 0113"):
        enricher.enrich_one_tile(str(path), skip_existing=False, backend=backend)
    monkeypatch.setattr(sys, "argv", ["enrich_tiles_s1.py", "--data-dir", str(tmp_path),
                                      "--s1-backend", backend])
    with pytest.raises(SystemExit) as exc:
        enricher.main()
    assert exc.value.code == 2
    assert "task 0113" in capsys.readouterr().err
