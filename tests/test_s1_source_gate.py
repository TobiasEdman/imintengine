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
from imint.training.s1_enrichment import S1_ENRICH_VERSION, S1_SOURCE, s1_source_of
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
    assert f"dropped 2 with s1_enrich_v!={S1_ENRICH_VERSION}" in log
    assert f"dropped 2 with s1_source!='{S1_SOURCE}'" in log


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
    ("reject_v4_cdse", "s1_source"),
    ("reject_v4_no_source", "s1_source"),
    ("reject_v3_rtc", "s1_enrich_v"),
    ("reject_unstamped", "s1_enrich_v"),
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
