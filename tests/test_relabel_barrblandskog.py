"""IM-023 label shortcut: NMD 113 → class 28 equals the canonical label chain.

The equivalence is shown three ways: the pure rule against ``merge_all_2023``
and ``apply_override`` on random inputs, and on disk through the real
``build_labels.build_tile_label`` (raster + LPIS/SKS parquet) and the real
``distill_forest_labels.process_tile`` with a stub forward pass.
"""
from pathlib import Path
import json
import sys

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import build_labels as builder
import distill_forest_labels as distill
import relabel_barrblandskog as relabel
from imint.schema import unified_schema as schema

pytest.importorskip("torch")


# The NMD2023 LUT of the R4-era schema: 113 → lövskog (3).
OLD_LUT = schema._NMD2023_TO_UNIFIED.copy()
OLD_LUT[113] = 3


def test_trainer_keys_match_dataset():
    from imint.training.unified_dataset import _LABEL_SIDECAR_KEYS
    assert relabel.TRAINER_KEYS == _LABEL_SIDECAR_KEYS


def test_nmd_rule_equals_merge_all_2023(monkeypatch):
    rng = np.random.default_rng(0)
    codes = np.array(sorted(schema.NMD2023_TO_UNIFIED) + [0, 999], dtype=np.uint16)
    raw = rng.choice(codes, (64, 64)).astype(np.uint16)
    raw[:8] = 113
    sjv = rng.choice(np.array([0, 4, 9, 60, 9999], dtype=np.uint16), raw.shape)
    harvest = (rng.random(raw.shape) < 0.3).astype(np.uint8)
    new = schema.merge_all_2023(raw, sjv, harvest)
    monkeypatch.setattr(schema, "_NMD2023_TO_UNIFIED", OLD_LUT)
    old = schema.merge_all_2023(raw, sjv, harvest)
    np.testing.assert_array_equal(relabel.relabel_nmd(old, raw), new)
    assert (new == 28).any() and ((raw == 113) & (new == schema.HARVEST_CLASS)).any()
    assert relabel.relabel_nmd(old, raw).dtype == new.dtype


def test_distilled_rule_equals_override():
    rng = np.random.default_rng(1)
    raw = rng.choice(np.array([111, 112, 113, 114, 115, 123, 3, 61], dtype=np.uint16), (40, 40))
    new_nmd = schema.nmd2023_to_unified(raw)
    old_nmd = np.where(new_nmd == 28, 3, new_nmd).astype(np.uint8)
    head = rng.integers(0, 5, (32, 32))
    old_distilled, _, _ = distill.apply_override(old_nmd, head, 4, 32)
    new_distilled, _, _ = distill.apply_override(new_nmd, head, 4, 32)
    assert (old_distilled != old_nmd).any()
    np.testing.assert_array_equal(relabel.relabel_distilled(old_distilled, new_nmd), new_distilled)


# ── on disk: the real builder and distiller ──────────────────────────────────

SIZE = 16


@pytest.fixture
def chain(tmp_path, monkeypatch):
    # Builder reference handles are process-wide; this test has its own refs.
    monkeypatch.setattr(builder, "_lpis_sp", {})
    monkeypatch.setattr(builder, "_sks_utforda_sp", None)
    monkeypatch.setattr(builder, "_sks_anmalda_sp", None)
    tiles = tmp_path / "tiles"
    tiles.mkdir()
    tile = tiles / "T1.npz"
    np.savez(tile, spectral=np.ones((24, SIZE, SIZE), dtype=np.float32),
             easting=np.float64(500000), northing=np.float64(6500000),
             dates=np.array(["2021-09-10", "2022-05-10", "2022-06-10", "2022-08-10"]))
    # 32×32 NMD2023 raster at 10 m around the tile: forest, 113, agri, water.
    raster = np.add.outer(np.arange(32), np.arange(32)) % 10
    raster = np.array([111, 112, 113, 113, 114, 115, 123, 3, 411, 61], dtype=np.uint16)[raster]
    nmd = tmp_path / "nmd.tif"
    with rasterio.open(nmd, "w", driver="GTiff", width=32, height=32, count=1,
                       dtype="uint16", crs="EPSG:3006",
                       transform=from_origin(499840, 6500160, 10, 10)) as dst:
        dst.write(raster, 1)
    refs = tmp_path / "refs"
    refs.mkdir()
    # LPIS over the west half, harvest over the north half, both partial.
    layers = {
        "jordbruksskiften_2022_spatial.parquet": ({"grdkod_mar": [4], "area_ha": [.04]},
                                                  box(499900, 6499900, 500000, 6500100)),
        "utforda_avverkningar_spatial.parquet": ({"Avvdatum": ["2022-05-01"]},
                                                 box(499900, 6500000, 500100, 6500100)),
        "avverkningsanmalningar_spatial.parquet": ({}, box(600000, 6600000, 600080, 6600080)),
    }
    for filename, (columns, geom) in layers.items():
        gdf = gpd.GeoDataFrame(columns, geometry=[geom], crs="EPSG:3006")
        for key in ("minx", "miny", "maxx", "maxy"):
            gdf[f"_bbox_{key}"] = gdf.geometry.bounds[key]
        gdf.to_parquet(refs / filename)

    def build(out: Path) -> None:
        out.mkdir()
        result = builder.build_tile_label(str(tile), str(nmd), str(refs), str(refs),
                                          nmd_version=2023, label_out_dir=str(out))
        assert result["status"] == "ok", result

    def distil(label_dir: Path, out: Path, seed: int = 7) -> None:
        import torch

        class Forward:
            @staticmethod
            def run_inference(model, tile_path, device, **kwargs):
                store["feat"] = torch.zeros(1, 256, 3, 3)

        store = {"feat": None}
        monkeypatch.setattr(distill, "head_predict_dense", lambda feat, head: np.random.default_rng(
            seed).integers(0, 5, feat.shape[:2]))
        out.mkdir()
        status = distill.process_tile(str(tile), None, {"head_sha": "h1"}, None, SIZE - 4,
                                      str(label_dir), str(out), Forward, store)
        assert status["status"] == "ok", status

    with monkeypatch.context() as m:
        m.setattr(schema, "_NMD2023_TO_UNIFIED", OLD_LUT)
        build(tmp_path / "nmd_old")                        # /cephfs/nmd2023_labels
    distil(tmp_path / "nmd_old", tmp_path / "stored")      # /cephfs/distill/tessera_r2
    build(tmp_path / "canon_nmd")                          # current schema: 113 → 28
    distil(tmp_path / "nmd_old", tmp_path / "canon_old")
    distil(tmp_path / "canon_nmd", tmp_path / "canon_new")
    return tmp_path, distil


def args(root: Path) -> list[str]:
    return ["--nmd-in", str(root / "nmd_old"), "--distill-in", str(root / "stored")]


def verify_args(root: Path) -> list[str]:
    return ["verify", *args(root), "--canon-nmd", str(root / "canon_nmd"),
            "--canon-distill-old", str(root / "canon_old"),
            "--canon-distill-new", str(root / "canon_new"), "--report", str(root / "report.json")]


def test_inputs_exercise_every_path(chain):
    root, _ = chain
    with np.load(root / "canon_nmd" / "T1.npz") as z:
        label, raw = z["label"], z["nmd_label_raw"]
    assert (label == 28).any()
    assert ((raw == 113) & (label == schema.HARVEST_CLASS)).any()
    assert ((raw == 3) & (label == 11)).any()
    with np.load(root / "stored" / "T1.npz") as z:
        assert ((raw == 113) & np.isin(z["label"], (1, 2, 4))).any()


def test_verify_passes_against_the_canonical_chain(chain):
    root, _ = chain
    assert relabel.main(verify_args(root)) == 0
    report = json.loads((root / "report.json").read_text())
    assert report["identical"] and report["stored_reproduced"]
    assert report["head_rerun_differs_px"] == 0 and report["tiles"] == ["T1"]


def test_verify_fails_on_a_canonical_difference(chain):
    root, _ = chain
    path = root / "canon_nmd" / "T1.npz"
    payload = relabel.load(path)
    payload["label"] = payload["label"].copy()
    payload["label"][0, 0] = 9
    np.savez_compressed(path, **payload)
    assert relabel.main(verify_args(root)) == 1
    report = json.loads((root / "report.json").read_text())
    assert report["nmd_problems"] == ["T1 nmd: label differs in 1 cells"]


def test_verify_separates_head_reruns_from_the_shortcut(chain):
    root, distil = chain
    for directory in ("canon_old", "stored"):
        for path in (root / directory).glob("*.npz"):
            path.unlink()
        (root / directory).rmdir()
    distil(root / "nmd_old", root / "canon_old", seed=8)
    distil(root / "nmd_old", root / "stored", seed=9)
    assert relabel.main(verify_args(root)) == 1
    report = json.loads((root / "report.json").read_text())
    assert report["head_rerun_differs_px"] > 0 and not report["stored_reproduced"]


def test_run_writes_canonical_labels_and_resumes(chain):
    root, _ = chain
    out = ["--nmd-out", str(root / "b28_nmd"), "--distill-out", str(root / "b28_distill")]
    assert relabel.main(["run", *args(root), *out]) == 0
    for produced, canonical in (("b28_nmd", "canon_nmd"), ("b28_distill", "canon_new")):
        written, expected = relabel.load(root / produced / "T1.npz"), relabel.load(root / canonical / "T1.npz")
        assert str(written["relabel_rule"]) == relabel.RULE
        for key in expected:
            np.testing.assert_array_equal(written[key], expected[key], err_msg=key)
            assert written[key].dtype == expected[key].dtype, key
    first = json.loads((root / "b28_distill" / relabel.MANIFEST).read_text())
    assert first == json.loads((root / "b28_nmd" / relabel.MANIFEST).read_text())
    assert (first["written"], first["nmd_tiles"], first["distilled_tiles"]) == (1, 1, 1)
    assert first["class_28_pixels"] > 0 and first["nmd_changed"] == first["class_28_pixels"]
    assert first["distilled_changed"] == first["class_28_pixels"]
    stamp = (root / "b28_distill" / "T1.npz").stat().st_mtime_ns
    assert relabel.main(["run", *args(root), *out]) == 0
    second = json.loads((root / "b28_distill" / relabel.MANIFEST).read_text())
    assert (second["written"], second["skipped_current"]) == (0, 1)
    assert {k: second[k] for k in ("nmd_changed", "distilled_changed", "class_28_pixels")} == \
           {k: first[k] for k in ("nmd_changed", "distilled_changed", "class_28_pixels")}
    assert (root / "b28_distill" / "T1.npz").stat().st_mtime_ns == stamp
    assert not list(root.rglob("*.tmp.npz"))


def test_run_refuses_input_directories_and_orphans(chain):
    root, _ = chain
    with pytest.raises(ValueError, match="never an input"):
        relabel.run(root / "nmd_old", root / "stored", root / "stored", root / "x")
    with pytest.raises(ValueError, match="never an input"):
        relabel.run(root / "nmd_old", root / "stored", root / "x", root / "x")
    (root / "nmd_old" / "T1.npz").rename(root / "nmd_old" / "T0.npz")
    with pytest.raises(ValueError, match="without an NMD sidecar"):
        relabel.run(root / "nmd_old", root / "stored", root / "a", root / "b")


def test_sample_is_deterministic(tmp_path):
    for directory in ("nmd", "distill"):
        (tmp_path / directory).mkdir()
    for i in range(12):
        raw = np.full((4, 4), 113 if i % 3 else 111, dtype=np.uint16)
        np.savez(tmp_path / "nmd" / f"t{i}.npz", nmd_label_raw=raw)
        np.savez(tmp_path / "distill" / f"t{i}.npz", label=np.zeros((4, 4), np.uint8))
    picked = relabel.sample(tmp_path / "nmd", tmp_path / "distill", 3, 2)
    assert picked == relabel.sample(tmp_path / "nmd", tmp_path / "distill", 3, 2)
    assert len(picked) == 5 and sum(int(n[1:]) % 3 == 0 for n in picked) == 2
    with pytest.raises(ValueError, match="only"):
        relabel.sample(tmp_path / "nmd", tmp_path / "distill", 9, 2)
