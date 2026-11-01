"""Range-read original TESSERA tiles: bit for bit the tile path, cached bands, pinned sizes."""
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from functools import partial
import threading
from types import SimpleNamespace

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

pytest.importorskip("geotessera")
from geotessera.registry import tile_to_embedding_paths

from imint.inference.lclu import Grid, InputUnavailable
from scripts import prepare_lclu as prep


class RangeHandler(SimpleHTTPRequestHandler):
    """Static files with single byte ranges, as object stores serve them."""
    requests = []

    def log_message(self, *args):
        pass

    def send_head(self):
        RangeHandler.requests.append((self.command, self.path, self.headers.get("Range")))
        spec = self.headers.get("Range")
        if self.command != "GET" or not spec:
            return super().send_head()
        path = self.translate_path(self.path)
        data = open(path, "rb").read()
        start, _, end = spec.removeprefix("bytes=").partition("-")
        start, end = int(start), min(int(end) if end else len(data) - 1, len(data) - 1)
        body = data[start:end + 1]
        self.send_response(206)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Range", f"bytes {start}-{end}/{len(data)}")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        import io
        return io.BytesIO(body)


@pytest.fixture
def served(tmp_path):
    root = tmp_path / "public"
    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(RangeHandler, directory=str(root)))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    root.mkdir()
    thread.start()
    RangeHandler.requests = []
    yield root, f"http://127.0.0.1:{server.server_address[1]}/npy/v1"
    server.shutdown()


def two_blocks(tmp_path, root, grid):
    """Two overlapping blocks off the output grid, as tests/lclu/test_cog.py builds them."""
    arrays, refs, sizes = {}, {}, {}
    for i, lon in enumerate((17.95, 18.05)):
        tx = from_origin(grid.east - 3000 + 2980 * i + .3, grid.north + 3000.7, 10, 10)
        q = np.full((600, 302, 128), 20 + i, np.int8)
        q[100:120, 100:120] = 0
        q[:, -1, :] = 0
        q[300:, :, 5] = -7 - i
        scales = np.linspace(.001, .02, 600 * 302, dtype=np.float32).reshape(600, 302)
        q_rel, s_rel = tile_to_embedding_paths(lon, 59.3, 2025)
        for rel, values in ((q_rel, q), (s_rel, scales)):
            path = root / "npy" / "v1" / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, values)
        sizes[lon] = tuple((root / "npy" / "v1" / rel).stat().st_size for rel in (q_rel, s_rel))
        ref = tmp_path / f"{i}-ref.tif"
        with rasterio.open(ref, "w", driver="GTiff", width=302, height=600, count=1,
                           dtype="uint8", crs="EPSG:3006", transform=tx) as dst:
            dst.write(np.ones((600, 302), np.uint8), 1)
        arrays[lon] = (q.astype(np.float32) * scales[:, :, None], "EPSG:3006", tx)
        refs[lon] = ref
    registry = SimpleNamespace(
        load_blocks_for_region=lambda **kw: [(2025, lon, 59.3) for lon in arrays],
        fetch_landmask=lambda **kw: refs[kw["lon"]],
        get_tile_file_size=lambda year, lon, lat: sizes[lon][0],
        get_scales_file_size=lambda year, lon, lat: sizes[lon][1])
    return SimpleNamespace(registry=registry, fetch_embedding=lambda **kw: arrays[kw["lon"]]), refs


def test_range_reads_match_the_tile_path_and_reuse_cached_bands(tmp_path, served, monkeypatch):
    root, url = served
    grid = Grid.from_lonlat(18, 59.3)
    gt, refs = two_blocks(tmp_path, root, grid)
    monkeypatch.setattr(prep, "embedding_files", lambda *_: {p: p.stat().st_size for p in refs.values()})
    expected, expected_valid, _ = prep.assemble_embedding(gt, grid, 2025, max_blocks=4, max_bytes=1)

    result, valid, provenance = prep.assemble_remote_embedding(
        gt, grid, 2025, root=url, cache=tmp_path / "bands", max_blocks=4)
    np.testing.assert_array_equal(result, expected)
    np.testing.assert_array_equal(valid, expected_valid)
    assert valid.sum() > 200_000 and not valid.all()
    ranged = [r for r in RangeHandler.requests if r[2]]
    assert ranged and all(r[0] == "GET" for r in ranged)
    # One small header read per file, then one request per band of rows.
    headers = [r for r in ranged if r[2] == "bytes=0-4095"]
    assert len(headers) == 4
    assert provenance["bands"]["fetched"] == len(ranged) - len(headers)
    assert provenance["bands"]["cached"] == 0
    assert provenance["storage"] == "remote-npy-range" and provenance["blocks"] == 2

    RangeHandler.requests = []
    again, valid_again, second = prep.assemble_remote_embedding(
        gt, grid, 2025, root=url, cache=tmp_path / "bands", max_blocks=4)
    assert RangeHandler.requests == []          # every band came from disk
    assert second["bands"] == {"fetched": 0, "cached": provenance["bands"]["fetched"]}
    np.testing.assert_array_equal(again, expected)
    np.testing.assert_array_equal(valid_again, expected_valid)


def test_a_tile_whose_remote_size_changed_is_refused(tmp_path, served):
    root, url = served
    grid = Grid.from_lonlat(18, 59.3)
    gt, _ = two_blocks(tmp_path, root, grid)
    gt.registry.get_tile_file_size = lambda year, lon, lat: 123
    with pytest.raises(ValueError, match="pinned registry"):
        prep.assemble_remote_embedding(gt, grid, 2025, root=url, cache=tmp_path / "bands",
                                       max_blocks=4)
    assert not list((tmp_path / "bands").rglob("*.npy"))


def test_block_budget_and_exact_year_still_apply(tmp_path, served):
    root, url = served
    grid = Grid.from_lonlat(18, 59.3)
    gt, _ = two_blocks(tmp_path, root, grid)
    with pytest.raises(InputUnavailable, match="block budget"):
        prep.assemble_remote_embedding(gt, grid, 2025, root=url, cache=tmp_path / "b",
                                       max_blocks=1)
    gt.registry.load_blocks_for_region = lambda **kw: [(2024, 17.95, 59.3)]
    with pytest.raises(InputUnavailable, match="exact-year"):
        prep.assemble_remote_embedding(gt, grid, 2025, root=url, cache=tmp_path / "b",
                                       max_blocks=4)


def test_prepare_embedding_routes_the_published_root_to_range_reads(tmp_path, monkeypatch):
    import geotessera
    from geotessera.registry import TESSERA_NPY_MIRROR_URL
    from scripts.cache_predict import file_identity

    registry = tmp_path / "registry"
    registry.mkdir()
    for name in ("manifest.parquet", "landmasks.parquet"):
        (registry / name).write_bytes(b"pinned")
    config = {"registry": {**file_identity(registry / "manifest.parquet"), "path": "registry/manifest.parquet"},
              "landmasks": {**file_identity(registry / "landmasks.parquet"), "path": "registry/landmasks.parquet"},
              "max_blocks": 4, "embedding_cache": "cache",
              "remote_tiles": f"{TESSERA_NPY_MIRROR_URL}/v1"}
    monkeypatch.setattr(geotessera, "GeoTessera", lambda **kw: SimpleNamespace(**kw))
    calls = []
    def remote(gt, grid, year, **kw):
        calls.append(kw)
        return np.zeros((128, 512, 512), np.float16), np.zeros((512, 512), bool), {"storage": "remote-npy-range"}
    monkeypatch.setattr(prep, "assemble_remote_embedding", remote)
    monkeypatch.setattr(prep, "assemble_embedding", lambda *a, **kw: pytest.fail("downloaded whole tiles"))
    grid = Grid.from_lonlat(18, 59.3)
    _, _, provenance = prep.prepare_embedding(config, tmp_path, grid, 2025)
    assert calls[0]["root"] == f"{TESSERA_NPY_MIRROR_URL}/v1"
    assert calls[0]["cache"].name == "geotessera-0.10.2-v1-vultr-bands"
    assert provenance["registry"]["sha256"] == config["registry"]["sha256"]

    with pytest.raises(ValueError, match="published v1/vultr"):
        prep.prepare_embedding({**config, "remote_tiles": "https://example.org/npy/v1"},
                               tmp_path, grid, 2025)
    whole = {k: v for k, v in config.items() if k != "remote_tiles"}
    with pytest.raises(KeyError, match="max_download_bytes"):
        prep.prepare_embedding(whole, tmp_path, grid, 2025)
