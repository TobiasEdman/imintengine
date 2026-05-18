"""GCS ``_$folder$`` directory-placeholder handling in fetch_l1c_safe_by_name.

The Sentinel-2 public bucket marks an empty directory with a sibling
``<dir>_$folder$`` zero-byte object. Downloading it as a *file* leaves the
directory non-existent, so sen2cor's ``L2A_ProcessDataStrip`` crashes on
``os.listdir(.../QI_DATA)``. The fetcher must materialise these as real
directories instead.
"""
import urllib.request

from imint import fetch
from imint.fetch import _gcs_marker_dir


# ── _gcs_marker_dir predicate ─────────────────────────────────────────────

def test_marker_dir_strips_suffix():
    assert _gcs_marker_dir("DATASTRIP/DS_X/QI_DATA_$folder$") == \
        "DATASTRIP/DS_X/QI_DATA"


def test_marker_dir_ignores_real_files():
    assert _gcs_marker_dir("GRANULE/L1C_X/IMG_DATA/B02.jp2") is None
    assert _gcs_marker_dir("MTD_MSIL1C.xml") is None


def test_marker_dir_root_placeholder():
    assert _gcs_marker_dir("_$folder$") == ""


# ── fetch_l1c_safe_by_name end-to-end ─────────────────────────────────────

class _FakeResp:
    def __init__(self, data: bytes):
        self._data, self._pos = data, 0

    def read(self, n: int) -> bytes:
        chunk = self._data[self._pos:self._pos + n]
        self._pos += n
        return chunk

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_fetch_materialises_folder_marker_as_directory(tmp_path, monkeypatch):
    safe = "S2A_MSIL1C_20160714T103022_N0204_R108_T33WXR_20160714T103025.SAFE"
    prefix = f"tiles/33/W/XR/{safe}/"
    payload = b"hello world"

    monkeypatch.setattr(fetch, "_gcp_resolve_safe_name", lambda n: n)
    monkeypatch.setattr(fetch, "_gcp_list_safe_files", lambda n: [
        {"name": prefix + "DATASTRIP/DS_FOO/QI_DATA_$folder$", "size": 0},
        {"name": prefix + "MTD_MSIL1C.xml", "size": len(payload)},
    ])
    monkeypatch.setattr(
        urllib.request, "urlopen",
        lambda url, timeout=0: _FakeResp(payload),
    )

    safe_root = fetch.fetch_l1c_safe_by_name(safe, str(tmp_path))

    qi_data = safe_root / "DATASTRIP" / "DS_FOO" / "QI_DATA"
    assert qi_data.is_dir(), "QI_DATA placeholder must become a directory"
    assert not (safe_root / "DATASTRIP" / "DS_FOO" / "QI_DATA_$folder$").exists(), \
        "the _$folder$ marker must not be written as a file"
    mtd = safe_root / "MTD_MSIL1C.xml"
    assert mtd.read_bytes() == payload, "real files still download intact"
