"""Check presentation against the pinned official QML and native class pixels."""
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image
import rasterio

from imint.inference.lclu import Grid, export_map, implementation_identity
from imint.inference.lclu_legend import MAP_COLORS, NMD_SWATCHES, legend


def test_official_swatches_exports_and_class_ids_agree(tmp_path):
    qml = Path(__file__).parents[2] / 'docs/verification/lclu-service/NMD2023bas_v2_1.qml'
    assert hashlib.sha256(qml.read_bytes()).hexdigest() == (
        '0154ee34d2ccc150246ea4901adaac3765b9cf02b1ea5c2d35ff07fc800a4322')
    official = {int(item.attrib['value']): item.attrib['color']
                for item in ET.parse(qml).iter('paletteEntry')}
    items = {item['id']: item for item in legend()}
    assert set(items) == set(range(28)) | {255}
    for key, (code, color) in NMD_SWATCHES.items():
        assert items[key]['color'] == color == official[code]
    # Cover all classes with wide stripes that survive nearest reprojection.
    pred = np.repeat(np.arange(28, dtype=np.uint8), 18)[:, None].repeat(504, axis=1)
    valid = np.ones_like(pred, dtype=bool)
    valid[:5, :5] = False
    export_map(pred, valid, Grid.from_lonlat(20.25, 67.85), tmp_path)
    with rasterio.open(tmp_path/'classification.tif') as src:
        np.testing.assert_array_equal(src.read(1), np.where(valid, pred, 255))
        for key, rgb in MAP_COLORS.items():
            assert src.colormap(1)[key] == (*rgb, 255)
    rgba = np.array(Image.open(tmp_path/'overlay.png'))
    assert set(map(tuple, np.unique(rgba.reshape(-1, 4), axis=0))) == {
        *((*color, 255) for color in MAP_COLORS.values()), (0, 0, 0, 0)}
    assert json.loads((tmp_path/'legend.json').read_text()) == legend()
    assert 'imint/inference/lclu_legend.py' in implementation_identity()


def test_aggregates_and_extensions_explain_their_limits():
    items = {item['id']: item for item in legend()}
    assert 'barrblandskog' in items[3]['name']
    for key in (3, 5, 7, 8, 9, 10, 24, 25, 26):
        assert 'sammanslag' in (items[key]['note'] + items[key]['name']).lower()
    for key in range(11, 23):
        assert 'ingen separat NMD-klass' in items[key]['note']
