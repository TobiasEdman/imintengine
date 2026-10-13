"""Regressions for optional grazing metadata in actual tabbed exports."""
import json
from pathlib import Path

from PIL import Image
import pytest

from imint.exporters import html_report


@pytest.mark.parametrize('metadata, shape', [
    ('bands', (37, 59)),
    ('absent', (573, 324)),
    ('grazing', (23, 41)),
])
def test_grazing_report_shape_and_images(tmp_path, monkeypatch, metadata, shape):
    fire, marine, grazing = (tmp_path / name for name in ('fire', 'marine', 'grazing'))
    for directory in (fire, marine, grazing):
        directory.mkdir()
    image = grazing / 'rgb.png'
    Image.new('RGB', (59, 37), 'green').save(image)
    bands = grazing / 'bands'
    bands.mkdir()
    # Conflicting dated metadata must not override the documented undated input.
    (bands / '2026-10-12_bands_meta.json').write_text(json.dumps({'shape': [99, 101]}))
    if metadata != 'absent':
        (bands / 'bands_meta.json').write_text(json.dumps({'shape': [37, 59]}))
    if metadata == 'grazing':
        (grazing / 'grazing_meta.json').write_text(json.dumps({'shape': [23, 41]}))
    # This export regression checks HTML/data output; browser tests load real libraries.
    monkeypatch.setattr(html_report, '_fetch_lib', lambda *args, **kwargs: '')
    output = tmp_path / 'report.html'
    result = html_report.save_tabbed_report(
        str(fire), str(marine), str(output),
        grazing_dir=str(grazing), grazing_date='2026-10-12',
    )
    assert Path(result) == output
    html = output.read_text()
    assert f'initMaps(GRAZING_VIEWERS, GRAZING_IMAGES, {shape[0]}, {shape[1]}, false, LPIS_GEOJSON)' in html
    assert 'showcase/grazing/rgb.png' in html
    assert 'id="g-rgb"' in html
    assert (tmp_path / 'showcase/grazing/rgb.png').read_bytes() == image.read_bytes()
