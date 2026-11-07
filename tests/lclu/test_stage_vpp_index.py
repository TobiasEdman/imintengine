"""stage() reads HR-VPP from the index it is given (task 0151)."""
import json

import pytest

from scripts import stage_lclu_demo


class Stop(Exception):
    pass


def test_stage_selects_hrvpp_from_the_named_index_or_the_shared_default(tmp_path, monkeypatch):
    seen = []

    def select(index, vpp_year, bounds):
        seen.append((index, vpp_year))
        raise Stop  # selection is the first read of the index; nothing is staged

    monkeypatch.setattr(stage_lclu_demo, 'select_vpp', select)
    scope = tmp_path / 'scope.json'
    scope.write_text(json.dumps({
        'version': 1, 'bounds_3006': [520640, 6864400, 525760, 6872300],
        'blocks': [{'version': '1.0', 'variant': 'vultr', 'year': year, 'lon': 15.45,
                    'lat': 61.95, 'grid_size': 1000, 'scales_size': 100}
                   for year in (2019, 2025)]}))
    for name, content in (('vpp_wekeo', '{"shared": {}}'), ('vpp_wekeo_fire', '{"fire": {}}')):
        (tmp_path / name).mkdir()
        (tmp_path / name / 'index.json').write_text(content)
    with pytest.raises(Stop):
        stage_lclu_demo.stage(scope, 2019, tmp_path, tmp_path / 'a',
                              vpp_index=tmp_path / 'vpp_wekeo_fire/index.json')
    with pytest.raises(Stop):
        stage_lclu_demo.stage(scope, 2025, tmp_path, tmp_path / 'b')
    # 2025 reads HR-VPP 2024 (lclu.VPP_YEAR_SUBSTITUTIONS) from the default cache.
    assert seen == [({'fire': {}}, 2019), ({'shared': {}}, 2024)]
