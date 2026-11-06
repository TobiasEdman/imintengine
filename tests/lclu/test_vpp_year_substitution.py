"""2025 maps use HR-VPP 2024 through one explicit table, and say so (Tobias 2026-11-03).

Every other year keeps its own HR-VPP year; any other pairing is rejected.
"""
import copy
import json

import numpy as np
import pytest

from imint.inference.lclu import (
    HISTORICAL_NOTE, VPP_YEAR_SUBSTITUTIONS, YEARS, BundleCatalog, EMBEDDING, LcluService,
    public_provenance, validate_bundle, vpp_substitutions, vpp_year_for,
)
from scripts.cache_predict import file_identity
from tests.lclu.test_service import SyntheticModel, bundle  # noqa: F401  (fixture)


def test_the_table_substitutes_only_2025_with_2024():
    assert {year: vpp_year_for(year) for year in YEARS if vpp_year_for(year) != year} == {2025: 2024}
    assert all(vpp_year_for(year) == year for year in YEARS if year != 2025)
    # The page's "~7 %" is the control's median for a one-year shift only.
    assert all(year in YEARS and vpp == year - 1 for year, vpp in VPP_YEAR_SUBSTITUTIONS.items())


def test_catalogue_notice_and_historical_note_name_the_substitution():
    assert vpp_substitutions() == [{
        'year': 2025, 'vpp_year': 2024, 'label': 'växtfenologi från 2024',
        'note': 'Uppskattningsvis ~7 % av pixlarna kan avvika jämfört med en karta med årets växtfenologi.'}]
    assert 'utom 2025: HR-VPP 2025 är inte publicerad, så 2025 använder HR-VPP 2024' in HISTORICAL_NOTE


def _write(path, arrays, year, vpp_year):
    np.savez_compressed(path, **{**arrays, 'year': np.int32(year), 'tessera_year': np.int32(year),
                                 'vpp_year': np.int32(vpp_year)})
    return file_identity(path)


@pytest.mark.parametrize('year,vpp_year,accepted', [
    (2025, 2024, True),    # the tabled substitution
    (2024, 2024, True),
    (2025, 2025, False),   # the table names 2024 for 2025, nothing else
    (2025, 2023, False),
    (2024, 2023, False),   # no other year falls back
    (2023, 2022, False),
    (2024, 2025, False),
])
def test_bundle_accepts_only_the_tabled_vpp_year(bundle, year, vpp_year, accepted):
    grid, arrays, _, path = bundle
    record = _write(path, arrays, year, vpp_year)
    if accepted:
        validate_bundle(path, record, grid, year)
    else:
        with pytest.raises(ValueError, match='HR-VPP year'):
            validate_bundle(path, record, grid, year)


def _entry(entry, year, vpp_year):
    changed = copy.deepcopy(entry)
    changed['year'] = year
    changed['auxiliary_provenance']['vpp']['year'] = vpp_year
    return changed


@pytest.mark.parametrize('year,vpp_year,substituted', [(2025, 2024, True), (2024, 2024, False),
                                                        (2017, 2017, False)])
def test_public_provenance_records_vpp_year_and_marker(bundle, year, vpp_year, substituted):
    _, _, entry, _ = bundle
    public = public_provenance(_entry(entry, year, vpp_year))
    assert public['vpp_year'] == vpp_year and public['vpp_year_substituted'] is substituted
    assert public['auxiliary_provenance']['vpp']['year'] == vpp_year


@pytest.mark.parametrize('year,vpp_year', [(2025, 2025), (2025, 2023), (2024, 2023), (2023, 2022)])
def test_public_provenance_rejects_any_other_pairing(bundle, year, vpp_year):
    _, _, entry, _ = bundle
    with pytest.raises(ValueError, match='exact HR-VPP year'):
        public_provenance(_entry(entry, year, vpp_year))


def test_a_2025_map_records_hrvpp_2024_in_provenance_and_result(bundle, tmp_path):
    grid, arrays, entry, path = bundle
    entries = []
    for year in (2024, 2025):
        year_path = tmp_path / f'public-{year}.npz'
        record = _write(year_path, arrays, year, vpp_year_for(year))
        entries.append({**_entry(entry, year, vpp_year_for(year)), 'input': record})
    app = LcluService(SyntheticModel(), BundleCatalog(entries, tmp_path), tmp_path / 'outputs')
    assert app.sources.years(grid) == [2024, 2025]
    for year, substituted in ((2024, False), (2025, True)):
        result = app.run(grid, year, lambda _: None)
        directory = app.output_root / result['cache_key']
        saved = json.loads((directory / 'result.json').read_text())
        provenance = json.loads((directory / 'provenance.json').read_text())
        for record in (result, saved, provenance):
            assert record['year'] == year
            assert record['vpp_year'] == 2024 and record['vpp_year_substituted'] is substituted
        assert provenance['auxiliary_provenance']['vpp']['year'] == 2024
        assert 'HR-VPP 2024' in provenance['historical_note']
        cached = app.run(grid, year, lambda _: None)
        assert cached['cache_hit'] and cached['vpp_year_substituted'] is substituted


@pytest.mark.parametrize('year', [2024, 2025])
def test_preparation_reads_the_tabled_hrvpp_year_and_stamps_it(tmp_path, monkeypatch, year):
    from imint.inference.lclu import AUX_NAMES, Grid
    from imint.training import skg_open
    from scripts import prepare_lclu as prep

    grid = Grid.from_lonlat(17.64, 59.86)
    metadata = tmp_path / 'MANIFEST.json'
    metadata.write_text('{}')
    config = tmp_path / 'sources.json'
    config.write_text(json.dumps({'version': 1, 'embedding': EMBEDDING,
                                  'skg_manifest': file_identity(metadata), 'identity_cache': 'identities'}))
    read = []

    def prepare_aux(config, base, grid, vpp_year):
        read.append(vpp_year)
        aux = {name: np.ones((512, 512), np.float32) for name in AUX_NAMES}
        return aux, {'forestry': {'counties': [1]},
                     'vpp': {'year': vpp_year, 'season': 1, 'files': [],
                             'index': {'sha256': 'a' * 64, 'size': 1}}}
    monkeypatch.setattr(prep, 'prepare_aux', prepare_aux)
    monkeypatch.setattr(prep, 'prepare_embedding', lambda *a: (
        np.full((128, 512, 512), .125, np.float16), np.ones((512, 512), bool), {}))
    monkeypatch.setattr(prep, 'verify_county_masks', lambda *a: {})
    monkeypatch.setattr(skg_open, 'county_cells', lambda *a: np.ones((512, 512), bool))
    out = tmp_path / 'prepared'
    prep.prepare(config, grid, year, out)
    assert read == [2024]
    with np.load(out / 'input.npz', allow_pickle=False) as saved:
        assert (int(saved['year']), int(saved['tessera_year']), int(saved['vpp_year'])) == (year, year, 2024)
    entry = json.loads((out / 'entry.json').read_text())
    assert entry['year'] == year and entry['auxiliary_provenance']['vpp']['year'] == 2024
    validate_bundle(out / 'input.npz', {**entry['input'], 'path': 'input.npz'}, grid, year)
    vpp_only = {**entry, 'auxiliary_provenance': {'vpp': entry['auxiliary_provenance']['vpp']}}
    assert public_provenance(vpp_only)['vpp_year_substituted'] is (year == 2025)
