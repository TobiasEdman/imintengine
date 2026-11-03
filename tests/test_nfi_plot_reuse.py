"""Geographic reuse must not be inferred from recycled NFI identifiers."""
import numpy as np
import pandas as pd
import pytest

from scripts.audit_nfi_plot_reuse import compare_locations, coordinates
from imint.eval.fieldtruth import NFI_KEY


def frame(rows):
    return pd.DataFrame(rows, columns=NFI_KEY + ["Easting", "Northing"])


def test_recycled_id_is_not_same_location_and_changed_id_can_be():
    held = frame([(1, 1, 2024, 0, 0), (2, 1, 2024, 1000, 0)])
    teacher = frame([(1, 1, 2020, 10000, 0), (9, 9, 2020, 0, 0),
                     (8, 8, 2024, 1000, 0)])
    r = compare_locations(held, teacher)
    assert r['cross_year_shared_id_pairs'] == 1
    assert r['shared_id_min_distance_m'] == 10000
    assert r['cross_year_proximity_m']['0'] == {'pairs': 1, 'holdout_observations': 1}
    assert r['nearest_cross_year_distance_quantiles_m']['max'] == 1000


def test_distance_threshold_counts_pairs_and_observations_separately():
    held = frame([(1, 1, 2024, 0, 0)])
    teacher = frame([(2, 1, 2020, 7, 0), (3, 1, 2021, 0, 14), (4, 1, 2021, 0, 15)])
    r = compare_locations(held, teacher)
    assert r['cross_year_proximity_m']['7'] == {'pairs': 1, 'holdout_observations': 1}
    assert r['cross_year_proximity_m']['14'] == {'pairs': 2, 'holdout_observations': 1}
    assert r['cross_year_shared_id_pairs'] == 0


def test_same_year_only_is_reported_without_nonfinite_json():
    import json
    held = frame([(1, 1, 2024, 0, 0)])
    teacher = frame([(2, 1, 2024, 0, 0)])
    r = compare_locations(held, teacher)
    assert r['holdout_with_cross_year_reference'] == 0
    assert r['nearest_cross_year_distance_quantiles_m'] is None
    json.dumps(r, allow_nan=False)


@pytest.mark.parametrize('bad', ['duplicate', 'nonfinite', 'same_year_overlap', 'empty'])
def test_invalid_comparisons_fail_closed(bad):
    held = frame([(1, 1, 2024, 0, 0)])
    teacher = frame([(2, 1, 2020, 0, 0)])
    if bad == 'duplicate': teacher = pd.concat([teacher, teacher])
    if bad == 'nonfinite': teacher = teacher.assign(Easting=np.inf)
    if bad == 'same_year_overlap': teacher = held.copy()
    if bad == 'empty': teacher = teacher.iloc[:0]
    with pytest.raises(ValueError): compare_locations(held, teacher)


def test_coordinate_join_collapses_tile_copies_but_rejects_conflicts_and_missing():
    a = frame([(1, 1, 2020, 0, 0)])
    assert len(coordinates(a, pd.concat([a, a]))) == 1
    b = a.copy(); b.loc[0, 'Easting'] = 1
    with pytest.raises(ValueError, match='conflicting'): coordinates(a, pd.concat([a, b]))
    b.loc[0, 'Year'] = 2021
    with pytest.raises(ValueError, match='null'): coordinates(a, b)


@pytest.fixture
def frozen_audit(tmp_path):
    import json
    from tests.nfi_freeze_fixtures import complete_manifest
    from imint.eval.fieldtruth import NFI_MODELS, sha256_file
    held = frame([(1, 1, 2024, 0, 0)]).assign(tile_name='private_held')
    source = frame([(1, 1, 2020, 10000, 0), (2, 1, 2020, 0, 0)])
    source['tile_name'] = ['private_train', 'private_test']
    manifest = complete_manifest({'schema': 'nfi-fieldtruth-freeze-v1', 'inputs': [], 'recorded_training_plot_years': 1,
                                  'campaign': {'training_tiles': 0}})
    for name, table in [('holdout', held), ('training', source[NFI_KEY])]:
        p = tmp_path / (name + '.parquet'); table.to_parquet(p, index=False)
        manifest[name] = {'file': p.name, 'sha256': sha256_file(p), 'observations': len(table)}
    source_path = tmp_path / 'nfi_index_unified_v2_512.parquet'
    source.to_parquet(source_path, index=False)
    inputs = [source_path]
    for model in NFI_MODELS:
        fp = tmp_path / f'{model}_r2_plot_features.parquet'
        source.drop(columns='Year').assign(f000=1.).to_parquet(fp, index=False)
        sp = tmp_path / f'{model}_r2_split.json'
        sp.write_text(json.dumps({'train_tiles': ['private_train'], 'test_tiles': ['private_test'],
                                  'n_train_plots': 1, 'n_test_plots': 1}))
        inputs.extend([fp, sp])
    manifest['inputs'] = [{'path': str(p), 'sha256': sha256_file(p), 'bytes': p.stat().st_size} for p in inputs]
    mp = tmp_path / 'manifest.json'; mp.write_text(json.dumps(manifest))
    return mp, sha256_file(mp)


def test_cli_verifies_frozen_files_and_emits_only_aggregates(frozen_audit, tmp_path):
    import json
    import subprocess
    import sys
    from pathlib import Path
    mp, digest = frozen_audit
    script = Path(__file__).resolve().parents[1] / 'scripts/audit_nfi_plot_reuse.py'
    done = subprocess.run([sys.executable, str(script), '--manifest', str(mp),
                           '--manifest-sha256', digest], capture_output=True, text=True, check=True)
    output = tmp_path / 'audit.json'; output.write_text(done.stdout)
    r = json.loads(output.read_text())
    assert len(r['per_model']) == 7
    assert r['all_models']['recorded_feature_pool']['cross_year_proximity_m']['0']['pairs'] == 1
    assert r['all_models']['recorded_training_split']['cross_year_proximity_m']['0']['pairs'] == 0
    assert 'private_' not in done.stdout and 'tile_name' not in done.stdout
    assert r['manifest_sha256'] == digest


@pytest.mark.parametrize('changed', ['manifest', 'features', 'split', 'source', 'training', 'dependency'])
def test_changed_frozen_inputs_fail_closed(frozen_audit, changed):
    import json
    from scripts.audit_nfi_plot_reuse import audit
    from imint.eval.fieldtruth import sha256_file
    mp, digest = frozen_audit
    if changed in ('manifest', 'dependency'):
        m = json.loads(mp.read_text()); m['source_sha256']['imint/eval/fieldtruth.py'] = 'f' * 64
        mp.write_text(json.dumps(m))
        if changed == 'dependency': digest = sha256_file(mp)
    else:
        name = {'features': 'tessera_r2_plot_features.parquet', 'split': 'tessera_r2_split.json',
                'source': 'nfi_index_unified_v2_512.parquet', 'training': 'training.parquet'}[changed]
        p = mp.parent / name; p.write_bytes(p.read_bytes() + b'changed')
    expected = {
        'manifest': 'frozen manifest differs from the approved SHA256',
        'features': 'frozen parquet content changed',
        'split': 'frozen JSON input content changed',
        'source': 'frozen parquet content changed',
        'training': 'frozen parquet content changed',
        'dependency': 'audit dependency differs from frozen source',
    }
    with pytest.raises(ValueError, match=expected[changed]):
        audit(mp, digest)
