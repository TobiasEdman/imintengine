"""Planning counts must preserve true year identity and common pixel geometry."""
import json
import sys

import numpy as np
import pandas as pd
import pytest

from scripts import plan_nfi_tile_counts as plan
from scripts.prepare_nfi_holdout import select_balanced_years
from imint.eval.fieldtruth import NFI_KEY, NFI_YEAR_SAMPLING
from imint.training.nfi_colocate import point_to_pixel
from scripts.gen_ladder_manifests import DISTILL


def points(east, north=None, year=2024):
    return pd.DataFrame(dict(TractID=np.arange(len(east))+1,PlotID=1,Year=year,
                             Easting=east,Northing=north if north is not None else [6400000.]*len(east)))


def test_common_crop_matches_models_and_pixel_boundaries():
    assert plan.CROP_PX == min(c['img_size'] for c in DISTILL.values()) == 496
    box = plan.TILE.bbox_from_center(500000,6400000)
    frame = points([box['west']+x for x in (79.9,80.,5039.9,5040.,2560.,2560.)],
                   [box['north']-y for y in (2560.,2560.,2560.,2560.,79.9,5040.)])
    actual = plan.crop_mask(frame,box)
    expected=[]
    for row in frame.itertuples():
        rc=point_to_pixel(row.Easting,row.Northing,box,512)
        expected.append(rc is not None and all(8 <= x < 504 for x in rc))
    assert actual.tolist() == expected == [False,True,True,False,False,False]


def test_cover_splits_full_tile_span_that_does_not_fit_every_model():
    frame=points([512000.,517119.])
    got=plan.tile_cover(frame)
    assert got == dict(tiles=2,plots=2,original_lattice_cells=1,extra_crop_splits=1)
    assert plan.tile_cover(frame.iloc[::-1]) == got
    assert plan.tile_cover(frame.iloc[:0])['tiles'] == 0


def test_cover_never_shares_a_tile_across_inventory_years():
    frame=points([500000.,500001.]);frame.loc[1,'Year']=2023
    with pytest.raises(ValueError,match='one year'):
        plan.tile_cover(frame)


def test_planning_selection_matches_frozen_selection_protocol():
    frame=pd.concat([points([500000.,510000.,520000.],year=y) for y in (2023,2024)])
    expected,_=select_balanced_years(frame,dict(NFI_YEAR_SAMPLING,years=[2023,2024],observations_per_year=2))
    actual=plan.balanced_candidates(frame.sample(frac=1,random_state=91),[2023,2024],2)
    pd.testing.assert_frame_equal(actual,expected)


@pytest.mark.parametrize("bad_bbox", [False, True])
def test_main_counts_actual_exposure_and_legacy_geometric_reuse(tmp_path,monkeypatch,capsys,bad_bbox):
    for name in ('nfi','distill/heads','audits','holdout_val_512'):(tmp_path/name).mkdir(parents=True)
    frame=pd.concat([points([500000.,500100.,520000.],year=2023),
                     points([500000.,510000.,520000.,530000.],year=2024)],ignore_index=True)
    frame=frame.assign(VolPine=100.,VolContorta=0.,VolSpruce=0.,VolBirch=0.,VolOtherDec=0.)
    frame.to_parquet(tmp_path/'nfi/nfi_plots.parquet',index=False)
    source=frame[(frame.Year==2024)&(frame.TractID==4)].assign(tile_name='teacher')
    source.to_parquet(tmp_path/'nfi/nfi_index_unified_v2_512.parquet',index=False)
    for model in plan.NFI_MODELS:
        source.drop(columns='Year').to_parquet(tmp_path/f'distill/heads/{model}_r2_plot_features.parquet',index=False)
    frame[(frame.Year==2024)&(frame.TractID==1)].assign(row=256,col=256,tile_name='campaign').to_parquet(tmp_path/'nfi/nfi_index_v4_2026-09-30.parquet',index=False)
    bbox=plan.TILE.bbox_from_center(500000,6400000)
    (tmp_path/'audits/holdout_val_phaseA.json').write_text(json.dumps([dict(name='legacy',year=2023,bbox_3006=bbox)]))
    np.savez_compressed(tmp_path/'holdout_val_512/legacy.npz',spectral=np.zeros((24,512,512),dtype=np.float32),
                        year=2023,easting=500000.,northing=6400000.,tessera_source='old',
                        **({'bbox_3006':np.array([bbox['west'],np.nan,np.nan,bbox['north']])} if bad_bbox else {}))
    monkeypatch.setattr(sys,'argv',['plan','--data-root',str(tmp_path),'--years','2023','2024',
                                  '--observations-per-year','2'])
    plan.main();report=json.loads(capsys.readouterr().out)
    assert report['teacher_feature_union']==1
    assert report['balanced_planning_ceiling_per_year']==3
    assert report['legacy_metadata_verified_by_year']==({} if bad_bbox else {'2023':1})
    assert report['legacy_tessera_sources']==({} if bad_bbox else {'old':1})
    assert report['legacy_metadata_issues']==({'nonfinite_bbox':1} if bad_bbox else {})
    y23,y24=report['by_year']
    assert y23['geometric_reuse_all_roots']==(0 if bad_bbox else 2)
    assert y23['additional_balanced_if_legacy_reusable']['tiles']==(2 if bad_bbox else 1)
    assert y24['excluded_teacher_plot_years']==1
    assert y24['geometric_reuse_all_roots']==1
    assert report['input_freeze'] is False and report['fetch_started'] is False

    scenario, = report['additional_planning_scenarios']
    assert scenario['observations_per_year'] == 2
    assert [row['plot_years'] for row in scenario['by_year']] == [2, 2]
    for row in scenario['by_year']:
        assert row['geometric_reuse'] + row['additional_if_legacy_reusable']['plots'] == 2


def test_job_pins_identical_readonly_planner():
    import ast
    import hashlib
    from pathlib import Path
    import yaml

    root = Path(__file__).resolve().parents[1]
    source = (root/'scripts/plan_nfi_tile_counts.py').read_text()
    job = yaml.safe_load((root/'k8s/nfi0103-tile-counts-job.yaml').read_text())
    spec = job['spec']['template']['spec']
    container, = spec['containers']
    driver = container['command'][3]
    assignment, = [n for n in ast.parse(driver).body if isinstance(n, ast.Assign)
                   and isinstance(n.targets[0], ast.Name)
                   and n.targets[0].id == 'planner_source']
    assert ast.literal_eval(assignment.value) == source
    env = {v['name']: v['value'] for v in container['env']}
    assert env['NFI_PLANNER_SHA256'] == hashlib.sha256(source.encode()).hexdigest()
    assert '@sha256:' in container['image']
    assert container['image'] == env['NFI_RUNTIME_IMAGE']
    assert container['volumeMounts'][0]['readOnly'] is True
    assert spec['volumes'][0]['persistentVolumeClaim']['readOnly'] is True
    assert 'nvidia.com/gpu' not in container['resources']['limits']
    assert spec['automountServiceAccountToken'] is False
