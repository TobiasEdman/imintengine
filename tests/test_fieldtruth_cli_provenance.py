import importlib.util
import sys
import types
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]

def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / (name + '.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

nfi = load('validate_against_nfi')
lucas = load('validate_against_lucas')
inf = load('inference_comparison')
from imint.training.unified_dataset import TilePrerequisiteError


def tile(path, **overrides):
    data = dict(spectral=np.ones((24,8,8), dtype=np.float32) * .1,
                b08=np.ones((4,8,8), dtype=np.float32) * .1,
                rededge=np.ones((12,8,8), dtype=np.float32) * .1,
                doy=np.array([280,150,180,210]), temporal_mask=np.ones(4),
                year=np.int32(2022), easting=np.float64(500000),
                northing=np.float64(6500000))
    data.update(overrides)
    np.savez(path, **{k:v for k,v in data.items() if v is not None})
    return str(path)


def model(family='terramind'):
    return types.SimpleNamespace(fm_spec=types.SimpleNamespace(family=family),
                                 ck_cfg={}, num_frames=4)


def predictor(family='terramind', config_mismatch=False):
    m = model(family)
    if config_mismatch:
        m.ck_cfg = {'enabled_aux_names': ['dem']}
        m.n_aux_channels = 2
    def f(path):
        probs, _, _ = inf.run_inference(m, path, 'cpu', img_size=8, return_probs=True)
        return probs.argmax(0), probs
    return f


def index(kind, paths):
    rows=[]
    for i,p in enumerate(paths):
        row=dict(tile_name=Path(p).stem, tile_path=p, row=1, col=1, Year=2022)
        if kind=='nfi':
            row.update(TractID=i+1, PlotID=1, Easting=500000., Northing=6500000.,
                       VolPine=100., VolSpruce=0., VolBirch=0., VolContorta=0., VolOtherDec=0.)
        else:
            row.update(point_id=i+1, unified_class=1, unified_name='tallskog',
                       split='test', source='lucas', forest_dominant='pine')
        rows.append(row)
    return pd.DataFrame(rows)

@pytest.mark.parametrize('family',['croma','terramind'])
@pytest.mark.parametrize('version',[None,0,1,3])
def test_real_inference_produces_typed_s1_exception(tmp_path,family,version):
    path=tile(tmp_path/'bad.npz',s1_enrich_v=version)
    with pytest.raises(TilePrerequisiteError,match='s1_enrich_v'):
        predictor(family)(path)

@pytest.mark.parametrize('kind',['nfi','lucas'])
def test_real_config_failure_not_skipped(tmp_path,kind):
    path=tile(tmp_path/'bad.npz')
    score=getattr({'nfi':nfi,'lucas':lucas}[kind], 'score_against_'+kind)
    with pytest.raises(ValueError,match='aux conv takes 2'):
        score(index(kind,[path]),predictor(config_mismatch=True),skip_unscoreable=True)

@pytest.mark.parametrize('family,overrides,error',[('croma',{'b08':None},KeyError),
    ('terramind',{'s1_enrich_v':4},KeyError),
    ('terramind',{'s1_enrich_v':4,'s1_vv_vh':np.ones((8,8))},ValueError),
    ('tessera',{},KeyError)])
def test_other_tile_failures_remain_ineligible(tmp_path,family,overrides,error):
    path=tile(tmp_path/'bad.npz',**overrides)
    with pytest.raises(error) as caught:
        predictor(family)(path)
    assert not isinstance(caught.value,TilePrerequisiteError)

@pytest.mark.parametrize('kind',['nfi','lucas'])
@pytest.mark.parametrize('mixed',[False,True])
def test_real_cli_disk_output(tmp_path,monkeypatch,kind,mixed):
    import torch
    mod={'nfi':nfi,'lucas':lucas}[kind]
    paths=[tile(tmp_path/'bad.npz')]
    if mixed:
        paths.insert(0,tile(tmp_path/'good.npz',s1_enrich_v=4,
                           s1_vv_vh=np.full((2,8,8),.05,dtype=np.float32),has_s1=1))
    idx=tmp_path/'index.parquet'; index(kind,paths).to_parquet(idx,index=False)
    out=tmp_path/'result.json'; dump=tmp_path/'observations.parquet'
    cli=[kind,'--checkpoint','unused','--'+('plot' if kind=='nfi' else 'lucas')+'-index',
         str(idx),'--out',str(out),'--skip-unscoreable','--img-size','8',
         '--dump-per-'+('plot' if kind=='nfi' else 'point'),str(dump)]
    if kind=='lucas': cli += ['--data-dir',str(tmp_path),'--min-support','1']
    # Patch heavyweight model construction/forward only. NPZ preprocessing,
    # real S1 typed error, scoring, parquet and JSON output execute unchanged.
    monkeypatch.setattr(mod if kind=='nfi' else mod.van,'make_model_predict_fn',
                        lambda *a,**k:predictor())
    monkeypatch.setattr(inf,'_forward_from_inputs',
                        lambda *a,**k:torch.zeros((1,28,8,8)))
    monkeypatch.setattr(sys,'argv',cli)
    if mixed:
        mod.main()
        report=json.loads(out.read_text())
        assert report['skipped_tiles'][0]['tile']=='bad'
        assert report['skipped_tiles'][0]['plots' if kind=='nfi' else 'points']==1
        assert len(pd.read_parquet(dump))==1
        assert report['n_plots' if kind=='nfi' else '_meta']==(1 if kind=='nfi' else report['_meta'])
        if kind=='lucas': assert report['_meta']['points_scored']==1
    else:
        with pytest.raises(SystemExit,match='no (plot|point) could be scored'):
            mod.main()
        report = json.loads(out.read_text())
        assert report["status"] == "no_scored_observations"
        assert report["observations_scored"] == 0
        assert report["skipped_tiles"][0]["tile"] == "bad"
        assert report["skipped_tiles"][0]["plots" if kind == "nfi" else "points"] == 1
        assert pd.read_parquet(dump).empty
