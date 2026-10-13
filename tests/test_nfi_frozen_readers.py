import importlib.util
import importlib.abc
import sys
import json
import types
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
def load(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/'scripts'/(name+'.py'))
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod
nfi=load('validate_against_nfi');inf=load('inference_comparison')
from imint.eval.fieldtruth import sha256_file,verify_prediction_dump, evaluation_runtime_identity
from tests.nfi_freeze_fixtures import complete_manifest


def setup(tmp_path,heights=(8,)):
    paths=[];rows=[]
    for i,h in enumerate(heights):
        p=tmp_path/f'tile{i}.npz'
        np.savez(p,spectral=np.full((24,h,h),.1,np.float32),
                 tessera=np.full((128,h,h),.2,np.float32),year=2022)
        paths.append(p)
        rows.append(dict(TractID=i+1,PlotID=1,Year=2022,tile_name=p.stem,
                         tile_path=str(p),row=4,col=4,VolPine=100.,VolSpruce=0.,
                         VolBirch=0.,VolOtherDec=0.,VolContorta=0.,Easting=500000.,Northing=6500000.))
    held=pd.DataFrame(rows);hpath=tmp_path/'holdout.parquet';held.to_parquet(hpath,index=False)
    tpath=tmp_path/'training.parquet';held[['TractID','PlotID','Year']].iloc[:0].to_parquet(tpath,index=False)
    cp=tmp_path/'checkpoint.pt';cp.write_bytes(b'fixture-not-model')
    def identity(p):return dict(path=str(p),bytes=p.stat().st_size,sha256=sha256_file(p))
    manifest=dict(schema='nfi-fieldtruth-freeze-v1',source_sha256={},campaign={'training_tiles':0},
        holdout=dict(file=hpath.name,sha256=sha256_file(hpath),observations=len(held)),
        training=dict(file=tpath.name,sha256=sha256_file(tpath),observations=0),
        cells={'tessera_r1':dict(checkpoint=identity(cp),img_size=8,num_classes=23,backbone='tessera_v1')},
        protocol={'truth_dominant_fraction':.7},inputs=[],tiles={p.stem:dict(identity(p), geometry={"height": h, "width": h}) for p,h in zip(paths,heights)})
    mp=tmp_path/'manifest.json';mp.write_text(json.dumps(complete_manifest(manifest)))
    return paths,cp,mp


def patch_model(monkeypatch,after_preflight=lambda:None):
    m=types.SimpleNamespace(fm_spec=types.SimpleNamespace(family='tessera'),ck_cfg={},num_frames=1)
    seen={}
    def fake_load(*args,**kwargs):
        seen['checkpoint_kwargs']=kwargs
        after_preflight()
        return m,1,.5,8
    def fake_forward(model,inputs,device,**kwargs):
        seen['embedding']=float(inputs['img5d'].mean())
        logits=torch.zeros((1,23,8,8))
        for r in range(8): logits[0,r,r,:]=1
        return logits
    monkeypatch.setattr(inf,'_forward_from_inputs',fake_forward)
    original=importlib.util.spec_from_file_location
    class Loader(importlib.abc.Loader):
        def create_module(self,spec):return None
        def exec_module(self,mod):
            mod.load_model=fake_load
            mod.run_inference=inf.run_inference
    monkeypatch.setattr(importlib.util,'spec_from_file_location',
        lambda name,*a,**kw:importlib.util.spec_from_loader(name,Loader()) if name=='_infcmp' else original(name,*a,**kw))
    return seen


def cli(monkeypatch,tmp_path,cp,mp):
    # This suite isolates authenticated readers; full runtime checks have their own tests.
    monkeypatch.setattr(nfi, 'verify_evaluation_source', evaluation_runtime_identity)
    out=tmp_path/'results.json';dump=tmp_path/'predictions.parquet'
    monkeypatch.setattr(sys,'argv',['nfi','--checkpoint',str(cp),'--holdout-manifest',str(mp),
        '--expected-manifest-sha256',sha256_file(mp),'--cell','tessera_r1','--out',str(out),'--dump-per-plot',str(dump),'--device','cpu'])
    nfi.main()
    return out,dump


def test_frozen_tile_changed_after_check_is_rejected(tmp_path,monkeypatch):
    paths,cp,mp=setup(tmp_path)
    def rewrite_tile():
        np.savez(paths[0],spectral=np.full((24,8,8),.1,np.float32),
                 tessera=np.full((128,8,8),.8,np.float32),year=2022)
    seen=patch_model(monkeypatch,rewrite_tile)
    with pytest.raises(Exception, match="frozen tile.*mismatch"):
        cli(monkeypatch,tmp_path,cp,mp)
    assert seen["checkpoint_kwargs"]["expected_checkpoint_sha256"] == sha256_file(cp)
    assert seen["checkpoint_kwargs"]["expected_checkpoint_size"] == cp.stat().st_size
    assert "embedding" not in seen
    assert not (tmp_path / "predictions.parquet.meta.json").exists()


def test_frozen_mixed_tile_sizes_use_their_own_crop(tmp_path,monkeypatch):
    _,cp,mp=setup(tmp_path,heights=(8,12))
    patch_model(monkeypatch)
    _,dump=cli(monkeypatch,tmp_path,cp,mp)
    actual=pd.read_parquet(dump)['model_pred'].tolist()
    # Native row 4 is row 2 in the second tile's 8x8 centre crop.
    assert actual==[4,2]
