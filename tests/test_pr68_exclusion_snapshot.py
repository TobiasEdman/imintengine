"""Boole review regressions: synthetic files only; no live freeze."""
import sys
import socket
import numpy as np
import pandas as pd
import pytest
from scripts import prepare_nfi_holdout as p
from tests.test_nfi_preparation_e2e import prepared_inputs

@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError("network prohibited in synthetic regression")
    for name in ("connect", "connect_ex", "bind"):
        monkeypatch.setattr(socket.socket, name, denied)
    monkeypatch.setattr(socket, "getaddrinfo", denied)

@pytest.mark.parametrize('mode',['all','campaign','balanced'])
def test_cohort_identity_excluded_before_population_filter(prepared_inputs,monkeypatch,mode):
    path=prepared_inputs['plot-index']; frame=pd.read_parquet(path)
    # Tract 3 is a new cohort-exposed identity, absent from teacher features.
    # Its duplicate campaign record must not bypass the identity exclusion.
    duplicate=frame.iloc[[0]].assign(TractID=3)
    survivor=frame.iloc[[2]].assign(TractID=99)
    pd.concat([frame,duplicate,survivor],ignore_index=True).to_parquet(path)
    args=['--population',mode]
    if mode=='balanced':
        extra=prepared_inputs['out-dir'].parent/'extra2019'; extra.mkdir()
        with np.load(prepared_inputs['staging-dir']/'campaign000.npz') as data: tile=dict(data)
        tile['dates']=np.array(['2018-10-07','2019-05-29','2019-06-28','2019-07-28'])
        np.savez(extra/'y2019.npz',**tile)
        frame=pd.read_parquet(path)
        pd.concat([frame,frame.iloc[[2]].assign(TractID=98,Year=2019,tile_name='y2019')],ignore_index=True).to_parquet(path)
        args += ['--inventory-years','2019','2024','--observations-per-year','1','--evaluation-root',f'nfi2019:2019:{extra}']
    monkeypatch.setattr(sys,'argv',sys.argv+args)
    p.main()
    held=pd.read_parquet(prepared_inputs['out-dir']/'holdout.parquet')
    assert sorted(held.TractID.tolist())==([98,99] if mode=='balanced' else [99]), held[['TractID','tile_role']].to_dict('records')


@pytest.mark.parametrize("indexed", [True, False])
def test_indexed_training_footprint_cannot_change_before_freeze(prepared_inputs,monkeypatch,indexed):
    if not indexed:
        ip=prepared_inputs["plot-index"]; frame=pd.read_parquet(ip)
        frame[frame.tile_name!="train"].to_parquet(ip)
    original=p.teacher_training_set
    changed=False
    path=prepared_inputs['cohort-dir']/'train.npz'
    def replace_after_geometry(*args,**kwargs):
        nonlocal changed
        result=original(*args,**kwargs)
        if not changed:
            with np.load(path) as data: tile=dict(data)
            tile['bbox_3006']=np.array([499990.,6499990.,500010.,6500010.])
            np.savez(path,**tile)
            changed=True
        return result
    monkeypatch.setattr(p,'teacher_training_set',replace_after_geometry)
    with pytest.raises(ValueError,match='changed'):
        p.main()
    assert not prepared_inputs['out-dir'].exists()
