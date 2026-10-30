"""Positive supervision-mask contract from real dataset sidecars through loss."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from imint.training.losses import FractionLoss
from imint.training.unified_dataset import UnifiedDataset
from tests.test_tessera_wiring import tessera_tile_dir, TILE_H


@pytest.mark.parametrize('loss_type',['l1','smooth_l1'])
@pytest.mark.parametrize('case',['reliable','unreliable','zero','missing','corrupt'])
def test_dataset_mask_selects_reliable_signalled_pixels(tmp_path,loss_type,case):
    path=tmp_path/'synthetic.npz'
    if case=='corrupt':
        path.write_bytes(b'not an npz')
    elif case!='missing':
        frac=np.array([80,10,5,5],dtype=np.uint8).reshape(4,1,1)
        if case=='zero': frac[:]=0
        np.savez_compressed(path,frac=frac,frac_unreliable=np.array([[case=='unreliable']]))
    bundle=UnifiedDataset._load_frac_bundle(SimpleNamespace(frac_dir=tmp_path),path.name,1,1)
    target=torch.from_numpy(bundle[:-1])[None]
    mask=torch.from_numpy(bundle[-1])[None]
    logits=torch.zeros_like(target,requires_grad=True)
    loss=FractionLoss(loss_type)(logits,target,mask)
    loss.backward()
    if case=='reliable':
        assert loss.item()==pytest.approx(0.4 if loss_type=='l1' else 0.35)
        assert torch.count_nonzero(logits.grad)==4
        assert logits.grad[0,0,0,0]<0
        assert (logits.grad[0,1:,0,0]>0).all()
    else:
        assert loss.item()==0
        assert torch.count_nonzero(logits.grad)==0


@pytest.mark.parametrize('loss_type',['l1','smooth_l1'])
@pytest.mark.parametrize('family',['tessera','prithvi'])
def test_full_dataset_crop_to_fraction_gradient(tessera_tile_dir,tmp_path,loss_type,family):
    directory=tmp_path/'fracs'
    directory.mkdir()
    frac=np.zeros((4,TILE_H,TILE_H),dtype=np.uint8)
    frac[0]=80; frac[1]=10; frac[2:]=5
    frac[:,28:36]=0
    unreliable=np.zeros((TILE_H,TILE_H),dtype=np.uint8)
    unreliable[:,32:]=1
    np.savez_compressed(directory/'tile_0001.npz',frac=frac,frac_unreliable=unreliable)
    dataset=UnifiedDataset(lulc_dir=tessera_tile_dir,split='train',patch_size=32,
                           augment_override=False,backbone_family=family,frac_dir=directory)
    sample=dataset[0]
    expected_target=torch.from_numpy(frac[:,16:48,16:48].astype(np.float32)/100)
    expected_keep=torch.from_numpy(((frac>0).any(axis=0)&~unreliable.astype(bool))[16:48,16:48])
    torch.testing.assert_close(sample['frac'],expected_target)
    assert torch.equal(sample['frac_mask'].bool(),expected_keep)
    logits=torch.zeros_like(sample['frac'][None],requires_grad=True)
    loss=FractionLoss(loss_type)(logits,sample['frac'][None],sample['frac_mask'][None])
    loss.backward()
    # All positive targets differ from sigmoid(0), so each supervised pixel
    # must receive a nonzero derivative and each excluded pixel must not.
    actual=logits.grad.abs().sum(dim=1)[0]>0
    assert expected_keep.any() and (~expected_keep).any()
    assert torch.equal(actual,expected_keep)
    assert loss.item()==pytest.approx(0.4 if loss_type=='l1' else 0.35)


def test_one_optimizer_step_updates_only_supervised_logits(tmp_path):
    frac=np.array([[80,100,0],[10,0,0],[5,0,0],[5,0,0]],dtype=np.uint8)[:,None,:]
    np.savez_compressed(tmp_path/'tile.npz',frac=frac,frac_unreliable=np.array([[0,1,0]],dtype=np.uint8))
    bundle=UnifiedDataset._load_frac_bundle(SimpleNamespace(frac_dir=tmp_path),'tile.npz',1,3)
    target=torch.from_numpy(bundle[:-1])[None]
    mask=torch.from_numpy(bundle[-1])[None]
    logits=torch.nn.Parameter(torch.zeros_like(target))
    optimizer=torch.optim.SGD([logits],lr=0.1)
    criterion=FractionLoss()
    before=criterion(logits,target,mask)
    before.backward()
    optimizer.step()
    assert torch.count_nonzero(logits[...,0])==4
    assert torch.count_nonzero(logits[...,1:])==0
    assert criterion(logits,target,mask).item()<before.item()
