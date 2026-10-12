"""Package merge must retain evaluation guards and canonical dependencies."""
import importlib
from pathlib import Path
import socket
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError('No network in package integration regression')
    for name in ('connect', 'connect_ex', 'bind'):
        monkeypatch.setattr(socket.socket, name, denied)
    monkeypatch.setattr(socket, 'getaddrinfo', denied)


def test_compare_retains_fieldtruth_and_canonical_schema():
    comparison = importlib.import_module('scripts.compare_nmd2023_nfi')
    from imint.eval.fieldtruth import load_frozen_holdout, validate_year_balance
    from imint.schema.class_schema import nmd_raster_to_lulc
    from imint.schema.unified_schema import nmd19_to_unified
    from scripts.nfi_nmd_coverage import nmd_covered
    assert comparison.load_frozen_holdout is load_frozen_holdout
    assert comparison.validate_year_balance is validate_year_balance
    assert comparison.nmd_raster_to_lulc is nmd_raster_to_lulc
    assert comparison.nmd19_to_unified is nmd19_to_unified
    assert comparison.nmd_covered is nmd_covered


def test_lucas_retains_skip_contract_and_canonical_schema():
    lucas = importlib.import_module('scripts.validate_against_lucas')
    from imint.training.errors import TilePrerequisiteError
    from imint.eval.fieldtruth import NoScoredObservations
    from imint.schema.unified_schema import UNIFIED_CLASSES
    assert lucas.TilePrerequisiteError is TilePrerequisiteError
    assert lucas.NoScoredObservations is NoScoredObservations
    assert lucas.UNIFIED_CLASSES is UNIFIED_CLASSES
    assert lucas.class_name(1) == UNIFIED_CLASSES[1]


def test_ladder_uses_and_fingerprints_canonical_metrics(tmp_path, monkeypatch):
    import builtins
    original = builtins.__import__
    def checked(name, *args, **kwargs):
        if name in {'imint.training.evaluate', 'imint.training.unified_schema'}:
            raise AssertionError('obsolete ladder dependency: ' + name)
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', checked)
    ladder = importlib.import_module('scripts.ladder_holdout_eval')
    from imint import metrics
    from imint.eval.fieldtruth import sha256_file
    assert ladder.compute_miou is metrics.compute_miou
    result = ladder.compute_miou(np.array([1, 0, 2]), np.array([1, 1, 2]), 28)
    assert result['confusion_matrix'][1, 0] == 1
    assert result['confusion_matrix'].sum() == 3
    assert ladder.derive(result['confusion_matrix'])['overall_accuracy'] == pytest.approx(2/3, abs=1e-4)
    args = SimpleNamespace(holdout_dir=tmp_path, label_dir=tmp_path, ckpt_root=tmp_path)
    (tmp_path/'fixture.npz').write_bytes(b'synthetic-fingerprint-input')
    fingerprint = ladder.run_fingerprint(['fixture'], args)
    expected = Path(metrics.__file__)
    assert fingerprint['code_sha256']['imint/metrics.py'] == sha256_file(expected)
    assert 'imint/training/evaluate.py' not in fingerprint['code_sha256']
