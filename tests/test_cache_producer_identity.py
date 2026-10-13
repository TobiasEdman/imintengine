"""Red/green regressions derived from the independent RA05 review witnesses.

Each scenario uses a disposable source copy and a fresh interpreter: changing
code in the pytest process would deliberately poison its producer snapshot.
No weights or real tiles are used; preprocessing and cache orchestration are
real, with the deterministic dual-head fixture from the cache parity suite.
"""
from pathlib import Path
import os
import shutil
import subprocess
import sys
import textwrap

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def producer(tmp_path):
    root = tmp_path / "producer"
    for folder in ("imint", "scripts"):
        shutil.copytree(ROOT / folder, root / folder,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (root / "tests").mkdir()
    shutil.copyfile(ROOT / "tests/test_cached_validation_parity.py",
                    root / "tests/test_cached_validation_parity.py")
    return root


BOOT = '''
import importlib.util
import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
root = Path.cwd()
spec = importlib.util.spec_from_file_location("fixture", root / "tests/test_cached_validation_parity.py")
t = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t)
data = root / "tiles"
data.mkdir(exist_ok=True)
if not list(data.glob("*.npz")):
    for i in range(3):
        t._write_tile(data / f"tileX_{i}.npz", 100 + i)
cache = root / "cache"
cache.mkdir(exist_ok=True)
checkpoint = root / "checkpoint.pt"
checkpoint.write_bytes(b"synthetic checkpoint")
model = t.FakeDualHead()
tile = data / "tileX_0.npz"
'''


def run(root, code):
    result = subprocess.run(
        [sys.executable, "-c", BOOT + textwrap.dedent(code)], cwd=root,
        env={**os.environ, "PYTHONPATH": str(root), "OMP_NUM_THREADS": "1",
             "MKL_NUM_THREADS": "1", "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True, text=True, timeout=90)
    print(result.stdout)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("dependency", ["imint/training/sampler.py", "imint/training/unified_dataset.py",
                                         "imint/training/tile_time.py"])
def test_dependency_change_preserves_pinned_read_but_refuses_resume(producer, dependency):
    run(producer, '''
        sha, counts = t._run_stage_a(model, data, cache, checkpoint)
        assert counts == (3, 0)
        assert t._run_stage_a(model, data, cache, checkpoint)[1] == (0, 3)
        (root / "old-sha").write_text(sha)
        np.savez(root / "probe.npz", spectral=np.zeros((6, 16, 16), np.float32),
                 doy=np.array([180]), year=np.array(2022))
        inp = t.infcmp._build_inference_inputs(root / "probe.npz", t.torch.device("cpu"), 16, None)
        (root / "coords.json").write_text(json.dumps(inp["location_coords"].tolist()))
        t.infcmp._close_inference_inputs(inp)
    ''')
    path = producer / dependency
    source = path.read_text()
    if "sampler" in dependency:
        assert "math.radians(15.0)" in source
        source = source.replace("math.radians(15.0)", "math.radians(16.0)")
    elif path.suffix == ".json":
        source += "\n"
    else:
        source += "\n# next segmentation implementation deployment\n"
    path.write_text(source)
    run(producer, f"dependency = {dependency!r}\n" + textwrap.dedent('''
        sha = (root / "old-sha").read_text()
        # Stage B consumes recorded producer provenance, not its local code.
        t._cached_predict_fn(cache, sha, want_fracs=False)(tile)
        with patch.object(t._FakeInfcmp, 'load_model', side_effect=AssertionError('model loaded')):
            try:
                t._run_stage_a(model, data, cache, checkpoint)
            except ValueError as exc:
                assert 'producer_sources' in str(exc)
            else:
                raise AssertionError('changed producer silently regenerated')
        assert t._run_stage_a(model, data, cache, checkpoint, regenerate_invalid=True)[1] == (3, 0)
        assert t._run_stage_a(model, data, cache, checkpoint)[1] == (0, 3)
        if "sampler" in dependency:
            inp = t.infcmp._build_inference_inputs(root / "probe.npz", t.torch.device("cpu"), 16, None)
            before = json.loads((root / "coords.json").read_text())
            after = inp["location_coords"].tolist()
            assert before != after
            assert before[0][1] == 15.0 and after[0][1] == 16.0
            t.infcmp._close_inference_inputs(inp)
            print("UNTRACKED_PREPROCESSOR: coords changed 15 -> 16; old cache rejected")
        print("dependency changed: read rejected, regenerated 3, then reused 3")
    '''))


def test_loaded_old_normalizer_cannot_stamp_new_disk_revision(producer):
    # Same semantic mutation as cache-review-witness.py (PRODUCER_SKEW).
    run(producer, '''
        import imint.training.unified_dataset
        source = root / "imint/training/unified_dataset.py"
        source.write_text(source.read_text() + '\\nAUX_NORM["height"] = (0.0, 0.001)\\n')
        try:
            t._run_stage_a(model, data, cache, checkpoint)
        except RuntimeError as exc:
            assert "producer" in str(exc)
        else:
            raise AssertionError("loaded old normalizer stamped new disk identity")
        assert not list(cache.glob("*/*.npz"))
        print("PRODUCER_SKEW: refused before publication")
    ''')


def test_replacement_before_worker_preprocessing_is_rejected(producer):
    run(producer, '''
        from unittest.mock import patch
        original = t.infer_tiles.TileInferenceDataset.__getitem__
        changed = False
        def replace_then_preprocess(self, index):
            global changed
            if not changed:
                path = root / "imint/training/unified_dataset.py"
                path.write_text(path.read_text() + '\\nAUX_NORM["height"] = (0.0, 0.001)\\n')
                changed = True
            return original(self, index)
        with patch.object(t.infer_tiles.TileInferenceDataset, "__getitem__", replace_then_preprocess):
            try:
                t._run_stage_a(model, data, cache, checkpoint)
            except RuntimeError as exc:
                assert "producer" in str(exc)
            else:
                raise AssertionError("worker accepted changed source")
        assert not list(cache.glob("*/*.npz"))
        print("worker preprocessing source replacement: rejected")
    ''')


def test_replacement_during_forward_prevents_publication(producer):
    run(producer, '''
        original = model.forward
        def replace_then_forward(*args, **kwargs):
            path = root / "imint/training/sampler.py"
            path.write_text(path.read_text() + "\\n# changed during forward\\n")
            return original(*args, **kwargs)
        model.forward = replace_then_forward
        try:
            t._run_stage_a(model, data, cache, checkpoint)
        except RuntimeError as exc:
            assert "producer" in str(exc)
        else:
            raise AssertionError("published after producer drift")
        assert not list(cache.glob("*/*.npz"))
        print("forward source replacement: no entry published")
    ''')


def test_foreign_loaded_normalizer_is_rejected(producer):
    run(producer, '''
        import sys
        path = root / "foreign.py"
        path.write_text((root / "imint/training/unified_dataset.py").read_text()
                        + '\\nAUX_NORM["height"] = (0.0, 0.001)\\n')
        spec = importlib.util.spec_from_file_location("imint.training.unified_dataset", path)
        other = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(other)
        with patch.dict(sys.modules, {"imint.training.unified_dataset": other}):
            try:
                t._run_stage_a(model, data, cache, checkpoint)
            except RuntimeError as exc:
                assert "origin" in str(exc)
            else:
                raise AssertionError("accepted foreign loaded normalizer")
        assert not list(cache.glob("*/*.npz"))
        print("foreign module origin: rejected")
    ''')


def test_stale_bytecode_cannot_impersonate_new_source(producer):
    # Timestamp-based pyc can contain old code even with a fresh process and
    # unchanged source size/mtime. Compare executed code, not only source files.
    import py_compile
    source = producer / "imint/training/sampler.py"
    before = source.stat()
    py_compile.compile(str(source), doraise=True,
                       invalidation_mode=py_compile.PycInvalidationMode.TIMESTAMP)
    source.write_text(source.read_text().replace("math.radians(15.0)", "math.radians(16.0)"))
    os.utime(source, ns=(before.st_atime_ns, before.st_mtime_ns))
    run(producer, '''
        import imint.training.sampler
        try:
            t._run_stage_a(model, data, cache, checkpoint)
        except RuntimeError as exc:
            assert "loaded producer code" in str(exc)
        else:
            raise AssertionError("stale bytecode accepted as new source")
        assert not list(cache.glob("*/*.npz"))
        print("actual executed bytecode mismatch: rejected")
    ''')


def test_pinned_reader_independent_of_reader_source_drift(producer):
    run(producer, '''
        sha, _ = t._run_stage_a(model, data, cache, checkpoint)
        reader = t._cached_predict_fn(cache, sha, want_fracs=False)
        reader(tile)
        source = root / "imint/training/sampler.py"
        source.write_text(source.read_text().replace("math.radians(15.0)", "math.radians(16.0)"))
        reader(tile)
        print("pinned reader retains original producer provenance")
    ''')
