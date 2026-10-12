"""Ordinary PR59 library imports must not inherit cache-provenance failures."""
from pathlib import Path
import os
import shutil
import subprocess
import sys
import textwrap

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def source(tmp_path):
    root = tmp_path / "source"
    for folder in ("imint", "scripts"):
        shutil.copytree(ROOT / folder, root / folder,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    return root


def run(source, body):
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(body)], cwd=source,
        env={**os.environ, "PYTHONPATH": str(source), "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    print(result.stdout)


def test_symlinked_source_does_not_break_import_and_failure_is_sticky(source):
    path = source / "imint/training/sampler.py"
    real = source / "sampler-source.py"
    path.rename(real)
    path.symlink_to(real)
    run(source, '''
        import sys
        from pathlib import Path
        import imint
        assert "imint.engine" not in sys.modules
        assert "torch" not in sys.modules
        from imint import _cache_identity as identity
        for repair in (False, True):
            if repair:
                p = Path("imint/training/sampler.py")
                p.unlink()
                p.write_bytes(Path("sampler-source.py").read_bytes())
            try:
                identity.producer_revision()
            except identity.ProducerIdentityError as exc:
                assert "link" in str(exc)
            else:
                raise AssertionError("unsafe initial capture was retried or accepted")
        print("ordinary import succeeded; symlink capture remains refused after repair")
    ''')


def test_unavailable_metadata_is_deferred_and_producer_failure_is_sticky(source):
    run(source, '''
        import sys
        import importlib.metadata as metadata
        real = metadata.distributions
        calls = []
        def unavailable():
            calls.append(1)
            raise OSError("metadata unavailable")
        metadata.distributions = unavailable
        import imint
        from imint import _cache_identity as identity
        assert calls == []
        assert "imint.engine" not in sys.modules
        assert "torch" not in sys.modules
        identity.producer_revision()  # Source-only validation doesn't enumerate packages.
        for repair in (False, True):
            if repair:
                metadata.distributions = real
            try:
                identity.producer_environment()
            except identity.ProducerIdentityError as exc:
                assert "metadata unavailable" in str(exc)
            else:
                raise AssertionError("missing initial runtime identity was accepted/retried")
        assert calls == [1]
        print("metadata deferred; producer refuses permanently on first-use failure")
    ''')


def test_source_read_error_does_not_break_noncache_import(source):
    run(source, '''
        from pathlib import Path
        from unittest.mock import patch
        original = Path.open
        def unreadable(path, *args, **kwargs):
            if path.name == "sampler.py":
                raise PermissionError("source unreadable")
            return original(path, *args, **kwargs)
        with patch.object(Path, "open", unreadable):
            import imint
        from imint import _cache_identity as identity
        try:
            identity.producer_revision()
        except identity.ProducerIdentityError as exc:
            assert "source unreadable" in str(exc)
        else:
            raise AssertionError("failed source capture was accepted")
        print("ordinary import tolerates source read failure; cache refuses")
    ''')


def test_source_drift_during_capture_does_not_break_noncache_import(source):
    run(source, '''
        from pathlib import Path
        from unittest.mock import patch
        import os
        original = os.fstat
        calls = 0
        def drift(fd):
            global calls
            calls += 1
            value = original(fd)
            if calls == 2:
                from types import SimpleNamespace
                return SimpleNamespace(st_dev=value.st_dev, st_ino=value.st_ino,
                    st_size=value.st_size + 1, st_mtime_ns=value.st_mtime_ns,
                    st_ctime_ns=value.st_ctime_ns)
            return value
        with patch.object(os, "fstat", drift):
            import imint
        from imint import _cache_identity as identity
        try:
            identity.producer_revision()
        except identity.ProducerIdentityError as exc:
            assert "changed during capture" in str(exc)
        else:
            raise AssertionError("source drift was accepted")
        print("ordinary import tolerates capture drift; cache refuses")
    ''')


def test_observer_fast_skips_unrelated_execution_without_filesystem_calls(source):
    run(source, '''
        from unittest.mock import patch
        import imint
        from imint import _cache_identity as identity
        code = compile("value = 42", "/outside-watched-roots/unrelated.py", "exec")
        scope = {}
        with patch.object(identity, "Path", side_effect=AssertionError("unexpected filesystem work")):
            exec(code, scope)
        assert scope["value"] == 42
        identity.producer_revision()  # No observer error should have been stored.
        print("unrelated execution skipped without path/filesystem work")
    ''')


def test_observer_errors_do_not_interrupt_execution_but_invalidate_cache(source):
    run(source, '''
        from pathlib import Path
        from unittest.mock import patch
        import imint
        from imint import _cache_identity as identity
        code = compile("value = 42", str(Path("imint/observer-probe.py").resolve()), "exec")
        scope = {}
        with patch.object(identity, "_record", side_effect=OSError("observer failed")):
            exec(code, scope)
        assert scope["value"] == 42
        try:
            identity.producer_revision()
        except identity.ProducerIdentityError as exc:
            assert "observer failed" in str(exc)
        else:
            raise AssertionError("observer failure did not invalidate cache")
        print("observed execution continued; cache refused after observer failure")
    ''')


def test_producer_versions_are_pinned_on_first_use_without_loading_models(source):
    run(source, '''
        import sys
        import importlib.metadata as metadata
        from unittest.mock import patch
        import imint
        from imint import _cache_identity as identity
        assert "torch" not in sys.modules and "numpy" not in sys.modules
        runtime = identity.producer_environment()
        versions = dict(runtime["packages"])
        assert versions["torch"] == metadata.version("torch")
        assert versions["numpy"] == metadata.version("numpy")
        assert runtime["python"] == sys.version
        with patch.object(metadata, "distributions", side_effect=AssertionError("recapture")):
            assert identity.producer_environment() == runtime
        runtime["packages"].clear()
        assert identity.producer_environment()["packages"]
        assert "torch" not in sys.modules and "numpy" not in sys.modules
        print("Python/torch/numpy versions pinned lazily; no numerical modules loaded")
    ''')
