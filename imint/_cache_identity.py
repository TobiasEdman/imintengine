"""Fail-closed source identity for the two-stage prediction cache.

Capture the package before its first imports. The exec audit event records the
actual module code (including code loaded from pyc), not a later source read.
This observer does not change Python's import machinery. A running producer
cannot adopt a new deployment: restart it after any source replacement.
"""
from __future__ import annotations

import hashlib
import importlib.util
import os
from pathlib import Path
import sys

_ROOT = Path(__file__).absolute().parents[1]
_CAPTURE_ERROR = None
_WATCH_PREFIXES = ()
_SCRIPTS = ("scripts/inference_comparison.py", "scripts/infer_tiles.py",
            "scripts/cache_predict.py")


class ProducerIdentityError(RuntimeError):
    """Loaded code or source no longer belongs to the captured producer."""


# These upstream encoders can be source checkouts rather than distributions.
# Capture them before model loading too; installed library versions are recorded
# separately. No discovery imports or model/weight loads are performed here.
_EXTERNAL = {}

def _source_path(name):
    if name.startswith("@"):
        package, relative = name[1:].split("/", 1)
        return _EXTERNAL[package] / relative
    return _ROOT / name


def _tree_files(root):
    # Include encoder JSON configurations, not only Python implementations.
    return [p for p in root.rglob("*")
            if p.suffix in {".py", ".json", ".yaml", ".yml", ".toml"}]


def _paths():
    # Whole package: lazy registry/loaders, coordinates, normalizers, forwards,
    # and both pre-/post-PR59 canonical module paths without an allowlist.
    names = [p.relative_to(_ROOT).as_posix()
             for p in _tree_files(_ROOT / "imint")]
    names += [p for p in _SCRIPTS if (_ROOT / p).is_file()]
    for package, root in _EXTERNAL.items():
        names += [f"@{package}/{p.relative_to(root).as_posix()}"
                  for p in _tree_files(root)]
    return sorted(names)


def _stat(value):
    return (value.st_dev, value.st_ino, value.st_size,
            value.st_mtime_ns, value.st_ctime_ns)


_SOURCES = {}
_STATS = {}
_REVISION = None
_EXECUTED = {}
_EXPECTED = {}
_ENVIRONMENT = None
_ENVIRONMENT_ERROR = None

# Capture once, before package imports. Any failure is sticky cache state,
# never a reason to fail an ordinary library import or recapture newer code.
try:
    _lexical_root = _ROOT
    _ROOT = _ROOT.resolve()
    for _module in ("use_croma", "claymodel"):
        _spec = importlib.util.find_spec(_module)
        if _spec is not None and _spec.origin:
            _EXTERNAL[_module] = Path(_spec.origin).resolve().parent
    _WATCH_PREFIXES = tuple(str(p) + os.sep for p in
                            (_ROOT, _lexical_root, *_EXTERNAL.values()))
    for _name in _paths():
        _path = _source_path(_name)
        if _path.is_symlink():
            raise ProducerIdentityError(f"producer source is a link: {_path}")
        with _path.open("rb") as _stream:
            _before = _stat(os.fstat(_stream.fileno()))
            _SOURCES[_name] = _stream.read()
            if _before != _stat(os.fstat(_stream.fileno())):
                raise ProducerIdentityError(f"producer changed during capture: {_path}")
        if _before != _stat(_path.stat()):
            raise ProducerIdentityError(f"producer path changed during capture: {_path}")
        _STATS[_name] = _before
    _digest = hashlib.sha256()
    for _name, _source in _SOURCES.items():
        _digest.update(_name.encode() + b"\0" + _source)
    _REVISION = _digest.hexdigest()
except Exception as _exc:
    _CAPTURE_ERROR = _exc


def _record(code):
    path = Path(code.co_filename).resolve()
    name = path.relative_to(_ROOT).as_posix() if path.is_relative_to(_ROOT) else None
    if name not in _SOURCES:
        for package, root in _EXTERNAL.items():
            if path.is_relative_to(root):
                name = f"@{package}/{path.relative_to(root).as_posix()}"
                break
    if name in _SOURCES:
        _EXECUTED.setdefault(name, set()).add(code)


def _audit(event, args):
    # No filesystem work for unrelated execution. Observation failures must
    # never interrupt the code being observed; only cache validation refuses.
    global _CAPTURE_ERROR
    if event != "exec" or _CAPTURE_ERROR is not None:
        return
    try:
        code = args[0]
        if code.co_name != "<module>" or not code.co_filename.startswith(_WATCH_PREFIXES):
            return
        _record(code)
    except BaseException as exc:
        _CAPTURE_ERROR = exc


try:
    sys.addaudithook(_audit)
except Exception as _exc:
    _CAPTURE_ERROR = _exc
_audit("exec", (sys._getframe().f_code,))


def record_caller():
    """Record entry modules whose exec began before the observer was installed."""
    _audit("exec", (sys._getframe(1).f_code,))


def _check_module_code(name):
    executed = _EXECUTED.get(name)
    if not executed:
        raise ProducerIdentityError(f"producer module was loaded before capture: {name}")
    if name not in _EXPECTED:
        _EXPECTED[name] = compile(_SOURCES[name], str(_source_path(name)), "exec", dont_inherit=True)
    if any(code != _EXPECTED[name] for code in executed):
        raise ProducerIdentityError(f"loaded producer code differs from snapshot: {name}")


def producer_environment():
    """Lazily pin installed versions once; readers need not match this runtime."""
    global _ENVIRONMENT, _ENVIRONMENT_ERROR
    producer_revision()
    if _ENVIRONMENT is None and _ENVIRONMENT_ERROR is None:
        try:
            from importlib.metadata import distributions
            versions = tuple(sorted((d.metadata["Name"], d.version)
                                    for d in distributions()))
            if (any(not name or not version for name, version in versions)
                    or not {"numpy", "torch"}.issubset({name.lower() for name, _ in versions})):
                raise ValueError("producer numpy/torch distribution metadata is incomplete")
            _ENVIRONMENT = versions
        except Exception as exc:
            _ENVIRONMENT_ERROR = exc
    if _ENVIRONMENT_ERROR is not None:
        raise ProducerIdentityError(
            f"producer environment capture failed; restart: {_ENVIRONMENT_ERROR}"
        ) from _ENVIRONMENT_ERROR
    return {"python": sys.version, "packages": [list(pair) for pair in _ENVIRONMENT]}


def producer_revision(*, expected=None, helpers=None):
    """Check stable sources, module origins and actual executed module code."""
    if _CAPTURE_ERROR is not None:
        raise ProducerIdentityError(
            f"producer capture failed; restart: {_CAPTURE_ERROR}"
        ) from _CAPTURE_ERROR
    try:
        if _paths() != list(_SOURCES):
            raise ProducerIdentityError("producer source inventory changed; restart")
        for name, before in _STATS.items():
            if _stat(_source_path(name).stat()) != before:
                raise ProducerIdentityError(f"producer source changed: {name}; restart")
        for name, module in tuple(sys.modules.items()):
            if name != "imint" and not name.startswith("imint."):
                continue
            if module is None:
                continue
            origin = getattr(module, "__file__", None)
            relative = name.replace(".", "/")
            candidates = (relative + ".py", relative + "/__init__.py")
            match = next((p for p in candidates if p in _SOURCES), None)
            if match is None or origin is None or Path(origin).resolve() != _ROOT / match:
                raise ProducerIdentityError(f"foreign producer module origin: {name}")
            _check_module_code(match)
        for alias, relative in {
            "prithvi_mae": "imint/fm/prithvi_mae/prithvi_mae.py",
            "use_croma": "@use_croma/use_croma.py",
            "claymodel": "@claymodel/__init__.py",
        }.items():
            module = sys.modules.get(alias)
            if module is not None:
                if (relative not in _SOURCES or
                        Path(module.__file__).resolve() != _source_path(relative)):
                    raise ProducerIdentityError(f"foreign encoder module origin: {alias}")
                _check_module_code(relative)
        for name in _EXECUTED:
            _check_module_code(name)
        if helpers is not None:
            fn = helpers._build_inference_inputs
            origin = Path(fn.__code__.co_filename).resolve()
            if origin != _ROOT / "scripts/inference_comparison.py":
                raise ProducerIdentityError("foreign inference helper origin")
            _check_module_code("scripts/inference_comparison.py")
        if expected is not None and expected != _REVISION:
            raise ProducerIdentityError("worker producer differs from parent snapshot")
    except OSError as exc:
        raise ProducerIdentityError(f"producer source unavailable: {exc}") from exc
    return _REVISION
