"""Opt-in, loaded-code identity for prediction-cache entry points only.

Ordinary library imports do nothing. Capture starts before producer imports;
only executed repository/encoder modules and opened model configs participate.
Readers retain the producer's recorded provenance, not their own runtime.
"""
from __future__ import annotations

import hashlib
import builtins
import importlib.util
import os
from pathlib import Path
import sys
import types

_ROOT = Path(__file__).resolve().parents[1]
_SELF_CODE = sys._getframe().f_code
_ACTIVE = False
_PRODUCER = False
_SNAPSHOT = {}
_SNAPSHOT_ERRORS = {}
_BUSY = False
_CAPTURE_ERROR = None
_WATCH_PREFIXES = ()
_EXTERNAL = {}
_SOURCES = {}
_STATS = {}
_EXECUTED = {}
_EXECUTED_FILES = set()
_EXPECTED = {}
_GLOBALS = []
_USED_MODULES = {}
_VERSIONS = {}
_VERSION_ERROR = None
_USED_PACKAGES = {'numpy', 'torch'}
_REAL_IMPORT = builtins.__import__
_REAL_DYNAMIC_IMPORT = importlib._bootstrap._gcd_import
_NUMERICAL_PACKAGES = {'numpy': 'numpy', 'torch': 'torch', 'scipy': 'scipy',
    'pyproj': 'pyproj', 'PIL': 'Pillow', 'timm': 'timm', 'einops': 'einops',
    'torchvision': 'torchvision', 'terratorch': 'terratorch',
    'transformers': 'transformers', 'omegaconf': 'omegaconf',
    'safetensors': 'safetensors', 'lightning': 'lightning'}


def _observed_import(name, globals=None, locals=None, fromlist=(), level=0):
    # Preserve import semantics, including cached imports which emit no audit
    # exec event. Enabled only by the inference entry point. Observation errors
    # are sticky cache failures, never exceptions in unrelated execution.
    global _CAPTURE_ERROR
    result = _REAL_IMPORT(name, globals, locals, fromlist, level)
    origin = (globals or {}).get('__file__', '')
    if (_BUSY or not _PRODUCER or _CAPTURE_ERROR is not None
            or not isinstance(origin, str) or not origin.startswith(_WATCH_PREFIXES)
            or origin == __file__):
        return result
    try:
        if _source_name(origin) not in _EXECUTED:
            return result
        absolute = (importlib.util.resolve_name('.' * level + name, globals['__package__'])
                    if level else name)
        names = [absolute] + [absolute + '.' + x for x in (fromlist or ()) if x != '*']
        for module_name in names:
            _note_import(module_name, sys.modules.get(module_name))
    except BaseException as exc:
        _CAPTURE_ERROR = exc
    return result


def _note_import(module_name, module):
    if module is not None:
        origin = getattr(module, '__file__', None)
        if module_name.startswith('imint.') or module_name.split('.')[0] in _EXTERNAL:
            if not isinstance(origin, str) or _source_name(origin) is None:
                raise ProducerIdentityError('foreign producer module origin: ' + module_name)
        _USED_MODULES[id(module)] = module
    top = module_name.split('.')[0]
    if top in _NUMERICAL_PACKAGES:
        _USED_PACKAGES.add(top)


def _observed_dynamic_import(name, package=None, level=0):
    # import_module bypasses builtins.__import__ for cached modules. Observe
    # its shared importlib dispatch so aliases bound BEFORE activation are
    # covered too. Preserve the original result/exception and observe only
    # calls reached from producer code; observation errors remain cache-only.
    global _CAPTURE_ERROR
    result = _REAL_DYNAMIC_IMPORT(name, package, level)
    if _BUSY or not _PRODUCER or _CAPTURE_ERROR is not None:
        return result
    try:
        frame = sys._getframe(1)
        while frame is not None:
            origin = frame.f_code.co_filename
            if origin != __file__ and origin in _EXECUTED_FILES:
                absolute = (importlib.util.resolve_name('.' * level + name, package)
                            if level else name)
                _note_import(absolute, result)
                break
            frame = frame.f_back
    except BaseException as exc:
        _CAPTURE_ERROR = exc
    return result

_SCRIPTS = {'scripts/inference_comparison.py', 'scripts/infer_tiles.py',
            'scripts/cache_predict.py'}


class ProducerIdentityError(RuntimeError):
    """Loaded code or source no longer belongs to the captured producer."""


def _stat(value):
    return (value.st_dev, value.st_ino, value.st_size,
            value.st_mtime_ns, value.st_ctime_ns)


def _source_path(name):
    if name.startswith('@'):
        package, relative = name[1:].split('/', 1)
        return _EXTERNAL[package] / relative
    return _ROOT / name


def _source_name(path):
    path = Path(path).absolute()
    if path.is_relative_to(_ROOT):
        name = path.relative_to(_ROOT).as_posix()
        if name.startswith('imint/') or name in _SCRIPTS:
            return name
    for package, root in _EXTERNAL.items():
        if path.is_relative_to(root):
            return f'@{package}/{path.relative_to(root).as_posix()}'
    return None


def _capture(name):
    if name in _SOURCES:
        return
    if _PRODUCER and name in _SNAPSHOT_ERRORS:
        raise _SNAPSHOT_ERRORS[name]
    if _PRODUCER and name in _SNAPSHOT:
        _SOURCES[name], _STATS[name] = _SNAPSHOT[name]
        return
    if _PRODUCER:
        raise ProducerIdentityError(f'producer source absent from initial snapshot: {name}')
    path = _source_path(name)
    if path.is_symlink() or path.resolve() != path:
        raise ProducerIdentityError(f'producer source is a link: {path}')
    with path.open('rb') as stream:
        before = _stat(os.fstat(stream.fileno()))
        source = stream.read()
        if before != _stat(os.fstat(stream.fileno())):
            raise ProducerIdentityError(f'producer changed during capture: {path}')
    if before != _stat(path.stat()):
        raise ProducerIdentityError(f'producer path changed during capture: {path}')
    _SOURCES[name], _STATS[name] = source, before


def _record(code):
    name = _source_name(code.co_filename)
    if name is not None:
        _capture(name)
        _EXECUTED.setdefault(name, set()).add(code)
        _EXECUTED_FILES.add(code.co_filename)


def _audit(event, args):
    global _BUSY, _CAPTURE_ERROR
    if not _ACTIVE or _BUSY or _CAPTURE_ERROR is not None:
        return
    if event not in ('exec', 'open'):
        return
    try:
        value = args[0]
        filename = value.co_filename if event == 'exec' else value
        if not isinstance(filename, str):
            return
        relative_resource = (event == 'open' and not os.path.isabs(filename)
                             and filename.endswith(('.json', '.yaml', '.yml', '.toml')))
        if not filename.startswith(_WATCH_PREFIXES) and not relative_resource:
            return
        # Attribute relative resource reads before normalizing them. Unrelated
        # execution still skips without Path/stat/read work.
        frame = sys._getframe(1)
        relevant = False
        while frame is not None:
            observed_file = frame.f_code.co_filename
            if observed_file != __file__ and observed_file in _EXECUTED_FILES:
                relevant = True
                break
            frame = frame.f_back
        if not relevant:
            return
        if relative_resource:
            filename = os.path.abspath(filename)
            if not filename.startswith(_WATCH_PREFIXES):
                return
        if event == 'exec' and value.co_name != '<module>':
            return
        # Python files are captured at exec (and compared to actual code).
        # Only model resources are captured on open; arbitrary package JSON
        # and installed-but-unused encoders do not participate.
        if event == 'open' and (Path(filename).suffix not in {'.json', '.yaml', '.yml', '.toml'}
                or not (filename.startswith(str(_ROOT / 'imint/fm') + os.sep)
                        or any(filename.startswith(str(p) + os.sep) for p in _EXTERNAL.values()))):
            return
        _BUSY = True
        if event == 'exec':
            _record(value)
        else:
            name = _source_name(filename)
            if name is not None:
                _capture(name)
    except BaseException as exc:
        _CAPTURE_ERROR = exc  # Observation never raises into the observed code.
    finally:
        _BUSY = False


def start_capture(*, producer=False):
    """Explicit entry-point activation; producer bytes are snapshotted once."""
    global _ACTIVE, _PRODUCER, _WATCH_PREFIXES, _CAPTURE_ERROR, _BUSY
    if not _ACTIVE:
        _ACTIVE = True
        try:
            for module in ('use_croma', 'claymodel'):
                spec = importlib.util.find_spec(module)
                if spec is not None and spec.origin:
                    _EXTERNAL[module] = Path(spec.origin).resolve().parent
            _WATCH_PREFIXES = tuple(str(p) + os.sep for p in
                (_ROOT / 'imint', _ROOT / 'scripts', *_EXTERNAL.values()))
            sys.addaudithook(_audit)
            _BUSY = True
            _record(_SELF_CODE)
        except Exception as exc:
            _CAPTURE_ERROR = exc
        finally:
            _BUSY = False
    if producer and not _PRODUCER:
        # Snapshot candidates early; only actual producer dependencies enter
        # the identity. Unused unreadable/symlinked candidates are irrelevant.
        # This one producer-only pass prevents late imports blessing new disk
        # bytes after model loading has started. Readers do not scan this tree.
        _BUSY = True
        try:
            paths = list((_ROOT / 'imint').rglob('*.py'))
            paths += [p for p in (_ROOT / 'imint/fm').rglob('*')
                      if p.suffix in {'.json', '.yaml', '.yml', '.toml'}]
            paths += [_ROOT / p for p in _SCRIPTS]
            for root in _EXTERNAL.values():
                paths += [p for p in root.rglob('*')
                          if p.suffix in {'.py', '.json', '.yaml', '.yml', '.toml'}]
            for path in paths:
                name = _source_name(str(path))
                if name is None:
                    continue
                try:
                    _capture(name)
                    _SNAPSHOT[name] = (_SOURCES[name], _STATS[name])
                except Exception as exc:
                    _SNAPSHOT_ERRORS[name] = exc
            # Keep only already executed sources, not the candidate inventory.
            for name in list(_SOURCES):
                if name not in _EXECUTED:
                    del _SOURCES[name]
                    del _STATS[name]
            _PRODUCER = True
            builtins.__import__ = _observed_import
            importlib._bootstrap._gcd_import = _observed_dynamic_import
        finally:
            _BUSY = False
    _BUSY = True
    try:
        caller = sys._getframe(1)
        _record(caller.f_code)
        _GLOBALS.append(caller.f_globals)
    except Exception as exc:
        _CAPTURE_ERROR = exc
    finally:
        _BUSY = False


def record_caller():
    # Explicit module opt-in also works for loaders that do not register their
    # module alias in sys.modules. Keep the actual executing code object.
    global _BUSY, _CAPTURE_ERROR
    if _ACTIVE:
        caller = sys._getframe(1)
        _BUSY = True
        try:
            _record(caller.f_code)
            _GLOBALS.append(caller.f_globals)
        except BaseException as exc:
            _CAPTURE_ERROR = exc
        finally:
            _BUSY = False


def _check_module_code(name):
    executed = _EXECUTED.get(name)
    if not executed:
        raise ProducerIdentityError(f'producer module was loaded before capture: {name}')
    if name not in _EXPECTED:
        _EXPECTED[name] = compile(_SOURCES[name], str(_source_path(name)), 'exec', dont_inherit=True)
    if any(code != _EXPECTED[name] for code in executed):
        raise ProducerIdentityError(f'loaded producer code differs from snapshot: {name}')


def _check_references():
    """Catch cached imports and aliases without blessing post-import disk bytes.

    Imports satisfied from sys.modules emit no exec event. Follow repository
    modules/functions/classes referenced by observed module globals and require
    their original exec observation too. Unrelated modules remain irrelevant.
    """
    modules = tuple(m for m in sys.modules.values() if isinstance(m, types.ModuleType))
    pending = [m for m in modules if getattr(m, '__file__', None)
               and _source_name(m.__file__) in _EXECUTED]
    pending.extend(_USED_MODULES.values())
    namespaces = list(_GLOBALS)
    visited = set()
    while pending or namespaces:
        if namespaces:
            values = tuple(namespaces.pop().values())
        else:
            module = pending.pop()
            if id(module) in visited:
                continue
            visited.add(id(module))
            origin = getattr(module, '__file__', None)
            own_name = _source_name(origin) if isinstance(origin, str) else None
            if own_name is not None and module.__name__ != 'imint':
                _capture(own_name)
                _check_module_code(own_name)
            else:
                continue
            values = tuple(vars(module).values())
        for value in values:
            if isinstance(value, types.ModuleType):
                dependency = value
            elif isinstance(value, (types.FunctionType, type)):
                dependency = sys.modules.get(value.__module__)
            else:
                continue
            if not isinstance(dependency, types.ModuleType) or dependency.__name__ == 'imint':
                continue
            origin = getattr(dependency, '__file__', None)
            name = _source_name(origin) if isinstance(origin, str) else None
            if name is None:
                if dependency.__name__.startswith('imint.'):
                    raise ProducerIdentityError('foreign producer module origin: ' + dependency.__name__)
                continue
            _capture(name)
            _check_module_code(name)
            pending.append(dependency)


def producer_revision(*, expected=None, helpers=None, expected_sources=None):
    if not _ACTIVE:
        raise ProducerIdentityError('cache capture is not active; import a cache entry point first')
    if _CAPTURE_ERROR is not None:
        raise ProducerIdentityError(f'producer capture failed; restart: {_CAPTURE_ERROR}') from _CAPTURE_ERROR
    try:
        _check_references()
        for name, before in tuple(_STATS.items()):
            if _stat(_source_path(name).stat()) != before:
                raise ProducerIdentityError(f'producer source changed: {name}; restart')
        for name, module in tuple(sys.modules.items()):
            if name == 'imint' or name == __name__ or module is None:
                continue
            if not name.startswith('imint.'):
                continue
            relative = name.replace('.', '/')
            choices = (relative + '.py', relative + '/__init__.py')
            origin = getattr(module, '__file__', None)
            match = next((p for p in choices if p in _SOURCES), None)
            if match is None:
                continue  # Incidental/scorer-only imports are not producer inputs.
            if origin is None or Path(origin).resolve() != _source_path(match):
                raise ProducerIdentityError(f'foreign or preloaded producer module origin: {name}')
            _check_module_code(match)
        for name in tuple(_EXECUTED):
            _check_module_code(name)
        if helpers is not None:
            fn = helpers._build_inference_inputs
            if Path(fn.__code__.co_filename).resolve() != _ROOT / 'scripts/inference_comparison.py':
                raise ProducerIdentityError('foreign inference helper origin')
            _check_module_code('scripts/inference_comparison.py')
            _check_module_code('imint/training/unified_dataset.py')
            _check_module_code('imint/training/sampler.py')
        sources = producer_sources()
        if expected_sources is not None:
            if any(expected_sources.get(name) != digest for name, digest in sources.items()):
                raise ProducerIdentityError('worker producer dependencies differ from parent snapshot')
            # Workers execute only preprocessing, but bind it to the complete
            # parent producer contract; never adopt fresh worker disk bytes.
            revision = source_revision(expected_sources)
        else:
            revision = source_revision(sources)
        if expected is not None and expected != revision:
            raise ProducerIdentityError('producer loaded dependencies differ from pinned snapshot; restart')
        return revision
    except OSError as exc:
        raise ProducerIdentityError(f'producer source unavailable: {exc}') from exc


def producer_environment(device='cpu'):
    """Only loaded numerical/model dependencies, plus the producing backend."""
    global _VERSION_ERROR
    producer_revision()
    from importlib.metadata import version
    import torch
    try:
        if _VERSION_ERROR is not None:
            raise _VERSION_ERROR
        for module in sorted(_USED_PACKAGES):
            distribution = _NUMERICAL_PACKAGES[module]
            if distribution not in _VERSIONS:
                _VERSIONS[distribution] = version(distribution)
        packages = sorted(_VERSIONS.items())
    except Exception as exc:
        _VERSION_ERROR = exc
        raise ProducerIdentityError(f'producer numerical package metadata unavailable: {exc}') from exc
    dev = torch.device(device)
    backend = {'type': dev.type}
    if dev.type == 'cuda':
        backend.update(name=torch.cuda.get_device_name(dev),
                       capability=list(torch.cuda.get_device_capability(dev)),
                       cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version())
    else:
        import platform
        backend['machine'] = platform.machine()
    return {'python': sys.version.split()[0], 'packages': [list(p) for p in packages],
            'device': backend, 'numerics': {
                'float32_matmul_precision': torch.get_float32_matmul_precision(),
                'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
                'cudnn_deterministic': torch.backends.cudnn.deterministic,
                'cudnn_benchmark': torch.backends.cudnn.benchmark,
                'cuda_matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32}}


def producer_sources():
    return {name: hashlib.sha256(source).hexdigest() for name, source in sorted(_SOURCES.items())}


def source_revision(sources):
    import json
    return hashlib.sha256(json.dumps(sources, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def validate_recorded_sources(sources):
    """Preflight prior producer dependencies against the initial snapshot."""
    if not _PRODUCER:
        raise ProducerIdentityError('producer preflight requires early capture')
    failures = []
    for name, expected in sources.items():
        item = _SNAPSHOT.get(name)
        if item is None or hashlib.sha256(item[0]).hexdigest() != expected:
            failures.append('producer_sources.' + name)
    return failures
