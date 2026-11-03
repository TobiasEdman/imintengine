"""Real filesystem and competing-writer checks for the public demo budget."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
from types import SimpleNamespace

import pytest

from imint.inference.lclu import EMBEDDING, Grid
from imint.inference.lclu_sources import (
    PREPARATION_LOG_BYTES, RegionalSources, RepositorySources, run_preparation,
)
from imint.inference.lclu_storage import PREPARATION_BYTES, StorageBudget, StorageLimitError
from scripts.cache_predict import file_identity


def budget(tmp_path, max_bytes=100):
    return StorageBudget(tmp_path, max_bytes=max_bytes, min_free_bytes=1)


def test_retained_bytes_survive_restart_and_block_new_writes(tmp_path):
    stored = tmp_path/'verified-result'
    stored.write_bytes(b'a'*60)
    guard = budget(tmp_path)
    with guard.reserve(40, paths=(tmp_path/'new-result',)):
        (tmp_path/'new-result').write_bytes(b'b'*40)
    restarted = budget(tmp_path)
    with pytest.raises(StorageLimitError, match='byte budget'):
        with restarted.reserve(1):
            pytest.fail('Full cache admitted a new write')
    assert stored.read_bytes() == b'a'*60
    assert (tmp_path/'new-result').read_bytes() == b'b'*40


def test_concurrent_writers_cannot_both_consume_last_allowance(tmp_path):
    guard = budget(tmp_path)
    entered, release = threading.Event(), threading.Event()

    def first():
        with guard.reserve(60, paths=(tmp_path/'first',)):
            entered.set()
            assert release.wait(5)
            (tmp_path/'first').write_bytes(b'x'*60)

    with ThreadPoolExecutor(2) as pool:
        running = pool.submit(first)
        try:
            assert entered.wait(5)
            with pytest.raises(StorageLimitError):
                with guard.reserve(60):
                    pytest.fail('Competing writer oversubscribed the cache')
        finally:
            release.set()
        running.result()
    with guard.reserve(40, paths=(tmp_path/'second',)):
        (tmp_path/'second').write_bytes(b'y'*40)
    assert guard.used_bytes() == 100


def test_free_space_reserve_is_checked_even_with_cache_capacity(tmp_path, monkeypatch):
    guard = StorageBudget(tmp_path, max_bytes=1000, min_free_bytes=100)
    monkeypatch.setattr('imint.inference.lclu_storage.shutil.disk_usage',
                        lambda _: SimpleNamespace(free=150))
    with guard.reserve(30):
        with pytest.raises(StorageLimitError, match='filesystem reserve'):
            guard.check(21)
    with guard.reserve(50):
        pass
    assert not list(tmp_path.iterdir())


def test_failed_work_releases_reservation_but_retained_logs_still_count(tmp_path):
    guard = budget(tmp_path)
    with pytest.raises(RuntimeError, match='failed'):
        with guard.reserve(80, paths=(tmp_path/'failure.log',)):
            (tmp_path/'failure.log').write_bytes(b'e'*25)
            raise RuntimeError('failed')
    with guard.reserve(75):
        pass
    with pytest.raises(StorageLimitError):
        guard.check(76)


@pytest.mark.parametrize('maximum,minimum', [(0,1),(-1,1),(True,1),(100,0),(100,-1)])
def test_invalid_configuration_fails_closed(tmp_path, maximum, minimum):
    with pytest.raises(ValueError):
        StorageBudget(tmp_path, max_bytes=maximum, min_free_bytes=minimum)


def source_settings(tmp_path):
    grid = Grid.from_lonlat(17.64, 59.86)
    bounds = grid.bounds()
    path = tmp_path/'source.json'
    path.write_text(json.dumps({'version':1, 'embedding':EMBEDDING, 'qualified_regions':[
        {'country':'SE', 'verification':'synthetic-test', 'years':[2024],
         'bounds_3006':[bounds[key] for key in ('west','south','east','north')]}]}))
    return grid, {'config':file_identity(path), 'python':'unused', 'timeout_seconds':10}


def test_full_preparation_cache_never_starts_child_or_changes_existing_data(tmp_path, monkeypatch):
    grid, settings = source_settings(tmp_path)
    root = tmp_path/'results'
    root.mkdir()
    retained = root/'old-result'
    retained.write_bytes(b'keep this data')
    guard = budget(root, PREPARATION_BYTES)
    source = RepositorySources(settings, tmp_path, root, storage=guard)
    monkeypatch.setattr(source, '_prepare', lambda *_: pytest.fail('Started preparation at quota'))
    before = sorted((str(p.relative_to(root)),p.read_bytes()) for p in root.rglob('*') if p.is_file())
    with pytest.raises(StorageLimitError):
        source.prepare(grid, 2024)
    after = sorted((str(p.relative_to(root)),p.read_bytes()) for p in root.rglob('*') if p.is_file())
    assert before == after and not list(root.rglob('.preparing-*'))


def test_regional_preparation_shares_reservations(tmp_path, monkeypatch):
    grid, settings = source_settings(tmp_path)
    root = tmp_path/'results'
    guard = budget(root, PREPARATION_BYTES)
    regional = RegionalSources([settings, settings], tmp_path, root, storage=guard)
    # Distinct regional adapters still draw from the one release allowance.
    monkeypatch.setattr(regional.sources[1], '_prepare', lambda *_: pytest.fail('Oversubscribed regions'))
    with regional.sources[0].storage.reserve(PREPARATION_BYTES):
        with pytest.raises(StorageLimitError):
            regional.sources[1].prepare(grid, 2024)


@pytest.mark.parametrize('exit_code', [0, 7])
def test_real_child_output_is_drained_and_persisted_with_a_hard_cap(tmp_path, exit_code):
    log = tmp_path/'preparation.log'
    command = [sys.executable, '-c',
               f'import sys; sys.stdout.buffer.write(b"x" * {4 * PREPARATION_LOG_BYTES}); '
               f'sys.stdout.flush(); print("last diagnostic", file=sys.stderr); sys.exit({exit_code})']
    if exit_code:
        with pytest.raises(subprocess.CalledProcessError) as failure:
            run_preparation(command, timeout=10, log_path=log)
        assert failure.value.returncode == exit_code
    else:
        run_preparation(command, timeout=10, log_path=log)
    assert log.stat().st_size == PREPARATION_LOG_BYTES
    assert log.read_bytes().startswith(b'[Earlier preparation output truncated]\n')
    assert log.read_bytes().endswith(b'last diagnostic\n')


def test_real_child_timeout_reaps_process_and_preserves_small_diagnostic(tmp_path):
    log = tmp_path/'preparation.log'
    command = [sys.executable, '-c',
               'import os, time; print(os.getpid(), flush=True); time.sleep(30)']
    with pytest.raises(subprocess.TimeoutExpired):
        run_preparation(command, timeout=1, log_path=log)
    pid = int(log.read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


def test_failed_preparation_retains_only_bounded_log(tmp_path):
    grid, settings = source_settings(tmp_path)
    # A real executable stands in for the preparation interpreter, and emits
    # output larger than the log cap through the unchanged source adapter.
    child = tmp_path/'noisy-child'
    child.write_text(f'#!{sys.executable}\nimport sys\n'
                     f'sys.stdout.buffer.write(b"x" * {3 * PREPARATION_LOG_BYTES})\n'
                     'sys.stdout.flush()\nprint("failed", file=sys.stderr)\nsys.exit(7)\n')
    child.chmod(0o700)
    settings['python'] = str(child)
    guard = budget(tmp_path/'results', PREPARATION_BYTES)
    source = RepositorySources(settings, tmp_path, guard.root, storage=guard)
    with pytest.raises(RuntimeError, match='operator log'):
        source.prepare(grid, 2024)
    assert not list(source.cache.glob('.preparing-*'))
    logs = list((source.cache/'failures').glob('*.log'))
    assert len(logs) == 1 and logs[0].read_bytes().endswith(b'failed\n')
    assert guard.used_bytes() == PREPARATION_LOG_BYTES
    guard.check(PREPARATION_BYTES - PREPARATION_LOG_BYTES)


@pytest.mark.parametrize('oversize', [None, 'arrays', 'metadata'])
def test_real_native_package_fits_allowance_and_rejects_oversize(tmp_path, monkeypatch, oversize):
    import numpy as np
    from imint.inference.lclu import AUX_NAMES
    from imint.inference.lclu_storage import PREPARATION_ARRAY_BYTES, PREPARATION_METADATA_BYTES
    from imint.training import skg_open
    from scripts import prepare_lclu as prep

    grid = Grid.from_lonlat(17.64, 59.86)
    metadata = tmp_path/'MANIFEST.json'
    metadata.write_text('{}')
    config = tmp_path/'sources.json'
    config.write_text(json.dumps({'version':1, 'embedding':EMBEDDING,
                                 'skg_manifest':file_identity(metadata), 'identity_cache':'identities'}))
    rng = np.random.default_rng(7)
    embedding = rng.standard_normal((128,512,512), dtype=np.float32).astype(np.float16)
    aux = {name:rng.random((512,512), dtype=np.float32) for name in AUX_NAMES}
    if oversize == 'arrays':
        embedding = embedding.astype(np.float32)  # twice the allowed native precision
    provenance = {'forestry':{'counties':[1]}}
    if oversize == 'metadata':
        provenance['excess'] = 'x' * PREPARATION_METADATA_BYTES
    monkeypatch.setattr(prep, 'prepare_aux', lambda *a: (aux, provenance))
    monkeypatch.setattr(prep, 'prepare_embedding',
                        lambda *a: (embedding, np.ones((512,512), bool), {}))
    monkeypatch.setattr(prep, 'verify_county_masks', lambda *a: {})
    monkeypatch.setattr(skg_open, 'county_cells', lambda *a: np.ones((512,512), bool))
    out = tmp_path/'prepared'
    if oversize:
        with pytest.raises(ValueError, match='storage allowance'):
            prep.prepare(config, grid, 2024, out)
        assert not (out/'entry.json').exists()
        if oversize == 'arrays':
            assert not (out/'input.npz').exists()
        return
    prep.prepare(config, grid, 2024, out)
    with np.load(out/'input.npz', allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved['tessera'], embedding)
        assert saved['tessera'].shape == (128,512,512)
    sizes = sum(p.stat().st_size for p in out.iterdir())
    # High-entropy native data, written by the actual producer, plus a full
    # diagnostic tail fits the writer's reservation with substantial margin.
    assert sizes + PREPARATION_LOG_BYTES < PREPARATION_BYTES
    assert (out/'input.npz').stat().st_size < PREPARATION_ARRAY_BYTES


def test_rendered_campaign_reserves_cache_and_free_space_on_existing_volume(tmp_path):
    from docker.lclu.render_job import resources
    rendered = resources('example/image@sha256:' + 'a'*64, 2024,
                         job_name='lclu-stockholm-0124-r9', reuse_volume=True,
                         campaign_years=tuple(range(2017,2026)), storage_gib=694,
                         max_original_bytes=305680252544, isolate_campaign=True)
    job = next(item for item in rendered['items'] if item['kind'] == 'Job')
    env = {item['name']:item['value'] for item in job['spec']['template']['spec']['containers'][0]['env']}
    assert int(env['IMINT_LCLU_STORAGE_MAX_BYTES']) == 8 * 2**30
    assert int(env['IMINT_LCLU_STORAGE_MIN_FREE_BYTES']) == 2 * 2**30
    assert not any(item['kind'] == 'PersistentVolumeClaim' for item in rendered['items'])
    guard = StorageBudget.from_environment(tmp_path)
    assert guard.max_bytes == 8 * 2**30 and guard.min_free_bytes == 2 * 2**30


def test_admission_does_not_scan_disk_and_writes_reconcile_only_their_paths(tmp_path, monkeypatch):
    from imint.inference import lclu_storage
    root = tmp_path/'results'
    guard = budget(root, 1000)
    original = lclu_storage._bytes
    checked = []

    def scoped(path):
        assert path != root, 'An admission or write rescanned the entire cache'
        checked.append(path)
        return original(path)

    monkeypatch.setattr(lclu_storage, '_bytes', scoped)
    for _ in range(50):
        guard.check(100)
    assert checked == []
    out = root/'map'
    with guard.reserve(100, paths=(out,)):
        out.mkdir()
        (out/'result').write_bytes(b'x'*90)
    assert checked == [out, out]
    assert guard.used_bytes() == 90


def test_startup_reclaims_only_abandoned_staging(tmp_path):
    pending = tmp_path/'lclu'/'.pending-interrupted'
    preparing = tmp_path/'lclu-inputs'/'.preparing-interrupted'
    result = tmp_path/'lclu'/('a'*64)
    source = tmp_path/'source-data'
    for directory in (pending, preparing, result, source):
        directory.mkdir(parents=True)
        (directory/'valuable').write_bytes(b'preserved')
    StorageBudget.cleanup_staging(tmp_path)
    assert not pending.exists() and not preparing.exists()
    assert (result/'valuable').read_bytes() == (source/'valuable').read_bytes() == b'preserved'
    assert budget(tmp_path).used_bytes() == 18


def test_lru_respects_a_real_concurrent_reader_and_survives_restart(tmp_path):
    guard = budget(tmp_path, 160)
    paths = [tmp_path/'lclu-inputs'/f'{i:064x}' for i in range(4)]
    for path in paths[:2]:
        with guard.reserve(60, paths=(path,)):
            path.mkdir(parents=True)
            (path/'input.npz').write_bytes(b'x'*60)
    entered, release = threading.Event(), threading.Event()

    def read_input():
        with guard.lease(paths[0]):
            entered.set()
            assert release.wait(5)
            assert (paths[0]/'input.npz').read_bytes() == b'x'*60

    with ThreadPoolExecutor(2) as pool:
        reading = pool.submit(read_input)
        try:
            assert entered.wait(5)
            with guard.reserve(60, paths=(paths[2],)):
                paths[2].mkdir()
                (paths[2]/'input.npz').write_bytes(b'z'*60)
            assert paths[0].exists() and not paths[1].exists()
        finally:
            release.set()
        reading.result()
    restarted = budget(tmp_path, 160)
    with restarted.reserve(60, paths=(paths[3],)):
        paths[3].mkdir()
        (paths[3]/'input.npz').write_bytes(b'y'*60)
    # The leased input was most recently used, including after the restart.
    assert paths[0].exists() and not paths[2].exists()
    assert restarted.used_bytes() == 120


def test_repeated_failed_preparation_overwrites_one_bounded_log(tmp_path):
    grid, settings = source_settings(tmp_path)
    child = tmp_path/'failed-child'
    child.write_text(f'#!{sys.executable}\nimport sys\nprint("failed"); sys.exit(7)\n')
    child.chmod(0o700)
    settings['python'] = str(child)
    source = RepositorySources(settings, tmp_path, tmp_path/'results')
    for _ in range(3):
        with pytest.raises(RuntimeError, match='operator log'):
            source.prepare(grid, 2024)
    logs = list((source.cache/'failures').glob('*.log'))
    assert len(logs) == 1 and source.storage.used_bytes() == len(b'failed\n')


def test_prepared_key_skips_admission_but_new_key_checks_capacity(tmp_path, monkeypatch):
    grid, settings = source_settings(tmp_path)
    source = RepositorySources(settings, tmp_path, tmp_path/'results')
    target = source._target(grid, 2024)
    target.mkdir()
    source.storage.max_bytes = 1
    source.check_preparation(grid, 2024)
    target.rmdir()
    with pytest.raises(StorageLimitError):
        source.check_preparation(grid, 2024)


def test_lazy_filesystem_free_space_does_not_drain_all_inputs(tmp_path, monkeypatch):
    paths = [tmp_path/'lclu-inputs'/f'{i:064x}' for i in range(5)]
    for path in paths:
        path.mkdir(parents=True)
        (path/'input.npz').write_bytes(b'x'*60)
    guard = StorageBudget(tmp_path, max_bytes=1000, min_free_bytes=100)
    calls = []
    def delayed_statfs(root):
        calls.append(root)
        return SimpleNamespace(free=150)  # deleted space isn't reported yet
    monkeypatch.setattr('imint.inference.lclu_storage.shutil.disk_usage', delayed_statfs)
    with guard.reserve(100):
        assert not paths[0].exists()
        assert all(p.exists() for p in paths[1:])
    assert len(calls) == 1 and guard.used_bytes() == 240


def test_interrupted_eviction_is_not_a_prepared_input_and_is_cleaned_on_restart(tmp_path, monkeypatch):
    from imint.inference import lclu_storage
    target = tmp_path/'lclu-inputs'/('a'*64)
    target.mkdir(parents=True)
    (target/'input.npz').write_bytes(b'x'*60)
    (target/'entry.json').write_bytes(b'y'*20)
    guard = budget(tmp_path, 100)
    real_rmtree = lclu_storage.shutil.rmtree

    def interrupted(path):
        assert path.name.startswith('.preparing-')
        assert not target.exists()  # renamed before the first destructive step
        (path/'input.npz').unlink()
        raise OSError('interrupted deletion')

    monkeypatch.setattr(lclu_storage.shutil, 'rmtree', interrupted)
    with pytest.raises(OSError, match='interrupted deletion'):
        with guard.reserve(50):
            pytest.fail('Started writer after interrupted cleanup')
    assert not target.exists() and guard.used_bytes() == 20
    assert not guard._writing and guard._reserved == 0
    assert len(list((tmp_path/'lclu-inputs').glob('.preparing-*'))) == 1
    monkeypatch.setattr(lclu_storage.shutil, 'rmtree', real_rmtree)
    StorageBudget.cleanup_staging(tmp_path)
    assert budget(tmp_path).used_bytes() == 0


def test_delayed_statfs_preserves_reserve_across_real_writes(tmp_path, monkeypatch):
    paths = [tmp_path/'lclu-inputs'/f'{i:064x}' for i in range(5)]
    for path in paths:
        path.mkdir(parents=True)
        (path/'input.npz').write_bytes(b'x'*60)
    guard = StorageBudget(tmp_path, max_bytes=1000, min_free_bytes=100)
    monkeypatch.setattr('imint.inference.lclu_storage.shutil.disk_usage',
                        lambda _: SimpleNamespace(free=150))
    initial_used = guard.used_bytes()
    for i in range(3):
        result = tmp_path/f'map-{i}'
        with guard.reserve(100, paths=(result,)):
            result.write_bytes(b'm'*100)
        actual_free = 150 + initial_used - guard.used_bytes()
        assert actual_free >= 100
    assert all(not p.exists() for p in paths)
    with pytest.raises(StorageLimitError, match='filesystem reserve'):
        with guard.reserve(100):
            pytest.fail('Stale statfs erased completed write debits')
    assert all((tmp_path/f'map-{i}').read_bytes() == b'm'*100 for i in range(3))


def test_lower_free_observation_is_kept_when_later_statfs_is_higher(tmp_path, monkeypatch):
    free = [1000]
    monkeypatch.setattr('imint.inference.lclu_storage.shutil.disk_usage',
                        lambda _: SimpleNamespace(free=free[0]))
    guard = StorageBudget(tmp_path, max_bytes=1000, min_free_bytes=100)
    result = tmp_path/'map'
    with guard.reserve(100, paths=(result,)):
        result.write_bytes(b'm'*100)
    free[0] = 120  # another PVC consumer reduces available space
    with pytest.raises(StorageLimitError, match='filesystem reserve'):
        guard.check(21)
    free[0] = 1000  # a delayed higher observation cannot reset the bound
    with pytest.raises(StorageLimitError, match='filesystem reserve'):
        with guard.reserve(21):
            pytest.fail('A later high observation erased the conservative bound')


def test_retained_failure_bytes_debit_free_estimate(tmp_path, monkeypatch):
    monkeypatch.setattr('imint.inference.lclu_storage.shutil.disk_usage',
                        lambda _: SimpleNamespace(free=150))
    guard = StorageBudget(tmp_path, max_bytes=1000, min_free_bytes=100)
    log = tmp_path/'failure.log'
    with pytest.raises(RuntimeError, match='failed'):
        with guard.reserve(40, paths=(log,)):
            log.write_bytes(b'e'*30)
            raise RuntimeError('failed')
    with pytest.raises(StorageLimitError, match='filesystem reserve'):
        guard.check(21)


def test_inflight_statfs_drop_is_not_debited_again_at_completion(tmp_path, monkeypatch):
    free = [300]
    monkeypatch.setattr('imint.inference.lclu_storage.shutil.disk_usage',
                        lambda _: SimpleNamespace(free=free[0]))
    guard = StorageBudget(tmp_path, max_bytes=1000, min_free_bytes=100)
    result = tmp_path/'map'
    with guard.reserve(100, paths=(result,)):
        result.write_bytes(b'm'*100)
        free[0] = 200  # statfs observes this writer before its counter update
        with pytest.raises(StorageLimitError):
            guard.check(1)  # conservatively still includes its full reservation
    with guard.reserve(100):
        pass  # the completed 100-byte write must only be debited once
    assert guard.used_bytes() == 100
