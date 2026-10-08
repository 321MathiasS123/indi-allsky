from multiprocessing import Process
from types import SimpleNamespace

import pytest

from indi_allsky.render_backlog import RenderBacklogState, ResourceBackoff


def _record_in_worker(state, filename):
    assert state.observe_file(filename)
    assert state.record(25.0)


def test_state_is_shared_and_keeps_only_last_ten_renders(tmp_path):
    frame = tmp_path / 'capture.fit'
    frame.write_bytes(b'x' * 256)
    state = RenderBacklogState()
    child = Process(target=_record_in_worker, args=(state, str(frame)))
    child.start()
    child.join(15)
    if child.is_alive():
        child.terminate()
        child.join(5)
        pytest.fail('Shared render telemetry worker did not exit')
    assert child.exitcode == 0
    assert state.snapshot() == {
        'mean_seconds': 25.0, 'samples': 1, 'frame_bytes': 256,
        'spool_path': str(tmp_path.resolve()),
    }
    for seconds in range(1, 13):
        assert state.record(seconds)
    assert state.snapshot()['samples'] == 10
    assert state.snapshot()['mean_seconds'] == 7.5  # renders 3 through 12
    for seconds in (0, -1, float('nan'), float('inf')):
        assert not state.record(seconds)
    assert state.snapshot()['samples'] == 10


def test_missing_empty_files_and_busy_worker_lock_do_not_block(tmp_path):
    state = RenderBacklogState()
    frame = tmp_path / 'capture.fit'
    assert not state.observe_file(frame)
    frame.touch()
    assert not state.observe_file(frame)
    frame.write_bytes(b'frame')
    with state._lock:
        assert state.snapshot() is None
        assert not state.record(25)
        assert not state.observe_file(frame)
    assert state.snapshot()['samples'] == 0


def guard_fixture(monkeypatch, tmp_path, queue_max=3, frame_bytes=100, samples=True):
    frame = tmp_path / 'capture.fit'
    frame.write_bytes(b'x' * frame_bytes)
    state = RenderBacklogState()
    assert state.observe_file(frame)
    if samples:
        assert state.record(25)
    free = SimpleNamespace(value=100000)

    def disk_usage(path):
        assert path == str(tmp_path.resolve())
        return SimpleNamespace(free=free.value)

    monkeypatch.setattr('indi_allsky.render_backlog.shutil.disk_usage', disk_usage)
    monkeypatch.setattr('indi_allsky.render_backlog.psutil.disk_partitions',
                        lambda all: [SimpleNamespace(mountpoint=str(tmp_path.anchor), fstype='ext4')])
    return ResourceBackoff(state, queue_max), state, free


@pytest.mark.parametrize('queue_max,frame_bytes,reserve,release', [
    (0, 100, 300, 500), (3, 100, 500, 800), (8, 200, 2000, 3600),
])
def test_reserve_and_hysteresis_follow_queue_configuration_and_actual_file_size(
        monkeypatch, tmp_path, queue_max, frame_bytes, reserve, release):
    guard, state, free = guard_fixture(monkeypatch, tmp_path, queue_max, frame_bytes)
    free.value = reserve
    assert guard.delay(20, 1) == 0
    free.value -= 1
    assert guard.delay(20, 1) == pytest.approx(7.5)
    assert guard.latched
    assert guard.can_capture
    free.value = release - 1
    assert guard.delay(20, 1) == pytest.approx(7.5)
    free.value = release
    assert guard.delay(20, 1) == 0
    assert not guard.latched


@pytest.mark.parametrize('period,exposure,delay', [
    (20, 1, 7.5), (20, 25, 7.5), (20, 30, 0), (30, 1, 0), (5, 10, 22.5),
])
def test_guard_matches_start_plus_period_scheduler(monkeypatch, tmp_path, period, exposure, delay):
    guard, state, free = guard_fixture(monkeypatch, tmp_path)
    free.value = 400
    actual = guard.delay(period, exposure)
    assert actual == pytest.approx(delay)
    assert max(exposure, period + actual) >= 27.5


def test_critical_space_pauses_even_without_timing_and_recovers(monkeypatch, tmp_path):
    guard, state, free = guard_fixture(monkeypatch, tmp_path, samples=False)
    free.value = 199
    assert guard.delay(20, 1) == 0
    assert not guard.can_capture
    assert guard.latched
    free.value = 200
    assert guard.delay(20, 1) == 0
    assert guard.can_capture
    assert guard.latched


def test_busy_telemetry_uses_last_snapshot_and_storage_failure_keeps_guard(monkeypatch, tmp_path):
    guard, state, free = guard_fixture(monkeypatch, tmp_path)
    assert guard.delay(20, 1) == 0
    free.value = 199
    with state._lock:
        assert guard.delay(20, 1) == pytest.approx(7.5)
    assert not guard.can_capture

    def unavailable(path):
        raise OSError('temporarily unavailable')

    monkeypatch.setattr('indi_allsky.render_backlog.shutil.disk_usage', unavailable)
    assert guard.delay(20, 1) == pytest.approx(7.5)
    assert not guard.can_capture


def test_twilight_backlog_does_not_change_cadence_and_drains_at_longer_exposure(monkeypatch, tmp_path):
    guard, state, free = guard_fixture(monkeypatch, tmp_path)
    # Twenty minutes at 20-second captures / 25-second renders adds 12 frames.
    backlog = 0.0
    for _ in range(60):
        delay = guard.delay(20, 1)
        assert delay == 0
        cadence = max(1, 20 + delay)
        backlog += 1 - cadence / 25
        free.value -= 100 * (1 - cadence / 25)
    assert backlog == pytest.approx(12)
    # Fresh controls reach a genuine 30-second exposure; it takes 30 more
    # minutes to drain because new captures continue during recovery.
    for _ in range(60):
        delay = guard.delay(20, 30)
        assert delay == 0
        cadence = max(30, 20 + delay)
        backlog += 1 - cadence / 25
        free.value -= 100 * (1 - cadence / 25)
    assert backlog == pytest.approx(0)


def test_new_frame_size_and_render_throughput_are_observed(monkeypatch, tmp_path):
    guard, state, free = guard_fixture(monkeypatch, tmp_path)
    assert guard.delay(20, 1) == 0
    frame = tmp_path / 'larger.fit'
    frame.write_bytes(b'x' * 1000)
    assert state.observe_file(frame)
    for _ in range(10):
        assert state.record(30)
    free.value = 4500
    assert guard.delay(20, 1) == pytest.approx(13)
    assert guard.can_capture


@pytest.mark.parametrize('fstype', ['tmpfs', 'ramfs'])
@pytest.mark.parametrize('frame_bytes', [100, 10 * 1024 * 1024])
def test_memory_spool_pauses_before_disk_full_and_recovers_with_hysteresis(
        monkeypatch, tmp_path, fstype, frame_bytes):
    guard, state, free = guard_fixture(monkeypatch, tmp_path, frame_bytes=frame_bytes)
    free.value = 1024 * 1024 * 1024
    working_reserve = max(256 * 1024 * 1024, 32 * frame_bytes)
    memory = SimpleNamespace(available=working_reserve + 2 * frame_bytes - 1)
    monkeypatch.setattr('indi_allsky.render_backlog.psutil.virtual_memory', lambda: memory)
    monkeypatch.setattr('indi_allsky.render_backlog.psutil.disk_partitions', lambda all: [
        SimpleNamespace(mountpoint=str(tmp_path), fstype=fstype),
        SimpleNamespace(mountpoint=str(tmp_path.anchor), fstype='ext4'),
    ])
    assert guard.delay(20, 1) == pytest.approx(7.5)
    assert not guard.can_capture
    assert guard.latched
    memory.available = working_reserve - 1
    assert guard.delay(20, 1) == pytest.approx(7.5)
    assert not guard.can_capture
    memory.available = working_reserve + 2 * frame_bytes
    assert guard.delay(20, 1) == pytest.approx(7.5)
    assert guard.can_capture
    memory.available = working_reserve + 8 * frame_bytes - 1
    assert guard.delay(20, 1) == pytest.approx(7.5)
    memory.available += 1
    assert guard.delay(20, 1) == 0
    assert not guard.latched


def test_nested_disk_mount_does_not_use_ram_guard(monkeypatch, tmp_path):
    guard, state, free = guard_fixture(monkeypatch, tmp_path)
    monkeypatch.setattr('indi_allsky.render_backlog.psutil.disk_partitions', lambda all: [
        SimpleNamespace(mountpoint=str(tmp_path), fstype='ext4'),
        SimpleNamespace(mountpoint=str(tmp_path.parent), fstype='tmpfs'),
        SimpleNamespace(mountpoint=str(tmp_path) + '-other', fstype='ramfs'),
    ])

    def memory_must_not_be_used():
        pytest.fail('Disk-backed spool must not consult available RAM')

    monkeypatch.setattr('indi_allsky.render_backlog.psutil.virtual_memory', memory_must_not_be_used)
    assert guard.delay(20, 1) == 0
    assert guard.can_capture
    assert not guard.latched


def test_memory_mount_classification_is_cached_per_spool_path(monkeypatch, tmp_path):
    guard, state, free = guard_fixture(monkeypatch, tmp_path)
    calls = []
    disk_path = tmp_path / 'disk'
    disk_path.mkdir()

    def mounts(all):
        calls.append(all)
        return [SimpleNamespace(mountpoint=str(tmp_path), fstype='tmpfs'),
                SimpleNamespace(mountpoint=str(disk_path), fstype='ext4')]

    monkeypatch.setattr('indi_allsky.render_backlog.psutil.disk_partitions', mounts)
    monkeypatch.setattr('indi_allsky.render_backlog.psutil.virtual_memory',
                        lambda: SimpleNamespace(available=0))
    assert guard.delay(20, 1) == pytest.approx(7.5)
    assert not guard.can_capture
    assert guard.delay(20, 1) == pytest.approx(7.5)
    assert calls == [True]
    frame = disk_path / 'capture.fit'
    frame.write_bytes(b'x' * 100)
    assert state.observe_file(frame)
    monkeypatch.setattr('indi_allsky.render_backlog.shutil.disk_usage',
                        lambda path: SimpleNamespace(free=100000))
    assert guard.delay(20, 1) == 0
    assert guard.can_capture
    assert calls == [True, True]
