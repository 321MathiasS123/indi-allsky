import json
from types import SimpleNamespace

import pytest

from indi_allsky import automation
from indi_allsky.shutdown import join_worker


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(automation, 'start_helper', lambda: None)
    return automation.ControlStore({'AUTOMATION_STATE_DIR': str(tmp_path)})


def test_duplicate_requests_survive_reopening_journal(store):
    first = automation.queue_recovery(store, 'recover', 'incident-1', 'boot-1', 1000)
    reopened = automation.ControlStore({'AUTOMATION_STATE_DIR': str(store.folder)})
    again = automation.queue_recovery(reopened, 'recover', 'incident-1', 'boot-1', 1001)
    assert first == again
    with pytest.raises(automation.ControlError, match='maintenance_active'):
        automation.queue_recovery(store, 'reboot', 'incident-2', 'boot-1', 1002)


def test_helper_can_start_before_dispatch_returns(store, monkeypatch):
    system = services()
    monkeypatch.setattr(automation, 'start_helper', lambda:
        automation.execute_recovery(store, system, 'boot', now=lambda: 1001))
    automation.queue_recovery(store, 'recover', 'incident', 'boot', 1000)
    assert store.read()['recovery']['state'] == 'complete'
    assert system.calls[-1] == ('start', 'capture.service')


def test_systemd_failed_stop_requires_inactive_confirmation(monkeypatch):
    import subprocess
    calls = []
    active = ['inactive']
    def run(args, **kwargs):
        calls.append(args)
        if args[2] == 'stop':
            raise subprocess.CalledProcessError(1, args)
        return SimpleNamespace(stdout=active[0])
    monkeypatch.setattr(automation.subprocess, 'run', run)
    system = automation.SystemServices({})
    system.service('stop', system.capture)
    assert calls[-1][2] == 'show'
    active[0] = 'deactivating'
    with pytest.raises(subprocess.CalledProcessError):
        system.service('stop', system.capture)


def test_second_controller_cannot_acquire_lock(store):
    with store.locked():
        with pytest.raises(automation.ControlError, match='control_busy'):
            with store.locked():
                pytest.fail('Concurrent controller acquired the lock')


@pytest.mark.parametrize('value', ['', None, True, 'a' * 129, '../command', 'a;reboot'])
def test_invalid_request_identifiers_are_rejected(value):
    with pytest.raises(automation.ControlError):
        automation.request_id(value)


def test_corrupt_state_does_not_forget_reboot(store):
    (store.folder / 'control.json').write_text('broken')
    with pytest.raises(json.JSONDecodeError):
        automation.queue_recovery(store, 'reboot', 'incident', 'boot', 1000)


def services(fail=False):
    calls = []
    def service(action, unit, **kwargs):
        calls.append((action, unit))
        if fail:
            raise RuntimeError('unit could not be stopped')
    return SimpleNamespace(capture='capture.service', driver='driver.service',
        restart_driver=True, service=service, reboot=lambda: calls.append(('reboot',)), calls=calls)


def test_recovery_stops_capture_before_driver_and_returns_status(store):
    automation.queue_recovery(store, 'recover', 'incident', 'boot', 1000)
    system = services()
    automation.execute_recovery(store, system, 'boot', now=lambda: 1001)
    assert system.calls == [('stop', 'capture.service'), ('restart', 'driver.service'), ('start', 'capture.service')]
    assert store.read()['recovery']['state'] == 'complete'
    automation.execute_recovery(store, system, 'boot', now=lambda: 1002)
    assert len(system.calls) == 3
    with pytest.raises(automation.ControlError, match='recovery_cooldown'):
        automation.queue_recovery(store, 'recover', 'another', 'boot', 1100)


def test_reboot_is_acknowledged_by_new_boot_not_http_acceptance(store):
    automation.queue_recovery(store, 'reboot', 'incident', 'boot', 1000)
    system = services()
    automation.execute_recovery(store, system, 'boot', now=lambda: 1001)
    assert system.calls == [('stop', 'capture.service'), ('reboot',)]
    assert store.read()['recovery']['state'] == 'rebooting'
    with store.locked():
        assert store.reconcile(store.read(), 'new-boot', 1100)['state'] == 'complete'
    duplicate = automation.queue_recovery(store, 'reboot', 'incident', 'new-boot', 1101)
    assert duplicate['state'] == 'complete'
    with pytest.raises(automation.ControlError, match='recovery_cooldown'):
        automation.queue_recovery(store, 'reboot', 'another', 'new-boot', 1102)


def test_failed_stop_does_not_reboot_and_is_reported(store):
    automation.queue_recovery(store, 'reboot', 'incident', 'boot', 1000)
    system = services(fail=True)
    with pytest.raises(RuntimeError):
        automation.execute_recovery(store, system, 'boot', now=lambda: 1001)
    assert system.calls == [('stop', 'capture.service')]
    assert store.read()['recovery']['error'] == 'service_control_failed'


def test_denied_reboot_attempts_to_restore_capture(store):
    automation.queue_recovery(store, 'reboot', 'incident', 'boot', 1000)
    system = services()
    def denied():
        system.calls.append(('reboot',))
        raise PermissionError('reboot denied')
    system.reboot = denied
    with pytest.raises(PermissionError):
        automation.execute_recovery(store, system, 'boot', now=lambda: 1001)
    assert system.calls == [('stop', 'capture.service'), ('reboot',), ('start', 'capture.service')]
    assert store.read()['recovery']['state'] == 'failed'


def test_dispatch_failure_releases_maintenance_but_remembers_request(store, monkeypatch):
    monkeypatch.setattr(automation, 'start_helper', lambda: (_ for _ in ()).throw(RuntimeError()))
    with pytest.raises(automation.ControlError, match='helper_unavailable'):
        automation.queue_recovery(store, 'recover', 'incident', 'boot', 1000)
    assert store.read()['recovery']['state'] == 'failed'
    assert automation.queue_recovery(store, 'recover', 'incident', 'boot', 1001)['state'] == 'failed'


def test_expired_request_is_not_executed_and_maintenance_can_clear(store):
    automation.queue_recovery(store, 'recover', 'incident', 'boot', 1000)
    system = services()
    automation.execute_recovery(store, system, 'boot', now=lambda: 2000)
    assert system.calls == []
    with store.locked():
        assert store.reconcile(store.read(), 'boot', 2000)['error'] == 'recovery_timeout'


def test_bounded_shutdown_kills_hung_process_but_not_threads(monkeypatch):
    from indi_allsky import shutdown
    monkeypatch.setattr(shutdown.time, 'monotonic', lambda: 100)
    calls = []
    worker = SimpleNamespace(name='hung', join=lambda **kw: calls.append(('join', kw)),
        is_alive=lambda: True, terminate=lambda: calls.append('terminate'), kill=lambda: calls.append('kill'))
    assert not join_worker(worker, deadline=110)
    assert calls == [('join', {'timeout': 10}), 'terminate', ('join', {'timeout': 2}), 'kill', ('join', {'timeout': 2})]
    del worker.terminate
    calls.clear()
    assert not join_worker(worker, deadline=90)
    assert calls == [('join', {'timeout': 0})]


def test_finished_worker_does_not_require_force():
    worker = SimpleNamespace(join=lambda **kw: None, is_alive=lambda: False, exitcode=0)
    assert join_worker(worker, deadline=0)
    worker.exitcode = -9
    assert not join_worker(worker, deadline=0)
