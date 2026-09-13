"""Durable coordination for short automation requests and a separate recovery job.

This module deliberately needs neither Flask nor the capture daemon. The systemd
helper must remain usable when capture, its database session, or Gunicorn stalls.
"""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import re
import subprocess
import time


RECOVERY_UNIT = 'indi-allsky-automation.service'
ACTIVE = ('requested', 'running', 'rebooting')


class ControlError(Exception):
    def __init__(self, code, status=409):
        super().__init__(code)
        self.code = code
        self.status = status


def boot_id():
    return Path('/proc/sys/kernel/random/boot_id').read_text().strip()


def request_id(value):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9_.:-]{1,128}', value):
        raise ControlError('invalid_request_id', 400)
    return value


def maintenance_active(config):
    operation = ControlStore(config).read().get('recovery', {})
    return (operation.get('state') in ACTIVE and operation['boot_id'] == boot_id()
            and time.time() <= operation['deadline'])


def shutdown_notice(config):
    try:
        operation = ControlStore(config).read().get('recovery', {})
        if operation.get('started') and 0 <= time.time() - operation['requested'] < 900:
            return 'Previous indi-allsky shutdown was incomplete during requested recovery; check the capture log'
    except (OSError, ValueError, KeyError):
        pass
    return 'indi-allsky was abnormally shutdown'


class ControlStore:
    def __init__(self, config):
        self.folder = Path(config.get('AUTOMATION_STATE_DIR', '/var/lib/indi-allsky/automation'))

    @contextmanager
    def locked(self, wait_seconds=0):
        self.folder.mkdir(mode=0o700, parents=True, exist_ok=True)
        with (self.folder / 'control.lock').open('a+b') as lock:
            # Windows is supported for isolated tests; production uses flock.
            if os.name == 'nt':
                import msvcrt
                lock.write(b'\0')
                lock.flush()
                lock.seek(0)
                acquire = lambda: msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                acquire = lambda: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            deadline = time.monotonic() + wait_seconds
            while True:
                try:
                    acquire()
                    break
                except OSError as exc:
                    if time.monotonic() >= deadline:
                        raise ControlError('control_busy') from exc
                    time.sleep(0.05)
            try:
                yield self
            finally:
                if os.name == 'nt':
                    lock.seek(0)
                    msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(lock, fcntl.LOCK_UN)

    def read(self):
        try:
            return json.loads((self.folder / 'control.json').read_text())
        except FileNotFoundError:
            return {}
        # A corrupt journal must fail closed, never silently forget a reboot.

    def write(self, data):
        path = self.folder / 'control.tmp'
        with path.open('w') as stream:
            os.chmod(path, 0o600)
            json.dump(data, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(path, self.folder / 'control.json')
        if os.name != 'nt':
            directory = os.open(self.folder, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)

    def reconcile(self, data, current_boot, now):
        operation = data.get('recovery', {})
        if operation.get('state') not in ACTIVE:
            return operation
        if operation['boot_id'] != current_boot:
            operation.update(state='complete' if operation['state'] == 'rebooting' else 'interrupted', finished=now)
        elif now > operation['deadline']:
            operation.update(state='failed', error='recovery_timeout', finished=now)
        else:
            return operation
        data.setdefault('requests', {})[operation['action']] = dict(operation)
        self.write(data)
        return operation


def start_helper():
    # A fixed user service: requests cannot supply commands, units, or arguments.
    import dbus
    manager = dbus.Interface(dbus.SessionBus().get_object(
        'org.freedesktop.systemd1', '/org/freedesktop/systemd1'),
        'org.freedesktop.systemd1.Manager')
    manager.StartUnit(RECOVERY_UNIT, 'fail', timeout=5)


def queue_recovery(store, action, identifier, current_boot, now):
    if action not in ('recover', 'reboot'):
        raise ControlError('invalid_action', 400)
    request_id(identifier)
    with store.locked():
        data = store.read()
        operation = store.reconcile(data, current_boot, now)
        previous = data.get('requests', {}).get(action)
        if previous and previous['request_id'] == identifier:
            return previous
        if operation.get('state') in ACTIVE:
            raise ControlError('maintenance_active')
        cooldown = 3600 if action == 'reboot' else 600
        if previous and now - previous['requested'] < cooldown:
            raise ControlError('recovery_cooldown', 429)
        operation = dict(action=action, request_id=identifier, state='requested',
                         boot_id=current_boot, requested=now, deadline=now + 900)
        data['recovery'] = operation
        data.setdefault('requests', {})[action] = operation
        store.write(data)
    # Release before dispatch: the helper may start before StartUnit replies.
    try:
        start_helper()
    except Exception as exc:
        with store.locked(wait_seconds=5):
            data = store.read()
            current = data.get('recovery', {})
            if current.get('request_id') == identifier and current.get('state') == 'requested':
                current.update(state='failed', error='helper_unavailable', finished=now)
                data['requests'][action] = dict(current)
                store.write(data)
        raise ControlError('helper_unavailable', 503) from exc
    return operation


class SystemServices:
    def __init__(self, config):
        # Names come exclusively from the administrator's local configuration.
        self.capture = config.get('ALLSKY_SERVICE_NAME', 'indi-allsky.service')
        self.driver = config.get('INDISERVER_SERVICE_NAME', 'indiserver.service')
        self.restart_driver = config.get('AUTOMATION_RESTART_INDISERVER', False)

    def service(self, action, unit, timeout=270):
        try:
            return subprocess.run(['systemctl', '--user', action, unit],
                                  check=True, capture_output=True, timeout=timeout)
        except subprocess.CalledProcessError:
            if action != 'stop':
                raise
            # A timed-out stop may have killed a hung process successfully.
            # Continue only after systemd confirms that the unit is inactive.
            result = subprocess.run(['systemctl', '--user', 'show', unit,
                                     '--property=ActiveState', '--value'],
                                    check=True, capture_output=True, timeout=10, text=True)
            if result.stdout.strip() not in ('inactive', 'failed'):
                raise

    def reboot(self):
        import dbus
        manager = dbus.Interface(dbus.SystemBus().get_object(
            'org.freedesktop.login1', '/org/freedesktop/login1'),
            'org.freedesktop.login1.Manager')
        manager.Reboot(False, timeout=10)


def execute_recovery(store, services, current_boot, now=time.time):
    """Run from the oneshot unit, never from a web worker or capture queue."""
    with store.locked(wait_seconds=5):
        data = store.read()
        operation = data.get('recovery', {})
        if (operation.get('state') != 'requested' or operation.get('boot_id') != current_boot
                or now() > operation['deadline']):
            return
        identifier = operation['request_id']
        operation.update(state='running', started=now())
        store.write(data)

    def update(**values):
        with store.locked(wait_seconds=5):
            data = store.read()
            if data.get('recovery', {}).get('request_id') != identifier:
                raise RuntimeError('Recovery operation changed')
            data['recovery'].update(values)
            data['requests'][operation['action']] = dict(data['recovery'])
            store.write(data)

    try:
        # systemctl waits for the stop job. The capture unit gives its parent a
        # bounded grace period before killing stuck processes in the cgroup.
        services.service('stop', services.capture)
        if operation['action'] == 'recover':
            if services.restart_driver:
                services.service('restart', services.driver, timeout=90)
            services.service('start', services.capture, timeout=90)
            update(state='complete', finished=now())
        else:
            # Persist intent before losing the machine. A changed boot ID is
            # the acknowledgement; accepting the D-Bus call is not completion.
            update(state='rebooting')
            services.reboot()
    except Exception:
        update(state='failed', error='service_control_failed', finished=now())
        raise
