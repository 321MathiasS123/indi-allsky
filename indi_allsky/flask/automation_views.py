"""Token-authenticated automation endpoints, independent of browser sessions."""
import hmac
import ipaddress
import time

from flask import Blueprint, current_app as app, jsonify, request

from ..automation import ACTIVE, ControlError, ControlStore, boot_id, queue_recovery, request_id
from ..automation_health import capture_health


bp_automation_allsky = Blueprint('automation_indi_allsky', __name__,
                                url_prefix='/indi-allsky/automation')


def configuration():
    from ..config import IndiAllSkyConfig
    return IndiAllSkyConfig()


def sync_module():
    # Keep this feature usable on main without pulling in an unrelated branch.
    # Production combines it with feature/syncapi-on-demand.
    try:
        from .. import syncapi_sync
    except ImportError as exc:
        raise ControlError('sync_not_installed', 501) from exc
    return syncapi_sync


@bp_automation_allsky.before_request
def authorize():
    token = app.config.get('AUTOMATION_TOKEN', '')
    if not isinstance(token, str) or len(token) < 32:
        raise ControlError('automation_disabled', 503)
    supplied = request.headers.get('Authorization', '')
    if not hmac.compare_digest(supplied.encode(), ('Bearer ' + token).encode()):
        raise ControlError('authentication_failed', 401)
    networks = app.config.get('AUTOMATION_ALLOWED_NETWORKS', [])
    try:
        address = ipaddress.ip_address(request.remote_addr)
        allowed = any(address in ipaddress.ip_network(net) for net in networks)
    except ValueError:
        allowed = False
    if not allowed:
        raise ControlError('network_not_allowed', 403)


@bp_automation_allsky.errorhandler(ControlError)
def control_error(error):
    return jsonify(error=error.code), error.status


def payload():
    if request.content_length and request.content_length > 4096:
        raise ControlError('request_too_large', 413)
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        raise ControlError('invalid_payload', 400)
    return data


@bp_automation_allsky.get('/status')
def status():
    store = ControlStore(app.config)
    current_boot = boot_id()
    with store.locked():
        data = store.read()
        recovery = store.reconcile(data, current_boot, time.time())
    maintenance = recovery.get('state') in ACTIVE
    try:
        sync = sync_module().status()
        sync['available'] = True
    except ControlError:
        sync = dict(available=False, active=False)
    try:
        capture = capture_health(configuration().config, maintenance)
    except ControlError as exc:
        if exc.code != 'capture_health_not_installed':
            raise
        capture = dict(expected=False, stale=False, error=exc.code)
    return jsonify(boot_id=current_boot, maintenance=maintenance, recovery=recovery,
                   capture=capture, sync=sync)


@bp_automation_allsky.post('/sync/start')
def start_sync():
    data = payload()
    identifier = request_id(data.get('request_id'))
    sync = sync_module()
    config = configuration()
    if int(sync.get_state('CONFIG_ID', 0)) != config.config_id:
        raise ControlError('configuration_not_applied')
    store = ControlStore(app.config)
    with store.locked():
        journal = store.read()
        recovery = store.reconcile(journal, boot_id(), time.time())
        if recovery.get('state') in ACTIVE:
            raise ControlError('maintenance_active')
        previous = journal.get('sync_request', {})
        if previous.get('request_id') == identifier:
            return jsonify(previous), 200
        current = sync.status()
        if current.get('cancel_requested') or (current.get('state') == 'cancelled' and
                current.get('reason', 'manual_cancel') == 'manual_cancel'):
            raise ControlError('sync_manually_cancelled')
        try:
            task = sync.request_sync(config.config, data.get('types', sync.DEFAULT_TYPES))
        except ValueError as exc:
            return jsonify(error='sync_configuration', message=str(exc)), 400
        result = dict(request_id=identifier, task_id=task.id)
        journal['sync_request'] = result
        store.write(journal)
    return jsonify(result), 202


@bp_automation_allsky.post('/sync/cancel')
def cancel_sync():
    data = payload()
    if type(data.get('task_id')) is not int or data['task_id'] <= 0:
        raise ControlError('invalid_task_id', 400)
    sync = sync_module()
    with ControlStore(app.config).locked():
        sync.cancel_sync(data['task_id'])
    return jsonify(sync.status())


@bp_automation_allsky.post('/system/<action>')
def recover(action):
    if action not in ('recover', 'reboot'):
        raise ControlError('invalid_action', 404)
    if not app.config.get('AUTOMATION_RECOVERY_ENABLE', False):
        raise ControlError('recovery_disabled', 403)
    data = payload()
    # The watchdog must cross-check its MQTT alarm against current local data.
    # Explicit manual recovery can be requested with require_stale=false.
    if data.get('require_stale', True) is not False:
        health = capture_health(configuration().config)
        if not health['stale']:
            raise ControlError('capture_not_stalled')
    result = queue_recovery(ControlStore(app.config), action, data.get('request_id'),
                            boot_id(), time.time())
    return jsonify(result), 202
