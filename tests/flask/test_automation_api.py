import importlib.util
from pathlib import Path
import sys
import types

from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_wtf.csrf import CSRFProtect
import pytest

from indi_allsky import automation


@pytest.fixture
def control_env(tmp_path, monkeypatch):
    # Load real models and the real blueprint without Linux camera dependencies.
    root = Path(__file__).resolve().parents[2]
    package = types.ModuleType('indi_allsky.flask')
    package.__path__ = [str(root / 'indi_allsky/flask')]
    package.db = database = SQLAlchemy()
    monkeypatch.setitem(sys.modules, package.__name__, package)

    def load(name, filename):
        spec = importlib.util.spec_from_file_location(name, root / filename)
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        return module

    models = load('indi_allsky.flask.models', 'indi_allsky/flask/models.py')
    package.models = models
    views = load('indi_allsky.flask.automation_views', 'indi_allsky/flask/automation_views.py')
    app = Flask(__name__)
    app.config.update(TESTING=True, SECRET_KEY='test-secret', AUTOMATION_TOKEN='x' * 40,
        AUTOMATION_ALLOWED_NETWORKS=['127.0.0.1/32'], AUTOMATION_RECOVERY_ENABLE=True,
        AUTOMATION_STATE_DIR=str(tmp_path), SQLALCHEMY_DATABASE_URI='sqlite://', LOGIN_DISABLED=True)
    database.init_app(app)
    app.register_blueprint(views.bp_automation_allsky)
    csrf = CSRFProtect(app)
    csrf.exempt(views.bp_automation_allsky)
    monkeypatch.setattr(views, 'boot_id', lambda: 'boot')
    monkeypatch.setattr(automation, 'boot_id', lambda: 'boot')
    monkeypatch.setattr(automation, 'start_helper', lambda: None)
    config = types.SimpleNamespace(config={}, config_id=1)
    monkeypatch.setattr(views, 'configuration', lambda: config)
    health = dict(stale=True, expected=True, last_capture=1)
    monkeypatch.setattr(views, 'capture_health', lambda *a: health)
    sync_status = dict(state='idle', active=False)
    calls = []
    sync = types.SimpleNamespace(DEFAULT_TYPES=['image'], status=lambda: dict(sync_status),
        get_state=lambda *a: 1, request_sync=lambda *a: calls.append(a) or types.SimpleNamespace(id=7),
        cancel_sync=lambda task_id: calls.append(('cancel', task_id)))
    monkeypatch.setattr(views, 'sync_module', lambda: sync)
    with app.app_context():
        database.create_all()
        yield types.SimpleNamespace(app=app, client=app.test_client(), views=views, db=database,
            models=models, config=config, health=health, sync=sync, sync_status=sync_status, calls=calls,
            headers={'Authorization': 'Bearer ' + app.config['AUTOMATION_TOKEN']})
        database.session.remove()
        database.drop_all()
        database.engine.dispose()


def post(env, path, data):
    return env.client.post('/indi-allsky/automation/' + path, json=data, headers=env.headers)


def test_requires_token_even_with_login_disabled_and_has_no_csrf_requirement(control_env):
    env = control_env
    assert env.client.get('/indi-allsky/automation/status').status_code == 401
    assert post(env, 'sync/start', {'request_id': 'one'}).status_code == 202
    env.app.config['AUTOMATION_TOKEN'] = ''
    assert post(env, 'sync/start', {'request_id': 'two'}).status_code == 503


def test_source_network_and_method_are_restricted(control_env):
    env = control_env
    env.app.config['AUTOMATION_ALLOWED_NETWORKS'] = ['192.168.1.2/32']
    assert post(env, 'system/reboot', {'request_id': 'one'}).status_code == 403
    assert env.client.get('/indi-allsky/automation/system/reboot', headers=env.headers).status_code == 405
    assert env.calls == []


def test_start_deduplicates_and_blocks_during_recovery(control_env):
    env = control_env
    assert post(env, 'sync/start', {'request_id': 'one'}).status_code == 202
    assert post(env, 'sync/start', {'request_id': 'one'}).status_code == 200
    assert len(env.calls) == 1
    assert post(env, 'system/recover', {'request_id': 'recovery'}).status_code == 202
    assert post(env, 'sync/start', {'request_id': 'two'}).get_json()['error'] == 'maintenance_active'


def test_failed_mqtt_alone_cannot_trigger_reboot(control_env):
    env = control_env
    env.health['stale'] = False
    assert post(env, 'system/reboot', {'request_id': 'one'}).get_json()['error'] == 'capture_not_stalled'
    assert post(env, 'system/reboot', {'request_id': 'manual', 'require_stale': False}).status_code == 202


def test_manual_cancel_is_not_automatically_undone(control_env):
    env = control_env
    env.sync_status.update(state='cancelled', reason='manual_cancel')
    assert post(env, 'sync/start', {'request_id': 'one'}).get_json()['error'] == 'sync_manually_cancelled'
    env.sync_status.update(state='interrupted', reason='maintenance')
    assert post(env, 'sync/start', {'request_id': 'two'}).status_code == 202


def test_unapplied_configuration_and_disabled_recovery(control_env):
    env = control_env
    env.config.config_id = 2
    assert post(env, 'sync/start', {'request_id': 'one'}).get_json()['error'] == 'configuration_not_applied'
    env.app.config['AUTOMATION_RECOVERY_ENABLE'] = False
    assert post(env, 'system/reboot', {'request_id': 'one'}).get_json()['error'] == 'recovery_disabled'


@pytest.mark.parametrize('data', [None, [], {}, {'request_id': True}, {'request_id': '../../run'}])
def test_malformed_start_has_no_side_effects(control_env, data):
    env = control_env
    assert post(env, 'sync/start', data).status_code == 400
    assert env.calls == []


def test_shared_latest_page_health_is_used_in_production(control_env, monkeypatch, tmp_path):
    shared = pytest.importorskip('indi_allsky.capture_health')
    if not hasattr(shared, 'local_capture_health'):
        pytest.skip('Updated capture-health feature is installed during production integration')
    from datetime import datetime
    from indi_allsky import automation_health, constants
    env = control_env
    camera = env.models.IndiAllSkyDbCameraTable(name='local', local=True, latitude=50, longitude=8,
        nightSunAlt=-6, capture_pause=False, daytime_capture=True, connectDate=datetime.now())
    env.db.session.add(camera)
    env.db.session.flush()
    env.db.session.add_all([
        env.models.IndiAllSkyDbStateTable(key='DB_CAMERA_ID', value=str(camera.id)),
        env.models.IndiAllSkyDbStateTable(key='STATUS', value=str(constants.STATUS_RUNNING)),
    ])
    env.db.session.commit()
    config = {'IMAGE_FOLDER': str(tmp_path), 'IMAGE_FILE_TYPE': 'jpg'}
    latest = tmp_path / 'latest.jpg'
    latest.write_bytes(b'image')
    import os
    import time
    os.utime(latest, (time.time() - 700, time.time() - 700))
    original = Path.read_text
    monkeypatch.setattr(Path, 'read_text', lambda path, *a, **kw:
        '10000 0' if str(path).replace('\\', '/') == '/proc/uptime' else original(path, *a, **kw))
    result = automation_health.capture_health(config)
    page = shared.local_capture_health(config, camera, True)
    assert result['health']['reason'] == page['reason'] == 'stale_image'
    assert result['age_seconds'] == page['last_success_age_s']
    assert result['stale']
    assert not automation_health.capture_health(config, maintenance=True)['stale']
    camera.capture_pause = True
    assert not automation_health.capture_health(config)['stale']
