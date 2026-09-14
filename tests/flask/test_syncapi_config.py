"""The configuration save transaction, without unrelated camera form fields."""
import ast
from copy import deepcopy
from datetime import datetime, timezone
import importlib
from pathlib import Path
from types import SimpleNamespace

import flask
from flask.views import View
from flask_login import LoginManager, current_user, login_required
from flask_wtf.csrf import CSRFProtect, generate_csrf
import pytest
from sqlalchemy import select

from indi_allsky.exceptions import ConfigSaveException


@pytest.fixture
def config_endpoint(sync_env):
    env = sync_env
    schedule = importlib.import_module('indi_allsky.syncapi_schedule')
    root = Path(__file__).resolve().parents[2]
    config_source = root / 'indi_allsky/config.py'
    config_class = next(node for node in ast.parse(config_source.read_text(encoding='utf-8')).body
                        if isinstance(node, ast.ClassDef) and node.name == 'IndiAllSkyConfig')
    persist = next(node for node in config_class.body if isinstance(node, ast.FunctionDef) and node.name == '_setConfigEntry')
    config_namespace = dict(db=env.db, datetime=datetime, timezone=timezone, __config_level__='test',
                            IndiAllSkyDbConfigTable=env.models.IndiAllSkyDbConfigTable)
    exec(compile(ast.Module(body=[persist], type_ignores=[]), str(config_source), 'exec'), config_namespace)
    control = SimpleNamespace(fail_save=False, before_commit=[])

    class Writer:
        _setConfigEntry = config_namespace['_setConfigEntry']

        def save(self, username, note):
            # A separate reader cannot see either pending change before the
            # actual configuration writer commits its SQLAlchemy session.
            with env.db.engine.connect() as connection:
                control.before_commit.append(connection.execute(select(env.models.IndiAllSkyDbStateTable.value)
                    .where(env.models.IndiAllSkyDbStateTable.key == schedule.SETTINGS_KEY)).scalar())
            if control.fail_save:
                env.db.session.flush()
                raise ConfigSaveException('Configuration save failed')
            return self._setConfigEntry(deepcopy(env.config), SimpleNamespace(id=None), note, False)

    class BaseView(View):
        def __init__(self):
            self.indi_allsky_config = env.config
            self._indi_allsky_config_obj = Writer()
            self._miscDb = SimpleNamespace(setState=lambda key, value: env.sync.set_state(key, value))

    source = root / 'indi_allsky/flask/views.py'
    tree = ast.parse(source.read_text(encoding='utf-8'))
    view = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'AjaxConfigView')
    method = next(node for node in view.body if isinstance(node, ast.FunctionDef) and node.name == 'dispatch_request')
    start = next(i for i, node in enumerate(method.body) if isinstance(node, ast.ImportFrom) and node.module == 'syncapi')
    # Keep real authentication, form rejection and the entire final save path.
    # Camera-field conversion is outside this feature and needs Pi services.
    method.body = method.body[:4] + ast.parse("config_note = 'test save'\nreload_on_save = False").body + method.body[start:]
    namespace = dict(__name__='indi_allsky.flask.views', __package__='indi_allsky.flask', BaseView=BaseView,
        login_required=login_required, current_user=current_user, app=flask.current_app, request=flask.request,
        jsonify=flask.jsonify, db=env.db, constants=env.sync.constants, ConfigSaveException=ConfigSaveException,
        IndiAllskyConfigForm=lambda data: SimpleNamespace(errors={}, validate=lambda: not data.get('invalid_other_field')),
        IndiAllSkyDbTaskQueueTable=env.models.IndiAllSkyDbTaskQueueTable,
        TaskQueueState=env.models.TaskQueueState, TaskQueueQueue=env.models.TaskQueueQueue)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[view], type_ignores=[])), str(source), 'exec'), namespace)
    env.app.add_url_rule('/ajax/config', view_func=namespace['AjaxConfigView'].as_view('config_save'))
    env.app.add_url_rule('/csrf', view_func=lambda: flask.jsonify(token=generate_csrf()))
    login = LoginManager(env.app)
    login.user_loader(lambda user_id: env.db.session.get(env.models.IndiAllSkyDbUserTable, int(user_id)))
    CSRFProtect(env.app)
    user = env.models.IndiAllSkyDbUserTable(username='admin', password='test', email='admin@example.invalid', admin=True)
    env.db.session.add(user)
    env.db.session.commit()
    client = env.app.test_client()
    with client.session_transaction() as session:
        session['_user_id'], session['_fresh'] = str(user.id), True
    headers = {'X-CSRFToken': client.get('/csrf').get_json()['token']}
    return SimpleNamespace(env=env, schedule=schedule, control=control, client=client, headers=headers, user=user)


def options():
    return dict(enabled=True, interval=5, delay=0, upload_limit=256, types=['image', 'rawimage'])


def test_config_save_commits_schedule_and_requests_reload(config_endpoint):
    ctx = config_endpoint
    response = ctx.client.post('/ajax/config', json={'SYNCAPI_SCHEDULE': options()}, headers=ctx.headers)
    assert response.status_code == 200
    assert 'Reloading' in response.get_json()['success-message']
    assert ctx.control.before_commit == [None]
    assert all(ctx.schedule.settings()[key] == value for key, value in options().items())
    assert ctx.env.models.IndiAllSkyDbConfigTable.query.count() == 2
    assert ctx.env.models.IndiAllSkyDbTaskQueueTable.query.one().data['action'] == 'reload'
    assert ctx.env.calls == []


def test_failed_config_save_rolls_back_staged_schedule(config_endpoint):
    ctx = config_endpoint
    ctx.schedule.save_settings(ctx.env.config, dict(options(), enabled=False))
    saved = ctx.schedule.settings()
    ctx.control.fail_save = True
    response = ctx.client.post('/ajax/config', json={'SYNCAPI_SCHEDULE': options()}, headers=ctx.headers)
    assert response.status_code == 400
    assert ctx.schedule.settings() == saved
    assert ctx.env.models.IndiAllSkyDbConfigTable.query.count() == 1
    assert ctx.env.models.IndiAllSkyDbTaskQueueTable.query.count() == 0


@pytest.mark.parametrize('payload', [None, [], dict(options(), enabled='true'), dict(options(), interval=0),
    dict(options(), delay=None), dict(options(), delay=1.5), dict(options(), types=[]),
    dict(options(), upload_limit=-1), dict(options(), upload_limit=True), dict(options(), upload_limit=123),
    dict(options(), upload_limit='256'), dict(options(), upload_limit=None)])
def test_invalid_schedule_rejects_whole_configuration(config_endpoint, payload):
    ctx = config_endpoint
    response = ctx.client.post('/ajax/config', json={'SYNCAPI_SCHEDULE': payload}, headers=ctx.headers)
    assert response.status_code == 400
    assert 'syncapi-run-schedule-controls' in response.get_json()
    assert ctx.control.before_commit == []
    assert ctx.env.models.IndiAllSkyDbConfigTable.query.count() == 1
    assert not ctx.schedule.settings()['enabled']


def test_old_config_pages_and_unchanged_active_schedules_are_preserved(config_endpoint):
    ctx = config_endpoint
    ctx.schedule.save_settings(ctx.env.config, options())
    saved = ctx.schedule.settings()
    task = ctx.env.sync.request_sync(ctx.env.config, ['image'], schedule_revision=saved['revision'])
    for payload in ({}, {'SYNCAPI_SCHEDULE': options()}):
        response = ctx.client.post('/ajax/config', json=payload, headers=ctx.headers)
        assert response.status_code == 200
        assert 'Reloading' not in response.get_json()['success-message']
        assert ctx.schedule.settings() == saved
        assert ctx.env.sync.active_task().id == task.id
    response = ctx.client.post('/ajax/config', json={'SYNCAPI_SCHEDULE': dict(options(), interval=7)}, headers=ctx.headers)
    assert response.status_code == 400
    assert ctx.schedule.settings() == saved


def test_config_auth_csrf_and_other_validation_precede_schedule_changes(config_endpoint):
    ctx = config_endpoint
    payload = {'SYNCAPI_SCHEDULE': options()}
    assert ctx.client.post('/ajax/config', json=payload).status_code == 400
    assert ctx.client.post('/ajax/config', json=dict(payload, invalid_other_field=True), headers=ctx.headers).status_code == 400
    ctx.user.admin = False
    ctx.env.db.session.commit()
    assert ctx.client.post('/ajax/config', json=payload, headers=ctx.headers).status_code == 400
    assert not ctx.schedule.settings()['enabled']
    assert ctx.env.models.IndiAllSkyDbConfigTable.query.count() == 1


def test_schedule_preference_can_be_saved_with_syncapi_disabled(config_endpoint):
    ctx = config_endpoint
    ctx.env.config['SYNCAPI']['ENABLE'] = False
    response = ctx.client.post('/ajax/config', json={'SYNCAPI_SCHEDULE': options()}, headers=ctx.headers)
    assert response.status_code == 200
    assert ctx.schedule.settings()['enabled']
    assert ctx.env.calls == [] and ctx.env.sync.active_task() is None
