import ast
from pathlib import Path

import flask
from flask.views import View
from flask_login import LoginManager, login_required, current_user
from flask_wtf.csrf import CSRFProtect, generate_csrf
from sqlalchemy.orm.exc import NoResultFound
import pytest


@pytest.fixture
def sync_endpoint(sync_env):
    env = sync_env
    class BaseView(View):
        def __init__(self):
            self.indi_allsky_config = env.config
            self.indi_allsky_config_id = 1
            self._miscDb = type('State', (), {'getState': lambda _, key: str(env.sync.get_state(key))})()

    source = Path(__file__).resolve().parents[2] / 'indi_allsky/flask/views.py'
    node = next(item for item in ast.parse(source.read_text(encoding='utf-8')).body
                if isinstance(item, ast.ClassDef) and item.name == 'AjaxSyncApiRunView')
    namespace = dict(__name__='indi_allsky.flask.views', __package__='indi_allsky.flask', BaseView=BaseView,
                     login_required=login_required, current_user=current_user, app=flask.current_app,
                     request=flask.request, jsonify=flask.jsonify, NoResultFound=NoResultFound)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), namespace)
    env.app.add_url_rule('/ajax/syncapi/run', view_func=namespace['AjaxSyncApiRunView'].as_view('sync_run'))
    env.app.add_url_rule('/token', view_func=lambda: flask.jsonify(token=generate_csrf()))
    login = LoginManager(env.app)
    login.user_loader(lambda user_id: env.db.session.get(env.models.IndiAllSkyDbUserTable, int(user_id)))
    CSRFProtect(env.app)
    user = env.models.IndiAllSkyDbUserTable(username='admin', password='test', email='admin@example.invalid', admin=True)
    env.db.session.add(user)
    env.db.session.commit()
    client = env.app.test_client()
    with client.session_transaction() as session:
        session['_user_id'] = str(user.id)
        session['_fresh'] = True
    token = client.get('/token').get_json()['token']
    return env, client, user, {'X-CSRFToken': token}


def test_start_returns_job_without_network(sync_endpoint):
    env, client, _, headers = sync_endpoint
    response = client.post('/ajax/syncapi/run', json={'action': 'start', 'types': ['image']}, headers=headers)
    assert response.status_code == 200
    assert response.get_json()['state'] == 'queued'
    assert env.calls == []
    again = client.post('/ajax/syncapi/run', json={'action': 'start', 'types': ['video']}, headers=headers)
    assert again.get_json()['task_id'] == response.get_json()['task_id']
    result = client.post('/ajax/syncapi/run', json={'action': 'cancel', 'task_id': response.get_json()['task_id']}, headers=headers)
    assert result.get_json()['cancel_requested']


def test_requires_csrf_and_admin(sync_endpoint):
    env, client, user, headers = sync_endpoint
    assert client.post('/ajax/syncapi/run', json={'action': 'start'}).status_code == 400
    user.admin = False
    env.db.session.commit()
    assert client.get('/ajax/syncapi/run').status_code == 403
    assert client.post('/ajax/syncapi/run', json={'action': 'start'}, headers=headers).status_code == 403
    assert env.sync.active_task() is None


@pytest.mark.parametrize('payload', [[], None, {'action': 'start', 'types': []}, {'action': 'start', 'types': ['invalid']}, {'action': 'cancel', 'task_id': True}])
def test_rejects_invalid_requests(sync_endpoint, payload):
    env, client, _, headers = sync_endpoint
    response = client.post('/ajax/syncapi/run', data=flask.json.dumps(payload), content_type='application/json', headers=headers)
    assert response.status_code == 400
    assert env.sync.active_task() is None and env.calls == []


def test_waits_for_configuration_reload(sync_endpoint):
    env, client, _, headers = sync_endpoint
    env.sync.set_state('CONFIG_ID', 0)
    response = client.post('/ajax/syncapi/run', json={'action': 'start'}, headers=headers)
    assert response.status_code == 400
    assert 'reload' in response.get_json()['error']
    assert env.calls == []
