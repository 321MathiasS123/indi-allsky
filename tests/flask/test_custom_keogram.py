"""Exercise the real routes, database selection and worker with isolated services."""

import ast
from datetime import datetime, timedelta
import logging
import os
from pathlib import Path
from types import SimpleNamespace
import shutil
import subprocess

import flask
from flask.views import View
from flask_sqlalchemy import SQLAlchemy
import numpy
from PIL import Image
import pytest

from indi_allsky import customKeogram


ROOT = Path(__file__).resolve().parents[2]
CONFIG = {'KEOGRAM_ANGLE': 0, 'KEOGRAM_H_SCALE': 100, 'KEOGRAM_V_SCALE': 100,
          'KEOGRAM_LABEL': False, 'IMAGE_EXIF_PRIVACY': True,
          'ORB_PROPERTIES': {'RADIUS': 0},
          'IMAGE_FILE_COMPRESSION': {'jpg': 95}}


def load_nodes(path, names, namespace):
    # Load real view code without importing Linux-only camera/D-Bus services.
    tree = ast.parse(path.read_text(encoding='utf-8'))
    tree.body = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name in names]
    exec(compile(tree, str(path), 'exec'), namespace)


@pytest.fixture
def environment(tmp_path):
    # Each test owns its database/session, including under pytest-xdist.
    db = SQLAlchemy()
    path = ROOT / 'indi_allsky/flask/models.py'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    tree.body = [node for node in tree.body if not (
        isinstance(node, ast.ImportFrom) and node.level == 1 and node.module is None)]
    namespace = {'__name__': 'custom_keogram_test_models', 'db': db}
    exec(compile(tree, str(path), 'exec'), namespace)
    app = flask.Flask(__name__, template_folder=str(ROOT / 'indi_allsky/flask/templates'))
    app.config.update(SQLALCHEMY_DATABASE_URI='sqlite://', INDI_ALLSKY_IMAGE_FOLDER=str(tmp_path),
                      SECRET_KEY='test', LOGIN_DISABLED=True)
    db.init_app(app)
    namespace.update(BaseView=View, TemplateView=View, login_required=lambda f: f,
                     app=flask.current_app, current_user=SimpleNamespace(is_admin=True),
                     request=flask.request, jsonify=flask.jsonify, send_file=flask.send_file,
                     url_for=flask.url_for, customKeogram=customKeogram,
                     datetime=datetime, timedelta=timedelta, os=os,
                     asi676mc_calibration=SimpleNamespace(cleanup_expired_sessions=lambda: 0))
    load_nodes(ROOT / 'indi_allsky/flask/views.py', {'AjaxCustomKeogramView', 'CustomKeogramView'}, namespace)
    handler = namespace['AjaxCustomKeogramView']()
    handler.login_disabled = False
    app.add_url_rule('/ajax/custom_keogram', 'indi_allsky.ajax_custom_keogram_view',
                     handler.dispatch_request, methods=['GET', 'POST'])
    path = ROOT / 'indi_allsky/video.py'
    # Keep the worker action intact while omitting process/hardware startup.
    worker_tree = ast.parse(path.read_text(encoding='utf-8'))
    worker_node = next(node for node in worker_tree.body if isinstance(node, ast.ClassDef) and node.name == 'VideoWorker')
    worker_node.bases = []
    worker_node.body = [node for node in worker_node.body if isinstance(node, ast.FunctionDef)
                        and node.name in {'generateCustomKeogram', 'expireData', '_deleteAssets',
                                          '_getFolderFolders', '_getFolderFilesAll'}]
    namespace['logger'] = logging.getLogger('custom-keogram-test')
    exec(compile(ast.Module(body=[worker_node], type_ignores=[]), str(path), 'exec'), namespace)
    worker = namespace['VideoWorker']()
    worker.config, worker.image_dir = CONFIG, tmp_path
    worker.scratch_base_dir = tmp_path / 'scratch'
    with app.app_context():
        db.create_all()
        Camera = namespace['IndiAllSkyDbCameraTable']
        db.session.add_all([Camera(id=1, uuid='one', name='Camera one', lensName='Test', local=True),
                            Camera(id=2, uuid='two', name='Camera two', lensName='Test', local=True)])
        db.session.commit()
        yield SimpleNamespace(app=app, client=app.test_client(), db=db, models=namespace,
                              worker=worker, handler=handler, path=tmp_path)
        db.session.remove()
        db.drop_all()


def add_image(env, hour, *, night=False, camera_id=1, exclude=False, size=(64, 64), missing=False, color=(20, 40, 60)):
    Model = env.models['IndiAllSkyDbImageTable']
    count = Model.query.count()
    path = env.path / ('frame_{0}.png'.format(count))
    if not missing:
        Image.new('RGB', size, color).save(path)
    entry = Model(filename=str(path), createDate=datetime(2026, 10, 1, hour),
                  dayDate=datetime(2026, 10, 1).date(), exposure=1, gain=1, adu=20,
                  night=night, camera_id=camera_id, exclude=exclude)
    env.db.session.add(entry)
    env.db.session.commit()
    return entry


def submit(env, **kwargs):
    return env.client.post('/ajax/custom_keogram', json={
        'camera_id': 1, 'start': '2026-10-01T15:00', 'end': '2026-10-01T22:00', **kwargs})


def status(env, task_id, **kwargs):
    return env.client.get('/ajax/custom_keogram', query_string={'task_id': task_id, 'camera_id': 1, **kwargs})


def run_job(env, response):
    task_id = response.json['task_id']
    task = env.db.session.get(env.models['IndiAllSkyDbTaskQueueTable'], task_id)
    # The worker dispatcher marks tasks running before calling their action.
    task.setRunning()
    env.worker.generateCustomKeogram(task, **task.data['kwargs'])
    return task_id, task


def test_crosses_day_night_uses_exact_range_and_downloads(environment):
    env = environment
    # Deliberately insert out of order, including other cameras/excluded frames.
    add_image(env, 22, night=True, color=(100, 120, 140))
    add_image(env, 14)
    add_image(env, 16, camera_id=2)
    add_image(env, 15)
    add_image(env, 23, night=True)
    add_image(env, 17, exclude=True)
    add_image(env, 20, night=True, missing=True)
    add_image(env, 21, night=True, size=(128, 128))
    response = submit(env)
    assert response.status_code == 202
    task_id = response.json['task_id']
    assert status(env, task_id).json['state'] == 'MANUAL'
    task_id, task = run_job(env, response)
    result = status(env, task_id).json
    assert result['state'] == 'SUCCESS'
    assert (result['frames'], result['skipped'], result['resized']) == (3, 1, 1)
    assert result['first'] == '2026-10-01 15:00:00'
    assert result['last'] == '2026-10-01 22:00:00'
    with Image.open(customKeogram.preview_path(env.path, 1)) as image:
        assert image.size == (3, 64)
    download = env.client.get(result['image_url'] + '&download=1')
    assert download.status_code == 200
    assert download.mimetype == 'image/jpeg'
    assert 'attachment;' in download.headers['Content-Disposition']
    assert '2026-10-01T1500' in download.headers['Content-Disposition']
    download.close()
    assert env.models['IndiAllSkyDbKeogramTable'].query.count() == 0


@pytest.mark.parametrize('payload', [None, [], 1, 'bad', {},
    {'camera_id': True}, {'camera_id': 1, 'start': '2026-10-01T22:00', 'end': '2026-10-01T15:00'},
    {'camera_id': 1, 'start': '2026-10-01T15:00', 'end': '2026-10-01T15:00'},
    {'camera_id': 1, 'start': '2026-10-01T15:00Z', 'end': '2026-10-01T22:00'}])
def test_invalid_requests_never_queue(environment, payload):
    env = environment
    response = env.client.post('/ajax/custom_keogram', json=payload)
    assert response.status_code == 400
    assert env.models['IndiAllSkyDbTaskQueueTable'].query.count() == 0


def test_empty_remote_and_unauthorized_requests(environment):
    env = environment
    assert submit(env).status_code == 400
    add_image(env, 15)
    camera = env.db.session.get(env.models['IndiAllSkyDbCameraTable'], 1)
    camera.local = False
    env.db.session.commit()
    assert 'capture server' in submit(env).json['message']
    env.models['current_user'].is_admin = False
    assert submit(env).status_code == 403
    assert env.models['IndiAllSkyDbTaskQueueTable'].query.count() == 0


def test_all_missing_fails_visibly_without_killing_worker(environment):
    env = environment
    add_image(env, 15, missing=True)
    task_id, task = run_job(env, submit(env))
    assert task.state.name == 'FAILED'
    assert 'No readable local images' in status(env, task_id).json['message']
    assert not customKeogram.output_path(env.path, task_id).exists()


def test_shape_change_reports_failure_and_removes_partial_output(environment):
    env = environment
    add_image(env, 15)
    add_image(env, 20, night=True, size=(128, 64))
    task_id, task = run_job(env, submit(env))
    assert task.state.name == 'FAILED'
    assert 'image shape changed' in task.result
    assert not customKeogram.output_path(env.path, task_id).exists()


@pytest.mark.parametrize('cleanup_denied', [False, True])
def test_write_failure_is_reported_even_if_cleanup_fails(environment, monkeypatch, cleanup_denied, caplog):
    env = environment
    add_image(env, 15)
    response = submit(env)
    outfile = customKeogram.output_path(env.path, response.json['task_id'])

    def interrupted_write(generator, path, camera):
        path.write_bytes(b'partial image')
        raise OSError('Interrupted image write')

    monkeypatch.setattr(customKeogram.KeogramGenerator, 'finalize', interrupted_write)
    if cleanup_denied:
        original_unlink = Path.unlink

        def denied_unlink(path, **kwargs):
            if path == outfile:
                raise PermissionError('File is busy')
            return original_unlink(path, **kwargs)

        monkeypatch.setattr(Path, 'unlink', denied_unlink)
    task_id, task = run_job(env, response)
    assert task.state.name == 'FAILED'
    assert 'Check the worker log' in task.result
    assert outfile.exists() == cleanup_denied
    assert 'Interrupted image write' in caplog.text
    if cleanup_denied:
        assert 'Unable to remove custom keogram' in caplog.text
    # A leftover partial file must never be offered as a successful preview.
    preview = status(env, task_id, image=1)
    assert preview.mimetype == 'application/json'
    assert preview.json['state'] == 'FAILED'
    assert 'image_url' not in preview.json


def test_wrong_camera_other_task_and_expired_preview(environment):
    env = environment
    add_image(env, 15)
    task_id, task = run_job(env, submit(env))
    assert status(env, task_id, camera_id=2).status_code == 404
    assert status(env, 1000).status_code == 404
    path = customKeogram.preview_path(env.path, 1)
    path.unlink()
    assert status(env, task_id).status_code == 410
    task.data = dict(task.data, action='generateVideo')
    env.db.session.commit()
    assert status(env, task_id).status_code == 404


def test_frame_limit_checked_before_queue_and_in_worker(environment, monkeypatch):
    env = environment
    add_image(env, 15)
    add_image(env, 16)
    queued = submit(env)
    monkeypatch.setattr(customKeogram, 'MAX_FRAMES', 1)
    rejected = submit(env)
    assert rejected.status_code == 400
    assert 'up to 1 images' in rejected.json['message']
    _, task = run_job(env, queued)
    assert task.state.name == 'FAILED'


def test_returning_restores_last_result_and_new_success_replaces_it(environment):
    env = environment
    assert status(env, None).json == {'state': 'EMPTY', 'preview': None}
    add_image(env, 15)
    add_image(env, 22, night=True, color=(100, 120, 140))
    first_id, _ = run_job(env, submit(env))
    restored = status(env, None).json
    assert restored['task_id'] == first_id
    assert restored['preview']['start'] == '2026-10-01T15:00'
    assert restored['preview']['end'] == '2026-10-01T22:00'
    assert status(env, None, camera_id=2).json['state'] == 'EMPTY'
    path = customKeogram.preview_path(env.path, 1)
    original = path.read_bytes()
    replacement = submit(env, start='2026-10-01T21:00')
    pending = status(env, None).json
    assert pending['state'] == 'MANUAL'
    assert pending['task_id'] == replacement.json['task_id']
    assert pending['preview']['task_id'] == first_id
    assert path.read_bytes() == original
    second_id, _ = run_job(env, replacement)
    assert path.read_bytes() != original
    assert list(path.parent.iterdir()) == [path]
    restored = status(env, None).json
    assert restored['task_id'] == second_id
    assert restored['preview']['start'] == '2026-10-01T21:00'
    # An old bookmark must not download the new image with the old range's name.
    assert status(env, first_id, image=1, download=1).status_code == 410
    download = status(env, second_id, image=1, download=1)
    assert '2026-10-01T2100' in download.headers['Content-Disposition']
    download.close()


def test_failed_replacement_preserves_the_last_image(environment):
    env = environment
    add_image(env, 15)
    first_id, _ = run_job(env, submit(env))
    path = customKeogram.preview_path(env.path, 1)
    original = path.read_bytes()
    add_image(env, 22, night=True, missing=True)
    failed_id, task = run_job(env, submit(env, start='2026-10-01T21:00'))
    assert task.state.name == 'FAILED'
    assert path.read_bytes() == original
    assert status(env, failed_id).json['preview']['task_id'] == first_id
    assert status(env, None).json['task_id'] == first_id


def test_preview_expires_at_24_hours_and_existing_cleanup_removes_it(environment, monkeypatch):
    env = environment
    add_image(env, 15)
    task_id, _ = run_job(env, submit(env))
    path = customKeogram.preview_path(env.path, 1)
    written = path.stat().st_mtime
    monkeypatch.setattr(customKeogram.time, 'time', lambda: written + 86400 - 1)
    assert status(env, None).json['task_id'] == task_id
    monkeypatch.setattr(customKeogram.time, 'time', lambda: written + 86400)
    assert status(env, None).json['state'] == 'EMPTY'
    assert status(env, task_id, image=1).status_code == 410
    assert path.exists()  # The scheduled worker removes the now-hidden file.
    old = (datetime.now() - timedelta(hours=25)).timestamp()
    os.utime(path, (old, old))
    fresh = customKeogram.preview_path(env.path, 2)
    fresh.write_bytes(b'other camera')
    Task = env.models['IndiAllSkyDbTaskQueueTable']
    cleanup = Task(queue=env.models['TaskQueueQueue'].VIDEO, state=env.models['TaskQueueState'].QUEUED, data={})
    env.db.session.add(cleanup)
    env.db.session.commit()
    env.worker.expireData(cleanup, camera_id=1)
    assert cleanup.state.name == 'SUCCESS'
    assert not path.exists()
    assert fresh.exists()


def test_generator_preserves_chronological_pixels_and_all_frames(tmp_path):
    camera = SimpleNamespace(name='Test', lensName='Test')
    entries = []
    for index in range(30):
        path = tmp_path / ('source_{0}.png'.format(index))
        Image.new('RGB', (64, 64), (index * 8, index * 8, index * 8)).save(path)
        entries.append(SimpleNamespace(getFilesystemPath=lambda p=path: p,
                                       createDate=datetime(2026, 10, 1, 15) + timedelta(minutes=index)))
    updates = []
    result = customKeogram.generate(CONFIG, camera, entries, tmp_path / 'output.jpg', updates.append)
    assert updates[0]['frames'] == 25
    assert result['frames'] == 30
    with Image.open(tmp_path / 'output.jpg') as image:
        assert image.size == (30, 64)
        pixels = numpy.array(image)[32, :, 0]
        assert numpy.all(numpy.abs(pixels.astype(int) - numpy.arange(30) * 8) <= 2)


def test_custom_keogram_javascript():
    # Include executable browser behavior tests in the shared pytest/npm suite.
    node = shutil.which('node')
    assert node is not None, 'Node.js is required to run the custom keogram tests'
    result = subprocess.run([node, '--test', str(Path(__file__).with_name('custom_keogram.test.cjs'))],
                            capture_output=True, text=True, encoding='utf-8', timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
