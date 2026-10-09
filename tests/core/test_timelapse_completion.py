"""Exercise automatic scheduling and completion with a real SQLite task queue.

Load worker methods without the Linux camera/DBus imports, as these tests also
run on Windows. Encoding is replaced; task persistence and file checks are real.
"""

import ast
from datetime import datetime, timedelta
import enum
import json
import logging
import io
from pathlib import Path
from queue import Queue
import runpy
import socket
import ssl
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock

from flask import Flask
from flask_sqlalchemy import SQLAlchemy
import pytest
from sqlalchemy.orm.exc import NoResultFound

from indi_allsky import constants
from indi_allsky.timelapse_completion import completion_payload, record_media


ROOT = Path(__file__).resolve().parents[2]


def load_class(path, name, namespace, methods=None):
    tree = ast.parse((ROOT / path).read_text(encoding='utf-8'))
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
    if methods is not None:
        node.bases = []
        node.body = [n for n in node.body if isinstance(n, ast.FunctionDef) and n.name in methods]
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(ROOT / path), 'exec'), namespace)
    return namespace[name]


@pytest.fixture
def runtime(tmp_path):
    app = Flask(__name__)
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + str(tmp_path / 'tasks.sqlite')
    db = SQLAlchemy(app)
    namespace = dict(db=db, enum=enum, constants=constants, json=json,
                     logger=logging.getLogger(__name__), NoResultFound=NoResultFound,
                     completion_payload=completion_payload)
    for name in ('TaskQueueState', 'TaskQueueQueue', 'IndiAllSkyDbTaskQueueTable'):
        load_class('indi_allsky/flask/models.py', name, namespace)

    class Camera(db.Model):
        id = db.Column(db.Integer, primary_key=True)

    namespace['IndiAllSkyDbCameraTable'] = Camera
    capture_type = load_class('indi_allsky/capture.py', 'CaptureWorker', namespace, {
        '_generateDayKeogram', '_generateNightKeogram',
        '_generateDayTimelapse', '_generateNightTimelapse',
        '_queueVideoTask', '_queuePeriodEnd',
    })
    worker_type = load_class('indi_allsky/video.py', 'VideoWorker', namespace, {'processTask'})
    upload_type = load_class('indi_allsky/miscUpload.py', 'miscUpload', namespace, {'mqtt_publish_event'})
    capture, worker, upload = capture_type(), worker_type(), upload_type()
    config = {'MQTTPUBLISH': {'ENABLE': True}, 'FISH2PANO': {'ENABLE': True}}
    capture.config = worker.config = upload.config = config
    capture.video_q = Queue()
    upload.upload_q = Queue()
    worker._miscUpload = upload
    task_type = namespace['IndiAllSkyDbTaskQueueTable']

    def encode(task, **kwargs):
        action = task.data['action']
        names = {'generateVideo': ['timelapse'], 'generatePanoramaVideo': ['panorama'],
                 'generateKeogramStarTrails': ['keogram', 'startrail', 'startrail_timelapse']}[action]
        outputs = {}
        for name in names:
            path = tmp_path / (str(task.id) + '-' + name)
            path.write_bytes(b'generated-media')
            skipped = name.startswith('startrail') and not kwargs['night']
            outputs[name] = (None if skipped else SimpleNamespace(success=True), path)
        record_media(task, outputs)
        task.setSuccess('Generated')

    for action in ('generateVideo', 'generatePanoramaVideo', 'generateKeogramStarTrails'):
        setattr(worker, action, encode)

    with app.app_context():
        db.create_all()
        db.session.add(Camera(id=7))
        db.session.commit()
        yield SimpleNamespace(db=db, capture=capture, worker=worker, upload=upload,
                              Task=task_type, states=namespace['TaskQueueState'],
                              queues=namespace['TaskQueueQueue'], tmp_path=tmp_path)
        db.session.remove()
        db.drop_all()


def schedule(runtime, night=True):
    period = 'Night' if night else 'Day'
    keogram = getattr(runtime.capture, '_generate' + period + 'Keogram')('20261008', 7)
    getattr(runtime.capture, '_generate' + period + 'Timelapse')(
        '20261008', 7, completion_task_ids=[keogram],
    )
    jobs = []
    while not runtime.capture.video_q.empty():
        jobs.append(runtime.capture.video_q.get_nowait())
    return jobs


def event(runtime):
    item = runtime.upload.upload_q.get_nowait()
    task = runtime.db.session.get(runtime.Task, item['task_id'])
    assert task.data['mqtt_event'] is True
    assert not task.data.get('local_file')
    return json.loads(task.data['metadata']['timelapse/complete'])


@pytest.mark.parametrize('night', [False, True])
@pytest.mark.parametrize('panorama', [False, True])
def test_one_event_only_after_all_local_jobs(runtime, night, panorama):
    runtime.capture.config['FISH2PANO']['ENABLE'] = panorama
    jobs = schedule(runtime, night)
    assert len(jobs) == (3 if panorama else 2)
    for job in jobs[:-1]:
        runtime.worker.processTask(job)
        assert runtime.upload.upload_q.empty()
    # Persisted metadata must survive worker/session replacement.
    runtime.db.session.remove()
    runtime.worker.processTask(jobs[-1])
    payload = event(runtime)
    assert payload['status'] == 'success'
    assert payload['camera_id'] == 7
    assert payload['date'] == '2026-10-08'
    assert payload['period'] == ('night' if night else 'day')
    assert ('panorama' in payload['outputs']) is panorama
    assert payload['outputs']['startrail_timelapse'] == ('success' if night else 'skipped')
    assert datetime.fromisoformat(payload['completed_at']).tzinfo is not None
    runtime.worker.processTask(jobs[-1])
    assert runtime.upload.upload_q.empty()


@pytest.mark.parametrize('failure', ['encoder', 'missing_file', 'empty_file', 'startrail', 'missing_task', 'expired'])
def test_final_panorama_success_does_not_hide_earlier_failure(runtime, failure):
    jobs = schedule(runtime)
    for job in jobs[:-1]:
        runtime.worker.processTask(job)
    first = runtime.db.session.get(runtime.Task, jobs[0]['task_id'])
    if failure == 'encoder':
        first.setFailed('Encoder failed')
    elif failure == 'expired':
        first.setExpired()
    elif failure == 'missing_task':
        runtime.db.session.delete(first)
        runtime.db.session.commit()
    elif failure == 'startrail':
        media = dict(first.data['local_media'])
        media['startrail_timelapse'] = dict(media['startrail_timelapse'], status='failed')
        first.data = dict(first.data, local_media=media)
        runtime.db.session.commit()
    else:
        path = Path(first.data['local_media']['keogram']['path'])
        path.unlink() if failure == 'missing_file' else path.write_bytes(b'')
    runtime.worker.processTask(jobs[-1])
    payload = event(runtime)
    assert payload['status'] == 'failed'
    assert payload['outputs']['panorama'] == 'success'


def test_failed_final_job_still_reports_completion(runtime):
    jobs = schedule(runtime)
    runtime.worker.generatePanoramaVideo = lambda task, **kwargs: task.setFailed('Encoder failed')
    for job in jobs:
        runtime.worker.processTask(job)
    payload = event(runtime)
    assert payload['status'] == 'failed'
    assert jobs[-1]['task_id'] in payload['failed_task_ids']


def test_startrail_below_minimum_is_skipped(runtime):
    jobs = schedule(runtime)
    runtime.worker.processTask(jobs[0])
    task = runtime.db.session.get(runtime.Task, jobs[0]['task_id'])
    media = dict(task.data['local_media'])
    media['startrail_timelapse'] = {'status': 'skipped', 'path': None}
    task.data = dict(task.data, local_media=media)
    runtime.db.session.commit()
    for job in jobs[1:]:
        runtime.worker.processTask(job)
    assert event(runtime)['status'] == 'success'


def test_mqtt_disabled_and_manual_generation_do_not_notify(runtime):
    runtime.capture.config['MQTTPUBLISH']['ENABLE'] = False
    for job in schedule(runtime):
        runtime.worker.processTask(job)
    assert runtime.upload.upload_q.empty()
    runtime.capture.config['MQTTPUBLISH']['ENABLE'] = True
    runtime.capture._generateNightTimelapse('20261009', 7)
    while not runtime.capture.video_q.empty():
        runtime.worker.processTask(runtime.capture.video_q.get_nowait())
    assert runtime.upload.upload_q.empty()


@pytest.mark.parametrize('setting', ['TIMELAPSE_ENABLE', 'DAYTIME_TIMELAPSE'])
def test_disabled_generation_does_not_schedule_completion(runtime, setting):
    runtime.capture.config[setting] = False
    assert schedule(runtime, night=False) == []


@pytest.mark.parametrize('night', [False, True])
def test_period_barrier_keeps_completion_behind_pending_images(runtime, night):
    if not hasattr(runtime.capture, '_queuePeriodEnd'):
        pytest.skip('Capture-period barriers are not present on this branch')
    capture = runtime.capture
    capture.camera_id = 7
    capture._expireData = Mock()
    capture._uploadAllskyEndOfNight = Mock()
    capture._period_queue = Mock()
    capture._queuePeriodEnd(datetime(2026, 10, 8).date(), night)
    assert capture.video_q.empty()
    assert runtime.upload.upload_q.empty()
    jobs = capture._period_queue.end_period.call_args.args[3]
    assert len(jobs) == 3
    for job in jobs[:-1]:
        runtime.worker.processTask(job)
        assert runtime.upload.upload_q.empty()
    runtime.worker.processTask(jobs[-1])
    assert event(runtime)['status'] == 'success'


def test_notification_failure_preserves_generation_success(runtime):
    runtime.upload.mqtt_publish_event = Mock(side_effect=RuntimeError('queue unavailable'))
    jobs = schedule(runtime)
    for job in jobs:
        runtime.worker.processTask(job)
    assert runtime.db.session.get(runtime.Task, jobs[-1]['task_id']).state == runtime.states.SUCCESS


def test_record_media_keeps_failed_encoder_even_when_partial_file_exists(runtime):
    jobs = schedule(runtime)
    task = runtime.db.session.get(runtime.Task, jobs[0]['task_id'])
    path = runtime.tmp_path / 'partial.mp4'
    path.write_bytes(b'partial')
    record_media(task, {'startrail_timelapse': (SimpleNamespace(success=False), path)})
    task.setSuccess('Combined job completed')
    assert completion_payload(task, {task.id: task})['status'] == 'failed'


def mqtt_client_type():
    namespace = dict(Path=Path, io=io, socket=socket, ssl=ssl, time=time,
                     logger=logging.getLogger(__name__))
    namespace.update(runpy.run_path(str(ROOT / 'indi_allsky/filetransfer/exceptions.py')))
    namespace['GenericFileTransfer'] = runpy.run_path(
        str(ROOT / 'indi_allsky/filetransfer/generic.py')
    )['GenericFileTransfer']
    return load_class('indi_allsky/filetransfer/paho_mqtt.py', 'paho_mqtt', namespace)


def receive_packet(connection):
    header = connection.recv(1)[0]
    length, multiplier = 0, 1
    while True:
        byte = connection.recv(1)[0]
        length += (byte & 127) * multiplier
        if not byte & 128:
            break
        multiplier *= 128
    body = b''
    while len(body) < length:
        chunk = connection.recv(length - len(body))
        if not chunk:
            raise RuntimeError('MQTT connection closed before complete packet')
        body += chunk
    return header, body


@pytest.mark.parametrize('protocol', ['MQTTv311', 'MQTTv5'])
@pytest.mark.parametrize('publish_image', [False, True])
def test_event_through_real_paho_tcp_publish(runtime, protocol, publish_image):
    """Receive actual MQTT wire packets on loopback, with no external broker."""
    for job in schedule(runtime):
        runtime.worker.processTask(job)
    upload_job = runtime.upload.upload_q.get_nowait()
    mqtt_type = mqtt_client_type()
    exceptions = SimpleNamespace(**mqtt_type.put.__globals__)
    models = SimpleNamespace(IndiAllSkyDbTaskQueueTable=runtime.Task,
                             TaskQueueState=runtime.states, TaskQueueQueue=runtime.queues)
    namespace = dict(__name__='indi_allsky.uploader', __package__='indi_allsky',
                     models=models, db=runtime.db, constants=constants, Path=Path,
                     NoResultFound=NoResultFound, time=time, timedelta=timedelta,
                     logger=logging.getLogger(__name__),
                     filetransfer=SimpleNamespace(paho_mqtt=mqtt_type, exceptions=exceptions))
    uploader_type = load_class('indi_allsky/uploader.py', 'FileUploader', namespace,
                               {'processUpload', 'cleanup'})
    uploader = uploader_type()
    uploader._miscDb = Mock()

    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        listener.listen(1)
        listener.settimeout(5)
        uploader.config = {'MQTTPUBLISH': {
            'TRANSPORT': 'tcp', 'PROTOCOL': protocol, 'HOST': '127.0.0.1',
            'PORT': listener.getsockname()[1], 'USERNAME': '', 'PASSWORD': '',
            'TLS': False, 'BASE_TOPIC': 'test/allsky', 'QOS': 1,
            'PUBLISH_IMAGE': publish_image,
        }}

        def broker():
            connection, _ = listener.accept()
            with connection:
                connection.settimeout(5)
                assert receive_packet(connection)[0] == 0x10  # CONNECT
                connection.sendall(b'\x20\x03\x00\x00\x00' if protocol == 'MQTTv5'
                                   else b'\x20\x02\x00\x00')
                header, body = receive_packet(connection)
                topic_length = int.from_bytes(body[:2], 'big')
                topic = body[2:2 + topic_length].decode()
                position = 2 + topic_length
                packet_id = body[position:position + 2]
                position += 2
                if protocol == 'MQTTv5':
                    assert body[position] == 0  # no publish properties
                    position += 1
                connection.sendall(b'\x40\x02' + packet_id)
                assert receive_packet(connection)[0] == 0xe0  # DISCONNECT
                return header, topic, json.loads(body[position:])

        with ThreadPoolExecutor(max_workers=1) as executor:
            received = executor.submit(broker)
            uploader.processUpload(upload_job)
            header, topic, payload = received.result(timeout=10)
    assert header == 0x32  # PUBLISH, QoS 1, retain OFF
    assert topic == 'test/allsky/timelapse/complete'
    assert payload['status'] == 'success'
    assert runtime.db.session.get(runtime.Task, upload_job['task_id']).state == runtime.states.SUCCESS
    uploader._miscDb.addNotification.assert_not_called()


def test_existing_images_and_sensors_remain_retained(runtime, monkeypatch):
    import paho.mqtt.publish

    publish = Mock()
    monkeypatch.setattr(paho.mqtt.publish, 'multiple', publish)
    path = runtime.tmp_path / 'latest.jpg'
    path.write_bytes(b'unchanged image payload')
    client = mqtt_client_type()({})
    client.connect(transport='tcp', protocol='MQTTv311', hostname='localhost',
                   username='', password='', tls=False, cert_bypass=False)
    client.put(local_file=path, base_topic='indi-allsky', qos=0,
               mq_data={'stars': 123}, image_topic='latest', publish_image=True)
    messages = publish.call_args.args[0]
    assert messages == [
        {'topic': 'indi-allsky/latest', 'payload': b'unchanged image payload', 'qos': 0, 'retain': True},
        {'topic': 'indi-allsky/stars', 'payload': 123, 'qos': 0, 'retain': True},
    ]
