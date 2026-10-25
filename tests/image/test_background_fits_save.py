"""Exercise real FITS save methods with isolated application services."""
import ast
import copy
from contextlib import nullcontext
from datetime import date, datetime, timedelta, timezone
import gzip
import importlib
import io
import json
import logging
from pathlib import Path
import queue
import shutil
import tempfile
import time
from types import SimpleNamespace
from unittest.mock import Mock

from astropy.io import fits
import numpy
import pytest

from indi_allsky import asi676mc, constants


ROOT = Path(__file__).resolve().parents[2]


class Field:
    def __init__(self, name):
        self.name = name

    def __eq__(self, expected):
        return lambda row: self.value(row) == expected

    def in_(self, expected):
        return lambda row: self.value(row) in expected

    def __getitem__(self, key):
        return Field((self.name, key))

    def as_string(self):
        return self

    def asc(self):
        return self

    def value(self, row):
        if isinstance(self.name, tuple):
            return getattr(row, self.name[0])[self.name[1]]
        return getattr(row, self.name)


class Query:
    def __init__(self, rows, predicates=(), order=()):
        self.rows, self.predicates, self.order = rows, predicates, order

    def filter(self, *predicates):
        return Query(self.rows, self.predicates + predicates, self.order)

    def filter_by(self, **kwargs):
        return self.filter(*(Field(name) == value for name, value in kwargs.items()))

    def order_by(self, *fields):
        return Query(self.rows, self.predicates, fields)

    def all(self):
        rows = [row for row in self.rows
                if all(predicate(row) for predicate in self.predicates)]
        return sorted(rows, key=lambda row: tuple(field.value(row) for field in self.order))

    def first(self):
        return next(iter(self.all()), None)

    def count(self):
        return len(self.all())

    def __iter__(self):
        return iter(self.all())


class Task:
    id = Field('id')
    state = Field('state')
    queue = Field('queue')
    data = Field('data')

    def __init__(self, **kwargs):
        self.id = kwargs.pop('id', 42)
        self.__dict__.update(kwargs)

    def setRunning(self):
        self.state = 'Running'

    def setQueued(self):
        self.state = 'Queued'

    def setSuccess(self, result):
        self.state, self.result = 'Success', result

    def setFailed(self, result):
        self.state, self.result = 'Failed', result


STATES = SimpleNamespace(QUEUED='Queued', RUNNING='Running', SUCCESS='Success',
                         FAILED='Failed', MANUAL='Manual', EXPIRED='Expired')
QUEUES = SimpleNamespace(VIDEO='video_q', UPLOAD='upload_q')


def load_methods(filename, class_name, names, namespace):
    """Avoid importing camera/Flask startup while executing unchanged methods."""
    path = ROOT / 'indi_allsky' / filename
    tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    cls = next(node for node in tree.body
               if isinstance(node, ast.ClassDef) and node.name == class_name)
    methods = [node for node in cls.body
               if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in methods} == set(names)
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(path), 'exec'), namespace)
    return SimpleNamespace(**{name: namespace[name] for name in names})


def frame():
    header = fits.Header()
    header['DATE-OBS'] = '2026-10-25T19:00:00'
    header['BAYERPAT'] = 'RGGB'
    header['OBSERVER'] = 'snapshot before later frame changes'
    data = numpy.arange(64 * 64, dtype=numpy.uint16).reshape(64, 64)
    hdus = fits.HDUList([
        fits.PrimaryHDU(data=data, header=header),
        fits.ImageHDU(data=numpy.arange(12, dtype=numpy.int16).reshape(3, 4),
                      name='EXTRA'),
    ])
    return SimpleNamespace(
        hdulist=hdus, exp_date=datetime(2026, 10, 25, 19, tzinfo=timezone.utc),
        day_date=date(2026, 10, 25), camera_id=1, camera_uuid='test-camera',
        exposure=30., gain=15., binning=1, sqm_value=19.2,
        stars=[(1, 2)], lines=[], kpindex=1., ovation_max=2., smoke_rating=0,
        aurora_mag_bt=3., aurora_mag_gsm_bz=-1., aurora_plasma_density=4.,
        aurora_plasma_speed=350., aurora_plasma_temp=1000.,
        aurora_n_hemi_gw=5., aurora_s_hemi_gw=6.,
        asi676mc_repair_result=None,
    )


def fits_bytes(hdus):
    stream = io.BytesIO()
    hdus.writeto(stream)
    return stream.getvalue()


def image_fixture(tmp_path, compressed=True, video_queue=None, extra_namespace=None):
    session = Mock()
    tasks = []
    session.add.side_effect = tasks.append
    namespace = dict(
        Path=Path, io=io, tempfile=tempfile, shutil=shutil, time=time, json=json,
        logger=logging.getLogger(__name__), asi676mc=asi676mc, constants=constants,
        copy=copy, queue=queue, numpy=numpy, db=SimpleNamespace(session=session),
        TaskQueueState=STATES, TaskQueueQueue=QUEUES,
        IndiAllSkyDbTaskQueueTable=Task,
    )
    if (ROOT / 'indi_allsky/fits_save.py').is_file():
        namespace['fits_save'] = importlib.import_module('indi_allsky.fits_save')
    namespace.update(extra_namespace or {})
    methods = load_methods('image.py', 'ImageWorker', ['write_fit'], namespace)
    database = Mock()
    entry = SimpleNamespace(id=7)
    database.addFitsImage.return_value = entry
    folder = tmp_path / 'fits' / '20261025' / 'night' / '25_19'

    def image_folder(*args):
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    worker = SimpleNamespace(
        config={'IMAGE_SAVE_FITS_COMPRESSED': compressed},
        video_q=video_queue, image_dir=tmp_path,
        next_save_fits_time=0., next_save_fits_offset=600.,
        night_av=[True, False], filename_t='ccd{0:d}_{1:s}.{2:s}',
        _getImageFolder=image_folder, _miscDb=database, _miscUpload=Mock(),
        image_processor=SimpleNamespace(
            astrometric_data={'moon_phase': 61.}, camera_sqm_raw_mag=19.),
        saved_tasks=tasks,
    )
    return methods.write_fit, worker, folder


@pytest.mark.parametrize('compressed', [False, True])
def test_no_queue_retains_lossless_synchronous_save(tmp_path, compressed):
    write, worker, folder = image_fixture(tmp_path, compressed=compressed)
    ref = frame()
    expected = fits_bytes(ref.hdulist)

    write(worker, ref, SimpleNamespace(uuid='test-camera'))

    output = next(folder.glob('*.fit*'))
    actual = gzip.decompress(output.read_bytes()) if compressed else output.read_bytes()
    assert actual == expected
    if compressed:
        assert output.read_bytes()[8] == 2  # gzip's maximum-compression flag
    assert fits_bytes(ref.hdulist) == expected
    metadata = worker._miscDb.addFitsImage.call_args.args[2]
    assert metadata['fileSize'] == output.stat().st_size
    worker._miscUpload.s3_upload_fits.assert_called_once()
    worker._miscUpload.upload_fits_image.assert_called_once()


@pytest.mark.parametrize('reason', ['not_due', 'day_save_disabled'])
def test_ineligible_frame_does_not_schedule_or_save(tmp_path, reason):
    output_queue = Mock()
    write, worker, folder = image_fixture(tmp_path, video_queue=output_queue)
    if reason == 'not_due':
        worker.next_save_fits_time = time.time() + 1000
    else:
        worker.night_av = [False, False]
        worker.config['DAYTIME_CAPTURE_SAVE'] = False

    write(worker, frame(), SimpleNamespace(uuid='test-camera'))

    output_queue.put.assert_not_called()
    worker._miscDb.addFitsImage.assert_not_called()
    assert not folder.exists()


def test_uncompressed_save_does_not_use_video_queue(tmp_path):
    output_queue = Mock()
    write, worker, folder = image_fixture(tmp_path, compressed=False,
                                          video_queue=output_queue)

    write(worker, frame(), SimpleNamespace(uuid='test-camera'))

    assert len(list(folder.glob('*.fit'))) == 1
    assert not worker.saved_tasks
    output_queue.put.assert_not_called()


@pytest.mark.parametrize('pre_dark', [False, True])
def test_deferred_snapshot_survives_later_pixel_header_and_mode_changes(
        tmp_path, monkeypatch, pre_dark):
    fits_save = importlib.import_module('indi_allsky.fits_save')
    output_queue = Mock()
    write, worker, folder = image_fixture(tmp_path, video_queue=output_queue)
    worker.config['IMAGE_SAVE_FITS_PRE_DARK'] = pre_dark
    ref = frame()
    if not pre_dark:
        ref.hdulist[0].data += 1000
    ref.hdulist[0].header['CALSTATE'] = 'raw' if pre_dark else 'calibrated'
    expected = fits_bytes(ref.hdulist)
    monkeypatch.setattr(gzip, 'compress', Mock(side_effect=AssertionError(
        'Compression must run in the background worker')))

    write(worker, ref, SimpleNamespace(uuid='test-camera'))

    assert len(worker.saved_tasks) == 1
    task = worker.saved_tasks[0]
    assert task.data['action'] == 'saveFits'
    job = task.data['kwargs']
    source = fits_save.source_path(tmp_path, job['source'])
    assert source.read_bytes() == expected
    metadata = copy.deepcopy(job['metadata'])
    ref.hdulist[0].data[:] = 0
    ref.hdulist[0].header['OBSERVER'] = 'changed after submission'
    ref.hdulist[1].data[:] = -123
    ref.gain = 200
    worker.night_av[:] = [False, True]
    worker.image_processor.astrometric_data['moon_phase'] = 100
    assert source.read_bytes() == expected
    assert job['metadata'] == metadata
    assert metadata['gain'] == 15
    assert metadata['night'] is True
    assert metadata['data']['moonphase'] == 61
    assert metadata['fileSize'] is None
    assert not list(folder.glob('*.gz'))
    output_queue.put.assert_called_once_with({'task_id': task.id})
    worker._miscDb.addFitsImage.assert_not_called()
    worker._miscUpload.s3_upload_fits.assert_not_called()
    worker._miscUpload.upload_fits_image.assert_not_called()


def test_full_pending_capacity_falls_back_to_lossless_level_nine(tmp_path, monkeypatch):
    fits_save = importlib.import_module('indi_allsky.fits_save')
    monkeypatch.setattr(fits_save, 'pending_count', lambda _: fits_save.MAX_PENDING)
    output_queue = Mock()
    write, worker, folder = image_fixture(tmp_path, video_queue=output_queue)
    ref = frame()
    expected = fits_bytes(ref.hdulist)

    write(worker, ref, SimpleNamespace(uuid='test-camera'))

    output = next(folder.glob('*.fit.gz'))
    assert gzip.decompress(output.read_bytes()) == expected
    assert output.read_bytes()[8] == 2
    output_queue.put.assert_not_called()
    assert not worker.saved_tasks


@pytest.mark.parametrize('failure', ['count', 'create', 'write'])
def test_unavailable_staging_falls_back_without_losing_or_leaking_a_snapshot(
        tmp_path, monkeypatch, failure):
    fits_save = importlib.import_module('indi_allsky.fits_save')
    output_queue = Mock()
    write, worker, folder = image_fixture(tmp_path, video_queue=output_queue)
    ref = frame()
    expected = fits_bytes(ref.hdulist)
    if failure == 'count':
        monkeypatch.setattr(fits_save, 'pending_count', Mock(
            side_effect=PermissionError('staging is not readable')))
    elif failure == 'create':
        original = tempfile.NamedTemporaryFile

        def unavailable(*args, **kwargs):
            if kwargs.get('dir') is not None:
                raise OSError('staging is not writable')
            return original(*args, **kwargs)

        monkeypatch.setattr(tempfile, 'NamedTemporaryFile', unavailable)
    else:
        original = ref.hdulist.writeto

        def interrupted(target, *args, **kwargs):
            if Path(getattr(target, 'name', '')).parent.name == '.fits_pending':
                target.write(b'incomplete snapshot')
                raise OSError('snapshot write interrupted')
            return original(target, *args, **kwargs)

        monkeypatch.setattr(ref.hdulist, 'writeto', interrupted)

    write(worker, ref, SimpleNamespace(uuid='test-camera'))

    output = next(folder.glob('*.fit.gz'))
    assert gzip.decompress(output.read_bytes()) == expected
    assert output.read_bytes()[8] == 2
    assert not list((tmp_path / '.fits_pending').glob('*.fit'))
    output_queue.put.assert_not_called()
    assert not worker.saved_tasks


def test_image_directory_symlink_keeps_queued_paths_relative_to_the_archive(tmp_path):
    fits_save = importlib.import_module('indi_allsky.fits_save')
    actual = tmp_path / 'actual-images'
    actual.mkdir()
    alias = tmp_path / 'configured-images'
    try:
        alias.symlink_to(actual, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip('Directory symlinks require unavailable operating-system privileges')
    write, worker, _ = image_fixture(alias, video_queue=Mock())
    ref = frame()
    expected = fits_bytes(ref.hdulist)

    write(worker, ref, SimpleNamespace(uuid='test-camera'))

    job = worker.saved_tasks[0].data['kwargs']
    assert not Path(job['source']).is_absolute()
    assert not Path(job['filename']).is_absolute()
    assert fits_save.source_path(alias, job['source']).read_bytes() == expected
    assert fits_save.destination_path(alias, job['filename']).is_relative_to(actual.resolve())


def video_fixture(tmp_path, entries=None):
    fits_save = importlib.import_module('indi_allsky.fits_save')
    entries = [] if entries is None else entries
    events = []
    model = SimpleNamespace(filename=Field('filename'), camera_id=Field('camera_id'),
                            query=Query(entries))
    session = Mock()
    namespace = dict(
        Path=Path, time=time, logger=logging.getLogger(__name__), fits_save=fits_save,
        db=SimpleNamespace(session=session), IndiAllSkyDbFitsImageTable=model,
        TaskQueueState=STATES, TaskQueueQueue=QUEUES, json=json,
        timedelta=timedelta, NotificationCategory=SimpleNamespace(WORKER='worker'),
    )
    methods = load_methods('video.py', 'VideoWorker', ['saveFits'], namespace)
    database = Mock()
    uploads = Mock()

    def add(filename, camera_id, metadata):
        destination = tmp_path / filename
        assert destination.is_file(), 'Publish complete file before creating its database row'
        assert metadata['fileSize'] == destination.stat().st_size
        events.append('database')
        entry = SimpleNamespace(id=7, filename=str(filename), camera_id=camera_id)
        entries.append(entry)
        return entry

    def uploaded(*args, **kwargs):
        assert entries, 'Upload must follow successful database publication'
        assert (tmp_path / entries[0].filename).is_file()
        events.append('upload')

    database.addFitsImage.side_effect = add
    uploads.s3_upload_fits.side_effect = uploaded
    uploads.upload_fits_image.side_effect = uploaded
    worker = SimpleNamespace(image_dir=tmp_path, _miscDb=database,
                             _miscUpload=uploads)
    return methods.saveFits, worker, entries, events, session


def pending_job(tmp_path):
    fits_save = importlib.import_module('indi_allsky.fits_save')
    source = fits_save.pending_directory(tmp_path) / 'immutable-source.fit'
    expected = fits_bytes(frame().hdulist)
    source.write_bytes(expected)
    filename = Path('fits/20261025/night/25_19/ccd1_20261025_190000.fit.gz')
    metadata = dict(createDate=1792954800, dayDate='20261025', exposure=30.,
                    gain=15., binmode=1, night=True, fileSize=None,
                    width=64, height=64, camera_uuid='test-camera', data={'moonphase': 61.})
    job = dict(source=str(source.relative_to(tmp_path)), filename=str(filename),
               camera_id=1, metadata=metadata, upload_timestamp=1792954800.5)
    task = Task(queue=QUEUES.VIDEO, state=STATES.RUNNING,
                data={'action': 'saveFits', 'kwargs': copy.deepcopy(job)})
    return task, job, source, tmp_path / filename, expected


def test_video_save_publishes_lossless_file_then_database_and_uploads(tmp_path):
    save, worker, entries, events, _ = video_fixture(tmp_path)
    task, job, source, destination, expected = pending_job(tmp_path)
    metadata = copy.deepcopy(job['metadata'])
    original_success = task.setSuccess

    def success(message):
        assert source.exists(), 'Keep recoverable source until task success is committed'
        events.append('success')
        original_success(message)

    task.setSuccess = success

    save(worker, task, **job)

    assert gzip.decompress(destination.read_bytes()) == expected
    assert destination.read_bytes()[8] == 2
    assert len(entries) == 1
    assert events == ['database', 'upload', 'upload', 'success']
    assert task.state == STATES.SUCCESS
    assert not source.exists()
    assert job['metadata'] == metadata
    upload_call = worker._miscUpload.upload_fits_image.call_args
    assert job['upload_timestamp'] in (*upload_call.args, *upload_call.kwargs.values())


def test_publish_failure_keeps_source_and_does_not_advertise_a_fits(tmp_path, monkeypatch):
    fits_save = importlib.import_module('indi_allsky.fits_save')
    save, worker, entries, _, session = video_fixture(tmp_path)
    task, job, source, destination, expected = pending_job(tmp_path)
    monkeypatch.setattr(fits_save, 'publish', Mock(side_effect=OSError('disk write failed')))

    save(worker, task, **job)

    assert source.read_bytes() == expected
    assert not destination.exists()
    assert entries == []
    assert task.state == STATES.FAILED
    session.rollback.assert_called_once()
    worker._miscUpload.s3_upload_fits.assert_not_called()
    worker._miscUpload.upload_fits_image.assert_not_called()


@pytest.mark.parametrize('failure_stage', ['database', 'upload'])
def test_retry_after_publication_reuses_file_and_avoids_duplicate_rows(tmp_path, failure_stage):
    save, worker, entries, _, session = video_fixture(tmp_path)
    task, job, source, destination, expected = pending_job(tmp_path)
    add = worker._miscDb.addFitsImage.side_effect
    if failure_stage == 'database':
        worker._miscDb.addFitsImage.side_effect = OSError('database unavailable')
    else:
        worker._miscUpload.upload_fits_image.side_effect = OSError('upload queue unavailable')

    save(worker, task, **job)

    assert task.state == STATES.FAILED
    assert source.read_bytes() == expected
    assert len(entries) == (0 if failure_stage == 'database' else 1)
    published = destination.read_bytes()
    mtime = destination.stat().st_mtime_ns
    worker._miscUpload.upload_fits_image.side_effect = None
    worker._miscDb.addFitsImage.side_effect = add
    task.state = STATES.RUNNING

    save(worker, task, **job)

    assert task.state == STATES.SUCCESS
    assert len(entries) == 1
    assert worker._miscDb.addFitsImage.call_count == (2 if failure_stage == 'database' else 1)
    assert destination.read_bytes() == published
    assert destination.stat().st_mtime_ns == mtime
    assert gzip.decompress(published) == expected
    assert not source.exists()
    session.rollback.assert_called_once()


def test_queue_dispatch_failure_keeps_snapshot_for_redispatch(tmp_path):
    fits_save = importlib.import_module('indi_allsky.fits_save')
    output_queue = Mock()
    output_queue.put.side_effect = OSError('queue connection lost')
    write, worker, _ = image_fixture(tmp_path, video_queue=output_queue)
    ref = frame()
    expected = fits_bytes(ref.hdulist)

    write(worker, ref, SimpleNamespace(uuid='test-camera'))

    task = worker.saved_tasks[0]
    assert task.state == STATES.MANUAL
    assert fits_save.source_path(tmp_path, task.data['kwargs']['source']).read_bytes() == expected
    worker._miscDb.addFitsImage.assert_not_called()


def test_video_fifo_finishes_queued_fits_before_period_job(tmp_path):
    video_queue = queue.Queue()
    write, image, _ = image_fixture(tmp_path, video_queue=video_queue)
    ref = frame()
    expected = fits_bytes(ref.hdulist)
    write(image, ref, SimpleNamespace(uuid='test-camera'))
    task = image.saved_tasks[0]
    job = task.data['kwargs']
    save, video, entries, events, _ = video_fixture(tmp_path)
    video_queue.put({'task_id': 99})
    video_queue.put({'stop': True})

    def process(item):
        if item['task_id'] == task.id:
            save(video, task, **job)
        else:
            assert task.state == STATES.SUCCESS
            assert len(entries) == 1
            assert gzip.decompress((tmp_path / job['filename']).read_bytes()) == expected
            events.append('period-end job')

    methods = load_methods('video.py', 'VideoWorker', ['saferun'], dict(
        queue=queue, logger=logging.getLogger(__name__),
        app=SimpleNamespace(app_context=nullcontext),
    ))
    video.video_q = video_queue
    video._shutdown = False
    video.processTask = process

    methods.saferun(video)

    assert events[-1] == 'period-end job'
    assert video_queue.empty()


@pytest.mark.parametrize('source_already_cleaned', [False, True])
def test_restart_resumes_only_fits_and_cleans_successful_staging(
        tmp_path, source_already_cleaned):
    fits_save = importlib.import_module('indi_allsky.fits_save')
    completed, job, source, destination, _ = pending_job(tmp_path)
    fits_save.publish(source, destination)
    if source_already_cleaned:
        source.unlink()
    completed.id, completed.state = 2, STATES.SUCCESS
    queued = Task(id=3, queue=QUEUES.VIDEO, state=STATES.QUEUED,
                  data={'action': 'saveFits', 'kwargs': job})
    running = Task(id=1, queue=QUEUES.VIDEO, state=STATES.RUNNING,
                   data={'action': 'saveFits', 'kwargs': job})
    unrelated = Task(id=4, queue=QUEUES.VIDEO, state=STATES.QUEUED,
                     data={'action': 'generateVideo'})
    failed = Task(id=5, queue=QUEUES.VIDEO, state=STATES.FAILED,
                  data={'action': 'saveFits', 'kwargs': job})
    rows = [queued, unrelated, completed, failed, running]
    model = SimpleNamespace(query=Query(rows), queue=Task.queue, state=Task.state,
                            id=Task.id, data=Task.data)
    logger = Mock()
    methods = load_methods('video.py', 'VideoWorker', ['_resumeFitsSaves'], dict(
        fits_save=fits_save, TaskQueueState=STATES, TaskQueueQueue=QUEUES,
        IndiAllSkyDbTaskQueueTable=model, logger=logger,
    ))
    resumed = []
    video = SimpleNamespace(image_dir=tmp_path,
                             processTask=lambda item: resumed.append(item['task_id']))

    methods._resumeFitsSaves(video)

    assert resumed == [1, 3]
    assert running.state == queued.state == STATES.QUEUED
    assert completed.state == STATES.SUCCESS
    assert unrelated.state == STATES.QUEUED
    assert failed.state == STATES.FAILED
    assert not source.exists()
    logger.exception.assert_not_called()


def test_daemon_restart_retains_recoverable_fits_jobs_and_expires_other_jobs():
    rows = [Task(id=index, queue=QUEUES.VIDEO, state=state,
                 data={'action': action})
            for index, (action, state) in enumerate([
                ('saveFits', STATES.QUEUED), ('saveFits', STATES.RUNNING),
                ('saveFits', STATES.MANUAL), ('saveFits', STATES.SUCCESS),
                ('saveFits', STATES.FAILED), ('generateVideo', STATES.QUEUED),
            ])]
    model = SimpleNamespace(query=Query(rows), state=Task.state)
    session = Mock()
    methods = load_methods('allsky.py', 'IndiAllSky', ['_expireOrphanedTasks'], dict(
        TaskQueueState=STATES, TaskQueueQueue=QUEUES,
        IndiAllSkyDbTaskQueueTable=model, logger=Mock(),
        db=SimpleNamespace(session=session),
    ))

    methods._expireOrphanedTasks(SimpleNamespace())

    assert [row.state for row in rows] == [
        STATES.QUEUED, STATES.QUEUED, STATES.QUEUED, STATES.SUCCESS,
        STATES.FAILED, STATES.EXPIRED,
    ]
    session.commit.assert_called_once()


def test_supervisor_supplies_existing_video_fifo_to_image_worker():
    tree = ast.parse((ROOT / 'indi_allsky/allsky.py').read_text(encoding='utf-8'))
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name) and node.func.id == 'ImageWorker']
    assert len(calls) == 1
    keyword = next(item for item in calls[0].keywords if item.arg == 'video_q')
    assert isinstance(keyword.value, ast.Attribute)
    assert isinstance(keyword.value.value, ast.Name)
    assert keyword.value.value.id == 'self' and keyword.value.attr == 'video_q'


def test_truncated_existing_archive_is_preserved_and_not_accepted_as_a_retry(tmp_path):
    fits_save = importlib.import_module('indi_allsky.fits_save')
    _, _, source, destination, expected = pending_job(tmp_path)
    destination.parent.mkdir(parents=True)
    truncated = gzip.compress(expected)[:-7]
    destination.write_bytes(truncated)

    with pytest.raises(FileExistsError):
        fits_save.publish(source, destination)

    assert destination.read_bytes() == truncated
    assert source.read_bytes() == expected
    assert not list(destination.parent.glob('.fits-save-*.tmp'))


@pytest.mark.parametrize('winner_matches', [False, True])
def test_atomic_publication_race_never_overwrites_the_winner_and_cleans_temporary(
        tmp_path, monkeypatch, winner_matches):
    fits_save = importlib.import_module('indi_allsky.fits_save')
    _, _, source, destination, expected = pending_job(tmp_path)
    winner = gzip.compress(expected if winner_matches else b'another saved frame')

    def concurrent_winner(staged, target):
        assert gzip.decompress(Path(staged).read_bytes()) == expected
        Path(target).write_bytes(winner)
        raise FileExistsError('another publisher won')

    monkeypatch.setattr(fits_save.os, 'link', concurrent_winner)
    if winner_matches:
        assert fits_save.publish(source, destination) == len(winner)
    else:
        with pytest.raises(FileExistsError):
            fits_save.publish(source, destination)

    assert destination.read_bytes() == winner
    assert source.read_bytes() == expected
    assert not list(destination.parent.glob('.fits-save-*.tmp'))


@pytest.mark.parametrize('deferred', [False, True])
def test_fits_upload_name_uses_captured_enqueue_clock_across_midnight(deferred):
    queued_time = datetime(2026, 10, 25, 23, 59)
    worker_time = datetime(2026, 10, 26, 0, 35)

    class Clock:
        @staticmethod
        def now():
            return worker_time

        @staticmethod
        def fromtimestamp(value):
            return datetime.fromtimestamp(value)

    rows = []
    session = Mock()
    session.add.side_effect = rows.append
    methods = load_methods('miscUpload.py', 'miscUpload', ['upload_fits_image'], dict(
        Path=Path, datetime=Clock, constants=constants,
        TaskQueueState=STATES, TaskQueueQueue=QUEUES,
        IndiAllSkyDbTaskQueueTable=Task, db=SimpleNamespace(session=session),
    ))
    uploader = SimpleNamespace(upload_q=Mock(), config={'FILETRANSFER': {
        'UPLOAD_FITS': True,
        'REMOTE_FITS_FOLDER': 'fits/{timestamp:%Y%m%d}/{day_date:%Y%m%d}/{tod}/{camera_uuid}',
        'REMOTE_FITS_NAME': '{ts:%Y%m%d_%H%M%S}_ccd{camera_id}.{ext}',
    }})
    entry = SimpleNamespace(id=7, filename='ccd1_20261025_235859.fit.gz',
                            dayDate=date(2026, 10, 25), night=True,
                            camera=SimpleNamespace(id=1, uuid='test-camera'))

    methods.upload_fits_image(uploader, entry,
                              timestamp=queued_time.timestamp() if deferred else None)

    timestamp = queued_time if deferred else worker_time
    expected = Path('fits') / timestamp.strftime('%Y%m%d') / '20261025' / 'night' / 'test-camera'
    expected /= timestamp.strftime('%Y%m%d_%H%M%S') + '_ccd1.gz'
    assert len(rows) == 1
    assert Path(rows[0].data['remote_file']) == expected
    uploader.upload_q.put.assert_called_once_with({'task_id': rows[0].id})
