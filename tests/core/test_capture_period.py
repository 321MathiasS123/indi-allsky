"""Exercise delivery/render ordering without camera, Flask or database services."""
import ast
import ctypes
from contextlib import nullcontext
from datetime import date, datetime, timedelta
import logging
import multiprocessing
import os
from pathlib import Path
import queue
from threading import Event, Thread
from types import SimpleNamespace

import pytest

from indi_allsky.capture_period import (
    CapturePeriodQueue, CaptureSequenceTracker, FrameTemperatureValues, failure_key, frame_mode,
    read_inflight, set_inflight,
)


ROOT = Path(__file__).resolve().parents[2]
DAY = date(2026, 10, 8)
PERIOD = '1:2026-10-08:1'
STREAM = 'a' * 32


def methods(filename, classname, names, namespace):
    """Run actual worker methods while replacing only their external services."""
    tree = ast.parse((ROOT / 'indi_allsky' / filename).read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == classname)
    selected = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in selected} == set(names)
    selected_class = ast.ClassDef(name='Worker', bases=[], keywords=[], body=selected, decorator_list=[])
    module = ast.fix_missing_locations(ast.Module(body=[selected_class], type_ignores=[]))
    exec(compile(module, str(ROOT / 'indi_allsky' / filename), 'exec'), namespace)
    return namespace['Worker']


def test_delayed_callback_keeps_mode_date_and_orders_boundary():
    outgoing = queue.Queue()
    mode = [1, 0]
    temperature = [12.0]
    producer = CapturePeriodQueue(outgoing, temperature=lambda: temperature[0])
    producer.begin(1, mode, DAY, 20)
    mode[:] = [0, 1]
    producer.end_period(1, DAY.isoformat(), True, [{'task_id': 7}])
    assert outgoing.empty()
    assert producer.waiting
    with pytest.raises(RuntimeError, match='Previous camera frame'):
        producer.begin(1, mode, DAY, 10)
    temperature[0] = 12.5
    source = {'filename': 'last-night.fit', 'exp_time': 42}
    producer.put(source)
    temperature[0] = 18
    frame, marker = outgoing.get_nowait(), outgoing.get_nowait()
    assert frame['capture_mode'] == (1, 0)
    assert frame['capture_day_date'] == DAY.isoformat()
    assert frame['capture_period'] == 20
    assert frame['capture_temperature'] == 12.5
    assert frame['capture_sequence'] == 1
    assert frame['capture_period_id'] == PERIOD
    assert marker == {'period_end': {'period_id': PERIOD, 'tasks': [{'task_id': 7}],
                                    'stream_id': frame['capture_stream_id'], 'last_sequence': 1}}
    assert source == {'filename': 'last-night.fit', 'exp_time': 42}
    assert not producer.waiting


def test_failed_delivery_never_releases_marker_or_allows_next_exposure():
    class BrokenQueue:
        def put(self, item):
            raise OSError('pipe closed')
    producer = CapturePeriodQueue(BrokenQueue())
    producer.begin(1, (1, 0), DAY, 20)
    producer.end_period(1, DAY.isoformat(), True, [{'task_id': 7}])
    with pytest.raises(OSError):
        producer.put({'filename': 'frame.fit'})
    assert producer.waiting


def test_passive_camera_accepts_unsolicited_frames_without_claiming_exposure_ownership():
    outgoing = queue.Queue()
    producer = CapturePeriodQueue(outgoing, passive=True)
    first = {'filename': 'before-first-command.fit'}
    producer.put(first)
    producer.begin(1, (1, 0), DAY, 20)
    producer.begin(1, (0, 0), DAY, 5)
    producer.put({'filename': 'external-1.fit'})
    producer.put({'filename': 'external-2.fit'})
    producer.end_period(1, DAY.isoformat(), True, [{'task_id': 7}])
    producer.put({'filename': 'arrived-after-boundary.fit'})
    assert not producer.waiting
    assert outgoing.get_nowait() is first
    assert outgoing.get_nowait() == {'filename': 'external-1.fit'}
    assert outgoing.get_nowait() == {'filename': 'external-2.fit'}
    marker = outgoing.get_nowait()['period_end']
    assert marker == {'period_id': PERIOD, 'tasks': [{'task_id': 7}]}
    assert not CaptureSequenceTracker().check(marker, marker=True)
    assert outgoing.get_nowait() == {'filename': 'arrived-after-boundary.fit'}


def test_controlled_camera_still_rejects_unsolicited_or_duplicate_callbacks():
    outgoing = queue.Queue()
    producer = CapturePeriodQueue(outgoing)
    with pytest.raises(RuntimeError, match='without an active exposure'):
        producer.put({'filename': 'unsolicited.fit'})
    producer.begin(1, (1, 0), DAY, 20)
    producer.put({'filename': 'expected.fit'})
    with pytest.raises(RuntimeError, match='without an active exposure'):
        producer.put({'filename': 'duplicate.fit'})
    assert outgoing.qsize() == 1


def test_callback_and_transition_writes_cannot_overtake_each_other():
    entered, proceed = Event(), Event()
    items = []
    class SlowQueue:
        def put(self, item):
            if 'filename' in item:
                entered.set()
                assert proceed.wait(3)
            items.append(item)
    producer = CapturePeriodQueue(SlowQueue())
    producer.begin(1, (1, 0), DAY, 20)
    callback = Thread(target=producer.put, args=({'filename': 'last.fit'},))
    callback.start()
    assert entered.wait(3)
    closing = Thread(target=producer.end_period, args=(1, DAY.isoformat(), True, [{'task_id': 8}]))
    closing.start()
    proceed.set()
    callback.join(3)
    closing.join(3)
    assert not callback.is_alive() and not closing.is_alive()
    assert ['frame' if 'filename' in item else 'end' for item in items] == ['frame', 'end']


@pytest.mark.parametrize('night,extra_expire,expected', [
    (True, False, ['expire', 'night-keogram', 'night-video', 'end-night']),
    (False, False, ['expire', 'day-keogram', 'day-video']),
    (False, True, ['expire', 'day-keogram', 'day-video', 'expire']),
])
def test_real_capture_job_builder_defers_all_jobs_until_last_callback(night, extra_expire, expected):
    Worker = methods('capture.py', 'CaptureWorker', ['_queuePeriodEnd'], {})
    worker = Worker()
    worker.camera_id = 1
    worker.generate_timelapse_flag = True
    outgoing = queue.Queue()
    worker._period_queue = CapturePeriodQueue(outgoing)
    worker._period_queue.begin(1, (int(night), 0), DAY, 20)
    calls = []
    for method, label in [('_expireData', 'expire'), ('_generateNightKeogram', 'night-keogram'),
                          ('_generateNightTimelapse', 'night-video'), ('_uploadAllskyEndOfNight', 'end-night'),
                          ('_generateDayKeogram', 'day-keogram'), ('_generateDayTimelapse', 'day-video')]:
        def create(*args, _label=label, task_ids):
            calls.append(_label)
            task_ids.append({'task_id': len(calls)})
        setattr(worker, method, create)
    worker._queuePeriodEnd(DAY, night, expire_twice=extra_expire)
    assert calls == expected
    assert not worker.generate_timelapse_flag
    assert outgoing.empty()
    # This requires neither a next frame nor reconfigureCcd: dawn with daytime
    # capture disabled, a paused camera and polar period closure all drain.
    worker._period_queue.put({'filename': 'last.fit'})
    assert 'filename' in outgoing.get_nowait()
    assert len(outgoing.get_nowait()['period_end']['tasks']) == len(expected)


class MissingState(Exception):
    pass


class StateStore:
    def __init__(self):
        self.values = {}
    def setState(self, key, value):
        self.values[key] = value
    def getState(self, key):
        if key not in self.values:
            raise MissingState(key)
        return self.values[key]


def image_worker(receipt, state=None, task_rows=None, prefix=None):
    namespace = dict(app=SimpleNamespace(app_context=nullcontext), queue=queue,
                     logger=logging.getLogger(__name__), frame_mode=frame_mode, datetime=datetime,
                     set_inflight=set_inflight, read_inflight=read_inflight, failure_key=failure_key,
                     NoResultFound=MissingState, TaskQueueState=SimpleNamespace(QUEUED='queued'),
                     constants=SimpleNamespace(NIGHT_NIGHT=0))
    task_rows = task_rows if task_rows is not None else {}
    namespace['IndiAllSkyDbTaskQueueTable'] = SimpleNamespace(query=SimpleNamespace(
        filter_by=lambda id: SimpleNamespace(one=lambda: task_rows[id])))
    Worker = methods('image.py', 'ImageWorker', ['saferun', '_releasePeriodEnd', '_setFrameContext', '_checkCaptureSequence',
                                              '_failCapturePeriod', '_checkCaptureImageSaved'], namespace)
    worker = Worker()
    worker.period_inflight = receipt
    worker.capture_sequence = CaptureSequenceTracker(prefix)
    worker._miscDb = state or StateStore()
    worker._shutdown = False
    worker.image_q = queue.Queue()
    worker.video_q = queue.Queue()
    worker.live_night_av = [0, 0]
    worker.config = {'DAYTIME_CAPTURE': True}
    worker.night_av = [-1, -1]
    worker.sensors_temp_av = FrameTemperatureValues([18., 99.], 0)
    worker.events = []
    worker.image_processor = SimpleNamespace(realtimeKeogramDataSave=lambda: worker.events.append('flush'))
    return worker


def test_private_frame_context_does_not_relabel_backlog_or_other_sensor_slots():
    worker = image_worker(None)
    held_context = worker.night_av
    result = worker._setFrameContext({'capture_mode': (1, 1), 'capture_day_date': '2026-10-07', 'capture_temperature': 12.5})
    assert result == date(2026, 10, 7)
    assert held_context == [1, 1]
    assert worker.live_night_av == [0, 0]
    worker.sensors_temp_av.live_values[:] = [25., 77.]
    assert worker.sensors_temp_av[0] == 12.5
    assert worker.sensors_temp_av[1] == 77.
    worker.sensors_temp_av[1] = 78.
    assert worker.sensors_temp_av.live_values[1] == 78.
    assert worker._setFrameContext({}) is None
    assert held_context == [0, 0]
    assert worker.sensors_temp_av[0] == 25.


def test_image_worker_releases_jobs_only_after_save_returns_and_before_stop():
    receipt = multiprocessing.RawArray(ctypes.c_char, 256)
    worker = image_worker(receipt)
    def save(frame):
        assert read_inflight(receipt) == PERIOD
        assert worker.video_q.empty()
        worker.events.extend(['save-' + frame['filename'], 'commit'])
    worker.processImage = save
    worker.image_q.put({'filename': 'last.fit', 'capture_period_id': PERIOD})
    worker.image_q.put({'period_end': {'period_id': PERIOD, 'tasks': [{'task_id': 3}, {'task_id': 4}]}})
    worker.image_q.put({'stop': True})
    worker.saferun()
    assert worker.events == ['save-last.fit', 'commit', 'flush', 'flush']
    assert [worker.video_q.get_nowait(), worker.video_q.get_nowait()] == [{'task_id': 3}, {'task_id': 4}]
    assert read_inflight(receipt) is None


def die_with_frame(receipt):
    set_inflight(receipt, PERIOD)
    os._exit(9)


def test_worker_death_receipt_survives_and_replacement_fails_closed():
    ctx = multiprocessing.get_context('spawn')
    receipt = ctx.RawArray(ctypes.c_char, 256)
    process = ctx.Process(target=die_with_frame, args=(receipt,))
    process.start()
    process.join(15)
    if process.is_alive():
        process.terminate()
        process.join(5)
        pytest.fail('Child did not exit')
    assert process.exitcode == 9
    assert read_inflight(receipt) == PERIOD
    failed = []
    task = SimpleNamespace(state='queued', setFailed=failed.append)
    worker = image_worker(receipt, task_rows={5: task})
    worker.processImage = lambda item: None
    worker.image_q.put({'period_end': {'period_id': PERIOD, 'tasks': [{'task_id': 5}]}})
    worker.image_q.put({'stop': True})
    worker.saferun()
    assert worker.video_q.empty()
    assert len(failed) == 1
    assert failure_key(PERIOD) in worker._miscDb.values
    assert read_inflight(receipt) is None


@pytest.mark.parametrize('value', [b'partial:1', b'\xff', b'UNKNOWN'])
def test_corrupt_receipt_blocks_any_period(value):
    receipt = multiprocessing.RawArray(ctypes.c_char, 256)
    receipt.value = value
    failed = []
    worker = image_worker(receipt, task_rows={5: SimpleNamespace(state='queued', setFailed=failed.append)})
    worker.image_q.put({'period_end': {'period_id': PERIOD, 'tasks': [{'task_id': 5}]}})
    worker.image_q.put({'stop': True})
    worker.saferun()
    assert worker.video_q.empty()
    assert failed


def test_oversize_receipt_never_truncates_to_another_valid_period():
    receipt = multiprocessing.RawArray(ctypes.c_char, 256)
    with pytest.raises(ValueError, match='too long'):
        set_inflight(receipt, '1' * 256)
    assert read_inflight(receipt) == 'UNKNOWN'


def test_processing_exception_keeps_receipt_for_replacement():
    receipt = multiprocessing.RawArray(ctypes.c_char, 256)
    worker = image_worker(receipt)
    def fail(frame):
        raise OSError('disk write failed')
    worker.processImage = fail
    worker.image_q.put({'filename': 'last.fit', 'capture_period_id': PERIOD})
    with pytest.raises(OSError):
        worker.saferun()
    assert read_inflight(receipt) == PERIOD
    assert worker.video_q.empty()


@pytest.mark.parametrize('reason', ['Captured frame file was not found', 'Captured frame file was empty',
                                   'Captured frame could not be decoded'])
def test_failed_frame_blocks_finalization_even_when_processing_returns(reason):
    failed = []
    worker = image_worker(None, task_rows={5: SimpleNamespace(state='queued', setFailed=failed.append)})
    worker._failCapturePeriod({'capture_period_id': PERIOD}, reason)
    worker._releasePeriodEnd({'period_id': PERIOD, 'tasks': [{'task_id': 5}]})
    assert worker.video_q.empty()
    assert failed


@pytest.mark.parametrize('mode,config,filename,expected_failure', [
    ([1, 0], {}, None, True),
    ([0, 0], {}, None, True),
    ([1, 0], {}, 'saved.jpg', False),
    ([1, 0], {'FOCUS_MODE': True}, None, False),
    ([0, 0], {'DAYTIME_CAPTURE_SAVE': False}, None, False),
])
def test_unexpected_unsaved_output_fails_but_intentional_nosave_does_not(mode, config, filename, expected_failure):
    worker = image_worker(None)
    worker.night_av[:] = mode
    worker.config.update(config)
    worker._checkCaptureImageSaved({'capture_period_id': PERIOD}, filename)
    assert bool(worker._miscDb.values) == expected_failure


def test_sqm_failure_does_not_make_timelapse_period_incomplete():
    worker = image_worker(None)
    worker._failCapturePeriod({'capture_period_id': PERIOD, 'sqm_exposure': True}, 'Bad SQM input')
    assert not worker._miscDb.values


@pytest.mark.parametrize('old_mode,new_night,forced,expected_date,expected_night,extra', [
    ((1, 0), False, False, date(2026, 10, 7), True, False),
    ((0, 0), True, False, DAY, False, False),
    ((1, 0), True, True, date(2026, 10, 7), True, False),
    ((0, 0), False, True, date(2026, 10, 7), False, True),
])
def test_actual_capture_boundary_block_routes_dawn_dusk_and_polar_periods(old_mode, new_night, forced,
                                                                        expected_date, expected_night, extra):
    tree = ast.parse((ROOT / 'indi_allsky/capture.py').read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'CaptureWorker')
    run = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'saferun')
    loop = next(n for n in run.body if isinstance(n, ast.While))
    context = next(n for n in loop.body if isinstance(n, ast.With))
    boundary = context.body[0]
    assert isinstance(boundary, ast.If)
    namespace = dict(datetime=datetime, timedelta=timedelta, logger=logging.getLogger(__name__),
                     constants=SimpleNamespace(NIGHT_NIGHT=0, NIGHT_MOONMODE=1))
    function = ast.FunctionDef(name='boundary', args=ast.arguments(posonlyargs=[], args=[ast.arg(arg='self'),
                              ast.arg(arg='loop_start_time')], kwonlyargs=[], kw_defaults=[], defaults=[]),
                              body=[boundary], decorator_list=[])
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])), '<boundary>', 'exec'), namespace)
    calls = []
    worker = SimpleNamespace(night_av=old_mode, night=new_night, moonmode=False, generate_timelapse_flag=True,
                             next_forced_transition_time=0 if forced else 2,
                             _dateCalcs=SimpleNamespace(getDayDate=lambda: DAY,
                                                       getNextDayNightTransition=lambda: datetime(2026, 10, 9)))
    worker._queuePeriodEnd = lambda day, night, expire_twice=False: calls.append((day, night, expire_twice))
    namespace['boundary'](worker, 1)
    assert calls == [(expected_date, expected_night, extra)]


def test_shoot_reserves_metadata_before_a_synchronous_camera_callback():
    Worker = methods('capture.py', 'CaptureWorker', ['shoot'],
                     dict(logger=logging.getLogger(__name__), constants=SimpleNamespace(NIGHT_NIGHT=0)))
    worker = Worker()
    worker.camera_id = 1
    worker.night_av = [1, 0]
    worker.config = {'EXPOSURE_PERIOD': 20, 'EXPOSURE_PERIOD_DAY': 5}
    worker._dateCalcs = SimpleNamespace(getDayDate=lambda: DAY)
    outgoing = queue.Queue()
    worker._period_queue = CapturePeriodQueue(outgoing)
    worker.indiclient = SimpleNamespace(setCcdExposure=lambda *a, **kw: worker._period_queue.put({'filename': 'frame.fit'}))
    worker.shoot(10, 50, 1, sync=False)
    assert outgoing.get_nowait()['capture_mode'] == (1, 0)
    assert not worker._period_queue.waiting


def numbered_frame(sequence, period=PERIOD, stream=STREAM, **extra):
    return dict(filename='frame-{0}.fit'.format(sequence), capture_period_id=period,
                capture_stream_id=stream, capture_sequence=sequence, **extra)


def closing_marker(sequence, period=PERIOD, stream=STREAM):
    return dict(period_id=period, stream_id=stream, last_sequence=sequence, tasks=[{'task_id': 5}])


@pytest.mark.parametrize('delivered,last_sequence', [
    ([2, 3], 3),          # first frame lost by a producer/feeder
    ([1, 3], 3),          # middle frame lost
    ([1, 2], 3),          # final frame lost, only visible at the marker
    ([1, 2, 2, 3], 3),    # duplicated frame
    ([1, 3, 2], 3),       # out-of-order delivery
    ([], 1),              # entire single-frame period lost
])
def test_renderer_detects_lost_feeder_frames_even_without_an_inflight_receipt(delivered, last_sequence):
    failed = []
    worker = image_worker(None, task_rows={5: SimpleNamespace(state='queued', setFailed=failed.append)})
    worker.processImage = lambda item: None
    for sequence in delivered:
        worker.image_q.put(numbered_frame(sequence))
    worker.image_q.put({'period_end': closing_marker(last_sequence)})
    worker.image_q.put({'stop': True})
    worker.saferun()
    assert worker.video_q.empty()
    assert failed
    assert failure_key(PERIOD) in worker._miscDb.values


def test_gap_across_sunrise_marks_both_possible_periods():
    tracker = CaptureSequenceTracker()
    night = numbered_frame(1)
    assert not tracker.check(night)
    tracker.complete(night)
    assert tracker.check(numbered_frame(3, period='1:2026-10-09:0')) == {PERIOD, '1:2026-10-09:0'}


def test_sqm_is_part_of_the_same_ordered_prefix():
    worker = image_worker(None)
    processed = []
    worker.processImage = lambda item: processed.append(item.get('sqm_exposure', False))
    worker.image_q.put(numbered_frame(1))
    worker.image_q.put(numbered_frame(2, sqm_exposure=True))
    worker.image_q.put(numbered_frame(3))
    worker.image_q.put({'period_end': closing_marker(3)})
    worker.image_q.put({'stop': True})
    worker.saferun()
    assert processed == [False, True, False]
    assert worker.video_q.get_nowait() == {'task_id': 5}
    assert not worker._miscDb.values


def test_renderer_replacement_keeps_completed_prefix_and_detects_next_missing_frame():
    shared = multiprocessing.RawArray(ctypes.c_char, 1024)
    first = CaptureSequenceTracker(shared)
    first.complete(numbered_frame(1))
    replacement = CaptureSequenceTracker(shared)
    assert not replacement.check(numbered_frame(2))
    assert replacement.check(numbered_frame(3)) == {PERIOD}
    assert replacement.check(closing_marker(2), marker=True) == {PERIOD}


def test_new_capture_worker_starts_a_new_stream_without_forgetting_old_queue_order():
    tracker = CaptureSequenceTracker()
    tracker.complete(numbered_frame(20))
    next_frame = numbered_frame(1, stream='b' * 32)
    assert not tracker.check(next_frame)
    tracker.complete(next_frame)
    assert tracker.check(numbered_frame(21)) == {PERIOD}


def test_new_capture_stream_without_frames_can_close_previous_period():
    tracker = CaptureSequenceTracker()
    tracker.complete(numbered_frame(20))
    empty = closing_marker(0, stream='b' * 32)
    assert not tracker.check(empty, marker=True)
    tracker.complete(empty, marker=True)
    assert not tracker.check(numbered_frame(1, stream='b' * 32))


def test_failed_marker_settles_missing_prefix_without_rewinding_or_poisoning_next_period():
    tracker = CaptureSequenceTracker()
    tracker.complete(numbered_frame(1))
    assert tracker.check(closing_marker(2), marker=True) == {PERIOD}
    tracker.complete(closing_marker(2), marker=True)
    assert not tracker.check(numbered_frame(3, period='1:2026-10-09:0'))
    tracker.complete(numbered_frame(1))
    assert tracker.current[1] == 2


@pytest.mark.parametrize('damage', [b'UNKNOWN', b'{"last_sequence":7}', b'\xff\ninvalid-checksum'])
def test_torn_or_unparseable_prefix_checkpoint_fails_closed(damage):
    shared = multiprocessing.RawArray(ctypes.c_char, 1024)
    shared.value = damage
    tracker = CaptureSequenceTracker(shared)
    assert 'UNKNOWN' in tracker.check(numbered_frame(1))
    assert 'UNKNOWN' in tracker.check(closing_marker(0), marker=True)


def test_valid_looking_torn_prefix_cannot_hide_a_dropped_frame():
    shared = multiprocessing.RawArray(ctypes.c_char, 1024)
    tracker = CaptureSequenceTracker(shared)
    tracker.complete(numbered_frame(1))
    shared.value = shared.value.replace(b'"last_sequence":1', b'"last_sequence":2')
    assert 'UNKNOWN' in CaptureSequenceTracker(shared).check(numbered_frame(3))


@pytest.mark.parametrize('broken', [dict(capture_sequence=1),
                                    dict(capture_stream_id=STREAM, capture_sequence=True, capture_period_id=PERIOD),
                                    dict(capture_stream_id=STREAM, capture_sequence=0, capture_period_id=PERIOD)])
def test_partial_or_invalid_numbering_fails_closed(broken):
    assert CaptureSequenceTracker().check(broken) == {'UNKNOWN'}


def producer_with_paused_feeder(output, put_returned, hold):
    import multiprocessing.queues
    # Stop the real multiprocessing feeder before it acquires the pipe lock.
    # put() still returns, exactly the acknowledgment window under test.
    def paused(*args):
        hold.wait(30)
    multiprocessing.queues.Queue._feed = staticmethod(paused)
    output.put(numbered_frame(2))
    put_returned.set()
    hold.wait(30)


def test_killed_producer_after_put_cannot_hide_a_frame_lost_in_its_feeder():
    ctx = multiprocessing.get_context('spawn')
    output = ctx.Queue()
    returned, hold = ctx.Event(), ctx.Event()
    producer = ctx.Process(target=producer_with_paused_feeder, args=(output, returned, hold))
    producer.start()
    try:
        assert returned.wait(15), 'Producer did not acknowledge put()'
        producer.terminate()
        producer.join(5)
        assert not producer.is_alive()
        # A replacement can enqueue the next frame, but the acknowledged frame
        # from the killed process never reached the consumer's pipe.
        output.put(numbered_frame(3))
        delivered = output.get(timeout=5)
        assert delivered['capture_sequence'] == 3
        tracker = CaptureSequenceTracker()
        tracker.complete(numbered_frame(1))
        assert tracker.check(delivered) == {PERIOD}
    finally:
        if producer.is_alive():
            producer.terminate()
            producer.join(5)
        output.close()
        output.join_thread()
