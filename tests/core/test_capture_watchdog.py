import ast
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from indi_allsky import capture_watchdog as watchdog_module
from indi_allsky.capture_watchdog import CaptureTimeoutError, CaptureWatchdog, FrameArrivalQueue, FrameDeadline, ProcessingAllowance
from indi_allsky.capture_control import drain_worker_control_queue, request_worker_stop
from indi_allsky.twilight import capture_period


@pytest.fixture
def clock(monkeypatch):
    clock = SimpleNamespace(now=100.0)
    monkeypatch.setattr(watchdog_module, 'time', SimpleNamespace(monotonic=lambda: clock.now))
    return clock


class AdvancingEvent:
    def __init__(self, clock, stop_at=float('inf')):
        self.clock = clock
        self.stop_at = stop_at

    def wait(self, seconds):
        self.clock.now += seconds
        return self.clock.now >= self.stop_at


@pytest.mark.parametrize('timeout', [1, 80, 150, 330])
@pytest.mark.parametrize('poll_offset', [0, 0.01, 0.51, 0.99])
def test_deadline_reacts_within_one_second_and_kills_within_four(clock, monkeypatch, timeout, poll_offset):
    deadline = FrameDeadline(timeout)
    deadline.begin_exposure()
    started = clock.now
    # Phase of the parent's polling is independent of the last frame.
    clock.now += poll_offset
    worker = Mock(pid=123, is_alive=Mock(return_value=True))
    actions = []
    worker.kill.side_effect = lambda: actions.append(('kill', clock.now))
    monkeypatch.setattr(watchdog_module, 'os', SimpleNamespace(kill=lambda *args: actions.append(('abort', clock.now))))
    monkeypatch.setattr(watchdog_module, 'signal', SimpleNamespace(SIGUSR1=10))
    watchdog = CaptureWatchdog(worker, deadline)
    watchdog._stop_event = AdvancingEvent(clock)

    watchdog.run()

    assert watchdog.timed_out
    assert [action for action, _ in actions] == ['abort', 'kill']
    assert timeout <= actions[0][1] - started < timeout + 1.01
    assert timeout + 3 <= actions[1][1] - started < timeout + 4.01


def test_ready_without_frames_and_reissued_exposures_do_not_extend_deadline(clock):
    deadline = FrameDeadline(80)
    deadline.begin_exposure()
    for elapsed in [10, 30, 60, 79]:
        clock.now = 100 + elapsed
        deadline.begin_exposure()
        deadline.schedule_next(clock.now + 15)
        assert deadline.snapshot() == 180


@pytest.mark.parametrize('arrival_before_schedule', [False, True])
def test_long_period_and_queue_backoff_are_allowed_only_after_a_frame(clock, arrival_before_schedule):
    deadline = FrameDeadline(80)
    deadline.begin_exposure()
    if arrival_before_schedule:
        clock.now = 130
        deadline.received()
    deadline.schedule_next(400)  # night/day period including queue backoff
    if not arrival_before_schedule:
        assert deadline.snapshot() == 180  # first frame is still required
        clock.now = 130
        deadline.received()
    assert deadline.snapshot() == 480
    clock.now = 400
    deadline.begin_exposure()
    assert deadline.snapshot() == 480


def test_new_frame_resets_timeout_independently_of_processing(clock):
    deadline = FrameDeadline(80)
    deadline.begin_exposure()
    downstream = Mock()
    frame_queue = FrameArrivalQueue(downstream, deadline)
    clock.now = 179.5
    frame = {'filename': 'frame.fit'}
    frame_queue.put(frame)
    downstream.put.assert_called_once_with(frame)
    assert deadline.snapshot() == 259.5


@pytest.mark.parametrize('period', [15, 30])
def test_fixed_80_second_limit_includes_the_normal_idle_gap(clock, period):
    deadline = FrameDeadline(80)
    deadline.begin_exposure(5, period)
    deadline.schedule_next(100 + period, period)
    clock.now = 105
    deadline.received()
    assert deadline.snapshot() == 185


def test_failed_frame_delivery_does_not_reset_timeout(clock):
    deadline = FrameDeadline(80)
    deadline.begin_exposure()
    downstream = Mock(put=Mock(side_effect=OSError('queue unavailable')))
    clock.now = 170
    with pytest.raises(OSError):
        FrameArrivalQueue(downstream, deadline).put({'filename': 'frame.fit'})
    assert deadline.snapshot() == 180


def test_pause_and_resume_do_not_inherit_an_expired_deadline(clock):
    deadline = FrameDeadline(80)
    deadline.begin_exposure()
    deadline.suspend()
    clock.now = 10000
    deadline.received()  # late frame during pause must not rearm capture
    assert deadline.snapshot() == 0
    deadline.begin_exposure()
    assert deadline.snapshot() == 10080


def test_wall_clock_changes_do_not_move_deadline(clock, monkeypatch):
    deadline = FrameDeadline(80)
    deadline.begin_exposure()
    monkeypatch.setattr(watchdog_module.time, 'time', lambda: -9999999, raising=False)
    assert deadline.snapshot() == 180
    clock.now = 130
    deadline.received()
    assert deadline.snapshot() == 210


@pytest.mark.parametrize(('exposure', 'period', 'processing', 'expected'), [
    (5, 30, 20, 80), (30, 15, 20, 80), (1, 120, 7, 247),
    (60, 0, 12, 132), (0.01, 4, 2, 10),
])
def test_automatic_timeout_uses_actual_cadence_and_processing_peak(clock, exposure, period, processing, expected):
    allowance = ProcessingAllowance()
    allowance.record(processing)
    deadline = FrameDeadline(0, allowance)
    deadline.begin_exposure(exposure, period)
    assert deadline.snapshot() == 100 + expected


def test_automatic_period_is_not_added_twice(clock):
    allowance = ProcessingAllowance()
    allowance.record(20)
    deadline = FrameDeadline(0, allowance)
    deadline.begin_exposure(5, 30)
    deadline.schedule_next(130, 30)
    clock.now = 105
    deadline.received()
    assert deadline.snapshot() == 185  # 105 + 2 * 30 + 20, no extra idle gap


def test_current_exposure_and_day_night_period_changes_are_used(clock):
    allowance = ProcessingAllowance()
    allowance.record(20)
    deadline = FrameDeadline(0, allowance)
    deadline.begin_exposure(5, 15)
    clock.now = 105
    deadline.received()
    deadline.begin_exposure(30, 15)
    assert deadline.snapshot() == 185
    clock.now = 135
    deadline.received()
    deadline.begin_exposure(1, 120)
    assert deadline.snapshot() == 395


def test_automatic_deadline_does_not_keep_moving_without_frames(clock):
    allowance = ProcessingAllowance()
    allowance.record(20)
    deadline = FrameDeadline(0, allowance)
    deadline.begin_exposure(30, 15)
    for elapsed in [10, 30, 60, 79]:
        clock.now = 100 + elapsed
        allowance.record(100)  # processing backlog cannot postpone recovery
        deadline.begin_exposure(30, 15)
        deadline.schedule_next(clock.now + 15, 15)
        assert deadline.snapshot() == 180


def test_processing_allowance_uses_recent_peak_and_ignores_invalid_samples():
    allowance = ProcessingAllowance()
    assert allowance.maximum() == 10
    allowance.record(20)
    for _ in range(9):
        allowance.record(2)
    for bad in [float('nan'), float('inf'), -2, 0]:
        allowance.record(bad)
    assert allowance.maximum() == 20
    allowance.record(3)
    assert allowance.maximum() == 3


def test_manual_timeout_is_not_overridden_by_exposure_or_processing(clock):
    allowance = ProcessingAllowance()
    allowance.record(500)
    deadline = FrameDeadline(80, allowance)
    deadline.begin_exposure(60, 30)
    assert deadline.snapshot() == 180


def test_cooperative_abort_is_not_force_killed(clock, monkeypatch):
    deadline = FrameDeadline(80)
    deadline.begin_exposure()
    worker = Mock(pid=123, is_alive=Mock(return_value=True))
    def exit_worker(*args):
        worker.is_alive.return_value = False
    monkeypatch.setattr(watchdog_module, 'os', SimpleNamespace(kill=exit_worker))
    monkeypatch.setattr(watchdog_module, 'signal', SimpleNamespace(SIGUSR1=10))
    watchdog = CaptureWatchdog(worker, deadline)
    watchdog._stop_event = AdvancingEvent(clock)
    watchdog.run()
    assert watchdog.timed_out
    worker.kill.assert_not_called()


def test_frame_arriving_during_deadline_read_is_not_falsely_expired(clock, monkeypatch):
    deadline = FrameDeadline(80)
    deadline.begin_exposure()
    snapshot = deadline.snapshot
    def concurrent_arrival():
        previous = snapshot()
        clock.now = 179.999
        deadline.received()
        clock.now = 180.01
        return previous
    monkeypatch.setattr(deadline, 'snapshot', concurrent_arrival)
    clock.now = 178.98
    send_signal = Mock()
    monkeypatch.setattr(watchdog_module, 'os', SimpleNamespace(kill=send_signal))
    watchdog = CaptureWatchdog(Mock(is_alive=Mock(return_value=True)), deadline)
    watchdog._stop_event = AdvancingEvent(clock, stop_at=181)

    watchdog.run()

    assert not watchdog.timed_out
    send_signal.assert_not_called()


def test_stuck_progress_lock_cannot_block_recovery(clock, monkeypatch):
    deadline = Mock(snapshot=Mock(side_effect=[180] + [None] * 100))
    worker = Mock(pid=123, is_alive=Mock(return_value=True))
    sent = []
    monkeypatch.setattr(watchdog_module, 'os', SimpleNamespace(kill=lambda *args: sent.append(clock.now)))
    monkeypatch.setattr(watchdog_module, 'signal', SimpleNamespace(SIGUSR1=10))
    watchdog = CaptureWatchdog(worker, deadline)
    watchdog._stop_event = AdvancingEvent(clock)
    watchdog.run()
    assert sent == [180]
    worker.kill.assert_called_once_with()


@pytest.mark.parametrize('suspended', [True, False])
def test_inactive_or_exited_worker_is_not_signalled(clock, monkeypatch, suspended):
    deadline = FrameDeadline(80)
    worker = Mock(is_alive=Mock(return_value=suspended))
    send_signal = Mock()
    monkeypatch.setattr(watchdog_module, 'os', SimpleNamespace(kill=send_signal))
    watchdog = CaptureWatchdog(worker, deadline)
    watchdog._stop_event = AdvancingEvent(clock, stop_at=500)
    watchdog.run()
    assert not watchdog.timed_out
    send_signal.assert_not_called()


def load_methods(filename, class_name, names, namespace):
    # Exercise real worker/supervisor methods without importing Linux camera,
    # D-Bus or application/database services on the test host.
    path = Path(__file__).resolve().parents[2] / 'indi_allsky' / filename
    tree = ast.parse(path.read_text(encoding='utf-8'))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    methods = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in names]
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


def test_worker_unwinds_and_aborts_before_disconnect_and_error_reporting():
    actions = []
    namespace = load_methods('capture.py', 'CaptureWorker', ['run'], {
        'signal': Mock(), 'CaptureTimeoutError': CaptureTimeoutError,
        'traceback': SimpleNamespace(format_exc=lambda: 'trace'),
    })
    worker = Mock()
    worker.saferun.side_effect = CaptureTimeoutError('deadline')
    worker.indiclient.abortCcdExposure.side_effect = lambda: actions.append('abort')
    worker.indiclient.disconnectServer.side_effect = lambda: actions.append('disconnect')
    worker.error_q.put.side_effect = lambda error: actions.append('report')
    with pytest.raises(CaptureTimeoutError):
        namespace['run'](worker)
    assert actions == ['abort', 'disconnect', 'report']


def test_deadline_is_armed_before_a_blocking_exposure_command(clock):
    namespace = load_methods('capture.py', 'CaptureWorker', ['shoot'], {'logger': Mock()})
    deadline = FrameDeadline(80)
    def blocked_exposure(*args, **kwargs):
        assert deadline.snapshot() == 180
        raise RuntimeError('camera command blocked')
    worker = SimpleNamespace(
        frame_deadline=deadline, indiclient=SimpleNamespace(setCcdExposure=blocked_exposure),
        focus_mode=False, night=True, config={'EXPOSURE_PERIOD': 15}, add_period_delay=0,
    )
    with pytest.raises(RuntimeError):
        namespace['shoot'](worker, 30, 10, 1, sync=False)


def test_supervisor_keeps_monitor_running_until_worker_has_stopped():
    actions = []
    namespace = load_methods('allsky.py', 'IndiAllSky', ['_stopCaptureWorker', '_requestCaptureWorkerStop'], {
        'logger': Mock(), 'request_worker_stop': request_worker_stop,
    })
    parent = SimpleNamespace(
        capture_watchdog=Mock(stop=Mock(side_effect=lambda: actions.append('stop monitor'))),
        capture_worker=Mock(is_alive=Mock(return_value=True), join=Mock(side_effect=lambda: actions.append('join'))),
        capture_q=Mock(), _terminate=False, _capture_worker_stop_requested=False,
    )
    parent._requestCaptureWorkerStop = lambda: namespace['_requestCaptureWorkerStop'](parent)
    namespace['_stopCaptureWorker'](parent)
    assert actions == ['join', 'stop monitor']
    parent.capture_q.put.assert_called_once_with({'stop': True})


@pytest.mark.parametrize('shutdown', [False, True])
def test_dark_maintenance_stops_monitor_and_respects_service_shutdown(shutdown):
    actions = []
    dark = Mock()
    namespace = load_methods('allsky.py', 'IndiAllSky', ['_runDarkAutomationTask', '_stopCaptureWorker'], {
        '__file__': __file__, 'Path': Path, 'logger': Mock(), 'app': Mock(), 'dark_automation': dark,
    })
    parent = Mock(_dark_automation_task_id=7, _shutdown=False, _terminate=False)
    parent.capture_worker.is_alive.return_value = True
    parent.capture_worker.join.side_effect = lambda: actions.append('join')
    parent.capture_watchdog.stop.side_effect = lambda: actions.append('stop monitor')
    parent._stopCaptureWorker.side_effect = lambda: namespace['_stopCaptureWorker'](parent)
    parent._startCaptureWorker.side_effect = lambda: actions.append('restart')
    def maintenance(*args, **kwargs):
        assert actions == ['join', 'stop monitor']
        actions.append('maintenance')
        parent._shutdown = shutdown
        assert kwargs['stop_requested']() == shutdown
    dark.run_task.side_effect = maintenance

    namespace['_runDarkAutomationTask'](parent)

    assert parent._dark_automation_task_id is None
    assert actions == ['join', 'stop monitor', 'maintenance'] + ([] if shutdown else ['restart'])
    assert dark.mark_capture_restored.call_count == int(not shutdown)


def test_supervisor_starts_replacement_before_notifying(monkeypatch):
    import queue
    import sys
    actions = []
    replacement = Mock(start=Mock(side_effect=lambda: actions.append('restart')))
    monkeypatch.setitem(sys.modules, 'indi_allsky.capture', SimpleNamespace(CaptureWorker=Mock(return_value=replacement)))
    namespace = load_methods('allsky.py', 'IndiAllSky', ['_startCaptureWorker'], {
        '__name__': 'indi_allsky.allsky', 'queue': queue, 'logger': Mock(),
        'FrameDeadline': FrameDeadline, 'CaptureWatchdog': Mock(),
        'app': SimpleNamespace(app_context=nullcontext),
        'NotificationCategory': SimpleNamespace(CAMERA='camera'), 'timedelta': timedelta,
    })
    parent = Mock(config={'CCD_EXPOSURE_TIMEOUT': 80}, capture_worker_idx=1)
    parent.capture_worker.is_alive.return_value = False
    parent.capture_error_q.get_nowait.side_effect = queue.Empty
    parent.capture_watchdog.timed_out = True
    parent._miscDb.addNotification.side_effect = lambda *args, **kwargs: actions.append('notify')
    namespace['_startCaptureWorker'](parent)
    assert actions == ['restart', 'notify']
    assert parent.capture_worker is replacement
    assert namespace['CaptureWatchdog'].call_args.args[1].timeout == 80


class EndCaptureTest(BaseException):
    pass


class SharedModes(list):
    def get_lock(self):
        return nullcontext()


def run_capture_loop(clock, *, night=True, paused=False, day_enabled=True, deliver=True, period=30, exposure=5, stop_after=90, twilight=False):
    import queue
    from indi_allsky import constants
    start = clock.now
    def sleep(seconds):
        clock.now += seconds
        if clock.now >= start + stop_after:
            raise EndCaptureTest()
    fake_time = SimpleNamespace(time=lambda: clock.now, monotonic=lambda: clock.now, sleep=sleep)
    namespace = load_methods('capture.py', 'CaptureWorker', ['saferun', 'shoot', '_processCaptureControlQueue'], {
        'time': fake_time, 'app': SimpleNamespace(app_context=nullcontext),
        'logger': Mock(), 'constants': constants, 'queue': queue,
        'timedelta': timedelta, 'datetime': __import__('datetime').datetime,
        'NotificationCategory': SimpleNamespace(CAMERA='camera'),
        'drain_worker_control_queue': drain_worker_control_queue, 'capture_period': capture_period,
    })
    allowance = ProcessingAllowance()
    allowance.record(20)
    deadline = FrameDeadline(0, allowance)
    image_queue = Mock(qsize=Mock(return_value=0))
    delivery_queue = FrameArrivalQueue(image_queue, deadline)
    in_flight = []
    def start_exposure(seconds, *args, **kwargs):
        in_flight.append(clock.now + seconds)
    def status():
        if not in_flight:
            return True, 'READY'
        if not deliver:
            return True, 'READY'  # faulty driver sends no frame
        if clock.now < in_flight[0]:
            return False, 'BUSY'
        in_flight.pop(0)
        delivery_queue.put({'filename': 'camera.fit'})
        return True, 'READY'
    worker = SimpleNamespace(
        night=night, moonmode=False, night_av=SharedModes([int(night), 0]),
        astro_av={constants.ASTRO_SUN_ALT: -9.0},
        detectNight=lambda: None, _initialize=lambda: None, _pre_run_tasks=lambda: None,
        _dateCalcs=SimpleNamespace(getNextDayNightTransition=lambda: SimpleNamespace(timestamp=lambda: 999999)),
        indiclient=SimpleNamespace(disconnected=False, ccd_removed=False, getCcdExposureStatus=status,
                                  setCcdExposure=start_exposure),
        capture_q=queue.Queue(), frame_deadline=deadline,
        config={'CAPTURE_PAUSE': paused, 'DAYTIME_CAPTURE': day_enabled,
                'EXPOSURE_PERIOD': period, 'EXPOSURE_PERIOD_DAY': period, 'CAMERA_INTERFACE': 'indi'},
        getCcdTemperature=lambda: None, getTelescopeRaDec=lambda: None, getGpsPosition=lambda: None,
        _shutdown=False, _miscDb=Mock(), reconfigureCcd=lambda: None,
        periodic_tasks_time=999999, update_time_offset=None, capture_pre_hook=lambda: None,
        sqm_camera_enable=False, focus_mode=False, image_q=image_queue, image_queue_min=1,
        image_queue_max=3, image_queue_backoff=0.5, add_period_delay=0,
        _expUtils=SimpleNamespace(EXPOSURE_NEXT=exposure, EXPOSURE_CURRENT=exposure,
                                 GAIN_NEXT=1, BINNING_NEXT=1),
    )
    if twilight:
        worker.config.update(EXPOSURE_PERIOD=15, EXPOSURE_PERIOD_DAY=120,
                             NIGHT_SUN_ALT_DEG=-6, TWILIGHT_TRANSITION={'ENABLE': True, 'NIGHT_ALT': -12})
    worker._processCaptureControlQueue = lambda: namespace['_processCaptureControlQueue'](worker)
    worker.shoot = lambda *args, **kwargs: namespace['shoot'](worker, *args, **kwargs)
    with pytest.raises(EndCaptureTest):
        namespace['saferun'](worker)
    return deadline, image_queue


def test_real_capture_loop_does_not_treat_ready_as_frame_progress(clock):
    deadline, images = run_capture_loop(clock, deliver=False, exposure=30, period=15)
    assert 180 <= deadline.snapshot() <= 180.1
    images.put.assert_not_called()


@pytest.mark.parametrize('deliver', [True, False])
def test_twilight_interval_applies_even_before_the_first_frame(clock, deliver):
    deadline, _ = run_capture_loop(clock, twilight=True, deliver=deliver, stop_after=10)
    # Halfway from 120s day to 15s night: 2 * 67.5 + 20 = 155s.
    reference = 105 if deliver else 100
    assert reference + 155 <= deadline.snapshot() <= reference + 155.2


@pytest.mark.parametrize('night', [True, False])
def test_real_capture_loop_allows_the_selected_day_or_night_interval(clock, night):
    deadline, images = run_capture_loop(clock, night=night, period=120)
    images.put.assert_called_once()
    assert 365 <= deadline.snapshot() <= 365.2  # frame at 105 + 240 + 20


@pytest.mark.parametrize(('paused', 'night', 'day_enabled'), [(True, True, True), (False, False, False)])
def test_real_capture_loop_does_not_arm_during_intentional_inactivity(clock, paused, night, day_enabled):
    deadline, images = run_capture_loop(clock, paused=paused, night=night, day_enabled=day_enabled)
    assert deadline.snapshot() == 0
    images.put.assert_not_called()
