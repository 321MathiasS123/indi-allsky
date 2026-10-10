"""Dispatch real exposure requests on a clock independent of render completion."""
import ast
from contextlib import nullcontext
from datetime import date, datetime, timedelta
import heapq
import logging
from pathlib import Path
import queue
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from indi_allsky import constants
from indi_allsky.capture_period import CapturePeriodQueue
from indi_allsky.highlight import HighlightMeasurement
from indi_allsky.highlight_meter import OutputFeedbackGate
from test_capture_period import methods
from test_highlight_exposure import controller


def capture_method(name='shoot'):
    # Execute the real dispatch boundary without importing camera/DBus drivers.
    source = Path(__file__).parents[2] / 'indi_allsky' / 'capture.py'
    tree = ast.parse(source.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'CaptureWorker')
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == name)
    namespace = {'logger': logging.getLogger(__name__)}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), 'exec'), namespace)
    return namespace[name]


@pytest.mark.parametrize('transition_period,expected', [(None, 10), (0, 0), (67.5, 67.5)])
def test_optional_capture_period_hook_keeps_blended_resource_cadence(transition_period, expected):
    capture = SimpleNamespace(focus_mode=False, night=False,
                              config={'EXPOSURE_PERIOD': 20, 'EXPOSURE_PERIOD_DAY': 10},
                              _twilightCapturePeriod=lambda: transition_period)
    assert capture_method('_configuredCapturePeriod')(capture) == expected


def test_focus_cadence_takes_priority_over_optional_transition_hook():
    hook = Mock(side_effect=AssertionError('Focus capture must not consult the transition hook'))
    capture = SimpleNamespace(focus_mode=True, config={'FOCUS_DELAY': 1.5}, _twilightCapturePeriod=hook)
    assert capture_method('_configuredCapturePeriod')(capture) == 1.5
    hook.assert_not_called()


def run_capture_clock(initial, render_seconds):
    control = controller('exposure_basic', night=False)
    control.config.update(EXPOSURE_PERIOD=20, EXPOSURE_PERIOD_DAY=20)
    control.config['HIGHLIGHT_PROTECTION']['OUTPUT_ENABLE'] = True
    state = control._expUtils
    state.EXPOSURE_NEXT, state.GAIN_NEXT, state.BINNING_NEXT = initial, 0, 1
    gate = OutputFeedbackGate()
    events = [(1000., 'capture', 0)]
    jobs, commands, decisions, displayed = {}, [], [], []
    render_free = meter_free = 0.
    feedback = None
    last_decision = None
    now = 0.

    def camera_command(exposure, gain, binning, **kwargs):
        # The driver records CURRENT when issuing the hardware command.
        state.EXPOSURE_CURRENT, state.GAIN_CURRENT = exposure, gain
        commands.append((now, exposure, gain, last_decision))

    capture = SimpleNamespace(
        indiclient=SimpleNamespace(setCcdExposure=camera_command),
        config=control.config, focus_mode=False, night=False, add_period_delay=0,
        night_av=control.night_av, _period_queue=Mock(),
        camera_id=1, _dateCalcs=Mock(),
    )
    shoot = capture_method()
    while events:
        now, kind, index = heapq.heappop(events)
        if kind == 'capture':
            shoot(capture, state.EXPOSURE_NEXT, state.GAIN_NEXT, state.BINNING_NEXT, sync=False)
            exposure = commands[-1][1]
            arrival = now + exposure + .9
            jobs[index] = dict(exp_time=arrival, exposure=exposure, exp_elapsed=exposure + .9,
                               gain=0, binning=1, camera_id=1, capture_mode=(0, 0), capture_period=20)
            heapq.heappush(events, (arrival, 'raw', index))
            if index < 39:
                # Production starts when both the period and camera readiness
                # permit it; it does not wait for either meter or renderer.
                heapq.heappush(events, (max(now + 20, arrival + .02), 'capture', index + 1))
        elif kind == 'raw':
            meter_free = max(now, meter_free) + .85
            heapq.heappush(events, (meter_free, 'meter', index))
        elif kind == 'meter':
            job = jobs[index]
            gate.update(control, job, feedback, allowance=.85)
            scene = 1.06 ** min(index, 20) * .94 ** max(index - 20, 0)
            measurement = HighlightMeasurement(0, 0, 110 * scene * job['exposure'] / initial, 0, 0)
            control.compare_highlights(measurement, job['exposure'], job['gain'])
            last_decision = index
            decisions.append((now, index, state.EXPOSURE_NEXT, state.GAIN_NEXT))
            render_free = max(now, render_free) + render_seconds
            heapq.heappush(events, (render_free, 'render', index))
        else:
            feedback = dict(jobs[index], measurement=(0, 0, 80, 0, 0), trusted=True)
            displayed.append((now, index))
    return commands, decisions, displayed


@pytest.mark.parametrize('initial', [1., 17.5, 18.5, 30.])
def test_renderer_crossing_the_capture_period_does_not_add_a_reaction_frame(initial):
    fast, slow = [run_capture_clock(initial, duration) for duration in (16, 21)]
    assert slow[0] == fast[0]  # Actual camera dispatches, not just queued NEXT.
    assert slow[1] == fast[1]  # Raw decisions finish at exactly the same times.
    assert slow[2][-1][0] > fast[2][-1][0] + 20  # Only display accumulates lag.
    for commands, decisions, _ in (fast, slow):
        for start, exposure, gain, decision_index in commands[1:]:
            ready = [decision for decision in decisions if decision[0] <= start]
            if ready:
                _, expected_index, expected_exposure, expected_gain = ready[-1]
                assert (exposure, gain, decision_index) == (expected_exposure, expected_gain, expected_index)
        if initial > 18:
            # Metering completes after the next long exposure has started.
            assert commands[1][1:] == (initial, 0, None)
            assert commands[2][1] == pytest.approx(initial * .9, abs=1e-6)
        else:
            assert commands[1][1] == pytest.approx(initial * .9, abs=1e-6)
            assert commands[1][3] == 0


@pytest.mark.parametrize('resource_wait', [None, 'blocked', 'slow'])
@pytest.mark.parametrize('stalled', [False, True])
def test_real_capture_loop_resource_waits_preserve_ordinary_busy_timeout(resource_wait, stalled):
    class EndCaptureTest(BaseException):
        pass

    class SharedMode(list):
        def get_lock(self):
            return nullcontext()

    clock = SimpleNamespace(now=100.)
    commands, delivered, aborted, in_flight = [], [], [], []

    def sleep(seconds):
        clock.now += seconds
        if clock.now >= 180.:
            raise EndCaptureTest()

    fake_time = SimpleNamespace(time=lambda: clock.now, sleep=sleep)
    Worker = methods('capture.py', 'CaptureWorker', ['saferun', 'shoot', '_configuredCapturePeriod'], {
        'time': fake_time, 'app': SimpleNamespace(app_context=nullcontext),
        'logger': logging.getLogger(__name__), 'constants': constants, 'queue': queue,
        'timedelta': timedelta, 'datetime': datetime,
        'NotificationCategory': SimpleNamespace(CAMERA='camera'),
    })
    output = SimpleNamespace(put=delivered.append, qsize=lambda: 0)
    producer = CapturePeriodQueue(output)

    def expose(seconds, *args, **kwargs):
        commands.append(clock.now)
        in_flight.append(clock.now + seconds)

    def status():
        if not in_flight:
            return True, 'READY'
        if stalled or clock.now < in_flight[0]:
            return False, 'BUSY'
        in_flight.pop(0)
        producer.put({'filename': 'camera.fit'})
        return True, 'READY'

    resource = SimpleNamespace(can_capture=True)

    def resource_delay(period, exposure):
        assert period == 5 and exposure == 1
        resource.can_capture = resource_wait != 'blocked' or clock.now >= 145.
        return 40. if resource_wait == 'slow' else 0.

    resource.delay = resource_delay
    worker = Worker()
    worker.__dict__.update(
        night=True, moonmode=False, night_av=SharedMode([1, 0]),
        detectNight=lambda: None, _initialize=lambda: None, _pre_run_tasks=lambda: None,
        _dateCalcs=SimpleNamespace(getNextDayNightTransition=lambda: SimpleNamespace(timestamp=lambda: 999999),
                                  getDayDate=lambda: date(2026, 10, 10)),
        indiclient=SimpleNamespace(disconnected=False, ccd_removed=False, getCcdExposureStatus=status,
                                  setCcdExposure=expose, abortCcdExposure=lambda: aborted.append(clock.now)),
        capture_q=queue.Queue(), exposure_timeout=10,
        config={'CAPTURE_PAUSE': False, 'DAYTIME_CAPTURE': True,
                'EXPOSURE_PERIOD': 5, 'EXPOSURE_PERIOD_DAY': 5, 'CAMERA_INTERFACE': 'indi'},
        getCcdTemperature=lambda: None, getTelescopeRaDec=lambda: None, getGpsPosition=lambda: None,
        _shutdown=False, _miscDb=Mock(), reconfigureCcd=lambda: None,
        periodic_tasks_time=999999, update_time_offset=None, capture_pre_hook=lambda: None,
        sqm_camera_enable=False, focus_mode=False, image_q=output, image_queue_min=1,
        image_queue_max=3, image_queue_backoff=0.5, add_period_delay=0,
        resource_backoff=resource if resource_wait else None, _period_queue=producer, camera_id=1,
        _expUtils=SimpleNamespace(EXPOSURE_NEXT=1, EXPOSURE_CURRENT=1, GAIN_NEXT=1, BINNING_NEXT=1),
    )
    with pytest.raises(RuntimeError if stalled else EndCaptureTest):
        worker.saferun()
    assert commands
    if resource_wait:
        assert 145 <= commands[0] < 146  # Waited longer than the ordinary timeout.
    if stalled:
        assert len(aborted) == 1 and not delivered
        assert 10 < aborted[0] - commands[0] < 22.1
        assert worker._miscDb.addNotification.call_args.args[1] == 'last_ready'
    else:
        assert delivered and not aborted
        worker._miscDb.addNotification.assert_not_called()
