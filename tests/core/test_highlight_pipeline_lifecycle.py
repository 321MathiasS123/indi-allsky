"""Exercise parent worker ordering without loading platform-specific services."""
import ast
import logging
from pathlib import Path
from queue import Queue
from types import SimpleNamespace

import pytest

from indi_allsky.capture_control import request_worker_stop


def parent_methods():
    tree = ast.parse((Path(__file__).parents[2] / 'indi_allsky' / 'allsky.py').read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'IndiAllSky')
    names = {'_highlightMeterEnabled', '_stopHighlightWorker', '_stopImageWorker', '_stopCaptureWorker',
             '_requestCaptureWorkerStop'}
    code = ast.Module(body=[n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names], type_ignores=[])
    namespace = {'logger': logging.getLogger(__name__), 'request_worker_stop': request_worker_stop}
    exec(compile(code, '<parent lifecycle>', 'exec'), namespace)
    return type('Parent', (), {n: namespace[n] for n in names})


class Worker:
    def __init__(self, events, name, alive=True):
        self.events, self.name, self.alive = events, name, alive
        self.exitcode = 0

    def is_alive(self):
        return self.alive

    def join(self, timeout=None):
        self.events.append(self.name + '-drained')
        self.alive = False

    def terminate(self):
        self.events.append(self.name + '-terminated')


class RecordedQueue(Queue):
    def __init__(self, events, name):
        super().__init__()
        self.events, self.name = events, name

    def put(self, value):
        self.events.append((self.name, value))
        super().put(value)


@pytest.mark.parametrize('enabled,focus,expected', [(True, False, True), (False, False, False), (True, True, False)])
def test_meter_selection_follows_current_config(enabled, focus, expected):
    parent = parent_methods()()
    parent.config = {'HIGHLIGHT_PROTECTION': {'ENABLE': enabled}, 'FOCUS_MODE': focus}
    assert parent._highlightMeterEnabled() is expected


def test_normal_stop_drains_meter_before_render_stop():
    events = []
    parent = parent_methods()()
    parent._terminate = False
    parent.highlight_worker = Worker(events, 'meter')
    parent.image_worker = Worker(events, 'render')
    parent.highlight_input_q = RecordedQueue(events, 'input')
    parent.image_q = RecordedQueue(events, 'images')
    parent._stopImageWorker()
    assert events == [('input', {'stop': True}), 'meter-drained', ('images', {'stop': True}), 'render-drained']
    assert parent.highlight_worker is None


def test_failed_renderer_is_restarted_to_drain_captured_files():
    events = []
    parent = parent_methods()()
    parent._terminate = False
    parent.highlight_worker = Worker(events, 'meter')
    parent.image_worker = Worker(events, 'old-render', alive=False)
    parent.highlight_input_q = RecordedQueue(events, 'input')
    parent.image_q = RecordedQueue(events, 'images')

    def restart():
        events.append('render-restarted')
        parent.image_worker = Worker(events, 'new-render')

    parent._startImageWorker = restart
    parent._stopImageWorker()
    assert events == ['render-restarted', ('input', {'stop': True}), 'meter-drained',
                      ('images', {'stop': True}), 'new-render-drained']


def test_failed_meter_is_restarted_before_graceful_drain():
    events = []
    parent = parent_methods()()
    parent._terminate = False
    parent.highlight_worker = Worker(events, 'old-meter', alive=False)
    parent.highlight_input_q = RecordedQueue(events, 'input')

    def restart():
        events.append('meter-restarted')
        parent.highlight_worker = Worker(events, 'new-meter')

    parent._startHighlightWorker = restart
    parent._stopHighlightWorker()
    assert events == ['meter-restarted', ('input', {'stop': True}), 'new-meter-drained']


def test_forced_stop_does_not_leave_sentinels_in_reused_queues():
    events = []
    parent = parent_methods()()
    parent._terminate = True
    parent.highlight_worker = Worker(events, 'meter')
    parent.highlight_input_q = RecordedQueue(events, 'input')
    parent._stopHighlightWorker()
    assert events == ['meter-terminated', 'meter-drained']
    assert parent.highlight_input_q.empty()


def test_capture_drain_starts_consumers_before_waiting_for_queue_flush():
    events = []
    parent = parent_methods()()
    parent._terminate = False
    parent._capture_worker_stop_requested = False
    parent.capture_watchdog = None
    parent.capture_worker = Worker(events, 'capture')
    parent.capture_q = RecordedQueue(events, 'capture-control')
    parent._startImageWorker = lambda: events.append('consumers-ready')
    parent._stopCaptureWorker()
    assert events == ['consumers-ready', ('capture-control', {'stop': True}), 'capture-drained']


def test_meter_failure_during_drain_restarts_without_duplicate_stop():
    events = []
    parent = parent_methods()()
    parent._terminate = False
    parent.highlight_worker = Worker(events, 'failed-meter')
    parent.highlight_worker.exitcode = 1
    parent.highlight_input_q = RecordedQueue(events, 'input')

    def restart():
        events.append('meter-restarted')
        parent.highlight_worker = Worker(events, 'new-meter')

    parent._startHighlightWorker = restart
    parent._stopHighlightWorker()
    assert events == [('input', {'stop': True}), 'failed-meter-drained',
                      'meter-restarted', 'new-meter-drained']


def test_renderer_failure_during_drain_does_not_restart_stopped_meter():
    events = []
    parent = parent_methods()()
    parent._terminate = False
    parent.highlight_worker = None
    parent.image_worker = Worker(events, 'failed-render')
    parent.image_worker.exitcode = 1
    parent.image_q = RecordedQueue(events, 'images')

    def restart(manage_meter=True):
        assert manage_meter is False
        events.append('render-restarted')
        parent.image_worker = Worker(events, 'new-render')

    parent._startImageWorker = restart
    parent._stopImageWorker()
    assert events == [('images', {'stop': True}), 'failed-render-drained',
                      'render-restarted', 'new-render-drained']
