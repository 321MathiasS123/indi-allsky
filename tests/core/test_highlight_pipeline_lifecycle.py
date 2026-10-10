"""Exercise parent worker ordering without loading platform-specific services."""
import ast
import logging
from pathlib import Path
from queue import Queue
import subprocess
import sys
import textwrap

import pytest

from indi_allsky.capture_period import failure_key, read_inflight, set_inflight


def parent_methods():
    tree = ast.parse((Path(__file__).parents[2] / 'indi_allsky' / 'allsky.py').read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'IndiAllSky')
    names = {'_highlightMeterEnabled', '_stopHighlightWorker', '_stopImageWorker',
             '_stopCaptureWorker', '_resetHighlightFeedbackQueue', '_settleCapturePeriods'}
    code = ast.Module(body=[n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names], type_ignores=[])
    namespace = {'logger': logging.getLogger(__name__), 'Queue': FeedbackQueue,
                 'read_inflight': read_inflight, 'set_inflight': set_inflight, 'failure_key': failure_key}
    exec(compile(code, '<parent lifecycle>', 'exec'), namespace)
    methods = {n: namespace[n] for n in names}
    def init(self):
        self.highlight_feedback_q = FeedbackQueue()
        self.capture_receipts = ()
        self._capture_worker_stop_requested = False
    methods['__init__'] = init
    return type('Parent', (), methods)


class FeedbackQueue(Queue):
    def close(self):
        self.closed = True


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


@pytest.mark.parametrize('terminate,renderer', [
    (False, 'alive'), (True, 'alive'), (False, 'absent'),
    (False, 'dead'), (True, 'dead'),
])
def test_stop_replaces_only_feedback_after_both_workers_stop(terminate, renderer):
    events = []
    parent = parent_methods()()
    parent._terminate = terminate
    parent.highlight_worker = Worker(events, 'meter')
    parent.image_worker = (None if renderer == 'absent' else
                           Worker(events, 'render', alive=renderer == 'alive'))
    parent._startImageWorker = lambda: setattr(parent, 'image_worker', Worker(events, 'replacement'))
    old_feedback = parent.highlight_feedback_q
    old_feedback.put({'exp_time': 100, 'measurement': 'old configuration'})
    image_queue, input_queue = Queue(), Queue()
    parent.image_q, parent.highlight_input_q = image_queue, input_queue
    image_queue.put({'filename': 'retained.fit'})
    input_queue.put({'filename': 'fresh.fit'})

    def close_feedback():
        assert parent.highlight_worker is None
        assert parent.image_worker is None or not parent.image_worker.is_alive()
        events.append('feedback-closed')

    old_feedback.close = close_feedback
    parent._stopImageWorker()
    assert events[-1] == 'feedback-closed'
    assert parent.highlight_feedback_q is not old_feedback
    assert parent.highlight_feedback_q.empty()
    assert parent.image_q is image_queue
    assert parent.highlight_input_q is input_queue
    assert image_queue.get_nowait() == {'filename': 'retained.fit'}
    assert input_queue.get_nowait() == {'filename': 'fresh.fit'}


def test_renderer_exit_drops_unused_feedback_but_flushes_frame_queue(tmp_path):
    # A real spawned child is essential: Queue.put() alone cannot expose the
    # feeder-thread join that occurs only when the producing process exits.
    script = tmp_path / 'renderer_feedback_exit.py'
    script.write_text(textwrap.dedent('''\
        import ast
        import multiprocessing as mp
        from pathlib import Path
        from queue import Queue
        import sys
        import traceback
        from types import SimpleNamespace

        def render(source, feedback, frames):
            tree = ast.parse(Path(source).read_text())
            cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'ImageWorker')
            run = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'run')
            signal = SimpleNamespace(SIGHUP=1, SIGTERM=2, SIGINT=3, SIGALRM=4,
                                     signal=lambda *args: None)
            namespace = {'signal': signal, 'traceback': traceback}
            exec(compile(ast.Module(body=[run], type_ignores=[]), source, 'exec'), namespace)
            def work():
                # Far beyond pipe capacity, intentionally never consumed.
                for index in range(4096):
                    feedback.put((index, b'x' * 4096))
                for index in range(400):
                    frames.put((index, b'y' * 2048))
            worker = SimpleNamespace(
                highlight_feedback_q=feedback, error_q=Queue(), saferun=work,
                sighup_handler_worker=lambda *args: None,
                sigterm_handler_worker=lambda *args: None,
                sigint_handler_worker=lambda *args: None,
                sigalarm_handler_worker=lambda *args: None,
            )
            namespace['run'](worker)

        if __name__ == '__main__':
            context = mp.get_context('spawn')
            feedback, frames = context.Queue(), context.Queue()
            child = context.Process(target=render, args=(sys.argv[1], feedback, frames))
            child.start()
            try:
                for index in range(400):
                    assert frames.get(timeout=10) == (index, b'y' * 2048)
                child.join(timeout=5)
                assert not child.is_alive(), 'renderer waits for feedback with no consumer'
                assert child.exitcode == 0
            finally:
                if child.is_alive():
                    child.terminate()
                    child.join(timeout=5)
                feedback.close()
                frames.close()
    '''))
    source = Path(__file__).parents[2] / 'indi_allsky' / 'image.py'
    result = subprocess.run([sys.executable, str(script), str(source)],
                            capture_output=True, text=True, timeout=25)
    assert result.returncode == 0, result.stdout + result.stderr
