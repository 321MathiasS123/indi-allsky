"""Worker scheduling checks without starting Flask or a camera service."""

import ast
import json
import logging
from pathlib import Path
import queue
import subprocess
import sys
import textwrap
from types import SimpleNamespace
from unittest.mock import Mock

import psutil
import pytest


SOURCE = Path(__file__).resolve().parents[2] / 'indi_allsky' / 'video.py'


def worker_source():
    tree = ast.parse(SOURCE.read_text(encoding='utf-8'))
    worker = next(node for node in tree.body
                  if isinstance(node, ast.ClassDef) and node.name == 'VideoWorker')
    worker.body = [node for node in worker.body if isinstance(node, ast.FunctionDef)
                   and (node.name in ('__init__', 'run') or node.name.endswith('_handler_worker'))]
    return ast.unparse(worker)


@pytest.fixture
def worker_environment():
    events = []
    cpu = Mock(side_effect=lambda value: events.append(('cpu', value)))
    io = Mock(side_effect=lambda value: events.append(('io', value)))
    process = Mock(return_value=SimpleNamespace(ionice=io))
    thread = Mock()
    thread.start.side_effect = lambda: events.append(('thread',))
    namespace = {
        'Process': object,
        'os': SimpleNamespace(nice=cpu),
        'psutil': SimpleNamespace(Process=process, IOPRIO_CLASS_IDLE=3, Error=psutil.Error),
        'miscDb': Mock(), 'miscUpload': Mock(), 'Path': Path,
        'signal': SimpleNamespace(signal=Mock(), SIGHUP=1, SIGTERM=15, SIGINT=2, SIGALRM=14),
        'threading': SimpleNamespace(Thread=Mock(return_value=thread)),
        'queue': queue, 'logger': logging.getLogger(__name__),
    }
    exec(compile(worker_source(), str(SOURCE), 'exec'), namespace)
    worker = namespace['VideoWorker'](1, {'IMAGE_FOLDER': '.'}, queue.Queue(),
                                      queue.Queue(), queue.Queue(), None, None)
    worker._asi676mcCalibrationWorker = lambda: None
    worker.saferun = lambda: events.append(('work',))
    return worker, namespace, events, cpu, io


def test_constructor_does_not_change_supervisor_priority(worker_environment):
    _, namespace, events, cpu, io = worker_environment
    cpu.assert_not_called()
    io.assert_not_called()
    namespace['psutil'].Process.assert_not_called()
    assert events == []


def test_child_sets_priorities_before_starting_threads_and_work(worker_environment):
    worker, _, events, _, _ = worker_environment
    worker.run()
    assert events == [('cpu', 19), ('io', 3), ('thread',), ('work',)]
    assert worker._asi676mc_calibration_q.get_nowait() is None


@pytest.mark.parametrize('failure', ['cpu_denied', 'cpu_unsupported', 'io_denied',
                                     'io_oserror', 'io_unsupported', 'io_constant_missing'])
def test_priority_failure_does_not_prevent_work(worker_environment, failure):
    worker, namespace, events, cpu, io = worker_environment
    if failure == 'cpu_denied':
        cpu.side_effect = PermissionError('denied')
    elif failure == 'cpu_unsupported':
        del namespace['os'].nice
    elif failure == 'io_denied':
        io.side_effect = psutil.AccessDenied()
    elif failure == 'io_oserror':
        io.side_effect = OSError('unsupported')
    elif failure == 'io_unsupported':
        namespace['psutil'].Process.return_value = object()
    else:
        del namespace['psutil'].IOPRIO_CLASS_IDLE
    worker.run()
    assert events[-2:] == [('thread',), ('work',)]
    if failure.startswith('cpu'):
        io.assert_called_once_with(3)
    else:
        cpu.assert_called_once_with(19)
    assert worker._asi676mc_calibration_q.get_nowait() is None


@pytest.mark.skipif(not sys.platform.startswith('linux'), reason='Linux scheduling inheritance')
def test_linux_supervisor_unchanged_and_worker_descendants_inherit_priority():
    # Isolate even a failing constructor in a disposable interpreter: no test
    # runner or unrelated process may be reniced by this regression check.
    script = textwrap.dedent('''
        import json, logging, multiprocessing, os, queue, signal, subprocess, sys, threading, traceback
        from pathlib import Path
        import psutil
        Process = multiprocessing.get_context('fork').Process
        miscDb = miscUpload = lambda *args: None
        logger = logging.getLogger('priority-test')
        exec(sys.argv[1])
        def priorities():
            p = psutil.Process(threading.get_native_id())
            return [p.nice(), list(p.ionice())]
        before = priorities()
        reports = multiprocessing.get_context('fork').Queue()
        worker = VideoWorker(1, {'IMAGE_FOLDER': '.'}, reports, None, None, None, None)
        after_construction = priorities()
        thread_report = queue.Queue()
        worker._asi676mcCalibrationWorker = lambda: thread_report.put(priorities())
        def work():
            child = json.loads(subprocess.check_output([sys.executable, '-c',
                'import json,psutil; p=psutil.Process(); print(json.dumps([p.nice(), list(p.ionice())]))'],
                text=True, timeout=5))
            reports.put([priorities(), thread_report.get(timeout=5), child])
        worker.saferun = work
        worker.start()
        worker.join(10)
        if worker.is_alive():
            worker.kill()
            worker.join(5)
            raise RuntimeError('worker failed to exit')
        assert worker.exitcode == 0
        report = reports.get(timeout=5)
        print(json.dumps([before, after_construction, priorities(), report]))
    ''')
    result = subprocess.run([sys.executable, '-c', script, worker_source()],
                            text=True, capture_output=True, timeout=25)
    assert result.returncode == 0, result.stdout + result.stderr
    before, constructed, after, descendants = json.loads(result.stdout)
    assert constructed == before == after
    expected = [min(19, before[0] + 19), [int(psutil.IOPRIO_CLASS_IDLE), 0]]
    assert descendants == [expected, expected, expected]
