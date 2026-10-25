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
    def affinity(*values):
        if values:
            events.append(('affinity', values[0]))
        else:
            return [2, 4, 6, 8]
    process = Mock(return_value=SimpleNamespace(ionice=io, cpu_affinity=Mock(side_effect=affinity)))
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
    assert events == [('cpu', 19), ('io', 3), ('affinity', [8]), ('thread',), ('work',)]
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


@pytest.mark.parametrize('limit,available,expected', [
    (0, [2, 4, 6, 8], None),
    (2, [2, 4, 6, 8], [6, 8]),
    (8, [2, 4, 6, 8], [4, 6, 8]),
    (1, [7], None),
    (1, [1, 5], [5]),
])
def test_cpu_budget_respects_allowed_set_and_keeps_live_capacity(worker_environment,
                                                               limit, available, expected):
    worker, namespace, events, _, _ = worker_environment
    worker.config['VIDEO_CPU_LIMIT'] = limit
    affinity = namespace['psutil'].Process.return_value.cpu_affinity
    affinity.side_effect = lambda *values: available if not values else events.append(('affinity', values[0]))
    worker.run()
    assert [event for event in events if event[0] == 'affinity'] == ([] if expected is None
                                                                 else [('affinity', expected)])
    if limit == 0:
        affinity.assert_not_called()
    else:
        assert affinity.call_args_list[0].args == ()
    assert events[-2:] == [('thread',), ('work',)]


@pytest.mark.parametrize('failure', ['unsupported', 'denied', 'oserror', 'invalid_setting'])
def test_cpu_budget_failure_does_not_prevent_work(worker_environment, failure):
    worker, namespace, events, _, _ = worker_environment
    process = namespace['psutil'].Process.return_value
    if failure == 'unsupported':
        del process.cpu_affinity
    elif failure == 'denied':
        process.cpu_affinity.side_effect = psutil.AccessDenied()
    elif failure == 'oserror':
        process.cpu_affinity.side_effect = OSError('unsupported')
    else:
        worker.config['VIDEO_CPU_LIMIT'] = 'invalid'
    worker.run()
    assert events[-2:] == [('thread',), ('work',)]
    assert worker._asi676mc_calibration_q.get_nowait() is None


@pytest.mark.skipif(not hasattr(psutil.Process, 'cpu_affinity'), reason='CPU affinity unsupported')
def test_actual_child_threads_and_encoder_inherit_cpu_budget():
    before = psutil.Process().cpu_affinity()
    script = textwrap.dedent('''
        import json, logging, os, queue, signal, subprocess, sys, threading, traceback
        from pathlib import Path
        from types import SimpleNamespace
        import psutil
        Process = object
        signal = SimpleNamespace(signal=lambda *args: None, SIGHUP=1, SIGTERM=15,
                                 SIGINT=2, SIGALRM=14)
        miscDb = miscUpload = lambda *args: None
        logger = logging.getLogger('cpu-budget-test')
        exec(sys.argv[1])
        def cpus():
            if hasattr(os, 'sched_getaffinity'):
                return sorted(os.sched_getaffinity(0))
            return psutil.Process().cpu_affinity()
        inherited = cpus()
        worker = VideoWorker(1, {'IMAGE_FOLDER': '.'}, queue.Queue(), None, None, None, None)
        constructed = cpus()
        result = queue.Queue()
        worker._asi676mcCalibrationWorker = lambda: result.put(cpus())
        def work():
            child = json.loads(subprocess.check_output([sys.executable, '-c',
                'import json,psutil; print(json.dumps(psutil.Process().cpu_affinity()))'],
                text=True, timeout=5))
            print(json.dumps([inherited, constructed, cpus(), result.get(timeout=5), child]))
        worker.saferun = work
        worker.run()
    ''')
    result = subprocess.run([sys.executable, '-c', script, worker_source()],
                            text=True, capture_output=True, timeout=20)
    assert result.returncode == 0, result.stdout + result.stderr
    inherited, constructed, worker, thread, encoder = json.loads(result.stdout)
    assert constructed == inherited
    assert worker == thread == encoder == inherited[-1:]
    assert psutil.Process().cpu_affinity() == before


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
            return [p.nice(), list(p.ionice()), p.cpu_affinity()]
        before = priorities()
        reports = multiprocessing.get_context('fork').Queue()
        worker = VideoWorker(1, {'IMAGE_FOLDER': '.'}, reports, None, None, None, None)
        after_construction = priorities()
        thread_report = queue.Queue()
        worker._asi676mcCalibrationWorker = lambda: thread_report.put(priorities())
        def work():
            child = json.loads(subprocess.check_output([sys.executable, '-c',
                'import json,psutil; p=psutil.Process(); print(json.dumps([p.nice(), list(p.ionice()), p.cpu_affinity()]))'],
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
    expected = [min(19, before[0] + 19), [int(psutil.IOPRIO_CLASS_IDLE), 0], before[2][-1:]]
    assert descendants == [expected, expected, expected]
