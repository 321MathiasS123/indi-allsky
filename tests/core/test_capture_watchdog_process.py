"""Real Linux signal/blocked-process regression; also runnable with unittest."""
import ast
import multiprocessing
import os
from pathlib import Path
import signal
import time
import traceback
from types import MethodType, SimpleNamespace
import unittest

from indi_allsky.capture_watchdog import CaptureTimeoutError, CaptureWatchdog, FrameDeadline, ProcessingAllowance


def blocked_capture(deadline, reports, ignore_signal):
    path = Path(__file__).resolve().parents[2] / 'indi_allsky/capture.py'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'CaptureWorker')
    methods = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in ('run', 'frame_timeout_handler')]
    namespace = {'signal': signal, 'traceback': traceback, 'CaptureTimeoutError': CaptureTimeoutError}
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(path), 'exec'), namespace)

    def capture():
        if ignore_signal:
            signal.signal(signal.SIGUSR1, signal.SIG_IGN)
        deadline.begin_exposure(30, 15)
        reports.send(('armed', time.monotonic()))
        while True:
            time.sleep(60)  # no capture-loop checks, READY updates or frames

    worker = SimpleNamespace(
        frame_deadline=deadline, saferun=capture,
        indiclient=SimpleNamespace(
            abortCcdExposure=lambda: reports.send(('abort', time.monotonic())),
            disconnectServer=lambda: None,
        ),
        error_q=SimpleNamespace(put=lambda error: None),
        sighup_handler_worker=lambda *args: None,
        sigterm_handler_worker=lambda *args: None,
        sigint_handler_worker=lambda *args: None,
        sigalarm_handler_worker=lambda *args: None,
    )
    worker.frame_timeout_handler = MethodType(namespace['frame_timeout_handler'], worker)
    try:
        namespace['run'](worker)
    except CaptureTimeoutError:
        pass


@unittest.skipUnless(os.name == 'posix', 'Capture recovery uses Linux process signals')
class CaptureWatchdogProcessTest(unittest.TestCase):
    def test_actual_80_second_deadline_for_blocked_and_unresponsive_workers(self):
        runs = []
        try:
            for ignore_signal in (False, True):
                processing = ProcessingAllowance()
                processing.record(20)
                # Automatic: 2 * max(30, 15) + 20 = 80. The unresponsive
                # worker uses the adjustable fixed 80-second override.
                deadline = FrameDeadline(80 if ignore_signal else 0, processing)
                receiver, sender = multiprocessing.Pipe(duplex=False)
                worker = multiprocessing.Process(target=blocked_capture, args=(deadline, sender, ignore_signal))
                worker.start()
                self.assertTrue(receiver.poll(10), 'capture did not arm its deadline')
                event, started = receiver.recv()
                self.assertEqual(event, 'armed')
                watchdog = CaptureWatchdog(worker, deadline)
                watchdog.start()
                runs.append((worker, watchdog, receiver, started, ignore_signal))

            # The parent main thread is blocked here. Its watchdog threads must
            # still recover both children without the parent's 13-second loop.
            for worker, watchdog, receiver, started, ignore_signal in runs:
                worker.join(timeout=max(0, started + 85 - time.monotonic()))
                self.assertFalse(worker.is_alive(), 'capture survived its recovery deadline')
                if ignore_signal:
                    self.assertEqual(worker.exitcode, -signal.SIGKILL)
                    self.assertLessEqual(time.monotonic() - started, 85)
                else:
                    self.assertTrue(receiver.poll(1), 'abort was not attempted')
                    event, aborted = receiver.recv()
                    self.assertEqual(event, 'abort')
                    self.assertGreaterEqual(aborted - started, 79.9)
                    self.assertLessEqual(aborted - started, 85)
                print('Capture watchdog ({0}): recovery after {1:.3f}s'.format(
                    'unresponsive' if ignore_signal else 'cooperative',
                    time.monotonic() - started if ignore_signal else aborted - started,
                ), flush=True)
        finally:
            for worker, watchdog, receiver, _, _ in runs:
                if worker.is_alive():
                    worker.kill()
                worker.join(timeout=2)
                watchdog.stop()
                receiver.close()


if __name__ == '__main__':
    unittest.main()
