import os
import math
import signal
import time
from multiprocessing import Array
from threading import Event, Thread


class CaptureTimeoutError(TimeoutError):
    pass


class ProcessingAllowance:
    """Recent processing peak, shared without database access in the watchdog."""

    def __init__(self):
        self._samples = Array('d', [0.0] * 11)  # ring index and ten durations
        self._last_peak = 10.0  # startup allowance until a frame is processed

    def record(self, seconds):
        if not math.isfinite(seconds) or seconds <= 0:
            return
        lock = self._samples.get_lock()
        if not lock.acquire(timeout=0.05):
            return
        try:
            index = int(self._samples[0])
            self._samples[index + 1] = seconds
            self._samples[0] = (index + 1) % 10
        finally:
            lock.release()

    def maximum(self):
        lock = self._samples.get_lock()
        if not lock.acquire(timeout=0.05):
            return self._last_peak
        try:
            self._last_peak = max(self._samples[1:]) or 10.0
            return self._last_peak
        finally:
            lock.release()


class FrameDeadline:
    """Frame progress shared by the capture process and its parent."""

    def __init__(self, timeout, processing=None):
        self.timeout = float(timeout)
        self.processing = processing or ProcessingAllowance()
        # deadline (zero means suspended), awaiting frame, last frame/start,
        # scheduled start of the next exposure, exposure, period.
        # Never share this lock with
        # another worker; a forcibly stopped capture gets a fresh instance.
        self._state = Array('d', [0.0] * 6)

    def _timeout(self):
        if self.timeout:
            return self.timeout
        return 2 * max(self._state[4], self._state[5]) + self.processing.maximum()

    def _deadline(self):
        reference = self._state[2]
        # Ordinary intervals are already inside the fixed timeout. Only defer
        # for an intentional idle gap that itself exceeds the whole timeout.
        if self.timeout and self._state[3] > reference + self.timeout:
            reference = self._state[3]
        return reference + self._timeout()

    def begin_exposure(self, exposure=0.0, period=0.0):
        now = time.monotonic()
        with self._state.get_lock():
            if self._state[0] and self._state[1]:
                return
            starting = not self._state[0]
            self._state[4] = exposure
            self._state[5] = period
            self._state[3] = now
            if starting:
                self._state[2] = now
            if self.timeout and not starting:
                self._state[0] = min(self._state[0], now + self.timeout)
            else:
                self._state[0] = self._deadline()
            # Reissuing an exposure without receiving a frame cannot postpone
            # the deadline, even if the driver keeps reporting READY.
            self._state[1] = 1.0

    def received(self):
        now = time.monotonic()
        with self._state.get_lock():
            if self._state[0]:
                self._state[2] = now
                self._state[1] = 0.0
                self._state[0] = self._deadline()

    def schedule_next(self, when, period=None):
        with self._state.get_lock():
            self._state[3] = when
            if period is not None:
                self._state[5] = period
            if self._state[0] and not self._state[1]:
                self._state[0] = self._deadline()

    def suspend(self):
        with self._state.get_lock():
            self._state[0] = 0.0

    def snapshot(self):
        # A stuck child must never hold up its supervisor. On contention the
        # supervisor keeps the last deadline it successfully read.
        lock = self._state.get_lock()
        if not lock.acquire(timeout=0.05):
            return None
        try:
            return self._state[0]
        finally:
            lock.release()


class FrameArrivalQueue:
    """Observe complete camera frames without depending on image processing."""

    def __init__(self, image_queue, deadline):
        self.image_queue = image_queue
        self.deadline = deadline

    def put(self, frame):
        self.image_queue.put(frame)
        self.deadline.received()


class CaptureWatchdog(Thread):
    check_interval = 1.0
    stop_grace = 3.0

    def __init__(self, worker, deadline):
        super().__init__(name='CaptureWatchdog', daemon=True)
        self.worker = worker
        self.deadline = deadline
        self.timed_out = False
        self._stop_event = Event()

    def stop(self):
        self._stop_event.set()
        self.join(timeout=self.stop_grace + 1.0)

    def run(self):
        deadline = 0.0
        while not self._stop_event.wait(self.check_interval):
            if not self.worker.is_alive():
                return

            current = self.deadline.snapshot()
            if current is not None:
                deadline = current
            if not deadline or time.monotonic() < deadline:
                continue

            self.timed_out = True
            try:
                # Unwind the capture loop before attempting an abort. No DB,
                # logging or camera calls may delay this independent deadline.
                os.kill(self.worker.pid, signal.SIGUSR1)
            except ProcessLookupError:
                return

            self._stop_event.wait(self.stop_grace)
            if self.worker.is_alive():
                # Also handles a driver stuck in native code or a stuck abort.
                self.worker.kill()
            return
