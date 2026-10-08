"""Bounded render telemetry and emergency protection for the capture spool."""
import math
from multiprocessing import Array, Lock, Value
import os
from pathlib import Path
import shutil


class RenderBacklogState:
    """Share small measurements, never queued pixels, between the workers.

    Access is nonblocking: a terminated worker must not strand capture waiting
    on a multiprocessing lock. A missed measurement is safe to retry later.
    """

    def __init__(self):
        self._lock = Lock()
        self._seconds = Array('d', 10, lock=False)
        self._count = Value('i', 0, lock=False)
        self._next = Value('i', 0, lock=False)
        self._frame_bytes = Value('Q', 0, lock=False)
        self._path = Array('B', 4096, lock=False)
        self._path_length = Value('i', 0, lock=False)

    def observe_file(self, filename):
        """Observe the original before forwarding it to its consuming renderer."""
        try:
            path = Path(filename)
            size = path.stat().st_size
            directory = os.fsencode(str(path.resolve().parent))
        except (OSError, ValueError):
            return False
        if size <= 0 or len(directory) > len(self._path):
            return False
        if not self._lock.acquire(block=False):
            return False
        try:
            self._frame_bytes.value = size
            self._path[:len(directory)] = directory
            self._path_length.value = len(directory)
        finally:
            self._lock.release()
        return True

    def record(self, seconds):
        """Record one completed render's entire service time, excluding backlog."""
        seconds = float(seconds)
        if not math.isfinite(seconds) or seconds <= 0:
            return False
        if not self._lock.acquire(block=False):
            return False
        try:
            self._seconds[self._next.value] = seconds
            self._next.value = (self._next.value + 1) % len(self._seconds)
            self._count.value = min(self._count.value + 1, len(self._seconds))
        finally:
            self._lock.release()
        return True

    def snapshot(self):
        if not self._lock.acquire(block=False):
            return None
        try:
            count = self._count.value
            return {
                'mean_seconds': sum(self._seconds[:count]) / count if count else 0.0,
                'samples': count,
                'frame_bytes': self._frame_bytes.value,
                'spool_path': os.fsdecode(bytes(self._path[:self._path_length.value])),
            }
        finally:
            self._lock.release()


class ResourceBackoff:
    """Keep configured cadence until the actual spool filesystem needs relief.

    Call delay() before starting each exposure and check can_capture. At less
    than two frames of free space, wait and recheck instead of starting another
    exposure. Files already captured always remain available for rendering.
    """

    def __init__(self, state, queue_max):
        self.state = state
        self.queue_max = max(int(queue_max), 1)
        self.latched = False
        self.can_capture = True
        self._snapshot = None

    def delay(self, period, exposure):
        current = self.state.snapshot()
        if current is not None:
            self._snapshot = current
        if not self._snapshot or not self._snapshot['frame_bytes']:
            return 0.0
        frame_bytes = self._snapshot['frame_bytes']
        try:
            free = shutil.disk_usage(self._snapshot['spool_path']).free
        except OSError:
            # Keep an established guard while storage cannot be inspected.
            pass
        else:
            self.can_capture = free >= 2 * frame_bytes
            reserve = (self.queue_max + 2) * frame_bytes
            release = reserve + max(self.queue_max, 2) * frame_bytes
            if free < reserve:
                self.latched = True
            elif free >= release:
                self.latched = False
        target = 1.1 * self._snapshot['mean_seconds']
        if self.latched and target > max(float(period), float(exposure)):
            # The scheduler uses start + period + delay, while exposure itself
            # can independently set a longer cadence.
            return max(0.0, target - float(period))
        return 0.0
