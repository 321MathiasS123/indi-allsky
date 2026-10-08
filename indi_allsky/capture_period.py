"""Keep frame metadata and period-end work ordered at camera delivery."""

from threading import RLock
from uuid import uuid4
import re


def period_id(camera_id, day_date, night):
    return '{0}:{1}:{2}'.format(camera_id, day_date, int(bool(night)))


class CapturePeriodQueue:
    """One camera producer, including asynchronous callbacks, feeding one FIFO.

    A camera's READY property can precede its image callback.  Reserve metadata
    before exposure starts and close an outgoing period only after that callback
    has queued the actual frame.  The lock also orders callback and capture-loop
    writes to the multiprocessing queue.
    """

    def __init__(self, image_queue, temperature=None):
        self.image_queue = image_queue
        self._temperature = temperature
        self._lock = RLock()
        self._context = None
        self._waiting = False
        self._closing = []
        self._sequence = 0
        self._stream_id = uuid4().hex

    @property
    def waiting(self):
        with self._lock:
            return self._waiting

    def begin(self, camera_id, mode, day_date, period):
        with self._lock:
            if self._waiting:
                raise RuntimeError('Previous camera frame has not arrived; capture period incomplete')
            self._sequence += 1
            self._context = {
                'capture_mode': tuple(mode),
                'capture_day_date': day_date.isoformat(),
                'capture_period': float(period),
                'capture_period_id': period_id(camera_id, day_date.isoformat(), mode[0]),
                'capture_stream_id': self._stream_id,
                'capture_sequence': self._sequence,
            }
            self._waiting = True

    def put(self, frame):
        with self._lock:
            if not self._waiting:
                # Do not silently attach an unsolicited/late callback to a
                # different exposure or put it after a completed period.
                raise RuntimeError('Camera frame arrived without an active exposure; capture period incomplete')
            item = dict(frame)
            item.update(self._context)
            if self._temperature is not None:
                item['capture_temperature'] = float(self._temperature())
            self.image_queue.put(item)
            self._waiting = False
            for marker in self._closing:
                self.image_queue.put(marker)
            self._closing.clear()

    def end_period(self, camera_id, day_date, night, tasks):
        marker = {'period_end': {
            'period_id': period_id(camera_id, day_date, night),
            'tasks': list(tasks),
        }}
        with self._lock:
            if self._waiting:
                self._closing.append(marker)
            else:
                self.image_queue.put(marker)


def frame_mode(frame, live_mode):
    """Legacy callers retain their existing fallback; captured frames are fixed."""
    return tuple(frame.get('capture_mode', live_mode))


class FrameTemperatureValues:
    """Freeze the calibration temperature without changing other sensor slots."""

    def __init__(self, live_values, ccd_slot):
        self.live_values = live_values
        self.ccd_slot = ccd_slot
        self.temperature = live_values[ccd_slot]

    def __getitem__(self, slot):
        return self.temperature if slot == self.ccd_slot else self.live_values[slot]

    def __setitem__(self, slot, value):
        self.live_values[slot] = value

    def __len__(self):
        return len(self.live_values)

    def get_lock(self):
        return self.live_values.get_lock()

    def set_frame(self, frame):
        self.temperature = frame.get('capture_temperature', self.live_values[self.ccd_slot])


def failure_key(identifier):
    return 'CAPTURE_PERIOD_FAILURE_{0}'.format(identifier)


def read_inflight(receipt):
    if receipt is None or not receipt.value:
        return None
    try:
        identifier = receipt.value.decode('utf-8')
    except UnicodeDecodeError:
        return 'UNKNOWN'
    if not re.fullmatch(r'[0-9]+:[0-9]{4}-[0-9]{2}-[0-9]{2}:[01]', identifier):
        return 'UNKNOWN'
    return identifier


def set_inflight(receipt, identifier):
    if receipt is None:
        return
    value = identifier.encode('utf-8') if identifier else b''
    if len(value) >= len(receipt):
        receipt.value = b'UNKNOWN'
        raise ValueError('Capture period receipt is too long; refusing unsafe completion')
    receipt.value = value
