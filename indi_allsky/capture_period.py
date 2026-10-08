"""Keep frame metadata and period-end work ordered at camera delivery."""

from threading import RLock
from uuid import uuid4
import hashlib
import json
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

    def __init__(self, image_queue, temperature=None, passive=False):
        self.image_queue = image_queue
        self._temperature = temperature
        self._passive = passive
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
            if self._passive:
                # An external client owns the exposures; this call merely
                # updates camera metadata and cannot reserve the next image.
                return
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
            if self._passive:
                self.image_queue.put(frame)
                return
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
        with self._lock:
            marker = {'period_end': {
                'period_id': period_id(camera_id, day_date, night),
                'tasks': list(tasks),
            }}
            if not self._passive:
                marker['period_end'].update(stream_id=self._stream_id, last_sequence=self._sequence)
            if self._waiting:
                self._closing.append(marker)
            else:
                self.image_queue.put(marker)


class CaptureSequenceTracker:
    """Check the delivered prefix, including SQM frames and its closing marker.

    A multiprocessing Queue producer can die after put() but before its feeder
    transmits the frame. These sequence numbers detect that missing frame even
    when the producer has already cleared its own in-flight receipt. Preserve
    the consumed prefix across renderer replacement; a checksum makes a torn
    shared-buffer write fail closed instead of accepting a plausible prefix.
    """

    def __init__(self, shared=None):
        self.shared = shared
        self.current = None
        self.invalid_checkpoint = False
        if shared is not None and shared.value:
            try:
                encoded, digest = shared.value.rsplit(b'\n', 1)
                if hashlib.sha256(encoded).hexdigest().encode('ascii') != digest:
                    raise ValueError('Incomplete prefix checkpoint')
                saved = json.loads(encoded)
                self.current = self._position(saved, marker=True)
                if self.current is None:
                    raise ValueError('Missing prefix checkpoint')
            except (ValueError, TypeError, KeyError, UnicodeDecodeError):
                self.invalid_checkpoint = True

    @staticmethod
    def _position(item, marker=False):
        stream_key, sequence_key, period_key = (
            ('stream_id', 'last_sequence', 'period_id') if marker else
            ('capture_stream_id', 'capture_sequence', 'capture_period_id')
        )
        # Standalone/legacy image callers do not belong to a capture stream.
        if stream_key not in item and sequence_key not in item:
            return None
        stream, sequence, identifier = item[stream_key], item[sequence_key], item[period_key]
        if (not isinstance(stream, str) or not re.fullmatch(r'[0-9a-f]{32}', stream)
                or type(sequence) is not int or sequence < (0 if marker else 1)
                or not isinstance(identifier, str)
                or not re.fullmatch(r'[0-9]+:[0-9]{4}-[0-9]{2}-[0-9]{2}:[01]', identifier)):
            raise ValueError('Invalid capture stream position')
        return stream, sequence, identifier

    def check(self, item, marker=False):
        failed = {'UNKNOWN'} if self.invalid_checkpoint else set()
        try:
            position = self._position(item, marker)
        except (KeyError, TypeError, ValueError):
            return failed | {'UNKNOWN'}
        if position is None:
            return failed
        stream, sequence, identifier = position
        previous = self.current
        expected = previous[1] if previous and previous[0] == stream else 0
        if not marker:
            expected += 1
        if sequence != expected:
            failed.add(identifier)
            if previous:
                # A gap can straddle sunset/sunrise. Neither side may claim
                # completion when the missing frame's period is unknown.
                failed.add(previous[2])
        return failed

    def complete(self, item, marker=False):
        try:
            position = self._position(item, marker)
        except (KeyError, TypeError, ValueError):
            return
        if position is None:
            return
        if self.current and self.current[0] == position[0] and self.current[1] > position[1]:
            return  # Never rewind after a duplicate/out-of-order item.
        if self.shared is not None:
            encoded = json.dumps(dict(stream_id=position[0], last_sequence=position[1],
                                      period_id=position[2]), separators=(',', ':')).encode('ascii')
            payload = encoded + b'\n' + hashlib.sha256(encoded).hexdigest().encode('ascii')
            if len(payload) >= len(self.shared):
                self.shared.value = b'UNKNOWN'
                raise ValueError('Capture prefix checkpoint is too long')
            self.shared.value = payload
        self.current = position
        self.invalid_checkpoint = False


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
