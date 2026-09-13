import json
import math
import time


CAPTURE_PIPELINE_ERROR_STATE = 'CAPTURE_PIPELINE_ERROR'

MIN_STALE_SECONDS = 60
MAX_ERROR_MESSAGE_LENGTH = 180


def _positive_number(value, default):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)

    if not math.isfinite(number) or number <= 0:
        return float(default)

    return number


def capture_stale_seconds(config, night):
    exposure_max = _positive_number(config.get('CCD_EXPOSURE_MAX'), 15.0)

    if night:
        period = _positive_number(config.get('EXPOSURE_PERIOD'), 15.0)
    else:
        period = _positive_number(config.get('EXPOSURE_PERIOD_DAY'), 15.0)

    # Allow two complete capture cycles before declaring the output stale.
    return int(max(MIN_STALE_SECONDS, 2 * max(exposure_max, period)))


def capture_error_payload(component, error, timestamp=None):
    if timestamp is None:
        timestamp = time.time()

    message = ' '.join(str(error).split())
    if not message:
        message = 'Unknown worker error'

    return {
        'component': str(component),
        'message': message[:MAX_ERROR_MESSAGE_LENGTH],
        'timestamp': float(timestamp),
    }


def serialize_capture_error(component, error, timestamp=None):
    return json.dumps(
        capture_error_payload(component, error, timestamp=timestamp),
        separators=(',', ':'),
        sort_keys=True,
    )


def parse_capture_error(value):
    try:
        payload = json.loads(value)
        component = str(payload['component']).strip()
        message = str(payload['message']).strip()
        timestamp = float(payload['timestamp'])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None

    if not component or not message:
        return None

    if not math.isfinite(timestamp) or timestamp <= 0:
        return None

    return {
        'component': component,
        'message': message,
        'timestamp': timestamp,
    }


def format_age(seconds):
    seconds = max(0, int(seconds))

    if seconds < 60:
        unit = 'second' if seconds == 1 else 'seconds'
        return '{0:d} {1:s}'.format(seconds, unit)

    minutes = int(seconds / 60)
    if minutes < 60:
        unit = 'minute' if minutes == 1 else 'minutes'
        return '{0:d} {1:s}'.format(minutes, unit)

    hours = int(minutes / 60)
    unit = 'hour' if hours == 1 else 'hours'
    return '{0:d} {1:s}'.format(hours, unit)


def capture_health_status(
        now_timestamp,
        latest_success_timestamp,
        failure,
        stale_seconds,
        capture_expected=True,
        system_error=None):
    health = {
        'status': 'ok',
        'reason': '',
        'message': '',
        'last_success_age_s': None,
    }

    now_timestamp = float(now_timestamp)

    if latest_success_timestamp is not None:
        latest_success_timestamp = float(latest_success_timestamp)
        health['last_success_age_s'] = max(
            0,
            int(now_timestamp - latest_success_timestamp),
        )

    if not capture_expected and not system_error:
        return health

    if failure:
        failure_timestamp = float(failure['timestamp'])
        failure_is_active = (
            latest_success_timestamp is None
            or failure_timestamp >= latest_success_timestamp
        )

        if failure_is_active:
            message = '{0:s} stopped unexpectedly {1:s} ago.'.format(
                failure['component'],
                format_age(now_timestamp - failure_timestamp),
            )

            if health['last_success_age_s'] is not None:
                message += ' Last successful image: {0:s} ago.'.format(
                    format_age(health['last_success_age_s']),
                )

            message += ' Check Capture Log.'

            health.update({
                'status': 'error',
                'reason': 'worker_error',
                'message': message,
            })
            return health

    if system_error:
        health.update({
            'status': 'error',
            'reason': 'system_error',
            'message': str(system_error),
        })
        return health

    if not capture_expected or health['last_success_age_s'] is None:
        return health

    if health['last_success_age_s'] > stale_seconds:
        health.update({
            'status': 'error',
            'reason': 'stale_image',
            'message': (
                'Capture appears stalled. No successful image for {0:s}. '
                'Check Capture Log.'
            ).format(format_age(health['last_success_age_s'])),
        })

    return health
