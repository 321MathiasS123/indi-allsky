import json

from indi_allsky.capture_health import capture_error_payload
from indi_allsky.capture_health import capture_health_status
from indi_allsky.capture_health import capture_stale_seconds
from indi_allsky.capture_health import parse_capture_error
from indi_allsky.capture_health import serialize_capture_error


def test_capture_error_serialization_is_safe_and_bounded():
    serialized = serialize_capture_error(
        'Image worker',
        'first line\nsecond line ' + ('x' * 300),
        timestamp=1000,
    )

    payload = parse_capture_error(serialized)

    assert payload['component'] == 'Image worker'
    assert '\n' not in payload['message']
    assert len(payload['message']) == 180
    assert payload['timestamp'] == 1000


def test_capture_error_parser_rejects_invalid_payloads():
    assert parse_capture_error('not-json') is None
    assert parse_capture_error(json.dumps({'component': 'Image worker'})) is None
    assert parse_capture_error(json.dumps({
        'component': 'Image worker',
        'message': 'failed',
        'timestamp': float('nan'),
    })) is None


def test_capture_stale_threshold_uses_active_capture_period():
    config = {
        'CCD_EXPOSURE_MAX': 30,
        'EXPOSURE_PERIOD': 120,
        'EXPOSURE_PERIOD_DAY': 10,
    }

    assert capture_stale_seconds(config, night=True) == 240
    assert capture_stale_seconds(config, night=False) == 60


def test_worker_error_remains_active_until_a_new_image_succeeds():
    failure = capture_error_payload('Image worker', 'failed', timestamp=100)

    failed_health = capture_health_status(
        now_timestamp=130,
        latest_success_timestamp=90,
        failure=failure,
        stale_seconds=60,
    )

    assert failed_health['status'] == 'error'
    assert failed_health['reason'] == 'worker_error'
    assert 'Image worker stopped unexpectedly 30 seconds ago.' in failed_health['message']
    assert 'Last successful image: 40 seconds ago.' in failed_health['message']


    recovered_health = capture_health_status(
        now_timestamp=130,
        latest_success_timestamp=110,
        failure=failure,
        stale_seconds=60,
    )

    assert recovered_health['status'] == 'ok'
    assert recovered_health['message'] == ''


def test_stale_capture_is_reported_after_dynamic_threshold():
    health = capture_health_status(
        now_timestamp=200,
        latest_success_timestamp=100,
        failure=None,
        stale_seconds=60,
    )

    assert health['status'] == 'error'
    assert health['reason'] == 'stale_image'
    assert '1 minute' in health['message']


def test_intentionally_inactive_capture_suppresses_old_failures():
    health = capture_health_status(
        now_timestamp=200,
        latest_success_timestamp=100,
        failure=capture_error_payload('Capture worker', 'failed', timestamp=150),
        stale_seconds=60,
        capture_expected=False,
    )

    assert health['status'] == 'ok'


def test_camera_status_error_is_reported():
    health = capture_health_status(
        now_timestamp=200,
        latest_success_timestamp=190,
        failure=None,
        stale_seconds=60,
        system_error='Capture unavailable: camera error.',
    )

    assert health['status'] == 'error'
    assert health['reason'] == 'system_error'
    assert health['message'] == 'Capture unavailable: camera error.'
