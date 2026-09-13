"""The automation consumer receives the same age and expectation as the page."""
from indi_allsky.capture_health import capture_health_status


def test_capture_health_includes_machine_readable_evidence():
    result = capture_health_status(1000, 300, None, 60, capture_expected=True)
    assert result['last_success_timestamp'] == 300
    assert result['last_success_age_s'] == 700
    assert result['stale_seconds'] == 60
    assert result['capture_expected'] is True
    assert result['reason'] == 'stale_image'


def test_paused_capture_keeps_age_without_recommending_recovery():
    result = capture_health_status(1000, 300, None, 60, capture_expected=False)
    assert result['last_success_age_s'] == 700
    assert result['capture_expected'] is False
    assert result['status'] == 'ok'
