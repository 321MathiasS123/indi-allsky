"""Closed-loop scenes and actuator boundaries beyond the initial smoke tests."""
import numpy as np
import pytest

from indi_allsky.highlight import HighlightMeasurement, exposure_decision, measure
from test_highlight_exposure import MODE_NAMES, controller


@pytest.mark.parametrize('name', MODE_NAMES)
def test_pre_dark_clipping_reduces_signal_when_calibration_hides_the_plateau(name, caplog):
    raw = np.full((100, 100), 70 * 256 + 3000, dtype=np.uint16)
    raw[20:30, 20:40] = 65535
    calibrated = raw - 3000
    mask = np.ones(raw.shape, dtype=np.uint8)
    old = measure(calibrated, mask, 16)
    corrected = measure(raw, mask, 16)._replace(adu=old.adu)
    assert old.full == old.any == 0
    assert corrected.full == corrected.any == 2
    assert exposure_decision(old, 70, 10, {})[0] == 1
    instance = controller(name)
    gain = instance.gain_min if name != 'exposure_basic' else instance.gain_max
    with caplog.at_level('INFO', logger='indi_allsky'):
        instance.compare_highlights(corrected, 0.1, gain)
    assert instance._expUtils.EXPOSURE_NEXT == pytest.approx(0.09)
    assert instance._expUtils.GAIN_NEXT == gain
    assert 'Highlight patches (pre-dark): full 2.000%, any 2.000%; calibrated ADU' in caplog.text


def test_pre_dark_clipping_can_lower_gain_at_the_live_moon_exposure_limit():
    instance = controller('exposure_autogain_exp_prio_db_1_10')
    instance._expUtils.GAIN_MAX_NIGHT = 300
    instance.compare_highlights(HighlightMeasurement(1.2, 2.0, 70), 30, 254.935)
    assert instance._expUtils.EXPOSURE_NEXT == 30
    assert instance._expUtils.GAIN_NEXT < 254.935


@pytest.mark.parametrize('name,minimum', [
    ('exposure_autogain_exp_prio_db_1_10', 100),
    ('exposure_autogain_exp_prio_db', 6),
    ('exposure_autogain_exp_prio_iso', 200),
    ('exposure_autogain_exp_prio_iso_1_100', 2),
])
def test_reducing_across_nonzero_gain_floor_reduces_signal(name, minimum):
    instance = controller(name)
    instance._expUtils.GAIN_MIN_NIGHT = minimum
    instance._expUtils.GAIN_MAX_NIGHT = minimum * 4
    gain = instance.dB2gain(instance.gain2dB(minimum) + 0.2)
    before = 29 * 10 ** (instance.gain2dB(gain) / 20)
    instance.compare_highlights(HighlightMeasurement(2, 0, 70), 29, gain)
    after = instance._expUtils.EXPOSURE_NEXT * 10 ** (instance.gain2dB(instance._expUtils.GAIN_NEXT) / 20)
    assert after == pytest.approx(before * 0.9, rel=0.001)
    assert instance.exposure_min <= instance._expUtils.EXPOSURE_NEXT <= instance.exposure_max


@pytest.mark.parametrize('maximum', [0.05, 0.5, 1.0, 30.0])
def test_legacy_short_maximum_never_requests_negative_exposure(maximum):
    instance = controller('exposure_legacy_autogain')
    instance._expUtils.EXPOSURE_MAX = maximum
    instance.compare_highlights(HighlightMeasurement(0, 0, 10), maximum / 2, instance.gain_min)
    assert instance.exposure_min <= instance._expUtils.EXPOSURE_NEXT <= maximum


def test_legacy_at_exposure_ceiling_can_increase_gain():
    instance = controller('exposure_legacy_autogain')
    instance.compare_highlights(HighlightMeasurement(0, 0, 10), 30, instance.gain_min)
    assert instance._expUtils.GAIN_NEXT > instance.gain_min


def test_flat_bright_cloud_patch_settles_without_permanent_flapping():
    # A nearly uniform cloud/light source crosses the threshold as one region.
    # No exposure can put its area inside the requested percentage deadband.
    scene = np.full((100, 100, 3), 0.15)
    scene[40:60, 40:60] = 1.1
    mask = np.ones((100, 100), dtype=np.uint8)
    instance = controller('exposure_basic', night=False)
    exposure = 1.0
    history = []
    for _ in range(120):
        data = np.minimum(scene * exposure * 65535, 65535).astype(np.uint16)
        metrics = measure(data, mask, 16)
        instance._expUtils.EXPOSURE_NEXT = exposure
        instance.compare_highlights(metrics, exposure, 0)
        exposure = instance._expUtils.EXPOSURE_NEXT
        history.append(exposure)
    assert max(history[-30:]) / min(history[-30:]) < 1.01
    assert metrics.full <= 1


def test_small_mean_noise_at_lift_limit_does_not_chase_each_frame():
    instance = controller('exposure_basic', night=False)
    exposure = 1.0
    rng = np.random.default_rng(676)
    history = []
    for _ in range(150):
        adu = 20 * exposure * (1 + rng.normal(0, 0.005))
        instance._expUtils.EXPOSURE_NEXT = exposure
        instance.compare_highlights(HighlightMeasurement(5, 10, adu), exposure, 0)
        exposure = instance._expUtils.EXPOSURE_NEXT
        history.append(exposure)
    changes = np.count_nonzero(np.abs(np.diff(history[-100:])) > 1e-6)
    assert changes < 10


@pytest.mark.parametrize('stops', [0, 1, 2, 4])
def test_shadow_floor_has_a_small_deadband(stops):
    floor = 80 / 2 ** stops
    for adu in (floor * 0.99, floor, floor * 1.01):
        assert exposure_decision(HighlightMeasurement(5, 10, adu), 80, 10, {'MAX_BOOST': stops})[0] == 1
    assert exposure_decision(HighlightMeasurement(5, 10, floor * 0.97), 80, 10, {'MAX_BOOST': stops})[0] > 1
    assert exposure_decision(HighlightMeasurement(5, 10, floor * 1.03), 80, 10, {'MAX_BOOST': stops})[0] < 1


def test_predictive_headroom_uses_largest_region_and_either_channel_limit():
    mask = np.ones((100, 100), dtype=np.uint8)
    data = np.full((100, 100, 3), 8000, dtype=np.uint16)
    data[20:40, 20:40, 0] = 60000  # 4% blue-only plateau, just below clipping
    metrics = measure(data, mask, 16)
    assert metrics.full == metrics.any == metrics.full_next == 0
    assert metrics.any_next == 4
    assert exposure_decision(metrics, 80, 10, {})[0] == 1
    data[20:40, 20:40] = 8000
    data[::5, ::5] = 60000  # same total, but isolated points do not block recovery
    metrics = measure(data, mask, 16)
    assert metrics.full_next == metrics.any_next == 0.01
    assert exposure_decision(metrics, 80, 10, {})[0] == 1.1


@pytest.mark.parametrize('minimum,maximum', [(0.001, 0.05), (30, 30)])
def test_legacy_gain_steps_keep_exposure_valid_and_do_not_reverse_direction(minimum, maximum):
    instance = controller('exposure_legacy_autogain')
    instance._expUtils.EXPOSURE_MIN_NIGHT = minimum
    instance._expUtils.EXPOSURE_MAX = maximum
    instance._expUtils.GAIN_MAX_NIGHT = 1  # small gain steps must not raise exposure on a reduction
    instance.post_init()
    gain = instance.auto_gain_step_list[3]
    low = instance.auto_gain_exposure_cutoff_low
    instance.compare_highlights(HighlightMeasurement(5, 10, 70), low, gain)
    assert instance._expUtils.EXPOSURE_NEXT == pytest.approx(low, abs=1e-6)
    assert instance._expUtils.GAIN_NEXT < gain
    high = instance.auto_gain_exposure_cutoff_high
    instance.compare_highlights(HighlightMeasurement(0, 0, 10), high, gain)
    assert instance._expUtils.EXPOSURE_NEXT == pytest.approx(high, abs=1e-6)
    assert instance._expUtils.GAIN_NEXT > gain


@pytest.mark.parametrize('name', MODE_NAMES)
def test_flat_patch_with_fixed_gain_settles_in_every_mode(name):
    instance = controller(name)
    instance._expUtils.GAIN_MAX_NIGHT = instance.gain_min
    scene = np.full((80, 80, 3), 0.15)
    scene[30:50, 30:50] = 1.1
    exposure = 1.0
    history = []
    for _ in range(80):
        metrics = measure(np.minimum(scene * exposure * 65535, 65535).astype(np.uint16),
                          np.ones((80, 80), dtype=np.uint8), 16)
        instance._expUtils.EXPOSURE_NEXT = exposure
        instance.compare_highlights(metrics, exposure, instance.gain_min)
        exposure = instance._expUtils.EXPOSURE_NEXT
        history.append(exposure)
    assert max(history[-20:]) / min(history[-20:]) < 1.01
    assert metrics.full <= 1


@pytest.mark.parametrize('delay', [0, 1, 2])
@pytest.mark.parametrize('radius', [5, 8])
def test_sustained_dawn_catch_up_with_frames_already_in_flight(delay, radius):
    y, x = np.ogrid[:128, :128]
    scene = .3 + 3 * np.exp(-((x - 64) ** 2 + (y - 64) ** 2) / (2 * radius ** 2))
    mask = np.ones(scene.shape, dtype=np.uint8)

    def run():
        instance = controller('exposure_basic', night=False)
        instance._expUtils.EXPOSURE_NEXT = 1.0
        pending = [1.0] * (delay + 1)
        history, clipping = [], []
        for frame in range(150):
            exposure = pending.pop(0)
            illumination = 1.03 ** min(frame, 40)
            data = np.minimum(scene * illumination * exposure * 65535, 65535).astype(np.uint16)
            metrics = measure(data, mask, 16)
            instance.compare_highlights(metrics, exposure, 0)
            next_exposure = instance._expUtils.EXPOSURE_NEXT
            assert .8 * exposure - 1e-6 <= next_exposure <= 1.1 * exposure + 1e-6
            pending.append(next_exposure)
            history.append(exposure)
            clipping.append(max(0, metrics.full - 1))
        return np.array(history), np.array(clipping)

    improved, new_clipping = run()
    assert max(improved[-30:]) / min(improved[-30:]) < 1.01
    assert new_clipping[-1] == 0
