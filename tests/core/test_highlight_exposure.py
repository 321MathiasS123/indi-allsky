from copy import deepcopy
from multiprocessing import Array

import pytest
import numpy as np

from indi_allsky import constants
from indi_allsky import exposure as modes
from indi_allsky.highlight import HighlightMeasurement, exposure_scale, exposure_decision, measure


@pytest.mark.parametrize('full,any_channel,adu,direction', [
    (1.01, 2.0, 80, -1), (0.8, 2.41, 80, -1),
    (0.2, 3.0, 80, -1), (1.2, 1.4, 80, -1),  # either limit wins
    (0.5, 1.5, 40, 1), (0.8, 1.5, 40, 0), (0.5, 2.0, 40, 0),
    (0.6, 1.6, 40, 0), (1.0, 2.4, 40, 0),  # inclusive deadband
    (0, 0, 80, 0), (0, 0, 70, 0), (0, 0, 69, 1),
    (0, 0, 100, -1), (20, 30, 20, 0), (20, 30, 10, 1),
])
def test_dual_deadband_and_shadow_floor(full, any_channel, adu, direction):
    scale = exposure_scale(HighlightMeasurement(full, any_channel, adu), 80, 10, {})
    assert (scale > 1) - (scale < 1) == direction
    assert 0.8 <= scale <= 1.1


def test_no_reduction_past_lift_limit_and_custom_settings():
    assert exposure_scale(HighlightMeasurement(10, 20, 21), 80, 10, {}) == pytest.approx(20 / 21)
    assert exposure_scale(HighlightMeasurement(0.8, 2, 80), 80, 10,
                          {'FULL_TARGET': 0.4, 'FULL_DEV': 0.1}) == pytest.approx(0.8)
    assert exposure_scale(HighlightMeasurement(10, 20, 80), 80, 10, {'MAX_BOOST': 0}) == 1


@pytest.mark.parametrize('adu,expected', [(69, 70 / 69), (69.99, 70 / 69.99),
                                         (70, 1), (80, 1), (90, 1),
                                         (90.01, 90 / 90.01), (91, 90 / 91)])
def test_brightness_recovery_also_tapers_at_the_adu_band(adu, expected):
    assert exposure_scale(HighlightMeasurement(0, 0, adu), 80, 10, {}) == pytest.approx(expected)


MODE_NAMES = [name for name in dir(modes) if name.startswith('exposure_')]


def controller(name, night=True, moon=False):
    config = {'TARGET_ADU': 70, 'TARGET_ADU_DAY': 80, 'TARGET_ADU_DEV': 10,
              'TARGET_ADU_DEV_DAY': 10, 'CCD_CONFIG': {'AUTO_GAIN_LEVELS': 8},
              'HIGHLIGHT_PROTECTION': {'ENABLE': True}}
    instance = getattr(modes, name)(config, Array('i', 7), Array('i', 10), Array('i', 6),
                                  Array('i', [int(night), int(moon)]))
    utils = instance._expUtils
    utils.EXPOSURE_MIN_DAY = 0.000032
    utils.EXPOSURE_MIN_NIGHT = 0.001
    utils.EXPOSURE_MAX = 30
    minimum = 100 if name.endswith('iso') else 1 if name.endswith('iso_1_100') else 0
    maximum = 800 if name.endswith('iso') else 8 if name.endswith('iso_1_100') else 100
    for key, value in [('GAIN_MIN_NIGHT', minimum), ('GAIN_MAX_NIGHT', maximum),
                       ('GAIN_MIN_DAY', minimum), ('GAIN_MAX_DAY', minimum),
                       ('GAIN_MIN_MOONMODE', minimum), ('GAIN_MAX_MOONMODE', maximum),
                       ('BINNING_DAY', 1), ('BINNING_NIGHT', 2), ('BINNING_MOONMODE', 3)]:
        setattr(utils, key, value)
    return instance


@pytest.mark.parametrize('name', MODE_NAMES)
@pytest.mark.parametrize('night,moon,binning', [(False, False, 1), (True, False, 2), (True, True, 3)])
def test_highlight_requests_retain_every_mode_policy_and_binning(name, night, moon, binning):
    actual = controller(name, night, moon)
    reference = controller(name, night, moon)
    config_before = deepcopy(actual.config)
    gain = actual.gain_min if name != 'exposure_basic' else actual.gain_max
    expected = reference.adjust_exposure_gain(0.1, gain, 0.09)
    actual.compare_highlights(HighlightMeasurement(5, 10, 80), 0.1, gain)
    assert actual._expUtils.EXPOSURE_NEXT == pytest.approx(expected[0], abs=0.000001)
    assert actual._expUtils.GAIN_NEXT == pytest.approx(expected[1], abs=0.001)
    assert actual._expUtils.BINNING_NEXT == binning
    assert actual.config == config_before
    assert not actual.target_adu_found


@pytest.mark.parametrize('name', MODE_NAMES)
def test_physical_exposure_and_gain_limits(name):
    instance = controller(name, night=False)
    for _ in range(2):
        instance.compare_highlights(HighlightMeasurement(10, 20, 80), 0.000032, instance.gain_min)
        assert instance._expUtils.EXPOSURE_NEXT >= instance.exposure_min
        assert instance.gain_min <= instance._expUtils.GAIN_NEXT <= instance.gain_max
    instance.compare_highlights(HighlightMeasurement(0, 0, 1), 30, instance.gain_max)
    assert instance._expUtils.EXPOSURE_NEXT <= 30
    assert instance.gain_min <= instance._expUtils.GAIN_NEXT <= instance.gain_max


@pytest.mark.parametrize('name', MODE_NAMES)
def test_standard_adu_behavior_is_unchanged_when_feature_is_not_used(name):
    instance = controller(name)
    instance.config['HIGHLIGHT_PROTECTION']['ENABLE'] = False
    gain = instance.gain_min if name != 'exposure_basic' else instance.gain_max
    instance.compare_exposure(35, 0.1, gain)
    assert instance._expUtils.EXPOSURE_NEXT == pytest.approx(0.2)
    instance.compare_exposure(70, 0.2, gain)
    assert instance.target_adu_found
    for _ in range(6):
        instance.compare_exposure(90, 0.2, gain)
    assert not instance.target_adu_found


@pytest.mark.parametrize('gain', [0, 100])
def test_fixed_gain_including_camera_without_gain_control(gain):
    instance = controller('exposure_basic')
    instance._expUtils.GAIN_MIN_NIGHT = gain
    instance._expUtils.GAIN_MAX_NIGHT = gain
    instance.compare_highlights(HighlightMeasurement(5, 10, 70), 1, gain)
    assert instance._expUtils.GAIN_NEXT == gain
    assert instance._expUtils.EXPOSURE_NEXT == pytest.approx(0.9)


@pytest.mark.parametrize('name', [n for n in MODE_NAMES if 'exp_prio' in n])
def test_fixed_exposure_uses_available_auto_gain(name):
    instance = controller(name)
    instance._expUtils.EXPOSURE_MIN_NIGHT = 30
    instance.compare_highlights(HighlightMeasurement(0, 0, 1), 30, instance.gain_min)
    assert instance._expUtils.EXPOSURE_NEXT == 30
    assert instance._expUtils.GAIN_NEXT > instance.gain_min


def test_closed_loop_settles_then_recovers_when_bright_source_disappears():
    instance = controller('exposure_basic', night=False)
    y, x = np.ogrid[:256, :256]
    scene = 0.3 + 3 * np.exp(-((x - 128) ** 2 + (y - 128) ** 2) / (2 * 11 ** 2))
    mask = np.ones(scene.shape, dtype=np.uint8)
    exposure = 1.0
    history = []
    for frame in range(90):  # allow the proportional fine adjustment to settle
        data = np.minimum(scene * exposure * 65535, 65535).astype(np.uint16)
        metrics = measure(data, mask, 16)
        instance._expUtils.EXPOSURE_NEXT = exposure
        instance.compare_highlights(metrics, exposure, 0)
        exposure = instance._expUtils.EXPOSURE_NEXT
        history.append(exposure)
    assert len(set(history[-10:])) == 1
    assert metrics.full <= 1.0
    assert metrics.adu >= 20
    protected_exposure = exposure
    for frame in range(60):
        data = np.full(scene.shape, min(65535, 0.3 * exposure * 65535), dtype=np.uint16)
        metrics = measure(data, mask, 16)
        instance._expUtils.EXPOSURE_NEXT = exposure
        instance.compare_highlights(metrics, exposure, 0)
        exposure = instance._expUtils.EXPOSURE_NEXT
    assert exposure > protected_exposure
    assert 70 <= metrics.adu <= 90
    assert metrics.full == metrics.any == 0


def test_pending_frames_do_not_compound_unapplied_requests():
    instance = controller('exposure_basic', night=False)
    for pending in range(6):
        instance.compare_highlights(HighlightMeasurement(3, 0, 80), 1.0, 0)
        assert instance._expUtils.EXPOSURE_NEXT == pytest.approx(0.9)
    instance.compare_highlights(HighlightMeasurement(1.1, 0, 80), 0.9, 0)
    assert instance._expUtils.EXPOSURE_NEXT == pytest.approx(0.855, abs=1e-6)


@pytest.mark.parametrize('reset', ['ordinary', 'invalid_frame', 'mode', 'shadow_floor', 'inside_band'])
def test_proportional_strength_resets_between_control_periods(reset):
    instance = controller('exposure_basic', night=False)
    exposure = 1.0
    for factor in (.9, .85, .825):
        instance.compare_highlights(HighlightMeasurement(3, 0, 80), exposure, 0)
        assert instance._expUtils.EXPOSURE_NEXT == pytest.approx(exposure * factor, abs=1e-6)
        exposure = instance._expUtils.EXPOSURE_NEXT
    if reset == 'ordinary':
        instance.compare_exposure(80, exposure, 0)
    elif reset == 'invalid_frame':
        instance.reset_highlights()
    elif reset == 'mode':
        instance.night_av[constants.NIGHT_NIGHT] = True
    elif reset == 'shadow_floor':
        instance.compare_highlights(HighlightMeasurement(3, 0, 20), exposure, 0)
    else:
        instance.compare_highlights(HighlightMeasurement(.8, 2, 80), exposure, 0)
    instance.compare_highlights(HighlightMeasurement(3, 0, 80), exposure, 0)
    assert instance._expUtils.EXPOSURE_NEXT == pytest.approx(exposure * .9, abs=1e-6)


@pytest.mark.parametrize('missing', [False, True])
def test_disabled_legacy_boundaries_match_main(missing):
    instance = controller('exposure_legacy_autogain')
    if missing:
        instance.config.pop('HIGHLIGHT_PROTECTION')
    else:
        instance.config['HIGHLIGHT_PROTECTION']['ENABLE'] = False
    instance.post_init()
    assert instance.auto_gain_exposure_cutoff_low == 24
    assert instance.auto_gain_exposure_cutoff_mid == 26.75
    assert instance.auto_gain_exposure_cutoff_high == 29.5
    # Preserve main's behavior at the exposure ceiling and gain-step boundary.
    assert instance.adjust_exposure_gain(30, 0, 33) == (30, 0, 0, 0)
    assert instance.adjust_exposure_gain(24, 42.857, 21.6) == pytest.approx((26.75, 28.571, 0, -14.286))
    instance._expUtils.EXPOSURE_MAX = .05
    instance.post_init()
    assert instance.auto_gain_exposure_cutoff_high == -.45


@pytest.mark.parametrize('missing', [False, True])
@pytest.mark.parametrize('name,minimum,expected', [
    ('exposure_autogain_exp_prio_db_1_10', 100, 9.0),
    ('exposure_autogain_exp_prio_db', 6, 3.582964534981475),
    ('exposure_autogain_exp_prio_iso', 200, 3.6),
    ('exposure_autogain_exp_prio_iso_1_100', 2, 3.6),
])
def test_disabled_nonzero_gain_floor_matches_main(name, minimum, expected, missing):
    instance = controller(name)
    if missing:
        instance.config.pop('HIGHLIGHT_PROTECTION')
    else:
        instance.config['HIGHLIGHT_PROTECTION']['ENABLE'] = False
    instance._expUtils.GAIN_MIN_NIGHT = minimum
    instance._expUtils.GAIN_MAX_NIGHT = minimum * 4
    assert instance.reduce_gain(1, minimum, .9)[0] == pytest.approx(expected)


@pytest.mark.parametrize('ratio,expected', [(1, 1), (1.01, .995), (1.1, .95), (1.2, .9), (1.3, .85), (1.4, .8), (100, .8)])
@pytest.mark.parametrize('channel', ['full', 'any'])
def test_proportional_reduction_tapers_at_either_upper_limit(channel, ratio, expected):
    metrics = HighlightMeasurement(ratio if channel == 'full' else 0,
                                   ratio * 2.4 if channel == 'any' else 0, 80)
    assert exposure_scale(metrics, 80, 10, {}) == pytest.approx(expected)


@pytest.mark.parametrize('name', MODE_NAMES)
@pytest.mark.parametrize('full,factor', [(1.02, .995), (1.1, .975), (1.2, .95), (1.4, .9)])
def test_proportional_request_uses_every_modes_existing_actuator_policy(name, full, factor):
    actual = controller(name)
    reference = controller(name)
    gain = actual.gain_min if name != 'exposure_basic' else actual.gain_max
    exposure = 1.0
    expected = reference.adjust_exposure_gain(exposure, gain, exposure * factor)
    actual.compare_highlights(HighlightMeasurement(full, 0, 80), exposure, gain)
    assert actual._expUtils.EXPOSURE_NEXT == pytest.approx(expected[0], abs=1e-6)
    assert actual._expUtils.GAIN_NEXT == pytest.approx(expected[1], abs=.001)


@pytest.mark.parametrize('adu,expected,reason', [
    (10, 1.1, 'recover shadow floor'), (20, 1, 'shadow floor'),
    (21, 20 / 21, 'full+any clipping'), (80, .8, 'full+any clipping'),
])
def test_faster_clipping_reduction_retains_the_shadow_floor(adu, expected, reason):
    scale, actual_reason = exposure_decision(HighlightMeasurement(3, 8, adu), 80, 10, {})
    assert scale == pytest.approx(expected)
    assert actual_reason == reason


def test_early_diagnostic_reports_decision_without_inventing_render_lift(caplog):
    instance = controller('exposure_basic', night=False)
    with caplog.at_level('INFO', logger='indi_allsky'):
        instance.compare_highlights(HighlightMeasurement(0, 3, 80), .1, 0)
    assert 'reason: any clipping' in caplog.text
    assert 'shadow lift' not in caplog.text  # reported later from the rendered frame
    assert 'exposure request 0.938x' in caplog.text
