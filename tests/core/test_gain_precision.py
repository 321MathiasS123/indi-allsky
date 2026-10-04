"""Hardware gain resolution must agree across requests, captures and limits."""
import pytest

from indi_allsky.gain import gain_limits, gain_quantum, quantize_gain
from indi_allsky.highlight import HighlightMeasurement
from test_highlight_exposure import MODE_NAMES, controller


@pytest.mark.parametrize('driver', ['libcamera-still', 'rpicam-still', 'indi_libcamera_ccd',
                                   'indi_pylibcamera', 'indi_qhy_ccd', 'future_driver'])
def test_fractional_and_unknown_interfaces_are_not_forced_to_integer_gain(driver):
    assert quantize_gain(1.875, gain_quantum(driver)) == 1.875


@pytest.mark.parametrize('minimum,maximum,quantum,values,expected', [
    (0, 300.9, 1, [], (0, 300)), (10.2, 300.9, 1, [], (11, 300)),
    (85.912, 85.912, 1, [], (86, 86)), (0, 0, 1, [], (0, 0)),
    (1.875, 2.9, 0, [], (1.875, 2.9)),
    (150, 750, 0, [100, 200, 400, 800], (200, 400)),
    (350, 350, 0, [100, 200, 400, 800], (400, 400)),
])
def test_effective_limits_are_supported_and_stay_inside_ranges_when_possible(minimum, maximum, quantum, values, expected):
    assert gain_limits(minimum, maximum, quantum, values) == expected


@pytest.mark.parametrize('name', MODE_NAMES)
@pytest.mark.parametrize('highlight', [False, True])
def test_every_mode_publishes_integer_commands_and_matching_deltas(name, highlight):
    instance = controller(name)
    instance.gain_quantum = 1
    instance.config['HIGHLIGHT_PROTECTION']['ENABLE'] = highlight
    current_gain = instance.gain_max if name == 'exposure_basic' else instance.gain_min
    instance._set_exposure(30, current_gain, 33, highlight=highlight)
    gain = instance._expUtils.GAIN_NEXT
    assert gain == round(gain)
    assert instance.gain_min <= gain <= instance.gain_max
    assert instance._expUtils.GAIN_DELTA == pytest.approx(gain - current_gain, abs=.001)


@pytest.mark.parametrize('clipping,expected', [(True, 85), (False, 87)])
def test_sub_step_highlight_correction_moves_one_real_gain_step(clipping, expected):
    instance = controller('exposure_autogain_exp_prio_db_1_10')
    instance.gain_quantum = 1
    instance._expUtils.EXPOSURE_NEXT, instance._expUtils.GAIN_NEXT = 30, 86
    metrics = HighlightMeasurement(1.001, 0, 70) if clipping else HighlightMeasurement(0, 0, 59.999)
    instance.compare_highlights(metrics, 30, 86)
    assert instance._expUtils.GAIN_NEXT == expected
    assert instance._expUtils.EXPOSURE_NEXT == 30
    # Matching the applied integer must clear the pending guard.
    assert not instance._highlight_request_pending(30, expected)
    assert instance._highlight_request_pending(30, 86)


def test_fractional_request_is_normalized_for_pending_check_after_restart():
    instance = controller('exposure_autogain_exp_prio_db_1_10')
    instance.gain_quantum = 1
    instance._expUtils.EXPOSURE_NEXT, instance._expUtils.GAIN_NEXT = 30, 85.912
    assert not instance._highlight_request_pending(30, 86)
    instance.gain_quantum = 0
    assert instance._highlight_request_pending(30, 86)


def test_fractional_configured_ceiling_does_not_keep_protection_waiting():
    instance = controller('exposure_autogain_exp_prio_db_1_10')
    instance.gain_quantum = 1
    instance._expUtils.GAIN_MAX_NIGHT = 100.4
    instance._expUtils.EXPOSURE_NEXT, instance._expUtils.GAIN_NEXT = 30, 100
    instance.highlight_transition.active = True
    instance.compare_highlights(HighlightMeasurement(0, 0, 30, 10, 20), 30, 100)
    assert instance._expUtils.GAIN_NEXT == 100
    assert not instance.highlight_transition.active
    assert instance.highlight_transition.reason == 'achievable exposure/gain ceiling'


@pytest.mark.parametrize('quantum,values', [(1, []), (0, [100, 200, 400, 800])])
def test_legacy_gain_ladder_recognizes_applied_steps_without_reset_or_duplicates(quantum, values, caplog):
    instance = controller('exposure_legacy_autogain')
    instance.gain_quantum, instance.gain_values = quantum, values
    instance._expUtils.GAIN_MIN_NIGHT = 150 if values else 0
    instance._expUtils.GAIN_MAX_NIGHT = 750 if values else 3
    instance.post_init()
    ladder = instance.auto_gain_step_list
    assert ladder == ([200, 400] if values else [0, 1, 2, 3])
    for low, high in zip(ladder, ladder[1:]):
        instance._set_exposure(30, low, 33, highlight=True)
        assert instance._expUtils.GAIN_NEXT == high
        instance._set_exposure(instance.exposure_min, high, instance.exposure_min * .9, highlight=True)
        assert instance._expUtils.GAIN_NEXT == low
    assert 'Current gain not found' not in caplog.text


@pytest.mark.parametrize('name', MODE_NAMES)
def test_fixed_fractional_setting_is_resolved_consistently_in_every_mode(name):
    instance = controller(name)
    instance.gain_quantum = 1
    instance._expUtils.GAIN_MIN_NIGHT = instance._expUtils.GAIN_MAX_NIGHT = 85.912
    assert instance.gain_min == instance.gain_max == 86
    instance._set_exposure(30, 86, 33, highlight=True)
    assert instance._expUtils.GAIN_NEXT == 86
    assert not instance._highlight_request_pending(30, 86)


@pytest.mark.parametrize('delay', [0, 1, 3])
def test_integer_gain_recovery_with_delayed_captures_releases_at_real_ceiling(delay):
    instance = controller('exposure_autogain_exp_prio_db_1_10')
    instance.gain_quantum = 1
    instance._expUtils.GAIN_MAX_NIGHT = 100.4
    instance._expUtils.EXPOSURE_NEXT, instance._expUtils.GAIN_NEXT = 30, 77
    state = instance.highlight_transition
    state.active, state.reference, state.gamma_mix = True, 70, 1
    pending = [77] * (delay + 1)
    applied = []
    for frame in range(160):
        gain = pending.pop(0)
        # The simulated device really applies integer commands, rather than
        # treating a fractional request as if it changed the physical signal.
        assert gain == round(gain)
        adu = 50 * .99 ** min(frame, 30) * 10 ** ((gain - 77) / 200)
        instance.compare_highlights(HighlightMeasurement(0, 0, adu), 30, gain)
        pending.append(instance._expUtils.GAIN_NEXT)
        applied.append(gain)
        state.render_target(adu, 70, 2)
        state.gamma(.87, .87)
    assert all(a <= b for a, b in zip(applied, applied[1:]))
    assert applied[-1] == 100 and adu < 60
    assert state.phase == 'normal' and state.lift == 0


@pytest.mark.parametrize('quantum', [0, 1])
@pytest.mark.parametrize('reverse', [False, True])
def test_optional_twilight_fixed_gain_blends_use_the_same_camera_precision(quantum, reverse):
    twilight = pytest.importorskip('indi_allsky.twilight')
    instance = controller('exposure_basic')
    instance.gain_quantum = quantum
    instance._expUtils.GAIN_MIN_NIGHT = instance._expUtils.GAIN_MAX_NIGHT = 100
    instance._expUtils.EXPOSURE_NEXT = 30
    instance.config['TWILIGHT_TRANSITION'] = {'ENABLE': True}
    transition = twilight.TwilightTransition(instance.config)
    instance.config = transition.config
    altitudes = [-6, -6.3, -7.1, -8.3, -9.7, -11.1, -12]
    for altitude in reversed(altitudes) if reverse else altitudes:
        transition.apply(altitude)
        instance.night_av[0] = altitude < -6
        desired = quantize_gain(100 * transition.weight, quantum)
        instance.apply_transition_limits()
        assert instance.gain_min == instance.gain_max == pytest.approx(desired)
        assert instance._expUtils.GAIN_NEXT == pytest.approx(desired, abs=.001)
        if quantum:
            assert instance._expUtils.GAIN_NEXT == round(instance._expUtils.GAIN_NEXT)
        assert not instance._highlight_request_pending(30, instance._expUtils.GAIN_NEXT)
