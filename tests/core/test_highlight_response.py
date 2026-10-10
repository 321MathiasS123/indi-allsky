"""Adaptive raw-meter response with a capture already in flight."""
import math

import numpy as np
import pytest

from indi_allsky.highlight import HighlightMeasurement, measure
from test_highlight_exposure import MODE_NAMES, controller


def measurement(adu, **changes):
    values = dict(full=0, any=0, adu=adu, full_next=0, any_next=0,
                  full_fast=0, any_fast=0)
    values.update(changes)
    return HighlightMeasurement(**values)


def settings(instance):
    return instance._expUtils.EXPOSURE_NEXT, instance._expUtils.GAIN_NEXT


def set_inflight(instance, exposure, gain):
    instance._expUtils.EXPOSURE_CURRENT = exposure
    instance._expUtils.GAIN_CURRENT = gain
    instance._expUtils.EXPOSURE_NEXT = exposure
    instance._expUtils.GAIN_NEXT = gain


def signal(instance, capture):
    exposure, gain = capture
    return exposure * (10 ** (instance.gain2dB(gain) / 20)
                       if hasattr(instance, 'gain2dB') else 1)


def fixed_gain_controller():
    instance = controller('exposure_basic')
    instance._expUtils.GAIN_MIN_NIGHT = 0
    instance._expUtils.GAIN_MAX_NIGHT = 0
    instance._expUtils.EXPOSURE_MAX = 120
    return instance


@pytest.mark.parametrize('actuator', ['exposure', 'gain'])
@pytest.mark.parametrize('direction', [-1, 1])
def test_continuous_scene_change_updates_every_capture_despite_one_frame_delay(actuator, direction):
    if actuator == 'gain':
        instance = controller('exposure_autogain_exp_prio_db_1_10')
        instance._expUtils.GAIN_MAX_NIGHT = 400
        instance.gain_quantum = 1
        initial = (30, 150)
    else:
        instance = fixed_gain_controller()
        initial = (25, 0)
    source = inflight = initial
    initial_signal = signal(instance, initial)
    brightness = 60 if direction > 0 else 80
    commands = [initial]
    for frame in range(18):
        # The next capture starts before this completed capture is metered.
        # Scene brightness belongs to the source, not the pending command.
        set_inflight(instance, *inflight)
        adu = brightness * math.exp(-direction * .045 * frame) * signal(instance, source) / initial_signal
        instance.compare_highlights(measurement(adu), *source)
        command = settings(instance)
        previous = signal(instance, inflight)
        current = signal(instance, command)
        # Integer gain may round by a hardware step.
        assert .79 <= current / previous <= 1.27
        commands.append(command)
        source, inflight = inflight, command
    values = np.array([signal(instance, command) for command in commands])
    # Ignore startup/deadband acquisition. The reported alternating hold must
    # not recur once a sustained brightening/darkening trend is established.
    assert np.all(direction * np.diff(values[5:]) > 0)
    if actuator == 'gain':
        assert all(command[0] == 30 for command in commands)
    else:
        assert all(command[1] == 0 for command in commands)


@pytest.mark.parametrize('near,far,direction', [(59, 20, 1), (81, 200, -1)])
def test_large_brightness_errors_receive_larger_but_bounded_commands(near, far, direction):
    results = []
    for adu in (near, far):
        instance = fixed_gain_controller()
        set_inflight(instance, 20, 0)
        instance.compare_highlights(measurement(adu), 20, 0)
        ratio = settings(instance)[0] / 20
        assert .8 - 1e-7 <= ratio <= 1.25 + 1e-7
        results.append(math.log(ratio))
    assert direction * results[1] > direction * results[0] > 0
    assert abs(results[1]) > math.log(1.1)


@pytest.mark.parametrize('pending,adu', [(.9, 30), (1.7, 50), (.6, 100)])
def test_stale_brightness_measurement_never_undoes_a_stronger_pending_request(pending, adu):
    instance = fixed_gain_controller()
    set_inflight(instance, pending, 0)
    before = list(instance.exposure_av), list(instance.gain_av), list(instance.binning_av)
    instance.compare_highlights(measurement(adu), 1, 0)
    assert (list(instance.exposure_av), list(instance.gain_av), list(instance.binning_av)) == before


@pytest.mark.parametrize('adu,direction,bound', [(30, 1, 2), (160, -1, .5)])
def test_repeated_delayed_measurements_approach_absolute_target_without_compounding(adu, direction, bound):
    instance = fixed_gain_controller()
    set_inflight(instance, 1, 0)
    prior = 1.0
    for _ in range(30):
        # Intentionally keep the original source while advancing requests.
        # A controller multiplying pending by the old error would run away.
        instance.compare_highlights(measurement(adu), 1, 0)
        current = settings(instance)[0]
        assert direction * (current - prior) >= -1e-6
        assert min(1, bound) - 1e-6 <= current <= max(1, bound) + 1e-6
        assert .8 - 1e-6 <= current / prior <= 1.25 + 1e-6
        prior = current
    assert abs(math.log(prior / bound)) < .001


@pytest.mark.parametrize('field,limit', [('full_fast', 1), ('any_fast', 2.4)])
def test_fast_recovery_requires_both_raw_headroom_limits(field, limit):
    ratios = []
    for value in (limit, limit + .001):
        instance = fixed_gain_controller()
        set_inflight(instance, 1, 0)
        instance.compare_highlights(measurement(20, **{field: value}), 1, 0)
        ratios.append(settings(instance)[0])
    assert 1.1 < ratios[0] <= 1.25
    assert 1 < ratios[1] <= 1.1 + 1e-6


def test_missing_fast_headroom_retains_conservative_growth_ceiling():
    instance = fixed_gain_controller()
    set_inflight(instance, 1, 0)
    instance.compare_highlights(HighlightMeasurement(0, 0, 20), 1, 0)
    assert 1 < settings(instance)[0] <= 1.1 + 1e-6


def test_unsafe_fast_headroom_caps_total_growth_from_old_source_not_pending():
    instance = fixed_gain_controller()
    set_inflight(instance, 1.05, 0)
    for _ in range(5):
        instance.compare_highlights(measurement(20, full_fast=10), 1, 0)
        assert 1.05 <= settings(instance)[0] <= 1.1 + 1e-6


@pytest.mark.parametrize('field,value', [('full_next', 1.01), ('any_next', 2.41)])
def test_predicted_clipping_blocks_even_fast_recovery(field, value):
    instance = fixed_gain_controller()
    set_inflight(instance, 1, 0)
    instance.compare_highlights(measurement(30, **{field: value}), 1, 0)
    assert settings(instance) == (1, 0)


def test_urgent_clipping_can_cancel_an_inflight_brightness_increase():
    instance = fixed_gain_controller()
    set_inflight(instance, 1.2, 0)
    instance.compare_highlights(measurement(70, full=5, any=10), 1, 0)
    assert .8 <= settings(instance)[0] < 1


@pytest.mark.parametrize('actuator', ['exposure', 'gain'])
@pytest.mark.parametrize('pending_ratio', [1.25, 1.5, 2.0])
def test_overbright_source_cancels_pending_growth_before_ordinary_adu_reduction(actuator, pending_ratio):
    if actuator == 'gain':
        instance = controller('exposure_autogain_exp_prio_db_1_10')
        instance._expUtils.GAIN_MAX_NIGHT = 400
        instance.gain_quantum = 1
        source = (30, 150)
        pending = (30, instance.effective_gain(150 + 200 * math.log10(pending_ratio)))
    else:
        instance = fixed_gain_controller()
        source, pending = (25, 0), (25 * pending_ratio, 0)
    set_inflight(instance, *pending)
    instance.compare_highlights(measurement(160), *source)
    # A bounded reduction from the pending increase alone could still leave
    # the request brighter than the already overbright source capture.
    assert signal(instance, settings(instance)) / signal(instance, source) <= .91


def test_output_protection_remains_a_constraint_on_adaptive_recovery():
    instance = fixed_gain_controller()
    instance.config['HIGHLIGHT_PROTECTION']['OUTPUT_ENABLE'] = True
    instance.highlight_output.observe(
        HighlightMeasurement(10, 20, 0), 1, 0, tuple(instance.night_av),
        instance.config['HIGHLIGHT_PROTECTION'])
    set_inflight(instance, 1, 0)
    instance.compare_highlights(measurement(30), 1, 0)
    assert settings(instance)[0] <= 1


def test_fast_headroom_counts_all_predicted_pixels_conservatively_and_honors_mask():
    data = np.zeros((100, 100, 3), dtype=np.uint16)
    data[10:16, 10:20] = 40000
    data[30:36, 30:40] = 40000
    data[50:53, 50:60, 0] = 40000
    mask = np.ones((100, 100), dtype=np.uint8)
    result = measure(data, mask, 16)
    assert result.full == result.any == result.full_next == result.any_next == 0
    assert result.full_fast == pytest.approx(1.2)
    assert result.any_fast == pytest.approx(1.5)
    mask[30:36, 30:40] = 0
    result = measure(data, mask, 16)
    assert result.full_fast == pytest.approx(60 / 9940 * 100)
    assert result.any_fast == pytest.approx(90 / 9940 * 100)


@pytest.mark.parametrize('name', MODE_NAMES)
def test_disabled_feature_keeps_the_existing_adu_controller(name):
    absent, disabled = controller(name), controller(name)
    absent.config.pop('HIGHLIGHT_PROTECTION')
    disabled.config['HIGHLIGHT_PROTECTION']['ENABLE'] = False
    for instance in (absent, disabled):
        if hasattr(instance, 'post_init'):
            instance.post_init()
    gain = absent.gain_max if name == 'exposure_basic' else absent.gain_min
    for adu, exposure in [(35, 10), (70, 20), (110, 20), (30, 15)]:
        for instance in (absent, disabled):
            instance.compare_exposure(adu, exposure, gain)
        assert list(absent.exposure_av) == list(disabled.exposure_av)
        assert list(absent.gain_av) == list(disabled.gain_av)
        assert list(absent.binning_av) == list(disabled.binning_av)
