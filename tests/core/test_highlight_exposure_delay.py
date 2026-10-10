"""Command smoothing uses capture progress, never renderer completion."""
import math

import numpy as np
import pytest

from indi_allsky.highlight import HighlightMeasurement
from test_highlight_exposure import controller


def set_command(instance, exposure, gain, inflight=True):
    instance._expUtils.EXPOSURE_NEXT = exposure
    instance._expUtils.GAIN_NEXT = gain
    if inflight:
        instance._expUtils.EXPOSURE_CURRENT = exposure
        instance._expUtils.GAIN_CURRENT = gain


def requested(instance):
    return instance._expUtils.EXPOSURE_NEXT, instance._expUtils.GAIN_NEXT


def signal(instance, exposure, gain):
    return exposure * (10 ** (instance.gain2dB(gain) / 20) if hasattr(instance, 'gain2dB') else 1)


@pytest.mark.parametrize('inflight', [False, True])
def test_adu_response_accounts_for_pending_command_even_before_camera_dispatch(inflight):
    instance = controller('exposure_basic', night=False)
    instance.compare_highlights(HighlightMeasurement(0, 0, 110), 1.1, 0)
    instance._expUtils.EXPOSURE_CURRENT = 1
    set_command(instance, .99, 0, inflight)
    instance.compare_highlights(HighlightMeasurement(0, 0, 110), 1, 0)
    # The same absolute ADU target applies whether the latest command has
    # reached the camera or is still awaiting dispatch.
    expected = math.sqrt(.99 * (90 / 110))
    assert requested(instance) == pytest.approx((expected, 0), abs=1e-6)


def test_caught_up_measurement_uses_half_the_remaining_logarithmic_error():
    instance = controller('exposure_basic', night=False)
    set_command(instance, .99, 0)
    instance.compare_highlights(HighlightMeasurement(0, 0, 110), .99, 0)
    assert requested(instance) == pytest.approx((.99 * math.sqrt(90 / 110), 0), abs=1e-6)


def test_brightening_after_an_inflight_increase_keeps_the_urgent_absolute_cut():
    instance = controller('exposure_basic', night=False)
    set_command(instance, 1.1, 0)
    instance.compare_highlights(HighlightMeasurement(5, 10, 80), 1, 0)
    # A clipped-patch demand remains urgent; linear ADU response must not
    # postpone it merely because a brighter capture is already in flight.
    assert requested(instance) == pytest.approx((.9, 0), abs=1e-6)


def test_first_adu_observation_moves_from_pending_toward_absolute_target():
    instance = controller('exposure_basic', night=False)
    set_command(instance, .99, 0)
    instance.compare_highlights(HighlightMeasurement(0, 0, 110), 1, 0)
    assert requested(instance) == pytest.approx((math.sqrt(.99 * 90 / 110), 0), abs=1e-6)


@pytest.mark.parametrize('initial_scale', [.999, .97, .91])
def test_sharper_fresh_demand_on_an_existing_downward_track_is_immediate(initial_scale):
    instance = controller('exposure_basic', night=False)
    set_command(instance, 1, 0)
    instance.compare_highlights(HighlightMeasurement(0, 0, 90 / initial_scale), 1, 0)
    set_command(instance, *requested(instance))
    instance.compare_highlights(HighlightMeasurement(5, 10, 80), 1, 0)
    assert requested(instance) == pytest.approx((.9, 0), abs=1e-6)


@pytest.mark.parametrize('boundary', ['reset', 'mode', 'recovery', 'hold'])
def test_adu_response_recomputes_target_after_reset_mode_or_recovery_boundaries(boundary):
    instance = controller('exposure_basic', night=False)
    instance.compare_highlights(HighlightMeasurement(0, 0, 110), 1.1, 0)
    if boundary == 'reset':
        instance.reset_highlights()
    elif boundary == 'mode':
        instance._expUtils.GAIN_MAX_NIGHT = 0
        instance.night_av[0] = 1
    else:
        set_command(instance, 1, 0)
        adu = 40 if boundary == 'recovery' else 80
        instance.compare_highlights(HighlightMeasurement(0, 0, adu), 1, 0)
    set_command(instance, .99, 0)
    instance.compare_highlights(HighlightMeasurement(0, 0, 110), 1, 0)
    upper = 80 if boundary == 'mode' else 90
    assert requested(instance) == pytest.approx((math.sqrt(.99 * upper / 110), 0), abs=1e-6)


def test_sustained_dawn_gain_steps_no_longer_alternate_six_and_three():
    instance = controller('exposure_autogain_exp_prio_db_1_10')
    instance._expUtils.GAIN_MAX_NIGHT = 300
    instance.gain_quantum = 1
    gains = [216, 210]
    for _ in range(12):
        set_command(instance, 30, gains[-1])
        instance.compare_highlights(HighlightMeasurement(0, 0, 100), 30, gains[-2])
        # Gain units are tenths of a dB. Calculate the absolute ADU target
        # independently, then allow only the unavoidable integer quantization.
        expected = (gains[-1] + gains[-2] + 200 * math.log10(80 / 100)) / 2
        assert requested(instance)[1] == pytest.approx(expected, abs=.5)
        gains.append(requested(instance)[1])
    steps = np.diff(gains[3:])
    assert np.all(steps < 0)
    assert np.ptp(steps) <= 1
    assert requested(instance)[0] == 30


def test_long_exposure_delay_converges_to_equal_percentage_steps_then_catches_up():
    instance = controller('exposure_basic', night=False)
    exposures = [25.449827, 23.546463]
    for _ in range(8):
        set_command(instance, exposures[-1], 0)
        instance.compare_highlights(HighlightMeasurement(0, 0, 110), exposures[-2], 0)
        expected = math.sqrt(exposures[-1] * exposures[-2] * 90 / 110)
        assert requested(instance)[0] == pytest.approx(expected, abs=1e-6)
        exposures.append(requested(instance)[0])
    # Under this repeated source-relative error, the delayed recurrence's
    # steady ratio is the cube root of the measured ADU correction.
    assert exposures[-1] / exposures[-2] == pytest.approx((90 / 110) ** (1 / 3), abs=.0002)
    set_command(instance, exposures[-1], 0)
    instance.compare_highlights(HighlightMeasurement(0, 0, 110), exposures[-1], 0)
    assert requested(instance)[0] == pytest.approx(exposures[-1] * math.sqrt(90 / 110), abs=1e-6)


@pytest.mark.parametrize('scale', [.999, .97, .9, .8, .6])
def test_changing_severity_never_cuts_past_the_measured_target_or_raises_pending(scale):
    instance = controller('exposure_autogain_exp_prio_db_1_10')
    instance._expUtils.GAIN_MAX_NIGHT = 300
    instance.gain_quantum = 1
    measured, pending = (30, 216), (30, 210)
    set_command(instance, *pending)
    target = instance._calculate_exposure(*measured, measured[0] * scale, highlight=True)[:2]
    instance._set_exposure(*measured, measured[0] * scale, highlight=True)
    result = signal(instance, *requested(instance))
    assert min(signal(instance, *target), signal(instance, *pending)) <= result + 1e-9
    assert result <= signal(instance, *pending) + 1e-9
    # Repeating an old measurement cannot blindly compound the pending cut.
    for _ in range(4):
        instance._set_exposure(*measured, measured[0] * scale, highlight=True)
        assert signal(instance, *requested(instance)) >= min(signal(instance, *target), signal(instance, *pending)) - 1e-9


def test_slew_crosses_gain_floor_with_the_remaining_signal_change_in_exposure():
    instance = controller('exposure_autogain_exp_prio_db_1_10')
    instance.gain_quantum = 1
    instance.compare_highlights(HighlightMeasurement(0, 0, 100), 30, 10)
    set_command(instance, 30, 2)
    instance.compare_highlights(HighlightMeasurement(0, 0, 100), 30, 6)
    exposure, gain = requested(instance)
    assert gain == 0 and 28 < exposure < 30
    absolute_target = signal(instance, 30, 6) * 80 / 100
    expected = math.sqrt(signal(instance, 30, 2) * absolute_target)
    assert signal(instance, exposure, gain) == pytest.approx(expected, abs=1e-6)


def test_unmodelled_legacy_gain_steps_keep_existing_policy():
    original, delayed = [controller('exposure_legacy_autogain') for _ in range(2)]
    for instance in (original, delayed):
        instance.post_init()
        gain = instance.auto_gain_step_list[-1]
        pending_gain = instance.auto_gain_step_list[-2]
        set_command(instance, 20, pending_gain, instance is delayed)
        instance._set_exposure(20, gain, 16, highlight=True)
    assert requested(original) == requested(delayed)
