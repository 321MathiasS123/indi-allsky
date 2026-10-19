"""Bound twilight commands after ordinary recovery and the solar fade blend."""
import pytest
import numpy as np

from indi_allsky.highlight import HighlightMeasurement, measure
from test_highlight_exposure import controller
from test_highlight_twilight_control import measurement, set_inflight, settings, signal, twilight_controller


def fast_measurement(adu=10, peak=.12, weight=1, time=100):
    return measurement(adu, peak, weight, time)._replace(full_fast=0, any_fast=0)


@pytest.mark.parametrize('weight', [1, .5, .05, .00001])
@pytest.mark.parametrize('peak', [0, .12])
@pytest.mark.parametrize('actuator', ['exposure', 'gain'])
def test_fast_recovery_switch_stays_bounded_through_entire_solar_fade(weight, peak, actuator):
    instance = twilight_controller(actuator)
    source = (30, 142) if actuator == 'gain' else (1, 0)
    set_inflight(instance, source)
    instance.compare_highlights(fast_measurement(peak=peak, weight=weight), *source)
    ratio = signal(instance, settings(instance)) / signal(instance, source)
    # A ZWO gain step may round the 10% bound by one hardware increment.
    quantum = 10 ** (instance.gain_quantum / 200) if actuator == 'gain' else 1.000002
    assert 1 < ratio <= 1.1 * quantum


@pytest.mark.parametrize('weight', [1, .5, .00001])
@pytest.mark.parametrize('source', [(30, 142), (30, 2), (1, 0)])
def test_normal_and_horizon_cuts_are_slewed_through_gain_floor(weight, source):
    instance = twilight_controller('gain')
    set_inflight(instance, source)
    clipped = HighlightMeasurement(3, 8, 70, twilight_peak=1,
                                   twilight_weight=weight, twilight_time=100)
    instance.compare_highlights(clipped, *source)
    ratio = signal(instance, settings(instance)) / signal(instance, source)
    assert .8 / 10 ** .005 <= ratio < 1
    assert settings(instance)[0] <= source[0] and settings(instance)[1] <= source[1]


@pytest.mark.parametrize('direction', ['dawn', 'dusk'])
def test_two_inflight_captures_use_newest_pending_command_without_signal_pumping(direction):
    instance = twilight_controller('gain')
    instance.config['EXPOSURE_PERIOD'] = 40
    source = (30, 220)
    pending = [source, source, source]
    reference = signal(instance, source)
    commands = []
    for frame in range(50):
        source = pending.pop(0)
        # Shared NEXT retains the newest command while two older captures wait.
        set_inflight(instance, pending[-1])
        growth = (1.04 if direction == 'dawn' else .96) ** frame
        brightness = growth * signal(instance, source) / reference
        instance.compare_highlights(
            fast_measurement(20 * brightness, min(.85 * brightness, 1), time=100 + frame * 40), *source)
        command = settings(instance)
        if commands:
            ratio = signal(instance, command) / signal(instance, commands[-1])
            assert .8 / 10 ** .005 <= ratio <= 1.1 * 10 ** .005
        commands.append(command)
        pending.append(command)
    values = [signal(instance, capture) for capture in commands[12:]]
    if direction == 'dawn':
        assert all(after <= before for before, after in zip(values, values[1:]))
        assert all(after[0] <= before[0] and after[1] <= before[1]
                   for before, after in zip(commands[12:], commands[13:]))
    else:
        assert all(after >= before for before, after in zip(values, values[1:]))


def test_known_twelve_percent_radiance_growth_tracks_with_two_older_captures_inflight():
    instance = twilight_controller('gain')
    instance.config['EXPOSURE_PERIOD'] = 20
    initial = (10, 240)
    pending = [initial] * 3
    reference = signal(instance, initial)
    radiance = np.full((96, 96), .08)
    radiance[28:48, 12:72] = .75
    mask = np.ones(radiance.shape, np.uint8)
    commands, peaks = [], []
    for frame in range(70):
        capture = pending.pop(0)
        set_inflight(instance, pending[-1])
        # Independently known scene radiance and the mode's actual gain model;
        # no rendered-image inversion or fitted candidate output is involved.
        linear = radiance * 1.12 ** frame * signal(instance, capture) / reference
        raw = (np.minimum(linear, 1) * 65535).astype(np.uint16)
        sample = measure(raw, mask, 16)._replace(twilight_peak=min(float(linear.max()), 1),
                                               twilight_weight=1, twilight_time=100 + frame * 20)
        instance.compare_highlights(sample, *capture)
        command = settings(instance)
        if commands:
            ratio = signal(instance, command) / signal(instance, commands[-1])
            assert .8 / 10 ** .005 <= ratio <= 1.1 * 10 ** .005
        commands.append(command)
        peaks.append(float(linear.max()))
        pending.append(command)
    # The two-interval forecast and three-capture queue leave a small overshoot,
    # but the horizon must not run away while recovery remains limited to 10%.
    assert max(peaks[-20:]) < 1.02


def test_repeated_or_older_capture_cannot_compound_its_slewed_cut():
    instance = twilight_controller('gain')
    source = (30, 200)
    set_inflight(instance, source)
    sample = measurement(70, 1, time=100)
    instance.compare_highlights(sample, *source)
    first = settings(instance)
    for sample_time in (100, 100, 90, 100):
        instance.compare_highlights(sample._replace(twilight_time=sample_time), *source)
        assert settings(instance) == first


@pytest.mark.parametrize('direction', ['cut', 'recovery'])
def test_unmodelled_legacy_gain_step_keeps_its_smallest_camera_policy_step(direction):
    instance = controller('exposure_legacy_autogain')
    instance.config['HIGHLIGHT_PROTECTION']['TWILIGHT_ENABLE'] = True
    instance.post_init()
    exposure = (instance.auto_gain_exposure_cutoff_low if direction == 'cut'
                else instance.auto_gain_exposure_cutoff_high)
    source = (exposure, instance.auto_gain_step_list[3])
    set_inflight(instance, source)
    sample = measurement(70, 1) if direction == 'cut' else fast_measurement()
    instance.compare_highlights(sample, *source)
    command = settings(instance)
    assert command[0] == pytest.approx(exposure)
    assert command[1] == instance.auto_gain_step_list[2 if direction == 'cut' else 4]
    assert instance._highlight_signal_change(*command, *source) is None


@pytest.mark.parametrize('timestamp', [0, -1, float('nan'), float('inf')])
@pytest.mark.parametrize('previous_twilight', [False, True])
def test_invalid_twilight_timestamp_neither_activates_nor_retains_slew(timestamp, previous_twilight):
    actual, reference = twilight_controller(), twilight_controller()
    source = (1, 0)
    if previous_twilight:
        set_inflight(actual, source)
        actual.compare_highlights(measurement(70, 1), *source)
        source = settings(actual)
    for instance in (actual, reference):
        set_inflight(instance, source)
    sample = fast_measurement(time=timestamp)
    actual.compare_highlights(sample, *source)
    reference.compare_highlights(sample._replace(twilight_peak=0, twilight_weight=0, twilight_time=0), *source)
    assert settings(actual) == settings(reference)
    assert actual._twilight_slew_state is None


def test_zero_weight_finishes_recovery_smoothly_then_restores_ordinary_control():
    instance = twilight_controller()
    source = (1, 0)
    for frame in range(30):
        set_inflight(instance, source)
        # Fade has ended after the first frame; the remaining release is finite.
        instance.compare_highlights(fast_measurement(10 * source[0], weight=1 if frame == 0 else 0,
                                                    time=100 + frame * 20), *source)
        command = settings(instance)
        assert command[0] <= source[0] * 1.1 + .000002
        source = command
    assert instance._twilight_slew_state is None
    set_inflight(instance, source)
    instance.compare_highlights(fast_measurement(adu=1, weight=0, time=800), *source)
    assert settings(instance)[0] > source[0] * 1.1


@pytest.mark.parametrize('hold', ['ordinary', 'await_pending'])
def test_release_survives_a_hold_while_last_twilight_command_is_still_pending(hold, caplog):
    instance = twilight_controller()
    source = (1, 0)
    set_inflight(instance, source)
    instance.compare_highlights(measurement(70, 1), *source)
    pending = settings(instance)
    if hold == 'await_pending':
        instance.config['HIGHLIGHT_PROTECTION']['OUTPUT_ENABLE'] = True
        instance.highlight_output.observe(HighlightMeasurement(3, 8, 0), *source,
                                          tuple(instance.night_av), instance.config['HIGHLIGHT_PROTECTION'])
    with caplog.at_level('INFO', logger='indi_allsky'):
        instance.compare_highlights(fast_measurement(10 if hold == 'await_pending' else 70,
                                                    weight=0, time=120), *source)
    assert settings(instance) == pending
    assert instance._twilight_slew_state is not None
    if hold == 'await_pending':
        assert 'await pending exposure/gain' in caplog.text
    instance.highlight_output.reset()
    set_inflight(instance, pending)
    instance.compare_highlights(fast_measurement(10, weight=0, time=140), *pending)
    assert pending[0] < settings(instance)[0] <= pending[0] * 1.1 + .000002


@pytest.mark.parametrize('gain,quantum', [(142, 1), (142, .1), (.01, .001), (100, 100)])
def test_fixed_32_microsecond_exposure_uses_bounded_gain_or_smallest_hardware_step(gain, quantum):
    instance = twilight_controller('gain')
    instance.gain_quantum = quantum
    instance._expUtils.EXPOSURE_MIN_NIGHT = .000032
    instance._expUtils.EXPOSURE_MAX = .000032
    source = (.000032, gain)
    set_inflight(instance, source)
    instance.compare_highlights(measurement(70, 1), *source)
    command = settings(instance)
    assert command[0] == pytest.approx(source[0])
    assert 0 <= command[1] < gain
    ratio = signal(instance, command) / signal(instance, source)
    assert ratio >= .8 / 10 ** (quantum / 200)


@pytest.mark.parametrize('reset', ['disabled', 'mode', 'gap'])
def test_release_does_not_persist_after_disable_mode_change_or_large_capture_gap(reset):
    instance = twilight_controller()
    source = (1, 0)
    set_inflight(instance, source)
    instance.compare_highlights(fast_measurement(), *source)
    assert instance._twilight_slew_state is not None
    source = settings(instance)
    set_inflight(instance, source)
    if reset == 'disabled':
        instance.config['HIGHLIGHT_PROTECTION']['TWILIGHT_ENABLE'] = False
    elif reset == 'mode':
        instance.night_av[0] = 0
    instance.compare_highlights(fast_measurement(adu=1, weight=0,
                                                time=1000 if reset == 'gap' else 120), *source)
    assert instance._twilight_slew_state is None
    assert settings(instance)[0] > source[0] * 1.1
