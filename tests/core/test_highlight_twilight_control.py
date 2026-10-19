"""Twilight headroom control under changing light and delayed camera commands."""
import math

import pytest

from indi_allsky.highlight import HighlightMeasurement
from test_highlight_exposure import MODE_NAMES, controller


def measurement(adu=70, peak=.85, weight=1.0, time=100.0):
    return HighlightMeasurement(0, 0, adu, twilight_peak=peak,
                                twilight_weight=weight, twilight_time=time)


def settings(instance):
    return instance._expUtils.EXPOSURE_NEXT, instance._expUtils.GAIN_NEXT


def set_inflight(instance, capture):
    exposure, gain = capture
    instance._expUtils.EXPOSURE_CURRENT = exposure
    instance._expUtils.GAIN_CURRENT = gain
    instance._expUtils.EXPOSURE_NEXT = exposure
    instance._expUtils.GAIN_NEXT = gain


def signal(instance, capture):
    exposure, gain = capture
    return exposure * (10 ** (instance.gain2dB(gain) / 20)
                       if hasattr(instance, 'gain2dB') else 1)


def twilight_controller(actuator='exposure'):
    if actuator.startswith('gain'):
        instance = controller('exposure_autogain_exp_prio_db_1_10')
        instance._expUtils.GAIN_MAX_NIGHT = 400
        instance.gain_quantum = 1
    else:
        instance = controller('exposure_basic')
        instance._expUtils.GAIN_MIN_NIGHT = 0
        instance._expUtils.GAIN_MAX_NIGHT = 0
    instance.config['HIGHLIGHT_PROTECTION']['TWILIGHT_ENABLE'] = True
    return instance


@pytest.mark.parametrize('name', MODE_NAMES)
@pytest.mark.parametrize('night,moon', [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize('inactive', ['disabled', 'outside_twilight'])
def test_disabled_or_outside_twilight_retains_every_actuator_and_mode(name, night, moon, inactive):
    actual, reference = controller(name, night, moon), controller(name, night, moon)
    actual.config['HIGHLIGHT_PROTECTION']['TWILIGHT_ENABLE'] = inactive != 'disabled'
    capture = (1, actual.gain_max if name == 'exposure_basic' else actual.gain_min)
    for frame, (adu, peak) in enumerate(((30, 1), (70, .95), (120, .98), (10, 1))):
        for instance in (actual, reference):
            set_inflight(instance, capture)
        actual.compare_highlights(measurement(adu, peak, 0 if inactive == 'outside_twilight' else 1,
                                              100 + frame * 20), *capture)
        reference.compare_highlights(HighlightMeasurement(0, 0, adu), *capture)
        assert list(actual.exposure_av) == list(reference.exposure_av)
        assert list(actual.gain_av) == list(reference.gain_av)
        assert list(actual.binning_av) == list(reference.binning_av)
        assert actual.target_adu_found == reference.target_adu_found
        capture = settings(reference)


def test_partial_twilight_weight_changes_headroom_constraint_smoothly():
    ratios = []
    for weight in (0, .05, .25, .5, .75, .95, 1):
        instance = twilight_controller()
        set_inflight(instance, (1, 0))
        instance.compare_highlights(measurement(10, 1, weight), 1, 0)
        ratios.append(settings(instance)[0])
    assert ratios[0] > 1
    assert ratios[-1] < 1
    assert all(before > after for before, after in zip(ratios, ratios[1:]))
    assert all(before - after < .1 for before, after in zip(ratios, ratios[1:]))


@pytest.mark.parametrize('adu', [1, 10, 17.5, 30, 70])
def test_broad_twilight_near_clipping_can_override_shadow_recovery(adu):
    instance = twilight_controller()
    instance.config['HIGHLIGHT_PROTECTION']['MAX_BOOST'] = 0
    set_inflight(instance, (1, 0))
    instance.compare_highlights(measurement(adu, 1), 1, 0)
    assert 0 < settings(instance)[0] < 1


@pytest.mark.parametrize('actuator', ['exposure', 'gain', 'gain_then_exposure'])
def test_brightening_dawn_tracks_without_clipping_or_repeated_signal_reversals(actuator):
    instance = twilight_controller(actuator)
    instance.config['EXPOSURE_PERIOD'] = 40
    initial = (30, 260) if actuator == 'gain' else (30, 70) if actuator == 'gain_then_exposure' else (20, 0)
    source = inflight = initial
    initial_signal = signal(instance, initial)
    commands, peaks = [], []
    for frame in range(36):
        # A capture is already in flight when the preceding frame completes.
        # Scene growth is independent of camera exposure and gain changes.
        set_inflight(instance, inflight)
        brightness = 1.06 ** frame * signal(instance, source) / initial_signal
        peak = .85 * brightness
        peaks.append(peak)
        instance.compare_highlights(measurement(15 * brightness, min(peak, 1), time=100 + frame * 40),
                                    *source)
        command = settings(instance)
        commands.append(command)
        source, inflight = inflight, command
    values = [signal(instance, command) for command in commands]
    # Ignore initial trend acquisition; sustained dawn must not cause the
    # repeating cut/recovery cycle of a clipping-only controller.
    assert max(peaks[6:]) < .99
    assert all(after <= before * 1.001 for before, after in zip(values[6:], values[7:]))
    assert values[-1] < values[6] * .6
    if actuator == 'gain':
        assert all(exposure == 30 for exposure, _ in commands)
    elif actuator == 'gain_then_exposure':
        assert commands[-1][0] < 30 and commands[-1][1] == 0
        assert all(after[0] <= before[0] and after[1] <= before[1]
                   for before, after in zip(commands[6:], commands[7:]))
    else:
        assert all(gain == 0 for _, gain in commands)


@pytest.mark.parametrize('name', MODE_NAMES)
def test_older_twilight_frame_cannot_undo_a_stronger_pending_cut(name):
    instance = controller(name)
    instance.config['HIGHLIGHT_PROTECTION']['TWILIGHT_ENABLE'] = True
    source = (1, instance.gain_max if name == 'exposure_basic' else instance.gain_min)
    set_inflight(instance, source)
    instance.compare_highlights(measurement(10, 1), *source)
    pending = settings(instance)
    assert pending != source
    instance.compare_highlights(measurement(10, .95), *source)
    assert settings(instance) == pending
    instance.compare_highlights(measurement(10, .1), *source)
    assert settings(instance) == pending


def test_gain_change_is_normalized_before_predicting_dawn_growth():
    instance = twilight_controller('gain')
    source = (30, 100)
    set_inflight(instance, source)
    instance.compare_highlights(measurement(35, .42), *source)
    brighter_capture = (30, 160)
    ratio = signal(instance, brighter_capture) / signal(instance, source)
    set_inflight(instance, brighter_capture)
    instance.compare_highlights(measurement(35 * ratio, .42 * ratio, time=120), *brighter_capture)
    # The raw increase is entirely explained by applied gain. Inventing a
    # second sky-brightening trend here would make an unnecessary exposure cut.
    assert settings(instance) == brighter_capture


def test_stale_dawn_trend_is_not_reused_after_a_capture_gap():
    instance = twilight_controller()
    for peak, time in ((.6, 100), (.85, 120)):
        set_inflight(instance, (1, 0))
        instance.compare_highlights(measurement(70, peak, time=time), 1, 0)
    assert settings(instance)[0] < 1
    set_inflight(instance, (1, 0))
    instance.compare_highlights(measurement(70, .85, time=1000), 1, 0)
    assert settings(instance) == (1, 0)


def test_compact_source_exclusion_leaves_dark_sky_recovery_available():
    instance = twilight_controller()
    set_inflight(instance, (1, 0))
    # The image meter has excluded the bright Moon/stars. The broad horizon
    # statistic therefore remains low even though individual pixels clip.
    instance.compare_highlights(measurement(10, .12), 1, 0)
    assert settings(instance)[0] > 1


def test_fading_sunset_recovers_smoothly_without_crossing_horizon_ceiling():
    instance = twilight_controller()
    instance.config['EXPOSURE_PERIOD'] = 40
    source = (1., 0)
    commands, peaks = [], []
    for frame in range(30):
        set_inflight(instance, source)
        brightness = .94 ** frame * source[0]
        peak = .85 * brightness
        peaks.append(peak)
        instance.compare_highlights(measurement(20 * brightness, peak, time=100 + frame * 40), *source)
        source = settings(instance)
        commands.append(source[0])
    assert max(peaks) <= .901
    assert all(after >= before for before, after in zip(commands, commands[1:]))
    assert commands[-1] > commands[0] * 3


def test_day_night_mode_change_discards_previous_twilight_scene_trend():
    instance = twilight_controller()
    set_inflight(instance, (1, 0))
    instance.compare_highlights(measurement(70, .6), 1, 0)
    instance.night_av[0] = 0
    set_inflight(instance, (1, 0))
    instance.compare_highlights(measurement(80, .85, time=120), 1, 0)
    assert settings(instance) == (1, 0)
