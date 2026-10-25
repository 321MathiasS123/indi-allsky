"""Delayed final-image feedback keeps one bounded source-relative ceiling."""
import math

import numpy as np
import pytest

from indi_allsky.highlight import HighlightMeasurement
from indi_allsky.highlight_output_control import TwilightOutputControl
from test_highlight_exposure import MODE_NAMES, controller


MODE = (1, 0)
SETTINGS = {'ENABLE': True, 'TWILIGHT_ENABLE': True, 'TWILIGHT_OUTPUT_ENABLE': True,
            'TWILIGHT_OUTPUT_TARGET': 252, 'MAX_BOOST': 2}


def raw(adu=70, time=120, weight=1, **changes):
    values = dict(full=0, any=0, adu=adu, twilight_peak=0,
                  twilight_time=time, twilight_weight=weight)
    values.update(changes)
    return HighlightMeasurement(**values)


def rendered(peak=255, area=1, adu=70, time=100):
    return raw(adu, time, twilight_peak=peak / 255, any=area)


def physical_change(exposure, gain, source_exposure, source_gain):
    return math.log(exposure / source_exposure) + (gain - source_gain) * math.log(10) / 200


def ceiling(state, current=None, capture=(1, 0), pending=None, settings=SETTINGS, mode=MODE):
    return state.limit(current or raw(), *capture, mode, settings, physical_change,
                       *(pending or capture), 70)


@pytest.mark.parametrize('peak,area', [(255, 1), (255, .1), (253, .001), (252, 0), (240, 0)])
def test_target_and_area_drive_a_bounded_appearance_ceiling(peak, area):
    state = TwilightOutputControl()
    state.observe(rendered(peak, area), 1, 0, MODE)
    value = ceiling(state)
    assert .98 <= value <= 1.1
    if peak > 252 or area > 0:
        assert value < 1
    elif peak == 252:
        assert value == 1
    else:
        assert value > 1  # Safe output may release existing raw ADU recovery.


def test_censored_tail_retains_area_pressure_at_the_same_peak():
    weak, strong = TwilightOutputControl(), TwilightOutputControl()
    weak.observe(rendered(area=.1), 1, 0, MODE)
    strong.observe(rendered(area=4), 1, 0, MODE)
    assert .98 <= ceiling(strong) < ceiling(weak) < 1


def test_repeated_output_cannot_compound_a_cut_onto_newer_pending_settings():
    state = TwilightOutputControl()
    original = rendered()
    state.observe(original, 1, 0, MODE)
    assert ceiling(state) == pytest.approx(.98)
    for _ in range(5):
        state.observe(original, 1, 0, MODE)
        assert ceiling(state, pending=(.98, 0)) == pytest.approx(.98)
        assert ceiling(state, current=raw(68.6), capture=(.98, 0)) == pytest.approx(1)
    state.observe(rendered(adu=68.6, time=101), .98, 0, MODE)
    assert ceiling(state, current=raw(68.6), capture=(.98, 0)) == pytest.approx(.98)


def test_newer_or_weaker_pending_command_is_not_used_as_a_compounding_base():
    state = TwilightOutputControl()
    state.observe(rendered(), 1, 0, MODE)
    assert ceiling(state, pending=(1.1, 0)) == pytest.approx(1.1 * .98)
    assert ceiling(state, pending=(.8, 0)) == pytest.approx(1.1 * .98)


def test_known_gain_and_exposure_changes_are_normalized_in_physical_signal_units():
    state = TwilightOutputControl()
    state.observe(rendered(), 30, 100, MODE)
    first = ceiling(state, capture=(30, 100))
    new_gain = 100 + 200 * math.log10(first)
    current = raw(70 * first)
    assert ceiling(state, current, capture=(30, new_gain)) == pytest.approx(1)
    assert ceiling(state, current, capture=(30 * first, 100)) == pytest.approx(1)


def test_duplicate_or_regressing_output_ids_do_not_reanchor_the_ceiling():
    state = TwilightOutputControl()
    state.observe(rendered(), 1, 0, MODE)
    before = ceiling(state)
    anchor = state.anchor
    for time in (100, 99):
        state.observe(rendered(peak=240, area=0, time=time), 2, 0, MODE)
        assert ceiling(state) == before
        assert state.anchor == anchor


@pytest.mark.parametrize('adu', np.linspace(15, 24, 31))
def test_persistent_clipping_tapers_into_the_shadow_reserve(adu):
    state = TwilightOutputControl()
    state.observe(rendered(adu=adu), 1, 0, MODE)
    limit = ceiling(state, raw(adu))
    assert limit >= min(1.1, 19.25 / adu)


def configured(name='exposure_basic', night=True, moon=False):
    result = controller(name, night, moon)
    result.config.update(EXPOSURE_PERIOD=40, EXPOSURE_PERIOD_DAY=40)
    result.config['HIGHLIGHT_PROTECTION'].update(SETTINGS)
    if name == 'exposure_basic':
        result._expUtils.GAIN_MIN_NIGHT = result._expUtils.GAIN_MAX_NIGHT = 0
    return result


def pending(instance, capture):
    instance._expUtils.EXPOSURE_CURRENT, instance._expUtils.GAIN_CURRENT = capture
    instance._expUtils.EXPOSURE_NEXT, instance._expUtils.GAIN_NEXT = capture


def requested(instance):
    return instance._expUtils.EXPOSURE_NEXT, instance._expUtils.GAIN_NEXT


@pytest.mark.parametrize('name', MODE_NAMES)
@pytest.mark.parametrize('night,moon', [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize('inactive', ['off', 'day_or_night'])
def test_option_off_or_outside_twilight_preserves_all_actuator_decisions(name, night, moon, inactive):
    actual, reference = configured(name, night, moon), configured(name, night, moon)
    actual.config['HIGHLIGHT_PROTECTION']['TWILIGHT_OUTPUT_ENABLE'] = inactive != 'off'
    reference.config['HIGHLIGHT_PROTECTION']['TWILIGHT_OUTPUT_ENABLE'] = False
    capture = (1, actual.gain_min if name != 'exposure_basic' else actual.gain_max)
    mode = (int(night), int(moon))
    for index, adu in enumerate((30, 70, 120, 10)):
        item = raw(adu, time=120 + index * 40, weight=0 if inactive == 'day_or_night' else 1)
        for instance in (actual, reference):
            pending(instance, capture)
        actual.twilight_output.observe(rendered(time=100 + index * 40), *capture, mode)
        actual.compare_highlights(item, *capture)
        reference.compare_highlights(item, *capture)
        assert list(actual.exposure_av) == list(reference.exposure_av)
        assert list(actual.gain_av) == list(reference.gain_av)
        assert list(actual.binning_av) == list(reference.binning_av)
        capture = requested(reference)


def test_output_constraint_fades_continuously_into_the_existing_request():
    values = []
    for weight in np.linspace(0, 1, 21):
        instance = configured()
        pending(instance, (1, 0))
        instance.twilight_output.observe(rendered(), 1, 0, MODE)
        instance.compare_highlights(raw(weight=float(weight)), 1, 0)
        values.append(requested(instance)[0])
    assert values[0] == 1
    assert values[-1] == pytest.approx(.98, abs=1e-6)
    assert np.all(np.diff(values) <= 0)
    assert max(abs(np.diff(values))) < .002


def test_fresh_raw_forecast_retains_priority_over_an_older_output_ceiling():
    instance = configured()
    pending(instance, (1, 0))
    instance.twilight_output.observe(rendered(peak=240, area=0), 1, 0, MODE)
    instance.compare_highlights(raw(full_next=2), 1, 0)
    assert requested(instance)[0] <= .95 + 1e-6


@pytest.mark.parametrize('quantized', [False, True])
def test_static_highlight_closes_loop_with_a_30_second_capture_already_in_flight(quantized):
    instance = configured()
    initial = (30, 0)
    source = inflight = initial
    old_output = None
    commands, peaks = [], []
    for index in range(50):
        time = 100 + index * 40
        brightness = source[0] / initial[0]
        # Independent monotone, gamma-like display response; output saturation
        # hides the true peak until exposure has recovered some headroom.
        peak = min(255, 260 * brightness ** (1 / 1.8))
        if quantized:
            peak = round(peak)
        area = max(0, min(3, (260 * brightness ** (1 / 1.8) - 252) / 8))
        current = raw(60 * brightness, time=time)
        pending(instance, inflight)
        if old_output is not None:
            instance.twilight_output.observe(*old_output)
        instance.compare_highlights(current, *source)
        command = requested(instance)
        commands.append(command[0])
        peaks.append(peak)
        old_output = (rendered(peak, area, current.adu, time), *source, MODE)
        source, inflight = inflight, command
    assert max(peaks[-15:]) <= 253
    assert min(peaks[-15:]) >= 250
    assert max(commands[-15:]) / min(commands[-15:]) < 1.02
    assert all(.979 <= after / before <= 1.101 for before, after in zip(commands, commands[1:]))


def test_persistent_rendered_plateau_cannot_drive_a_fading_sky_below_its_lift_reserve():
    instance = configured()
    instance._expUtils.EXPOSURE_MAX = 120
    source = inflight = (10, 0)
    old_output = None
    adus = []
    for index in range(100):
        time = 100 + index * 40
        adu = 40 * .98 ** index * source[0] / 10
        current = raw(adu, time=time)
        pending(instance, inflight)
        if old_output is not None:
            instance.twilight_output.observe(*old_output)
        instance.compare_highlights(current, *source)
        command = requested(instance)
        old_output = (rendered(255, 3, adu, time), *source, MODE)
        adus.append(adu)
        source, inflight = inflight, command
    assert min(adus[-30:]) >= 17.5
    assert max(adus[-20:]) / min(adus[-20:]) < 1.05
