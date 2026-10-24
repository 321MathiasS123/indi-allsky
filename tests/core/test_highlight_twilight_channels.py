"""Colour-tail forecasts supplement twilight's existing spatial warning."""
import itertools
import math

import numpy as np
import pytest

from indi_allsky.highlight import HighlightMeasurement, measure
from test_highlight_exposure import MODE_NAMES, controller
from test_highlight_twilight_control import set_inflight, settings, twilight_controller


def sample(levels=(.918, .3, .25), time=120, adu=70, warning=True, weight=1, raw=None, quantum=1 / 65535):
    raw = levels if raw is None else raw
    return HighlightMeasurement(0, 0, adu, any_next=3 if warning else 0,
                                twilight_weight=weight, twilight_time=time,
                                twilight_blue=levels[0], twilight_green=levels[1],
                                twilight_red=levels[2], twilight_blue_raw=raw[0],
                                twilight_green_raw=raw[1], twilight_red_raw=raw[2],
                                twilight_quantum=quantum)


def pair(old=(.85, .3, .25), new=(.918, .3, .25), weight=1, **changes):
    instance = twilight_controller()
    instance.config['EXPOSURE_PERIOD'] = 20
    set_inflight(instance, (1, 0))
    instance.compare_highlights(sample(old, time=100, warning=False, weight=weight), 1, 0)
    set_inflight(instance, (1, 0))
    instance.compare_highlights(sample(new, weight=weight, **changes), 1, 0)
    return instance


@pytest.mark.parametrize('order', list(itertools.permutations(range(3))))
def test_an_approaching_channel_tightens_the_reserve_even_when_gray_is_steady(order):
    before, after = (.85, .3, .25), (.918, .3, .25)
    actual = pair(tuple(before[i] for i in order), tuple(after[i] for i in order))
    reference = pair((0, 0, 0), (0, 0, 0))
    assert settings(actual)[0] == pytest.approx(.9, abs=.000002)
    assert settings(reference)[0] == pytest.approx(.95, abs=.000002)


def test_separate_channel_history_sees_the_next_dominant_colour():
    instance = pair((.83, .91, .3), (.94, .91, .3))
    assert settings(instance)[0] == pytest.approx(.9, abs=.000002)


def test_fast_growth_in_a_faint_channel_does_not_overrule_existing_headroom():
    instance = pair((.93, .3, .1), (.93, .3, .2))
    assert settings(instance)[0] == pytest.approx(.95, abs=.000002)


def test_colour_statistic_cannot_create_its_own_clipping_warning():
    instance = pair(warning=False)
    assert settings(instance)[0] == 1
    assert instance._twilight_prediction_previous[-1] is False


def test_raw_headroom_seeds_growth_even_with_a_substantial_additive_dark_level():
    instance = twilight_controller()
    instance.config['EXPOSURE_PERIOD'] = 20
    set_inflight(instance, (1, 0))
    instance.compare_highlights(sample((.8, .3, .25), raw=(.9, .3, .25),
                                       time=100, warning=False), 1, 0)
    set_inflight(instance, (1, 0))
    instance.compare_highlights(sample((.816, .3, .25), raw=(.916, .3, .25)), 1, 0)
    assert settings(instance)[0] == pytest.approx(.95 / 1.02 ** 2, abs=.000002)


def test_projected_raw_intensity_can_identify_an_approaching_channel_before_its_near_band():
    instance = pair((.7, .95, .25), (.88, .95, .25))
    # Blue is outside the current +10% near-clipping band, but its projected
    # growth overtakes green and crosses the configured raw clipping level.
    assert settings(instance)[0] == pytest.approx(.9, abs=.000002)


@pytest.mark.parametrize('raw', [.995, 1, float('nan'), float('inf'), 0])
def test_raw_certificate_remains_conservative_when_clipping_threshold_is_100_percent(raw):
    instance = twilight_controller()
    instance.config['HIGHLIGHT_PROTECTION']['THRESHOLD'] = 100
    set_inflight(instance, (1, 0))
    instance.compare_highlights(sample((.85, .3, .25), raw=(raw, .3, .25),
                                       time=100, warning=False), 1, 0)
    set_inflight(instance, (1, 0))
    instance.compare_highlights(sample(raw=(raw, .3, .25)), 1, 0)
    assert settings(instance)[0] == pytest.approx(.95, abs=.000002)


def test_a_calibrated_value_without_its_raw_certificate_cannot_seed_later_growth():
    instance = twilight_controller()
    set_inflight(instance, (1, 0))
    instance.compare_highlights(sample((.85, .3, .25), raw=(0, .3, .25),
                                       time=100, warning=False), 1, 0)
    assert instance._twilight_prediction_previous[5][0] == 0
    set_inflight(instance, (1, 0))
    instance.compare_highlights(sample(), 1, 0)
    assert settings(instance)[0] == pytest.approx(.95, abs=.000002)


@pytest.mark.parametrize('quantum', [0, -1, 2, float('nan'), float('inf')])
def test_missing_or_invalid_precision_retains_the_older_forecast(quantum):
    instance = twilight_controller()
    set_inflight(instance, (1, 0))
    instance.compare_highlights(sample((.85, .3, .25), time=100,
                                       warning=False, quantum=quantum), 1, 0)
    set_inflight(instance, (1, 0))
    instance.compare_highlights(sample(quantum=quantum), 1, 0)
    assert settings(instance)[0] == pytest.approx(.95, abs=.000002)


def test_a_bit_depth_change_restarts_colour_growth_history():
    instance = twilight_controller()
    set_inflight(instance, (1, 0))
    instance.compare_highlights(sample((.85, .3, .25), time=100, warning=False), 1, 0)
    set_inflight(instance, (1, 0))
    instance.compare_highlights(sample(quantum=1 / 255), 1, 0)
    assert settings(instance)[0] == pytest.approx(.95, abs=.000002)
    set_inflight(instance, (1, 0))
    instance.compare_highlights(sample((.98, .3, .25), time=140, quantum=1 / 255), 1, 0)
    assert settings(instance)[0] == pytest.approx(.9, abs=.000002)


@pytest.mark.parametrize('depth', [8, 12, 16])
def test_one_code_rounding_during_a_real_exposure_cut_does_not_become_scene_growth(depth):
    quantum = 1 / ((1 << depth) - 1)
    before = round(.94 / quantum) * quantum
    after = math.ceil(.95 * before / quantum) * quantum
    instance = twilight_controller()
    set_inflight(instance, (1, 0))
    instance.compare_highlights(sample((before, .3, .25), time=100,
                                       warning=False, quantum=quantum), 1, 0)
    set_inflight(instance, (.95, 0))
    instance.compare_highlights(sample((after, .285, .2375), quantum=quantum, adu=70 * .95), .95, 0)
    assert settings(instance)[0] == pytest.approx(.95 ** 2, abs=.000002)


@pytest.mark.parametrize('value', [0, -.1, .99, 1, float('nan'), float('inf')])
@pytest.mark.parametrize('unavailable', ['before', 'after'])
def test_invalid_or_censored_channels_do_not_supply_a_growth_rate(value, unavailable):
    before = (value, .3, .25) if unavailable == 'before' else (.85, .3, .25)
    after = (value, .3, .25) if unavailable == 'after' else (.918, .3, .25)
    instance = pair(before, after)
    assert settings(instance)[0] == pytest.approx(.95, abs=.000002)


@pytest.mark.parametrize('weight', [0, .00001, .05, .25, .5, .75, 1])
def test_channel_reserve_retains_the_existing_solar_fade(weight):
    instance = pair(weight=weight)
    assert settings(instance)[0] == pytest.approx(.9 ** weight, abs=.000002)


@pytest.mark.parametrize('adu', [1, 10, 17.5, 17.8])
def test_channel_forecast_cannot_override_the_normal_shadow_floor(adu):
    actual = pair(adu=adu)
    reference = pair((0, 0, 0), (0, 0, 0), adu=adu)
    assert settings(actual) == settings(reference)


def test_colour_cut_near_the_floor_only_uses_permitted_lift():
    instance = pair(adu=18)
    assert settings(instance)[0] * 18 == pytest.approx(17.5, abs=.00002)


def test_shadow_floor_release_clears_all_colour_threat_history():
    instance = pair(adu=18)
    set_inflight(instance, settings(instance))
    instance.compare_highlights(sample(adu=17.5, time=140), *settings(instance))
    assert instance._twilight_prediction_previous[-1] is False
    assert instance._twilight_prediction_previous[-2] == (False, False, False)


@pytest.mark.parametrize('reset', ['gap', 'mode', 'off', 'outside'])
def test_channel_history_is_discarded_at_invalid_continuity(reset):
    instance = twilight_controller()
    set_inflight(instance, (1, 0))
    instance.compare_highlights(sample((.85, .3, .25), time=100, warning=False), 1, 0)
    if reset == 'mode':
        instance.night_av[0] = 0
    if reset in ('off', 'outside'):
        instance.config['HIGHLIGHT_PROTECTION']['TWILIGHT_ENABLE'] = reset != 'off'
        set_inflight(instance, (1, 0))
        instance.compare_highlights(sample(time=110, warning=False, weight=0), 1, 0)
        instance.config['HIGHLIGHT_PROTECTION']['TWILIGHT_ENABLE'] = True
    set_inflight(instance, (1, 0))
    instance.compare_highlights(sample(time=1000 if reset == 'gap' else 120), 1, 0)
    assert settings(instance)[0] == pytest.approx(.95, abs=.000002)


def test_actual_exposure_and_gain_changes_do_not_become_colour_growth():
    instance = twilight_controller('gain')
    before, after = (30, 100), (30, 160)
    multiplier = 10 ** .3
    set_inflight(instance, before)
    instance.compare_highlights(sample((.4, .3, .25), time=100, adu=40, warning=False), *before)
    set_inflight(instance, after)
    instance.compare_highlights(sample(tuple(level * multiplier for level in (.4, .3, .25)),
                                       adu=40 * multiplier), *after)
    next_exposure, next_gain = settings(instance)
    ratio = next_exposure / after[0] * 10 ** ((next_gain - after[1]) / 200)
    assert ratio == pytest.approx(.95, rel=.012)


def test_a_new_colour_warning_preserves_a_stronger_pending_cut():
    instance = twilight_controller()
    set_inflight(instance, (1, 0))
    instance.compare_highlights(sample((.85, .3, .25), time=100, warning=False), 1, 0)
    set_inflight(instance, (.8, 0))
    instance.compare_highlights(sample(), 1, 0)
    assert settings(instance)[0] == pytest.approx(.8, abs=.000002)
    for time in (120, 110, 120):
        instance.compare_highlights(sample(time=time), 1, 0)
        assert settings(instance)[0] == pytest.approx(.8, abs=.000002)


@pytest.mark.parametrize('name', MODE_NAMES)
@pytest.mark.parametrize('night,moon', [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize('inactive', ['disabled', 'outside'])
def test_colour_forecasts_preserve_every_actuator_and_ordinary_mode(name, night, moon, inactive):
    actual, reference = controller(name, night, moon), controller(name, night, moon)
    actual.config['HIGHLIGHT_PROTECTION']['TWILIGHT_ENABLE'] = inactive != 'disabled'
    capture = (1, actual.gain_max if name == 'exposure_basic' else actual.gain_min)
    for index, (adu, levels) in enumerate([(70, (.85, .3, .25)), (70, (.918, .3, .25)),
                                         (10, (.98, .95, .8)), (120, (.2, .93, .4))]):
        for instance in (actual, reference):
            set_inflight(instance, capture)
        actual.compare_highlights(sample(levels, time=100 + index * 20, adu=adu,
                                         weight=0 if inactive == 'outside' else 1), *capture)
        reference.compare_highlights(sample((0, 0, 0), time=0, weight=0, adu=adu), *capture)
        assert list(actual.exposure_av) == list(reference.exposure_av)
        assert list(actual.gain_av) == list(reference.gain_av)
        assert list(actual.binning_av) == list(reference.binning_av)
        capture = settings(reference)


def with_levels(measured, raw, time):
    levels = [float(np.percentile(raw[:, :, index], 98.7)) / 65535 for index in range(3)]
    levels = [level if 0 < level < .99 else 0 for level in levels]
    return measured._replace(twilight_weight=1, twilight_time=time,
                             twilight_blue=levels[0], twilight_green=levels[1],
                             twilight_red=levels[2], twilight_blue_raw=levels[0],
                             twilight_green_raw=levels[1], twilight_red_raw=levels[2],
                             twilight_quantum=1 / 65535)


@pytest.mark.parametrize('peak', [.94, 1.0])
def test_static_moon_warning_retains_the_same_settled_exposure(peak):
    actual, reference = twilight_controller(), twilight_controller()
    field = np.full((96, 96, 3), .27)
    field[40:48, 28:68] = peak
    mask = np.ones(field.shape[:2], np.uint8)
    capture = (1., 0)
    for frame in range(30):
        raw = (np.minimum(field * capture[0], 1) * 65535).astype(np.uint16)
        measured = measure(raw, mask, 16)
        for instance in (actual, reference):
            set_inflight(instance, capture)
        actual.compare_highlights(with_levels(measured, raw, 100 + frame * 20), *capture)
        reference.compare_highlights(measured._replace(twilight_weight=1, twilight_time=100 + frame * 20), *capture)
        assert settings(actual) == pytest.approx(settings(reference), abs=.000002)
        capture = settings(reference)


@pytest.mark.parametrize('peak', [.94, 1.0])
def test_fresh_static_moon_captures_do_not_ratchet_after_their_small_reserve(peak):
    instance = twilight_controller()
    field = np.full((96, 96, 3), .27)
    field[40:48, 28:68] = peak
    mask = np.ones(field.shape[:2], np.uint8)
    capture = (1., 0)
    final = []
    for frame in range(40):
        set_inflight(instance, capture)
        raw = (np.minimum(field * capture[0], 1) * 65535).astype(np.uint16)
        measured = with_levels(measure(raw, mask, 16), raw, 100 + frame * 20)
        instance.compare_highlights(measured, *capture)
        capture = settings(instance)
        final.append(capture[0])
    assert min(final[-10:]) == pytest.approx(max(final[-10:]), abs=.000002)
    assert min(final[-10:]) == pytest.approx(.8 if peak == 1 else .95, abs=.000002)


@pytest.mark.parametrize('queued', [1, 3])
def test_colour_protection_keeps_published_commands_bounded_with_older_captures_queued(queued):
    instance = twilight_controller()
    instance.config['EXPOSURE_PERIOD'] = 20
    field = np.full((96, 96, 3), .12)
    field[:, :, 1], field[:, :, 2] = .36, .15
    patch = np.zeros(field.shape[:2], bool)
    patch[24:72, 24:72] = True
    mask = np.ones(field.shape[:2], np.uint8)
    pending = [(1., 0)] * queued
    commands, peaks = [], []
    for frame in range(12):
        capture = pending.pop(0)
        set_inflight(instance, pending[-1] if pending else capture)
        blue = .85 * 1.08 ** frame
        sky = field.copy()
        sky[patch, 0] = blue
        # A slight green change hides blue growth in mean grayscale radiance.
        sky[:, :, 1] -= .25 * .114 / .587 * (blue - .85)
        linear = sky * capture[0]
        raw = (np.minimum(linear, 1) * 65535).astype(np.uint16)
        measured = with_levels(measure(raw, mask, 16), raw, 100 + frame * 20)
        instance.compare_highlights(measured, *capture)
        command = settings(instance)
        if commands:
            assert .8 - .000002 <= command[0] / commands[-1][0] <= 1.1 + .000002
        commands.append(command)
        peaks.append(float(linear.max()))
        pending.append(command)
    assert max(peaks) < (1.25 if queued == 3 else 1.05)
    assert instance.highlight_transition.active


@pytest.mark.parametrize('growth,frames', [(1.02, 36), (1.08, 12)])
def test_channel_headroom_hold_does_not_recover_into_continuing_chromatic_dawn(growth, frames):
    instance = twilight_controller()
    instance.config['EXPOSURE_PERIOD'] = 20
    field = np.full((96, 96, 3), .12)
    field[:, :, 1], field[:, :, 2] = .36, .15
    patch = np.zeros(field.shape[:2], bool)
    patch[24:72, 24:72] = True
    mask = np.ones(field.shape[:2], np.uint8)
    capture = (1., 0)
    for frame in range(frames):
        set_inflight(instance, capture)
        blue = .905 * growth ** frame
        sky = field.copy()
        sky[patch, 0] = blue
        sky[:, :, 1] -= .25 * .114 / .587 * (blue - .905)
        linear = sky * capture[0]
        raw = (np.minimum(linear, 1) * 65535).astype(np.uint16)
        measured = with_levels(measure(raw, mask, 16), raw, 100 + frame * 20)
        instance.compare_highlights(measured, *capture)
        command = settings(instance)
        assert measured.adu > 17.5 * 1.02
        assert command[0] <= capture[0] + .000002
        assert float(linear.max()) < .99
        capture = command
