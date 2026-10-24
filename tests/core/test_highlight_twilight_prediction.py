"""Twilight headroom protects central cloud fields before they actually clip."""
import math

import numpy as np
import pytest

from indi_allsky.highlight import HighlightMeasurement, measure
from test_highlight_exposure import MODE_NAMES, controller
from test_highlight_twilight_control import set_inflight, settings, signal, twilight_controller


def prediction(adu=70, time=100, weight=1, **changes):
    values = dict(full=0, any=0, adu=adu, full_next=0, any_next=3,
                  twilight_peak=0, twilight_weight=weight, twilight_time=time)
    values.update(changes)
    return HighlightMeasurement(**values)


@pytest.mark.parametrize('channel', ['full_next', 'any_next'])
def test_prediction_cuts_before_an_actual_patch_exceeds_either_limit(channel):
    instance = twilight_controller()
    set_inflight(instance, (1, 0))
    sample = prediction(**{'any_next': 0, channel: 3})
    instance.compare_highlights(sample, 1, 0)
    assert sample.full == sample.any == sample.twilight_peak == 0
    assert settings(instance)[0] == pytest.approx(.95, abs=.000002)


@pytest.mark.parametrize('excess', [.000001, 1, 50])
def test_patch_area_excess_is_not_treated_as_linear_signal(excess):
    instance = twilight_controller()
    set_inflight(instance, (1, 0))
    instance.compare_highlights(prediction(any_next=2.4 + excess), 1, 0)
    assert settings(instance)[0] == pytest.approx(.95, abs=.000002)


def test_normalized_adu_trend_tightens_achieved_half_error_response():
    instance = twilight_controller()
    instance.config['EXPOSURE_PERIOD'] = 20
    # Today's ordinary ADU branch asks for .9 but eases the real command to
    # about .95. Central-cloud prediction must constrain that achieved command.
    set_inflight(instance, (1, 0))
    instance.compare_highlights(prediction(93.88, any_next=0), 1, 0)
    capture = settings(instance)
    set_inflight(instance, capture)
    instance.compare_highlights(prediction(93.88 * 1.06 * capture[0], time=120), *capture)
    assert settings(instance)[0] / capture[0] == pytest.approx(.9, abs=.000002)


@pytest.mark.parametrize('weight', [0, .00001, .05, .25, .5, .75, 1])
def test_solar_fade_blends_the_achieved_command_instead_of_removing_adu_easing(weight):
    instance = twilight_controller()
    set_inflight(instance, (1, 0))
    instance.compare_highlights(prediction(adu=81, weight=weight), 1, 0)
    ordinary = math.sqrt(80 / 81)
    assert settings(instance)[0] == pytest.approx(min(ordinary, .95 ** weight), abs=.000002)


@pytest.mark.parametrize('adu', [0, 1, 10, 17.5, 17.8])
def test_global_moon_warning_does_not_cut_below_normal_shadow_floor(adu):
    actual, reference = twilight_controller(), twilight_controller()
    reference.config['HIGHLIGHT_PROTECTION']['TWILIGHT_ENABLE'] = False
    for instance in (actual, reference):
        set_inflight(instance, (1, 0))
    actual.compare_highlights(prediction(adu), 1, 0)
    reference.compare_highlights(prediction(adu), 1, 0)
    assert settings(actual) == settings(reference)


def test_near_floor_prediction_tapers_to_the_permitted_brightness():
    instance = twilight_controller()
    set_inflight(instance, (1, 0))
    instance.compare_highlights(prediction(18), 1, 0)
    assert settings(instance)[0] * 18 == pytest.approx(17.5, abs=.00002)


@pytest.mark.parametrize('name', MODE_NAMES)
@pytest.mark.parametrize('inactive', ['disabled', 'outside'])
def test_new_prediction_retains_ordinary_behavior_in_every_mode_outside_twilight(name, inactive):
    actual, reference = controller(name), controller(name)
    actual.config['HIGHLIGHT_PROTECTION']['TWILIGHT_ENABLE'] = inactive != 'disabled'
    capture = (1, actual.gain_max if name == 'exposure_basic' else actual.gain_min)
    for frame, adu in enumerate((94, 70, 10, 120)):
        for instance in (actual, reference):
            set_inflight(instance, capture)
        sample = prediction(adu, 100 + frame * 20, weight=0 if inactive == 'outside' else 1)
        actual.compare_highlights(sample, *capture)
        reference.compare_highlights(sample._replace(twilight_weight=0, twilight_time=0), *capture)
        assert list(actual.exposure_av) == list(reference.exposure_av)
        assert list(actual.gain_av) == list(reference.gain_av)
        assert list(actual.binning_av) == list(reference.binning_av)
        capture = settings(reference)


def test_applied_gain_change_is_not_invented_dawn_growth():
    instance = twilight_controller('gain')
    source = (30, 100)
    set_inflight(instance, source)
    instance.compare_highlights(prediction(40, any_next=0), *source)
    source = (30, 160)
    set_inflight(instance, source)
    instance.compare_highlights(prediction(40 * 10 ** .3, time=120), *source)
    assert signal(instance, settings(instance)) / signal(instance, source) == pytest.approx(.95, rel=.012)


@pytest.mark.parametrize('reset', ['mode', 'gap', 'outside', 'disabled'])
def test_scene_trend_does_not_cross_a_mode_gap_or_inactive_envelope(reset):
    instance = twilight_controller()
    for time, adu in ((100, 50), (120, 70)):
        set_inflight(instance, (1, 0))
        instance.compare_highlights(prediction(adu, time, any_next=0), 1, 0)
    if reset == 'mode':
        instance.night_av[0] = 0
    if reset in ('outside', 'disabled'):
        if reset == 'disabled':
            instance.config['HIGHLIGHT_PROTECTION']['TWILIGHT_ENABLE'] = False
        set_inflight(instance, (1, 0))
        instance.compare_highlights(prediction(70, 140, weight=0), 1, 0)
        instance.config['HIGHLIGHT_PROTECTION']['TWILIGHT_ENABLE'] = True
    set_inflight(instance, (1, 0))
    instance.compare_highlights(prediction(70, 1000 if reset == 'gap' else 160), 1, 0)
    assert settings(instance)[0] == pytest.approx(.95, abs=.000002)


def test_repeated_captures_cannot_ratchet_a_static_moon_warning():
    instance = twilight_controller()
    set_inflight(instance, (1, 0))
    instance.compare_highlights(prediction(), 1, 0)
    first = settings(instance)
    for time in (100, 90, 100):
        instance.compare_highlights(prediction(time=time), 1, 0)
        assert settings(instance) == first


def test_covered_adu_target_still_accepts_a_new_tighter_prediction():
    instance = twilight_controller()
    # Pending .97 already covers this capture's ordinary ADU target, but not
    # the .95 headroom ceiling. The existing early ADU return must not hide it.
    set_inflight(instance, (.97, 0))
    instance.compare_highlights(prediction(81), 1, 0)
    assert settings(instance)[0] == pytest.approx(.95, abs=.000002)
    set_inflight(instance, (.8, 0))
    instance.compare_highlights(prediction(81, time=120), 1, 0)
    assert settings(instance)[0] == pytest.approx(.8, abs=.000002)


def test_unmodelled_legacy_gain_retains_one_policy_step_and_a_stronger_pending_cut():
    instance = controller('exposure_legacy_autogain')
    instance.config['HIGHLIGHT_PROTECTION']['TWILIGHT_ENABLE'] = True
    instance.post_init()
    source = (instance.auto_gain_exposure_cutoff_low, instance.auto_gain_step_list[3])
    set_inflight(instance, source)
    instance.compare_highlights(prediction(), *source)
    assert settings(instance) == (source[0], instance.auto_gain_step_list[2])
    pending = (source[0], instance.auto_gain_step_list[1])
    set_inflight(instance, pending)
    instance.compare_highlights(prediction(time=120), *source)
    assert settings(instance) == pending


@pytest.mark.parametrize('background', [.025, .27])
def test_fresh_static_moon_captures_settle_without_darkening_past_shadow_floor(background):
    instance = twilight_controller()
    field = np.full((96, 96, 3), background)
    field[40:48, 28:68] = .94
    mask = np.ones(field.shape[:2], np.uint8)
    capture = (1., 0)
    adus, commands = [], []
    for frame in range(40):
        set_inflight(instance, capture)
        raw = (np.minimum(field * capture[0], 1) * 65535).astype(np.uint16)
        sample = measure(raw, mask, 16)._replace(twilight_weight=1, twilight_time=100 + frame * 20)
        instance.compare_highlights(sample, *capture)
        capture = settings(instance)
        commands.append(capture)
        adus.append(sample.adu)
    assert min(adus[-10:]) >= 17.5 * .98
    assert max(command[0] for command in commands[-10:]) == pytest.approx(
        min(command[0] for command in commands[-10:]), abs=.000002)
    if background == .27:
        assert commands[-1][0] == pytest.approx(.95, abs=.000002)
    assert instance._twilight_prediction_previous[-1] is False


def test_predictive_protection_engages_compensation_without_lifting_above_target():
    instance = twilight_controller()
    set_inflight(instance, (1, 0))
    instance.compare_highlights(prediction(94), 1, 0)
    assert instance.highlight_transition.active
    assert instance.highlight_transition.render_target(94, 70, 2) <= 94
    assert instance.highlight_transition.lift == 0
    # A later protected cut can be compensated as soon as it falls below target.
    assert instance.highlight_transition.render_target(60, 70, 2) == pytest.approx(70)
    assert instance.highlight_transition.lift > 0


@pytest.mark.parametrize('growth,queued', [(1.06, 1), (1.08, 1), (1.12, 3)])
def test_known_central_cloud_radiance_uses_prediction_before_actual_backstop(growth, queued):
    instance = twilight_controller('gain')
    instance.config['EXPOSURE_PERIOD'] = 20
    initial = (10, 240)
    pending = [initial] * queued
    reference = signal(instance, initial)
    # A central blue plateau; no bright horizon statistic is supplied.
    field = np.full((96, 96, 3), .12)
    field[28:68, 28:68, 0] = .88
    mask = np.ones(field.shape[:2], np.uint8)
    commands, peaks, early = [], [], False
    for frame in range(70):
        capture = pending.pop(0)
        set_inflight(instance, pending[-1] if pending else capture)
        linear = field * growth ** frame * signal(instance, capture) / reference
        raw = (np.minimum(linear, 1) * 65535).astype(np.uint16)
        sample = measure(raw, mask, 16)._replace(twilight_weight=1, twilight_time=100 + frame * 20)
        instance.compare_highlights(sample, *capture)
        command = settings(instance)
        if sample.any == 0 and sample.any_next > 2.4 and signal(instance, command) < signal(instance, capture):
            early = True
        if commands:
            ratio = signal(instance, command) / signal(instance, commands[-1])
            assert .8 / 10 ** .005 <= ratio <= 1.1 * 10 ** .005
        commands.append(command)
        peaks.append(float(linear.max()))
        pending.append(command)
    assert early
    # Twelve-percent growth outruns the early 10% cut while older captures are
    # queued. The bounded actual-clipping backstop must prevent runaway; it
    # cannot promise zero overshoot from measurements three captures behind.
    assert max(peaks[-20:]) < (1.25 if queued == 3 else 1.05)
