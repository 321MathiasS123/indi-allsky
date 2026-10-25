"""Closed-loop capture/render regressions for compensated Moon-lit nights."""
import math
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from indi_allsky.highlight import HighlightMeasurement, HighlightTransition, compensate, measure
from indi_allsky.highlight_meter import apply_control_snapshot
from tests.core.test_highlight_exposure import controller


BASE_EXPOSURE = 29.171230
TARGET = 70.
MAX_BOOST = 2.
FULL_ADU = 65535 / 256
Y, X = np.ogrid[:100, :100]
MOON_CORE = (X - 75) ** 2 + (Y - 50) ** 2 <= 6 ** 2


def night_controller(maximum=30., minimum_gain=0., *, enabled=True):
    obj = controller('exposure_autogain_exp_prio_db_1_10', moon=True)
    obj.gain_quantum = 1.  # One camera unit is 0.1 dB on the ZWO camera.
    obj.config['CCD_EXPOSURE_MAX'] = maximum
    obj.config['HIGHLIGHT_PROTECTION'].update(
        NIGHT_MOON_EXCLUSION_ENABLE=enabled, MAX_BOOST=MAX_BOOST)
    utils = obj._expUtils
    utils.EXPOSURE_MAX = maximum
    for mode in ('NIGHT', 'MOONMODE'):
        setattr(utils, 'GAIN_MIN_' + mode, minimum_gain)
        setattr(utils, 'GAIN_MAX_' + mode, 300.)
    return obj


def source_frame(exposure, gain, *, base_adu=27.69, moon=True, outside=False,
                 weight=1., context=True, timestamp=1792954800., moon_adu=1024., moon_radius=6):
    """A fixed sky: linear photons, camera saturation and a local Moon mask."""
    core = MOON_CORE if moon_radius == 6 else (X - 75) ** 2 + (Y - 50) ** 2 <= moon_radius ** 2
    fraction = np.mean(core)
    background = (base_adu - fraction * FULL_ADU) / (1 - fraction)
    scene = np.full((100, 100), background, dtype=np.float64)
    if moon:
        scene[core] = moon_adu
    if outside:
        scene[5:25, 5:25] = 1024.
    signal = exposure / BASE_EXPOSURE * 10 ** (gain / 200)
    data = np.minimum(np.rint(scene * signal * 256), 65535).astype(np.uint16)
    mask = np.full(data.shape, 255, dtype=np.uint8)
    if moon and weight:
        cv2.circle(mask, (75, 50), int(math.ceil(6.5 * math.sqrt(weight))), 0, -1)
    clipped = bool(np.any((data >= .99 * 65535) & (mask == 0)))
    measurement = measure(data, mask, 16)._replace(
        adu=cv2.mean(data)[0] / 256, twilight_time=timestamp,
        night_moon_weight=weight if context and moon else 0.,
        night_moon_clipped=float(clipped and context),
    )
    return measurement, data


def acknowledge(obj, exposure, gain):
    # Match the real shared microsecond storage without synthetic pending
    # commands caused solely by truncating a binary float a second time.
    utils = obj._expUtils
    utils.EXPOSURE_CURRENT = utils.EXPOSURE_NEXT = math.nextafter(exposure, math.inf)
    utils.GAIN_CURRENT = utils.GAIN_NEXT = gain
    return utils.EXPOSURE_CURRENT, utils.GAIN_CURRENT


def render_capture(obj, renderer, measurement, data, startup):
    transition = obj.highlight_transition
    snapshot = {
        'transition': {
            'reset': False, 'trusted': transition.trusted, 'active': transition.active,
            'reason': transition.reason, 'seed': startup and transition.active,
            'target': TARGET,
        },
        'stable': obj.target_adu_found, 'current_adu_target': TARGET,
    }
    apply_control_snapshot(renderer, snapshot)
    transition = renderer.highlight_transition
    max_boost = obj.config['HIGHLIGHT_PROTECTION']['MAX_BOOST']
    target = transition.render_target(measurement.adu, TARGET, max_boost)
    gamma = transition.gamma(.87, .87)
    output = compensate(data, 16, measurement.adu, target, max_boost)
    background = (output[0, 0] / 65535) ** (1 / gamma) * 255
    return dict(render_target=target, lift=transition.lift, gamma=gamma,
                background=float(background), active=transition.active)


def replay(obj, *, exposure=BASE_EXPOSURE, gain=0., frames=100,
           base_adu=27.69, context=True, moon=True, outside=False, weights=None,
           renderer=None, timestamp=1792954800., moon_adu=1024., moon_radius=6):
    renderer = renderer or SimpleNamespace(highlight_transition=HighlightTransition())
    history = []
    for index in range(frames):
        exposure, gain = acknowledge(obj, exposure, gain)
        weight = weights[index] if weights is not None else 1.
        measurement, data = source_frame(
            exposure, gain, base_adu=base_adu, context=context, moon=moon,
            outside=outside, weight=weight, timestamp=timestamp + 31 * index,
            moon_adu=moon_adu, moon_radius=moon_radius)
        startup = obj.highlight_transition._startup
        obj.compare_highlights(measurement, exposure, gain)
        display = render_capture(obj, renderer, measurement, data, startup)
        history.append(dict(exposure=exposure, gain=gain, adu=measurement.adu,
                            measurement=measurement, **display))
        exposure, gain = obj._expUtils.EXPOSURE_NEXT, obj._expUtils.GAIN_NEXT
    return history, renderer


def test_previous_mask_only_policy_reproduces_the_gain_regression():
    history, _ = replay(night_controller(), context=False)
    assert history[0]['gain'] == 0
    assert history[0]['lift'] == 0
    assert history[-1]['exposure'] == 30
    assert 65 <= history[-1]['gain'] <= 75
    assert 60 <= history[-1]['adu'] <= 80


def test_moon_night_reaches_maximum_without_gain_or_startup_darkening():
    obj = night_controller()
    history, _ = replay(obj)
    first, last = history[0], history[-1]
    assert first['lift'] > 1
    assert first['render_target'] == pytest.approx(TARGET)
    assert last['exposure'] == 30
    assert all(row['gain'] == 0 for row in history)
    assert all(row['active'] for row in history)
    assert all(row['render_target'] == pytest.approx(TARGET) for row in history)
    assert min(row['background'] for row in history) >= first['background'] * .99
    assert max(row['background'] for row in history) <= first['background'] * 1.02
    # The meter never receives render-only lift from this replay.
    assert obj.highlight_transition.lift == 0


def test_already_increased_gain_recovers_to_minimum_with_brightness_compensation():
    history, _ = replay(night_controller(), exposure=30., gain=69.)
    assert history[-1]['exposure'] == 30
    assert history[-1]['gain'] == 0
    assert all(row['exposure'] == 30 for row in history)
    assert all(right['gain'] <= left['gain'] for left, right in zip(history, history[1:]))
    assert all(row['render_target'] == pytest.approx(TARGET) for row in history)
    # Every accepted protective cut is compensated before normal gamma.
    changes = [abs(math.log2(b['background'] / a['background']))
               for a, b in zip(history, history[1:])]
    assert max(changes) < .05


@pytest.mark.parametrize('maximum', [12., 30., 45.])
def test_long_running_policy_uses_configured_maximum_and_retains_real_gain_range(maximum):
    obj = night_controller(maximum)
    # Keep the sky bright enough at each independently configured maximum for
    # the original two-stop lift budget to permit minimum gain.
    moon_adu = np.mean(MOON_CORE) * FULL_ADU
    base_adu = (27.69 - moon_adu) * BASE_EXPOSURE / maximum + moon_adu
    history, _ = replay(obj, exposure=maximum * .6, base_adu=base_adu)
    assert history[-1]['exposure'] == maximum
    assert history[-1]['gain'] == 0
    assert obj._expUtils.GAIN_MAX_NIGHT == 300
    assert obj._expUtils.GAIN_MIN_NIGHT == 0
    assert obj._expUtils.EXPOSURE_MIN_NIGHT == .001
    assert obj._expUtils.EXPOSURE_MAX == maximum


@pytest.mark.parametrize('minimum', [20., 40.])
def test_configured_nonzero_gain_minimum_is_honoured(minimum):
    history, _ = replay(night_controller(minimum_gain=minimum), exposure=30., gain=100.)
    assert history[-1]['exposure'] == 30
    assert history[-1]['gain'] == minimum
    assert all(row['gain'] >= minimum for row in history)


def test_submaximum_exposure_and_high_gain_are_reallocated_without_stranding():
    history, _ = replay(night_controller(), exposure=12., gain=100., frames=120)
    assert history[-1]['exposure'] == 30
    assert history[-1]['gain'] == 0


def test_genuinely_dark_sky_can_still_raise_gain_to_meet_original_shadow_floor():
    obj = night_controller()
    history, _ = replay(obj, exposure=30., base_adu=12.)
    legacy, _ = replay(night_controller(), exposure=30., base_adu=12., context=False)
    last = history[-1]
    assert last['exposure'] == 30
    assert 0 < last['gain'] < legacy[-1]['gain']
    assert last['adu'] >= TARGET / 2 ** MAX_BOOST * .98
    assert last['adu'] < 60
    assert last['lift'] <= MAX_BOOST
    assert len({(row['exposure'], row['gain']) for row in history[-20:]}) == 1


def test_clipping_elsewhere_still_reduces_exposure_below_the_configured_maximum():
    history, _ = replay(night_controller(), exposure=30., outside=True)
    assert min(row['exposure'] for row in history) < 30
    assert history[-1]['exposure'] < 30
    assert all(row['gain'] == 0 for row in history)


@pytest.mark.parametrize('blocked', ['disabled', 'disabled_parent', 'no_moon', 'day', 'twilight'])
def test_unrelated_paths_keep_the_previous_controller_commands(blocked):
    actual, reference = night_controller(), night_controller(enabled=False)
    exposure, gain = 24., 0.
    for obj in (actual, reference):
        if blocked == 'day':
            obj.night_av[:] = [0, 0]
        if blocked == 'disabled':
            obj.config['HIGHLIGHT_PROTECTION']['NIGHT_MOON_EXCLUSION_ENABLE'] = False
        if blocked == 'disabled_parent':
            obj.config['HIGHLIGHT_PROTECTION']['ENABLE'] = False
        if blocked == 'twilight':
            obj.config['HIGHLIGHT_PROTECTION']['TWILIGHT_ENABLE'] = True
    for index in range(85):
        measurement, _ = source_frame(
            exposure, gain, moon=blocked not in ('no_moon', 'day'),
            context=blocked in ('disabled', 'disabled_parent', 'twilight'),
            timestamp=1792954800. + 31 * index)
        if blocked == 'twilight':
            measurement = measurement._replace(twilight_weight=.5)
        for obj in (actual, reference):
            acknowledge(obj, exposure, gain)
            obj.compare_highlights(measurement, exposure, gain)
        assert actual._expUtils.EXPOSURE_NEXT == reference._expUtils.EXPOSURE_NEXT
        assert actual._expUtils.GAIN_NEXT == reference._expUtils.GAIN_NEXT
        exposure, gain = reference._expUtils.EXPOSURE_NEXT, reference._expUtils.GAIN_NEXT


def test_old_or_duplicate_capture_cannot_compound_or_undo_newer_gain_recovery():
    obj = night_controller()
    exposure, gain = acknowledge(obj, 30., 69.)
    old, _ = source_frame(exposure, gain, timestamp=1792954800.)
    obj.compare_highlights(old, exposure, gain)
    requested = obj._expUtils.EXPOSURE_NEXT, obj._expUtils.GAIN_NEXT
    obj.compare_highlights(old, exposure, gain)
    assert (obj._expUtils.EXPOSURE_NEXT, obj._expUtils.GAIN_NEXT) == requested
    new_exposure, new_gain = acknowledge(obj, *requested)
    fresh, _ = source_frame(new_exposure, new_gain, timestamp=1792954831.)
    obj.compare_highlights(fresh, new_exposure, new_gain)
    newest = obj._expUtils.EXPOSURE_NEXT, obj._expUtils.GAIN_NEXT
    assert newest[1] <= requested[1]
    obj.compare_highlights(old, exposure, gain)
    assert (obj._expUtils.EXPOSURE_NEXT, obj._expUtils.GAIN_NEXT) == newest


def test_losing_valid_moon_context_releases_the_preference():
    obj = night_controller()
    history, renderer = replay(obj)
    assert history[-1]['gain'] == 0
    ordinary, _ = replay(obj, exposure=30., gain=0., moon=False,
                         renderer=renderer, timestamp=1792960000.)
    assert ordinary[-1]['gain'] > 0
    assert ordinary[-1]['adu'] >= 60
    assert not ordinary[-1]['active']


def test_previously_clipped_moon_can_fade_without_restarting_gain_pumping():
    obj = night_controller()
    history, renderer = replay(obj)
    assert history[-1]['gain'] == 0
    faded, _ = replay(obj, exposure=30., gain=0., moon_adu=64.,
                      renderer=renderer, timestamp=1792960000.)
    assert all(row['measurement'].night_moon_clipped == 0 for row in faded)
    assert all(row['gain'] == 0 for row in faded)
    assert all(row['active'] for row in faded)


def test_unclipped_moon_prediction_does_not_start_a_new_compensation_policy():
    actual, reference = night_controller(), night_controller(enabled=False)
    measurement, _ = source_frame(BASE_EXPOSURE, 0.)
    measurement = measurement._replace(night_moon_clipped=0.)
    for obj in (actual, reference):
        acknowledge(obj, BASE_EXPOSURE, 0.)
        obj.compare_highlights(measurement, BASE_EXPOSURE, 0.)
    assert actual._expUtils.EXPOSURE_NEXT == reference._expUtils.EXPOSURE_NEXT
    assert actual._expUtils.GAIN_NEXT == reference._expUtils.GAIN_NEXT
    assert actual.highlight_transition.active == reference.highlight_transition.active


def test_night_phase_fade_keeps_rendered_background_smooth_in_both_directions():
    weights = np.concatenate((np.linspace(0, 1, 50), np.ones(50), np.linspace(1, 0, 50)))
    history, _ = replay(night_controller(), weights=weights, frames=len(weights))
    assert history[99]['exposure'] == 30
    assert history[99]['gain'] == 0
    assert all(math.isfinite(row['background']) and row['background'] > 0 for row in history)
    changes = [abs(math.log2(b['background'] / a['background']))
               for a, b in zip(history, history[1:])]
    assert max(changes) < .06


def test_zero_shadow_lift_budget_keeps_the_original_raw_brightness_requirement():
    obj = night_controller()
    obj.config['HIGHLIGHT_PROTECTION']['MAX_BOOST'] = 0.
    history, _ = replay(obj, frames=120)
    assert history[-1]['exposure'] == 30
    assert history[-1]['gain'] > 0
    assert history[-1]['adu'] >= TARGET * .98
    assert all(row['lift'] == 0 for row in history)


def test_fresh_output_constraint_can_still_cut_below_maximum_exposure():
    obj = night_controller()
    obj.config['HIGHLIGHT_PROTECTION']['OUTPUT_ENABLE'] = True
    exposure, gain = acknowledge(obj, 30., 0.)
    measurement, _ = source_frame(exposure, gain)
    obj.highlight_output.observe(HighlightMeasurement(5., 5., 0.), exposure, gain,
                                 tuple(obj.night_av), obj.config['HIGHLIGHT_PROTECTION'])

    obj.compare_highlights(measurement, exposure, gain)

    assert obj.highlight_output.active
    assert obj._expUtils.EXPOSURE_NEXT < exposure
    assert obj._expUtils.GAIN_NEXT == 0


def test_partial_solar_weights_settle_without_gain_sawtooth_or_rendered_pumping():
    gains = []
    for weight in (.01, .1, .5, .9):
        # A compact clipped core fits even the early faded Moon halo; residual
        # raw clipping must not hide the independent phase-weighted policy.
        history, _ = replay(night_controller(), frames=180, moon_radius=1,
                             weights=np.full(180, weight))
        tail = {(row['exposure'], row['gain']) for row in history[-40:]}
        assert len(tail) == 1, (weight, tail)
        gains.append(history[-1]['gain'])
        changes = [abs(math.log2(b['background'] / a['background']))
                   for a, b in zip(history, history[1:])]
        assert max(changes) < .06, (weight, max(changes))
    assert all(left > right for left, right in zip(gains, gains[1:])), gains
