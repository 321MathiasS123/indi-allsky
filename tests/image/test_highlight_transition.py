"""Rendering engagement/release, separate from the exposure correction history."""
import math

import numpy as np
import pytest

from indi_allsky.highlight import HighlightMeasurement, HighlightTransition, compensate


SETTINGS = {'FULL_TARGET': .6, 'FULL_DEV': .15, 'ANY_TARGET': 1.3, 'ANY_DEV': .25}


def observe(state, adu=61, full=0, any_clip=0, full_next=0, any_next=0,
            pending=False, ceiling=False, predicted_block=False, settings=SETTINGS):
    state.observe(HighlightMeasurement(full, any_clip, adu, full_next, any_next),
                  70, 10, settings, pending, ceiling, predicted_block)


def established():
    state = HighlightTransition()
    observe(state, full=1)
    for _ in range(100):
        state.render_target(40, 70, 2)
        state.gamma(1.565, 1.85)
    assert state.reference == 70 and state.gamma_mix == 1
    return state


@pytest.mark.parametrize('adu', [0, 10, 40, 61, 70, 90])
@pytest.mark.parametrize('ceiling', [False, True])
def test_unclipped_startup_never_invents_lift_or_gamma(adu, ceiling):
    state = HighlightTransition()
    for _ in range(30):
        observe(state, adu=adu, ceiling=ceiling)
        assert state.render_target(adu, 70, 2) == max(adu, .1)
        assert state.gamma(1.565, 1.85) == 1.565
        assert state.phase == 'normal' and state.lift == 0


@pytest.mark.parametrize('untrusted_frames', [0, 3])
@pytest.mark.parametrize('max_boost', [0, .5, 2])
@pytest.mark.parametrize('normal,protected', [(1.565, 1.85), (.87, .87)])
def test_restart_uses_first_trusted_frame_without_rebuilding_compensation(untrusted_frames, max_boost, normal, protected):
    state = HighlightTransition()
    for _ in range(untrusted_frames):
        assert state.render_target(25, 80, max_boost) == 25
        assert state.gamma(normal, protected) == normal
    # Actual first capture after the 2026-10-05 09:20 restart: exposure and
    # calibrated brightness held steady, but the lost lift darkened the JPEG.
    measurement = HighlightMeasurement(.393, 1.201, 35.71, .453, 1.590)
    state.observe(measurement, 80, 10, SETTINGS, False, False, True)
    assert state.render_target(35.71, 80, max_boost) == 80
    assert state.lift == pytest.approx(min(max_boost, math.log2(80 / 35.71)))
    assert state.gamma(normal, protected) == protected


@pytest.mark.parametrize('full,any_clip,expected', [(0, 0, False), (.44, 1.05, False),
                                                  (.451, 0, True), (0, 1.051, True)])
def test_either_lower_limit_engages_without_aiming_to_create_clipping(full, any_clip, expected):
    state = HighlightTransition()
    observe(state, full=full, any_clip=any_clip)
    assert state.active is expected


@pytest.mark.parametrize('ceiling,expected', [(False, True), (True, False)])
def test_prediction_arms_only_for_an_achievable_needed_increase(ceiling, expected):
    state = HighlightTransition()
    observe(state, adu=40, any_next=4, predicted_block=True, ceiling=ceiling)
    assert state.active is expected


@pytest.mark.parametrize('kwargs,reason', [
    ({'adu': 40}, 'ordinary ADU recovery'),
    ({'pending': True}, 'pending exposure/gain'),
    ({'full': .42}, 'highlight headroom'),
    ({'any_clip': .99}, 'highlight headroom'),
    ({'any_next': 3.867}, 'highlight headroom'),  # observed sunrise cloud plateau
    ({'full_next': None, 'any_next': None}, 'highlight headroom'),
])
def test_clear_frame_does_not_prematurely_release_protection(kwargs, reason):
    state = established()
    observe(state, **kwargs)
    assert state.active and reason in state.reason


@pytest.mark.parametrize('ceiling', [False, True])
def test_safe_release_reaches_exact_ordinary_endpoint_with_bounded_steps(ceiling):
    state = established()
    state.render_target(61, 70, 2)
    observe(state, adu=30 if ceiling else 61, any_next=4 if ceiling else 0, ceiling=ceiling)
    assert not state.active
    previous_lift = state.lift
    previous_power = 1 / 1.85
    for index in range(100):
        # Scene variation during release must not be chased by the old reference.
        state.render_target(61 + index % 7, 70, 2)
        power = 1 / state.gamma(1.565, 1.85)
        assert 0 <= previous_lift - state.lift <= .040000001
        assert 0 <= power - previous_power <= .010000001
        previous_lift, previous_power = state.lift, power
    assert state.phase == 'normal' and state.lift == state.gamma_mix == 0
    assert state.gamma(1.565, 1.85) == 1.565


def test_entry_compensates_exposure_cuts_immediately_and_reentry_is_continuous():
    state = HighlightTransition()
    observe(state)  # Already running normally when the highlight appears.
    previous_reference = 61
    for adu in (61, 55, 49, 43, 38):
        observe(state, adu=adu, full=2)
        reference = state.render_target(adu, 70, 2)
        assert 0 <= math.log2(reference / previous_reference) <= .040000001
        assert adu * 2 ** state.lift == pytest.approx(reference)
        previous_reference = reference
    observe(state)
    state.render_target(61, 70, 2)
    assert 61 * 2 ** state.lift <= previous_reference  # no recovery-frame overshoot
    # A natural brightness increase partway through release may exceed target.
    state.render_target(100, 70, 2)
    partial_lift = state.lift
    assert not state.active and partial_lift > 0
    observe(state, full=2)
    reference = state.render_target(100, 70, 2)
    assert abs(math.log2(reference / (100 * 2 ** partial_lift))) <= .040000001


@pytest.mark.parametrize('active', [False, True])
def test_untrusted_frame_holds_decision_reference_and_gamma(active):
    state = established()
    if not active:
        observe(state)
    state.trusted = False
    before = state.active, state.reference, state.gamma_mix, state.lift
    for _ in range(30):
        state.render_target(25, 70, 2)  # repaired frame can use its own ADU
        assert state.gamma(1.565, 1.85) == 1.85
        assert (state.active, state.reference, state.gamma_mix) == before[:3]
    assert state.lift == (math.log2(70 / 25) if active else before[3])


@pytest.mark.parametrize('bits', [8, 12, 16])
@pytest.mark.parametrize('max_boost', [0, .5, 2])
def test_fully_protected_curve_is_pixel_identical(bits, max_boost):
    state = established()
    data = np.linspace(0, (1 << bits) - 1, 1200).reshape(20, 20, 3).astype(np.uint8 if bits == 8 else np.uint16)
    target = state.render_target(20, 70, max_boost)
    np.testing.assert_array_equal(compensate(data, bits, 20, target, max_boost),
                                  compensate(data, bits, 20, 70, max_boost))
    assert state.lift <= max_boost


def test_zero_limits_and_master_reset_have_exact_neutral_state():
    state = established()
    observe(state, settings={'FULL_TARGET': 0, 'FULL_DEV': 0, 'ANY_TARGET': 0, 'ANY_DEV': 0})
    assert not state.active
    state.reset()
    assert state.phase == 'normal' and not state.trusted
    assert state.render_target(30, 70, 2) == 30
    assert state.gamma(1.565, 1.85) == 1.565
    # Turning the feature back on must still ease in, not act like a restart.
    observe(state, adu=30, full=2)
    assert 0 < math.log2(state.render_target(30, 70, 2) / 30) <= .040000001
    assert 1.565 < state.gamma(1.565, 1.85) < 1.85
