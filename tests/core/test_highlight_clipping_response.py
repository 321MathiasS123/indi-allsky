"""Prompt clipped-patch response without compounding delayed captures."""
import pytest

from indi_allsky.highlight import HighlightMeasurement
from test_highlight_exposure import controller


def set_inflight(instance, exposure, gain=0):
    instance._expUtils.EXPOSURE_CURRENT = exposure
    instance._expUtils.GAIN_CURRENT = gain
    instance._expUtils.EXPOSURE_NEXT = exposure
    instance._expUtils.GAIN_NEXT = gain


def requested(instance):
    return instance._expUtils.EXPOSURE_NEXT, instance._expUtils.GAIN_NEXT


def signal(instance, settings):
    exposure, gain = settings
    return exposure * (10 ** (instance.gain2dB(gain) / 20)
                       if hasattr(instance, 'gain2dB') else 1)


@pytest.mark.parametrize('limits', [
    dict(FULL_TARGET=.6, FULL_DEV=.15, ANY_TARGET=1.3, ANY_DEV=.25),
    dict(FULL_TARGET=2.0, FULL_DEV=.3, ANY_TARGET=.4, ANY_DEV=.15),
])
@pytest.mark.parametrize('channel', ['full', 'any'])
@pytest.mark.parametrize('severity', [1.8, 2, 3, 10])
def test_large_clipping_spike_responds_immediately_using_configured_limits(limits, channel, severity):
    instance = controller('exposure_basic', night=False)
    instance.config['HIGHLIGHT_PROTECTION'].update(limits)
    limit = limits[channel.upper() + '_TARGET'] + limits[channel.upper() + '_DEV']
    patches = dict(full=0, any=0, adu=80)
    patches[channel] = limit * severity
    set_inflight(instance, 20)
    instance.compare_highlights(HighlightMeasurement(**patches), 20, 0)
    # A 40% reduction is the maximum justified cut. The first
    # observation must not spend additional frames ramping up that correction.
    assert requested(instance) == pytest.approx((12, 0), abs=1e-6)


@pytest.mark.parametrize('channel,limit', [('full', 1), ('any', 2.4)])
@pytest.mark.parametrize('boundary', [1.4, 1.8])
def test_severity_changes_do_not_introduce_a_command_discontinuity(channel, limit, boundary):
    commands = []
    for severity in (boundary - .00001, boundary, boundary + .00001):
        instance = controller('exposure_basic', night=False)
        patches = dict(full=0, any=0, adu=80)
        patches[channel] = severity * limit
        set_inflight(instance, 20)
        instance.compare_highlights(HighlightMeasurement(**patches), 20, 0)
        commands.append(requested(instance)[0])
    assert commands[0] >= commands[1] >= commands[2]
    assert max(commands) - min(commands) < .001


@pytest.mark.parametrize('severity,expected', [(1.02, .995), (1.2, .95), (1.4, .9)])
def test_small_clipping_excess_keeps_the_existing_gentle_first_step(severity, expected):
    instance = controller('exposure_basic', night=False)
    set_inflight(instance, 1)
    instance.compare_highlights(HighlightMeasurement(severity, 0, 80), 1, 0)
    assert requested(instance) == pytest.approx((expected, 0), abs=1e-6)


@pytest.mark.parametrize('pending_ratio', [.98, 1.1])
def test_severe_clipping_overrides_delayed_dawn_and_dusk_commands(pending_ratio):
    instance = controller('exposure_basic', night=False)
    # Establish an equally severe preceding observation so the ordinary
    # delayed-control slew would otherwise apply to a continuing dawn ramp.
    set_inflight(instance, 24)
    instance.compare_highlights(HighlightMeasurement(3, 8, 80), 24, 0)
    set_inflight(instance, 20 * pending_ratio)
    instance.compare_highlights(HighlightMeasurement(3, 8, 80), 20, 0)
    assert requested(instance) == pytest.approx((12, 0), abs=1e-6)


@pytest.mark.parametrize('pending_ratio', [.98, 1.1])
def test_repeated_old_clipped_capture_does_not_compound_its_absolute_cut(pending_ratio):
    instance = controller('exposure_basic', night=False)
    set_inflight(instance, 20 * pending_ratio)
    for _ in range(8):
        instance.compare_highlights(HighlightMeasurement(3, 8, 80), 20, 0)
        assert requested(instance) == pytest.approx((12, 0), abs=1e-6)


@pytest.mark.parametrize('actuator', ['exposure', 'db_gain', 'iso'])
@pytest.mark.parametrize('stronger_pending', [False, True])
def test_urgent_response_uses_achievable_signal_and_retains_stronger_pending_cut(actuator, stronger_pending):
    if actuator == 'exposure':
        instance = controller('exposure_basic', night=False)
        source = (20, 0)
        pending = (6 if stronger_pending else 19, 0)
    elif actuator == 'db_gain':
        instance = controller('exposure_autogain_exp_prio_db_1_10')
        instance._expUtils.GAIN_MAX_NIGHT = 300
        instance.gain_quantum = 1
        source = (30, 200)
        pending = (30, 90 if stronger_pending else 195)
    else:
        instance = controller('exposure_autogain_exp_prio_iso')
        instance.gain_values = [100, 200, 400, 800]
        source = (20, 400)
        pending = (20, 100) if stronger_pending else (30, 200)
    set_inflight(instance, *pending)
    instance.compare_highlights(HighlightMeasurement(3, 8, 80), *source)
    if stronger_pending:
        assert requested(instance) == pending
    else:
        assert signal(instance, requested(instance)) / signal(instance, source) == pytest.approx(.6, rel=.006)
    assert instance.exposure_min <= requested(instance)[0] <= instance.exposure_max
    assert instance.gain_min <= requested(instance)[1] <= instance.gain_max


@pytest.mark.parametrize('boost', [0, 1, 2, 3])
@pytest.mark.parametrize('above_floor', [1, 1.01, 1.05, 1.5, 2.0])
def test_urgent_clipping_cannot_spend_more_than_configured_shadow_headroom(boost, above_floor):
    instance = controller('exposure_basic', night=False)
    instance.config['HIGHLIGHT_PROTECTION']['MAX_BOOST'] = boost
    floor = 80 / 2 ** boost
    adu = floor * above_floor
    set_inflight(instance, 20)
    instance.compare_highlights(HighlightMeasurement(10, 20, adu), 20, 0)
    ratio = requested(instance)[0] / 20
    assert ratio >= max(.6, floor / adu) - 1e-6
    if above_floor <= 1.01:
        assert ratio == 1
    else:
        assert ratio < 1


def test_old_dark_dusk_capture_cannot_undo_newer_urgent_cut():
    instance = controller('exposure_basic', night=False)
    set_inflight(instance, 20)
    instance.compare_highlights(HighlightMeasurement(3, 8, 80), 20, 0)
    reduced = requested(instance)
    instance.compare_highlights(HighlightMeasurement(0, 0, 30, 0, 0, 0, 0), 20, 0)
    assert requested(instance) == reduced


def test_urgent_gain_reduction_spends_remaining_correction_in_exposure_at_gain_floor():
    instance = controller('exposure_autogain_exp_prio_db_1_10')
    instance.gain_quantum = 1
    source = (30, 10)
    set_inflight(instance, *source)
    instance.compare_highlights(HighlightMeasurement(3, 8, 70), *source)
    exposure, gain = requested(instance)
    assert gain == 0
    assert exposure == pytest.approx(18 * 10 ** (10 / 200), abs=1e-6)
    assert signal(instance, requested(instance)) / signal(instance, source) == pytest.approx(.6, abs=1e-7)
