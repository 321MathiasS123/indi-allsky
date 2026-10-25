"""Twilight feedback accepts known delayed settings and rejects unsafe input."""
import pytest

from indi_allsky.highlight_output_control import output_signature
from indi_allsky.highlight_meter import OutputFeedbackGate
from test_highlight_twilight_output_control import configured, raw, rendered, MODE


def job(**changes):
    return dict({'exp_time': 120, 'exp_elapsed': 20, 'exposure': 10, 'gain': 0,
                 'binning': 1, 'camera_id': 1, 'capture_mode': MODE, 'capture_period': 20}, **changes)


def feedback(control, **changes):
    values = dict(job(exp_time=100, exposure=9), measurement=tuple(rendered()), trusted=True,
                  twilight_output=True, render_signature=output_signature(control.config))
    values.update(changes)
    return values


def update(control=None, capture=None, returned=None, current=None):
    control = control or configured()
    return OutputFeedbackGate(output_signature(control.config)).update(
        control, capture or job(), returned or feedback(control), measurement=current or raw())


def test_known_exposure_change_uses_recent_output_instead_of_rejecting_its_settings():
    assert update()


def test_known_gain_change_can_use_output_from_the_previous_camera_settings():
    control = configured('exposure_autogain_exp_prio_db_1_10')
    assert update(control, job(gain=90), feedback(control, gain=100))


def test_legacy_gain_change_without_declared_signal_units_is_rejected():
    control = configured()
    assert not update(control, job(gain=1), feedback(control, gain=0))
    assert control.twilight_output.sample is None


@pytest.mark.parametrize('age,accepted', [(0, True), (20, True), (40, True), (40.001, False), (-1, False)])
def test_age_is_bounded_by_two_capture_cadences(age, accepted):
    control = configured()
    returned = feedback(control, exp_time=120 - age)
    assert update(control, returned=returned) is accepted


@pytest.mark.parametrize('change', [
    {'trusted': False}, {'twilight_output': False}, {'render_signature': 'other-config'},
    {'measurement': None}, {'camera_id': 2}, {'binning': 2}, {'capture_mode': (0, 0)},
    {'exposure': 0}, {'exposure': -1}, {'exposure': float('inf')},
    {'gain': float('nan')}, {'gain': float('inf')},
])
def test_stale_or_untrusted_returns_drop_the_twilight_constraint(change):
    control = configured()
    returned = feedback(control, **change)
    assert not update(control, returned=returned)
    assert control.twilight_output.sample is None


@pytest.mark.parametrize('field,value', [
    ('adu', 0), ('adu', float('nan')), ('adu', float('inf')),
    ('twilight_peak', 0), ('twilight_peak', 1.01), ('twilight_peak', float('nan')),
    ('twilight_weight', 0), ('twilight_weight', 1.01), ('twilight_weight', float('inf')),
    ('any', -1), ('any', 101), ('twilight_time', 0), ('twilight_time', 121),
])
def test_invalid_rendered_measurements_cannot_reach_control(field, value):
    control = configured()
    item = rendered()._replace(**{field: value})
    assert not update(control, returned=feedback(control, measurement=tuple(item)))


@pytest.mark.parametrize('field,value', [
    ('adu', float('nan')), ('adu', float('inf')), ('adu', 0),
    ('twilight_weight', float('inf')), ('twilight_weight', 1.01),
    ('twilight_time', float('nan')), ('full_next', float('inf')),
])
def test_invalid_current_raw_data_cannot_authorize_output_cuts(field, value):
    assert not update(current=raw()._replace(**{field: value}))


@pytest.mark.parametrize('exposure', [0, -1, float('inf'), float('nan')])
def test_invalid_current_capture_metadata_cannot_authorize_output_cuts(exposure):
    assert not update(capture=job(exposure=exposure))


def test_duplicate_and_regressing_returns_cannot_replace_a_newer_accepted_source():
    control = configured()
    gate = OutputFeedbackGate(output_signature(control.config))
    latest = feedback(control)
    assert gate.update(control, job(), latest, measurement=raw())
    before = control.twilight_output.sample
    for exp_time in (100, 90):
        old = feedback(control, exp_time=exp_time, measurement=tuple(rendered(240, 0, time=90)))
        assert gate.update(control, job(), old, measurement=raw())
        assert control.twilight_output.sample == before


def test_thirty_second_n_plus_two_capture_keeps_usable_output_without_compounding():
    control = configured()
    gate = OutputFeedbackGate(output_signature(control.config))
    current = raw(time=135)
    capture = job(exp_time=150, exposure=30, exp_elapsed=30.9)
    previous = feedback(control, exp_time=119.1, exposure=29.4,
                        measurement=tuple(rendered(adu=68.6, time=104.4)))
    assert gate.update(control, capture, previous, allowance=1.4, measurement=current)
    sample = control.twilight_output.sample
    assert gate.update(control, capture, None, allowance=1.4, measurement=current)
    assert control.twilight_output.sample == sample
    capture['exp_time'] = 182
    assert not gate.update(control, capture, None, allowance=1.4, measurement=raw(time=167))
    assert control.twilight_output.sample is None


@pytest.mark.parametrize('key,value', [('GAMMA_CORRECTION', 1.1), ('WBG_FACTOR', 1.2), ('SHARPEN_AMOUNT', .4)])
def test_saved_render_settings_change_the_feedback_signature(key, value):
    control = configured()
    signature = output_signature(control.config)
    control.config[key] = value
    assert output_signature(control.config) != signature


def test_gate_uses_saved_signature_while_capture_settings_are_interpolated():
    control = configured()
    signature = output_signature(control.config)
    gate = OutputFeedbackGate(signature)
    returned = feedback(control)
    # Both workers cache the saved signature before their per-frame add() blend.
    # These effective values change without changing either saved profile.
    control.config.update(GAMMA_CORRECTION=1.17, WBR_FACTOR=.82)
    assert output_signature(control.config) != signature
    assert gate.update(control, job(), returned, measurement=raw())
