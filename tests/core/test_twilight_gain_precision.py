"""Physical transition gains are valid with this feature installed alone."""
import pytest

from indi_allsky.twilight import transition_gain
from test_twilight_transition import controller


@pytest.mark.parametrize('info,value,expected', [
    ({'quantum': 1, 'step': 60}, 32.817, 33),
    ({'quantum': 1}, 32.5, 33),
    ({'quantum': 1}, 0, 0),
    ({'step': 60}, 32.817, 32.817),
    ({'quantum': 0}, 1.875, 1.875),
    ({'values': [100, 200, 400, 800]}, 350, 400),
    ({'values': [400, 100, 200]}, 300, 200),
    ({}, -1, -1),
])
def test_transition_respects_camera_precision_not_gui_step(info, value, expected):
    assert transition_gain(value, info) == expected


@pytest.mark.parametrize('info,minimum,maximum,pending,expected', [
    ({'quantum': 1}, 0, 100, 40.4, 40),
    ({'quantum': 1}, 0, 300, 254.9, 255),
    ({'quantum': 1}, .2, 99.8, 99.8, 99),
    ({'quantum': 1}, .2, 99.8, .2, 1),
    ({'quantum': .5}, 0, 100, 40.25, 40.5),
    ({'values': [100, 200, 400, 800]}, 150, 450, 350, 400),
    ({'values': [100, 200, 400, 800]}, 250, 750, 740, 400),
    ({'values': [100, 200, 400, 800]}, 250, 350, 300, 200),
    ({'quantum': 1}, .2, .4, .3, 0),
    ({'step': 60}, 0, 100, 32.817, 32.817),
    ({}, 0, 100, 40.4, 40.4),
])
def test_moving_limits_publish_camera_supported_autogain_without_other_features(info, minimum, maximum, pending, expected):
    obj, _ = controller('exposure_autogain_exp_prio_db_1_10')
    obj.twilight_gain_info = info
    obj._expUtils.GAIN_MIN_NIGHT, obj._expUtils.GAIN_MAX_NIGHT = minimum, maximum
    obj._expUtils.GAIN_NEXT = pending
    obj._expUtils.GAIN_DELTA = .25
    if hasattr(obj, 'effective_gain'):
        obj.gain_quantum = info.get('quantum', 0.0)
        obj.gain_values = info.get('values', [])
    obj.apply_transition_limits()
    assert obj._expUtils.GAIN_NEXT == pytest.approx(expected, abs=.001)
    assert obj._expUtils.GAIN_DELTA == pytest.approx(.25 + expected - pending, abs=.001)
    first_gain, first_delta = obj._expUtils.GAIN_NEXT, obj._expUtils.GAIN_DELTA
    obj.apply_transition_limits()
    assert obj._expUtils.GAIN_NEXT == first_gain
    assert obj._expUtils.GAIN_DELTA == first_delta


def test_moving_limits_preserve_pending_autogain_without_camera_precision():
    obj, _ = controller('exposure_autogain_exp_prio_db_1_10')
    if hasattr(obj, 'twilight_gain_info'):
        del obj.twilight_gain_info
    obj._expUtils.GAIN_NEXT, obj._expUtils.GAIN_DELTA = 40.4, .25
    obj.apply_transition_limits()
    assert obj._expUtils.GAIN_NEXT == 40.4
    assert obj._expUtils.GAIN_DELTA == .25


def test_disabled_transition_does_not_quantize_pending_autogain():
    obj, _ = controller('exposure_autogain_exp_prio_db_1_10')
    obj.config['TWILIGHT_TRANSITION']['ENABLE'] = False
    obj.twilight_gain_info = {'quantum': 1}
    obj._expUtils.GAIN_NEXT, obj._expUtils.GAIN_DELTA = 40.4, .25
    obj.apply_transition_limits()
    assert obj._expUtils.GAIN_NEXT == 40.4
    assert obj._expUtils.GAIN_DELTA == .25


@pytest.mark.parametrize('info', [{}, {'quantum': 1}, {'values': [0, 20, 40, 60, 80, 100]}])
@pytest.mark.parametrize('moon', [False, True])
@pytest.mark.parametrize('reverse', [False, True])
def test_dawn_dusk_and_moonmode_publish_reachable_fixed_gain_limits(info, moon, reverse):
    obj, transition = controller('exposure_basic')
    obj.twilight_gain_info = info
    if hasattr(obj, 'effective_gain_limits'):
        obj.gain_quantum = info.get('quantum', 0.0)
        obj.gain_values = info.get('values', [])
    obj.night_av[1] = moon
    obj._expUtils.GAIN_MIN_MOONMODE = obj._expUtils.GAIN_MAX_MOONMODE = 50
    altitudes = [-6, -6.3, -7.1, -8.3, -9.7, -11.1, -12]
    gains = []
    for altitude in reversed(altitudes) if reverse else altitudes:
        transition.apply(altitude)
        desired = transition_gain((50 if moon else 100) * transition.weight, info)
        obj.apply_transition_limits()
        assert obj.gain_min == obj.gain_max == pytest.approx(desired)
        assert obj._expUtils.GAIN_NEXT == pytest.approx(desired, abs=.001)
        gains.append(obj._expUtils.GAIN_NEXT)
    assert gains == sorted(gains, reverse=reverse)
    # Disabling restores the endpoint, subject to optional camera-wide precision.
    obj.config['TWILIGHT_TRANSITION']['ENABLE'] = False
    obj._expUtils.GAIN_MIN_NIGHT = obj._expUtils.GAIN_MAX_NIGHT = 85.912
    obj.night_av[1] = False
    expected = transition_gain(85.912, info) if hasattr(obj, 'effective_gain_limits') else 85.912
    assert obj.gain_max == expected
