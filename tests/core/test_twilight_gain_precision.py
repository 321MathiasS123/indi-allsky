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


def test_moving_limits_preserve_optional_camera_precision_inside_autogain_bounds():
    obj, _ = controller('exposure_autogain_exp_prio_db_1_10')
    obj._expUtils.GAIN_NEXT = 40.4
    obj._expUtils.GAIN_DELTA = .25
    camera_precision = hasattr(obj, 'effective_gain')
    if camera_precision:
        obj.gain_quantum = 1
    assert obj.gain_min == 0 and obj.gain_max == 100
    obj.apply_transition_limits()
    expected = 40 if camera_precision else 40.4
    assert obj._expUtils.GAIN_NEXT == pytest.approx(expected, abs=.001)
    assert obj._expUtils.GAIN_DELTA == pytest.approx(.25 + expected - 40.4, abs=.001)


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
