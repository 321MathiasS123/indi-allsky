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


@pytest.mark.parametrize('info', [{}, {'quantum': 1}, {'values': [0, 20, 40, 60, 80, 100]}])
@pytest.mark.parametrize('moon', [False, True])
@pytest.mark.parametrize('reverse', [False, True])
def test_dawn_dusk_and_moonmode_publish_reachable_fixed_gain_limits(info, moon, reverse):
    obj, transition = controller('exposure_basic')
    obj.twilight_gain_info = info
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
    # Disabling restores the original fixed endpoint; no residual blend/rounding.
    obj.config['TWILIGHT_TRANSITION']['ENABLE'] = False
    obj._expUtils.GAIN_MAX_NIGHT = 85.912
    obj.night_av[1] = False
    assert obj.gain_max == 85.912
