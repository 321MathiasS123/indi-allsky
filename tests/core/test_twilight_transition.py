from copy import deepcopy
import ast
from datetime import datetime, timedelta, timezone
import math
from multiprocessing import Array
from pathlib import Path
from types import SimpleNamespace

import ephem
import numpy as np
import pytest

from indi_allsky import constants, exposure as modes
from indi_allsky.twilight import (
    TwilightTransition, capture_period, day_altitude, exposure_minimum, interpolate, night_weight, observer_at, transition_forecast, transition_gain,
)


def config():
    return {
        'TWILIGHT_TRANSITION': {'ENABLE': True, 'NIGHT_ALT': -12.0},
        'NIGHT_SUN_ALT_DEG': -6.0, 'USE_NIGHT_COLOR': False,
        'TARGET_ADU': 100, 'TARGET_ADU_DAY': 50,
        'TARGET_ADU_DEV': 10, 'TARGET_ADU_DEV_DAY': 20,
        'GAMMA_CORRECTION': 1.0, 'GAMMA_CORRECTION_DAY': 2.0,
        'WBR_FACTOR': 1.8, 'WBR_FACTOR_DAY': 1.2,
        'EXPOSURE_PERIOD': 30, 'EXPOSURE_PERIOD_DAY': 10,
        'CCD_CONFIG': {'AUTO_GAIN_LEVELS': 8},
    }


@pytest.mark.parametrize('altitude,expected', [(10, 0), (-6, 0), (-9, .5), (-12, 1), (-30, 1)])
def test_curve_endpoints_and_midpoint(altitude, expected):
    assert night_weight(altitude) == expected


@pytest.mark.parametrize('start,end', [(0, -12), (-3, -9), (4, -18), (-6, -6.1)])
def test_custom_interval_is_symmetric_and_independent_of_mode_threshold(start, end):
    source = config()
    source['TWILIGHT_TRANSITION'].update(DAY_ALT=start, NIGHT_ALT=end)
    source['HIGHLIGHT_PROTECTION'] = {'ENABLE': True, 'GAMMA_DAY': 1.8, 'GAMMA': 0}
    before = deepcopy(source)
    t = TwilightTransition(source)
    # Include both endpoints and a partial reversal, not just a completed night.
    altitudes = [start + 1, start, (start + end) / 2, end, end - 1]
    for altitude, expected in zip(altitudes + altitudes[::-1], [0, 0, .5, 1, 1, 1, 1, .5, 0, 0]):
        t.apply(altitude)
        assert t.weight == pytest.approx(expected)
        assert t.config['TARGET_ADU'] == t.config['TARGET_ADU_DAY'] == pytest.approx(50 + 50 * expected)
        assert t.config['GAMMA_CORRECTION'] == pytest.approx(2 - expected)
        assert t.config['HIGHLIGHT_PROTECTION']['GAMMA'] == pytest.approx(1.8 - .8 * expected)
        assert capture_period(source, altitude) == pytest.approx(10 + 20 * expected)
        assert t.config['NIGHT_SUN_ALT_DEG'] == -6
    assert source == before


@pytest.mark.parametrize('explicit_null', [False, True])
def test_legacy_endpoint_preserves_custom_mode_threshold(explicit_null):
    source = config()
    source['NIGHT_SUN_ALT_DEG'] = -4
    if explicit_null:
        source['TWILIGHT_TRANSITION']['DAY_ALT'] = None
    t = TwilightTransition(source)
    t.apply(-8)
    assert t.weight == .5
    assert capture_period(source, -8) == 20


def test_twilight_reverses_at_partial_night_without_reset_or_rescaling():
    t = TwilightTransition(config())
    altitudes = [-5, -6, -7, -8, -9, -8, -7, -6, -5]
    weights = []
    for altitude in altitudes:
        t.apply(altitude)
        weights.append(t.weight)
    assert weights == weights[::-1]
    assert max(weights) == .5
    # Entering/exiting the band has a zero slope.
    assert night_weight(-6.001) < .000001
    assert 1 - night_weight(-11.999) < .000001


@pytest.mark.parametrize('start,end', [(-6, -6), (-6, 0), (float('nan'), -12), (-6, float('-inf'))])
def test_invalid_elevations_rejected(start, end):
    with pytest.raises(ValueError):
        night_weight(-9, start, end)


@pytest.mark.parametrize('missing', [False, True])
def test_disabled_feature_reuses_unmodified_config(missing):
    source = config()
    if missing:
        source.pop('TWILIGHT_TRANSITION')
    else:
        source['TWILIGHT_TRANSITION']['ENABLE'] = False
    before = deepcopy(source)
    t = TwilightTransition(source)
    t.apply(-9)
    t.update(datetime.now(timezone.utc), 57, 0, 0)
    assert t.config is source
    assert source == before
    assert t.weight is None


def test_runtime_endpoints_do_not_drift_or_modify_saved_config():
    source = config()
    before = deepcopy(source)
    t = TwilightTransition(source)
    for altitude in (-6, -9, -12, -8, -9, -6):
        t.apply(altitude)
        w = night_weight(altitude)
        assert t.config['TARGET_ADU'] == t.config['TARGET_ADU_DAY'] == 50 + 50 * w
        assert t.config['GAMMA_CORRECTION'] == 2 - w
        assert t.config['WBR_FACTOR'] == pytest.approx(1.2 + .6 * w)
    assert source == before


@pytest.mark.parametrize('day_override,night_override', [(0, 0), (1.8, 0), (0, .9), (1.8, .9)])
def test_highlight_gamma_resolves_inherit_before_interpolation(day_override, night_override):
    source = config()
    source['HIGHLIGHT_PROTECTION'] = {'ENABLE': True, 'GAMMA_DAY': day_override, 'GAMMA': night_override}
    before = deepcopy(source)
    t = TwilightTransition(source)
    t.apply(-9)
    gamma = ((day_override or 2.0) + (night_override or 1.0)) / 2
    assert t.config['HIGHLIGHT_PROTECTION']['GAMMA'] == gamma
    assert t.config['HIGHLIGHT_PROTECTION']['GAMMA_DAY'] == gamma
    assert source == before


def test_use_night_color_still_wins_with_highlight_overrides():
    source = config()
    source['USE_NIGHT_COLOR'] = True
    source['HIGHLIGHT_PROTECTION'] = {'ENABLE': True, 'GAMMA': .9, 'GAMMA_DAY': 1.8}
    t = TwilightTransition(source)
    t.apply(-9)
    assert t.config['GAMMA_CORRECTION'] == 1
    assert t.config['WBR_FACTOR'] == 1.8
    assert t.config['HIGHLIGHT_PROTECTION'] == source['HIGHLIGHT_PROTECTION']
    assert t.config['TARGET_ADU'] == 75


def test_capture_time_replay_and_restart_do_not_depend_on_processing_time():
    source = config()
    when = datetime(2026, 6, 21, 23, 45, tzinfo=timezone.utc)
    t = TwilightTransition(source)
    t.update(when, 57, 0, 0)
    original = t.weight
    t.update(when + timedelta(hours=6), 57, 0, 0)
    t.update(when, 57, 0, 0)
    fresh = TwilightTransition(source)
    fresh.update(when, 57, 0, 0)
    assert 0 < original < 1
    assert t.weight == fresh.weight == original


@pytest.mark.parametrize('month,day,expected', [(3, 20, 38.1), (6, 21, 65.7), (12, 21, 41.6)])
def test_seasonal_duration_uses_solar_crossings(month, day, expected):
    forecast = transition_forecast(config(), datetime(2026, month, day, 15, tzinfo=timezone.utc), 50, 0)
    assert forecast['dusk_minutes'] == pytest.approx(expected, abs=.2)
    assert forecast['dawn_minutes'] == pytest.approx(expected, abs=.2)
    assert forecast['maximum'] == 1


@pytest.mark.parametrize('latitude,month,expected', [(57, 6, .64), (-57, 12, .64), (65, 6, 0), (80, 12, 1)])
def test_partial_and_polar_cycles_have_no_false_completion(latitude, month, expected):
    forecast = transition_forecast(config(), datetime(2026, month, 21, 15, tzinfo=timezone.utc), latitude, 0)
    assert forecast['maximum'] == pytest.approx(expected, abs=.005)
    assert forecast['dusk_minutes'] is None
    assert forecast['dawn_minutes'] is None
    if latitude == 80:
        assert forecast['minimum'] == 1


def controller(name='exposure_basic', source=None):
    t = TwilightTransition(source or config())
    t.apply(-9)
    obj = getattr(modes, name)(t.config, Array('i', 7), Array('i', 10), Array('i', 6), Array('i', [1, 0]))
    u = obj._expUtils
    u.EXPOSURE_MIN_DAY, u.EXPOSURE_MIN_NIGHT, u.EXPOSURE_MAX = .0001, 1, 30
    minimum = 100 if name.endswith('iso') else 1 if name.endswith('iso_1_100') else 0
    maximum = minimum * 8 if minimum else 100
    u.GAIN_MIN_DAY = u.GAIN_MAX_DAY = minimum
    u.GAIN_MIN_NIGHT = u.GAIN_MIN_MOONMODE = maximum if name == 'exposure_basic' else minimum
    u.GAIN_MAX_NIGHT = u.GAIN_MAX_MOONMODE = maximum
    u.BINNING_DAY = u.BINNING_NIGHT = u.BINNING_MOONMODE = 1
    u.EXPOSURE_NEXT, u.GAIN_NEXT = 1, minimum
    return obj, t


def test_moving_target_cannot_remain_locked_on_old_brightness():
    obj, t = controller()
    t.apply(-6)
    obj.compare_exposure(50, 1, 0)
    assert obj.target_adu_found
    t.apply(-9)
    obj.compare_exposure(50, 1, 0)
    assert obj._expUtils.EXPOSURE_NEXT > 1
    assert not obj.target_adu_found


def test_target_tracks_gradually_inside_tolerance_and_reverses_at_partial_night():
    obj, t = controller()
    obj._expUtils.GAIN_MIN_NIGHT = obj._expUtils.GAIN_MAX_NIGHT = 0
    exposure = 1.0
    altitudes = [-6 - i / 10 for i in range(31)]
    for altitude in altitudes + altitudes[-2::-1]:
        t.apply(altitude)
        measured_adu = 50 * exposure  # constant scene, fixed gain
        obj.compare_exposure(measured_adu, exposure, 0)
        obj.apply_transition_limits()
        exposure = obj._expUtils.EXPOSURE_NEXT
        assert 50 * exposure == pytest.approx(t.config['TARGET_ADU'], abs=.002)
        assert obj.target_adu_found


@pytest.mark.parametrize('name', modes.__all__)
def test_resolved_exposure_minimum_moves_logarithmically_for_all_controllers(name):
    obj, t = controller(name)
    assert obj.exposure_min == pytest.approx(.01)
    t.apply(-6)
    assert obj.exposure_min == .0001
    t.apply(-12)
    assert obj.exposure_min == 1


def test_fixed_gain_and_exposure_limits_update_even_inside_brightness_deadband():
    obj, t = controller()
    obj.compare_exposure(75, 1, 0)
    obj.apply_transition_limits()
    assert obj._expUtils.GAIN_NEXT == 50
    t.apply(-12)
    obj._expUtils.EXPOSURE_NEXT = .01
    obj.apply_transition_limits()
    assert obj._expUtils.EXPOSURE_NEXT == 1
    assert obj._expUtils.GAIN_NEXT == 100


@pytest.mark.parametrize('name', modes.__all__)
@pytest.mark.parametrize('night', [False, True])
@pytest.mark.parametrize('enabled', [False, True])
def test_uninitialized_or_disabled_transition_keeps_original_exposure_minimum(name, night, enabled):
    obj, t = controller(name)
    # Before the first frame, use the selected mode. Disabled transitions must
    # also ignore any weight left over in the runtime configuration.
    if enabled:
        t.config.pop('_TWILIGHT_WEIGHT')
    t.config['TWILIGHT_TRANSITION']['ENABLE'] = enabled
    obj.night_av[0] = night
    assert obj.exposure_min == (1 if night else .0001)


@pytest.mark.parametrize('name', modes.__all__)
def test_disabled_limit_update_does_nothing(name):
    source = config()
    source['TWILIGHT_TRANSITION']['ENABLE'] = False
    obj, t = controller(name, source)
    before = (list(obj.exposure_av), list(obj.gain_av), list(obj.binning_av))
    obj.apply_transition_limits()
    assert (list(obj.exposure_av), list(obj.gain_av), list(obj.binning_av)) == before


def test_capture_period_has_same_elevation_curve():
    assert capture_period(config(), -6) == 10
    assert capture_period(config(), -9) == 20
    assert capture_period(config(), -12) == 30


@pytest.mark.parametrize('altitude,focus,sqm,enabled,expected', [
    (-5, False, False, True, 10), (-9, False, False, True, 20),
    (-13, False, False, True, 30), (-9, True, False, True, 4),
    (-9, False, True, True, 0), (-9, False, False, False, 30),
])
def test_capture_scheduler_and_frame_period_use_same_transition(
        altitude, focus, sqm, enabled, expected):
    import logging
    import queue
    from indi_allsky.capture_period import CapturePeriodQueue

    path = Path(__file__).resolve().parents[2] / 'indi_allsky/capture.py'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'CaptureWorker')
    shoot = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == 'shoot')
    period_hook = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == '_twilightCapturePeriod')
    schedule = next(node for node in ast.walk(cls) if isinstance(node, ast.If)
                    and ast.unparse(node.test) == 'self.focus_mode'
                    and any(isinstance(child, ast.Assign) and any(isinstance(target, ast.Name)
                            and target.id == 'next_frame_time' for target in child.targets) for child in node.body))
    source = config()
    source['TWILIGHT_TRANSITION']['ENABLE'] = enabled
    outgoing = queue.Queue()
    period_queue = CapturePeriodQueue(outgoing)
    worker = SimpleNamespace(config=source, focus_mode=focus, night=True, night_av=[1, 0],
                             astro_av=[altitude, 0, 0], add_period_delay=2, camera_id=1,
                             _period_queue=period_queue,
                             _dateCalcs=SimpleNamespace(getDayDate=lambda: datetime(2026, 10, 10).date()))
    worker.indiclient = SimpleNamespace(setCcdExposure=lambda *a, **kw: period_queue.put({'filename': 'frame.fit'}))
    namespace = dict(constants=constants, capture_period=capture_period, logger=logging.getLogger(__name__),
                     self=worker, waiting_for_sqm_frame=sqm, now_time=100, frame_start_time=95)
    exec(compile(ast.Module(body=[shoot, period_hook, schedule], type_ignores=[]), str(path), 'exec'), namespace)
    namespace['shoot'](worker, 1, 50, 1, sqm_exposure=sqm)
    assert namespace['_twilightCapturePeriod'](worker) == (capture_period(source, altitude) if enabled else None)
    assert outgoing.get_nowait()['capture_period'] == expected
    assert namespace['next_frame_time'] == (100 if focus else 95) + expected + (0 if sqm else 2)


@pytest.mark.parametrize('dawn', [False, True])
def test_short_exposures_do_not_accumulate_rounding_drift(dawn):
    obj, t = controller()
    u = obj._expUtils
    u.EXPOSURE_MIN_DAY = u.EXPOSURE_MIN_NIGHT = .000001
    u.GAIN_MIN_NIGHT = u.GAIN_MAX_NIGHT = 0
    exposure = .0001 if dawn else .00005
    u.EXPOSURE_NEXT = exposure
    for step in range(1501):
        t.apply(-12 + 6 * step / 1500 if dawn else -6 - 6 * step / 1500)
        obj.compare_exposure(int(exposure * 1_000_000), exposure, 0)
        obj.apply_transition_limits()
        next_exposure = u.EXPOSURE_NEXT
        # One microsecond is the existing shared-memory precision. Fractional
        # requests must accumulate instead of losing a microsecond every frame.
        assert abs(next_exposure * 1_000_000 - t.config['TARGET_ADU']) <= 1.01
        assert abs(next_exposure - exposure) <= .00000101
        exposure = next_exposure


@pytest.mark.parametrize('name', modes.__all__)
def test_moving_minimum_never_overrides_camera_maximum(name):
    obj, t = controller(name)
    obj._expUtils.EXPOSURE_MAX = .1  # camera is stricter than saved configuration
    t.apply(-12)
    obj.apply_transition_limits()
    assert obj.exposure_min <= .1
    assert obj._expUtils.EXPOSURE_NEXT <= .1


def test_return_after_excluded_frames_does_not_apply_stale_target_ratio():
    obj, t = controller()
    t.apply(-6)
    obj.compare_exposure(50, 1, 0)
    # Exposure control was held while frames were excluded; illumination now
    # already matches the new target. Do not replay the missed target changes.
    t.apply(-12)
    obj.compare_exposure(100, 1, 100)
    assert obj._expUtils.EXPOSURE_NEXT == 1


@pytest.mark.parametrize('name,minimum', [
    ('exposure_autogain_exp_prio_db_1_10', 100), ('exposure_autogain_exp_prio_db', 6),
    ('exposure_autogain_exp_prio_iso', 200), ('exposure_autogain_exp_prio_iso_1_100', 2),
])
def test_dawn_brightness_reduction_at_nonzero_gain_floor(name, minimum):
    obj, t = controller(name)
    obj._expUtils.GAIN_MIN_NIGHT = minimum
    obj._expUtils.GAIN_MAX_NIGHT = minimum * 4
    next_exposure, _, _, _ = obj.reduce_gain(1, minimum, .9)
    assert next_exposure == pytest.approx(.9)


@pytest.mark.parametrize('altitude', [0, -3, -6, -9, -12])
@pytest.mark.parametrize('start', [None, 0, -3])
@pytest.mark.parametrize('gain_info', [{}, {'quantum': 1}, {'values': [0, 40, 100]}])
def test_capture_restart_obeys_camera_limit_and_current_blend(altitude, start, gain_info):
    tree = ast.parse((Path(__file__).resolve().parents[2] / 'indi_allsky/capture.py').read_text(encoding='utf-8'))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'CaptureWorker')
    initialize = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_initialize')
    block = next(n for n in initialize.body if isinstance(n, ast.If) and 'TWILIGHT_TRANSITION' in ast.unparse(n.test))
    obj, _ = controller()
    obj._expUtils.EXPOSURE_MAX = .1
    worker = SimpleNamespace(config=config(), _expUtils=obj._expUtils, night=True,
                             night_av=[1, 0], astro_av=[altitude, 0, 0])
    worker.config['TWILIGHT_TRANSITION']['DAY_ALT'] = start
    namespace = dict(self=worker, constants=constants, night_weight=night_weight, interpolate=interpolate,
                     day_altitude=day_altitude,
                     exposure_minimum=exposure_minimum, maximum_exposure=.1, ccd_exposure_default=.05,
                     exposure_class_str='exposure_basic', gain_day=0, gain_night=100, gain_moonmode=50,
                     transition_gain=transition_gain, ccd_info={'GAIN_INFO': gain_info})
    exec(compile(ast.Module(body=[block], type_ignores=[]), 'capture-startup', 'exec'), namespace)
    assert 0 < namespace['ccd_exposure_default'] <= .1
    expected = night_weight(altitude, -6 if start is None else start, -12)
    assert namespace['ccd_gain_default'] == transition_gain(100 * expected, gain_info)


def test_wider_custom_interval_changes_forecast_without_changing_mode_boundary():
    source = config()
    when = datetime(2026, 10, 7, 15, tzinfo=timezone.utc)
    original = transition_forecast(source, when, 53, 11)
    source['TWILIGHT_TRANSITION']['DAY_ALT'] = 0
    custom = transition_forecast(source, when, 53, 11)
    for key in ('dusk_minutes', 'dawn_minutes'):
        assert 1.8 * original[key] < custom[key] < 2.2 * original[key]
    source['NIGHT_SUN_ALT_DEG'] = -18
    assert transition_forecast(source, when, 53, 11) == custom


@pytest.mark.parametrize('name', modes.__all__)
@pytest.mark.parametrize('start,end', [(0, -12), (-3, -9)])
def test_custom_exposure_limits_do_not_jump_at_operational_mode_switch(name, start, end):
    source = config()
    source['TWILIGHT_TRANSITION'].update(DAY_ALT=start, NIGHT_ALT=end)
    obj, t = controller(name, source)
    t.apply(-6)
    before = obj.exposure_min
    assert before == pytest.approx(.01)
    obj.night_av[constants.NIGHT_NIGHT] = False
    assert obj.exposure_min == before
    if name == 'exposure_basic':
        assert obj.gain_min == obj.gain_max == 50
    # Exercise both directions as well as the exact operational boundary.
    for altitude in [start, -6, end, -6, start]:
        t.apply(altitude)
        obj.night_av[constants.NIGHT_NIGHT] = altitude < -6
        obj._expUtils.EXPOSURE_NEXT = .00001
        obj._expUtils.GAIN_NEXT = obj.gain_min
        obj.apply_transition_limits()
        assert obj._expUtils.EXPOSURE_NEXT == pytest.approx(obj.exposure_min, abs=.000001)
        assert obj.gain_min <= obj._expUtils.GAIN_NEXT <= obj.gain_max


@pytest.mark.parametrize('latitude', [-90, -80, -65, -57, 0, 57, 65, 80, 90])
@pytest.mark.parametrize('month', [3, 6, 9, 12])
@pytest.mark.parametrize('start', [None, 0])
def test_forecast_agrees_with_sampled_solar_cycle(latitude, month, start):
    when = datetime(2026, month, 21, 15, tzinfo=timezone.utc)
    source = config()
    source['TWILIGHT_TRANSITION']['DAY_ALT'] = start
    forecast = transition_forecast(source, when, latitude, 0)
    obs = observer_at(when, latitude, 0, 0)
    noon = obs.previous_transit(ephem.Sun())
    samples = []
    for minute in range(0, 1441, 10):
        obs.date = noon + minute / 1440
        samples.append(night_weight(math.degrees(ephem.Sun(obs).alt), -6 if start is None else start, -12))
    assert forecast['maximum'] == pytest.approx(max(samples), abs=.001)
    assert forecast['minimum'] == pytest.approx(min(samples), abs=.001)
    for key in ('dusk_minutes', 'dawn_minutes'):
        assert forecast[key] is None or 0 < forecast[key] < 720


@pytest.mark.parametrize('latitude,month', [(90, 10), (-90, 4)])
def test_polar_partial_cycle_does_not_invent_a_midnight_reversal(latitude, month):
    forecast = transition_forecast(config(), datetime(2026, month, 20, 15, tzinfo=timezone.utc), latitude, 0)
    assert 0 < forecast['minimum'] < forecast['maximum'] < 1
    assert forecast['reversal_utc'] is None


@pytest.mark.parametrize('name', modes.__all__)
@pytest.mark.parametrize('latitude', [0, 57])
@pytest.mark.parametrize('highlight_enabled', [False, True])
def test_complete_and_partial_solar_cycles_with_feedback_and_restart(name, latitude, highlight_enabled):
    highlight = pytest.importorskip('indi_allsky.highlight') if highlight_enabled else None
    source = config()
    source['HIGHLIGHT_PROTECTION'] = {'ENABLE': highlight_enabled}
    obj, t = controller(name, source)
    u = obj._expUtils
    minimum, maximum = ((100, 800) if name.endswith('iso') else (1, 8) if name.endswith('iso_1_100')
                        else (6, 24) if name.endswith('_db') else (0, 100))
    u.GAIN_MIN_DAY = u.GAIN_MAX_DAY = minimum
    u.GAIN_MIN_NIGHT = maximum if name == 'exposure_basic' else minimum
    u.GAIN_MAX_NIGHT = maximum
    u.EXPOSURE_MIN_DAY, u.EXPOSURE_MIN_NIGHT = .000032, .001
    u.EXPOSURE_NEXT, u.GAIN_NEXT = .01, minimum
    y, x = np.ogrid[:64, :64]
    scene = 1 + 5 * np.exp(-((x - 32) ** 2 + (y - 32) ** 2) / 32)
    mask = np.ones(scene.shape, dtype=np.uint8)
    weights = []
    modes_seen = set()
    start = datetime(2026, 6, 21, 16, tzinfo=timezone.utc)
    for step in range(181):
        t.update(start + timedelta(minutes=6 * step), latitude, 0, 0)
        obj.night_av[constants.NIGHT_NIGHT] = t.altitude < -6
        modes_seen.add(bool(obj.night_av[constants.NIGHT_NIGHT]))
        weights.append(t.weight)
        if step == 90:
            # Restart the actual controller against retained camera settings.
            obj = getattr(modes, name)(t.config, obj.exposure_av, obj.gain_av, obj.binning_av, obj.night_av)
            u = obj._expUtils
        exposure, gain = u.EXPOSURE_NEXT, u.GAIN_NEXT
        physical_gain = (gain / 100 if name.endswith('iso') else gain if name.endswith('iso_1_100')
                         else 10 ** (gain / 20) if name.endswith('_db') else 10 ** (gain / 200))
        illumination = .00004 * math.exp(max(-4, min(14, (t.altitude + 12) * .8)))
        raw = np.clip(scene * illumination * exposure * physical_gain * 65535, 0, 65535).astype(np.uint16)
        if highlight:
            obj.compare_highlights(highlight.measure(raw, mask, 16), exposure, gain)
        else:
            obj.compare_exposure(int(raw.mean() / 256), exposure, gain)
        obj.apply_transition_limits()
        assert math.isfinite(u.EXPOSURE_NEXT) and math.isfinite(u.GAIN_NEXT)
        assert obj.exposure_min - .000001 <= u.EXPOSURE_NEXT <= obj.exposure_max
        assert obj.gain_min - .001 <= u.GAIN_NEXT <= obj.gain_max + .001
    assert modes_seen == {False, True}
    assert min(weights) == 0
    assert max(weights) == (1 if latitude == 0 else pytest.approx(.64, abs=.005))


@pytest.mark.parametrize('name', modes.__all__)
@pytest.mark.parametrize('transition_enabled', [False, True])
@pytest.mark.parametrize('highlight_enabled', [False, True])
def test_optional_highlight_controller_respects_blended_limits(name, transition_enabled, highlight_enabled):
    highlight = pytest.importorskip('indi_allsky.highlight')
    source = config()
    source['TWILIGHT_TRANSITION']['ENABLE'] = transition_enabled
    source['HIGHLIGHT_PROTECTION'] = {'ENABLE': highlight_enabled}
    obj, t = controller(name, source)
    gain = obj.gain_min
    obj._expUtils.EXPOSURE_NEXT, obj._expUtils.GAIN_NEXT = 10, gain
    if highlight_enabled:
        # Clipping must lower exposure even though the blended ADU target is met.
        obj.compare_highlights(highlight.HighlightMeasurement(3, 8, 75), 10, gain)
        assert obj._expUtils.EXPOSURE_NEXT < 10
    else:
        obj.compare_exposure(obj.config['TARGET_ADU'], 10, gain)
        assert obj.target_adu_found
    obj.apply_transition_limits()
    assert obj.exposure_min <= obj._expUtils.EXPOSURE_NEXT <= obj.exposure_max
    assert obj.gain_min <= obj._expUtils.GAIN_NEXT <= obj.gain_max
    assert obj.exposure_min == pytest.approx(.01 if transition_enabled else 1)
