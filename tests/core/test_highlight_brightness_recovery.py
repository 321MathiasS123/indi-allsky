"""Large ADU errors recover promptly without compounding delayed feedback."""
import math

import pytest

from test_highlight_exposure import controller
from test_highlight_response import (
    fixed_gain_controller, measurement, set_inflight, settings, signal,
)


@pytest.mark.parametrize('actuator', ['exposure', 'gain'])
@pytest.mark.parametrize('adu,expected', [(10, 1.5), (20, 1.5), (200, 2 / 3)])
def test_large_errors_exceed_ordinary_slew_without_overshooting_target(actuator, adu, expected):
    if actuator == 'gain':
        instance = controller('exposure_autogain_exp_prio_db_1_10')
        instance._expUtils.GAIN_MAX_NIGHT = 400
        instance.gain_quantum = 1
        source = (30, 150)
    else:
        instance = fixed_gain_controller()
        source = (20, 0)
    set_inflight(instance, *source)
    instance.compare_highlights(measurement(adu), *source)
    ratio = signal(instance, settings(instance)) / signal(instance, source)
    # Integer gain contributes at most one gain unit of rounding.
    assert ratio == pytest.approx(expected, rel=.012)
    assert ratio > 1.25 if adu < 60 else ratio < .8
    if actuator == 'gain':
        assert settings(instance)[0] == 30
    else:
        assert settings(instance)[1] == 0


@pytest.mark.parametrize('adu,bound', [(1, 2), (250, .5)])
def test_extreme_errors_retain_source_relative_signal_bounds(adu, bound):
    instance = fixed_gain_controller()
    instance.config.update(TARGET_ADU=20, TARGET_ADU_DEV=2)
    set_inflight(instance, 1, 0)
    previous = 1.0
    for _ in range(12):
        instance.compare_highlights(measurement(adu), 1, 0)
        command = settings(instance)[0]
        assert min(1, bound) <= command <= max(1, bound)
        assert 2 / 3 - .000001 <= command / previous <= 1.5 + .000001
        previous = command
    assert command == pytest.approx(bound)


@pytest.mark.parametrize('adu,target,direction', [(30, 2, 1), (160, .5, -1)])
def test_delayed_absolute_target_tapers_each_step_without_compounding(adu, target, direction):
    instance = fixed_gain_controller()
    set_inflight(instance, 1, 0)
    commands = [1.0]
    for _ in range(10):
        instance.compare_highlights(measurement(adu), 1, 0)
        commands.append(settings(instance)[0])
    steps = [abs(math.log(after / before)) for before, after in zip(commands, commands[1:])]
    assert steps[0] > math.log(1.25)
    assert all(later < earlier for earlier, later in zip(steps, steps[1:]))
    assert all(direction * (value - target) <= 0 for value in commands)
    assert commands[-1] == pytest.approx(target, rel=.001)


@pytest.mark.parametrize('delay', [0, 1, 3])
@pytest.mark.parametrize('initial_adu,target,direction', [(10, 60, 1), (200, 80, -1)])
def test_steady_light_recovery_with_delayed_captures_never_crosses_target(delay, initial_adu, target, direction):
    instance = fixed_gain_controller()
    captures = [(1, 0)] * (delay + 1)
    commands = [1.0]
    for _ in range(60):
        source = captures.pop(0)
        set_inflight(instance, *captures[-1] if captures else source)
        instance.compare_highlights(measurement(initial_adu * source[0]), *source)
        command = settings(instance)
        captures.append(command)
        commands.append(command[0])
        assert direction * (initial_adu * command[0] - target) <= .0001
    assert all(direction * (after - before) >= -1e-6
               for before, after in zip(commands, commands[1:]))
    assert initial_adu * commands[-1] == pytest.approx(target, abs=.002)


def test_severe_dark_measurement_cannot_undo_newer_cut():
    instance = fixed_gain_controller()
    set_inflight(instance, 1, 0)
    instance.compare_highlights(measurement(200), 1, 0)
    reduced = settings(instance)
    assert reduced[0] < .8
    instance.compare_highlights(measurement(1), 1, 0)
    assert settings(instance) == reduced


def test_severe_bright_measurement_cancels_pending_growth():
    instance = fixed_gain_controller()
    set_inflight(instance, 1, 0)
    instance.compare_highlights(measurement(1), 1, 0)
    assert settings(instance)[0] == 1.5
    instance.compare_highlights(measurement(200), 1, 0)
    assert settings(instance)[0] <= .9


def test_brief_dark_pulse_keeps_growth_bounded_and_reverses_on_fresh_brightness():
    instance = fixed_gain_controller()
    set_inflight(instance, 1, 0)
    instance.compare_highlights(measurement(10), 1, 0)
    command = settings(instance)
    assert command[0] == 1.5
    # Illumination returns to its original ADU 70 after one dark capture.
    # Fresh evidence of the brighter command must taper back toward the band.
    for _ in range(20):
        source = command
        set_inflight(instance, *source)
        instance.compare_highlights(measurement(70 * source[0]), *source)
        command = settings(instance)
        assert 80 / 70 - .000001 <= command[0] <= source[0]
    assert 70 * command[0] == pytest.approx(80, abs=.002)


@pytest.mark.parametrize('adu', [55, 59, 60, 70, 80, 81, 95])
def test_ordinary_brightness_corrections_keep_existing_half_error_response(adu):
    instance = fixed_gain_controller()
    set_inflight(instance, 1, 0)
    instance.compare_highlights(measurement(adu), 1, 0)
    expected = math.sqrt(60 / adu) if adu < 60 else math.sqrt(80 / adu) if adu > 80 else 1
    assert settings(instance)[0] == pytest.approx(expected, abs=.000001)


@pytest.mark.parametrize('night,exposure,target,deviation', [
    (True, 1, 70, 12), (False, 1, 80, 12),
    (True, .0009, 70, 7), (False, .0009, 80, 7),
])
@pytest.mark.parametrize('direction', [-1, 1])
def test_recovery_scales_with_configured_target_and_selected_deviation(night, exposure, target, deviation, direction):
    instance = controller('exposure_basic', night=night)
    instance.config.update(TARGET_ADU_DEV=12, TARGET_ADU_DEV_DAY=7)
    instance._expUtils.GAIN_MIN_NIGHT = instance._expUtils.GAIN_MAX_NIGHT = 0
    instance._expUtils.EXPOSURE_MIN_NIGHT = instance._expUtils.EXPOSURE_MIN_DAY = .000001
    set_inflight(instance, exposure, 0)
    adu = (target - deviation) / 2 if direction > 0 else (target + deviation) * 2
    instance.compare_highlights(measurement(adu), exposure, 0)
    assert settings(instance)[0] == pytest.approx(exposure * 2 ** (direction / 2), abs=.000001)
