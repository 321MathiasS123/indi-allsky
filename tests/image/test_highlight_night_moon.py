"""Night Moon exclusion uses a separate, capture-time solar envelope."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import math
from types import SimpleNamespace
from unittest.mock import Mock

import ephem
import pytest

from indi_allsky.highlight_output import moon_region, night_moon_weight


CAPTURE_END = datetime(2026, 10, 25, 18, tzinfo=timezone.utc)
POSITION = (52.854, 11.148, 23.0)


def solar_at(monkeypatch, altitude):
    observer = SimpleNamespace()
    monkeypatch.setattr(ephem, 'Observer', lambda: observer)
    sun = Mock(return_value=SimpleNamespace(alt=math.radians(altitude)))
    monkeypatch.setattr(ephem, 'Sun', sun)
    return observer, sun


@pytest.mark.parametrize('altitude,expected', [
    (5, 0), (-14, 0), (-16, 0), (-16.5, .15625), (-17, .5),
    (-17.5, .84375), (-18, 1), (-25, 1), (-90, 1),
])
def test_solar_envelope_has_exact_endpoints_and_smooth_fade(monkeypatch, altitude, expected):
    solar_at(monkeypatch, altitude)
    assert night_moon_weight(CAPTURE_END, 30, POSITION) == pytest.approx(expected)


def test_capture_midpoint_and_geometric_observer_are_used_without_rounding(monkeypatch):
    observer, sun = solar_at(monkeypatch, -17)
    position = (52.854123, 11.148456, 123.75)
    assert night_moon_weight(CAPTURE_END, 29.75, position) == pytest.approx(.5)
    sun.assert_called_once_with(observer)
    assert observer.date == CAPTURE_END - timedelta(seconds=29.75 / 2)
    assert observer.lat == math.radians(position[0])
    assert observer.lon == math.radians(position[1])
    assert observer.elevation == position[2]
    assert observer.pressure == 0


def test_real_ephemeris_uses_solar_centre_at_midpoint():
    observer = ephem.Observer()
    observer.lat, observer.lon = map(math.radians, POSITION[:2])
    observer.elevation, observer.pressure = POSITION[2], 0
    observer.horizon = '-17'
    observer.date = datetime(2026, 10, 25, 12, tzinfo=timezone.utc)
    midpoint = observer.next_setting(ephem.Sun(), use_center=True).datetime().replace(tzinfo=timezone.utc)
    end = midpoint + timedelta(seconds=15)
    # Event searches have a small numerical tolerance, well below a 15s shift.
    assert night_moon_weight(end, 30, POSITION) == pytest.approx(.5, abs=.0002)


def test_weight_is_monotonic_and_has_zero_endpoint_slopes(monkeypatch):
    observer, sun = solar_at(monkeypatch, 0)
    weights = []
    for index in range(101):
        sun.return_value.alt = math.radians(-16 - index / 50)
        weights.append(night_moon_weight(CAPTURE_END, 30, POSITION))
    assert all(before <= after for before, after in zip(weights, weights[1:]))
    epsilon = .0001
    for altitude in (-16, -18):
        sun.return_value.alt = math.radians(altitude - epsilon)
        left = night_moon_weight(CAPTURE_END, 30, POSITION)
        sun.return_value.alt = math.radians(altitude + epsilon)
        right = night_moon_weight(CAPTURE_END, 30, POSITION)
        assert abs(left - right) / (2 * epsilon) < .0001


@pytest.mark.parametrize('date,exposure,position', [
    (None, 30, POSITION),
    (CAPTURE_END, None, POSITION),
    (CAPTURE_END, 'bad', POSITION),
    (CAPTURE_END, -1, POSITION),
    (CAPTURE_END, float('nan'), POSITION),
    (CAPTURE_END, float('inf'), POSITION),
    (CAPTURE_END, 30, None),
    (CAPTURE_END, 30, (52, 11)),
    (CAPTURE_END, 30, (91, 11, 0)),
    (CAPTURE_END, 30, (52, -181, 0)),
    (CAPTURE_END, 30, (float('nan'), 11, 0)),
    (CAPTURE_END, 30, (52, float('inf'), 0)),
    (CAPTURE_END, 30, (52, 11, float('nan'))),
    (datetime(2200, 1, 1, tzinfo=timezone.utc), 30, POSITION),
    (datetime(1970, 1, 1, tzinfo=timezone.utc), 30, POSITION),
])
def test_invalid_capture_context_keeps_existing_safeguard(monkeypatch, date, exposure, position):
    monkeypatch.setattr(ephem, 'Observer', Mock(side_effect=AssertionError('invalid context')))
    assert night_moon_weight(date, exposure, position) == 0.0


@pytest.mark.parametrize('altitude', [float('nan'), float('inf'), -float('inf')])
def test_invalid_solar_result_has_no_exemption(monkeypatch, altitude):
    solar_at(monkeypatch, altitude)
    assert night_moon_weight(CAPTURE_END, 30, POSITION) == 0.0


def test_solar_calculation_failure_retains_existing_safeguard(monkeypatch):
    solar_at(monkeypatch, -17)
    monkeypatch.setattr(ephem, 'Sun', Mock(side_effect=ValueError('unavailable ephemeris')))
    assert night_moon_weight(CAPTURE_END, 30, POSITION) == 0.0


@pytest.mark.parametrize('phase', [0, .5, 1])
@pytest.mark.parametrize('binning', [1, 2])
def test_existing_moon_halo_remains_phase_bounded_and_geometry_is_unchanged(monkeypatch, phase, binning):
    config = {
        'VIRTUALSKY': {'IMAGE_CIRCLE_DIAMETER': 400},
        'LENS_IMAGE_CIRCLE': 400,
    }
    original = deepcopy(config)
    monkeypatch.setattr(ephem, 'Moon', lambda observer: SimpleNamespace(
        alt=math.radians(45), az=0., radius=math.radians(.25), moon_phase=phase))
    before = moon_region(config, (440, 440), binning, CAPTURE_END, 30, POSITION, native=True)
    assert before is not None
    assert before[0] == pytest.approx(220)
    assert before[1] == pytest.approx(220 - 200 * math.sin(math.pi / 8) / .70710678)
    assert before[2] == pytest.approx(200 / binning * (.025 + .105 * math.sqrt(phase)))
    assert 0 < before[2] <= .13 * 200 / binning
    monkeypatch.setattr(ephem, 'Sun', lambda observer: SimpleNamespace(alt=math.radians(-17)))
    assert night_moon_weight(CAPTURE_END, 30, POSITION) == pytest.approx(.5)
    assert moon_region(config, (440, 440), binning, CAPTURE_END, 30, POSITION, native=True) == before
    assert config == original
