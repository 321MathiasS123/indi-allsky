"""Physical Moon exclusion bridges solar phases without changing control fades."""
from datetime import datetime
import math
from types import SimpleNamespace
from unittest.mock import Mock

import cv2
import ephem
import numpy as np
import pytest

from indi_allsky import highlight, highlight_output, highlight_twilight
from tests.core.test_highlight_exposure import controller
from test_highlight_protection import highlight_processor
from test_highlight_night_moon_pipeline import set_pixels
from test_highlight_twilight_output_pipeline import prepare


def scene(processor, monkeypatch):
    ref = prepare(processor, NIGHT_MOON_EXCLUSION_ENABLE=True)
    processor.night_av = [True, True]
    data = np.full((100, 100, 3), 7680, np.uint16)
    cv2.circle(data, (75, 50), 6, (65535, 65535, 65535), -1)
    set_pixels(processor, ref, data)
    monkeypatch.setattr(highlight_output, 'moon_region',
                        lambda *args, **kwargs: (75., 50., 6.5))
    monkeypatch.setattr(highlight_twilight, 'capture_moon', lambda *args: (100., 30.))
    raw = Mock(wraps=highlight.measure)
    legacy = Mock(wraps=highlight.measure_rendered)
    tail = Mock(wraps=highlight_output.output_tail)
    monkeypatch.setattr(highlight, 'measure', raw)
    monkeypatch.setattr(highlight, 'measure_rendered', legacy)
    monkeypatch.setattr(highlight_output, 'output_tail', tail)
    return ref, data, raw, legacy, tail


def solar_position(monkeypatch, altitude):
    sun = {'altitude': altitude}
    monkeypatch.setattr(ephem, 'Sun',
                        lambda observer: SimpleNamespace(alt=math.radians(sun['altitude'])))
    return sun


def night_weight(altitude):
    strength = min(1., max(0., (-altitude - 16.) / 2.))
    return strength * strength * (3. - 2. * strength)


def expected_mask(strength):
    mask = np.full((100, 100), 255, np.uint8)
    if strength:
        cv2.circle(mask, (75, 50), math.ceil(6.5 * math.sqrt(strength)), 0, -1)
    return mask != 0


def assert_masks(processor, raw, legacy, tail, expected):
    raw.reset_mock()
    legacy.reset_mock()
    tail.reset_mock()
    measured = processor.calibrate_highlights(processor.measure_highlights())
    output = processor.measure_output_highlights()
    np.testing.assert_array_equal(raw.call_args.args[1] != 0, expected)
    meter = tail if tail.called else legacy
    np.testing.assert_array_equal(meter.call_args.args[1] != 0, expected)
    return measured, output


@pytest.mark.parametrize('direction', [1, -1], ids=['dawn', 'dusk'])
def test_dense_handover_has_full_identical_masks_and_unchanged_control_weights(
        highlight_processor, monkeypatch, direction):
    p = highlight_processor
    ref, data, raw, legacy, tail = scene(p, monkeypatch)
    sun = solar_position(monkeypatch, -18.)
    originals = (ref.hdulist[0].data.copy(), ref.opencv_data.copy(),
                 p.image.copy(), p._adu_mask_dict[1].copy())
    adu = cv2.mean(cv2.cvtColor(data, cv2.COLOR_BGR2GRAY))[0] / 256
    for altitude in np.linspace(-18., -12., 121)[::direction]:
        sun['altitude'] = float(altitude)
        measured, output = assert_masks(p, raw, legacy, tail, expected_mask(1.))
        assert measured.full == measured.any == output.full == output.any == 0
        assert measured.adu == pytest.approx(adu)
        assert measured.night_moon_weight == pytest.approx(night_weight(altitude))
        assert measured.night_moon_clipped == float(night_weight(altitude) > 0)
        assert measured.twilight_weight == pytest.approx(highlight_twilight.twilight_weight(altitude))
        assert bool(tail.called) == (highlight_twilight.twilight_weight(altitude) > 0)
    for actual, original in zip((ref.hdulist[0].data, ref.opencv_data,
                                 p.image, p._adu_mask_dict[1]), originals):
        np.testing.assert_array_equal(actual, original)
    np.testing.assert_array_equal(p._highlight_output_mask != 0, originals[3] != 0)


@pytest.mark.parametrize('altitude', [-25., -18., -12., -7., -2., -1., 0., 10.])
def test_unchanged_night_twilight_interior_and_daylight_endpoints(
        highlight_processor, monkeypatch, altitude):
    p = highlight_processor
    ref, _, raw, legacy, tail = scene(p, monkeypatch)
    solar_position(monkeypatch, altitude)
    old = max(night_weight(altitude), highlight_twilight.twilight_weight(altitude))
    new = highlight_output.moon_mask_weight(ref.exp_date, ref.exposure, p.position_av)
    assert new == pytest.approx(old)
    assert_masks(p, raw, legacy, tail, expected_mask(old))


@pytest.mark.parametrize('disabled', [
    'ENABLE', 'NIGHT_MOON_EXCLUSION_ENABLE', 'TWILIGHT_ENABLE', 'TWILIGHT_OUTPUT_ENABLE',
])
@pytest.mark.parametrize('altitude', [-17., -15., -13.])
def test_each_disabled_gate_preserves_its_existing_mask_path(
        highlight_processor, monkeypatch, disabled, altitude):
    p = highlight_processor
    _, _, raw, legacy, tail = scene(p, monkeypatch)
    solar_position(monkeypatch, altitude)
    settings = p.config['HIGHLIGHT_PROTECTION']
    settings[disabled] = False
    unexpected = Mock(side_effect=AssertionError('combined physical envelope while disabled'))
    monkeypatch.setattr(highlight_output, 'moon_mask_weight', unexpected)
    strength = 0.
    if settings['ENABLE']:
        if settings['NIGHT_MOON_EXCLUSION_ENABLE']:
            strength = night_weight(altitude)
        if settings['TWILIGHT_ENABLE'] and settings['TWILIGHT_OUTPUT_ENABLE']:
            strength = max(strength, highlight_twilight.twilight_weight(altitude))
    assert_masks(p, raw, legacy, tail, expected_mask(strength))
    unexpected.assert_not_called()


@pytest.mark.parametrize('context', ['missing-date', 'missing-position', 'nonfinite-position'])
def test_invalid_capture_context_retains_unmasked_safeguards(
        highlight_processor, monkeypatch, context):
    p = highlight_processor
    ref, data, raw, legacy, tail = scene(p, monkeypatch)
    if context == 'missing-date':
        ref.exp_date = None
    else:
        p.position_av = [] if context == 'missing-position' else [float('nan'), 11., 23.]
    assert highlight_output.moon_mask_weight(ref.exp_date, ref.exposure, p.position_av) == 0
    measured, output = assert_masks(p, raw, legacy, tail, expected_mask(0.))
    assert measured.full > 1 and output.full > 1
    assert measured.night_moon_weight == measured.night_moon_clipped == 0
    assert measured.adu == pytest.approx(cv2.mean(cv2.cvtColor(data, cv2.COLOR_BGR2GRAY))[0] / 256)


@pytest.mark.parametrize('altitude', [-15., -13.])
@pytest.mark.parametrize('region', [None, (50., 50., 100.)])
def test_invalid_or_entire_image_projection_cannot_hide_raw_clipping(
        highlight_processor, monkeypatch, altitude, region):
    p = highlight_processor
    _, _, raw, legacy, tail = scene(p, monkeypatch)
    solar_position(monkeypatch, altitude)
    monkeypatch.setattr(highlight_output, 'moon_region', lambda *args, **kwargs: region)
    measured = p.measure_highlights()
    output = p.measure_output_highlights()
    assert measured.full > 1 and measured.any > 1
    np.testing.assert_array_equal(raw.call_args.args[1] != 0, expected_mask(0.))
    assert measured.night_moon_weight == measured.night_moon_clipped == 0
    assert output is None or output.any > 1


def test_outside_moon_clipping_still_shortens_exposure_in_the_former_gap(
        highlight_processor, monkeypatch):
    p = highlight_processor
    ref, data, raw, legacy, tail = scene(p, monkeypatch)
    solar_position(monkeypatch, -15.)
    data[5:25, 5:25] = 65535
    set_pixels(p, ref, data)
    measured, output = assert_masks(p, raw, legacy, tail, expected_mask(1.))
    assert measured.full > 4 and output.any > 4
    assert measured.night_moon_weight == measured.twilight_weight == 0
    obj = controller('exposure_autogain_exp_prio_db_1_10', moon=True)
    obj._expUtils.EXPOSURE_CURRENT = obj._expUtils.EXPOSURE_NEXT = 30.
    obj._expUtils.GAIN_CURRENT = obj._expUtils.GAIN_NEXT = 0.
    obj.compare_highlights(measured, 30., 0.)
    assert obj._expUtils.EXPOSURE_NEXT < 30
    assert obj._expUtils.GAIN_NEXT == 0


def test_recorded_052613_capture_context_bridges_gap_without_extending_control_fades(
        highlight_processor, monkeypatch):
    p = highlight_processor
    ref, _, raw, legacy, tail = scene(p, monkeypatch)
    # Actual receipt/exposure/location from the problematic sunrise frame;
    # the synthetic bounded Moon above isolates its physical mask handover.
    ref.exp_date = datetime.fromisoformat('2026-10-26T05:26:13.186603+01:00')
    ref.exposure = 26.137889
    assert highlight_output.night_moon_weight(ref.exp_date, ref.exposure, p.position_av) == 0
    assert highlight_twilight.capture_twilight(ref.exp_date, ref.exposure, p.position_av)[0] == 0
    measured, output = assert_masks(p, raw, legacy, tail, expected_mask(1.))
    assert measured.full == measured.any == output.full == output.any == 0
    assert measured.night_moon_weight == measured.twilight_weight == 0
