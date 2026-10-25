"""Optional output twilight protection keeps capture and display masks separate."""
from copy import deepcopy
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import Mock

import cv2
import numpy as np
import pytest

from indi_allsky import highlight, highlight_output, highlight_twilight
from test_highlight_protection import highlight_processor


def prepare(processor, **settings):
    data = np.full((100, 100, 3), 30000, np.uint16)
    cv2.circle(data, (75, 50), 12, (65535, 65535, 65535), -1)
    data[5:15, 5:15] = 65535  # distant horizon remains included
    processor.config = {'LENS_IMAGE_CIRCLE': 100, 'HIGHLIGHT_PROTECTION': {
        'ENABLE': True, 'TWILIGHT_ENABLE': True, 'TWILIGHT_OUTPUT_ENABLE': True,
        'ANY_TARGET': 1.1, 'ANY_DEV': .2, 'TWILIGHT_STAR_ALLOWANCE': .1,
        **settings}}
    ref = SimpleNamespace(hdulist=[SimpleNamespace(data=np.moveaxis(data[:, :, ::-1], -1, 0))],
                          opencv_data=data.copy(), binning=1, image_bitpix=16,
                          image_bayerpat=None, camera_uuid='test-camera', exposure=30.,
                          exp_date=datetime(2026, 10, 25, 17, tzinfo=timezone.utc))
    processor.getLatestImage = lambda: ref
    processor.position_av = [52.854, 11.148, 23]
    processor._adu_mask_dict = {1: np.ones((100, 100), np.uint8)*255}
    processor._detection_mask_dict = {1: None}
    processor.image = (data >> 8).astype(np.uint8)
    return ref


def setup_moon(monkeypatch, weight=1.):
    solar = Mock(return_value=(weight, 100.))
    moon = Mock(return_value=(75., 50., 15.))
    monkeypatch.setattr(highlight_twilight, 'capture_twilight', solar)
    monkeypatch.setattr(highlight_twilight, 'capture_moon', lambda *args: (100., 20.))
    monkeypatch.setattr(highlight_output, 'moon_region', moon)
    return solar, moon


def test_moon_mask_is_reused_for_global_and_colour_metrics_but_not_calibrated_adu(highlight_processor, monkeypatch):
    p = highlight_processor
    ref = prepare(p)
    setup_moon(monkeypatch)
    original_mask = p._adu_mask_dict[1].copy()
    original_source, original_image = ref.hdulist[0].data.copy(), p.image.copy()
    meter = Mock(wraps=highlight.measure)
    channels = Mock(wraps=highlight_twilight.measure_channels)
    monkeypatch.setattr(highlight, 'measure', meter)
    monkeypatch.setattr(highlight_twilight, 'measure_channels', channels)
    measured = p.measure_highlights()
    assert meter.call_count == 1
    assert measured.full == measured.any == pytest.approx(10000/9291)
    assert p._highlight_twilight_mask is not p._adu_mask_dict[1]
    assert channels.call_args.args[1] is p._highlight_twilight_mask
    assert measured.twilight_blue_raw == pytest.approx(30000/65535)
    ref.opencv_data = np.maximum(ref.opencv_data.astype(np.int32)-2000, 0).astype(np.uint16)
    calibrated = p.calibrate_highlights(measured)
    assert channels.call_count == 2
    assert channels.call_args.args[1] is p._highlight_twilight_mask
    assert calibrated.twilight_blue == pytest.approx(28000/65535)
    expected_adu = cv2.mean(cv2.cvtColor(ref.opencv_data, cv2.COLOR_BGR2GRAY), mask=original_mask)[0]/256
    assert calibrated.adu == pytest.approx(expected_adu)
    assert calibrated.full == measured.full and calibrated.any == measured.any
    np.testing.assert_array_equal(p._adu_mask_dict[1], original_mask)
    np.testing.assert_array_equal(ref.hdulist[0].data, original_source)
    np.testing.assert_array_equal(p.image, original_image)


def test_new_option_keeps_existing_ring_phase_measurement_unchanged(highlight_processor, monkeypatch):
    p = highlight_processor
    prepare(p, TWILIGHT_OUTPUT_ENABLE=False)
    setup_moon(monkeypatch)
    old = p.measure_highlights()
    p.config['HIGHLIGHT_PROTECTION']['TWILIGHT_OUTPUT_ENABLE'] = True
    new = p.measure_highlights()
    assert new.full < old.full and new.any < old.any
    assert new.twilight_peak == old.twilight_peak
    assert new.twilight_weight == old.twilight_weight and new.twilight_time == old.twilight_time


@pytest.mark.parametrize('sun_alt', [-14., -13., -12., -2., -1., 0.])
def test_moon_exception_area_fades_with_solar_weight_in_raw_and_output(highlight_processor, monkeypatch, sun_alt):
    p = highlight_processor
    prepare(p)
    weight = highlight_twilight.twilight_weight(sun_alt)
    solar, moon = setup_moon(monkeypatch, weight)
    source = deepcopy(p._adu_mask_dict[1])
    raw = p.measure_highlights()
    assert solar.call_count == 1  # early weight is reused by the ring path
    if not weight:
        assert moon.call_count == 0
        assert p._highlight_twilight_mask is None
        assert raw == highlight.measure(np.moveaxis(p.getLatestImage().hdulist[0].data, 0, -1), source, 16)._replace(twilight_time=100.)
        assert p.measure_output_highlights() == highlight.measure_rendered(p.image, source)
        return
    expected = source.copy()
    cv2.circle(expected, (75, 50), int(np.ceil(15*np.sqrt(weight))), 0, -1)
    np.testing.assert_array_equal(p._highlight_twilight_mask, expected)
    output_meter = Mock(wraps=highlight_output.output_tail)
    monkeypatch.setattr(highlight_output, 'output_tail', output_meter)
    result = p.measure_output_highlights()
    np.testing.assert_array_equal(output_meter.call_args.args[1] != 0, expected != 0)
    assert result.twilight_weight == weight and result.twilight_time == 100
    assert result.full == result.adu == 0 and result.any > 0
    np.testing.assert_array_equal(p._adu_mask_dict[1], source)
    # Mask caching is static; the moving Moon must not alter the cached ROI.
    np.testing.assert_array_equal(p._highlight_output_mask != 0, source != 0)


@pytest.mark.parametrize('settings', [
    {'ENABLE': False}, {'TWILIGHT_ENABLE': False}, {'TWILIGHT_OUTPUT_ENABLE': False},
])
def test_switching_option_off_restores_existing_raw_and_output_paths(highlight_processor, monkeypatch, settings):
    p = highlight_processor
    ref = prepare(p, **settings)
    setup_moon(monkeypatch)
    moon = Mock(side_effect=AssertionError('optional Moon work while disabled'))
    monkeypatch.setattr(highlight_output, 'moon_region', moon)
    monkeypatch.setattr(highlight_output, 'output_tail', Mock(side_effect=AssertionError('optional output work while disabled')))
    p._highlight_twilight_mask = np.zeros((100, 100), np.uint8)  # previous frame's state
    raw = p.measure_highlights()
    expected = highlight.measure(np.moveaxis(ref.hdulist[0].data, 0, -1), p._adu_mask_dict[1], 16)
    assert raw.full == expected.full and raw.any == expected.any
    assert p._highlight_twilight_mask is None
    assert p.measure_output_highlights() == highlight.measure_rendered(p.image, p._adu_mask_dict[1])


@pytest.mark.parametrize('weight', [0., 1.])
def test_no_usable_projection_keeps_all_raw_pixels(highlight_processor, monkeypatch, weight):
    p = highlight_processor
    ref = prepare(p)
    setup_moon(monkeypatch, weight)
    monkeypatch.setattr(highlight_output, 'moon_region', lambda *args, **kwargs: None)
    result = p.measure_highlights()
    expected = highlight.measure(np.moveaxis(ref.hdulist[0].data, 0, -1), p._adu_mask_dict[1], 16)
    assert result.full == expected.full and result.any == expected.any
    assert p._highlight_twilight_mask is None


def test_empty_moon_mask_cannot_remove_the_entire_raw_safeguard(highlight_processor, monkeypatch):
    p = highlight_processor
    ref = prepare(p)
    setup_moon(monkeypatch)
    monkeypatch.setattr(highlight_output, 'moon_region', lambda *args, **kwargs: (50., 50., 100.))
    result = p.measure_highlights()
    expected = highlight.measure(np.moveaxis(ref.hdulist[0].data, 0, -1), p._adu_mask_dict[1], 16)
    assert result.full == expected.full and result.any == expected.any
    assert p._highlight_twilight_mask is None
    assert p.measure_output_highlights() is None


def test_output_target_and_small_source_margin_are_passed_without_changing_pixels(highlight_processor, monkeypatch):
    p = highlight_processor
    prepare(p, TWILIGHT_OUTPUT_TARGET=254, TWILIGHT_STAR_ALLOWANCE=.2)
    setup_moon(monkeypatch)
    output = Mock(return_value=(253/255, .004))
    monkeypatch.setattr(highlight_output, 'output_tail', output)
    before = p.image.copy()
    result = p.measure_output_highlights()
    assert output.call_args.kwargs == {'star_allowance': .002, 'target': 254}
    assert result.twilight_peak == 253/255 and result.any == .4
    np.testing.assert_array_equal(p.image, before)


def test_output_geometry_uses_native_source_mask_shape_and_preborder_moon_position(highlight_processor, monkeypatch):
    p = highlight_processor
    ref = prepare(p)
    p.config.update(IMAGE_CROP_IMAGE_CIRCLE=True, LENS_IMAGE_CIRCLE=60,
                    LENS_OFFSET_X=4, LENS_OFFSET_Y=3,
                    IMAGE_BORDER={'TOP': 80, 'LEFT': 0, 'RIGHT': 70, 'BOTTOM': 0})
    _, moon = setup_moon(monkeypatch)
    p.crop_image()
    output = Mock(return_value=(1., .01))
    monkeypatch.setattr(highlight_output, 'output_tail', output)
    p.measure_output_highlights()
    moon.assert_called_once_with(p.config, (100, 100), 1, ref.exp_date, ref.exposure,
                                 p.position_av, 'test-camera')
    assert output.call_args.args[1].shape == p.image.shape[:2] == (66, 68)
