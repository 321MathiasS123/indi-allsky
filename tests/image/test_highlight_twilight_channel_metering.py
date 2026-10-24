"""Twilight colour trends use uncensored calibrated data inside the metering mask."""
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import Mock

import cv2
import numpy as np
import pytest

from indi_allsky import highlight_twilight
from indi_allsky.highlight import HighlightMeasurement
from test_highlight_protection import highlight_processor


SETTINGS = {'ENABLE': True, 'TWILIGHT_ENABLE': True, 'ANY_TARGET': 1.1, 'ANY_DEV': .2}
FIELDS = ('twilight_blue', 'twilight_green', 'twilight_red')
RAW_FIELDS = tuple(field + '_raw' for field in FIELDS)


def prepare(processor, raw, calibrated=None, settings=None):
    processor.config = {'LENS_IMAGE_CIRCLE': 0,
                        'HIGHLIGHT_PROTECTION': dict(SETTINGS if settings is None else settings)}
    processor.position_av = [53., 11., 10.]
    processor.max_bit_depth = np.iinfo(raw.dtype).bits
    native = np.moveaxis(raw[:, :, ::-1], -1, 0).copy() if raw.ndim == 3 else raw.copy()
    ref = SimpleNamespace(hdulist=[SimpleNamespace(data=native)], binning=1,
                          image_bitpix=processor.max_bit_depth, image_bayerpat=None,
                          exposure=20., exp_date=datetime(2026, 10, 24, 5, tzinfo=timezone.utc),
                          opencv_data=raw.copy() if calibrated is None else calibrated)
    processor.getLatestImage = lambda: ref
    processor._adu_mask_dict = {1: np.ones(raw.shape[:2], dtype=np.uint8) * 255}
    processor._detection_mask_dict = {1: None}
    processor.image = np.zeros((10, 10, 3), dtype=np.uint8)
    return ref


def levels(measurement):
    return tuple(getattr(measurement, field) for field in FIELDS)


def raw_levels(measurement):
    return tuple(getattr(measurement, field) for field in RAW_FIELDS)


@pytest.mark.parametrize('dtype,depth', [(np.uint8, 8), (np.uint16, 12), (np.uint16, 16)])
def test_channel_histograms_follow_native_range_and_bgr_order(dtype, depth):
    maximum = (1 << depth) - 1
    data = np.empty((40, 50, 3), dtype=dtype)
    data[:] = [maximum // 4, maximum // 2, maximum * 3 // 4]
    original = data.copy()
    mask = np.ones(data.shape[:2], np.uint8)
    expected = tuple(float(value) / maximum for value in data[0, 0])
    assert highlight_twilight.measure_channels(data, mask, depth, {'HIGHLIGHT_PROTECTION': SETTINGS}) == expected
    np.testing.assert_array_equal(data, original)


@pytest.mark.parametrize('dtype,depth', [(np.uint8, 8), (np.uint16, 12), (np.uint16, 16)])
def test_pipeline_carries_actual_code_resolution_through_calibration(highlight_processor, monkeypatch, dtype, depth):
    maximum = (1 << depth) - 1
    raw = np.empty((100, 100, 3), dtype)
    raw[:] = [maximum // 2, maximum // 3, maximum // 4]
    prepare(highlight_processor, raw)
    highlight_processor.max_bit_depth = depth
    monkeypatch.setattr(highlight_twilight, 'capture_twilight', lambda *args: (1., 100.))
    measured = highlight_processor.measure_highlights()
    calibrated = highlight_processor.calibrate_highlights(measured)
    assert measured.twilight_quantum == calibrated.twilight_quantum == 1 / maximum
    assert raw_levels(calibrated) == pytest.approx(raw[0, 0] / maximum)


def test_tiny_stars_and_masked_bright_sources_do_not_dominate_channel_trends():
    data = np.full((100, 100, 3), 10000, np.uint16)
    mask = np.zeros(data.shape[:2], np.uint8)
    mask[20:80, 20:80] = 255
    data[:20] = 65535
    data[30:33, 30:33] = 65535
    assert highlight_twilight.measure_channels(data, mask, 16, {'HIGHLIGHT_PROTECTION': SETTINGS}) == (10000 / 65535,) * 3


def test_tail_area_comes_from_the_existing_any_channel_upper_limit():
    data = np.full((100, 100, 3), 10000, np.uint16)
    data[:2, :75, 0] = 60000  # 1.5% blue support.
    mask = np.ones(data.shape[:2], np.uint8)
    narrow = highlight_twilight.measure_channels(data, mask, 16, {'HIGHLIGHT_PROTECTION': SETTINGS})
    broad = highlight_twilight.measure_channels(data, mask, 16, {'HIGHLIGHT_PROTECTION': {'ANY_TARGET': 2., 'ANY_DEV': .4}})
    assert narrow == (60000 / 65535, 10000 / 65535, 10000 / 65535)
    assert broad == (10000 / 65535,) * 3


@pytest.mark.parametrize('invalid', ['mono', 'channels', 'dtype', 'mask_shape', 'mask_nan', 'empty', 'depth', 'fraction', 'range'])
def test_unavailable_statistics_are_finite_zeroes(invalid):
    data = np.full((20, 20, 3), 1000, np.uint16)
    mask = np.ones(data.shape[:2], np.uint8)
    depth, settings = 16, dict(SETTINGS)
    if invalid == 'mono':
        data = data[:, :, 0]
    elif invalid == 'channels':
        data = data[:, :, :2]
    elif invalid == 'dtype':
        data = data.astype(np.float32)
    elif invalid == 'mask_shape':
        mask = mask[:-1]
    elif invalid == 'mask_nan':
        mask = mask.astype(float)
        mask[0, 0] = np.nan
    elif invalid == 'empty':
        mask[:] = 0
    elif invalid == 'depth':
        depth = 17
    elif invalid == 'fraction':
        settings['ANY_TARGET'] = float('inf')
    elif invalid == 'range':
        depth = 8
    assert highlight_twilight.measure_channels(data, mask, depth, {'HIGHLIGHT_PROTECTION': settings}) == (0., 0., 0.)


def test_raw_censor_certificate_survives_dark_subtraction(highlight_processor, monkeypatch):
    raw = np.empty((100, 100, 3), np.uint16)
    raw[:] = [65534, 55000, 30000]
    calibrated = (raw - 3000).astype(np.uint16)
    ref = prepare(highlight_processor, raw, calibrated)
    original = ref.hdulist[0].data.copy()
    monkeypatch.setattr(highlight_twilight, 'capture_twilight', lambda *args: (1., 100.))
    measured = highlight_processor.measure_highlights()
    assert levels(measured) == (0., 55000 / 65535, 30000 / 65535)
    assert raw_levels(measured) == levels(measured)
    result = highlight_processor.calibrate_highlights(measured)
    assert levels(result) == (0., 52000 / 65535, 27000 / 65535)
    assert raw_levels(result) == raw_levels(measured)
    assert result.any == measured.any == 100
    assert result.adu == pytest.approx(cv2.mean(cv2.cvtColor(calibrated, cv2.COLOR_BGR2GRAY))[0] / 256)
    np.testing.assert_array_equal(ref.hdulist[0].data, original)
    np.testing.assert_array_equal(ref.opencv_data, calibrated)


def test_full_code_threshold_does_not_certify_recorded_saturation_plateaus(highlight_processor, monkeypatch):
    raw = np.empty((100, 100, 3), np.uint16)
    raw[:] = [65534, 65520, 65534]
    settings = dict(SETTINGS, THRESHOLD=100.)
    prepare(highlight_processor, raw, raw - 5000, settings=settings)
    monkeypatch.setattr(highlight_twilight, 'capture_twilight', lambda *args: (1., 100.))
    measured = highlight_processor.measure_highlights()
    calibrated = highlight_processor.calibrate_highlights(measured)
    assert levels(measured) == raw_levels(measured) == (0., 0., 0.)
    assert levels(calibrated) == raw_levels(calibrated) == (0., 0., 0.)
    # The optional trend certificate must not change the requested raw policy.
    assert measured.any == measured.full == 0
    assert measured.any_next == measured.full_next == 100


def test_calibrated_channel_signal_does_not_invent_growth_from_dark_bias(highlight_processor, monkeypatch):
    scene = np.empty((100, 100, 3), np.uint16)
    scene[:] = [45000, 35000, 25000]
    results = []
    monkeypatch.setattr(highlight_twilight, 'capture_twilight', lambda *args: (1., 100.))
    for signal in (1., .9):
        calibrated = (scene * signal).astype(np.uint16)
        prepare(highlight_processor, calibrated + 5000, calibrated)
        measured = highlight_processor.measure_highlights()
        results.append(levels(highlight_processor.calibrate_highlights(measured)))
    assert np.array(results[1]) / np.array(results[0]) == pytest.approx([.9, .9, .9])


def test_dark_offset_does_not_hide_the_raw_channel_headroom(highlight_processor, monkeypatch):
    raw = np.empty((100, 100, 3), np.uint16)
    raw[:] = [round(.92 * 65535), 53000, 29000]
    calibrated = raw - round(.15 * 65535)
    prepare(highlight_processor, raw, calibrated)
    monkeypatch.setattr(highlight_twilight, 'capture_twilight', lambda *args: (1., 100.))
    result = highlight_processor.calibrate_highlights(highlight_processor.measure_highlights())
    assert result.twilight_blue == pytest.approx(.77, abs=1 / 65535)
    assert result.twilight_blue_raw == pytest.approx(.92, abs=1 / 65535)
    assert raw_levels(result) == pytest.approx(raw[0, 0] / 65535)


@pytest.mark.parametrize('inactive', ['parent', 'option', 'day', 'night'])
def test_ordinary_paths_do_not_compute_extra_channel_histograms(highlight_processor, monkeypatch, inactive):
    raw = np.full((100, 100, 3), 10000, np.uint16)
    settings = dict(SETTINGS)
    if inactive == 'parent':
        settings['ENABLE'] = False
    elif inactive == 'option':
        settings['TWILIGHT_ENABLE'] = False
    prepare(highlight_processor, raw, settings=settings)
    monkeypatch.setattr(highlight_twilight, 'capture_twilight', lambda *args: (0., 100.))
    histogram = Mock(side_effect=AssertionError('ordinary path'))
    monkeypatch.setattr(highlight_twilight, 'measure_channels', histogram)
    measured = highlight_processor.measure_highlights()
    calibrated = highlight_processor.calibrate_highlights(measured)
    assert levels(measured) == levels(calibrated) == (0., 0., 0.)
    assert raw_levels(measured) == raw_levels(calibrated) == (0., 0., 0.)
    assert measured.twilight_quantum == calibrated.twilight_quantum == 0
    histogram.assert_not_called()


def test_display_grayscale_does_not_discard_the_calibrated_colour_trend(highlight_processor, monkeypatch):
    raw = np.full((100, 100), 10000, np.uint16)
    raw[::2, ::2] = 20000
    raw[1::2, 1::2] = 40000
    ref = prepare(highlight_processor, raw)
    ref.image_bayerpat = 'RGGB'
    ref.opencv_data = cv2.cvtColor(raw, cv2.COLOR_BAYER_BG2GRAY)
    monkeypatch.setattr(highlight_twilight, 'capture_twilight', lambda *args: (1., 100.))
    result = highlight_processor.calibrate_highlights(highlight_processor.measure_highlights())
    assert levels(result) == pytest.approx((40000 / 65535, 10000 / 65535, 20000 / 65535))
    assert raw_levels(result) == pytest.approx(levels(result))


def test_legacy_positional_measurements_retain_zero_channel_defaults():
    assert levels(HighlightMeasurement(0, 0, 70)) == (0., 0., 0.)
    assert raw_levels(HighlightMeasurement(0, 0, 70)) == (0., 0., 0.)
    assert HighlightMeasurement(0, 0, 70).twilight_quantum == 0
    assert all(np.isfinite(value) for value in HighlightMeasurement(0, 0, 70))
