"""The optional raw meter follows photographic geometry without altering it."""
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from indi_allsky.highlight import measure
from indi_allsky import highlight_twilight
from test_highlight_protection import highlight_processor


def prepare(processor, data, **config):
    processor.config = {'HIGHLIGHT_PROTECTION': {'ENABLE': True, 'TWILIGHT_ENABLE': True},
                        'LENS_IMAGE_CIRCLE': 400, **config}
    processor.position_av = [53., 11., 10.]
    ref = SimpleNamespace(hdulist=[SimpleNamespace(data=data)], binning=1,
                          image_bitpix=16, image_bayerpat=None, exposure=30.,
                          exp_date=datetime(2026, 10, 19, 5, tzinfo=timezone.utc))
    processor.getLatestImage = lambda: ref
    mask = np.zeros(data.shape[-2:], dtype=np.uint8)
    h, w = mask.shape
    mask[h//2-20:h//2+20, w//2-20:w//2+20] = 255
    processor._adu_mask_dict = {1: mask}
    processor._detection_mask_dict = {1: None}
    processor.image = np.full((10, 10, 3), 37, dtype=np.uint8)
    return ref


def broad_horizon(shape=(460, 500), center=(250, 230), radius=200):
    y, x = np.ogrid[:shape[0], :shape[1]]
    distance = np.hypot(x-center[0], y-center[1]) / radius
    angles = np.degrees(np.arctan2(y-center[1], x-center[0]))
    data = np.full(shape, 10000, np.uint16)
    data[(distance > .95) & (distance < .985) & (angles > -50) & (angles < 50)] = 65535
    return data


@pytest.mark.parametrize('geometry', [{}, {'IMAGE_ROTATE': 'ROTATE_90_CLOCKWISE'},
                                     {'IMAGE_FLIP_V': True, 'IMAGE_FLIP_H': True},
                                     {'IMAGE_ROTATE_ANGLE': -3, 'IMAGE_ROTATE_KEEP_SIZE': True},
                                     {'IMAGE_ROTATE_ANGLE': 30, 'IMAGE_ROTATE_KEEP_SIZE': False}])
def test_active_ring_follows_geometry_and_preserves_originals(highlight_processor, monkeypatch, geometry):
    processor = highlight_processor
    raw = broad_horizon()
    prepare(processor, raw, **geometry)
    raw_before, image_before = raw.copy(), processor.image
    mask_before = processor._adu_mask_dict[1].copy()
    monkeypatch.setattr(highlight_twilight, 'capture_twilight', lambda *args: (1., 100.))
    result = processor.measure_highlights()
    assert result.twilight_peak > .99
    assert result.twilight_weight == 1 and result.twilight_time == 100
    # Ordinary metering remains central and independent of the new ring.
    assert result.full == result.any == 0
    np.testing.assert_array_equal(raw, raw_before)
    np.testing.assert_array_equal(processor._adu_mask_dict[1], mask_before)
    assert processor.image is image_before


def test_postcrop_offsets_match_visible_circle_instead_of_jpeg_midpoint(highlight_processor, monkeypatch):
    # 900x700 -> cropped372x306. Circle centre222,150 -> raw522,344.
    # Display padding is deliberately asymmetric and must not enter raw control.
    raw = broad_horizon((700, 900), (522, 344), 140)
    prepare(highlight_processor, raw, LENS_IMAGE_CIRCLE=300, LENS_OFFSET_X=36, LENS_OFFSET_Y=3,
            IMAGE_CROP_IMAGE_CIRCLE=True, IMAGE_CIRCLE_MASK={'ENABLE': True, 'DIAMETER': 280},
            IMAGE_SCALE=50, IMAGE_BORDER={'TOP': 80, 'RIGHT': 70, 'BOTTOM': 0, 'LEFT': 0})
    monkeypatch.setattr(highlight_twilight, 'capture_twilight', lambda *args: (1., 100.))
    result = highlight_processor.measure_highlights()
    assert result.twilight_peak > .99
    assert result.full == result.any == 0


def test_active_bayer_meter_keeps_raw_single_colour_clipping_and_saved_pixels(highlight_processor, monkeypatch):
    raw = broad_horizon()
    # Only one Bayer colour saturates; metering must not average it to grey.
    raw[1::2, :] = 10000
    raw[::2, 1::2] = 10000
    ref = prepare(highlight_processor, raw)
    ref.image_bayerpat = 'RGGB'
    before = raw.copy()
    monkeypatch.setattr(highlight_twilight, 'capture_twilight', lambda *args: (1., 100.))
    assert highlight_processor.measure_highlights().twilight_peak > .99
    np.testing.assert_array_equal(raw, before)


def test_active_binned_eight_bit_meter_uses_native_geometry_and_full_scale(highlight_processor, monkeypatch):
    raw = (broad_horizon((230, 250), (125, 115), 100) >> 8).astype(np.uint8)
    ref = prepare(highlight_processor, raw)
    ref.binning, ref.image_bitpix = 2, 8
    highlight_processor.max_bit_depth = 8
    highlight_processor._adu_mask_dict[2] = highlight_processor._adu_mask_dict.pop(1)
    monkeypatch.setattr(highlight_twilight, 'capture_twilight', lambda *args: (1., 100.))
    assert highlight_processor.measure_highlights().twilight_peak > .99


@pytest.mark.parametrize('blocked', [False, True])
def test_configured_detection_mask_is_honoured_separately_from_adu_roi(highlight_processor, monkeypatch, blocked):
    raw = broad_horizon()
    prepare(highlight_processor, raw, DETECT_MASK='configured-sky-mask.png',
            IMAGE_ROTATE_ANGLE=-3, IMAGE_ROTATE_KEEP_SIZE=True)
    mask = np.ones(raw.shape, np.uint8) * 255
    if blocked:
        mask[:, raw.shape[1]//2+90:] = 0
    highlight_processor._detection_mask_dict = {1: mask}
    original = mask.copy()
    monkeypatch.setattr(highlight_twilight, 'capture_twilight', lambda *args: (1., 100.))
    result = highlight_processor.measure_highlights()
    if blocked:
        assert result.twilight_peak == pytest.approx(10000/65535)
    else:
        assert result.twilight_peak > .99
    np.testing.assert_array_equal(mask, original)


@pytest.mark.parametrize('parent,option', [(False, True), (True, False), (False, False)])
def test_disabled_path_does_not_run_optional_geometry(highlight_processor, monkeypatch, parent, option):
    raw = broad_horizon()
    prepare(highlight_processor, raw, HIGHLIGHT_PROTECTION={'ENABLE': parent, 'TWILIGHT_ENABLE': option})
    monkeypatch.setattr(highlight_twilight, 'capture_twilight', Mock(side_effect=AssertionError('inactive path')))
    expected = measure(raw, highlight_processor._adu_mask_dict[1], 16)
    assert highlight_processor.measure_highlights() == expected


def test_outside_solar_window_only_retains_midpoint_for_smooth_release(highlight_processor, monkeypatch):
    raw = broad_horizon()
    prepare(highlight_processor, raw)
    monkeypatch.setattr(highlight_twilight, 'capture_twilight', lambda *args: (0., 100.))
    monkeypatch.setattr(highlight_twilight, 'measure_ring', Mock(side_effect=AssertionError('inactive path')))
    expected = measure(raw, highlight_processor._adu_mask_dict[1], 16)._replace(twilight_time=100.)
    assert highlight_processor.measure_highlights() == expected


def test_valid_black_ring_retains_solar_strength_for_smooth_recovery(highlight_processor, monkeypatch):
    raw = np.zeros((460, 500), dtype=np.uint16)
    prepare(highlight_processor, raw)
    monkeypatch.setattr(highlight_twilight, 'capture_twilight', lambda *args: (1., 100.))
    result = highlight_processor.measure_highlights()
    assert result.twilight_peak == 0
    assert result.twilight_weight == 1 and result.twilight_time == 100


def test_invalid_ring_geometry_keeps_whole_image_prediction_available(highlight_processor, monkeypatch):
    raw = broad_horizon()
    raw[224:236, 244:256] = 61000
    prepare(highlight_processor, raw, LENS_IMAGE_CIRCLE=0)
    monkeypatch.setattr(highlight_twilight, 'capture_twilight', lambda *args: (1., 100.))
    expected = measure(raw, highlight_processor._adu_mask_dict[1], 16)
    result = highlight_processor.measure_highlights()
    assert expected.any == 0 and expected.any_next > 1.3
    assert result == expected._replace(twilight_weight=1., twilight_time=100., twilight_quantum=1 / 65535)


def test_moon_budget_uses_capture_context_instead_of_delayed_renderer_state(highlight_processor, monkeypatch):
    raw = broad_horizon()
    ref = prepare(highlight_processor, raw)
    stale = {'moon_phase': 1, 'moon_alt': -60, 'sun_alt': 30}
    highlight_processor.astrometric_data = stale.copy()
    moon = Mock(return_value=(77., -.5))
    budget = Mock(return_value=.012)
    meter = Mock(return_value=.8)
    monkeypatch.setattr(highlight_twilight, 'capture_twilight', lambda *args: (1., 100.))
    monkeypatch.setattr(highlight_twilight, 'capture_moon', moon)
    monkeypatch.setattr(highlight_twilight, 'clipping_budget', budget)
    monkeypatch.setattr(highlight_twilight, 'measure_ring', meter)
    result = highlight_processor.measure_highlights()
    moon.assert_called_once_with(ref.exp_date, ref.exposure, highlight_processor.position_av)
    budget.assert_called_once_with(77., -.5, highlight_processor.config)
    assert meter.call_args.kwargs['clipping_fraction'] == .012
    assert result.twilight_peak == .8 and result.twilight_time == 100
    assert highlight_processor.astrometric_data == stale


def test_solar_context_uses_exposure_midpoint_without_mutating_metadata(highlight_processor):
    raw = broad_horizon()
    ref = prepare(highlight_processor, raw)
    highlight_processor.astrometric_data = {'sun_alt': 50.0, 'moon_alt': 10.0}
    expected_weight, expected_time = highlight_twilight.capture_twilight(ref.exp_date, ref.exposure,
                                                                       highlight_processor.position_av)
    assert expected_weight == 1
    result = highlight_processor.measure_highlights()
    assert result.twilight_time == ref.exp_date.timestamp() - 15 == expected_time
    assert result.twilight_weight == expected_weight
    assert highlight_processor.astrometric_data == {'sun_alt': 50.0, 'moon_alt': 10.0}


def test_capture_solar_envelope_is_calculated_at_midpoint(monkeypatch):
    import ephem
    seen = []
    monkeypatch.setattr(ephem, 'Sun', lambda observer: seen.append(observer.date.datetime())
                        or SimpleNamespace(alt=np.radians(-13)))
    end = datetime(2026, 10, 19, 5, 0, 30, tzinfo=timezone.utc)
    weight, timestamp = highlight_twilight.capture_twilight(end, 30, [53, 11, 10])
    assert weight == .5
    assert timestamp == end.timestamp() - 15
    assert abs((seen[0] - datetime(2026, 10, 19, 5, 0, 15)).total_seconds()) < .0001


def test_capture_moon_matches_ephemeris_at_capture_midpoint():
    import ephem
    end = datetime(2026, 10, 19, 5, 0, 30, tzinfo=timezone.utc)
    phase, altitude = highlight_twilight.capture_moon(end, 30, [53, 11, 10])
    observer = ephem.Observer()
    observer.lat, observer.lon = '53', '11'
    observer.elevation, observer.pressure = 10, 0
    observer.date = end - timedelta(seconds=15)
    moon = ephem.Moon(observer)
    assert phase == pytest.approx(moon.moon_phase * 100)
    assert altitude == pytest.approx(np.degrees(moon.alt))


def test_capture_moon_receives_midpoint_without_atmospheric_refraction(monkeypatch):
    import ephem
    seen = []
    monkeypatch.setattr(ephem, 'Moon', lambda observer: seen.append((observer.date.datetime(), observer.pressure))
                        or SimpleNamespace(moon_phase=.5, alt=np.radians(-.5)))
    end = datetime(2026, 10, 19, 5, 0, 30, tzinfo=timezone.utc)
    assert highlight_twilight.capture_moon(end, 30, [53, 11, 10]) == pytest.approx((50, -.5))
    assert abs((seen[0][0] - datetime(2026, 10, 19, 5, 0, 15)).total_seconds()) < .0001
    assert seen[0][1] == 0


@pytest.mark.parametrize('position', [None, [], [float('nan'), 11, 10], [91, 11, 10], [53, 181, 10]])
def test_invalid_capture_position_disables_guard(position):
    assert highlight_twilight.capture_twilight(datetime.now(), 30, position) == (0., 0.)
    assert highlight_twilight.capture_moon(datetime.now(), 30, position) == (0., -90.)


@pytest.mark.parametrize('exposure', [None, float('nan'), float('inf'), 'bad'])
def test_invalid_capture_exposure_has_sane_ephemeris_fallback(exposure):
    end = datetime(2026, 10, 19, 5, tzinfo=timezone.utc)
    assert highlight_twilight.capture_twilight(end, exposure, [53, 11, 10]) == (0., 0.)
    assert highlight_twilight.capture_moon(end, exposure, [53, 11, 10]) == (0., -90.)


def test_moon_ephemeris_value_error_has_sane_fallback(monkeypatch):
    import ephem
    monkeypatch.setattr(ephem, 'Moon', Mock(side_effect=ValueError('invalid ephemeris input')))
    end = datetime(2026, 10, 19, 5, tzinfo=timezone.utc)
    assert highlight_twilight.capture_moon(end, 30, [53, 11, 10]) == (0., -90.)
