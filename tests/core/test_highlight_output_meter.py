"""Twilight display measurement and Moon geometry without camera services."""
import ast
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import logging
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from indi_allsky.highlight_output import _geometry, moon_region, output_tail


@pytest.fixture
def config():
    return {
        'IMAGE_ROTATE': '', 'IMAGE_ROTATE_ANGLE': -3, 'IMAGE_ROTATE_KEEP_SIZE': True,
        'IMAGE_FLIP_H': True, 'IMAGE_FLIP_V': True, 'IMAGE_CROP_IMAGE_CIRCLE': True,
        'IMAGE_SCALE': 100, 'IMAGE_BORDER': {'TOP': 80, 'LEFT': 0, 'RIGHT': 70, 'BOTTOM': 0},
        'LENS_IMAGE_CIRCLE': 2265, 'LENS_OFFSET_X': 36, 'LENS_OFFSET_Y': 3,
        'LENS_AZIMUTH': 359.6, 'LENS_ALTITUDE': 87.47,
        'IMAGE_CIRCLE_MASK': {'ENABLE': True, 'DIAMETER': 2200},
        'VIRTUALSKY': {'IMAGE_CIRCLE_DIAMETER': 2298, 'OFFSET_X': -6, 'OFFSET_Y': -37,
                       'POINTING_AZIMUTH': 170.55, 'LATITUDE_OFFSET': 0., 'LONGITUDE_OFFSET': 0.,
                       'RADIAL_DISTORTION': .078567, 'FLIP_H': False, 'FLIP_V': False,
                       'CALIBRATION_ENABLED': False, 'CALIBRATION': None},
    }


def test_tail_uses_worst_colour_at_every_pixel_and_has_one_shared_allowance():
    image = np.full((100, 100, 3), 100, np.uint8)
    image[:5, :10, 0] = 255
    image[5:10, :10, 1] = 255
    image[10:15, :10, 2] = 255
    before = image.copy()
    peak, over = output_tail(image, star_allowance=.001)
    assert peak == 1
    assert over == pytest.approx((150-10)/10000)
    np.testing.assert_array_equal(image, before)


@pytest.mark.parametrize('stars', [0, 1, 9, 10])
def test_tiny_stars_and_transients_cannot_drive_the_tail(stars):
    image = np.full((100, 100, 3), 200, np.uint8)
    image.reshape(-1, 3)[:stars] = 255
    assert output_tail(image) == pytest.approx((200/255, 0))


def test_broad_flat_plateau_retains_area_error_when_tail_is_censored():
    image = np.full((100, 100), 100, np.uint8)
    image[:20] = 255
    assert output_tail(image) == pytest.approx((1, .199))


def test_target_selects_only_codes_strictly_above_the_output_goal():
    image = np.array([[251, 252, 253, 254, 255]], np.uint8)
    assert output_tail(image, star_allowance=0, target=252) == pytest.approx((1, .6))
    assert output_tail(image, star_allowance=0, target=254) == pytest.approx((1, .2))


def test_mask_excludes_moon_and_uses_remaining_sky_denominator_without_mutation():
    image = np.full((100, 100, 3), 100, np.uint8)
    image[:50] = 255
    image[50:60, :10, 2] = 254
    mask = np.ones((100, 100), np.uint8)*255
    mask[:50] = 0
    before = mask.copy()
    assert output_tail(image, mask) == pytest.approx((254/255, 95/5000))
    np.testing.assert_array_equal(mask, before)


@pytest.mark.parametrize('data', [None, np.zeros((0, 10), np.uint8),
                                  np.zeros((10, 10), np.uint16),
                                  np.zeros((10, 10, 4), np.uint8)])
def test_invalid_image_has_no_optional_measurement(data):
    assert output_tail(data) == (0, 0)


@pytest.mark.parametrize('mask', [np.zeros((10, 10), np.uint8), np.ones((9, 10), np.uint8),
                                  np.full((10, 10), np.nan)])
def test_invalid_or_empty_mask_has_no_optional_measurement(mask):
    assert output_tail(np.full((10, 10, 3), 255, np.uint8), mask) == (0, 0)


@pytest.mark.parametrize('allowance', [-1, .02, float('nan'), float('inf'), 'bad'])
def test_invalid_star_allowance_does_not_disable_raw_safeguard(allowance):
    assert output_tail(np.ones((10, 10), np.uint8), star_allowance=allowance) == (0, 0)


@pytest.mark.parametrize('target', [-1, 255, 252.5, float('nan'), float('inf'), 'bad'])
def test_invalid_target_has_no_optional_measurement(target):
    assert output_tail(np.ones((10, 10), np.uint8), target=target) == (0, 0)


def test_recorded_october25_moon_matches_native_and_delivered_positions(config):
    # Independent recorded anchors: native ANY>=99% centroid2681.85,1778.09;
    # delivered JPEG>=250 centroid223.19,1168.90. Bloom need not be symmetric.
    end = datetime(2026, 10, 25, 18, 0, 56, tzinfo=timezone(timedelta(hours=1)))
    original = deepcopy(config)
    native = moon_region(config, (3552, 3552), 1, end, 21.56768, [52.854, 11.148, 23], native=True)
    display = moon_region(config, (3552, 3552), 1, end, 21.56768, [52.854, 11.148, 23])
    assert native[:2] == pytest.approx((2681.85, 1778.09), abs=8)
    assert (display[0], display[1]+80) == pytest.approx((223.19, 1168.90), abs=8)
    assert native[2] == display[2] <= 143
    # Tonight's widest nativeMoon component reaches130.8px from this position.
    assert display[2] > 131
    # Its distant western horizon stays protected.
    assert np.hypot(native[0]-682, native[1]-1518) > native[2]*10
    assert config == original


@pytest.mark.parametrize('position', [None, [], [float('nan'), 11, 23], [91, 11, 23], [53, 181, 23]])
def test_invalid_position_has_no_moon_exception(config, position):
    assert moon_region(config, (3552, 3552), 1, datetime.now(), 30, position) is None


@pytest.mark.parametrize('exposure', [None, -1, float('nan'), float('inf'), 'bad'])
def test_invalid_capture_has_no_moon_exception(config, exposure):
    assert moon_region(config, (3552, 3552), 1, datetime.now(), exposure, [53, 11, 23]) is None


@pytest.mark.parametrize('patch', [{}, {'IMAGE_CIRCLE_DIAMETER': 0}, {'RADIAL_DISTORTION': 1.1},
                                   {'FLIP_H': 'false'}, {'CALIBRATION_ENABLED': True, 'CALIBRATION': {'stale': True}}])
def test_unusable_alignment_retains_unexcluded_raw_safeguard(config, patch):
    if not patch:
        config.pop('VIRTUALSKY')
    else:
        config['VIRTUALSKY'].update(patch)
    end = datetime(2026, 10, 25, 17, tzinfo=timezone.utc)
    assert moon_region(config, (3552, 3552), 1, end, 30, [53, 11, 23]) is None


def test_midpoint_ephemeris_and_below_horizon_moon(config, monkeypatch):
    import ephem
    observed = []
    def moon(observer):
        observed.append((observer.date.datetime(), observer.pressure))
        return SimpleNamespace(alt=-1., az=0., radius=.005, moon_phase=1.)
    monkeypatch.setattr(ephem, 'Moon', moon)
    end = datetime(2026, 10, 25, 17, 0, 30, tzinfo=timezone.utc)
    assert moon_region(config, (3552, 3552), 1, end, 30, [53, 11, 23]) is None
    assert abs((observed[0][0]-datetime(2026, 10, 25, 17, 0, 15)).total_seconds()) < .0001
    assert observed[0][1] == 0


def test_phase_halo_is_local_and_grows_monotonically(config, monkeypatch):
    import ephem
    end = datetime(2026, 10, 25, 17, tzinfo=timezone.utc)
    radii = []
    for phase in (0., .25, .5, 1.):
        monkeypatch.setattr(ephem, 'Moon', lambda observer, phase=phase:
                            SimpleNamespace(alt=1., az=2., radius=.005, moon_phase=phase))
        region = moon_region(config, (3552, 3552), 1, end, 30, [53, 11, 23])
        assert region is not None
        radii.append(region[2])
    assert all(a < b for a, b in zip(radii, radii[1:]))
    assert radii[-1] == pytest.approx(143)


@pytest.fixture(scope='module')
def geometric_processor():
    # Exercise the actual photograph geometry methods without importing camera,
    # SQL or Linux-only services. These are also tested through worker replay.
    source = (Path(__file__).parents[2]/'indi_allsky/processing.py').read_text(encoding='utf8')
    tree = ast.parse(source)
    original = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'ImageProcessor')
    names = {'_rotate_90', '_rotate_angle', '_flip', '_crop_image', '_scale_image', '_add_border'}
    original.body = [node for node in original.body if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = dict(cv2=cv2, numpy=np, logger=logging.getLogger(__name__))
    exec(compile(ast.Module(body=[original], type_ignores=[]), '<actual-image-geometry>', 'exec'), namespace)
    return namespace['ImageProcessor']


@pytest.mark.parametrize('quarter', ['', 'ROTATE_90_CLOCKWISE', 'ROTATE_90_COUNTERCLOCKWISE', 'ROTATE_180'])
@pytest.mark.parametrize('angle,keep', [(0, False), (-3, True), (30, False)])
@pytest.mark.parametrize('flip', [(False, False), (True, False), (False, True), (True, True)])
@pytest.mark.parametrize('scale', [50, 100])
def test_sparse_geometry_matches_actual_photo_marker(geometric_processor, quarter, angle, keep, flip, scale):
    config = {'IMAGE_ROTATE': quarter, 'IMAGE_ROTATE_ANGLE': angle, 'IMAGE_ROTATE_KEEP_SIZE': keep,
              'IMAGE_FLIP_H': flip[0], 'IMAGE_FLIP_V': flip[1], 'IMAGE_CROP_IMAGE_CIRCLE': True,
              'LENS_IMAGE_CIRCLE': 100, 'LENS_OFFSET_X': 10, 'LENS_OFFSET_Y': 3,
              'IMAGE_SCALE': scale, 'IMAGE_BORDER': {'TOP': 7, 'LEFT': 3, 'RIGHT': 11, 'BOTTOM': 5}}
    image = np.zeros((120, 160, 3), np.uint8)
    image[50:53, 90:93] = 200
    processor = geometric_processor.__new__(geometric_processor)
    processor.config, processor.image = config, image
    if quarter:
        processor._rotate_90(getattr(cv2, quarter))
    if angle:
        processor._rotate_angle(angle, keep_size=keep)
    if flip[1]:
        processor.image = processor._flip(processor.image, 0)
    if flip[0]:
        processor.image = processor._flip(processor.image, 1)
    processor.image = processor._crop_image(SimpleNamespace(binning=1))
    matrix, shape, preborder, pre_shape = _geometry(config, image.shape, 1)
    assert processor.image.shape[:2] == pre_shape
    before = processor.image[:, :, 0]
    y, x = np.indices(before.shape)
    actual = np.array([(x*before).sum()/before.sum(), (y*before).sum()/before.sum(), 1.])
    np.testing.assert_allclose(actual, preborder @ [91, 51, 1], atol=.8)
    if scale != 100:
        processor._scale_image(scale)
    processor.image = processor._add_border(7, 3, 11, 5)
    assert processor.image.shape[:2] == shape
    before = processor.image[:, :, 0]
    y, x = np.indices(before.shape)
    actual = np.array([(x*before).sum()/before.sum(), (y*before).sum()/before.sum(), 1.])
    np.testing.assert_allclose(actual, matrix @ [91, 51, 1], atol=.8)


@pytest.mark.parametrize('binning', [1, 2, 3])
def test_roi_crop_uses_the_same_binning_as_photographic_pipeline(geometric_processor, binning):
    config = {'IMAGE_CROP_ROI': [15, 21, 180, 210], 'IMAGE_SCALE': 100}
    image = np.ones((100, 110, 3), np.uint8)
    processor = geometric_processor.__new__(geometric_processor)
    processor.config, processor.image = config, image
    processor.image = processor._crop_image(SimpleNamespace(binning=binning))
    _, shape, matrix, pre_shape = _geometry(config, image.shape, binning)
    assert processor.image.shape[:2] == pre_shape == shape
    np.testing.assert_array_equal(matrix @ [50, 50, 1], [50-int(15/binning), 50-int(21/binning), 1])
