"""Catalogue aid must agree with the solved overlay, without its import cost."""
from copy import deepcopy
from datetime import datetime, timezone
import subprocess
import sys

import numpy as np
import pytest

from indi_allsky import sky_reference as reference
from indi_allsky.maskProcessing import MaskProcessor


SHAPE = (3552, 3552, 3)
STAMP = datetime(2026, 10, 16, 21, 33, 51, tzinfo=timezone.utc).timestamp()


def config():
    return dict(LENS_AZIMUTH=359.6, LENS_ALTITUDE=87.47, IMAGE_SCALE=100,
                VIRTUALSKY=dict(IMAGE_CIRCLE_DIAMETER=2298, OFFSET_X=-6, OFFSET_Y=-37,
                    LATITUDE_OFFSET=0., LONGITUDE_OFFSET=0., POINTING_AZIMUTH=170.55,
                    RADIAL_DISTORTION=.078567, PRECESSION=True,
                    CALIBRATION_ENABLED=False, CALIBRATION=None))


def make(cfg=None, **kwargs):
    return reference.VirtualSkyReference(cfg if cfg is not None else config(),
        kwargs.pop('shape', SHAPE), kwargs.pop('binning', 1),
        kwargs.pop('camera_uuid', 'camera'), kwargs.pop('latitude', 53.),
        kwargs.pop('longitude', 11.))


def saved(cfg, version=2):
    vs = cfg['VIRTUALSKY']
    geometry = [cfg['LENS_AZIMUTH'], vs['LATITUDE_OFFSET'], vs['LONGITUDE_OFFSET'],
                vs['IMAGE_CIRCLE_DIAMETER'], vs['OFFSET_X'], vs['OFFSET_Y'],
                cfg['LENS_ALTITUDE'], vs['POINTING_AZIMUTH']]
    if version == 2:
        geometry += [vs['RADIAL_DISTORTION'], int(vs['PRECESSION'])]
    return dict(version=version, coefficients=np.zeros((10, 2)).tolist(),
                bounds=[-.9, -.9, .9, .9], geometry=geometry,
                image_size=[SHAPE[1], SHAPE[0]], context=[53, 11, 0],
                pipeline=reference._pipeline_signature(cfg), camera_uuid='camera', summary='Test')


def test_base_alignment_works_without_optional_residual_correction():
    cfg = config()
    original = deepcopy(cfg)
    result = make(cfg).predictions(STAMP)
    assert len(result) > 500
    assert np.all(np.isfinite(result)) and result[:, 2].max() <= 5.5
    assert cfg == original
    cfg['VIRTUALSKY']['CALIBRATION_ENABLED'] = True
    np.testing.assert_array_equal(make(cfg).predictions(STAMP), result)
    cfg['VIRTUALSKY'].update(CALIBRATION_ENABLED=False, CALIBRATION={'obsolete': True})
    np.testing.assert_array_equal(make(cfg).predictions(STAMP), result)


@pytest.mark.parametrize('timestamp', [float('nan'), float('inf'), -1, 4102444801, None, True])
def test_bad_capture_time_fails_open(timestamp):
    assert make().predictions(timestamp).shape == (0, 3)


@pytest.mark.parametrize('updates', [dict(shape=(0, 10)), dict(shape=(True, 10)),
    dict(binning=0), dict(binning=1.5), dict(latitude=91), dict(longitude=181)])
def test_bad_capture_geometry_fails_open(updates):
    assert make(**updates).predictions(STAMP).shape == (0, 3)


@pytest.mark.parametrize('key,value', [('IMAGE_ROTATE', 'nonsense'), ('IMAGE_ROTATE_ANGLE', float('nan')),
    ('IMAGE_CROP_ROI', [0, 0, 0, 1]), ('IMAGE_SCALE', 0), ('IMAGE_BORDER', {'LEFT': -1}),
    ('LENS_ALTITUDE', 95), ('LENS_AZIMUTH', float('inf'))])
def test_unsupported_config_fails_open(key, value):
    cfg = config()
    cfg[key] = value
    assert make(cfg).predictions(STAMP).shape == (0, 3)
    assert make({}).predictions(STAMP).shape == (0, 3)


@pytest.mark.parametrize('version', [1, 2])
def test_saved_model_and_ui_time_offset(version):
    cfg = config()
    if version == 1:
        cfg['VIRTUALSKY'].update(RADIAL_DISTORTION=0, PRECESSION=False)
    before = make(cfg).predictions(STAMP)
    cfg['VIRTUALSKY'].update(CALIBRATION_ENABLED=True, CALIBRATION=saved(cfg, version))
    cfg['VIRTUALSKY']['CALIBRATION']['context'][2] = 7200
    after = make(cfg).predictions(STAMP)
    # A remote UI timezone bridge is not a shift to physical UTC capture time.
    np.testing.assert_array_equal(after, before)


@pytest.mark.parametrize('key,value', [('version', True), ('version', 3), ('coefficients', [[True, 0]]*10),
    ('coefficients', [[float('nan'), 0]]*10), ('coefficients', [[1., 0]]*10),
    ('bounds', [-.01, -.01, .01, .01]), ('bounds', [-4., -1., 1., 1.]),
    ('geometry', [0]*10), ('image_size', [SHAPE[1]-2, SHAPE[0]]),
    ('pipeline', 'a'*64), ('camera_uuid', 'other-camera'), ('context', [54, 11, 0]),
    ('context', [53, 11, float('inf')]), ('summary', 'x'*501),
    ('orientation', [1, 0]), ('orientation', [True, False]), ('unexpected', 1)])
def test_invalid_or_stale_saved_model_is_not_used(key, value):
    cfg = config()
    model = saved(cfg)
    model[key] = value
    cfg['VIRTUALSKY'].update(CALIBRATION_ENABLED=True, CALIBRATION=model)
    aid = make(cfg)
    assert not aid.valid and aid.predictions(STAMP).shape == (0, 3)


def test_privacy_location_uses_same_rounding_as_overlay():
    cfg = config()
    expected = make(cfg).predictions(STAMP)
    cfg['PRIVACY_MODE'] = True
    cfg['VIRTUALSKY'].update(CALIBRATION_ENABLED=True, CALIBRATION=saved(cfg))
    np.testing.assert_array_equal(make(cfg, latitude=53.3, longitude=10.6).predictions(STAMP), expected)
    cfg['PRIVACY_MODE'] = False
    assert not make(cfg, latitude=53.3, longitude=10.6).valid


def test_saved_model_binds_transformed_dimensions_and_current_pipeline():
    cfg = config()
    cfg.update(IMAGE_ROTATE_ANGLE=-3, IMAGE_FLIP_H=True, IMAGE_FLIP_V=True,
               IMAGE_CROP_IMAGE_CIRCLE=True, LENS_IMAGE_CIRCLE=2265,
               LENS_OFFSET_X=36, LENS_OFFSET_Y=3, IMAGE_SCALE=75,
               IMAGE_BORDER=dict(LEFT=4, TOP=2, RIGHT=6, BOTTOM=8))
    base = make(cfg)
    assert base.valid
    model = saved(cfg)
    model['image_size'] = list(base._output_shape[::-1])
    cfg['VIRTUALSKY'].update(CALIBRATION_ENABLED=True, CALIBRATION=model)
    np.testing.assert_array_equal(make(cfg).predictions(STAMP), base.predictions(STAMP))
    cfg['IMAGE_FLIP_H'] = False  # same dimensions, incompatible saved mapping
    assert not make(cfg).valid


@pytest.mark.parametrize('flip_h,flip_v', [(False, False), (True, False), (False, True), (True, True)])
def test_optional_production_orientation_schema(flip_h, flip_v):
    cfg = config()
    cfg['VIRTUALSKY'].update(FLIP_H=flip_h, FLIP_V=flip_v)
    before = make(cfg).predictions(STAMP)
    model = saved(cfg)
    model['orientation'] = [flip_h, flip_v]
    cfg['VIRTUALSKY'].update(CALIBRATION_ENABLED=True, CALIBRATION=model)
    np.testing.assert_array_equal(make(cfg).predictions(STAMP), before)
    plain = make().predictions(STAMP)
    center = np.array([SHAPE[1]/2-6, SHAPE[0]/2+37])
    expected = (plain[:, :2]-center)*[(-1 if flip_h else 1), (-1 if flip_v else 1)]+center
    np.testing.assert_allclose(before[:, :2], expected, atol=1e-10)


@pytest.mark.parametrize('rotation', ['', 'ROTATE_90_CLOCKWISE', 'ROTATE_180', 'ROTATE_90_COUNTERCLOCKWISE'])
@pytest.mark.parametrize('keep_size', [False, True])
@pytest.mark.parametrize('crop', ['circle', 'roi'])
def test_inverse_matches_actual_mask_processing(rotation, keep_size, crop):
    cfg = dict(IMAGE_ROTATE=rotation, IMAGE_ROTATE_ANGLE=-13, IMAGE_ROTATE_KEEP_SIZE=keep_size,
               IMAGE_FLIP_H=True, IMAGE_FLIP_V=True, IMAGE_SCALE=73,
               IMAGE_BORDER=dict(LEFT=3, RIGHT=5, TOP=7, BOTTOM=11))
    if crop == 'circle':
        cfg.update(IMAGE_CROP_IMAGE_CIRCLE=True, LENS_IMAGE_CIRCLE=80, LENS_OFFSET_X=-5, LENS_OFFSET_Y=7)
    else:
        cfg['IMAGE_CROP_ROI'] = [22, 26, 178, 160]
    # Odd inputs test the half-pixel centre shift and even-size trim. Binning 2
    # distinguishes the unbinned circle-crop setting from the divided ROI.
    shape, binning = (123, 151), 2
    transform, output_shape, bounds = reference._sensor_to_output(cfg, shape, binning)
    images = []
    yy, xx = np.indices(shape, dtype=float)
    for source in (xx, yy, np.ones(shape)):
        processor = MaskProcessor(cfg)
        processor.binning = binning
        processor.image = source
        for operation in ('rotate_90', 'rotate_angle', 'flip_v', 'flip_h', 'crop_image', 'scale_image'):
            getattr(processor, operation)()
        images.append(np.pad(processor.image, ((7, 11), (3, 5))))
    assert images[0].shape == output_shape
    oy, ox = np.indices(output_shape)
    output = np.column_stack([ox.ravel(), oy.ravel(), np.ones(ox.size)])
    expected = output @ np.linalg.inv(transform).T
    valid = images[2].ravel() > .999999
    measured = np.column_stack([images[0].ravel(), images[1].ravel()])
    # OpenCV uses quantized interpolation weights. INTER_AREA's fractional
    # footprints can differ slightly from the geometric pixel centre.
    np.testing.assert_allclose(measured[valid], expected[valid, :2], atol=.16, rtol=0)
    assert bounds == (3, 7, output_shape[1]-5, output_shape[0]-11)


@pytest.mark.parametrize('year', [2000, 2026, 2045])
@pytest.mark.parametrize('precession', [False, True])
def test_projection_matches_existing_astropy_solver(year, precession):
    from indi_allsky.lens_solver.projection import precessCatalog, predictAltAz, projectToPixels, cameraAltAz

    cfg = config()
    cfg['VIRTUALSKY']['PRECESSION'] = precession
    vs = cfg['VIRTUALSKY']
    timestamp = datetime(year, 10, 16, 23, 59, 59, tzinfo=timezone.utc).timestamp()
    stars = reference._catalogue()
    catalog = precessCatalog(stars, timestamp) if precession else stars
    alt, az = predictAltAz(catalog, 53, 11, timestamp)
    params = [cfg['LENS_AZIMUTH'], 0, 0, vs['IMAGE_CIRCLE_DIAMETER'], vs['OFFSET_X'], vs['OFFSET_Y'], vs['RADIAL_DISTORTION']]
    x, y = projectToPixels(alt, az, params, SHAPE[1], SHAPE[0],
                          lens_altitude=cfg['LENS_ALTITUDE'], pointing_azimuth=vs['POINTING_AZIMUTH'])
    keep = (alt > np.radians(10)) & (cameraAltAz(alt, az, cfg['LENS_ALTITUDE'], vs['POINTING_AZIMUTH'])[0] > 0)
    keep &= (x >= 0) & (x < SHAPE[1]) & (y >= 0) & (y < SHAPE[0])
    expected = np.column_stack([x[keep], y[keep], stars[keep, 2]])
    actual = make(cfg).predictions(timestamp)
    np.testing.assert_array_equal(actual[:, 2], expected[:, 2])
    assert np.max(np.linalg.norm(actual[:, :2]-expected[:, :2], axis=1)) < .08


@pytest.mark.parametrize('latitude,altitude,heading,radial', [
    (-53, 40, 321, -.5), (0, 65, 70, 0), (89, 90, 0, .5), (-89, 70, 180, 1)])
def test_projection_for_other_observers_lens_curves_and_pointing(latitude, altitude, heading, radial):
    from indi_allsky.lens_solver.projection import precessCatalog, predictAltAz, projectToPixels, cameraAltAz

    cfg = config()
    cfg['LENS_ALTITUDE'] = altitude
    cfg['VIRTUALSKY'].update(POINTING_AZIMUTH=heading, RADIAL_DISTORTION=radial)
    stars = reference._catalogue()
    alt, az = predictAltAz(precessCatalog(stars, STAMP), latitude, 11, STAMP)
    x, y = projectToPixels(alt, az, [359.6, 0, 0, 2298, -6, -37, radial],
                          SHAPE[1], SHAPE[0], lens_altitude=altitude, pointing_azimuth=heading)
    keep = (alt > np.radians(10)) & (cameraAltAz(alt, az, altitude, heading)[0] > 0)
    keep &= (x >= 0) & (x < SHAPE[1]) & (y >= 0) & (y < SHAPE[0])
    expected = np.column_stack([x[keep], y[keep], stars[keep, 2]])
    actual = make(cfg, latitude=latitude).predictions(STAMP)
    np.testing.assert_array_equal(actual[:, 2], expected[:, 2])
    assert np.max(np.linalg.norm(actual[:, :2]-expected[:, :2], axis=1)) < .08


def test_saved_displacement_matches_solver_and_is_not_resampled():
    from indi_allsky.lens_solver.calibration import displacement, validateCalibration

    cfg = config()
    model = saved(cfg)
    model['coefficients'][1] = [.002, -.003]
    model['coefficients'][4] = [.001, .002]
    assert validateCalibration(model)
    points = np.random.default_rng(338).uniform(-1.2, 1.2, (200, 2))
    np.testing.assert_array_equal(reference._displacement(points, model), displacement(points, model))
    before = make(cfg).predictions(STAMP)
    cfg['VIRTUALSKY'].update(CALIBRATION_ENABLED=True, CALIBRATION=model)
    after = make(cfg).predictions(STAMP)
    center, radius = np.array([SHAPE[1]/2-6, SHAPE[0]/2+37]), 2298/2
    expected = before[:, :2]+radius*displacement((before[:, :2]-center)/radius, model)
    np.testing.assert_allclose(after[:, :2], expected, atol=1e-10)


def test_catalogue_and_day_precession_cache_do_not_freeze_sky_motion():
    reference._equatorial.cache_clear()
    aid = make()
    before, after = aid.predictions(STAMP), aid.predictions(STAMP+30)
    assert reference._equatorial.cache_info().hits == 1
    # Some sources cross the altitude cutoff; compare the common bright prefix.
    assert np.max(np.linalg.norm(before[:50, :2]-after[:50, :2], axis=1)) > 1
    assert not reference._catalogue().flags.writeable


def test_missing_catalogue_fails_open(monkeypatch):
    def missing():
        raise OSError('Missing optional data')
    monkeypatch.setattr(reference, '_catalogue', missing)
    assert make().predictions(STAMP).shape == (0, 3)


def test_normal_import_does_not_load_solver_or_astropy():
    code = ('import sys; from indi_allsky.sky_reference import VirtualSkyReference; '
            f'assert len(VirtualSkyReference({config()!r}, {SHAPE!r}, 1, "camera", 53, 11).predictions({STAMP!r})) > 0; '
            'assert "astropy" not in sys.modules; assert "indi_allsky.lens_solver" not in sys.modules')
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout+result.stderr
