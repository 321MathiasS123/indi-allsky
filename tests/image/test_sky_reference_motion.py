"""Predicted sky movement is sparse, locally calibrated and observation-gated."""
from copy import deepcopy
import numpy as np
import pytest
from scipy.spatial import cKDTree

from indi_allsky import sky_reference


STAMP = 1792186431.
SHAPE = (3552, 3552, 3)


def config():
    return dict(LENS_AZIMUTH=359.6, LENS_ALTITUDE=87.47, IMAGE_SCALE=100,
        VIRTUALSKY=dict(IMAGE_CIRCLE_DIAMETER=2298, OFFSET_X=-6, OFFSET_Y=-37,
            LATITUDE_OFFSET=0., LONGITUDE_OFFSET=0., POINTING_AZIMUTH=170.55,
            RADIAL_DISTORTION=.078567, PRECESSION=True,
            CALIBRATION_ENABLED=False, CALIBRATION=None))


def make(cfg=None, latitude=53.):
    return sky_reference.VirtualSkyReference(config() if cfg is None else cfg,
        SHAPE, 1, 'camera', latitude, 11.)


def correction(cfg):
    vs = cfg['VIRTUALSKY']
    coefficients = np.zeros((10, 2))
    coefficients[1] = [.003, -.002]
    coefficients[2] = [.001, .002]
    coefficients[4] = [.002, .001]
    return dict(version=2, coefficients=coefficients.tolist(),
        bounds=[-.9, -.9, .9, .9],
        geometry=[cfg['LENS_AZIMUTH'], 0., 0., vs['IMAGE_CIRCLE_DIAMETER'],
                  vs['OFFSET_X'], vs['OFFSET_Y'], cfg['LENS_ALTITUDE'],
                  vs['POINTING_AZIMUTH'], vs['RADIAL_DISTORTION'], 1],
        image_size=[SHAPE[1], SHAPE[0]], context=[53., 11., 0],
        pipeline=sky_reference._pipeline_signature(cfg), camera_uuid='camera', summary='Test')


def test_motion_requires_previously_observed_alignment_and_expires():
    aid = make()
    points = aid.predictions(STAMP)[:8, :2]
    assert aid.displacements(points, 30, reference_time=STAMP) is None
    aid.mark_validated(STAMP)
    assert aid.motion_available(STAMP)
    assert aid.motion_available(STAMP+6*3600)
    assert not aid.motion_available(STAMP-1)
    assert not aid.motion_available(STAMP+6*3600+1)
    assert aid.displacements(points, 30, reference_time=STAMP+6*3600+1) is None
    aid.mark_validated(STAMP+6*3600)
    assert aid.displacements(points, 30, reference_time=STAMP+6*3600+30) is not None


@pytest.mark.parametrize('radial', [-.5, 0, .078567, 1.])
@pytest.mark.parametrize('orientation', [(False, False), (True, False), (True, True)])
def test_inverse_and_rotation_agree_with_independent_catalogue_predictions(radial, orientation):
    cfg = config()
    cfg['VIRTUALSKY'].update(RADIAL_DISTORTION=radial,
        FLIP_H=orientation[0], FLIP_V=orientation[1])
    aid = make(cfg)
    aid.mark_validated(STAMP)
    before = aid.predictions(STAMP)[:, :2]
    original = before.copy()
    delta = aid.displacements(before, 300, reference_time=STAMP+300)
    supported = np.linalg.norm(delta, axis=1) > 1e-6
    distance, _ = cKDTree(aid.predictions(STAMP+300)[:, :2]).query((before+delta)[supported])
    assert len(distance) > 500
    assert distance.max() < .0005
    np.testing.assert_array_equal(before, original)


def test_empirical_correction_and_image_transform_are_inverted():
    cfg = config()
    cfg.update(IMAGE_ROTATE='ROTATE_90_CLOCKWISE', IMAGE_ROTATE_ANGLE=-3,
               IMAGE_ROTATE_KEEP_SIZE=True, IMAGE_FLIP_H=True, IMAGE_FLIP_V=True)
    cfg['VIRTUALSKY'].update(CALIBRATION_ENABLED=True, CALIBRATION=correction(cfg))
    original = deepcopy(cfg)
    aid = make(cfg)
    assert aid.valid
    aid.mark_validated(STAMP)
    before = aid.predictions(STAMP)[:, :2]
    delta = aid.displacements(before, 120, reference_time=STAMP+120)
    supported = np.linalg.norm(delta, axis=1) > 1e-6
    distance, _ = cKDTree(aid.predictions(STAMP+120)[:, :2]).query((before+delta)[supported])
    assert distance.max() < .0005
    assert cfg == original


@pytest.mark.parametrize('latitude', [-75., -23., 0., 53., 89.])
def test_local_sky_rotation_preserves_the_celestial_pole(monkeypatch, latitude):
    # The visible pole must remain stationary on either hemisphere. A nearby
    # source is slow, while a source 35 degrees from it moves substantially.
    dec = np.radians(np.array([90., 89.9, 55.])*(1 if latitude >= 0 else -1))
    ra = np.full(3, sky_reference._gmst(STAMP)+np.radians(11.))
    monkeypatch.setattr(sky_reference, '_equatorial', lambda epoch: (ra, np.sin(dec), np.cos(dec)))
    monkeypatch.setattr(sky_reference, '_catalogue', lambda: np.column_stack((ra, dec, [2., 3., 4.])))
    aid = make(latitude=latitude)
    aid.mark_validated(STAMP)
    before = aid.predictions(STAMP)
    if latitude == 0:
        # A pole at the horizon is deliberately outside prediction coverage.
        assert 2. not in before[:, 2]
        return
    distance = np.linalg.norm(aid.displacements(before[:, :2], 300, reference_time=STAMP+300), axis=1)
    assert distance[before[:, 2] == 2.][0] < 1e-9
    assert distance[before[:, 2] == 3.][0] < .1
    if 4. in before[:, 2]:
        assert distance[before[:, 2] == 4.][0] > 4.5


def test_scalar_and_per_source_elapsed_time_match():
    aid = make()
    aid.mark_validated(STAMP)
    points = aid.predictions(STAMP)[:4, :2]
    delta = aid.displacements(points, [0, 30, 60, 120], reference_time=STAMP+120)
    for index, age in enumerate((0, 30, 60, 120)):
        expected = aid.displacements(points[index:index+1], age, reference_time=STAMP+120)
        np.testing.assert_array_equal(delta[index:index+1], expected)


@pytest.mark.parametrize('points,elapsed', [([[1., 2.]], -1), ([[1., 2.]], 21601),
    ([[1., 2.]], float('nan')), ([[float('nan'), 2.]], 30), ([[1., 2., 3.]], 30),
    ([1., 2.], 30), ([[1., 2.]], [30., 60.])])
def test_bad_motion_arguments_fail_closed(points, elapsed):
    aid = make()
    aid.mark_validated(STAMP)
    assert aid.displacements(points, elapsed, reference_time=STAMP+30) is None


def test_unsupported_sensor_positions_cannot_gain_motion_proof():
    aid = make()
    aid.mark_validated(STAMP)
    points = np.array([[-1., 2.], [0., 0.], [3552., 3552.]])
    np.testing.assert_array_equal(aid.displacements(points, 300, reference_time=STAMP+300), np.zeros((3, 2)))
