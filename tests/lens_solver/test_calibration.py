import cv2
import numpy as np
import pytest

from indi_allsky.lens_solver.calibration import (
    fitCorrection, displacement, validateCalibration, pipelineSignature)
from indi_allsky.lens_solver.detection import StarDetector
from indi_allsky.lens_solver.request import parseSolverRequestValues, applySolvedValuesToConfig
from tests.flask.test_virtualsky_requests import VALUES


def field(layout='circle'):
    rng = np.random.default_rng(3128)
    points = rng.uniform(-1, 1, (10000, 2))
    keep = np.linalg.norm(points, axis=1) < 0.95
    if layout == 'landscape':
        keep &= np.abs(points[:, 1]) < 0.55
    elif layout == 'portrait':
        keep &= np.abs(points[:, 0]) < 0.55
    elif layout == 'offset_crop':
        keep &= (points[:, 0] > -0.2) & (np.abs(points[:, 1]) < 0.65)
    expected = points[keep]
    source = expected[:350]
    u, v = source.T
    # Independent radial and asymmetric distortions, in circle-radius units.
    shift = source*(0.018*(u*u+v*v))[:, None]
    shift += np.column_stack([0.004*(3*u*u+v*v), 0.008*u*v])
    target = source+shift+rng.normal(0, 0.0004, source.shape)
    return source, target, expected[350:]


@pytest.mark.parametrize('layout', ['circle', 'landscape', 'portrait', 'offset_crop'])
def test_learned_mapping_improves_unused_stars(layout):
    source, target, expected = field(layout)
    result, reason = fitCorrection(source, target, expected, 0.025)
    assert result is not None, reason
    model, stats = result
    assert stats['validation'] >= 15
    assert stats['after'] < stats['before']*0.5
    # Validate the learned mapping against independent physical truth too.
    # A few wrong initial identities can keep the reported validation RMS high;
    # the implementation deliberately does not discard those validation errors.
    u, v = expected.T
    truth = expected*(0.018*(u*u+v*v))[:, None]
    truth += np.column_stack([0.004*(3*u*u+v*v), 0.008*u*v])
    assert np.sqrt(np.mean(np.sum((displacement(expected, model)-truth)**2, axis=1))) < 0.002


def test_central_patch_cannot_validate_full_sensor():
    source, target, expected = field()
    keep = np.linalg.norm(source, axis=1) < 0.6
    result, reason = fitCorrection(source[keep], target[keep], expected, 0.025)
    assert result is None
    assert 'cover' in reason


def test_correct_mapping_is_retained_when_no_improvement_is_needed():
    source, _, expected = field()
    result, _ = fitCorrection(source, source, expected, 0.025)
    assert result is None


@pytest.mark.parametrize('kind', ['few', 'line', 'noise'])
def test_unreliable_calibration_is_refused(kind):
    source, target, expected = field()
    if kind == 'few':
        source, target = source[:30], target[:30]
    elif kind == 'line':
        source[:, 1] = source[:, 0]
        target = source.copy()
    else:
        target = np.random.default_rng(12).uniform(-1, 1, target.shape)
    assert fitCorrection(source, target, expected, 0.025)[0] is None


def saved_model():
    source, target, expected = field()
    model, _ = fitCorrection(source, target, expected, 0.025)[0]
    model.update(geometry=[VALUES[k] for k in ('AZIMUTH_ANGLE', 'LATITUDE_OFFSET',
        'LONGITUDE_OFFSET', 'IMAGE_CIRCLE_DIAMETER', 'OFFSET_X', 'OFFSET_Y')]+[90, 123],
        image_size=[2028, 1520], context=[53, 11, 0], camera_uuid='test-camera',
        pipeline=pipelineSignature({}), summary='Validation: 80 unused stars.')
    return model


@pytest.mark.parametrize('key,value', [('coefficients', [[float('nan'), 0]]*10),
    ('coefficients', [[True, False]]*10), ('coefficients', [[1, 1]]*10),
    ('bounds', [0, 0, 0, 0]), ('bounds', [-100, -100, 100, 100]),
    ('image_size', [0, 1]), ('geometry', [0]*7), ('version', 999),
    ('context', [None, 0, 0]), ('summary', '<x>'*1000)])
def test_invalid_models_are_rejected(key, value):
    model = saved_model()
    model[key] = value
    assert not validateCalibration(model)


def test_model_save_toggle_and_geometry_binding():
    model = saved_model()
    assert validateCalibration(model)
    payload = dict(VALUES, LENS_ALTITUDE=90, CALIBRATION_ENABLED=True, CALIBRATION=model)
    parsed, error = parseSolverRequestValues(payload, for_save=True)
    assert error is None
    config = applySolvedValuesToConfig({}, parsed)
    assert config['VIRTUALSKY']['CALIBRATION'] == model
    payload['CALIBRATION_ENABLED'] = False
    parsed, error = parseSolverRequestValues(payload, for_save=True)
    assert error is None
    applySolvedValuesToConfig(config, parsed)
    assert config['VIRTUALSKY']['CALIBRATION'] == model
    assert not config['VIRTUALSKY']['CALIBRATION_ENABLED']
    payload['OFFSET_X'] += 1
    assert parseSolverRequestValues(payload, for_save=True)[1]


@pytest.mark.parametrize('value', ['true', 'false', 0, 1, None, [], {}])
def test_calibration_flag_requires_json_boolean(value):
    assert parseSolverRequestValues(dict(VALUES, CALIBRATION_ENABLED=value), for_save=True)[1]


def test_taper_is_identity_far_outside_calibrated_region():
    model = saved_model()
    np.testing.assert_array_equal(displacement(np.array([[3, 0], [-3, 0], [0, 3]]), model), 0)


def test_roi_transform_uses_sensor_coordinates_binning_rotation_and_crop():
    detector = StarDetector({'SQM_ROI': [20, 10, 80, 50], 'IMAGE_ROTATE': 'ROTATE_90_CLOCKWISE',
                             'IMAGE_CROP_ROI': [0, 0, 60, 100]})
    detector.use_sky_hints = True
    detector.sensor_shape = (40, 60)  # unbinned 80 x 120 sensor
    detector.binning = 2
    detections = np.array([[20, 20, 100], [2, 20, 100], [20, 45, 100]])
    # Original ROI [10:40, 5:25] -> clockwise x=[15:35], y=[10:40], then crop.
    preferred = detector.preferredDetections(detections, (50, 30))
    np.testing.assert_array_equal(preferred, detections[:1])


@pytest.mark.parametrize('roi', [None, [], [1], [0, 0, -1, 2], [0, 0, 5000, 4000],
                               [0, 0, float('nan'), 20], [0, 0, True, 20]])
def test_invalid_roi_is_ignored(roi):
    detector = StarDetector({'SQM_ROI': roi})
    detector.sensor_shape = (100, 100)
    assert len(detector.preferredDetections(np.array([[50, 50, 100]]), (100, 100))) == 0


def test_detection_mask_excludes_lights_from_calibration_threshold(tmp_path):
    mask = np.zeros((200, 400), np.uint8)
    mask[:, :200] = 255
    path = tmp_path / 'mask.png'
    assert cv2.imwrite(str(path), mask)
    image = np.full(mask.shape, 15, np.uint8)
    for y in range(20, 200, 30):
        for x in range(20, 200, 30):
            cv2.circle(image, (x, y), 2, 35, -1)
    image[:, 200:] = np.random.default_rng(3).integers(0, 255, (200, 200), dtype=np.uint8)
    detector = StarDetector({'DETECT_MASK': str(path)})
    legacy = detector.detectStars(image)
    detector.use_sky_hints = True
    hinted = detector.detectStars(image)
    assert len(hinted) > len(legacy)
    assert np.all(hinted[:, 0] < 200)


@pytest.mark.parametrize('altitude,heading', [(90, 0), (54, 123)])
def test_solver_learns_from_rendered_catalogue(tmp_path, altitude, heading):
    from indi_allsky.lens_solver import IndiAllSkyLensSolver
    from tests.lens_solver.test_orientation import star_field, render_stars
    from tests.lens_solver.test_camera_tilt import PARAMS, KEYS

    _, detections, _, _ = star_field(altitude, heading)
    center = np.array([1014+PARAMS[4], 760-PARAMS[5]])
    q = (detections[:, :2]-center)/(PARAMS[3]/2)
    u, v = q.T
    shift = q*(0.025*(u*u+v*v))[:, None]
    shift += np.column_stack([0.006*(3*u*u+v*v), 0.012*u*v])
    detections[:, :2] += shift*(PARAMS[3]/2)
    path = tmp_path / 'distorted.png'
    render_stars(path, detections)
    initial = dict(zip(KEYS, PARAMS), CALIBRATION_ENABLED=True)
    result = IndiAllSkyLensSolver({}).solve(path, 46.51, 8, 1770000000, initial,
        lens_altitude=altitude, pointing_azimuth=heading)
    assert result['success'], result
    assert result['calibration'] is not None, result['message']
    assert validateCalibration(result['calibration'])
    assert result['calibration']['image_size'] == [2028, 1520]
