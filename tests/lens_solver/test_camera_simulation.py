"""Image-level camera simulations; geometry is independent of the fitted model.

Every case renders an actual PNG and runs normal detection and solving. FOV is
the angular span along the shorter sensor dimension before an off-centre crop.
Non-native distortions deliberately test the limits of a one-parameter lens.
Counterexamples remain failing regressions until the solver recovers accurate
pointing or refuses an unreliable result; they are not skipped or marked xfail.
"""
import json
from functools import lru_cache

import cv2
import numpy as np
import pytest
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.time import Time
import astropy.units as u

from indi_allsky.lens_solver import IndiAllSkyLensSolver


SENSORS = [(1600, 1600), (1600, 1200), (1920, 1080), (2400, 800), (1080, 1920)]
MODELS = ['equisolid', 'equidistant', 'stereographic', 'orthographic']
ALTITUDE, HEADING, ROLL = 87.4, 176., 200.


@lru_cache(maxsize=4)
def observed_sky(hour):
    timestamp = 1788731972 + hour*3600
    catalog = IndiAllSkyLensSolver({}).loadCatalog()
    stars = SkyCoord(ra=catalog[:, 0]*u.deg, dec=catalog[:, 1]*u.deg, frame='icrs')
    location = EarthLocation.from_geodetic(11*u.deg, 53*u.deg, 0*u.m)
    sky = stars.transform_to(AltAz(obstime=Time(timestamp, format='unix'), location=location))
    alt, az = sky.alt.rad, sky.az.rad
    return timestamp, alt, np.column_stack([
        np.cos(alt)*np.sin(az), np.cos(alt)*np.cos(az), np.sin(alt)])


def lens_radius(theta, model):
    # Native lens laws, normalized to radius one at 90 degrees off-axis.
    if model == 'equisolid':
        return np.sqrt(2)*np.sin(theta/2)
    if model == 'equidistant':
        return theta/(np.pi/2)
    if model == 'stereographic':
        return np.tan(theta/2)
    if model == 'orthographic':
        return np.sin(theta)
    raise ValueError(model)


def simulate_camera(path, sensor, fov, model, hour, distortion='none', strength=0,
                    crop=(0., 0.), noise=False, obstruction=False, guess_error=False):
    width, height = sensor
    timestamp, alt, world = observed_sky(hour)
    elevation, heading, roll = np.radians([ALTITUDE, HEADING, ROLL])
    axis = np.array([np.cos(elevation)*np.sin(heading),
                     np.cos(elevation)*np.cos(heading), np.sin(elevation)])
    v = np.cross(axis, [0., 0., 1.])
    skew = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation = np.eye(3)+skew+skew@skew/(1+axis[2])
    east, north, z = (world@rotation.T).T
    theta = np.arccos(np.clip(z, -1, 1))
    diameter = min(sensor)/lens_radius(np.radians(fov/2), model)
    radius = lens_radius(theta, model)
    norm = np.maximum(np.hypot(east, north), 1e-12)
    x = -radius*(east*np.cos(roll)-north*np.sin(roll))/norm
    y = -radius*(north*np.cos(roll)+east*np.sin(roll))/norm
    r2 = x*x+y*y
    if distortion == 'radial':
        x, y = x*(1+strength*r2), y*(1+strength*r2)
    elif distortion == 'wave':
        factor = 1+strength*np.sin(2*np.pi*radius)
        x, y = x*factor, y*factor
    elif distortion == 'elliptical':
        x, y = x*(1+strength), y*(1-strength)
    elif distortion == 'decentered':
        # Tangential displacement, with a different x/y dependence from a
        # shifted optical centre. It cannot be absorbed by a radial parameter.
        x, y = x+strength*(r2+2*x*x), y+strength*2*x*y
    elif distortion != 'none':
        raise ValueError(distortion)
    cx, cy = width/2+crop[0]*width, height/2+crop[1]*height
    x, y = cx+x*diameter/2, cy+y*diameter/2
    keep = (alt > np.radians(10)) & (z > 0) & (x > 5) & (x < width-5) & (y > 5) & (y < height-5)
    image = np.full((height, width), 10, np.uint8)
    for sx, sy in zip(x[keep], y[keep]):
        cv2.circle(image, (int(round(sx)), int(round(sy))), 2, 220, -1)
    image = cv2.GaussianBlur(image, (5, 5), 1.1)
    if noise:
        rng = np.random.default_rng(3127+hour)
        image = np.clip(image.astype(float)+rng.normal(0, 3, image.shape), 0, 255).astype(np.uint8)
        image[rng.integers(height, size=300), rng.integers(width, size=300)] = 255
    config = {}
    if obstruction:
        mask = np.full(image.shape, 255, np.uint8)
        mask[:, :width//3] = 0
        image[mask == 0] = 10
        mask_path = path.with_name('mask.png')
        assert cv2.imwrite(str(mask_path), mask)
        config['DETECT_MASK'] = str(mask_path)
    assert cv2.imwrite(str(path), image)
    initial = dict(AZIMUTH_ANGLE=ROLL, LATITUDE_OFFSET=0, LONGITUDE_OFFSET=0,
        IMAGE_CIRCLE_DIAMETER=diameter*(1.15 if guess_error else 1),
        OFFSET_X=cx-width/2+(40 if guess_error else 0),
        OFFSET_Y=height/2-cy-(30 if guess_error else 0),
        LENS_ALTITUDE=90, POINTING_AZIMUTH=0, PRECESSION=True, RADIAL_DISTORTION=0)
    result = IndiAllSkyLensSolver(config).solve(path, 53, 11, timestamp, initial)
    values = result.get('values', {})
    heading_error = abs((values['POINTING_AZIMUTH']-HEADING+180) % 360-180) if 'POINTING_AZIMUTH' in values else None
    elevation_error = abs(values['LENS_ALTITUDE']-ALTITUDE) if 'LENS_ALTITUDE' in values else None
    report = dict(sensor=sensor, fov=fov, model=model, hour=hour,
        distortion=distortion, strength=strength, crop=crop, noise=noise,
        obstruction=obstruction, guess_error=guess_error, stars_rendered=int(keep.sum()),
        heading_error=heading_error, elevation_error=elevation_error, result=result)
    path.with_suffix('.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


def assert_accurate(report):
    result = report['result']
    assert result['success'] and not result['partial'], report
    assert report['heading_error'] is not None and report['heading_error'] < 2, report
    assert report['elevation_error'] is not None and report['elevation_error'] < 0.5, report


@pytest.mark.parametrize('sensor', SENSORS, ids=['square', '4x3', '16x9', '3x1', 'portrait'])
@pytest.mark.parametrize('fov', [180, 140, 100, 60])
@pytest.mark.parametrize('model', MODELS)
@pytest.mark.parametrize('hour', [0, 4, 8, 18])
def test_sensor_fov_and_native_lens(tmp_path, sensor, fov, model, hour):
    assert_accurate(simulate_camera(tmp_path/'sky.png', sensor, fov, model, hour))


@pytest.mark.parametrize('sensor', SENSORS, ids=['square', '4x3', '16x9', '3x1', 'portrait'])
@pytest.mark.parametrize('model', MODELS)
@pytest.mark.parametrize('hour', [0, 4, 8, 18])
def test_narrow_field_is_accurate_or_refused(tmp_path, sensor, model, hour):
    report = simulate_camera(tmp_path/'sky.png', sensor, 30, model, hour)
    if report['result']['success']:
        assert_accurate(report)
    else:
        assert 'values' not in report['result'], report


@pytest.mark.parametrize('distortion', ['radial', 'wave', 'elliptical', 'decentered'])
@pytest.mark.parametrize('strength', [0.01, 0.05])
@pytest.mark.parametrize('fov', [160, 80])
@pytest.mark.parametrize('hour', [0, 4, 8, 18])
def test_non_native_distortion_is_accurate_or_refused(tmp_path, distortion, strength, fov, hour):
    report = simulate_camera(tmp_path/'sky.png', (1920, 1080), fov,
                             'equisolid', hour, distortion, strength)
    if report['result']['success']:
        assert_accurate(report)
    else:
        assert 'values' not in report['result'], report


@pytest.mark.parametrize('sensor', [(2400, 800), (1080, 1920)])
@pytest.mark.parametrize('crop', [(0.2, -0.15), (-0.2, 0.15)])
@pytest.mark.parametrize('hour', [0, 4, 8, 18])
def test_offset_noisy_masked_sensor(tmp_path, sensor, crop, hour):
    assert_accurate(simulate_camera(tmp_path/'sky.png', sensor, 140, 'equisolid', hour,
        crop=crop, noise=True, obstruction=True, guess_error=True))


@pytest.mark.parametrize('model', MODELS)
def test_solved_values_remain_a_valid_starting_point(tmp_path, model):
    path = tmp_path/'sky.png'
    report = simulate_camera(path, (1920, 1080), 140, model, 0)
    assert_accurate(report)
    values = report['result']['values']
    again = IndiAllSkyLensSolver({}).solve(path, 53, 11, observed_sky(0)[0], values,
        lens_altitude=values['LENS_ALTITUDE'], pointing_azimuth=values['POINTING_AZIMUTH'])
    assert again['success'] and not again['partial'], again
    assert abs((again['values']['POINTING_AZIMUTH']-HEADING+180) % 360-180) < 2, again
    assert abs(again['values']['LENS_ALTITUDE']-ALTITUDE) < 0.5, again
