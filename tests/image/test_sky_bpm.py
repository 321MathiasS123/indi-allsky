"""Known map positions still require a present narrow colour defect."""
import cv2
import numpy as np
import pytest
from scipy.spatial import cKDTree

from indi_allsky import sky_denoise
from indi_allsky.sky_catalogue import SkySourceCatalogue


def scene(channel=None):
    image = np.random.default_rng(674).normal(.08, .001, (192, 192, 3)).astype(np.float32)
    if channel is not None:
        image[96, 96, channel] += .3
    return image


def evidence(image, points, *, margin=0):
    return sky_denoise._source_evidence(image, np.ones(image.shape[:2], bool),
        catalogue=SkySourceCatalogue(), capture_context=dict(capture_time=1000,
            capture_interval=30, geometry_key='mapped', sensor_shape=(192+2*margin, 192+2*margin)),
        margin=margin, bad_pixel_map=points)


@pytest.mark.parametrize('channel', [0, 1, 2])
@pytest.mark.parametrize('margin', [0, 512])
def test_active_mapped_colour_defect_is_repaired_without_visible_stars(channel, margin):
    image = scene(channel)
    measured = evidence(image, cKDTree([[96+margin, 96+margin]]), margin=margin)
    assert measured['catalogue']['status'] == 'motion_fallback'
    assert measured['catalogue']['mapped_pixels'] == 1
    defects = measured['sensor_defects']
    np.testing.assert_array_equal(defects, [[96, 96, 1]])
    result = image.copy()
    sky_denoise._repair_sensor_pixels(result, defects, measured['sensor_clearance'])
    assert result[96, 96, channel] < .085
    assert np.all(result[96, 96] > .075)  # no dark hole in the other channels
    np.testing.assert_array_equal(result[:92], image[:92])
    np.testing.assert_array_equal(result[101:], image[101:])


@pytest.mark.parametrize('channel', [0, 1, 2])
def test_mapped_impulse_in_normal_source_table_preserves_early_filter_weights(channel):
    image = scene(channel)
    image[96, 96, (channel+1) % 3] += .02
    measured = evidence(image, np.array([[96, 96]]))
    assert measured['catalogue']['mapped_pixels'] == 1
    nearby = np.linalg.norm(measured['points'][:, :2] - [96, 96], axis=1) < 1
    assert nearby.sum() == 1
    assert not measured['catalogue']['stationary_mask'][nearby].any()
    assert measured.get('restore_mask', measured['mask'])[96, 96] == 1
    np.testing.assert_array_equal(measured['sensor_defects'], [[96, 96, 1]])


def test_mapped_normal_repair_does_not_feed_back_into_adaptive_filters():
    image = scene(2)
    image[96, 96, 0] += .02
    context = dict(capture_time=1000, capture_interval=30, geometry_key='mapped')
    baseline = sky_denoise.denoise(image, {}, catalogue=SkySourceCatalogue(), capture_context=context)
    result = sky_denoise.denoise(image, {}, catalogue=SkySourceCatalogue(), capture_context=context,
                                bad_pixel_map=cKDTree([[96, 96]]))
    y, x = np.mgrid[:192, :192]
    outside = (x-96)**2 + (y-96)**2 >= 4**2
    np.testing.assert_array_equal(result[outside], baseline[outside])
    assert baseline[96, 96, 2] - result[96, 96, 2] > .05


@pytest.mark.parametrize('colour', [(1, 1, 1), (.35, .7, 1), (1, .7, .35), (.1, .1, 1)])
def test_star_crossing_quiet_map_position_keeps_resolved_profile(colour):
    image = scene()
    y, x = np.mgrid[:192, :192]
    image += (.15 * np.exp(-((x-96)**2 + (y-96)**2) / 4))[:, :, None] * colour
    original = image.copy()
    baseline = evidence(image, None)
    measured = evidence(image, cKDTree([[96, 96]]))
    assert measured['catalogue']['mapped_pixels'] == 0
    for key in ('mask', 'points', 'noise', 'lum'):
        np.testing.assert_array_equal(measured[key], baseline[key])
    assert not len(measured.get('sensor_defects', ()))
    np.testing.assert_array_equal(image, original)


def test_absent_low_gain_defect_and_unmapped_impulses_remain_pixel_exact():
    image = scene(2)
    original = image.copy()
    context = dict(capture_time=1000, capture_interval=30, geometry_key='mapped')
    baseline = sky_denoise.denoise(image, {}, catalogue=SkySourceCatalogue(), capture_context=context)
    result = sky_denoise.denoise(image, {}, catalogue=SkySourceCatalogue(), capture_context=context,
                                bad_pixel_map=cKDTree([[45, 45], [70, 145]]))
    np.testing.assert_array_equal(result, baseline)
    np.testing.assert_array_equal(image, original)


def test_full_filter_mapped_repair_preserves_other_pixels_and_star_wings():
    image = scene(2)
    y, x = np.mgrid[:192, :192]
    image += (.06 * np.exp(-((x-104)**2 + (y-96)**2) / 3))[:, :, None]
    original = image.copy()
    context = dict(capture_time=1000, capture_interval=30, geometry_key='mapped')
    baseline = sky_denoise.denoise(image, {}, catalogue=SkySourceCatalogue(), capture_context=context)
    result = sky_denoise.denoise(image, {}, catalogue=SkySourceCatalogue(), capture_context=context,
                                bad_pixel_map=cKDTree([[96, 96]]))
    assert baseline[96, 96, 2] - result[96, 96, 2] > .05
    protected = (x-104)**2 + (y-96)**2 <= 6.5**2
    outside = (x-96)**2 + (y-96)**2 >= 4**2
    np.testing.assert_array_equal(result[protected | outside], baseline[protected | outside])
    np.testing.assert_array_equal(image, original)


@pytest.mark.parametrize('sun', [0, 5, 45])
def test_daytime_bypass_does_not_add_point_or_map_work(monkeypatch, sun):
    image = scene(2)
    baseline = sky_denoise.denoise(image, {}, sun_altitude=sun)
    def forbidden(*args):
        pytest.fail('Daytime must retain the fast path')
    monkeypatch.setattr(sky_denoise, '_mapped_impulses', forbidden)
    result = sky_denoise.denoise(image, {}, sun_altitude=sun, catalogue=SkySourceCatalogue(),
        capture_context=dict(capture_time=1000, geometry_key='day'), bad_pixel_map=cKDTree([[96, 96]]))
    np.testing.assert_array_equal(result, baseline)


@pytest.mark.parametrize('bad_map', [[], [[np.nan, 96]], [[96.5, 96]], [[96, 96, 1]],
                                    np.zeros((20001, 2)), [[5000, 5000]]])
def test_malformed_or_unrelated_map_falls_back(bad_map):
    measured = evidence(scene(2), bad_map)
    assert measured['catalogue']['mapped_pixels'] == 0
    assert not len(measured['sensor_defects'])


def test_debayered_map_impulse_is_recognized_at_sensor_coordinates():
    rng = np.random.default_rng(204)
    raw = np.rint(rng.normal(5000, 40, (192, 192))).astype(np.uint16)
    raw[96, 96] = 35000
    image = cv2.cvtColor(raw, cv2.COLOR_BAYER_BG2BGR).astype(np.float32) / 65535
    measured = evidence(image, cKDTree([[96, 96]]))
    assert measured['catalogue']['mapped_pixels'] == 1
    np.testing.assert_array_equal(measured['sensor_defects'], [[96, 96, 1]])
