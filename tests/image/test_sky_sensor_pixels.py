"""Single-colour sensor defects must not become astronomical source votes."""
import numpy as np
import pytest

from indi_allsky import sky_denoise
from indi_allsky.sky_catalogue import SkySourceCatalogue


class SensorCatalogue:
    def __init__(self, proven=False):
        self.proven = proven

    def weights(self, points, support, sensor_points, **kwargs):
        self.points, self.support, self.sensors = points, support, sensor_points
        return np.ones(len(points), np.float32), dict(
            sensor_stationary=np.full(len(sensor_points), self.proven, bool),
            sensor_weights=np.full(len(sensor_points), 0 if self.proven else 1, np.float32))


@pytest.mark.parametrize('channel', [0, 1, 2])
def test_single_colour_impulses_are_sensor_evidence_without_star_protection(channel):
    image = np.random.default_rng(674).normal(.08, .001, (192, 192, 3)).astype(np.float32)
    image[96, 96, channel] += .3
    valid = np.ones(image.shape[:2], bool)
    baseline = sky_denoise._source_evidence(image, valid)
    catalogue = SensorCatalogue()
    measured = sky_denoise._source_evidence(image, valid, catalogue=catalogue,
        capture_context={'capture_time': 1000}, margin=512)
    assert np.min(np.linalg.norm(catalogue.sensors[:, :2] - [608, 608], axis=1)) < 1
    assert not np.any(np.linalg.norm(catalogue.points[:, :2] - [608, 608], axis=1) < 4)
    for key in ('mask', 'points', 'lum', 'noise'):
        np.testing.assert_array_equal(measured[key], baseline[key])
    assert not len(measured['sensor_defects'])
    np.testing.assert_array_equal(
        sky_denoise.denoise(image, {}, catalogue=catalogue, capture_context={'capture_time': 1000}),
        sky_denoise.denoise(image, {}))


def test_sensor_candidates_reject_broad_signal_edges_and_nearby_stars():
    y, x = np.mgrid[:192, :192]
    image = np.full((192, 192, 3), .08, np.float32)
    image[:, :, 2] += .1 * np.exp(-((x-96)**2 + (y-96)**2) / 32)
    evidence = sky_denoise._source_evidence(image, np.ones((192,192), bool),
        catalogue=(catalogue := SensorCatalogue()), capture_context={'capture_time': 1000})
    assert not len(catalogue.sensors)
    assert not len(evidence['sensor_defects'])
    # An isolated sensor peak next to a measured star must not repair its wings.
    dog = np.zeros_like(image)
    dog[50,50,2] = 10
    scores = [dog[:,:,c] for c in range(3)]
    support = np.zeros((192,192), np.float32)
    distance = np.full_like(support, 20)
    distance[50,50] = 6
    assert not len(sky_denoise._single_colour_points(dog, scores, support, distance, None))
    distance[50,50] = 20
    assert len(sky_denoise._single_colour_points(dog, scores, support, distance, None)) == 1


def test_local_repair_preserves_gradient_dark_pixels_and_distant_detail():
    y, x = np.mgrid[:40, :40]
    image = np.repeat((.1 + x*.001 + y*.002)[:,:,None], 3, axis=2).astype(np.float32)
    background = image.copy()
    image[20,20] += [.02, .02, .2]
    image[21,21] -= .04
    original = image.copy()
    sky_denoise._repair_sensor_pixels(image, np.array([[20,20,1.]]))
    np.testing.assert_allclose(image[20,20], background[20,20], atol=1e-6)
    np.testing.assert_array_equal(image[21,21], original[21,21])
    assert np.all(image <= original)
    np.testing.assert_array_equal(image[:16], original[:16])
    np.testing.assert_array_equal(image[25:], original[25:])


def test_proven_defect_is_removed_from_filtered_output_without_changing_distant_pixels():
    image = np.random.default_rng(674).normal(.08, .001, (192,192,3)).astype(np.float32)
    image[96,96,2] += .3
    original = image.copy()
    baseline = sky_denoise.denoise(image, {})
    filtered = sky_denoise.denoise(image, {}, catalogue=SensorCatalogue(proven=True),
                                  capture_context={'capture_time': 1000})
    assert baseline[96,96,2] - filtered[96,96,2] > .05
    assert filtered[96,96,2] < .085
    np.testing.assert_array_equal(filtered[:92], baseline[:92])
    np.testing.assert_array_equal(filtered[101:], baseline[101:])
    np.testing.assert_array_equal(image, original)


def test_nearby_star_wings_remain_identical_when_the_repair_footprint_overlaps():
    y, x = np.mgrid[:192, :192]
    image = np.random.default_rng(674).normal(.08, .001, (192,192,3)).astype(np.float32)
    image += (.06 * np.exp(-((x-104)**2 + (y-96)**2)/3))[:, :, None]
    image[96,96,2] += .3
    context = {'capture_time': 1000}
    measured = sky_denoise._source_evidence(image, catalogue=SensorCatalogue(proven=True),
                                            capture_context=context)
    assert np.any(np.linalg.norm(measured['sensor_defects'][:, :2] - [96,96], axis=1) < 1)
    assert np.any(np.linalg.norm(measured['points'][:, :2] - [104,96], axis=1) < 1)
    protected = np.zeros((192,192), bool)
    for cx, cy, _ in measured['points']:
        protected |= (x-cx)**2 + (y-cy)**2 <= 6.5**2
    footprint = (x-96)**2 + (y-96)**2 < 4**2
    assert np.any(footprint & protected)
    baseline = sky_denoise.denoise(image, {})
    filtered = sky_denoise.denoise(image, {}, catalogue=SensorCatalogue(proven=True),
                                  capture_context=context)
    assert filtered[96,96,2] < baseline[96,96,2] - .05
    np.testing.assert_array_equal(filtered[protected], baseline[protected])


@pytest.mark.parametrize('context,protection', [(None, 1), ({'capture_time': 1000}, 0)])
def test_preview_and_daytime_do_not_collect_or_repair_sensor_points(context, protection):
    image = np.full((64,64,3), .08, np.float32)
    image[32,32,2] += .3
    catalogue = SkySourceCatalogue()
    evidence = sky_denoise._source_evidence(image, star_protection=protection,
                                            catalogue=catalogue, capture_context=context)
    assert 'sensor_defects' not in evidence
    assert not len(catalogue._fixed)


def scene(frame, channel=2, moving=False, broad=False):
    rng = np.random.default_rng(40 + frame)
    image = rng.normal(.08, .001, (400,400,3)).astype(np.float32)
    y,x = np.mgrid[:400,:400]
    for cy in np.arange(40,356,45):
        for cx in np.arange(40,356,45):
            image += (.08*np.exp(-((x-cx-frame*1.8)**2+(y-cy+frame*.7)**2)/3))[:,:,None]
    cx,cy = np.array([101.,111.]) + (np.array([1.8,-.7])*frame if moving else 0)
    if broad:
        image[:,:,channel] += .3*np.exp(-((x-cx)**2+(y-cy)**2)/32)
    else:
        image[round(cy),round(cx),channel] += .3
    return image, (round(cx), round(cy))


@pytest.mark.parametrize('channel', [0, 1, 2])
@pytest.mark.parametrize('moving', [False, True])
def test_capture_sequence_repairs_fixed_colour_impulses_but_not_moving_ones(channel, moving):
    catalogue = SkySourceCatalogue()
    detections = []
    for frame in range(7):
        image, center = scene(frame, channel, moving)
        context = dict(capture_time=1000+frame*30, capture_interval=30,
                       geometry_key='sensor-test', sensor_shape=(400,400))
        evidence = sky_denoise._source_evidence(image, np.ones((400,400), bool),
            catalogue=catalogue, capture_context=context)
        assert evidence['catalogue']['status'] == 'tracking'
        selected = evidence.get('sensor_defects', np.empty((0,3)))
        nearby = np.linalg.norm(selected[:,:2]-center, axis=1) < 2
        detections.append(float(selected[nearby,2].max()) if nearby.any() else 0)
    if moving:
        assert detections == [0]*7
    else:
        assert detections[:3] == [0]*3
        assert detections[-1] == 1
        before = image.copy()
        sky_denoise._repair_sensor_pixels(image, evidence['sensor_defects'])
        cx,cy = center
        assert image[cy,cx,channel] < before[cy,cx,channel] - .25
