"""Sparse shortcuts preserve the original source and restoration decisions."""
import cv2
import numpy as np
import pytest
from scipy.ndimage import maximum_filter

from indi_allsky import sky_denoise


@pytest.fixture
def source_scene():
    image = np.random.default_rng(713).normal(.15, .001, (192, 192, 3)).astype(np.float32)
    y, x = np.ogrid[:192, :192]
    for cx, cy, width in [(45, 50, 2), (95, 90, 10), (135, 140, 3)]:
        image += (.1 * np.exp(-((x-cx)**2 + (y-cy)**2) / width))[:, :, None]
    image[65, 130, 2] += .3
    return image


class RecordingCatalogue:
    def __init__(self, varied=False):
        self.varied = varied

    def weights(self, points, support, compact, **context):
        self.points = points.copy()
        self.compact = compact.copy()
        self.original_weights = (np.resize(np.array([1, .5, 0, .25], np.float32), len(points))
                                 if self.varied else np.ones(len(points), np.float32))
        self.stationary = np.arange(len(points)) % 3 == 0
        return self.original_weights.copy(), {'stationary_mask': self.stationary}


def reference_restoration(edge, points, weights, stationary=None):
    """Independent per-source full-grid envelopes, including zero weights."""
    result = edge.copy()
    y, x = np.indices(edge.shape)
    if stationary is not None:
        for (cx, cy, _), weight, fixed in zip(points, weights, stationary):
            if fixed:
                core = np.clip((4 - np.hypot(x-cx, y-cy)) / 2, 0, 1).astype(np.float32)
                result *= 1 - (1 - weight) * core
    for (cx, cy, _), weight in zip(points, weights):
        distance = np.sqrt((x-cx)**2 + (y-cy)**2)
        taper = np.clip((6.5 - distance) / 2.5, 0, 1).astype(np.float32)
        result = np.maximum(result, weight * taper)
    return result


@pytest.mark.parametrize('case', ['flat', 'ties', 'border'])
def test_dilated_source_maxima_match_original_filter(case, monkeypatch):
    image = np.full((80, 84, 3), .2, np.float32)
    if case == 'ties':
        image[30:33, 35:38] = .9
        image[46:49, 54:57] = .9
    elif case == 'border':
        image[:3] = .9
        image[:, 1] = .8
        image[40, 40] = 1
    measured = sky_denoise._source_evidence(image)
    original_dilate = cv2.dilate
    calls = []

    def original_peak_maximum(data, kernel, *args, **kwargs):
        if kernel.shape == (5, 5) and np.all(kernel == 1):
            calls.append(data.shape)
            return maximum_filter(data, size=5)
        return original_dilate(data, kernel, *args, **kwargs)

    monkeypatch.setattr(cv2, 'dilate', original_peak_maximum)
    expected = sky_denoise._source_evidence(image)
    assert calls == [image.shape[:2]]
    for key in ('lum', 'noise', 'points', 'mask'):
        np.testing.assert_array_equal(measured[key], expected[key])


def test_nan_source_maxima_keep_scipy_ordering(monkeypatch):
    image = np.random.default_rng(48).normal(.2, .001, (96, 96, 3)).astype(np.float32)
    image[48, 48, 1] = np.nan
    calls = []
    original_dilate = cv2.dilate

    def recorded_maximum(data, size):
        assert np.isnan(data).any()
        calls.append(size)
        return maximum_filter(data, size=size)

    def no_peak_dilation(data, kernel, *args, **kwargs):
        assert not (kernel.shape == (5, 5) and np.all(kernel == 1))
        return original_dilate(data, kernel, *args, **kwargs)

    monkeypatch.setattr(sky_denoise, 'maximum_filter', recorded_maximum)
    monkeypatch.setattr(cv2, 'dilate', no_peak_dilation)
    evidence = sky_denoise._source_evidence(image)
    assert calls == [5]
    assert np.isnan(evidence['lum'][48, 48])


def test_compact_subset_matches_original_all_source_wing_test(source_scene):
    catalogue = RecordingCatalogue()
    sky_denoise._source_evidence(source_scene, catalogue=catalogue,
        capture_context={'capture_time': 1000})
    x, y = catalogue.points[:, :2].astype(int).T
    lum = source_scene @ sky_denoise.WEIGHTS
    dog = cv2.GaussianBlur(lum, (0, 0), 1) - cv2.GaussianBlur(lum, (0, 0), 3.5)
    colour = cv2.GaussianBlur(source_scene, (0, 0), 1) - cv2.GaussianBlur(source_scene, (0, 0), 3.5)
    scores = [colour[:, :, c] / sky_denoise._noise_grid(colour[:, :, c]) for c in range(3)]
    outer = np.maximum.reduce([dog[y+dy, x+dx] for dy, dx in
        ((-4, 0), (4, 0), (0, -4), (0, 4), (-3, -3), (-3, 3), (3, -3), (3, 3))])
    compact = outer < .3 * dog[y, x]
    assert compact.any() and (~compact).any()
    core = colour[y, x]
    wings = np.zeros(len(x), np.uint8)
    for dy, dx in ((-3, 0), (3, 0), (0, -3), (0, 3), (-2, -2), (-2, 2), (2, -2), (2, 2)):
        significance = np.column_stack([s[y+dy, x+dx] for s in scores])
        wings += np.count_nonzero((significance > 3) & (colour[y+dy, x+dx] > .03 * core), axis=1) >= 2
    np.testing.assert_array_equal(catalogue.compact, compact & (wings < 4))


def test_only_unprotected_sources_need_trail_measurement(source_scene, monkeypatch):
    catalogue = RecordingCatalogue(varied=True)
    original_lines = sky_denoise._line_sources
    original_restoration = sky_denoise._weighted_points
    line_calls = []
    restoration = {}

    def recorded_lines(dog, noise, x, y):
        line_calls.append(np.column_stack((x, y)))
        return original_lines(dog, noise, x, y)

    def recorded_restoration(edge, points, weights, stationary):
        restoration.update(edge=edge.copy(), weights=weights.copy())
        return original_restoration(edge, points, weights, stationary)

    monkeypatch.setattr(sky_denoise, '_line_sources', recorded_lines)
    monkeypatch.setattr(sky_denoise, '_weighted_points', recorded_restoration)
    measured = sky_denoise._source_evidence(source_scene, catalogue=catalogue,
        capture_context={'capture_time': 1000})
    assert len(line_calls) == 1
    pending = catalogue.original_weights < 1
    np.testing.assert_array_equal(line_calls[0], catalogue.points[pending, :2])
    lum = measured['lum']
    dog = cv2.GaussianBlur(lum, (0, 0), 1) - cv2.GaussianBlur(lum, (0, 0), 3.5)
    x, y = catalogue.points[:, :2].astype(int).T
    expected_weights = catalogue.original_weights.copy()
    expected_weights[original_lines(dog, measured['noise'], x, y)] = 1
    np.testing.assert_array_equal(restoration['weights'], expected_weights)
    expected = reference_restoration(restoration['edge'], catalogue.points, expected_weights,
                                     catalogue.stationary)
    np.testing.assert_array_equal(measured['restore_mask'], expected)


@pytest.mark.parametrize('weights', [[0, 0, 0], [1, 1, 1], [0, .5, .25]])
def test_omitted_zero_splats_keep_overlapping_stationary_fades(weights):
    edge = np.random.default_rng(16).random((64, 64)).astype(np.float32)
    original = edge.copy()
    points = np.array([[25, 25, 8], [29, 26, 12], [35, 29, 4]])
    weights = np.array(weights, np.float32)
    stationary = np.array([True, True, False])
    expected = reference_restoration(edge, points, weights, stationary)
    result = sky_denoise._weighted_points(edge, points, weights, stationary)
    np.testing.assert_array_equal(result, expected)
    np.testing.assert_array_equal(edge, original)
