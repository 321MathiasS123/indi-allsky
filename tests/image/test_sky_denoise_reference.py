"""A projected catalogue can confirm measured stars, never invent image detail."""
import copy

import numpy as np
import pytest

from indi_allsky import sky_denoise
from indi_allsky.sky_catalogue import SkySourceCatalogue


ANCHORS = np.array([[45, 45], [140, 42], [250, 55], [275, 145], [258, 260],
                    [150, 275], [45, 255], [38, 145], [120, 125], [210, 200]], float)
TARGET = np.array([165, 190])
UNKNOWN = np.array([95, 200])
SENSOR = np.array([225, 110])
EMPTY = np.array([75, 115])


class Reference:
    def __init__(self, predictions):
        self.table = predictions
        self.times = []

    def predictions(self, timestamp):
        self.times.append(timestamp)
        return self.table


def tables():
    observed = np.vstack((np.column_stack((ANCHORS, np.full(len(ANCHORS), 8))),
                          [*TARGET, 3.2], [*UNKNOWN, 3.2]))
    predicted = np.vstack((np.column_stack((ANCHORS, np.full(len(ANCHORS), 3))),
                           [*TARGET, 5.5], [*SENSOR, 4], [*EMPTY, 5]))
    return observed, predicted


def test_reference_only_confirms_unambiguous_measured_sources():
    observed, predicted = tables()
    reference = Reference(predicted)
    matched = sky_denoise._reference_matches(observed, (320, 320), 1000.5, reference)
    assert matched.tolist() == [True] * (len(ANCHORS) + 1) + [False]
    assert reference.times == [1000.5]
    # Predictions at EMPTY and SENSOR did not introduce additional sources.
    assert len(matched) == len(observed)


@pytest.mark.parametrize('case', ['missing', 'empty', 'wrong', 'few', 'clustered',
                                 'large_residual', 'shift_aliases', 'too_faint', 'invalid'])
def test_uncertain_reference_keeps_existing_behavior(case):
    observed, predicted = tables()
    if case == 'missing':
        reference = None
    else:
        if case == 'empty':
            predicted = np.empty((0, 3))
        elif case == 'wrong':
            predicted[:, :2] += 20
        elif case == 'few':
            predicted = predicted[:7]
        elif case == 'clustered':
            observed[:, :2] = observed[:, :2] / 10 + 130
            predicted[:, :2] = predicted[:, :2] / 10 + 130
        elif case == 'large_residual':
            predicted[:, 0] += 2.5
        elif case == 'shift_aliases':
            observed = np.vstack([observed + [dx, dy, 0] for dx, dy in
                                   [(0, 0), (16, 0), (-16, 0), (0, 16), (0, -16)]])
        elif case == 'too_faint':
            predicted[:, 2] = 4.6
        elif case == 'invalid':
            predicted[0, 0] = np.nan
        reference = Reference(predicted)
    assert not sky_denoise._reference_matches(observed, (320, 320), 1000, reference).any()


def test_dense_unrelated_noise_cannot_validate_a_catalogue():
    rng = np.random.default_rng(711)
    observed = np.column_stack((rng.uniform(12, 500, (10000, 2)), np.full(10000, 8)))
    predicted = np.column_stack((rng.uniform(12, 500, (1000, 2)), np.full(1000, 3)))
    assert not sky_denoise._reference_matches(observed, (512, 512), 1000,
                                             Reference(predicted)).any()


@pytest.mark.parametrize('ambiguous', ['catalogue', 'observed'])
def test_crowded_matches_do_not_confirm_ambiguous_sources(ambiguous):
    observed, predicted = tables()
    if ambiguous == 'catalogue':
        predicted = np.vstack((predicted, [*(TARGET + [.5, 0]), 5]))
    else:
        observed = np.vstack((observed, [*(TARGET + [.5, 0]), 4]))
    matched = sky_denoise._reference_matches(observed, (320, 320), 1000, Reference(predicted))
    assert matched[:len(ANCHORS)].all()
    assert not matched[len(ANCHORS):].any()


@pytest.mark.parametrize('binning', [1, 2, 3])
def test_reference_radius_is_measured_in_native_sensor_pixels(binning):
    observed, predicted = tables()
    observed[:, :2] /= binning
    predicted[:, :2] /= binning
    shape = (320 / binning, 320 / binning)
    predicted[len(ANCHORS), 0] += 2.8 / binning
    assert sky_denoise._reference_matches(observed, shape, 1000, Reference(predicted),
                                          binning)[len(ANCHORS)]
    predicted[len(ANCHORS), 0] += .4 / binning
    assert not sky_denoise._reference_matches(observed, shape, 1000, Reference(predicted),
                                              binning)[len(ANCHORS)]


def test_catalogue_failures_and_invalid_capture_times_are_optional():
    observed, _ = tables()

    class Broken:
        def predictions(self, timestamp):
            raise OSError('Unavailable calibration')

    assert not sky_denoise._reference_matches(observed, (320, 320), 1000, Broken()).any()
    reference = Reference(np.empty((0, 3)))
    assert not sky_denoise._reference_matches(observed, (320, 320), np.nan, reference).any()
    assert not reference.times


@pytest.fixture
def scene():
    image = np.random.default_rng(823).normal(.08, .001, (320, 320, 3)).astype(np.float32)
    y, x = np.ogrid[:320, :320]
    for cx, cy in ANCHORS:
        image += (.12 * np.exp(-((x-cx)**2 + (y-cy)**2) / 1.5))[:, :, None]
    for cx, cy in [TARGET, UNKNOWN]:
        image += (.02 * np.exp(-((x-cx)**2 + (y-cy)**2) / .9))[:, :, None]
    image[SENSOR[1], SENSOR[0], 2] += .3
    return image


def at(points, coordinate):
    selected = np.flatnonzero(np.linalg.norm(points[:, :2] - coordinate, axis=1) < 1)
    assert len(selected) == 1
    return selected[0]


@pytest.mark.parametrize('aligned', [False, True])
def test_only_observed_matching_alignment_authorizes_cloud_motion(scene, aligned):
    class MotionReference(Reference):
        def __init__(self, predictions):
            super().__init__(predictions)
            self.validated = []

        def mark_validated(self, timestamp):
            self.validated.append(timestamp)

        def motion_available(self, timestamp):
            return bool(self.validated)

    class Recorder:
        def weights(self, points, support, **kwargs):
            self.context = kwargs
            return np.ones(len(points), np.float32), {}

    predicted = tables()[1].copy()
    if not aligned:
        predicted[:, :2] += 20
    reference = MotionReference(predicted)
    catalogue = Recorder()
    context = dict(capture_time=1000, reference_time=1015, geometry_key='camera')
    retained = context.copy()
    sky_denoise._source_evidence(scene, catalogue=catalogue, capture_context=context,
                                star_reference=reference)
    assert reference.validated == ([1015] if aligned else [])
    assert catalogue.context['motion_reference'] is reference
    assert catalogue.context['capture_time'] == 1000
    assert catalogue.context['reference_time'] == 1015
    assert context == retained


def test_reference_overrides_cached_false_sensor_proof_without_erasing_memory(scene):
    valid = np.ones(scene.shape[:2], bool)
    context = dict(capture_time=1000, capture_interval=30, geometry_key='camera', sensor_shape=valid.shape)
    catalogue = SkySourceCatalogue()
    initial = sky_denoise._source_evidence(scene, valid, catalogue=catalogue, capture_context=context)
    index = at(initial['points'], TARGET)
    catalogue._last_weights[index] = 0
    catalogue._last_stationary[index] = True
    reference = Reference(tables()[1])
    bad = sky_denoise._source_evidence(scene, valid, catalogue=catalogue, capture_context=context)
    assert bad['catalogue']['stationary_mask'][index]
    assert np.any(np.linalg.norm(bad['sensor_defects'][:, :2] - TARGET, axis=1) < 1)
    fixed = sky_denoise._source_evidence(scene, valid, catalogue=catalogue,
                                        capture_context=context, star_reference=reference)
    assert fixed['catalogue']['status'] == 'duplicate'
    assert fixed['catalogue']['reference_confirmed'] >= len(ANCHORS)
    assert not fixed['catalogue']['stationary_mask'][index]
    assert not np.any(np.linalg.norm(fixed['sensor_defects'][:, :2] - TARGET, axis=1) < 1)
    assert catalogue._last_weights[index] == 0 and catalogue._last_stationary[index]
    # Only the current observed star is protected; remembered sensor evidence is retained.
    control = copy.deepcopy(catalogue)
    control._last_weights[index] = 1
    control._last_stationary[index] = False
    expected = sky_denoise._denoise_linear(scene, valid, catalogue=control, capture_context=context)
    actual = sky_denoise._denoise_linear(scene, valid, catalogue=catalogue,
                                        capture_context=context, star_reference=reference)
    np.testing.assert_array_equal(actual, expected)


class MeasuredCatalogue:
    def weights(self, points, support, compact, sensor_points, **context):
        self.points, self.compact, self.sensors = points.copy(), compact.copy(), sensor_points.copy()
        weights = np.full(len(points), .25, np.float32)
        return weights, dict(stationary_mask=np.zeros(len(points), bool),
            sensor_stationary=np.ones(len(sensor_points), bool),
            sensor_weights=np.zeros(len(sensor_points), np.float32))

    def reset(self):
        pass


@pytest.mark.parametrize('reference_clock,enabled', [(1015, True), (1015, False), (None, True)])
def test_reference_clock_is_separate_from_tracking_without_mutating_context(
        scene, reference_clock, enabled):
    class ClockCatalogue(MeasuredCatalogue):
        def weights(self, points, support, compact, sensor_points, **context):
            self.context = context.copy()
            assert 'reference_time' not in context
            return super().weights(points, support, compact, sensor_points, **context)

    context = dict(capture_time=1000.5, sensor_shape=scene.shape[:2])
    if reference_clock is not None:
        context['reference_time'] = reference_clock
    original = context.copy()
    reference = Reference(tables()[1])
    catalogue = ClockCatalogue()
    sky_denoise._source_evidence(scene, catalogue=catalogue, capture_context=context,
                                 star_reference=reference if enabled else None)
    assert context == original
    assert catalogue.context['capture_time'] == 1000.5
    assert reference.times == ([reference_clock or 1000.5] if enabled else [])


@pytest.mark.parametrize('margin', [0, 128])
def test_only_normal_measured_points_gain_confidence_and_margin_is_sensor_relative(scene, margin):
    reference = tables()[1].copy()
    reference[:, :2] += margin
    catalogue = MeasuredCatalogue()
    context = dict(capture_time=1000, sensor_shape=(320 + 2 * margin, 320 + 2 * margin))
    result = sky_denoise._source_evidence(scene, catalogue=catalogue, capture_context=context,
        margin=margin, star_reference=Reference(reference))
    index = at(catalogue.points, TARGET + margin)
    assert not catalogue.compact[index]
    assert result['restore_mask'][TARGET[1], TARGET[0]] == 1
    assert result['restore_mask'][UNKNOWN[1], UNKNOWN[0]] < 1
    assert np.any(np.linalg.norm(result['sensor_defects'][:, :2] - SENSOR, axis=1) < 1)
    assert not np.any(np.linalg.norm(result['points'][:, :2] - EMPTY, axis=1) < 1)
    assert not np.any(np.linalg.norm(result['points'][:, :2] - SENSOR, axis=1) < 1)


def test_reference_confirmation_keeps_twilight_fade_and_daylight_fast_path(scene, monkeypatch):
    reference = Reference(tables()[1])
    context = dict(capture_time=1000, sensor_shape=scene.shape[:2])
    original = sky_denoise._weighted_points
    weights_at_star = []

    def recorded(edge, points, weights, stationary):
        weights_at_star.append(weights[at(points, TARGET)])
        return original(edge, points, weights, stationary)

    monkeypatch.setattr(sky_denoise, '_weighted_points', recorded)
    partial = sky_denoise._source_evidence(scene, star_protection=.5,
        catalogue=MeasuredCatalogue(), capture_context=context, star_reference=reference)
    full = sky_denoise._source_evidence(scene, catalogue=MeasuredCatalogue(),
        capture_context=context, star_reference=reference)
    assert weights_at_star == [.5, 1]
    # Existing resolved-edge protection can exceed the faded point weight.
    assert partial['restore_mask'][TARGET[1], TARGET[0]] < full['restore_mask'][TARGET[1], TARGET[0]]
    assert full['restore_mask'][TARGET[1], TARGET[0]] == 1
    reference.times.clear()
    sky_denoise._source_evidence(scene, star_protection=0, catalogue=MeasuredCatalogue(),
        capture_context=context, star_reference=reference, processing_only=True)
    assert not reference.times


@pytest.mark.parametrize('reference', [None, Reference(np.empty((0, 3)))])
def test_disabled_or_unusable_reference_leaves_every_evidence_array_exact(scene, reference):
    context = dict(capture_time=1000, sensor_shape=scene.shape[:2])
    expected = sky_denoise._source_evidence(scene, catalogue=MeasuredCatalogue(), capture_context=context)
    actual = sky_denoise._source_evidence(scene, catalogue=MeasuredCatalogue(),
                                         capture_context=context, star_reference=reference)
    for key in ('lum', 'noise', 'points', 'mask', 'restore_mask', 'sensor_defects'):
        np.testing.assert_array_equal(actual[key], expected[key])
    for a, b in zip(actual['sensor_clearance'], expected['sensor_clearance']):
        np.testing.assert_array_equal(a, b)
