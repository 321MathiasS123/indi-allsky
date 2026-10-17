"""Cloud fallback may prove sensor defects without inventing visible stars."""
from copy import deepcopy

import numpy as np
import pytest

from indi_allsky.sky_catalogue import SkySourceCatalogue


SHAPE = (1200, 1200)


class MotionReference:
    def __init__(self, validated=True, speed=.06):
        self.validated = validated
        self.speed = speed

    def displacements(self, positions, elapsed, *, reference_time):
        if not self.validated or reference_time is None:
            return None
        return np.column_stack((elapsed*self.speed, np.zeros(len(positions))))


def update(catalogue, frame, *, source=True, compact=True, reference=None,
           sensors=None, x=600., time=None, **kwargs):
    points = np.array([[x, 500., 12.]]) if source else np.empty((0, 3))
    timestamp = 1000+frame*30 if time is None else time
    return catalogue.weights(points, points, capture_time=timestamp,
        geometry_key=kwargs.pop('geometry_key', 'one'), sensor_shape=SHAPE,
        compact=np.full(len(points), compact), sensor_points=sensors,
        motion_reference=reference, reference_time=timestamp+15, **kwargs)


def same_state(left, right):
    assert vars(left).keys() == vars(right).keys()
    for key, value in vars(left).items():
        other = getattr(right, key)
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(value, other, err_msg=key)
        else:
            assert value == other, key


def test_unvalidated_reference_is_exact_existing_cloud_fallback():
    old, new = SkySourceCatalogue(), SkySourceCatalogue()
    reference = MotionReference(validated=False)
    for frame in range(12):
        left = update(old, frame)
        right = update(new, frame, reference=reference)
        np.testing.assert_array_equal(left[0], right[0])
        same_state(old, new)
    assert not len(new._fixed)


def test_cloudy_fixed_point_needs_four_of_six_sixty_seconds_and_displacement():
    catalogue = SkySourceCatalogue()
    reference = MotionReference()
    values, masks = [], []
    for frame in range(7):
        weights, diagnostics = update(catalogue, frame, reference=reference)
        values.append(weights[0])
        masks.append(bool(diagnostics['stationary_mask'][0]))
        assert not catalogue._history and not catalogue._observed_history
        assert not len(catalogue._tracks)  # No sky/source votes while cloudy.
    assert values == [1, 1, 1, .5, .25, 0, 0]
    assert masks == [False, False, False, True, True, True, True]


def test_cloud_votes_do_not_bypass_the_sixty_second_guard():
    catalogue = SkySourceCatalogue()
    reference = MotionReference(speed=2.)
    for frame in range(7):
        _, info = update(catalogue, frame, time=1000+frame*9, reference=reference)
        assert not info['stationary_mask'][0]
    _, info = update(catalogue, 7, time=1063, reference=reference)
    assert info['stationary_mask'][0]


def test_sparse_reappearances_do_not_satisfy_four_of_six():
    catalogue = SkySourceCatalogue()
    reference = MotionReference()
    for frame in range(18):
        present = frame % 2 == 0
        weights, diagnostics = update(catalogue, frame, source=present, reference=reference)
        if present:
            assert weights[0] == 1 and not diagnostics['stationary_mask'][0]


@pytest.mark.parametrize('compact,speed', [(False, .06), (True, .001)])
def test_resolved_or_local_slow_star_keeps_protection(compact, speed):
    catalogue = SkySourceCatalogue()
    for frame in range(11):
        weights, diagnostics = update(catalogue, frame, compact=compact,
            reference=MotionReference(speed=speed))
        assert weights[0] == 1 and not diagnostics['stationary_mask'][0]


def test_actual_movement_prevents_fixed_pixel_proof():
    catalogue = SkySourceCatalogue()
    for frame in range(12):
        weights, diagnostics = update(catalogue, frame, x=600.+.4*frame,
            reference=MotionReference())
        assert weights[0] == 1 and not diagnostics['stationary_mask'][0]


def test_single_colour_sensor_candidates_can_be_learned_without_star_votes():
    catalogue = SkySourceCatalogue()
    sensors = np.array([[600., 500., 12.]])
    for frame in range(6):
        _, diagnostics = update(catalogue, frame, source=False, sensors=sensors,
            reference=MotionReference())
    assert diagnostics['sensor_stationary'][0]
    assert diagnostics['sensor_weights'][0] == 0
    assert not catalogue._history and not len(catalogue._tracks)


def test_reference_loss_keeps_proven_defects_but_stops_new_learning():
    catalogue = SkySourceCatalogue()
    for frame in range(5):
        update(catalogue, frame, reference=MotionReference())
    weights, info = update(catalogue, 5)
    assert weights[0] == 0 and info['stationary_mask'][0]
    update(catalogue, 6, x=700.)
    assert len(catalogue._fixed) == 1


def test_duplicate_capture_and_geometry_reset_do_not_accumulate_proof():
    catalogue = SkySourceCatalogue()
    reference = MotionReference()
    for _ in range(10):
        weights, _ = update(catalogue, 0, reference=reference)
        assert weights[0] == 1
    assert catalogue._fixed[0, 4] == 1
    for frame in range(1, 4):
        update(catalogue, frame, reference=reference)
    assert catalogue._fixed[0, 8] == 1
    _, info = update(catalogue, 4, reference=reference, geometry_key='new-geometry')
    assert not info['stationary_mask'][0]
    assert catalogue._fixed[0, 4] == 1 and catalogue._fixed[0, 5] == 1120


@pytest.mark.parametrize('bad', [np.array([[float('nan'), 0]]), np.array([[1e30, 0]]),
                                np.array([6.]), np.zeros((0, 2))])
def test_invalid_reference_result_does_not_partially_update_sensor_proof(bad):
    catalogue = SkySourceCatalogue()
    for frame in range(3):
        update(catalogue, frame, reference=MotionReference())
    baseline = deepcopy(catalogue)
    broken = MotionReference()
    broken.displacements = lambda *args, **kwargs: bad
    left = update(catalogue, 3, reference=broken)
    right = update(baseline, 3)
    np.testing.assert_array_equal(left[0], right[0])
    same_state(catalogue, baseline)


def test_clear_recovery_does_not_count_motion_before_first_cloud_observation(monkeypatch):
    from indi_allsky import sky_catalogue
    catalogue = SkySourceCatalogue()
    rng = np.random.default_rng(47)
    anchors = np.column_stack((rng.uniform(150, 1050, (100, 2)), np.full(100, 12.)))
    catalogue.weights(anchors, anchors, capture_time=1000., geometry_key='one', sensor_shape=SHAPE)
    update(catalogue, 1, reference=MotionReference())  # First fixed observation is at 1030.
    coefficient = np.zeros((6, 2))
    coefficient[0] = [6., 0.]
    monkeypatch.setattr(sky_catalogue, '_motion', lambda *args, **kwargs: (coefficient, {'reason': 'matched'}))
    support = np.vstack((anchors+[6, 0, 0], [600., 500., 12.]))
    flags = np.zeros(len(support), bool)
    flags[-1] = True
    catalogue.weights(support, support, capture_time=1060., geometry_key='one', sensor_shape=SHAPE,
                      compact=flags)
    # Full fitted shift spans 60s; this candidate existed for only the last 30s.
    np.testing.assert_allclose(catalogue._fixed[0, 2:4], [603., 500.])
