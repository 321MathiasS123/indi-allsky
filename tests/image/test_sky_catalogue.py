"""Capture-time coordinate tracking without image buffers or source synthesis."""
import numpy as np
import pytest

from indi_allsky import sky_catalogue


SHAPE = (1200, 1200)


def anchors(frame):
    rng = np.random.default_rng(91)
    xy = rng.uniform(140, 1060, (160, 2))
    xy += np.array([1.8, -0.7]) * frame
    return np.column_stack((xy, np.full(len(xy), 12)))


def frame_points(frame, source=True, score=3.5, fixed=False):
    points = anchors(frame)
    if source:
        xy = np.array([601.3, 527.7]) + (0 if fixed else np.array([1.8, -0.7]) * frame)
        points = np.vstack((points, [*xy, score]))
    return points


def update(catalogue, frame, *, source=True, score=3.5, fixed=False, compact=False,
           support_only=False, time=None, **kwargs):
    support = frame_points(frame, source, score, fixed)
    points = support[:-1] if support_only and source else support
    flags = np.zeros(len(points), bool)
    if compact and source and not support_only:
        flags[-1] = True
    return catalogue.weights(points, support, capture_time=1000 + 30 * frame if time is None else time,
                             geometry_key='camera-one', sensor_shape=SHAPE,
                             compact=flags, **kwargs)


def test_weak_sources_start_partial_then_confirm_three_of_four():
    catalogue = sky_catalogue.SkySourceCatalogue()
    values = [float(update(catalogue, frame)[0][-1]) for frame in range(4)]
    assert values == [0.5, 0.5, 1, 1]
    assert catalogue._tracks.ndim == 2 and catalogue._tracks.shape[1] == 11


def test_insufficient_anchor_bootstrap_keeps_baseline_without_partial_pulsing():
    catalogue = sky_catalogue.SkySourceCatalogue()
    for frame in range(4):
        points = frame_points(frame)[-1:]
        weights, diagnostics = catalogue.weights(points, points, capture_time=1000+30*frame,
            geometry_key='camera-one', sensor_shape=SHAPE)
        assert weights[0] == 1 and diagnostics['status'] == 'motion_fallback'
        assert not len(catalogue._tracks) and not catalogue._history


def test_direct_three_of_four_survives_a_closer_stale_rejected_track():
    catalogue = sky_catalogue.SkySourceCatalogue()
    # A noisy first centroid fragments this one real moving source into two
    # tracks. The last point is closer to the old rejected identity, while its
    # current plus two prior measured centroids still satisfy the 1.5 px rule.
    for frame, jitter in enumerate((1.3, -1.3, -1.2, -1.1, -1.0, 0.35)):
        points = frame_points(frame)
        points[-1, 0] += jitter
        weights, _ = catalogue.weights(points, points, capture_time=1000+30*frame,
            geometry_key='camera-one', sensor_shape=SHAPE)
    assert weights[-1] == 1
    assert len(catalogue._history) == 3


def test_lower_historical_support_and_one_missing_frame_can_confirm():
    catalogue = sky_catalogue.SkySourceCatalogue()
    update(catalogue, 0)
    update(catalogue, 1, score=2.2, support_only=True)
    update(catalogue, 2, source=False)
    weights, diagnostics = update(catalogue, 3)
    assert weights[-1] == 1
    assert diagnostics['status'] == 'tracking'


def test_reappearance_does_not_get_new_grace_after_rejection():
    catalogue = sky_catalogue.SkySourceCatalogue()
    assert update(catalogue, 0)[0][-1] == 0.5
    for frame in range(1, 7):
        update(catalogue, frame, source=False)
    assert update(catalogue, 7)[0][-1] == 0
    assert update(catalogue, 8, source=False)[1]['status'] == 'tracking'
    assert update(catalogue, 9)[0][-1] == 0
    assert update(catalogue, 10)[0][-1] == 1  # Repeated evidence can rehabilitate.


def test_confirmed_source_survives_first_failed_window_then_fades():
    catalogue = sky_catalogue.SkySourceCatalogue()
    for frame in range(4):
        update(catalogue, frame)
    update(catalogue, 4, source=False)
    assert catalogue._tracks[-1, 5] == 1
    update(catalogue, 5, source=False)
    assert update(catalogue, 6)[0][-1] == 0.25
    update(catalogue, 7, source=False)
    assert update(catalogue, 8)[0][-1] == 0


def test_new_bright_transient_bypasses_history():
    catalogue = sky_catalogue.SkySourceCatalogue()
    assert update(catalogue, 0, score=6)[0][-1] == 1
    for frame in range(1, 7):
        update(catalogue, frame, source=False)
    assert update(catalogue, 7, score=6)[0][-1] == 1


def test_compact_fixed_residual_needs_repetition_and_discriminating_motion():
    catalogue = sky_catalogue.SkySourceCatalogue()
    values, stationary = [], []
    for frame in range(6):
        weights, diagnostics = update(catalogue, frame, fixed=True, compact=True, score=12)
        values.append(weights[-1])
        stationary.append(diagnostics['stationary_mask'][-1])
    assert values[:3] == [1, 1, 1]
    assert values[3:] == [0.5, 0.25, 0]
    assert stationary == [False, False, False, True, True, True]


def test_proven_sensor_residual_stays_rejected_after_intermittent_absence():
    catalogue = sky_catalogue.SkySourceCatalogue()
    for frame in range(6):
        update(catalogue, frame, fixed=True, compact=True, score=12)
    for frame in range(6, 10):
        update(catalogue, frame, source=False)
    weights, diagnostics = update(catalogue, 10, fixed=True, compact=True, score=12)
    assert weights[-1] == 0 and diagnostics['stationary_mask'][-1]


@pytest.mark.parametrize('spacing', [2, 4])
def test_intermittent_weak_sensor_noise_cannot_restart_grace(spacing):
    catalogue = sky_catalogue.SkySourceCatalogue()
    seen = []
    for frame in range(13):
        present = frame % spacing == 0
        weights, _ = update(catalogue, frame, source=present, fixed=True)
        if present:
            seen.append((frame, weights[-1]))
    assert seen[0][1] == 0.5
    assert all(weight == 0 for frame, weight in seen if frame > 4)
    assert len(catalogue._cold) > 0


def test_rejected_sensor_position_records_observed_not_predicted_coordinates():
    catalogue = sky_catalogue.SkySourceCatalogue()
    update(catalogue, 0)
    for frame in range(1, 6):
        update(catalogue, frame, source=False)
    np.testing.assert_allclose(catalogue._cold[0, :2], [601.3, 527.7])
    # Sky prediction has moved almost ten pixels; the rejection has not.
    assert np.linalg.norm(catalogue._tracks[-1, :2] - catalogue._cold[0, :2]) > 8


def test_first_current_rejection_fades_before_cold_memory_takes_effect():
    catalogue = sky_catalogue.SkySourceCatalogue()
    update(catalogue, 0)
    update(catalogue, 1, source=False)
    update(catalogue, 2)
    update(catalogue, 3, source=False)
    assert update(catalogue, 4)[0][-1] == 0.25
    assert len(catalogue._cold) == 1


def test_cloud_fallback_preserves_remembered_weak_sensor_rejection():
    catalogue = sky_catalogue.SkySourceCatalogue()
    update(catalogue, 0, fixed=True)
    for frame in range(1, 7):
        update(catalogue, frame, source=False)
    points = frame_points(7, fixed=True)[-1:]
    weights, diagnostics = catalogue.weights(points, points, capture_time=1210,
        geometry_key='camera-one', sensor_shape=SHAPE)
    assert diagnostics['status'] == 'motion_fallback'
    assert weights[0] == 0


def test_cold_position_expires_and_a_crossing_real_source_can_confirm():
    for expiry in (False, True):
        catalogue = sky_catalogue.SkySourceCatalogue()
        update(catalogue, 0, fixed=True)
        end = 14 if expiry else 7
        for frame in range(1, end):
            update(catalogue, frame, source=False)
        values = []
        for frame in range(end, end+3):
            points = frame_points(frame)
            points[-1, :2] = [601.3, 527.7] + np.array([1.8, -0.7]) * (frame-end)
            weights, _ = catalogue.weights(points, points, capture_time=1000+30*frame,
                                           geometry_key='camera-one', sensor_shape=SHAPE)
            values.append(weights[-1])
        assert values == ([0.5, 0.5, 1] if expiry else [0, 0, 1])


def test_sensor_cold_record_does_not_override_new_strong_detail():
    catalogue = sky_catalogue.SkySourceCatalogue()
    update(catalogue, 0, fixed=True)
    for frame in range(1, 7):
        update(catalogue, frame, source=False)
    assert update(catalogue, 7, fixed=True, score=6)[0][-1] == 1


def test_failed_motion_suspends_votes_and_preserves_known_fixed_rejection():
    catalogue = sky_catalogue.SkySourceCatalogue()
    for frame in range(3):
        update(catalogue, frame)
    before = catalogue._tracks[:, 2:6].copy()
    for frame in (3, 4):
        points = frame_points(frame)[-1:]
        catalogue.weights(points, points, capture_time=1000+30*frame,
                          geometry_key='camera-one', sensor_shape=SHAPE)
    np.testing.assert_array_equal(catalogue._tracks[:, 2:6], before)
    assert update(catalogue, 5)[0][-1] == 1

    fixed = sky_catalogue.SkySourceCatalogue()
    for frame in range(6):
        update(fixed, frame, fixed=True, compact=True, score=12)
    points = frame_points(6, fixed=True, score=12)[-1:]
    weights, diagnostics = fixed.weights(points, points, capture_time=1180,
        geometry_key='camera-one', sensor_shape=SHAPE, compact=[True])
    assert diagnostics['status'] == 'motion_fallback'
    assert weights[0] == 0 and diagnostics['stationary_mask'][0]

    # Continued captures through clouds must not turn a 120-second inability to
    # register into a geometry reset that forgets a proven sensor residual.
    for frame in range(7, 12):
        weights, diagnostics = fixed.weights(points, points, capture_time=1000+30*frame,
            geometry_key='camera-one', sensor_shape=SHAPE, compact=[True])
        assert weights[0] == 0 and diagnostics['stationary_mask'][0]
    assert update(fixed, 12, fixed=True, compact=True, score=12)[0][-1] == 0


def test_relaxed_support_does_not_create_tracks():
    catalogue = sky_catalogue.SkySourceCatalogue()
    update(catalogue, 0, score=2.2, support_only=True)
    assert len(catalogue._tracks) == len(anchors(0))
    weights, _ = update(catalogue, 1)
    assert len(catalogue._tracks) == len(anchors(1)) + 1
    assert weights[-1] == 0.5


def test_unclassified_bright_or_moving_compact_detail_is_not_stationary_suppressed():
    for fixed, compact in [(True, False), (False, True)]:
        catalogue = sky_catalogue.SkySourceCatalogue()
        for frame in range(6):
            weights, diagnostics = update(catalogue, frame, fixed=fixed, compact=compact, score=12)
            assert weights[-1] == 1
            assert not diagnostics['stationary_mask'][-1]


def test_insufficient_motion_does_not_label_stationary_sources():
    catalogue = sky_catalogue.SkySourceCatalogue()
    points = frame_points(0, score=12)
    for frame in range(8):
        weights, diagnostics = catalogue.weights(points, points, capture_time=1000+30*frame,
            geometry_key='still', sensor_shape=SHAPE, compact=np.ones(len(points), bool))
        np.testing.assert_array_equal(weights, 1)
        assert not diagnostics['stationary_mask'].any()


def test_duplicate_capture_preserves_weights_and_does_not_advance_history():
    catalogue = sky_catalogue.SkySourceCatalogue()
    first, _ = update(catalogue, 0)
    old = catalogue._tracks.copy()
    second, diagnostics = update(catalogue, 0)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(catalogue._tracks, old)
    assert diagnostics['status'] == 'duplicate'


@pytest.mark.parametrize('change', ['geometry', 'backwards', 'gap', 'invalid', 'stacked'])
def test_invalid_context_or_capture_discontinuity_resets_history(change):
    catalogue = sky_catalogue.SkySourceCatalogue()
    for frame in range(4):
        update(catalogue, frame)
    points = frame_points(4)
    kwargs = dict(capture_time=1120, geometry_key='camera-one', sensor_shape=SHAPE)
    if change == 'geometry':
        kwargs['geometry_key'] = 'other-camera'
    elif change == 'backwards':
        kwargs['capture_time'] = 999
    elif change == 'gap':
        kwargs['capture_time'] = 1600
    elif change == 'invalid':
        kwargs['capture_time'] = None
    else:
        kwargs['context_valid'] = False
    weights, _ = catalogue.weights(points, points, **kwargs)
    assert weights[-1] == (1 if change in ('invalid', 'stacked') else 0.5)


def test_configured_long_cadence_is_not_mistaken_for_a_gap():
    catalogue = sky_catalogue.SkySourceCatalogue()
    for frame in range(4):
        weights, diagnostics = update(catalogue, frame, time=1000+180*frame, capture_interval=180)
    assert weights[-1] == 1 and diagnostics['status'] == 'tracking'


def test_registration_failure_preserves_sources_and_recovers():
    catalogue = sky_catalogue.SkySourceCatalogue()
    update(catalogue, 0)
    points = frame_points(1)[-1:]
    weights, diagnostics = catalogue.weights(points, points, capture_time=1030,
                                             geometry_key='camera-one', sensor_shape=SHAPE)
    assert weights[0] == 1 and diagnostics['status'] == 'motion_fallback'
    weights, diagnostics = update(catalogue, 2)
    assert diagnostics['status'] == 'tracking'


def test_tracks_are_bounded_and_expire_by_capture_time(monkeypatch):
    monkeypatch.setattr(sky_catalogue, 'MAX_TRACKS', 200)
    catalogue = sky_catalogue.SkySourceCatalogue()
    update(catalogue, 0)
    rng = np.random.default_rng(17)
    for frame in range(1, 13):
        points = np.vstack((anchors(frame), np.column_stack((rng.uniform(100, 1100, (60, 2)), np.full(60, 3)))))
        catalogue.weights(points, points, capture_time=1000+frame*30,
                          geometry_key='camera-one', sensor_shape=SHAPE)
        assert len(catalogue._tracks) <= 200
    assert catalogue._tracks[:, 6].min() >= 1060


def test_matching_is_one_to_one_and_uses_local_alternative():
    left, right = sky_catalogue._pairs(np.array([[0., 0], [0.2, 0]]),
                                     np.array([[0.1, 0], [0.9, 0]]), 1)
    assert len(left) == len(set(left)) == len(set(right)) == 2


def test_vectorized_uncontested_matches_preserve_greedy_crowded_choices():
    from scipy.spatial import cKDTree

    rng = np.random.default_rng(620)
    for radius in (0.01, 0.5, 1.5, 4):
        previous = rng.uniform(0, 20, (120, 2))
        current = np.vstack((previous[:50] + rng.normal(0, .2, (50, 2)),
                             rng.uniform(0, 20, (100, 2))))
        distances, indices = cKDTree(current).query(previous, k=2, distance_upper_bound=radius)
        left, rank = np.where(np.isfinite(distances))
        right = indices[left, rank]
        expected, used_left, used_right = [], set(), set()
        for index in np.argsort(distances[left, rank], kind='stable'):
            a, b = int(left[index]), int(right[index])
            if a not in used_left and b not in used_right:
                expected.append((a, b))
                used_left.add(a)
                used_right.add(b)
        actual = sky_catalogue._pairs(previous, current, radius)
        np.testing.assert_array_equal(np.column_stack(actual), np.asarray(expected).reshape(-1, 2))


def test_additional_sensor_candidate_fades_without_entering_star_history():
    catalogue = sky_catalogue.SkySourceCatalogue()
    baseline = sky_catalogue.SkySourceCatalogue()
    sensor = np.array([[70., 75., 100.]])
    values, flags = [], []
    for frame in range(6):
        expected, _ = update(baseline, frame)
        weights, diagnostics = update(catalogue, frame, sensor_points=sensor)
        np.testing.assert_array_equal(weights, expected)
        np.testing.assert_array_equal(catalogue._tracks, baseline._tracks)
        np.testing.assert_array_equal(catalogue._anchors, baseline._anchors)
        for actual, original in zip(catalogue._history, baseline._history):
            np.testing.assert_array_equal(actual, original)
        values.append(diagnostics['sensor_weights'][0])
        flags.append(diagnostics['sensor_stationary'][0])
    assert values == [1, 1, 1, 0.5, 0.25, 0]
    assert flags == [False, False, False, True, True, True]


def test_normal_and_extra_stationary_candidates_age_once_per_capture():
    catalogue = sky_catalogue.SkySourceCatalogue()
    for frame in range(6):
        weights, diagnostics = update(catalogue, frame, score=12, fixed=True, compact=True,
                                      sensor_points=[[70, 75, 50]])
        assert weights[-1] == diagnostics['sensor_weights'][0]
        assert diagnostics['stationary_mask'][-1] == diagnostics['sensor_stationary'][0]


def test_moving_single_colour_candidate_is_not_rejected_or_used_as_a_star():
    catalogue = sky_catalogue.SkySourceCatalogue()
    for frame in range(10):
        sensor = np.array([[70., 75., 100.]])
        sensor[:, :2] += np.array([1.8, -0.7])*frame
        _, diagnostics = update(catalogue, frame, sensor_points=sensor)
        assert diagnostics['sensor_weights'][0] == 1
        assert not diagnostics['sensor_stationary'][0]
        assert len(catalogue._tracks) == len(frame_points(frame))


def test_single_colour_source_at_rotation_pole_has_insufficient_motion_for_rejection():
    catalogue = sky_catalogue.SkySourceCatalogue()
    initial = anchors(0)
    for frame in range(12):
        angle = frame * 0.002
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        points = initial.copy()
        points[:, :2] = (initial[:, :2] - 600) @ rotation.T + 600
        _, diagnostics = catalogue.weights(points, points, capture_time=1000+30*frame,
            geometry_key='rotation', sensor_shape=SHAPE, sensor_points=[[600, 600, 100]])
        assert diagnostics['sensor_weights'][0] == 1
        assert not diagnostics['sensor_stationary'][0]


def test_extra_bright_detections_cannot_bootstrap_sky_motion():
    catalogue = sky_catalogue.SkySourceCatalogue()
    for frame in range(4):
        points = frame_points(frame)[-1:]
        weights, diagnostics = catalogue.weights(points, points, capture_time=1000+30*frame,
            geometry_key='one-real-star', sensor_shape=SHAPE, sensor_points=anchors(frame))
        assert weights[0] == 1 and diagnostics['status'] == 'motion_fallback'
        np.testing.assert_array_equal(diagnostics['sensor_weights'], 1)
        assert not diagnostics['sensor_stationary'].any()
        assert not len(catalogue._tracks) and not len(catalogue._fixed) and not catalogue._history


def test_extra_duplicate_capture_reuses_aligned_results_without_learning():
    catalogue = sky_catalogue.SkySourceCatalogue()
    sensors = np.array([[70., 75., 100.], [1140., 80., 100.]])
    for frame in range(4):
        sensors[1, :2] = [1140+1.8*frame, 80-.7*frame]
        first, diagnostics = update(catalogue, frame, sensor_points=sensors)
    before = catalogue._fixed.copy()
    repeated, duplicate = update(catalogue, 3, sensor_points=sensors[::-1])
    np.testing.assert_array_equal(repeated, first)
    np.testing.assert_array_equal(duplicate['sensor_weights'], diagnostics['sensor_weights'][::-1])
    np.testing.assert_array_equal(duplicate['sensor_stationary'], diagnostics['sensor_stationary'][::-1])
    np.testing.assert_array_equal(catalogue._fixed, before)
    assert duplicate['status'] == 'duplicate'


def test_sensor_fallback_keeps_only_prior_proof_and_suspends_fade():
    for initial_frames, expected in [(3, 1), (4, 0.5), (6, 0)]:
        catalogue = sky_catalogue.SkySourceCatalogue()
        for frame in range(initial_frames):
            update(catalogue, frame, sensor_points=[[70, 75, 50]])
        history_and_fade = catalogue._fixed[:, [4, 8, 9]].copy()
        frame = initial_frames
        points = frame_points(frame)[-1:]
        _, diagnostics = catalogue.weights(points, points, capture_time=1000+30*frame,
            geometry_key='camera-one', sensor_shape=SHAPE,
            sensor_points=[[70, 75, 50], [80, 85, 100]])
        assert diagnostics['status'] == 'motion_fallback'
        np.testing.assert_array_equal(diagnostics['sensor_weights'], [expected, 1])
        np.testing.assert_array_equal(diagnostics['sensor_stationary'], [initial_frames >= 4, False])
        np.testing.assert_array_equal(catalogue._fixed[:, [4, 8, 9]], history_and_fade)


@pytest.mark.parametrize('change', ['gap', 'invalid', 'geometry'])
def test_sensor_proof_resets_with_capture_context(change):
    catalogue = sky_catalogue.SkySourceCatalogue()
    for frame in range(6):
        update(catalogue, frame, sensor_points=[[70, 75, 50]])
    points = frame_points(6)
    kwargs = dict(capture_time=1180, geometry_key='camera-one', sensor_shape=SHAPE)
    if change == 'gap':
        kwargs['capture_time'] = 1700
    elif change == 'invalid':
        kwargs['capture_time'] = None
    else:
        kwargs['geometry_key'] = 'other-camera'
    _, diagnostics = catalogue.weights(points, points, sensor_points=[[70, 75, 50]], **kwargs)
    assert diagnostics['sensor_weights'][0] == 1 and not diagnostics['sensor_stationary'][0]


@pytest.mark.parametrize('sensors', [
    [[70, 75, np.nan]], [[70, np.inf, 50]], [[-1, 75, 50]], [[1200, 75, 50]],
    [[70, 75]], 'invalid', np.zeros((sky_catalogue.MAX_POINTS+1, 3)),
])
def test_invalid_extra_table_is_skipped_without_resetting_stars(sensors):
    catalogue = sky_catalogue.SkySourceCatalogue()
    baseline = sky_catalogue.SkySourceCatalogue()
    for frame in range(3):
        update(catalogue, frame)
        update(baseline, frame)
    expected, _ = update(baseline, 3)
    weights, diagnostics = update(catalogue, 3, sensor_points=sensors)
    np.testing.assert_array_equal(weights, expected)
    np.testing.assert_array_equal(catalogue._tracks, baseline._tracks)
    assert diagnostics['sensor_skipped']
    assert not diagnostics['sensor_stationary'].any()
    np.testing.assert_array_equal(diagnostics['sensor_weights'], 1)


def test_optional_empty_extra_table_preserves_normal_results_exactly():
    baseline = sky_catalogue.SkySourceCatalogue()
    supplied = sky_catalogue.SkySourceCatalogue()
    for frame in range(7):
        kwargs = dict(fixed=True, compact=True, score=12)
        expected, first = update(baseline, frame, **kwargs)
        actual, second = update(supplied, frame, sensor_points=np.empty((0, 3)), **kwargs)
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(second['stationary_mask'], first['stationary_mask'])
        np.testing.assert_array_equal(supplied._fixed, baseline._fixed)
        assert not second['sensor_skipped'] and len(second['sensor_weights']) == 0


def test_extra_candidates_respect_the_shared_fixed_track_bound(monkeypatch):
    monkeypatch.setattr(sky_catalogue, 'MAX_TRACKS', 170)
    catalogue = sky_catalogue.SkySourceCatalogue()
    sensors = np.column_stack((np.arange(20)*3+60, np.full(20, 75), np.full(20, 50)))
    for frame in range(4):
        _, diagnostics = update(catalogue, frame, sensor_points=sensors)
        assert len(catalogue._fixed) <= 170 and len(catalogue._tracks) <= 170
        assert len(diagnostics['sensor_weights']) == len(sensors)
