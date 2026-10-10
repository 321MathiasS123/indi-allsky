"""Bounded coordinate history for display-denoiser point protection.

Only measured coordinates and confidence are retained. No previous image pixels
are combined with the current exposure. Uncertain registration preserves detail.
"""
import cv2
import numpy as np
from scipy.spatial import cKDTree


PROVISIONAL_WEIGHT = 0.5
STRONG_SCORE = 6.0
MATCH_RADIUS = 1.5
MEMORY_SECONDS = 300.0
# Already proven defects may be quiet through long cloud intervals. Only their
# coordinates survive longer; current compact evidence is still required.
SENSOR_MEMORY_SECONDS = 6 * 3600.0
MAX_TRACKS = 65536
MAX_POINTS = 20000


def _pairs(previous, current, radius):
    """Nearest local centroids, with each source used at most once."""
    if not len(previous) or not len(current):
        return np.empty(0, int), np.empty(0, int)
    distances, indices = cKDTree(current).query(previous, k=2,
                                               distance_upper_bound=radius)
    left, rank = np.where(np.isfinite(distances))
    right = indices[left, rank]
    order = np.argsort(distances[left, rank], kind='stable')
    # Most centroids have exactly one available partner on either side. Those
    # independent pairs cannot affect the crowded components' greedy choices.
    uncontested = ((np.bincount(left, minlength=len(previous))[left] == 1)
                   & (np.bincount(right, minlength=len(current))[right] == 1))
    used_left, used_right = set(), set()
    accepted = uncontested.copy()
    for index in order[~uncontested[order]]:
        a, b = int(left[index]), int(right[index])
        if a not in used_left and b not in used_right:
            accepted[index] = True
            used_left.add(a)
            used_right.add(b)
    chosen = order[accepted[order]]
    return left[chosen], right[chosen]


def _features(points, shape):
    height, width = shape
    x, y = ((points - [width / 2, height / 2]) / max(shape)).T
    return np.column_stack((np.ones(len(points)), x, y, x*x, x*y, y*y))


def _anchors(points, shape):
    """Bound work without letting a dense corner dominate the fit."""
    points = points[points[:, 2] >= STRONG_SCORE]
    cells = np.clip((points[:, :2] / np.array(shape[::-1]) * 8).astype(int), 0, 7)
    counts = np.zeros(64, int)
    selected = []
    for index in np.argsort(-points[:, 2], kind='stable'):
        cell = cells[index, 1] * 8 + cells[index, 0]
        if counts[cell] < 12:
            selected.append(index)
            counts[cell] += 1
    return points[selected, :2]


def _distributed(points, shape, minimum=24):
    if len(points) < minimum:
        return False
    span = np.ptp(points, axis=0) / np.array(shape[::-1])
    middle = np.array(shape[::-1]) / 2
    quadrants = (points[:, 0] > middle[0]) + 2 * (points[:, 1] > middle[1])
    return np.all(span > 0.25) and len(np.unique(quadrants)) >= 3


def _motion(previous, current, shape, warm=None):
    """Fit a small, smooth displacement from distributed bright anchors."""
    if not _distributed(previous, shape) or not _distributed(current, shape):
        return None, {'reason': 'insufficient_distributed_anchors'}
    design = _features(previous, shape)
    if warm is None:
        left, right = _pairs(previous, current, 16.0)
        if not _distributed(previous[left], shape):
            return None, {'reason': 'insufficient_bootstrap_matches'}
        affine, inliers = cv2.estimateAffinePartial2D(
            previous[left], current[right], method=cv2.RANSAC,
            ransacReprojThreshold=1.2, maxIters=1000, confidence=0.995)
        if affine is None or inliers.sum() < 24:
            return None, {'reason': 'bootstrap_failed'}
        predicted = np.column_stack((previous, np.ones(len(previous)))) @ affine.T
    else:
        predicted = previous + design @ warm
    coefficient = None
    for radius in (2.5, 1.5, 1.25):
        left, right = _pairs(predicted, current, radius)
        if not _distributed(previous[left], shape):
            return None, {'reason': 'insufficient_motion_matches'}
        matrix = design[left]
        if np.linalg.cond(matrix) > 1000:
            return None, {'reason': 'ill_conditioned_motion'}
        coefficient = np.linalg.lstsq(matrix, current[right] - previous[left], rcond=None)[0]
        predicted = previous + design @ coefficient
    left, right = _pairs(predicted, current, 1.25)
    error = np.linalg.norm(predicted[left] - current[right], axis=1)
    if (not _distributed(previous[left], shape) or len(left) < 0.25 * min(len(previous), len(current))
            or np.median(error) > 0.65 or np.percentile(error, 95) > 1.15):
        return None, {'reason': 'motion_quality_failed'}
    # Reject extrapolation that folds or magnifies the field. A camera move or
    # false match must not become evidence that a real star is sensor-fixed.
    yy, xx = np.mgrid[0:shape[0]:5j, 0:shape[1]:5j]
    grid = np.column_stack((xx.ravel(), yy.ravel()))
    shift = _features(grid, shape) @ coefficient
    if np.max(np.linalg.norm(shift, axis=1)) > 24:
        return None, {'reason': 'excessive_motion'}
    return coefficient, dict(reason='matched', anchors=len(left),
                             median_error=float(np.median(error)),
                             p95_error=float(np.percentile(error, 95)))


class SkySourceCatalogue:
    """One camera's capture-ordered point history; never store image arrays."""

    def __init__(self):
        self.reset()

    def reset(self):
        self._time = None
        self._key = None
        self._shape = None
        self._cadence = None
        self._anchor_time = None
        self._anchors = np.empty((0, 2))
        self._model = None
        self._model_dt = None
        # Three accepted support tables, projected into the latest accepted
        # frame (at most 3 * MAX_POINTS * 2 float64 coordinates). Current evidence
        # votes independently of mutable track identity, without image buffers.
        self._history = []
        # Matching uses projected positions; retain their measured coordinates
        # too so a real star crossing an old defect can demonstrate movement.
        self._observed_history = []
        # Predicted x,y, 4-bit history, age, failed windows, weight, last seen,
        # rejected, last ACTUALLY OBSERVED sensor x,y, last observed score.
        self._tracks = np.empty((0, 11), np.float64)
        # Rejected weak sensor positions and last actual sighting. These remain
        # fixed when the separate sky tracks move, so a gap cannot renew grace.
        self._cold = np.empty((0, 3), np.float64)
        # fixed x,y, predicted sky x,y, 6-bit history, first time, last seen,
        # maximum displacement from initial sensor position, proven, fade weight
        self._fixed = np.empty((0, 10), np.float64)
        self._last_points = np.empty((0, 3))
        self._last_weights = np.empty(0, np.float32)
        self._last_stationary = np.empty(0, bool)
        self._last_sensor_points = np.empty((0, 3))
        self._last_sensor_weights = np.empty(0, np.float32)
        self._last_sensor_stationary = np.empty(0, bool)

    def weights(self, points, support_points, *, capture_time, geometry_key,
                context_valid=True, capture_interval=None, sensor_shape=None,
                compact=None, sensor_points=None):
        """Return current-point weights and diagnostics; inputs are x,y,score.

        Capture time is the exposure midpoint, not processing time. The caller
        invalidates context for stacking or undated previews. Compact flags are
        required before a persistent sensor-position residual can lose strong
        protection. Exposure/gain changes alone must not change geometry_key.
        Voting uses the current plus three accepted observations. Unregistered
        cloud frames suspend voting, so these need not be consecutive captures.
        Optional sensor_points are separate compact single-colour candidates.
        They participate only in sensor-position proof, never star protection,
        motion fitting or voting. The caller excludes co-located normal points.
        """
        points = np.asarray(points, dtype=np.float64)
        support = np.asarray(support_points, dtype=np.float64)
        ones = np.ones(len(points), np.float32)
        try:
            sensors = (np.empty((0, 3)) if sensor_points is None
                       else np.asarray(sensor_points, dtype=np.float64))
            sensor_count = len(sensors) if sensors.ndim == 2 and len(sensors) <= MAX_POINTS else 0
            sensor_valid = (sensors.ndim == 2 and sensors.shape[1] == 3
                            and len(sensors) <= MAX_POINTS and np.isfinite(sensors).all())
        except (TypeError, ValueError, OverflowError):
            sensor_count, sensor_valid = 0, False
        sensor_result = dict(sensor_stationary=np.zeros(sensor_count, bool),
                             sensor_weights=np.ones(sensor_count, np.float32),
                             sensor_skipped=not sensor_valid)
        if not sensor_valid:
            sensors = np.empty((0, 3))
        try:
            timestamp = float(capture_time)
            shape = tuple(int(value) for value in sensor_shape)
            valid = (context_valid and geometry_key is not None and np.isfinite(timestamp)
                     and len(shape) == 2 and min(shape) > 0 and points.ndim == support.ndim == 2
                     and points.shape[1] == support.shape[1] == 3
                     and max(len(points), len(support)) <= MAX_POINTS
                     and np.isfinite(points).all() and np.isfinite(support).all())
        except (TypeError, ValueError, OverflowError):
            valid = False
        if not valid:
            self.reset()
            return ones, dict(status='invalid_context', stationary_mask=np.zeros(len(points), bool), **sensor_result)
        if len(sensors) and (np.any(sensors[:, :2] < 0)
                             or np.any(sensors[:, :2] >= np.array(shape[::-1]))):
            sensors = np.empty((0, 3))
            sensor_result['sensor_skipped'] = True
        compact = np.zeros(len(points), bool) if compact is None else np.asarray(compact, bool)
        if compact.shape != (len(points),):
            self.reset()
            return ones, dict(status='invalid_compact_flags', stationary_mask=np.zeros(len(points), bool), **sensor_result)
        # Every protected candidate must exist in the measured support table.
        if len(points):
            pi, si = _pairs(points[:, :2], support[:, :2], 0.01)
            if len(pi) != len(points):
                self.reset()
                return ones, dict(status='incomplete_support', stationary_mask=np.zeros(len(points), bool), **sensor_result)
            point_support = np.empty(len(points), int)
            point_support[pi] = si
        else:
            point_support = np.empty(0, int)
        if self._time is not None and geometry_key == self._key and shape == self._shape and timestamp == self._time:
            result = np.where(points[:, 2] >= STRONG_SCORE, 1.0, PROVISIONAL_WEIGHT).astype(np.float32)
            old, new = _pairs(self._last_points[:, :2], points[:, :2], 0.01)
            result[new] = self._last_weights[old]
            stationary = np.zeros(len(points), bool)
            stationary[new] = self._last_stationary[old]
            old, new = _pairs(self._last_sensor_points[:, :2], sensors[:, :2], 0.01)
            sensor_result['sensor_weights'][new] = self._last_sensor_weights[old]
            sensor_result['sensor_stationary'][new] = self._last_sensor_stationary[old]
            return result, dict(status='duplicate', tracks=len(self._tracks), stationary_mask=stationary,
                                **sensor_result)
        try:
            expected = float(capture_interval)
            if not np.isfinite(expected) or expected <= 0:
                expected = None
        except (TypeError, ValueError):
            expected = None
        expected = expected or self._cadence or 30.0
        if (self._time is not None and (geometry_key != self._key or shape != self._shape
                or timestamp < self._time or timestamp - self._time > max(120, 4 * expected))):
            self.reset()
        if self._time is not None:
            step = timestamp - self._time
            self._cadence = step if self._cadence is None else 0.8 * self._cadence + 0.2 * step
        self._key, self._shape = geometry_key, shape
        self._cold = self._cold[self._cold[:, 2] >= timestamp - MEMORY_SECONDS]
        lifetime = np.where(self._fixed[:, 8] != 0, SENSOR_MEMORY_SECONDS, MEMORY_SECONDS)
        self._fixed = self._fixed[self._fixed[:, 6] >= timestamp - lifetime]
        anchors = _anchors(support, shape)
        motion_info = {'reason': 'first_frame'}
        if (self._anchor_time is not None and _distributed(anchors, shape)
                and (timestamp - self._anchor_time > max(120, 4 * expected)
                     or not _distributed(self._anchors, shape))):
            # A cloud interval is not a camera/context reset. Re-bootstrap the
            # moving catalogue, retaining remembered sensor-position rejection.
            self._anchor_time = None
            self._model = self._model_dt = None
            self._tracks = np.empty((0, 11), np.float64)
            self._history = []
            self._observed_history = []
            motion_info = {'reason': 'motion_rebootstrap'}
        if self._anchor_time is None and not _distributed(anchors, shape):
            return self._fallback(points, compact, timestamp,
                                  {'reason': 'insufficient_distributed_anchors'}, sensors, sensor_result)
        if self._anchor_time is not None:
            dt = timestamp - self._anchor_time
            warm = self._model * (dt / self._model_dt) if self._model is not None else None
            model, motion_info = _motion(self._anchors, anchors, shape, warm)
            if model is None and warm is not None:
                model, motion_info = _motion(self._anchors, anchors, shape)
            if model is None:
                return self._fallback(points, compact, timestamp, motion_info, sensors, sensor_result)
            self._tracks[:, :2] += _features(self._tracks[:, :2], shape) @ model
            unproven = self._fixed[:, 8] == 0
            self._fixed[unproven, 2:4] += _features(self._fixed[unproven, 2:4], shape) @ model
            for previous in self._history:
                previous += _features(previous, shape) @ model
            self._model, self._model_dt = model, dt
        self._anchors, self._anchor_time = anchors, timestamp
        current_history = np.ones(len(points), np.uint8)
        moving = np.zeros(len(points), bool)
        for age, (previous, observed) in enumerate(zip(reversed(self._history),
                                                     reversed(self._observed_history)), 1):
            current, old = _pairs(points[:, :2], previous, MATCH_RADIUS)
            current_history[current] |= 1 << age
            moving[current] |= np.linalg.norm(points[current, :2] - observed[old], axis=1) > 0.75
        popcount = np.array([int(value).bit_count() for value in range(16)])
        current_passed = popcount[current_history] >= 3
        advance = 1
        self._tracks = self._tracks[self._tracks[:, 6] >= timestamp - MEMORY_SECONDS]
        tracks = self._tracks
        tracks[:, 2] = (tracks[:, 2].astype(np.uint8) << advance) & 15
        tracks[:, 3] = np.minimum(4, tracks[:, 3] + advance)
        old, new = _pairs(tracks[:, :2], support[:, :2], MATCH_RADIUS)
        tracks[old, :2] = support[new, :2]
        tracks[old, 2] += 1
        tracks[old, 6] = timestamp
        tracks[old, 8:11] = support[new]
        support_track = np.full(len(support), -1, int)
        support_track[new] = old
        remaining = point_support[support_track[point_support] < 0]
        capacity = MAX_TRACKS - len(tracks)
        if len(remaining) > capacity:
            remaining = remaining[np.argsort(-support[remaining, 2], kind='stable')[:capacity]]
        added = np.zeros((len(remaining), 11))
        added[:, :2] = support[remaining, :2]
        added[:, 2:4] = 1
        added[:, 5] = PROVISIONAL_WEIGHT
        added[:, 6] = timestamp
        added[:, 8:11] = support[remaining]
        support_track[remaining] = np.arange(len(tracks), len(tracks) + len(remaining))
        tracks = np.concatenate((tracks, added))
        indices = support_track[point_support]
        tracked = indices >= 0
        tracks[indices[tracked], 2] = current_history[tracked]
        counts = popcount[tracks[:, 2].astype(int)]
        mature = tracks[:, 3] >= 4
        passed = (tracks[:, 3] >= 3) & (counts >= 3) & ((tracks[:, 2].astype(int) & 1) != 0)
        # A closer stale track cannot veto three directly observed supports.
        # Conversely, inherited track bits cannot invent missing evidence.
        passed[indices[tracked]] = current_passed[tracked]
        tracks[passed, 4] = 0
        tracks[passed, 7] = 0
        tracks[mature & ~passed, 4] += 1
        tracks[passed, 5] = 1
        fading = mature & ~passed & (tracks[:, 4] >= 2)
        newly_rejected = fading & (tracks[:, 7] == 0) & (tracks[:, 10] < STRONG_SCORE)
        tracks[fading, 7] = 1
        # Two downward steps for provisional points, at most three for full
        # protection. Reappearance never resets a rejected track to new grace.
        tracks[fading, 5] = np.where(tracks[fading, 5] > 0.25, tracks[fading, 5] / 2, 0)
        self._tracks = tracks
        result = np.full(len(points), PROVISIONAL_WEIGHT, np.float32)
        result[tracked] = tracks[indices[tracked], 5]
        result[current_passed] = 1
        weak = np.flatnonzero(points[:, 2] < STRONG_SCORE)
        cold, current = _pairs(self._cold[:, :2], points[weak, :2], 0.75)
        current = weak[current]
        self._cold[cold, 2] = timestamp
        # A real star crossing a rejected sensor position may still establish
        # its own moving history; a single reappearance cannot reset probation.
        blocked = current[~current_passed[current]]
        result[blocked] = 0
        tracked_blocked = indices[blocked][indices[blocked] >= 0]
        tracks[tracked_blocked, 5] = 0
        tracks[tracked_blocked, 7] = 1
        # Record after using earlier rejection memory: the first failed point
        # gets its downward fade, rather than immediately vetoing itself.
        self._remember_rejected(tracks[newly_rejected][:, [8, 9, 6]])
        result[points[:, 2] >= STRONG_SCORE] = 1
        combined = np.concatenate((points, sensors)) if len(sensors) else points
        # A fixed hot pixel can pass sky matching near a slow rotation pole.
        # Exempt crossing stars only with independently observed movement too.
        fixed_candidates = compact & ~(current_passed & moving)
        flags = (np.concatenate((fixed_candidates, np.ones(len(sensors), bool)))
                 if len(sensors) else fixed_candidates)
        sky_confirmed = (np.concatenate((current_passed, np.zeros(len(sensors), bool)))
                         if len(sensors) else current_passed)
        stationary_all, weight_all = self._stationary(combined, flags, timestamp, advance, sky_confirmed)
        stationary, stationary_weight = stationary_all[:len(points)], weight_all[:len(points)]
        if len(sensors):
            sensor_result['sensor_stationary'] = stationary_all[len(points):]
            sensor_result['sensor_weights'] = weight_all[len(points):]
        result[stationary] = np.minimum(result[stationary], stationary_weight[stationary])
        self._time = timestamp
        self._last_points, self._last_weights = points.copy(), result.copy()
        self._last_stationary = stationary.copy()
        self._last_sensor_points = sensors.copy()
        self._last_sensor_weights = sensor_result['sensor_weights'][:len(sensors)].copy()
        self._last_sensor_stationary = sensor_result['sensor_stationary'][:len(sensors)].copy()
        self._history.append(support[:, :2].copy())
        self._history = self._history[-3:]
        self._observed_history.append(support[:, :2].copy())
        self._observed_history = self._observed_history[-3:]
        return result, dict(status='tracking', motion=motion_info, tracks=len(tracks),
                            sensor_tracks=len(self._fixed), confirmed=int(np.count_nonzero(result == 1)),
                            rejected_sensor_positions=len(self._cold),
                            provisional=int(np.count_nonzero((result > 0) & (result < 1))),
                            suppressed=int(np.count_nonzero(result == 0)),
                            stationary=int(stationary.sum()), stationary_mask=stationary, **sensor_result)

    def _fallback(self, points, compact, timestamp, motion_info, sensors, sensor_result):
        """Suspend votes, keeping sensor rejection that needs no new sky fit."""
        weights = np.ones(len(points), np.float32)
        combined = np.concatenate((points, sensors)) if len(sensors) else points
        flags = np.concatenate((compact, np.ones(len(sensors), bool))) if len(sensors) else compact
        stationary_all = np.zeros(len(combined), bool)
        stationary_weights = np.ones(len(combined), np.float32)
        proven = np.flatnonzero(self._fixed[:, 8] != 0)
        selected = np.flatnonzero(flags)
        fixed, current = _pairs(self._fixed[proven, :2], combined[selected, :2], 0.75)
        # Established sensor proof needs no new sky fit. Advance only when the
        # defect is detected again, never for an absent or duplicate capture.
        matched = proven[fixed]
        fade = self._fixed[matched, 9]
        self._fixed[matched, 9] = np.where(fade > 0.25, fade / 2, 0)
        stationary_all[selected[current]] = True
        stationary_weights[selected[current]] = self._fixed[matched, 9]
        stationary = stationary_all[:len(points)]
        if len(sensors):
            sensor_result['sensor_stationary'] = stationary_all[len(points):]
            sensor_result['sensor_weights'] = stationary_weights[len(points):]
        weak = np.flatnonzero(points[:, 2] < STRONG_SCORE)
        cold, recurrent = _pairs(self._cold[:, :2], points[weak, :2], 0.75)
        weights[weak[recurrent]] = 0
        self._cold[cold, 2] = timestamp
        weights[stationary] = np.minimum(weights[stationary], stationary_weights[:len(points)][stationary])
        self._fixed[matched, 6] = timestamp
        self._time = timestamp
        self._last_points, self._last_weights = points.copy(), weights.copy()
        self._last_stationary = stationary.copy()
        self._last_sensor_points = sensors.copy()
        self._last_sensor_weights = sensor_result['sensor_weights'][:len(sensors)].copy()
        self._last_sensor_stationary = sensor_result['sensor_stationary'][:len(sensors)].copy()
        return weights, dict(status='motion_fallback', motion=motion_info, tracks=len(self._tracks),
                             stationary_mask=stationary, **sensor_result)

    def _remember_rejected(self, positions):
        if not len(positions):
            return
        positions = positions[np.argsort(-positions[:, 2], kind='stable')]
        _, unique = np.unique(positions[:, :2], axis=0, return_index=True)
        positions = positions[unique]
        old, new = _pairs(self._cold[:, :2], positions[:, :2], 0.75)
        self._cold[old, 2] = np.maximum(self._cold[old, 2], positions[new, 2])
        unmatched = np.ones(len(positions), bool)
        unmatched[new] = False
        added = positions[unmatched][:MAX_TRACKS - len(self._cold)]
        self._cold = np.concatenate((self._cold, added))

    def _stationary(self, points, compact, timestamp, advance, sky_confirmed=None):
        """Require repeated compact detections despite clearly moving sky."""
        lifetime = np.where(self._fixed[:, 8] != 0, SENSOR_MEMORY_SECONDS, MEMORY_SECONDS)
        fixed = self._fixed[self._fixed[:, 6] >= timestamp - lifetime]
        fixed[:, 4] = (fixed[:, 4].astype(np.uint8) << min(advance, 6)) & 63
        selected = np.flatnonzero(compact)
        old, new = _pairs(fixed[:, :2], points[selected, :2], 0.75)
        current = selected[new]
        if sky_confirmed is not None:
            # A slow star can round to one pixel for several captures. Once
            # sky-confirmed, an old inactive defect must not erase that crossing
            # or renew its own proof from the star's coincident observations.
            crossing = ((fixed[old, 8] != 0) & (fixed[old, 6] < timestamp - MEMORY_SECONDS)
                        & sky_confirmed[current])
            old, current = old[~crossing], current[~crossing]
        fixed[old, 4] += 1
        fixed[old, 6] = timestamp
        deviation = np.linalg.norm(fixed[old, :2] - points[current, :2], axis=1)
        fixed[old, 7] = np.maximum(fixed[old, 7], deviation)
        counts = np.array([int(value).bit_count() for value in range(64)])[fixed[old, 4].astype(int)]
        movement = np.linalg.norm(fixed[old, 2:4] - fixed[old, :2], axis=1)
        suspicious = ((counts >= 4) & (movement > 4.5) & (fixed[old, 7] <= 0.6)
                      & (timestamp - fixed[old, 5] >= 60))
        fixed[old[suspicious], 8] = 1
        proven = fixed[:, 8] != 0
        fixed[proven, 9] = np.where(fixed[proven, 9] > 0.25, fixed[proven, 9] / 2, 0)
        result = np.zeros(len(points), bool)
        result[current] = fixed[old, 8] != 0
        weights = np.ones(len(points), np.float32)
        weights[current] = fixed[old, 9]
        unmatched = np.ones(len(selected), bool)
        unmatched[new] = False
        selected = selected[unmatched][:MAX_TRACKS - len(fixed)]
        added = np.zeros((len(selected), 10))
        added[:, :2] = added[:, 2:4] = points[selected, :2]
        added[:, 4] = 1
        added[:, 5:7] = timestamp
        added[:, 9] = 1
        self._fixed = np.concatenate((fixed, added))
        return result, weights
