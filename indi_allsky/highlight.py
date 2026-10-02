"""Optional highlight metering and a bounded, colour-preserving shadow lift."""
import logging
from typing import NamedTuple

import cv2
import numpy


logger = logging.getLogger('indi_allsky')


class HighlightMeasurement(NamedTuple):
    full: float
    any: float
    adu: float
    full_next: float = 0.0
    any_next: float = 0.0


def measure(data, mask, bit_depth, threshold=99.0):
    """Largest 8-connected patches, as percentages of the metering mask.

    Measure calibrated, linear pixels before stretching, white balance or
    stacking. The near-full-scale threshold tolerates small calibration offsets;
    it is a clipping proxy, not a claim about the sensor's exact saturation level.
    """
    if mask is None or mask.shape != data.shape[:2]:
        return None
    valid = mask != 0
    count = numpy.count_nonzero(valid)
    if not count:
        return None
    maximum = (1 << bit_depth) - 1
    if data.ndim == 3:
        lowest = data.min(axis=2)
        highest = data.max(axis=2)
        mono = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
    else:
        lowest = highest = data
        mono = data

    def largest(region):
        _, _, stats, _ = cv2.connectedComponentsWithStats(
            (region & valid).astype(numpy.uint8), connectivity=8,
        )
        return float(stats[1:, cv2.CC_STAT_AREA].max()) * 100 / count if len(stats) > 1 else 0.0

    adu = cv2.mean(mono, mask=valid.astype(numpy.uint8))[0] / (1 << (bit_depth - 8))
    cutoff = maximum * threshold / 100.0
    # Predict the next 10% increase from the current linear pixels. This avoids
    # repeatedly crossing a flat cloud/lamp plateau with no attainable deadband.
    return HighlightMeasurement(largest(lowest >= cutoff), largest(highest >= cutoff), adu,
                                largest(lowest >= cutoff / 1.1), largest(highest >= cutoff / 1.1))


def exposure_scale(measurement, target, deviation, settings):
    """Return a bounded request; the selected exposure mode chooses the actuators."""
    full_target = settings.get('FULL_TARGET', 0.8)
    full_dev = settings.get('FULL_DEV', 0.2)
    any_target = settings.get('ANY_TARGET', 2.0)
    any_dev = settings.get('ANY_DEV', 0.4)
    over = measurement.full > full_target + full_dev or measurement.any > any_target + any_dev
    under = measurement.full < full_target - full_dev and measurement.any < any_target - any_dev
    adu = max(measurement.adu, 0.1)
    floor = target / (2 ** settings.get('MAX_BOOST', 2.0))

    # Keep shadows within the permitted lift, including after a scene change.
    if adu < floor * 0.98:
        return min(1.1, floor / adu)
    if over:
        # A small brightness deadband prevents chasing noise at the lift limit.
        if adu <= floor * 1.02:
            logger.info('Highlight target limited by maximum shadow lift')
            return 1.0
        return max(0.9, floor / adu)
    if adu > target + deviation:
        return max(0.9, target / adu)
    # Never lengthen exposure just to create clipping in an otherwise dark sky.
    if under and adu < target - deviation:
        if measurement.full_next > full_target + full_dev or measurement.any_next > any_target + any_dev:
            return 1.0
        return min(1.1, target / adu)
    return 1.0


def compensate(data, bit_depth, adu, target, max_boost):
    """Lift low/mid tones before black clipping, rolling smoothly into highlights.

    A common multiplier per pixel preserves RGB ratios. The curve is linear
    up to half scale in the compensated image, then has a smooth shoulder that
    reaches full scale only for an already-full-scale input. No local contrast
    enhancement or changes to stored stretch/gamma settings are introduced.
    """
    boost = min(2 ** max_boost, max(1.0, target / max(adu, 0.1)))
    if boost == 1.0:
        return data
    maximum = (1 << bit_depth) - 1
    values = numpy.arange(maximum + 1, dtype=numpy.float32) / maximum
    knee = 0.5 / boost
    distance = numpy.maximum(values - knee, 0)
    curve = numpy.where(values <= knee, values * boost,
                        0.5 + boost * distance / (1 + (2 * boost - 1 / (1 - knee)) * distance))
    scale = numpy.divide(curve, values, out=numpy.ones_like(values), where=values > 0)
    peak = data.max(axis=2) if data.ndim == 3 else data
    multiplier = scale[numpy.minimum(peak, maximum)]
    if data.ndim == 3:
        multiplier = multiplier[:, :, None]
    return numpy.clip(numpy.rint(data * multiplier), 0, maximum).astype(data.dtype)
