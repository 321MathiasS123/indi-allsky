"""Optional highlight metering and a bounded, colour-preserving shadow lift."""
from typing import NamedTuple

import cv2
import numpy


# The look-ahead mask must cover the largest increase the controller can ask for.
MAX_EXPOSURE_INCREASE = 1.1


class HighlightMeasurement(NamedTuple):
    """Mask-area percentages and an average on the common 8-bit ADU scale.

    ``full`` requires every channel; ``any`` requires at least one. Each area
    is its largest connected patch, not the sum of all clipped pixels.
    ``*_next`` predicts those patches after the maximum exposure increase.
    """
    full: float
    any: float
    adu: float
    full_next: float = 0.0
    any_next: float = 0.0


def measure(data, mask, bit_depth, threshold=99.0):
    """Largest 8-connected patches, as percentages of the metering mask.

    Measure linear pixels before dark/black-level subtraction, stretching,
    white balance or stacking. Replace the returned ADU with calibrated
    brightness before controlling exposure. The threshold is a clipping proxy,
    not a claim about the sensor's exact saturation level.
    """
    if mask is None or mask.shape != data.shape[:2]:
        return None
    valid = mask != 0
    count = numpy.count_nonzero(valid)
    if not count:
        return None
    maximum = (1 << bit_depth) - 1
    if data.ndim == 3:
        # Min/max over channels distinguishes white clipping from lost colour.
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
        # Component zero is background; disconnected reflections do not add up.
        return float(stats[1:, cv2.CC_STAT_AREA].max()) * 100 / count if len(stats) > 1 else 0.0

    adu = cv2.mean(mono, mask=valid.astype(numpy.uint8))[0] / (1 << (bit_depth - 8))
    cutoff = maximum * threshold / 100.0
    # Predict the next 10% increase from the current linear pixels. This avoids
    # repeatedly crossing a flat cloud/lamp plateau with no attainable deadband.
    full = largest(lowest >= cutoff)
    full_next = largest(lowest >= cutoff / MAX_EXPOSURE_INCREASE)
    if data.ndim == 3:
        any_channel = largest(highest >= cutoff)
        any_next = largest(highest >= cutoff / MAX_EXPOSURE_INCREASE)
    else:
        # The two definitions coincide for monochrome; avoid two full-image scans.
        any_channel, any_next = full, full_next
    return HighlightMeasurement(full, any_channel, adu, full_next, any_next)


def exposure_decision(measurement, target, deviation, settings):
    """Return an exposure multiplier and diagnostic reason for this capture.

    Either upper clipping limit can reduce exposure. Recovery toward target ADU
    needs both lower limits and low ADU; it never aims to create clipping.
    The shadow floor takes priority when the frame would need excessive lift.
    The selected exposure mode translates this request into exposure and gain.
    """
    full_target = settings.get('FULL_TARGET', 0.8)
    full_dev = settings.get('FULL_DEV', 0.2)
    any_target = settings.get('ANY_TARGET', 2.0)
    any_dev = settings.get('ANY_DEV', 0.4)
    full_limit, any_limit = full_target + full_dev, any_target + any_dev
    full_over, any_over = measurement.full > full_limit, measurement.any > any_limit
    under = measurement.full < full_target - full_dev and measurement.any < any_target - any_dev
    adu = max(measurement.adu, 0.1)
    floor = target / (2 ** settings.get('MAX_BOOST', 2.0))

    # Keep shadows within the permitted lift, including after a scene change.
    if adu < floor * 0.98:
        recovery_target = floor
        if under and measurement.full_next <= full_limit and measurement.any_next <= any_limit:
            # With highlight headroom, recover toward the normal ADU band on
            # both sides of the floor. A darker frame must not ask for less
            # recovery merely because it crossed the maximum-lift boundary.
            recovery_target = max(floor, target - deviation)
        return min(MAX_EXPOSURE_INCREASE, recovery_target / adu), 'recover shadow floor'
    if full_over or any_over:
        # A small brightness deadband prevents chasing noise at the lift limit.
        if adu <= floor * 1.02:
            return 1.0, 'shadow floor'
        reason = 'full+any clipping' if full_over and any_over else ('full clipping' if full_over else 'any clipping')
        # Taper corrections to zero at the upper limits. Clipped area is not
        # linear in exposure, so use a conservative gain and cap the reduction.
        excess = max(measurement.full / full_limit - 1, measurement.any / any_limit - 1)
        reduction = min(0.2, 0.5 * excess)
        return max(1.0 - reduction, floor / adu), reason
    if adu > target + deviation:
        return max(0.9, (target + deviation) / adu), 'ADU above band'
    # Never lengthen exposure just to create clipping in an otherwise dark sky.
    if under and adu < target - deviation:
        if measurement.full_next > full_limit or measurement.any_next > any_limit:
            return 1.0, 'predicted clipping on increase'
        return min(MAX_EXPOSURE_INCREASE, (target - deviation) / adu), 'ADU below band'
    return 1.0, 'hold within control limits'


def shadow_boost(adu, target, max_boost):
    """Convert the stop limit into a linear lift; never darken a bright image."""
    return min(2 ** max_boost, max(1.0, target / max(adu, 0.1)))


def compensate(data, bit_depth, adu, target, max_boost):
    """Lift low/mid tones before black clipping, rolling smoothly into highlights.

    A common multiplier per pixel preserves RGB ratios. The curve is linear
    up to half scale in the compensated image, then has a smooth shoulder that
    reaches full scale only for an already-full-scale input. No local contrast
    enhancement or changes to stored stretch/gamma settings are introduced.
    """
    boost = shadow_boost(adu, target, max_boost)
    if boost == 1.0:
        return data
    maximum = (1 << bit_depth) - 1
    # Index the LUT by the brightest channel, then apply the same multiplier to
    # every channel. Separate channel curves would change colour ratios.
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
