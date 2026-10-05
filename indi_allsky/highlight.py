"""Optional highlight metering and a bounded, colour-preserving shadow lift."""
import math
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


class HighlightOutput:
    """Slow, optional feedback from the last clean rendered image.

    No accumulated correction: a stale render can hold recovery, but cannot
    request another cut until a capture with the same settings is measured.
    Raw protection and the shadow floor always retain priority.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.active = False
        self.measurement = None
        self.source = None
        self.mode = None

    def observe(self, measurement, exposure, gain, mode, settings):
        if mode != self.mode:
            self.reset()
        self.mode = mode
        self.measurement = measurement
        self.source = (exposure, gain)
        if measurement is None:
            return
        full = settings.get('OUTPUT_FULL_TARGET', 1.5)
        full_dev = settings.get('OUTPUT_FULL_DEV', 0.5)
        any_channel = settings.get('OUTPUT_ANY_TARGET', 2.5)
        any_dev = settings.get('OUTPUT_ANY_DEV', 0.5)
        if measurement.full > full + full_dev or measurement.any > any_channel + any_dev:
            self.active = True
        elif measurement.full < full - full_dev and measurement.any < any_channel - any_dev:
            # Release the extra constraint as soon as output has headroom.
            # Waiting for normal ADU first can strand a fading sky at max lift.
            self.active = False

    def constrain(self, scale, adu, target, deviation, exposure, gain, mode, settings):
        """Only tighten the raw request; never chase a minimum clipped area."""
        if not settings.get('OUTPUT_ENABLE', False) or mode != self.mode:
            self.reset()
            return scale, None
        if not self.active:
            return scale, None
        limit = 1.0
        if self.measurement is None:
            reason = 'await trusted output'
        else:
            matches = (math.isclose(exposure, self.source[0], rel_tol=0, abs_tol=0.0000005)
                       and math.isclose(gain, self.source[1], rel_tol=0, abs_tol=0.0005))
            reason = 'output deadband' if matches else 'await matching output exposure/gain'
            if matches:
                excess = max(
                    self.measurement.full / (settings.get('OUTPUT_FULL_TARGET', 1.5) + settings.get('OUTPUT_FULL_DEV', 0.5)) - 1,
                    self.measurement.any / (settings.get('OUTPUT_ANY_TARGET', 2.5) + settings.get('OUTPUT_ANY_DEV', 0.5)) - 1,
                    0.0,
                )
                limit = 1 - min(0.02, 0.1 * excess)
                if limit < 1:
                    reason = 'output bright patch'
        # Keep one maximum recovery step of brightness above the lift floor.
        # Taper cuts into proportional recovery instead of switching from hold
        # to a 10% rescue below the floor. Trusted raw ADU supplies this reserve,
        # even when output feedback is stale; raw/pending limits still win.
        floor = target / 2 ** settings.get('MAX_BOOST', 2.0)
        reserve = min(target, floor * MAX_EXPOSURE_INCREASE)
        floor_limit = min(MAX_EXPOSURE_INCREASE, reserve / max(adu, 0.1))
        if floor_limit > limit:
            limit, reason = floor_limit, 'output shadow reserve'
        return (limit, reason) if scale > limit else (scale, None)


class HighlightTransition:
    """Retain protection through recovery, with a separate rendering envelope.

    This state survives the exposure controller's short-lived correction history.
    Only trusted captures advance it; elapsed time and camera hangs do not.
    """

    def __init__(self):
        self.reset()
        self._startup = True

    def reset(self):
        # Disabling protection must retain gradual engagement when re-enabled.
        self._startup = False
        self.active = False
        self.trusted = False
        self.reference = None
        self.lift = 0.0
        self.gamma_mix = 0.0
        self.reason = 'ordinary rendering'

    @property
    def phase(self):
        return 'protected' if self.active else ('releasing' if self.lift or self.gamma_mix else 'normal')

    def observe(self, measurement, target, deviation, settings, pending, ceiling, predicted_block, output_needed=False):
        """Decide activity from the calibrated capture, never a rendered stack."""
        self.trusted = True
        full_low = settings.get('FULL_TARGET', 0.8) - settings.get('FULL_DEV', 0.2)
        any_low = settings.get('ANY_TARGET', 2.0) - settings.get('ANY_DEV', 0.4)
        needed = output_needed or measurement.full > full_low or measurement.any > any_low or (predicted_block and not ceiling)
        headroom = (measurement.full_next is not None and measurement.any_next is not None
                    and measurement.full_next <= full_low and measurement.any_next <= any_low)
        clear = measurement.full <= full_low * 0.9 and measurement.any <= any_low * 0.9
        if needed:
            if not self.active:
                # Start from the current appearance, including a partial release.
                self.reference = None
            self.active = True
            self.reason = 'output protection needed' if output_needed else 'clipping protection needed'
        elif self.active:
            if pending:
                self.reason = 'await pending exposure/gain'
            elif not clear or not (ceiling or headroom):
                self.reason = 'await highlight headroom'
            elif measurement.adu < target - deviation and not ceiling:
                self.reason = 'await ordinary ADU recovery'
            else:
                self.active = False
                self.reason = 'ordinary exposure recovered' if not ceiling else 'achievable exposure/gain ceiling'

        if self._startup:
            # A restarted worker may inherit an already reduced exposure. Seed
            # compensation once from its first trusted capture, not from zero.
            # A clear first capture consumes this too, so later entry still eases.
            if self.active:
                self.reference = target
                self.gamma_mix = 1.0
            self._startup = False

    @staticmethod
    def _approach(value, target, cap, epsilon):
        difference = target - value
        if abs(difference) <= epsilon:
            return target
        return value + max(-cap, min(cap, difference * 0.5))

    def render_target(self, adu, target, max_boost):
        """Ease appearance, but immediately compensate actual exposure cuts."""
        adu = max(adu, 0.1)
        if self.active:
            if self.trusted:
                if self.reference is None:
                    # A returning highlight during release must not snap an
                    # already brightened scene down to target in one frame.
                    self.reference = adu * 2 ** self.lift if self.lift else min(target, adu)
                # Ease the brightness reference, not the compensating gain: a
                # slow gain ramp would leave dark dips during exposure cuts.
                if abs(math.log2(target / self.reference)) <= 0.001:
                    self.reference = target
                else:
                    self.reference = 2 ** self._approach(math.log2(self.reference), math.log2(target), 0.04, 0.001)
            if self.reference is not None:
                self.lift = min(max_boost, math.log2(shadow_boost(adu, self.reference, max_boost)))
                return self.reference
        elif self.trusted:
            if self.reference is not None:
                # Account for the exposure recovery in this first release frame
                # before fading; otherwise the preceding darker frame's lift
                # would over-brighten it. Later frames only fade the extra lift.
                self.lift = min(max_boost, math.log2(shadow_boost(adu, self.reference, max_boost)))
                self.reference = None
            # Release extra processing without chasing natural sky changes.
            self.lift = self._approach(self.lift, 0.0, 0.04, 0.002)
        self.lift = min(self.lift, max_boost)
        return adu * 2 ** self.lift

    def gamma(self, normal, protected):
        """Blend power exponents, retaining exact ordinary/protected endpoints."""
        difference = abs(1.0 / protected - 1.0 / normal)
        if self.trusted:
            if difference < 1e-9:
                self.gamma_mix = float(self.active)
            else:
                self.gamma_mix = self._approach(self.gamma_mix, float(self.active), 0.01 / difference, 0.0002 / difference)
        if self.gamma_mix == 0.0:
            return normal
        if self.gamma_mix == 1.0:
            return protected
        return 1.0 / ((1 - self.gamma_mix) / normal + self.gamma_mix / protected)


def _largest_patch(region, valid, count):
    _, _, stats, _ = cv2.connectedComponentsWithStats(
        (region & valid).astype(numpy.uint8), connectivity=8,
    )
    # Component zero is background; disconnected reflections do not add up.
    return float(stats[1:, cv2.CC_STAT_AREA].max()) * 100 / count if len(stats) > 1 else 0.0


def measure_rendered(data, mask):
    """Near-white (all >= 240) and near-clipped (any >= 250) 8-bit patches.

    These describe appearance, not sensor saturation. Meter before overlays
    and compression; there is no linear exposure look-ahead for rendered data.
    """
    if mask is None or mask.shape != data.shape[:2]:
        return None
    valid = mask != 0
    count = numpy.count_nonzero(valid)
    if not count:
        return None
    lowest = data.min(axis=2) if data.ndim == 3 else data
    highest = data.max(axis=2) if data.ndim == 3 else data
    return HighlightMeasurement(_largest_patch(lowest >= 240, valid, count),
                                _largest_patch(highest >= 250, valid, count), 0.0, None, None)


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
        return _largest_patch(region, valid, count)

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
