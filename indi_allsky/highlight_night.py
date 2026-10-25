"""Exposure preference for a measured, localized night Moon exception."""
import math


class NightExposurePreference:
    def __init__(self):
        self.reset()

    def reset(self):
        self.active = False
        self.mode = None
        self.time = 0.0
        self.weight = 0.0
        self.duplicate = False

    def update(self, measurement, mode, settings):
        self.duplicate = False
        if (not mode[0] or not settings.get('ENABLE', False)
                or not settings.get('NIGHT_MOON_EXCLUSION_ENABLE', True)
                or measurement.twilight_weight > 0):
            self.reset()
            return False
        if mode != self.mode:
            self.reset()
            self.mode = mode
        stamp = measurement.twilight_time
        if math.isfinite(stamp) and stamp > 0:
            if self.time and stamp <= self.time:
                self.duplicate = True
                return self.active
            self.time = stamp
        weight, clipped = measurement.night_moon_weight, measurement.night_moon_clipped
        if (not math.isfinite(weight) or not 0 < weight <= 1
                or not math.isfinite(clipped) or clipped < 0):
            self.reset()
            return False
        self.weight = weight
        # Once the real Moon clipping has engaged the policy, clearing that
        # clipping must not alternate between this goal and raw-ADU recovery.
        self.active = self.active or clipped > 0
        return self.active

    def constrain(self, controller, measurement, exposure, gain, target, deviation, scale):
        """Keep the original raw safety ceiling, replacing only its ADU goal."""
        if not self.active:
            return None
        try:
            if not (callable(getattr(controller, 'gain2dB', None))
                    and callable(getattr(controller, 'dB2gain', None))):
                self.reset()
                return None
            low, high = controller.gain2dB(controller.gain_min), controller.gain2dB(controller.gain_max)
            if not all(math.isfinite(value) for value in (low, high)) or low > high:
                raise ValueError('Unavailable gain model')
            minimum = controller._highlight_signal_change(
                controller.exposure_max, controller.gain_min, exposure, gain)
            floor = target / 2 ** controller.config.get('HIGHLIGHT_PROTECTION', {}).get('MAX_BOOST', 2.0)
            if minimum is None or not math.isfinite(minimum) or floor <= 0:
                raise ValueError('Unavailable night signal goal')
            adu = max(measurement.adu, 0.1)
            # Prefer the longest shutter and lowest gain. A genuinely faint
            # sky can still use gain to retain the original shadow floor.
            minimum_signal = math.exp(minimum)
            minimum_adu = minimum_signal * adu
            # Fade the desired signal band, not a fractional cut repeated on
            # every frame. A hold band also avoids one-quantum gain sawtooth.
            low, high = max(minimum_adu, floor * 1.08), max(minimum_adu, floor * 1.12)
            if self.weight < 1:
                low = math.exp((1 - self.weight) * math.log(max(floor, target - deviation))
                               + self.weight * math.log(low))
                high = math.exp((1 - self.weight) * math.log(max(floor, target + deviation))
                                + self.weight * math.log(high))
            if low * (1 - 1e-9) <= adu <= high * (1 + 1e-9):
                goal = 1.0
            else:
                boundary = low if adu < low else high
                goal = minimum_signal if self.weight == 1 and boundary == minimum_adu else boundary / adu
            pending = 0.0
            if controller._highlight_request_pending(exposure, gain):
                pending = controller._highlight_signal_change(
                    controller._expUtils.EXPOSURE_NEXT,
                    controller.effective_gain(controller._expUtils.GAIN_NEXT), exposure, gain)
            if pending is None or not math.isfinite(pending):
                raise ValueError('Unavailable pending signal')
            newest = math.exp(pending)
            if adu * min(1.0, newest) >= floor * .98:
                goal = min(newest * 1.1, max(newest / 1.1, goal))
            # Outside-Moon clipping, predicted headroom and ordinary brightness
            # reductions retain their independently calculated source ceiling.
            return min(scale, goal)
        except Exception:
            # Fixed/legacy gain policies cannot certify a signal-preserving
            # exchange. Leave their ordinary controller untouched.
            self.reset()
            return None

    def reallocate(self, controller, exposure, gain):
        """Trade gain for shutter time without increasing the achieved signal."""
        try:
            maximum, minimum = controller.exposure_max, controller.gain_min
            change = controller._highlight_signal_change(exposure, gain, maximum, minimum)
            if change is None or not math.isfinite(change):
                return exposure, gain
            preferred = controller._calculate_exposure(
                maximum, minimum, maximum * math.exp(change), highlight=True)[:2]
            if self.weight < 1 and preferred[0] > exposure:
                mixed_exposure = exposure * math.exp(self.weight * math.log(preferred[0] / exposure))
                mixed_gain = controller.dB2gain(
                    controller.gain2dB(gain) + 20 * math.log10(exposure / mixed_exposure))
                preferred = controller._calculate_exposure(
                    mixed_exposure, mixed_gain, mixed_exposure, highlight=True)[:2]
            difference = controller._highlight_signal_change(*preferred, exposure, gain)
            if (self.weight == 1 and difference is not None and difference > 1e-6
                    and controller.gain_quantum):
                lower = controller.effective_gain(preferred[1] - controller.gain_quantum)
                if lower >= minimum:
                    preferred = controller._calculate_exposure(
                        preferred[0], lower, preferred[0], highlight=True)[:2]
                    difference = controller._highlight_signal_change(*preferred, exposure, gain)
            if (difference is not None and difference <= 1e-6
                    and preferred[0] >= exposure - .0000005):
                return preferred
        except Exception:
            pass
        return exposure, gain
