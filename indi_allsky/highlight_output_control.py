"""Source-anchored feedback from twilight's final display pixels."""
import hashlib
import json
import math


def output_signature(config):
    """Compare saved profiles, without rejecting each interpolated capture."""
    return hashlib.sha256(json.dumps(config, sort_keys=True, default=str).encode()).hexdigest()


class TwilightOutputControl:
    """Bounded appearance correction; encoded levels are not raw radiance.

    Each distinct returned image establishes one absolute signal ceiling.
    Re-reading it cannot compound a cut onto a newer pending camera command.
    Current calibrated ADU translates the source's brightness reserve; early
    raw/channel forecasting remains responsible for rapidly changing light.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.sample = None
        self.anchor = None
        self.active = False

    def observe(self, measurement, exposure, gain, mode):
        if (self.sample is not None and mode == self.sample[3]
                and measurement.twilight_time <= self.sample[0].twilight_time):
            return
        self.sample = (measurement, exposure, gain, mode)
        self.anchor = None

    def limit(self, measurement, exposure, gain, mode, settings, signal_change,
              pending_exposure, pending_gain, target, cadence=20):
        if (not settings.get('ENABLE', False) or not settings.get('TWILIGHT_ENABLE', False)
                or not settings.get('TWILIGHT_OUTPUT_ENABLE', False)
                or not all(math.isfinite(value) for value in measurement)
                or not 0 < measurement.twilight_weight <= 1 or self.sample is None
                or self.sample[3] != mode):
            self.reset()
            return None
        source, source_exposure, source_gain, _ = self.sample
        if not (all(math.isfinite(value) for value in source)
                and 0 < source.twilight_peak <= 1 and source.adu > 0 and measurement.adu > 0
                and 0 <= measurement.twilight_time - source.twilight_time <= min(120, 2 * cadence)):
            self.reset()
            return None
        change = signal_change(exposure, gain, source_exposure, source_gain)
        if change is None or not math.isfinite(change):
            self.reset()
            return None
        if self.anchor is None:
            peak = source.twilight_peak * 255
            # A censored 255 tail still has an area error. Both terms taper to
            # zero at the configured target; neither is a linear-light model.
            factor = min(1.1, max(.98, math.sqrt(settings.get('TWILIGHT_OUTPUT_TARGET', 252) / peak)))
            if source.any > 0:
                factor = min(factor, max(.98, 1 - 2 * source.any / 100))
            # Translate a source-relative brightness ceiling using the actual
            # calibrated capture. Do not multiply an encoded code by E/g.
            ceiling = factor * source.adu / measurement.adu
            pending = signal_change(pending_exposure, pending_gain, exposure, gain)
            if pending is None or not math.isfinite(pending):
                self.reset()
                return None
            # At most one 2% cut from the newest command for this output id.
            ceiling = max(ceiling, .98 * math.exp(pending))
            self.anchor = (exposure, gain, ceiling)
        anchor_exposure, anchor_gain, ceiling = self.anchor
        change = signal_change(exposure, gain, anchor_exposure, anchor_gain)
        if change is None or not math.isfinite(change):
            self.reset()
            return None
        ceiling *= math.exp(-change)
        # Preserve the established lift reserve while a fading sky recovers.
        reserve = target / 2 ** settings.get('MAX_BOOST', 2.0) * 1.1
        ceiling = max(ceiling, min(1.1, reserve / measurement.adu))
        self.active = ceiling < 1.1
        return ceiling
