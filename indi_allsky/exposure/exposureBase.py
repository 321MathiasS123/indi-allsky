import copy
import functools
import logging
import math

from ..twilight import runtime_weight
from .. import constants
from ..utils import IndiAllSkyExposureUtils
from ..highlight import exposure_decision


logger = logging.getLogger('indi_allsky')


class IndiAllSky_Exposure_Base(object):


    def __init__(self, *args, **kwargs):
        self.config = args[0]
        self.exposure_av = args[1]
        self.gain_av = args[2]
        self.binning_av = args[3]
        self.night_av = args[4]

        self._expUtils = IndiAllSkyExposureUtils(self.config, self.exposure_av, self.gain_av, self.binning_av)

        self._target_adu_found = False
        self._current_adu_target = 0
        self.hist_adu = []
        # Populated from camera metadata only for discrete ISO switch controls.
        self.gain_values = []
        self.reset_highlights()


    @property
    def target_adu_found(self):
        return self._target_adu_found

    @target_adu_found.setter
    def target_adu_found(self, new_target_adu_found):
        self._target_adu_found = bool(new_target_adu_found)


    @property
    def current_adu_target(self):
        return self._current_adu_target


    def compare_exposure(self, adu, exposure, gain):
        self.reset_highlights()
        if adu <= 0.0:
            # ensure we do not divide by zero
            logger.warning('Zero average, setting a default of 0.1')
            adu = 0.1


        if self.night_av[constants.NIGHT_NIGHT]:
            target_adu = self.config['TARGET_ADU']
        else:
            target_adu = self.config['TARGET_ADU_DAY']


        # Brightness when the sun is in view (very short exposures) can change drastically when clouds pass through the view
        # Setting a deviation that is too short can cause exposure flapping
        if exposure < 0.001000:
            # DAY
            adu_dev = float(self.config.get('TARGET_ADU_DEV_DAY', 20))

            target_adu_min = target_adu - adu_dev
            target_adu_max = target_adu + adu_dev
            current_adu_target_min = self.current_adu_target - adu_dev
            current_adu_target_max = self.current_adu_target + adu_dev

            exp_scale_factor = 0.50  # scale exposure calculation
            history_max_vals = 6     # number of entries to use to calculate average
        else:
            # NIGHT
            adu_dev = float(self.config.get('TARGET_ADU_DEV', 10))

            target_adu_min = target_adu - adu_dev
            target_adu_max = target_adu + adu_dev
            current_adu_target_min = self.current_adu_target - adu_dev
            current_adu_target_max = self.current_adu_target + adu_dev

            exp_scale_factor = 1.0  # scale exposure calculation
            history_max_vals = 6    # number of entries to use to calculate average


        if runtime_weight(self.config) is not None:
            previous_target = getattr(self, '_twilight_target', None)
            tracking = getattr(self, '_twilight_tracking', None)
            self._twilight_tracking = None
            if previous_target != target_adu:
                # Samples collected for a different target cannot establish a lock.
                self.target_adu_found = False
                self.hist_adu = []
                if (previous_target and exposure > 0 and target_adu_min <= adu <= target_adu_max
                        and abs(adu - previous_target) <= adu_dev):
                    # Follow a moving target inside the brightness tolerance,
                    # through the existing camera policy. Otherwise AE supplies
                    # the correction below. This avoids tolerance-sized steps.
                    # Carry fractions only once the previous request has taken
                    # effect; queued frames and gain changes start a new track.
                    base = tracking[2] if tracking and tracking[:2] == (exposure, gain) else exposure
                    self.recalculate_exposure(exposure, gain, previous_target * exposure / base, target_adu,
                                              target_adu, target_adu, 1.0)
                    ideal = base * target_adu / previous_target
                    if self.exposure_min <= ideal <= self.exposure_max and self._expUtils.GAIN_NEXT == gain:
                        # Retain requested exposure, gain, and the unrounded ideal.
                        self._twilight_tracking = (self._expUtils.EXPOSURE_NEXT, gain, ideal)
            self._twilight_target = target_adu


        if not self.target_adu_found:
            self.recalculate_exposure(exposure, gain, adu, target_adu, target_adu_min, target_adu_max, exp_scale_factor)
            return adu, 0.0


        self.hist_adu.append(adu)
        self.hist_adu = self.hist_adu[(history_max_vals * -1):]  # remove oldest values, up to history_max_vals

        adu_average = functools.reduce(lambda a, b: a + b, self.hist_adu) / len(self.hist_adu)

        #logger.info('ADU average: %0.2f', adu_average)
        #logger.info('Current target ADU: %0.2f (%0.2f/%0.2f)', self.current_adu_target, current_adu_target_min, current_adu_target_max)
        #logger.info('Current ADU history: (%d) [%s]', len(self.hist_adu), ', '.join(['{0:0.2f}'.format(x) for x in self.hist_adu]))


        ### Need at least x values to continue
        if len(self.hist_adu) < history_max_vals:
            return adu, 0.0


        ### only change exposure when 70% of the values exceed the max or minimum
        if adu_average > current_adu_target_max:
            logger.warning('ADU increasing beyond limits, recalculating next exposure')
            self.target_adu_found = False
        elif adu_average < current_adu_target_min:
            logger.warning('ADU decreasing beyond limits, recalculating next exposure')
            self.target_adu_found = False

        return adu, adu_average



    def recalculate_exposure(self, current_exposure, current_gain, adu, target_adu, target_adu_min, target_adu_max, exp_scale_factor):
        # There might be a race condition here if there is a day/night change but self.target_adu_found == True

        # Until we reach a good starting point, do not calculate a moving average
        if adu <= target_adu_max and adu >= target_adu_min:
            logger.warning('Found target value for exposure')
            self._current_adu_target = copy.copy(adu)
            self.target_adu_found = True
            self.hist_adu = []
            return


        # Scale the exposure up and down based on targets
        if adu > target_adu_max:
            next_exposure = current_exposure - ((current_exposure - (current_exposure * (target_adu / adu))) * exp_scale_factor)
        elif adu < target_adu_min:
            next_exposure = current_exposure - ((current_exposure - (current_exposure * (target_adu / adu))) * exp_scale_factor)
        else:
            next_exposure = current_exposure


        self._set_exposure(current_exposure, current_gain, next_exposure)


    def reset_highlights(self):
        """Discard correction history after a mode change or unusable capture."""
        self._highlight_reduction = 0.0
        self._highlight_previous_reduction = 0.0
        self._highlight_sample = None
        self._highlight_mode = None


    def compare_highlights(self, measurement, exposure, gain):
        """Use this capture's actual settings, never an unapplied queued request."""
        night = self.night_av[constants.NIGHT_NIGHT]
        target = self.config['TARGET_ADU' if night else 'TARGET_ADU_DAY']
        # Match normal ADU control's short-exposure deviation selection.
        deviation = (self.config.get('TARGET_ADU_DEV_DAY', 20) if exposure < 0.001
                     else self.config.get('TARGET_ADU_DEV', 10))
        settings = self.config.get('HIGHLIGHT_PROTECTION', {})
        mode = tuple(self.night_av)
        if mode != self._highlight_mode:
            self.reset_highlights()
        self._highlight_mode = mode
        scale, reason = exposure_decision(measurement, target, deviation, settings)
        if reason in ('full clipping', 'any clipping', 'full+any clipping'):
            sample = (exposure, gain)
            if sample != self._highlight_sample:
                self._highlight_previous_reduction = self._highlight_reduction
                self._highlight_sample = sample
            # Grow correction strength smoothly as settings take effect.
            # Frames already in flight must not compound a pending request.
            reduction = 1.0 - scale
            reduction = min(reduction, (self._highlight_previous_reduction + reduction) / 2)
            self._highlight_reduction = reduction
            scale = 1.0 - reduction
        else:
            self.reset_highlights()
        self.hist_adu = []
        # Keep existing status/telemetry fields useful without the ADU history
        # delay. The reason string and requested multiplier are logged once here.
        self._current_adu_target = measurement.adu
        self.target_adu_found = scale == 1.0
        logger.info('Highlight patches (pre-dark): full %.3f%%, any %.3f%%; calibrated ADU %.2f; exposure request %.3fx; reason: %s',
                    measurement.full, measurement.any, measurement.adu, scale, reason)
        if scale != 1.0:
            self._set_exposure(exposure, gain, exposure * scale, highlight=True)
            if self._expUtils.EXPOSURE_NEXT == exposure and self._expUtils.GAIN_NEXT == gain:
                logger.info('Highlight adjustment limited by exposure/gain settings')
        return measurement.adu, measurement.adu


    def _set_exposure(self, current_exposure, current_gain, next_exposure, highlight=False):
        next_exposure, next_gain, exposure_delta, gain_delta = self.adjust_exposure_gain(current_exposure, current_gain, next_exposure)


        # Do not exceed the gain limits
        if next_gain > self.gain_max:
            next_gain = self.gain_max
        elif next_gain < self.gain_min:
            next_gain = self.gain_min

        if highlight:
            # ISO switches cannot accept the continuous gain requested by the
            # controller. ISO is linear in signal (unlike dB gain); compensate
            # with exposure while retaining the selected mode's limits.
            values = [g for g in self.gain_values if g > 0 and self.gain_min <= g <= self.gain_max]
            if values:
                signal = next_exposure * next_gain
                feasible = [g for g in values if self.exposure_min <= signal / g <= self.exposure_max]
                if feasible:
                    next_gain = min(feasible, key=lambda g: abs(g - next_gain))
                    next_exposure = signal / next_gain
                else:
                    # With fixed exposure, the smallest available ISO step is
                    # the hardware limit on smoothness; do not stall between ISOs.
                    current_signal = current_exposure * current_gain
                    direction = signal - current_signal
                    candidates = [(g, min(self.exposure_max, max(self.exposure_min, signal / g))) for g in values]
                    directional = [(g, e) for g, e in candidates if (e * g - current_signal) * direction > 0]
                    next_gain, next_exposure = min(directional or candidates, key=lambda pair: abs(pair[1] * pair[0] - signal))

            # Shared exposure storage uses whole microseconds. A fractional
            # increase must not truncate back to the same value indefinitely.
            exposure_us = round(next_exposure * 1000000)
            current_us = round(current_exposure * 1000000)
            if next_exposure > current_exposure and exposure_us <= current_us:
                exposure_us = current_us + 1
            elif next_exposure < current_exposure and exposure_us >= current_us:
                exposure_us = current_us - 1
            exposure_us = min(math.floor(self.exposure_max * 1000000),
                              max(math.ceil(self.exposure_min * 1000000), exposure_us))
            # The shared setter truncates; avoid losing a microsecond to binary
            # floating-point error when it converts seconds back to integers.
            next_exposure = math.nextafter(exposure_us / 1000000, math.inf)
            exposure_delta = next_exposure - current_exposure
            gain_delta = next_gain - current_gain


        # Binning
        if self.night_av[constants.NIGHT_NIGHT]:
            if self.night_av[constants.NIGHT_MOONMODE]:
                next_binning = self._expUtils.BINNING_MOONMODE
            else:
                next_binning = self._expUtils.BINNING_NIGHT
        else:
            next_binning = self._expUtils.BINNING_DAY


        ### Check for exposure flapping
        # Flapping is defined when the exposure increases then immediately decreases (or the opposite)
        # and cannot find a stable value.  The result is the image brightness will flash
        #if self._expUtils.EXPOSURE_DELTA > 0 and exposure_delta < 0:
        #    # exposure is decreasing
        #    exposure_offset = exposure_delta / 2
        #    next_exposure -= exposure_offset  # offset will be negative
        #    exposure_delta -= exposure_offset

        #    logger.warning('DETECTED EXPOSURE FLAPPING - Attempting to mitigate by adjusting exposure by %+0.6fs', exposure_offset * -1)
        #elif self._expUtils.EXPOSURE_DELTA < 0 and exposure_delta > 0:
        #    # exposure is increasing
        #    exposure_offset = exposure_delta / 2
        #    next_exposure -= exposure_offset
        #    exposure_delta -= exposure_offset

        #    logger.warning('DETECTED EXPOSURE FLAPPING - Attempting to mitigate by adjusting exposure by %+0.6fs', exposure_offset * -1)


        logger.warning('New calculated exposure: %0.6fs (%+0.6f) @ gain %0.3f (%+0.3f) bin %d', next_exposure, exposure_delta, next_gain, gain_delta, next_binning)
        self._expUtils.EXPOSURE_NEXT = next_exposure
        self._expUtils.EXPOSURE_DELTA = exposure_delta

        self._expUtils.GAIN_NEXT = next_gain
        self._expUtils.GAIN_DELTA = gain_delta

        self._expUtils.BINNING_NEXT = next_binning



    def apply_transition_limits(self):
        """Apply moving limits even when a valid frame needs no AE correction."""
        if runtime_weight(self.config) is None:
            return
        exposure = self._expUtils.EXPOSURE_NEXT
        gain = self._expUtils.GAIN_NEXT
        next_exposure = max(self.exposure_min, min(self.exposure_max, exposure))
        next_gain = max(self.gain_min, min(self.gain_max, gain))
        if next_exposure != exposure:
            # Add the clamp to any correction already made by the AE controller.
            self._expUtils.EXPOSURE_NEXT = next_exposure
            self._expUtils.EXPOSURE_DELTA += next_exposure - exposure
        if next_gain != gain:
            self._expUtils.GAIN_NEXT = next_gain
            self._expUtils.GAIN_DELTA += next_gain - gain


    def adjust_exposure_gain(self, *args):
        raise Exception('Not implemented')

