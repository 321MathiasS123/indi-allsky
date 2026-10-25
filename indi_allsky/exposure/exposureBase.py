import copy
import functools
import logging
import math

from .. import constants
from ..utils import IndiAllSkyExposureUtils
from ..gain import gain_limits, quantize_gain
from ..highlight import MAX_EXPOSURE_INCREASE, HighlightOutput, HighlightTransition, exposure_decision
from ..highlight_output_control import TwilightOutputControl


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
        self.gain_quantum = 0.0
        self.highlight_transition = HighlightTransition()
        self.highlight_output = HighlightOutput()
        self.twilight_output = TwilightOutputControl()
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
        self.highlight_output.reset()
        self.twilight_output.reset()
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
        self._highlight_slew = None
        self._twilight_previous = None
        self._twilight_prediction_previous = None
        self._twilight_slew_state = None


    def compare_highlights(self, measurement, exposure, gain):
        """Use this capture's actual settings, never an unapplied queued request."""
        night = self.night_av[constants.NIGHT_NIGHT]
        target = self.config['TARGET_ADU' if night else 'TARGET_ADU_DAY']
        # Match normal ADU control's short-exposure deviation selection.
        deviation = (self.config.get('TARGET_ADU_DEV_DAY', 20) if exposure < 0.001
                     else self.config.get('TARGET_ADU_DEV', 10))
        settings = self.config.get('HIGHLIGHT_PROTECTION', {})
        mode = tuple(self.night_av)
        twilight_previous = self._twilight_previous
        twilight_prediction_previous = self._twilight_prediction_previous
        twilight_slew = self._twilight_slew_state
        cadence = max(exposure, self.config.get('EXPOSURE_PERIOD' if night else 'EXPOSURE_PERIOD_DAY', 15), 0.001)
        twilight_enabled = settings.get('ENABLE', False) and settings.get('TWILIGHT_ENABLE', False)
        twilight_valid_time = math.isfinite(measurement.twilight_time) and measurement.twilight_time > 0
        twilight_active = twilight_enabled and twilight_valid_time and 0 < measurement.twilight_weight <= 1
        if (twilight_slew is not None and (not twilight_enabled or twilight_slew[0] != mode
                or not twilight_valid_time
                or measurement.twilight_time - twilight_slew[1] > max(120, 5 * cadence))):
            twilight_slew = None
        twilight_duplicate = twilight_slew is not None and measurement.twilight_time <= twilight_slew[1]
        # Keep this decision's prior strength across the clipping-history
        # resets below. External resets and mode changes must still discard it.
        previous_scale = (self._highlight_slew[1] if self._highlight_slew is not None
                          and self._highlight_slew[0] == mode else None)
        if mode != self._highlight_mode:
            self.reset_highlights()
        self._highlight_mode = mode
        scale, reason = exposure_decision(measurement, target, deviation, settings)
        # ADU is linear signal, unlike clipped patch area. Retain its absolute
        # target separately from the source-relative highlight safety limit.
        response_target = (exposure_decision(measurement, target, deviation, settings, bounded=False)[0]
                           if reason in ('ADU below band', 'ADU above band', 'recover shadow floor') else None)
        # This also covers recovery limited to the shadow floor by prediction.
        predicted_block = (measurement.adu < target - deviation
                           and (measurement.full_next > settings.get('FULL_TARGET', 0.8) + settings.get('FULL_DEV', 0.2)
                                or measurement.any_next > settings.get('ANY_TARGET', 2.0) + settings.get('ANY_DEV', 0.4)))
        clipping_urgency = 0.0
        if reason in ('full clipping', 'any clipping', 'full+any clipping'):
            sample = (exposure, gain)
            if sample != self._highlight_sample:
                self._highlight_previous_reduction = self._highlight_reduction
                self._highlight_sample = sample
            # Keep gentle corrections smooth, but progressively release their
            # damping beyond a 20% cut. Severe clipping must not wait for a
            # long-exposure ramp to catch up. The shadow floor still bounds it.
            reduction = 1.0 - scale
            clipping_urgency = max(0.0, min(1.0, (reduction - 0.2) / 0.2))
            smoothed = min(reduction, (self._highlight_previous_reduction + reduction) / 2)
            reduction = smoothed + (reduction - smoothed) * clipping_urgency
            self._highlight_reduction = reduction
            scale = 1.0 - reduction
        else:
            self.reset_highlights()
        # Crossing a clipping limit must not weaken an ADU correction. Apply
        # this after clipping's smoothing so the ordinary brightness ceiling
        # still wins when it calls for the stronger (bounded) reduction.
        if measurement.adu > target + deviation:
            adu_scale = max(0.9, (target + deviation) / measurement.adu)
            if adu_scale < scale:
                scale, reason = adu_scale, reason + ' + ADU above band'
        twilight_output_active = twilight_active and settings.get('TWILIGHT_OUTPUT_ENABLE', False)
        scale, output_reason = ((scale, None) if twilight_output_active else self.highlight_output.constrain(
            scale, measurement.adu, target, deviation, exposure, gain, mode, settings))
        if output_reason:
            reason += ' + ' + output_reason
            response_target = None  # Output constraints retain their own bounds.
        twilight_needed = False
        twilight_prediction_limit = None
        if twilight_active and math.isfinite(measurement.adu) and measurement.adu > 0:
            forecast = 1.0
            threatened = False
            channels = (measurement.twilight_blue, measurement.twilight_green, measurement.twilight_red)
            raw_channels = (measurement.twilight_blue_raw, measurement.twilight_green_raw, measurement.twilight_red_raw)
            ceiling = settings.get('THRESHOLD', 99.0) / 100.0
            raw_ceiling = min(ceiling, .99)
            quantum = measurement.twilight_quantum
            if not math.isfinite(quantum) or not 0 < quantum <= 1:
                quantum = 0
            channels = tuple(value if quantum and math.isfinite(value) and math.isfinite(raw)
                             and 0 < value < ceiling and 0 < raw < raw_ceiling else 0
                             for value, raw in zip(channels, raw_channels))
            warning = (measurement.full_next > settings.get('FULL_TARGET', 0.8) + settings.get('FULL_DEV', 0.2)
                       or measurement.any_next > settings.get('ANY_TARGET', 2.0) + settings.get('ANY_DEV', 0.4))
            channel_threatened = [warning and math.isfinite(value) and math.isfinite(raw)
                                  and 0 < value < ceiling and ceiling / MAX_EXPOSURE_INCREASE <= raw < raw_ceiling
                                  for value, raw in zip(channels, raw_channels)]
            if twilight_prediction_previous is not None and twilight_prediction_previous[0] == mode:
                _, old_time, old_adu, old_exposure, old_gain, old_channels, old_quantum, old_channel_threatened, old_threatened = twilight_prediction_previous
                elapsed = measurement.twilight_time - old_time
                change = self._highlight_signal_change(exposure, gain, old_exposure, old_gain)
                if change is not None and 0 < elapsed <= max(120, 5 * cadence):
                    # Only linear ADU supplies a time trend. Predicted patch
                    # areas certify headroom; their size is not signal growth.
                    growth = max(0.0, math.log(measurement.adu / old_adu) - change) / elapsed
                    forecast = math.exp(min(math.log(2), growth * 2 * cadence))
                    for index, (value, old_value, raw) in enumerate(zip(channels, old_channels, raw_channels)):
                        if (quantum and quantum == old_quantum and math.isfinite(value) and math.isfinite(old_value)
                                and math.isfinite(raw) and 0 < raw < raw_ceiling
                                and 0 < value < ceiling and 0 < old_value < ceiling):
                            # Calibrated colour tails supply a scene trend; raw
                            # headroom excludes clipped plateaus and keeps faint
                            # fast-growing channels from taking over the reserve.
                            difference = math.log(value / old_value) - change
                            # Ignore the paired one-code quantile uncertainties;
                            # otherwise a static Moon can acquire a false slope.
                            noise = quantum / value + old_quantum / old_value
                            growth = difference / elapsed if difference > noise else 0.0
                            channel_forecast = math.exp(min(math.log(2), growth * 2 * cadence))
                            projected = raw + value * (channel_forecast - 1)
                            if (growth > 0 and (old_channel_threatened[index] or channel_threatened[index]
                                                or (warning and projected >= ceiling))):
                                # Track a previously threatened colour through
                                # headroom recovery until its scene stops rising.
                                channel_threatened[index] = True
                                forecast = max(forecast, channel_forecast)
                    threatened = old_threatened and forecast > 1.000001
            threatened |= warning
            if threatened:
                # Start with a small reserve and allow a 10% early cut as dawn
                # brightens. Ordinary clipping retains its stronger backstop.
                # Keep this whole-image reserve inside the normal shadow floor.
                floor = target / 2 ** settings.get('MAX_BOOST', 2.0)
                if measurement.adu > floor * 1.02:
                    # Once headroom is restored, track continuing dawn growth
                    # instead of immediately undoing protection with recovery.
                    # Static sources and fading scenes do not retain this hold.
                    limit = max(0.9, 0.95 / forecast if warning else 1 / math.sqrt(forecast),
                                floor / measurement.adu)
                    twilight_prediction_limit = math.exp(measurement.twilight_weight * math.log(limit))
                    reason += ' + twilight predicted clipping'
                    # Rendering already leaves signal above target unlifted;
                    # engaging now lets it compensate subsequent protected cuts.
                    twilight_needed = True
                    logger.info('Twilight whole-image headroom: forecast %.3fx; source signal ceiling %.3fx',
                                forecast, twilight_prediction_limit)
                else:
                    threatened = False
            if not threatened:
                channel_threatened = [False] * 3
            if twilight_prediction_previous is not None and measurement.twilight_time <= twilight_prediction_previous[1]:
                self._twilight_prediction_previous = twilight_prediction_previous
            else:
                self._twilight_prediction_previous = (mode, measurement.twilight_time, measurement.adu, exposure, gain,
                                                     channels, quantum, tuple(channel_threatened), threatened)
        else:
            self._twilight_prediction_previous = None
        output_limit = self.twilight_output.limit(
            measurement, exposure, gain, mode, settings, self._highlight_signal_change,
            self._expUtils.EXPOSURE_NEXT, self.effective_gain(self._expUtils.GAIN_NEXT), target, cadence)
        if output_limit is not None:
            requested = min(scale, twilight_prediction_limit) if twilight_prediction_limit is not None else scale
            if output_limit < requested:
                twilight_prediction_limit = math.exp(
                    (1 - measurement.twilight_weight) * math.log(requested)
                    + measurement.twilight_weight * math.log(output_limit))
                twilight_needed = True
                reason += ' + twilight output ceiling'
                logger.info('Twilight output: source %s; tail %.1f; excess %.3f%%; signal ceiling %.3fx',
                            self.twilight_output.sample[0].twilight_time,
                            self.twilight_output.sample[0].twilight_peak * 255,
                            self.twilight_output.sample[0].any, twilight_prediction_limit)
        if twilight_active and measurement.twilight_peak > 0:
            forecast = 1.0
            if twilight_previous is not None and twilight_previous[0] == mode:
                _, old_time, old_peak, old_exposure, old_gain = twilight_previous
                elapsed = measurement.twilight_time - old_time
                change = self._highlight_signal_change(exposure, gain, old_exposure, old_gain)
                if change is not None and 0 < elapsed <= max(120, 5 * cadence):
                    # Normalize out our own exposure/gain commands. Reserve two
                    # capture intervals for brightening while a frame is in flight.
                    growth = max(0.0, math.log(measurement.twilight_peak / old_peak) - change) / elapsed
                    forecast = math.exp(min(math.log(2), growth * 2 * cadence))
            limit = 0.90 / (measurement.twilight_peak * forecast)
            if limit < scale:
                # Blend the constraint in signal space; zero strength retains
                # the exact existing decision, including every recovery path.
                scale *= math.exp(measurement.twilight_weight * math.log(limit / scale))
                reason += ' + twilight horizon'
                response_target = None
                clipping_urgency = 1.0  # Pending-command easing must not weaken this ceiling.
            twilight_needed |= measurement.twilight_peak * forecast >= 0.88 and measurement.adu < target - deviation
            if twilight_previous is not None and twilight_previous[0] == mode and measurement.twilight_time <= twilight_previous[1]:
                self._twilight_previous = twilight_previous
            else:
                self._twilight_previous = (mode, measurement.twilight_time, measurement.twilight_peak, exposure, gain)
            logger.info('Twilight horizon: raw peak %.3f; strength %.3f; forecast %.3fx; limit %.3fx',
                        measurement.twilight_peak, measurement.twilight_weight, forecast, limit)
        else:
            self._twilight_previous = None
        # Retain the command slew through internal clipping-history resets and
        # briefly after the solar envelope ends. Release only once ordinary
        # control can take over without a larger recovery or reduction step.
        if twilight_active or twilight_slew is not None:
            self._twilight_slew_state = (mode, max(measurement.twilight_time, twilight_slew[1])
                                        if twilight_slew is not None else measurement.twilight_time)
        else:
            self._twilight_slew_state = None
        if scale > 1 and response_target is None and self._highlight_request_pending(exposure, gain):
            scale, reason = 1.0, 'await pending exposure/gain'
        self.hist_adu = []
        # Keep existing status/telemetry fields useful without the ADU history
        # delay. The reason string and requested multiplier are logged once here.
        self._current_adu_target = measurement.adu
        self.target_adu_found = scale == 1.0 and twilight_prediction_limit is None
        logger.info('Highlight patches (pre-dark): full %.3f%%, any %.3f%%; calibrated ADU %.2f; exposure request %.3fx; reason: %s',
                    measurement.full, measurement.any, measurement.adu, scale, reason)
        if (scale != 1.0 or twilight_prediction_limit is not None) and not twilight_duplicate:
            self._set_exposure(exposure, gain, exposure * scale, highlight=True,
                               previous_scale=previous_scale, response_target=response_target,
                               clipping_urgency=clipping_urgency,
                               twilight_prediction_limit=twilight_prediction_limit,
                               twilight_slew=self._twilight_slew_state is not None,
                               twilight_release=not twilight_active)
            if self._expUtils.EXPOSURE_NEXT == exposure and self._expUtils.GAIN_NEXT == gain:
                logger.info('Highlight adjustment limited by exposure/gain settings')
        elif scale == 1.0 and not twilight_active and not self._highlight_request_pending(exposure, gain):
            self._twilight_slew_state = None
        # Dry-run the same mode policy, ISO selection and storage rounding used
        # by real requests. Copy local policy state so a probe cannot initialize
        # the legacy gain ladder earlier than a real adjustment would.
        logger.info('Highlight preview only (not sent to camera): testing +10% exposure headroom')
        next_exposure, next_gain, _, _ = copy.copy(self)._calculate_exposure(exposure, gain, exposure * MAX_EXPOSURE_INCREASE, highlight=True)
        logger.info('Highlight preview result (not sent to camera): %.6fs @ gain %.3f', next_exposure, next_gain)
        ceiling = next_exposure <= exposure + 0.0000005 and next_gain <= gain + 0.0005
        self.highlight_transition.observe(measurement, target, deviation, settings,
                                          self._highlight_request_pending(exposure, gain), ceiling, predicted_block,
                                          output_needed=self.highlight_output.active or twilight_needed)
        logger.info('Highlight rendering control: %s; %s; predicted +10%% patches: full %.3f%%, any %.3f%%',
                    self.highlight_transition.phase, self.highlight_transition.reason, measurement.full_next, measurement.any_next)
        request_scale = min(scale, twilight_prediction_limit) if twilight_prediction_limit is not None else scale
        self._highlight_slew = (mode, request_scale) if 0 < request_scale < 1 else None
        return measurement.adu, measurement.adu


    def _highlight_request_pending(self, exposure, gain):
        pending_exposure = self._expUtils.EXPOSURE_NEXT
        pending_gain = self.effective_gain(self._expUtils.GAIN_NEXT)
        return (self.exposure_min <= pending_exposure <= self.exposure_max
                and self.gain_min <= pending_gain <= self.gain_max
                and (not math.isclose(exposure, pending_exposure, rel_tol=0, abs_tol=0.0000005)
                     or not math.isclose(gain, pending_gain, rel_tol=0, abs_tol=0.0005)))


    def effective_gain(self, gain):
        """Use the same representable values for limits, requests and captures."""
        return quantize_gain(gain, self.gain_quantum, self.gain_values)


    def effective_gain_limits(self, minimum, maximum):
        return gain_limits(minimum, maximum, self.gain_quantum, self.gain_values)


    def _calculate_exposure(self, current_exposure, current_gain, next_exposure, highlight=False):
        """Map a request to achievable settings without publishing it."""
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

        if self.gain_quantum or self.gain_values:
            requested_gain = next_gain
            next_gain = self.effective_gain(next_gain)
            if (highlight and self.gain_quantum and next_gain == current_gain
                    and not math.isclose(requested_gain, current_gain, rel_tol=0, abs_tol=1e-9)):
                # A sub-step correction must not stall forever outside the
                # deadband. Take one hardware step, bounded by the mode limits.
                next_gain = self.effective_gain(current_gain + math.copysign(self.gain_quantum, requested_gain - current_gain))
            next_gain = min(self.gain_max, max(self.gain_min, next_gain))
            gain_delta = next_gain - current_gain

        return next_exposure, next_gain, exposure_delta, gain_delta


    def _set_exposure(self, current_exposure, current_gain, next_exposure, highlight=False, previous_scale=None,
                      response_target=None, clipping_urgency=0.0, twilight_slew=False, twilight_release=False,
                      twilight_prediction_limit=None):
        reducing = next_exposure < current_exposure
        scale = next_exposure / current_exposure if highlight else 1.0
        next_exposure, next_gain, exposure_delta, gain_delta = self._calculate_exposure(current_exposure, current_gain, next_exposure, highlight)

        response_applied = False
        pending = highlight and self._highlight_request_pending(current_exposure, current_gain)
        if highlight and response_target is not None:
            pending_exposure = self._expUtils.EXPOSURE_NEXT if pending else current_exposure
            pending_gain = self.effective_gain(self._expUtils.GAIN_NEXT) if pending else current_gain
            previous_change = self._highlight_signal_change(
                pending_exposure, pending_gain, current_exposure, current_gain)
            modelled = self._highlight_signal_change(next_exposure, next_gain, current_exposure, current_gain)
            if previous_change is not None and modelled is not None:
                target_change = math.log(response_target)
                # An old dark frame must not undo a newer cut, nor may a stale
                # target weaken an already stronger request in either direction.
                if ((target_change > 0 and previous_change < -1e-9)
                        or (target_change > 0 and previous_change >= target_change)
                        or (target_change < 0 and previous_change <= target_change)):
                    logger.info('Highlight ADU target already covered by pending exposure/gain')
                    if twilight_prediction_limit is None:
                        return
                    # The ADU target can be covered while a newer twilight
                    # headroom warning still needs a tighter source ceiling.
                    target_change = previous_change
                # Half the remaining logarithmic error gives proportional,
                # diminishing steps even when the next capture is in flight.
                # Anchor the target to the measured capture, never multiply an
                # old correction onto pending settings. Permit larger steps
                # for large errors; the half-error response tapers them as the
                # target approaches. Raw headroom still bounds total growth
                # justified by this source independently of the command cap.
                step = min(math.log(1.5), max(math.log(2.0 / 3.0), (target_change - previous_change) / 2))
                change = previous_change + step
                if target_change > 0:
                    change = min(change, math.log(scale))
                    if change <= previous_change + 1e-9:
                        logger.info('Highlight growth held: await pending exposure/gain headroom')
                        if twilight_prediction_limit is None:
                            return
                        change = previous_change
                else:
                    if previous_change > 0:
                        # Cancel unmeasured growth immediately when the source
                        # is already too bright; its old increase is not a safe
                        # starting point for a leisurely reversal.
                        change = min(change, math.log(scale))
                    change = max(change, math.log(0.5))
                    if change >= previous_change - 1e-9:
                        if twilight_prediction_limit is None:
                            return
                        change = previous_change
                response_exposure, response_gain, _, _ = self._calculate_exposure(
                    pending_exposure, pending_gain,
                    pending_exposure * math.exp(change - previous_change), highlight=True)
                achieved = self._highlight_signal_change(
                    response_exposure, response_gain, pending_exposure, pending_gain)
                if achieved is not None:
                    next_exposure, next_gain = response_exposure, response_gain
                    exposure_delta = next_exposure - current_exposure
                    gain_delta = next_gain - current_gain
                    response_applied = True
                    logger.info('Highlight ADU response: target %.3fx captured signal; command %.3fx pending signal',
                                response_target, math.exp(achieved))

        if twilight_prediction_limit is not None:
            # Apply the reserve to achieved signal AFTER proportional ADU
            # easing. Comparing it with the abstract ADU scale would miss a
            # half-error response that is still too gentle for a brightening sky.
            change = self._highlight_signal_change(next_exposure, next_gain, current_exposure, current_gain)
            if change is None or change > math.log(twilight_prediction_limit):
                next_exposure, next_gain, _, _ = self._calculate_exposure(
                    current_exposure, current_gain, current_exposure * twilight_prediction_limit, highlight=True)
                exposure_delta = next_exposure - current_exposure
                gain_delta = next_gain - current_gain
                reducing = True
                # The ceiling is tied to this capture. Preserve a newer pending
                # cut below, and do not ease the protective reserve a second time.
                response_applied = False
                clipping_urgency = 1.0

        if pending and not reducing and response_target is not None and not response_applied:
            # Legacy gain ladders have no known signal conversion; retain the
            # conservative wait when the ADU response cannot be modelled.
            logger.info('Highlight growth held: await pending exposure/gain')
            return

        if highlight and reducing and pending and not response_applied:
            pending_exposure = self._expUtils.EXPOSURE_NEXT
            pending_gain = self.effective_gain(self._expUtils.GAIN_NEXT)
            # An older frame's "reduction" can still raise a newer request.
            # Compare achieved signal using the selected mode's gain model;
            # discrete ISO may trade lower gain for a longer exposure. Fixed
            # and legacy modes do not exchange exposure for gain on this path.
            change = self._highlight_signal_change(next_exposure, next_gain, pending_exposure, pending_gain)
            weaker = (change > 1e-9 if hasattr(self, 'gain2dB') else
                      next_exposure > pending_exposure + 0.0000005 or next_gain > pending_gain + 0.0005)
            if weaker:
                logger.info('Highlight reduction held: keeping stronger pending %.6fs @ gain %.3f; source %.6fs @ gain %.3f',
                            pending_exposure, pending_gain, current_exposure, current_gain)
                return

            # With a later capture already in flight, two interleaved control
            # tracks can alternate large/small cuts. Limit the step from that
            # command to half the measured logarithmic correction. Approach
            # the original absolute target; never compound a cut on pending
            # settings or predict how clipped patch areas would scale. A new
            # stronger demand bypasses this limit immediately. Release the
            # extra slew limit continuously as clipping becomes severe.
            slew_power = 0.5 + 0.5 * clipping_urgency
            previous_change = self._highlight_signal_change(pending_exposure, pending_gain, current_exposure, current_gain)
            if (change is not None and previous_change is not None and previous_change < -1e-9
                    and previous_scale is not None and scale >= previous_scale - 1e-9
                    and 0 < scale < 1 and change < math.log(scale) * slew_power
                    and math.isclose(self._expUtils.EXPOSURE_CURRENT, pending_exposure,
                                     rel_tol=0, abs_tol=0.0000005)
                    and math.isclose(self.effective_gain(self._expUtils.GAIN_CURRENT), pending_gain,
                                     rel_tol=0, abs_tol=0.0005)):
                limited_exposure, limited_gain, _, _ = self._calculate_exposure(
                    pending_exposure, pending_gain, pending_exposure * scale ** slew_power, highlight=True)
                limited_change = self._highlight_signal_change(
                    limited_exposure, limited_gain, pending_exposure, pending_gain)
                # Rounding, gain floors and discrete ISO can make a smaller
                # step unrepresentable. Retain the original safe request then.
                if limited_change is not None and change < limited_change < 0:
                    next_exposure, next_gain = limited_exposure, limited_gain
                    exposure_delta = next_exposure - current_exposure
                    gain_delta = next_gain - current_gain
                    logger.info('Highlight delayed reduction slewed from pending %.6fs @ gain %.3f',
                                pending_exposure, pending_gain)

        if twilight_slew:
            # The blended request can cross from 10% recovery to the fast ADU
            # branch while the guard fades. Bound the final achieved signal in
            # both directions, from the newest pending command rather than an
            # older capture. The source-relative request remains the target;
            # a delayed frame never compounds its full correction on pending.
            base_exposure = self._expUtils.EXPOSURE_NEXT if pending else current_exposure
            base_gain = self.effective_gain(self._expUtils.GAIN_NEXT) if pending else current_gain
            change = self._highlight_signal_change(next_exposure, next_gain, base_exposure, base_gain)
            cap = -math.log(0.8) if change is not None and change < 0 else math.log(MAX_EXPOSURE_INCREASE)
            if change is not None and abs(change) > cap:
                limited = self._calculate_exposure(
                    base_exposure, base_gain, base_exposure * math.exp(max(-cap, min(cap, change))), highlight=True)
                limited_change = self._highlight_signal_change(limited[0], limited[1], base_exposure, base_gain)
                # Hardware gain steps and integer exposure storage set the
                # smallest achievable step; keep that step when the cap rounds.
                if limited_change is not None and limited_change * change > 0 and abs(limited_change) < abs(change):
                    next_exposure, next_gain = limited[:2]
                    exposure_delta = next_exposure - current_exposure
                    gain_delta = next_gain - current_gain
                    logger.info('Twilight signal slew: %.3fx newest pending exposure/gain', math.exp(limited_change))
            elif twilight_release:
                self._twilight_slew_state = None

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


    def _highlight_signal_change(self, exposure, gain, reference_exposure, reference_gain):
        if hasattr(self, 'gain2dB'):
            gain_change = (self.gain2dB(gain) - self.gain2dB(reference_gain)) * math.log(10) / 20
        elif gain == reference_gain:
            gain_change = 0.0
        else:
            return None  # Legacy gain steps have no declared signal units.
        return math.log(exposure / reference_exposure) + gain_change



    def adjust_exposure_gain(self, *args):
        raise Exception('Not implemented')

